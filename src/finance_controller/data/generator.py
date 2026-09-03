"""Synthetic dataset generator for LedgerLens.

Three design principles are encoded structurally here — not just in comments:

Principle 1 — Split before observation.
    TrueTransaction objects are shuffled and partitioned into dev / held-out
    BEFORE any invoice, settlement, or bank-credit record is derived from
    them.  An invoice and its matching settlement item always share the same
    partition; there is no code path that can place them in different splits.

Principle 2 — Explicit ground truth for decoys.
    Each decoy settlement item (same vendor/amount as a real invoice, wrong
    order_reference) carries an explicit is_match=False MatchLabel against
    the specific invoice it targets.  Evaluation code reads labels; it never
    infers them from whichever pair scores highest at runtime.

Principle 3 — Amount mismatch ≠ non-match.
    True pairs where settlement.amount ≠ invoice.amount (partial payment /
    rounding discrepancy) are labeled is_match=True.  They may land in
    REVIEW or EXCEPTION during reconciliation — that is the matching
    engine's job, not the labeler's.
"""

from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from datetime import date
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import Final

from finance_controller.data.constants import (
    BANK_CREDIT_DELAY_DAYS,
    GATEWAY_FEE_RATE,
    GST_ON_FEE_RATE,
    RANDOM_SEED,
    SETTLEMENT_DELAY_DAYS,
    VENDOR_BASE_NAMES,
)
from finance_controller.data.noise import (
    drift_date,
    invoice_vendor_name,
    noisy_reference,
    perturb_amount,
    pick_legal_suffix,
    settlement_vendor_name,
)
from finance_controller.domain.enums import EntityType, Source
from finance_controller.domain.models import (
    BankCredit,
    Invoice,
    MatchLabel,
    SettlementItem,
)

# ---------------------------------------------------------------------------
# Generation parameters
# ---------------------------------------------------------------------------

N_TRUE: Final[int] = 200
"""True matched invoice←→settlement pairs, spread across both splits.

200 gives ~160 dev pairs and ~40 held-out pairs.  40 is the practical minimum
for the held-out set: a single misclassification moves precision by ~2.5 points,
which is defensible.  At 16 (80 × 0.2) it swings by ~5 points, which a reviewer
will notice.
"""

N_DECOYS: Final[int] = 50
"""Orphan settlement items that mimic a real invoice but are not a match.

Kept at ~25% of N_TRUE so the ratio of hard negatives to true pairs stays
constant as the dataset scales.  Each decoy targets one specific invoice
(same vendor, same amount, wrong order_reference) and carries an explicit
is_match=False label against it.  Decoys are distributed proportionally to
the dev/held-out split ratio.
"""

N_ORPHAN_INV: Final[int] = 25
"""Invoices that have no matching settlement item — stage-1 exceptions.

Kept at ~12.5% of N_TRUE so the exception rate stays constant as the
dataset scales.
"""

PERTURB_RATE: Final[float] = 0.15
"""Fraction of true pairs where settlement.amount differs from invoice.amount.

These are still labeled is_match=True.  The discrepancy (partial payment,
rounding) is an accounting exception for the engine to flag, not a reason
to change the ground-truth label.
"""

DROP_REF_RATE: Final[float] = 0.20
"""Fraction of true pairs where the settlement item carries no order_reference.

Forces the matching engine to rely on vendor+amount+date evidence alone
instead of a trivial exact-reference join.
"""

BATCH_SIZE_MIN: Final[int] = 2
BATCH_SIZE_MAX: Final[int] = 5

SETTLEMENT_SUM_MISMATCH_RATE: Final[float] = 0.08
"""Fraction of settlement batches where BankCredit.amount ≠ sum(net_amounts).

Stage-2 (batch↔bank) exception.  Rare by design — real discrepancies are
not the common case.  The mismatch lives only in BankCredit.amount; the
settlement items themselves are always internally consistent.
"""

DEV_RATIO: Final[float] = 0.80
BASE_DATE: Final[date] = date(2026, 8, 1)

_TWO_DP: Final[Decimal] = Decimal("0.01")
FEE_RATE: Final[Decimal] = Decimal(GATEWAY_FEE_RATE)
GST_RATE: Final[Decimal] = Decimal(GST_ON_FEE_RATE)
_AMOUNT_POOL: Final[list[int]] = list(range(5000, 200_001, 500))

# Sentinel values used while settlement items await batch assignment.
_PENDING_SETL = "PENDING"
_PENDING_UTR = "UTR_PENDING"


# ---------------------------------------------------------------------------
# Hidden ground-truth record — never serialised
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class TrueTransaction:
    """One real-world sale event; the hidden source for all three record types.

    This object is built first, split into dev/held-out second, and only
    then used to derive noisy invoice + settlement observations.  It is
    never written to any CSV.

    tx_ids on SplitDataset expose the set of tx_ids per partition so the
    leakage test can assert the two sets are disjoint without needing to
    reach into this private class.
    """

    tx_id: str
    split: str               # "dev" or "held_out"; set after shuffle
    canonical_name: str
    legal_suffix: str
    order_reference: str
    invoice_id: str
    payment_id: str
    amount: Decimal
    invoice_date: date
    has_amount_perturbation: bool
    drop_reference: bool     # settlement item omits order_reference when True


# ---------------------------------------------------------------------------
# Public output container
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SplitDataset:
    """All generated records for one split (dev or held_out)."""

    split: str
    invoices: list[Invoice]
    settlement_items: list[SettlementItem]
    bank_credits: list[BankCredit]
    match_labels: list[MatchLabel]
    tx_ids: frozenset[str]
    """Hidden transaction IDs for the leakage check test only.

    Never written to CSV.  The test asserts dev.tx_ids.isdisjoint(held.tx_ids).
    """


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _compute_net(amount: Decimal) -> tuple[Decimal, Decimal, Decimal]:
    """Return (fee, tax, net_amount) rounded to 2 dp."""
    fee = (amount * FEE_RATE).quantize(_TWO_DP, rounding=ROUND_HALF_UP)
    tax = (fee * GST_RATE).quantize(_TWO_DP, rounding=ROUND_HALF_UP)
    return fee, tax, amount - fee - tax


def _build_transactions(rng: random.Random) -> list[TrueTransaction]:
    """Create N_TRUE transactions, shuffle, then assign splits.

    The shuffle-then-split order is the guarantee that observations derived
    from the same transaction always land in the same partition.
    """
    txs: list[TrueTransaction] = [
        TrueTransaction(
            tx_id=f"TX-{i + 1:04d}",
            split="dev",  # placeholder; overwritten after shuffle below
            canonical_name=rng.choice(VENDOR_BASE_NAMES),
            legal_suffix=pick_legal_suffix(rng),
            order_reference=f"ORD-{i + 1:04d}",
            invoice_id=f"INV-{i + 1:04d}",
            payment_id=f"pay_TX{i + 1:04d}",
            amount=Decimal(rng.choice(_AMOUNT_POOL)),
            invoice_date=drift_date(BASE_DATE, 0, 30, rng),
            has_amount_perturbation=rng.random() < PERTURB_RATE,
            drop_reference=rng.random() < DROP_REF_RATE,
        )
        for i in range(N_TRUE)
    ]
    rng.shuffle(txs)
    n_dev = round(N_TRUE * DEV_RATIO)
    for i, tx in enumerate(txs):
        tx.split = "dev" if i < n_dev else "held_out"
    return txs


def _make_invoice(tx: TrueTransaction, rng: random.Random) -> Invoice:
    return Invoice(
        invoice_id=tx.invoice_id,
        order_reference=tx.order_reference,
        customer_name=invoice_vendor_name(tx.canonical_name, tx.legal_suffix, rng),
        amount=tx.amount,
        invoice_date=tx.invoice_date,
    )


def _make_settlement_pending(tx: TrueTransaction, rng: random.Random) -> SettlementItem:
    """Derive a settlement item from a true transaction.

    Uses settlement_vendor_name (upper-case, abbreviated suffixes, possible
    truncation) — a different noise path from invoice_vendor_name (title-case,
    double-space, trailing period).  The two never converge.

    settlement_id and settlement_utr are placeholders; _assign_batches
    rebuilds each item with real batch identifiers.
    """
    s_amount = perturb_amount(tx.amount, rng) if tx.has_amount_perturbation else tx.amount
    fee, tax, net = _compute_net(s_amount)
    created = drift_date(tx.invoice_date, SETTLEMENT_DELAY_DAYS[0], SETTLEMENT_DELAY_DAYS[1], rng)
    settled = drift_date(created, 0, 2, rng)
    return SettlementItem(
        entity_id=tx.payment_id,
        entity_type=EntityType.PAYMENT,
        order_reference=noisy_reference(tx.order_reference, rng, drop=tx.drop_reference),
        payment_id=tx.payment_id,
        customer_name=settlement_vendor_name(tx.canonical_name, tx.legal_suffix, rng),
        amount=s_amount,
        fee=fee,
        tax=tax,
        net_amount=net,
        settlement_id=_PENDING_SETL,
        settlement_utr=_PENDING_UTR,
        created_at=created,
        settled_at=settled,
    )


def _make_decoy_pending(
    idx: int,
    target: TrueTransaction,
    rng: random.Random,
) -> tuple[SettlementItem, MatchLabel]:
    """A settlement item that looks like target's invoice but is NOT a match.

    Identical vendor rendering and amount make this a hard negative for the
    matching engine.  The order_reference is a decoy-specific string
    ("DECOY-XXXX") that never matches any invoice's ORD-XXXX reference —
    this structural difference is what anchors the explicit is_match=False
    label to a specific, unambiguous record rather than leaving the labeling
    to evaluation-time inference.
    """
    pid = f"pay_DECOY{idx:04d}"
    fee, tax, net = _compute_net(target.amount)
    created = drift_date(
        target.invoice_date, SETTLEMENT_DELAY_DAYS[0], SETTLEMENT_DELAY_DAYS[1], rng
    )
    settled = drift_date(created, 0, 2, rng)
    item = SettlementItem(
        entity_id=pid,
        entity_type=EntityType.PAYMENT,
        order_reference=f"DECOY-{idx:04d}",  # never matches any ORD-XXXX
        payment_id=pid,
        customer_name=settlement_vendor_name(target.canonical_name, target.legal_suffix, rng),
        amount=target.amount,
        fee=fee,
        tax=tax,
        net_amount=net,
        settlement_id=_PENDING_SETL,
        settlement_utr=_PENDING_UTR,
        created_at=created,
        settled_at=settled,
    )
    label = MatchLabel(
        left_id=target.invoice_id,
        left_source=Source.INVOICE,
        right_id=pid,
        right_source=Source.SETTLEMENT_ITEM,
        is_match=False,  # explicit — not inferred at evaluation time
    )
    return item, label


def _make_orphan_invoice(idx: int, rng: random.Random) -> Invoice:
    """Invoice with no matching settlement item — a stage-1 exception."""
    return Invoice(
        invoice_id=f"INV-ORPH-{idx:04d}",
        order_reference=f"ORD-ORPH-{idx:04d}",
        customer_name=invoice_vendor_name(
            rng.choice(VENDOR_BASE_NAMES), pick_legal_suffix(rng), rng
        ),
        amount=Decimal(rng.choice(_AMOUNT_POOL)),
        invoice_date=drift_date(BASE_DATE, 0, 30, rng),
    )


def _assign_batches(
    pending: list[SettlementItem],
    prefix: str,
    rng: random.Random,
) -> tuple[list[SettlementItem], list[BankCredit]]:
    """Shuffle items into batches of 2–5 and generate one BankCredit per batch.

    Items are frozen dataclasses, so they are rebuilt with the real
    settlement_id / settlement_utr assigned during this step.

    SETTLEMENT_SUM_MISMATCH_RATE% of batches receive a corrupted
    BankCredit.amount (stage-2 exception).  The mismatch is in the credit
    only — every SettlementItem is always self-consistent.
    """
    pool = list(pending)
    rng.shuffle(pool)

    settled_items: list[SettlementItem] = []
    credits: list[BankCredit] = []
    batch_num = 0
    pos = 0

    while pos < len(pool):
        batch_num += 1
        remaining = len(pool) - pos
        if remaining <= BATCH_SIZE_MIN:
            # Fewer items than the minimum batch size — take them all.
            # This only arises when the pool itself is tiny (< BATCH_SIZE_MIN).
            size = remaining
        else:
            size = rng.randint(BATCH_SIZE_MIN, min(BATCH_SIZE_MAX, remaining))
            # If this choice would strand fewer than BATCH_SIZE_MIN items in the
            # next iteration, absorb them into the current batch now.  A batch
            # of BATCH_SIZE_MAX + 1 is preferable to a batch of 1.
            tail = remaining - size
            if 0 < tail < BATCH_SIZE_MIN:
                size = remaining
        chunk = pool[pos : pos + size]
        pos += size

        setl_id = f"SETL-{prefix}-{batch_num:04d}"
        utr = f"UTR{prefix}{batch_num:06d}"

        # Rebuild each frozen item with the real batch identifiers.
        for item in chunk:
            settled_items.append(
                SettlementItem(
                    entity_id=item.entity_id,
                    entity_type=item.entity_type,
                    order_reference=item.order_reference,
                    payment_id=item.payment_id,
                    customer_name=item.customer_name,
                    amount=item.amount,
                    fee=item.fee,
                    tax=item.tax,
                    net_amount=item.net_amount,
                    settlement_id=setl_id,
                    settlement_utr=utr,
                    created_at=item.created_at,
                    settled_at=item.settled_at,
                )
            )

        net_sum = sum((it.net_amount for it in chunk), Decimal("0"))

        # Inject stage-2 mismatch for ~8% of batches.
        if rng.random() < SETTLEMENT_SUM_MISMATCH_RATE:
            delta = Decimal(rng.choice([200, 500, 1000, 1500]))
            credit_amount = net_sum + delta
        else:
            credit_amount = net_sum

        latest_settled = max(
            (it.settled_at if it.settled_at is not None else it.created_at for it in chunk),
            default=BASE_DATE,
        )
        value_date = drift_date(
            latest_settled, BANK_CREDIT_DELAY_DAYS[0], BANK_CREDIT_DELAY_DAYS[1], rng
        )
        credits.append(
            BankCredit(
                credit_id=f"BC-{prefix}-{batch_num:04d}",
                utr=utr,
                amount=credit_amount,
                value_date=value_date,
            )
        )

    return settled_items, credits


def _build_split(
    txs: list[TrueTransaction],
    split_name: str,
    decoy_offset: int,
    orphan_offset: int,
    rng: random.Random,
) -> SplitDataset:
    split_txs = [tx for tx in txs if tx.split == split_name]
    ratio = DEV_RATIO if split_name == "dev" else (1.0 - DEV_RATIO)
    n_decoys = round(N_DECOYS * ratio)
    n_orphans = round(N_ORPHAN_INV * ratio)
    prefix = "DEV" if split_name == "dev" else "HLD"

    # --- Invoices ---
    true_invoices: list[Invoice] = [_make_invoice(tx, rng) for tx in split_txs]
    orphan_invoices: list[Invoice] = [
        _make_orphan_invoice(orphan_offset + i, rng) for i in range(n_orphans)
    ]

    # --- True match labels (is_match=True) ---
    true_labels: list[MatchLabel] = [
        MatchLabel(
            left_id=tx.invoice_id,
            left_source=Source.INVOICE,
            right_id=tx.payment_id,
            right_source=Source.SETTLEMENT_ITEM,
            is_match=True,
        )
        for tx in split_txs
    ]

    # --- Pending settlement items: true + decoys ---
    pending: list[SettlementItem] = [_make_settlement_pending(tx, rng) for tx in split_txs]

    decoy_labels: list[MatchLabel] = []
    # rng.choices allows the same invoice to attract multiple decoys,
    # which is realistic (an invoice can have more than one confusable
    # settlement item in the wild).
    decoy_targets = rng.choices(split_txs, k=n_decoys) if split_txs else []
    for j, target in enumerate(decoy_targets):
        item, label = _make_decoy_pending(decoy_offset + j, target, rng)
        pending.append(item)
        decoy_labels.append(label)

    # --- Batch and credit ---
    settled_items, credits = _assign_batches(pending, prefix, rng)

    return SplitDataset(
        split=split_name,
        invoices=true_invoices + orphan_invoices,
        settlement_items=settled_items,
        bank_credits=credits,
        match_labels=true_labels + decoy_labels,
        tx_ids=frozenset(tx.tx_id for tx in split_txs),
    )


# ---------------------------------------------------------------------------
# CSV serialisation (stdlib only — no pandas dependency)
# ---------------------------------------------------------------------------


def _write_invoices(invoices: list[Invoice], path: Path) -> None:
    with path.open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "invoice_id", "order_reference", "customer_name",
                "amount", "invoice_date", "description",
            ],
        )
        w.writeheader()
        for inv in invoices:
            w.writerow(
                {
                    "invoice_id": inv.invoice_id,
                    "order_reference": inv.order_reference,
                    "customer_name": inv.customer_name,
                    "amount": str(inv.amount),
                    "invoice_date": inv.invoice_date.isoformat(),
                    "description": inv.description or "",
                }
            )


def _write_settlement_items(items: list[SettlementItem], path: Path) -> None:
    with path.open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "entity_id", "entity_type", "order_reference", "payment_id",
                "customer_name", "amount", "fee", "tax", "net_amount",
                "settlement_id", "settlement_utr", "created_at", "settled_at",
            ],
        )
        w.writeheader()
        for it in items:
            w.writerow(
                {
                    "entity_id": it.entity_id,
                    "entity_type": it.entity_type.value,
                    "order_reference": it.order_reference or "",
                    "payment_id": it.payment_id,
                    "customer_name": it.customer_name,
                    "amount": str(it.amount),
                    "fee": str(it.fee),
                    "tax": str(it.tax),
                    "net_amount": str(it.net_amount),
                    "settlement_id": it.settlement_id,
                    "settlement_utr": it.settlement_utr,
                    "created_at": it.created_at.isoformat(),
                    "settled_at": it.settled_at.isoformat() if it.settled_at else "",
                }
            )


def _write_bank_credits(credits: list[BankCredit], path: Path) -> None:
    with path.open("w", newline="") as f:
        w = csv.DictWriter(
            f, fieldnames=["credit_id", "utr", "amount", "value_date", "narration"]
        )
        w.writeheader()
        for bc in credits:
            w.writerow(
                {
                    "credit_id": bc.credit_id,
                    "utr": bc.utr,
                    "amount": str(bc.amount),
                    "value_date": bc.value_date.isoformat(),
                    "narration": bc.narration or "",
                }
            )


def _write_match_labels(labels: list[MatchLabel], path: Path) -> None:
    with path.open("w", newline="") as f:
        w = csv.DictWriter(
            f, fieldnames=["left_id", "left_source", "right_id", "right_source", "is_match"]
        )
        w.writeheader()
        for lbl in labels:
            w.writerow(
                {
                    "left_id": lbl.left_id,
                    "left_source": lbl.left_source.value,
                    "right_id": lbl.right_id,
                    "right_source": lbl.right_source.value,
                    "is_match": lbl.is_match,
                }
            )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def generate(output_dir: Path | None = None) -> tuple[SplitDataset, SplitDataset]:
    """Generate the full synthetic dataset.

    Returns (dev, held_out) as SplitDataset objects.  If output_dir is
    supplied, each split is also written to:

        output_dir/<split>/invoices.csv
        output_dir/<split>/settlement_items.csv
        output_dir/<split>/bank_credits.csv
        output_dir/<split>/match_labels.csv

    The seed is fixed (RANDOM_SEED) so every call produces the same dataset.
    """
    rng = random.Random(RANDOM_SEED)
    txs = _build_transactions(rng)

    n_dev_decoys = round(N_DECOYS * DEV_RATIO)
    n_dev_orphans = round(N_ORPHAN_INV * DEV_RATIO)

    dev = _build_split(txs, "dev", decoy_offset=0, orphan_offset=0, rng=rng)
    held = _build_split(
        txs,
        "held_out",
        decoy_offset=n_dev_decoys,
        orphan_offset=n_dev_orphans,
        rng=rng,
    )

    if output_dir is not None:
        for ds in (dev, held):
            split_dir = output_dir / ds.split
            split_dir.mkdir(parents=True, exist_ok=True)
            _write_invoices(ds.invoices, split_dir / "invoices.csv")
            _write_settlement_items(ds.settlement_items, split_dir / "settlement_items.csv")
            _write_bank_credits(ds.bank_credits, split_dir / "bank_credits.csv")
            _write_match_labels(ds.match_labels, split_dir / "match_labels.csv")

    return dev, held
