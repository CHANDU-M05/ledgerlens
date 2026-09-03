"""Tests for the synthetic data generator.

Structural tests only — these verify the generator's design invariants,
not specific numeric outputs (which are seeded and stable but not what
we're testing here).

The most important test is test_no_transaction_leakage_between_splits,
which is the machine-checkable guarantee that Principle 1 actually holds:
if that test passes, the dev/held-out split cannot have leaked.
"""

from decimal import Decimal
from pathlib import Path

import pytest

from finance_controller.data.generator import (
    DEV_RATIO,
    N_TRUE,
    SplitDataset,
    generate,
)
from finance_controller.domain.enums import Source


@pytest.fixture(scope="module")
def datasets() -> tuple[SplitDataset, SplitDataset]:
    """Generate once per module — the seed is fixed so this is deterministic."""
    return generate()


# ---------------------------------------------------------------------------
# Principle 1 — split before observation (leakage check)
# ---------------------------------------------------------------------------


def test_no_transaction_leakage_between_splits(
    datasets: tuple[SplitDataset, SplitDataset],
) -> None:
    """No TrueTransaction appears in both dev and held-out partitions.

    This is the single most important structural guarantee in the generator.
    If this test passes, an invoice and its matching settlement item cannot
    have been placed on opposite sides of the split.
    """
    dev, held = datasets
    overlap = dev.tx_ids & held.tx_ids
    assert not overlap, (
        f"Leakage: {len(overlap)} transaction(s) appear in both splits: {overlap}"
    )


def test_split_sizes_are_proportional(
    datasets: tuple[SplitDataset, SplitDataset],
) -> None:
    """Dev gets ~80% of true transactions; held-out gets the rest."""
    dev, held = datasets
    n_dev_true = len(dev.tx_ids)
    n_held_true = len(held.tx_ids)
    assert n_dev_true + n_held_true == N_TRUE
    # Allow ±1 for rounding
    assert abs(n_dev_true - round(N_TRUE * DEV_RATIO)) <= 1


# ---------------------------------------------------------------------------
# Principle 2 — explicit decoy labels
# ---------------------------------------------------------------------------


def test_decoy_labels_are_explicit_false_matches(
    datasets: tuple[SplitDataset, SplitDataset],
) -> None:
    """Every decoy settlement item has an explicit is_match=False label
    against a real invoice — never against another decoy or orphan invoice.
    """
    for ds in datasets:
        true_invoice_ids = {
            inv.invoice_id
            for inv in ds.invoices
            if inv.invoice_id.startswith("INV-") and not inv.invoice_id.startswith("INV-ORPH")
        }
        decoy_payment_ids = {
            it.payment_id for it in ds.settlement_items if it.payment_id.startswith("pay_DECOY")
        }
        false_labels = [lbl for lbl in ds.match_labels if not lbl.is_match]

        for lbl in false_labels:
            assert lbl.right_id in decoy_payment_ids, (
                f"False label right_id {lbl.right_id!r} is not a decoy payment"
            )
            assert lbl.left_id in true_invoice_ids, (
                f"False label left_id {lbl.left_id!r} is not a real invoice"
            )


def test_decoy_order_references_never_match_any_invoice(
    datasets: tuple[SplitDataset, SplitDataset],
) -> None:
    """Decoy settlement items have order_reference starting with 'DECOY-'.

    No invoice carries that prefix, so a reference-equality join can never
    accidentally produce a true-match result for a decoy pair.
    """
    for ds in datasets:
        invoice_refs = {inv.order_reference for inv in ds.invoices}
        for item in ds.settlement_items:
            if item.payment_id.startswith("pay_DECOY"):
                assert item.order_reference is not None
                assert item.order_reference.startswith("DECOY-")
                assert item.order_reference not in invoice_refs, (
                    f"Decoy ref {item.order_reference!r} collides with an invoice"
                )


# ---------------------------------------------------------------------------
# Principle 3 — amount mismatch ≠ non-match
# ---------------------------------------------------------------------------


def test_amount_perturbed_pairs_labeled_as_match(
    datasets: tuple[SplitDataset, SplitDataset],
) -> None:
    """True pairs where settlement.amount ≠ invoice.amount must be is_match=True.

    Iterates ALL labels without pre-filtering on is_match — a pre-filter would
    make the subsequent assertion tautological (you can't assert a field you
    already filtered on).  Only true-transaction pairs are in scope: decoys
    intentionally carry the same amount as their target invoice, so amount
    difference is not a meaningful signal for them.
    """
    for ds in datasets:
        invoice_amount: dict[str, Decimal] = {
            inv.invoice_id: inv.amount for inv in ds.invoices
        }
        settlement_amount: dict[str, Decimal] = {
            it.payment_id: it.amount for it in ds.settlement_items
        }

        for lbl in ds.match_labels:  # all labels — NOT pre-filtered by is_match
            # Scope to true-transaction pairs only (pay_TX...).  Decoy payments
            # (pay_DECOY...) keep the same amount as the target invoice by design,
            # so they never exercise the perturbation path.
            if not lbl.right_id.startswith("pay_TX"):
                continue
            inv_amt = invoice_amount.get(lbl.left_id)
            setl_amt = settlement_amount.get(lbl.right_id)
            if inv_amt is not None and setl_amt is not None and inv_amt != setl_amt:
                assert lbl.is_match, (
                    f"Pair ({lbl.left_id}, {lbl.right_id}) has amount mismatch "
                    f"({inv_amt} vs {setl_amt}) but is labeled is_match=False — "
                    "a perturbed true pair must still carry is_match=True"
                )


# ---------------------------------------------------------------------------
# Stage-2: settlement-sum mismatch is at the batch level
# ---------------------------------------------------------------------------


def test_all_batches_meet_minimum_size(
    datasets: tuple[SplitDataset, SplitDataset],
) -> None:
    """No batch should have fewer than BATCH_SIZE_MIN items.

    A batch of size 1 means _assign_batches left a lone item rather than
    absorbing it into an adjacent batch — a silent invariant violation.
    """
    from finance_controller.data.generator import BATCH_SIZE_MIN

    for ds in datasets:
        by_utr: dict[str, int] = {}
        for it in ds.settlement_items:
            by_utr[it.settlement_utr] = by_utr.get(it.settlement_utr, 0) + 1
        for utr, count in by_utr.items():
            assert count >= BATCH_SIZE_MIN, (
                f"{ds.split}: batch {utr!r} has {count} item(s) — "
                f"violates BATCH_SIZE_MIN={BATCH_SIZE_MIN}"
            )



def test_settlement_sum_mismatch_is_rare_and_at_batch_level(
    datasets: tuple[SplitDataset, SplitDataset],
) -> None:
    """Only a small fraction of batches have BankCredit.amount ≠ sum(net_amounts).

    The mismatch is in the credit amount only — settlement items are correct.
    With SETTLEMENT_SUM_MISMATCH_RATE=0.08 and seed fixed, the combined
    total must stay below 20% to confirm it's an exception, not the norm.
    """
    for ds in datasets:
        # Group settlement items by settlement_utr
        by_utr: dict[str, list[Decimal]] = {}
        for it in ds.settlement_items:
            by_utr.setdefault(it.settlement_utr, []).append(it.net_amount)

        credit_by_utr: dict[str, Decimal] = {bc.utr: bc.amount for bc in ds.bank_credits}

        mismatches = 0
        total_batches = len(credit_by_utr)
        for utr, nets in by_utr.items():
            expected = sum(nets, Decimal("0"))
            actual = credit_by_utr.get(utr)
            if actual is not None and actual != expected:
                mismatches += 1

        if total_batches > 0:
            mismatch_rate = mismatches / total_batches
            assert mismatch_rate < 0.20, (
                f"Mismatch rate {mismatch_rate:.1%} exceeds 20% — "
                "stage-2 exceptions should be rare"
            )


# ---------------------------------------------------------------------------
# Record counts and source labels
# ---------------------------------------------------------------------------


def test_record_counts_are_plausible(
    datasets: tuple[SplitDataset, SplitDataset],
) -> None:
    """Each split must have at least a few records of each type."""
    for ds in datasets:
        assert len(ds.invoices) > 0, f"{ds.split}: no invoices"
        assert len(ds.settlement_items) > 0, f"{ds.split}: no settlement items"
        assert len(ds.bank_credits) > 0, f"{ds.split}: no bank credits"
        assert len(ds.match_labels) > 0, f"{ds.split}: no match labels"


def test_true_and_false_labels_both_present(
    datasets: tuple[SplitDataset, SplitDataset],
) -> None:
    """Both positive and negative examples exist in each split.

    A dataset with only is_match=True would train a trivially biased model.
    """
    for ds in datasets:
        true_count = sum(1 for lbl in ds.match_labels if lbl.is_match)
        false_count = sum(1 for lbl in ds.match_labels if not lbl.is_match)
        assert true_count > 0, f"{ds.split}: no positive labels"
        assert false_count > 0, f"{ds.split}: no negative labels"


def test_match_labels_source_fields_are_correct(
    datasets: tuple[SplitDataset, SplitDataset],
) -> None:
    """All match labels have left=INVOICE and right=SETTLEMENT_ITEM."""
    for ds in datasets:
        for lbl in ds.match_labels:
            assert lbl.left_source is Source.INVOICE
            assert lbl.right_source is Source.SETTLEMENT_ITEM


def test_settlement_items_have_no_pending_batch_ids(
    datasets: tuple[SplitDataset, SplitDataset],
) -> None:
    """No settlement item should still carry a placeholder settlement_id.

    If "PENDING" appears, _assign_batches failed to process some items.
    """
    for ds in datasets:
        for it in ds.settlement_items:
            assert it.settlement_id != "PENDING", (
                f"{it.entity_id} was never assigned a real settlement_id"
            )
            assert it.settlement_utr != "UTR_PENDING", (
                f"{it.entity_id} was never assigned a real UTR"
            )


def test_all_bank_credit_utrs_match_a_settlement_group(
    datasets: tuple[SplitDataset, SplitDataset],
) -> None:
    """Every BankCredit.utr has at least one corresponding settlement item."""
    for ds in datasets:
        settlement_utrs = {it.settlement_utr for it in ds.settlement_items}
        for bc in ds.bank_credits:
            assert bc.utr in settlement_utrs, (
                f"BankCredit {bc.credit_id} UTR {bc.utr!r} has no settlement items"
            )


def test_csv_output_is_written(tmp_path: Path) -> None:
    """generate(output_dir=...) writes all eight expected CSV files."""
    generate(output_dir=tmp_path)
    expected_files = [
        "dev/invoices.csv",
        "dev/settlement_items.csv",
        "dev/bank_credits.csv",
        "dev/match_labels.csv",
        "held_out/invoices.csv",
        "held_out/settlement_items.csv",
        "held_out/bank_credits.csv",
        "held_out/match_labels.csv",
    ]
    for rel in expected_files:
        assert (tmp_path / rel).exists(), f"Missing CSV: {rel}"
