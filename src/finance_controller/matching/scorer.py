"""Stage 1 scoring: candidate pair -> MatchEvidence -> confidence score.

Weights are stated and justified, not tuned to hit a target number:

    amount closeness   0.40  — strongest identity signal; unrelated
                                transactions rarely share an amount
    vendor similarity  0.35  — second strongest; noise is bounded
                                (see data/noise.py) so genuine matches
                                still score high
    reference match    0.15  — strong WHEN present, but only ~80% of
                                true pairs carry a usable reference by
                                design, so it can't outweigh amount/vendor
    date proximity     0.10  — weakest deliberately; settlement delay is
                                expected and varies 1-4+ days by design
"""

from __future__ import annotations

from decimal import Decimal

from rapidfuzz import fuzz

from finance_controller.domain.models import (
    Invoice,
    MatchEvidence,
    ReferenceStatus,
    SettlementItem,
)

_LEGAL_SUFFIX_TOKENS = ("PRIVATE", "LIMITED", "PVT", "LTD", "LLP", "ENTERPRISES", "ENTP")

W_AMOUNT = 0.40
W_VENDOR = 0.35
W_REFERENCE = 0.15
W_DATE = 0.10

DATE_SCORE_HALF_LIFE_DAYS = 5


def _normalize_vendor(name: str) -> str:
    tokens = name.upper().replace(".", "").split()
    return " ".join(t for t in tokens if t not in _LEGAL_SUFFIX_TOKENS)


def _normalize_reference(ref: str | None) -> str | None:
    """'ORD-0007' and 'PO-0007' both normalize to '0007' — this is what
    lets a genuine reference match survive the ORD-/PO- prefix swap
    injected by noisy_reference()."""
    if ref is None:
        return None
    digits = "".join(c for c in ref if c.isdigit())
    return digits or None


def _reference_status(inv_ref: str | None, item_ref: str | None) -> ReferenceStatus:
    if item_ref is None:
        return "absent"
    if inv_ref is not None and inv_ref == item_ref:
        return "match"
    return "mismatch"  # item HAS a reference and it does not match — strong negative


def vendor_similarity(invoice_name: str, item_name: str) -> float:
    a, b = _normalize_vendor(invoice_name), _normalize_vendor(item_name)
    return float(fuzz.token_sort_ratio(a, b) / 100.0)


def amount_score(diff: Decimal, invoice_amount: Decimal) -> float:
    if invoice_amount == 0:
        return 0.0
    relative = abs(diff) / invoice_amount
    return max(0.0, 1.0 - float(relative) / 0.10)  # >=10% off -> 0.0


def date_score(days_apart: int) -> float:
    # Exponential decay, never hits exactly zero — a distant date alone
    # shouldn't veto an otherwise strong amount+vendor match.
    return float(0.5 ** (days_apart / DATE_SCORE_HALF_LIFE_DAYS))


def compute_evidence(invoice: Invoice, item: SettlementItem) -> MatchEvidence:
    vendor_sim = vendor_similarity(invoice.customer_name, item.customer_name)
    amount_diff = item.amount - invoice.amount
    date_diff = abs((item.created_at - invoice.invoice_date).days)
    inv_ref = _normalize_reference(invoice.order_reference)
    item_ref = _normalize_reference(item.order_reference)
    return MatchEvidence(
        vendor_similarity=vendor_sim,
        amount_difference=amount_diff,
        date_difference_days=date_diff,
        reference_status=_reference_status(inv_ref, item_ref),
    )


def confidence(evidence: MatchEvidence, invoice_amount: Decimal) -> float:
    a_score = amount_score(evidence.amount_difference, invoice_amount)
    d_score = date_score(evidence.date_difference_days)
    # match: full credit. absent: no signal either way (0 contribution).
    # mismatch: an ACTIVE penalty, not just zero — a present, wrong
    # reference is close to disqualifying, unlike silence.
    r_score = {"match": 1.0, "absent": 0.0, "mismatch": -1.0}[evidence.reference_status]
    return (
        W_AMOUNT * a_score
        + W_VENDOR * evidence.vendor_similarity
        + W_REFERENCE * r_score
        + W_DATE * d_score
    )
