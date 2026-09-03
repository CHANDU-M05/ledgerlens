"""Stage 1 orchestration: candidate generation -> scoring -> ambiguity
check -> status decision. No AI here — the Day 7 AI reviewer consumes
the AI_REVIEW-status MatchPredictions this produces; it never replaces
this logic.
"""

from __future__ import annotations

from decimal import Decimal

from finance_controller.domain.enums import ExceptionReason, MatchStatus
from finance_controller.domain.models import (
    Invoice,
    MatchEvidence,
    MatchPrediction,
    SettlementItem,
)
from finance_controller.matching.calibration import Thresholds
from finance_controller.matching.candidates import generate_candidates
from finance_controller.matching.scorer import compute_evidence, confidence

# If the top two candidates score within this margin, the pick is not
# safe to automate even if the top score clears auto_match.
AMBIGUITY_MARGIN = 0.05

# Sentinel for "no candidate existed" — nothing to compute evidence FROM,
# so this is a typed placeholder, not a real score.
_NO_CANDIDATE_EVIDENCE = MatchEvidence(
    vendor_similarity=0.0,
    amount_difference=Decimal("0"),
    date_difference_days=-1,
    reference_status="absent",
)


def _reason_text(evidence: MatchEvidence, score: float) -> str:
    ref_map = {
        "match": "reference matches",
        "mismatch": "reference mismatch",
        "absent": "no reference",
    }
    ref_text = ref_map[evidence.reference_status]
    parts = [
        f"vendor similarity {evidence.vendor_similarity:.0%}",
        f"amount difference {evidence.amount_difference}",
        f"{evidence.date_difference_days} day(s) apart",
        ref_text,
    ]
    return f"score={score:.2f} (" + ", ".join(parts) + ")"


def reconcile_invoice(
    invoice: Invoice,
    candidates: list[SettlementItem],
    thresholds: Thresholds,
) -> MatchPrediction:
    """candidates must already be blocking-filtered — this only scores
    and decides, it does not re-block."""
    if not candidates:
        return MatchPrediction(
            left_id=invoice.invoice_id,
            right_id="",
            confidence=0.0,
            status=MatchStatus.EXCEPTION,
            evidence=_NO_CANDIDATE_EVIDENCE,
            reason="No settlement item found within the amount/date blocking window.",
            exception_reason=ExceptionReason.NO_CANDIDATE_FOUND,
        )

    scored = []
    for item in candidates:
        ev = compute_evidence(invoice, item)
        conf = confidence(ev, invoice.amount)
        scored.append((item, ev, conf))
    scored.sort(key=lambda t: t[2], reverse=True)

    best_item, best_evidence, best_score = scored[0]
    second_score = scored[1][2] if len(scored) > 1 else 0.0
    ambiguous = len(scored) > 1 and (best_score - second_score) < AMBIGUITY_MARGIN

    if best_score >= thresholds.auto_match and not ambiguous:
        status = MatchStatus.AUTO_MATCHED
        exception_reason = None
    elif ambiguous and best_score >= thresholds.review_floor:
        status = MatchStatus.REVIEW
        exception_reason = ExceptionReason.AMBIGUOUS_CANDIDATES
    elif best_score >= thresholds.review_floor:
        status = MatchStatus.REVIEW
        exception_reason = (
            ExceptionReason.AMOUNT_MISMATCH if best_evidence.amount_difference != 0 else None
        )
    else:
        status = MatchStatus.EXCEPTION
        exception_reason = ExceptionReason.LOW_CONFIDENCE

    reason = _reason_text(best_evidence, best_score)
    if ambiguous:
        reason += f" — ambiguous: second-best candidate scored {second_score:.2f}"

    return MatchPrediction(
        left_id=invoice.invoice_id,
        right_id=best_item.payment_id,
        confidence=best_score,
        status=status,
        evidence=best_evidence,
        reason=reason,
        exception_reason=exception_reason,
    )


def reconcile_all(
    invoices: list[Invoice],
    settlement_items: list[SettlementItem],
    thresholds: Thresholds,
) -> list[MatchPrediction]:
    candidates_by_invoice = generate_candidates(invoices, settlement_items)
    return [
        reconcile_invoice(
            inv, [c.settlement_item for c in candidates_by_invoice[inv.invoice_id]], thresholds
        )
        for inv in invoices
    ]
