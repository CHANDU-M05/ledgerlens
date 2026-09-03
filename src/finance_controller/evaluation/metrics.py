"""Evaluation against held-out ground truth. matching/* and ai/* must
NEVER import from here or read match_labels — that boundary is what
keeps both the deterministic engine and the AI reviewer honestly blind
to the answer key."""

from __future__ import annotations

from dataclasses import dataclass

from finance_controller.domain.enums import MatchStatus
from finance_controller.domain.models import MatchLabel, MatchPrediction


@dataclass(frozen=True, slots=True)
class Metrics:
    total: int
    auto_matched: int
    ai_approved: int
    review: int
    exception: int
    auto_match_precision: float
    """Precision of deterministic AUTO_MATCHED predictions only."""
    ai_approved_precision: float
    """Precision of AI_APPROVED predictions only — reported separately
    from auto_match_precision so a reviewer can see the AI layer's
    contribution isn't hidden inside a blended number."""
    combined_precision: float
    """Precision across AUTO_MATCHED + AI_APPROVED together."""
    recall: float
    auto_match_rate: float
    combined_resolution_rate: float
    """(auto_matched + ai_approved) / total — the fraction of invoices
    resolved without a human, combining both resolution paths."""


def _precision_of(
    preds: list[MatchPrediction], label_by_pair: dict[tuple[str, str], bool]
) -> float:
    if not preds:
        return 0.0
    correct = sum(1 for p in preds if label_by_pair.get((p.left_id, p.right_id)) is True)
    return correct / len(preds)


def evaluate(predictions: list[MatchPrediction], labels: list[MatchLabel]) -> Metrics:
    label_by_pair = {(lbl.left_id, lbl.right_id): lbl.is_match for lbl in labels}
    true_invoice_ids = {lbl.left_id for lbl in labels if lbl.is_match}

    auto_matched = [p for p in predictions if p.status == MatchStatus.AUTO_MATCHED]
    ai_approved = [p for p in predictions if p.status == MatchStatus.AI_APPROVED]
    review = [p for p in predictions if p.status == MatchStatus.REVIEW]
    exception = [p for p in predictions if p.status == MatchStatus.EXCEPTION]

    combined = auto_matched + ai_approved
    correct_combined = sum(
        1 for p in combined if label_by_pair.get((p.left_id, p.right_id)) is True
    )
    recall = correct_combined / len(true_invoice_ids) if true_invoice_ids else 0.0

    return Metrics(
        total=len(predictions),
        auto_matched=len(auto_matched),
        ai_approved=len(ai_approved),
        review=len(review),
        exception=len(exception),
        auto_match_precision=_precision_of(auto_matched, label_by_pair),
        ai_approved_precision=_precision_of(ai_approved, label_by_pair),
        combined_precision=(correct_combined / len(combined)) if combined else 0.0,
        recall=recall,
        auto_match_rate=len(auto_matched) / len(predictions) if predictions else 0.0,
        combined_resolution_rate=len(combined) / len(predictions) if predictions else 0.0,
    )
