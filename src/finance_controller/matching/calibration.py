"""Thresholds derived from the DEV split's labeled score distribution —
never hardcoded. Rerun calibrate() whenever the generator's parameters
change and the thresholds move with it. This is the direct answer to
"why these numbers": because this is where true-pair and decoy-pair
scores actually land on THIS dataset, not an assertion made in advance.
"""

from __future__ import annotations

from dataclasses import dataclass

from finance_controller.domain.models import Invoice, MatchLabel, SettlementItem
from finance_controller.matching.scorer import compute_evidence, confidence


@dataclass(frozen=True, slots=True)
class Thresholds:
    auto_match: float
    review_floor: float  # below this: EXCEPTION, no AI review attempted


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    idx = min(len(values) - 1, max(0, round(pct * (len(values) - 1))))
    return values[idx]


def calibrate(
    invoices: list[Invoice],
    settlement_items: list[SettlementItem],
    labels: list[MatchLabel],
) -> Thresholds:
    """auto_match = 5th percentile of TRUE-pair scores (~95% of genuine
    matches clear the bar). review_floor = 95th percentile of DECOY
    scores (~95% of decoys fall below where we'd even send them to AI
    review). Between the two: AI_REVIEW. Below review_floor: EXCEPTION.
    """
    invoice_by_id = {inv.invoice_id: inv for inv in invoices}
    item_by_id = {it.payment_id: it for it in settlement_items}

    true_scores: list[float] = []
    false_scores: list[float] = []

    for lbl in labels:
        inv = invoice_by_id.get(lbl.left_id)
        item = item_by_id.get(lbl.right_id)
        if inv is None or item is None:
            continue
        score = confidence(compute_evidence(inv, item), inv.amount)
        (true_scores if lbl.is_match else false_scores).append(score)

    auto_match = _percentile(true_scores, 0.05)
    review_floor = _percentile(false_scores, 0.95)

    if review_floor >= auto_match:
        # Distributions overlap more than expected on this dataset —
        # collapse to a single safe midpoint rather than produce an
        # inverted (unusable) threshold pair.
        midpoint = (review_floor + auto_match) / 2
        return Thresholds(auto_match=midpoint + 0.01, review_floor=midpoint - 0.01)

    return Thresholds(auto_match=auto_match, review_floor=review_floor)
