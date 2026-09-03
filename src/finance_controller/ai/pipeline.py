"""Batch entry point: runs AI review over every REVIEW-status prediction
in a Stage 1 result set. AUTO_MATCHED and EXCEPTION predictions pass
through untouched and never reach the LLM client."""

from __future__ import annotations

from finance_controller.ai.reviewer import LLMClient, review_prediction
from finance_controller.domain.enums import MatchStatus
from finance_controller.domain.models import Invoice, MatchPrediction, SettlementItem


def apply_ai_review(
    predictions: list[MatchPrediction],
    invoices: list[Invoice],
    settlement_items: list[SettlementItem],
    client: LLMClient,
) -> list[MatchPrediction]:
    invoice_by_id = {inv.invoice_id: inv for inv in invoices}
    item_by_id = {it.payment_id: it for it in settlement_items}

    results: list[MatchPrediction] = []
    for pred in predictions:
        if pred.status != MatchStatus.REVIEW:
            results.append(pred)
            continue
        invoice = invoice_by_id.get(pred.left_id)
        item = item_by_id.get(pred.right_id)
        if invoice is None or item is None:
            # Should be unreachable given how engine.py builds
            # predictions, but degrade the same way any AI failure
            # degrades rather than raising mid-batch.
            results.append(pred)
            continue
        results.append(review_prediction(pred, invoice, item, client))
    return results
