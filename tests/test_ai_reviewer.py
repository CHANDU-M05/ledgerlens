"""Tests for the AI review layer. FakeLLMClient stands in for the real
Anthropic client so these are fast, free, deterministic, and require
no API key."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from decimal import Decimal

from finance_controller.ai.pipeline import apply_ai_review
from finance_controller.ai.reviewer import review_prediction
from finance_controller.domain.enums import EntityType, ExceptionReason, MatchStatus
from finance_controller.domain.models import (
    Invoice,
    MatchEvidence,
    MatchPrediction,
    SettlementItem,
)


@dataclass
class FakeLLMClient:
    """Returns a fixed response or raises a fixed exception, regardless
    of prompt content — deterministic stand-in for the real API."""

    response: str | None = None
    exception: Exception | None = None

    def complete(self, prompt: str) -> str:
        if self.exception is not None:
            raise self.exception
        assert self.response is not None
        return self.response


def _make_review_prediction() -> tuple[MatchPrediction, Invoice, SettlementItem]:
    invoice = Invoice(
        invoice_id="INV-TEST",
        order_reference="ORD-TEST",
        customer_name="Test Vendor Pvt Ltd",
        amount=Decimal("10000"),
        invoice_date=date(2026, 8, 1),
    )
    item = SettlementItem(
        entity_id="pay_TEST",
        entity_type=EntityType.PAYMENT,
        order_reference="ORD-TEST",
        payment_id="pay_TEST",
        customer_name="TEST VENDOR PVT LTD",
        amount=Decimal("9500"),
        fee=Decimal("190"),
        tax=Decimal("34.2"),
        net_amount=Decimal("9275.8"),
        settlement_id="SETL-1",
        settlement_utr="UTR1",
        created_at=date(2026, 8, 2),
    )
    evidence = MatchEvidence(
        vendor_similarity=1.0,
        amount_difference=Decimal("-500"),
        date_difference_days=1,
        reference_status="match",
    )
    prediction = MatchPrediction(
        left_id=invoice.invoice_id,
        right_id=item.payment_id,
        confidence=0.75,
        status=MatchStatus.REVIEW,
        evidence=evidence,
        reason="score=0.75 (vendor similarity 100%, amount difference -500, "
        "1 day(s) apart, reference matches)",
    )
    return prediction, invoice, item


def test_approve_path_sets_ai_approved_status() -> None:
    prediction, invoice, item = _make_review_prediction()
    client = FakeLLMClient(
        response='{"decision": "approve", "rationale": '
        '"Vendor names match closely and the amount gap looks like a '
        'plausible partial payment."}'
    )
    result = review_prediction(prediction, invoice, item, client)
    assert result.status == MatchStatus.AI_APPROVED
    assert result.exception_reason is None
    assert "AI review: approved" in result.reason


def test_escalate_path_stays_review_with_ambiguous_reason() -> None:
    prediction, invoice, item = _make_review_prediction()
    client = FakeLLMClient(
        response='{"decision": "escalate", "rationale": '
        '"The amount gap is large enough that a human should confirm it."}'
    )
    result = review_prediction(prediction, invoice, item, client)
    assert result.status == MatchStatus.REVIEW
    assert result.exception_reason == ExceptionReason.AMBIGUOUS_CANDIDATES


def test_client_exception_falls_back_to_review_with_ai_unavailable() -> None:
    prediction, invoice, item = _make_review_prediction()
    client = FakeLLMClient(exception=TimeoutError("simulated timeout"))
    result = review_prediction(prediction, invoice, item, client)
    assert result.status == MatchStatus.REVIEW
    assert result.exception_reason == ExceptionReason.AI_REVIEW_UNAVAILABLE
    assert "AI review unavailable" in result.reason


def test_malformed_json_falls_back_to_review_with_ai_unavailable() -> None:
    prediction, invoice, item = _make_review_prediction()
    client = FakeLLMClient(response="not json at all")
    result = review_prediction(prediction, invoice, item, client)
    assert result.status == MatchStatus.REVIEW
    assert result.exception_reason == ExceptionReason.AI_REVIEW_UNAVAILABLE


def test_invalid_decision_value_falls_back() -> None:
    prediction, invoice, item = _make_review_prediction()
    client = FakeLLMClient(response='{"decision": "maybe", "rationale": "unsure"}')
    result = review_prediction(prediction, invoice, item, client)
    assert result.status == MatchStatus.REVIEW
    assert result.exception_reason == ExceptionReason.AI_REVIEW_UNAVAILABLE


def test_digit_in_rationale_is_rejected() -> None:
    """The digit ban is absolute — even a plausible-looking number must
    be rejected, since the guardrail's whole point is not having to
    judge whether a given number is legitimate."""
    prediction, invoice, item = _make_review_prediction()
    client = FakeLLMClient(
        response='{"decision": "approve", "rationale": '
        '"The amount differs by about 500 which seems fine."}'
    )
    result = review_prediction(prediction, invoice, item, client)
    assert result.status == MatchStatus.REVIEW
    assert result.exception_reason == ExceptionReason.AI_REVIEW_UNAVAILABLE
    assert "disallowed number" in result.reason


def test_non_review_predictions_pass_through_untouched() -> None:
    """review_prediction is a no-op for AUTO_MATCHED/EXCEPTION — the
    client must never even be invoked for those."""
    prediction, invoice, item = _make_review_prediction()
    already_matched = MatchPrediction(
        left_id=prediction.left_id,
        right_id=prediction.right_id,
        confidence=0.95,
        status=MatchStatus.AUTO_MATCHED,
        evidence=prediction.evidence,
        reason="deterministic auto-match",
    )
    client = FakeLLMClient(exception=RuntimeError("must never be called"))
    result = review_prediction(already_matched, invoice, item, client)
    assert result is already_matched


def test_pipeline_only_sends_review_status_to_the_client() -> None:
    prediction, invoice, item = _make_review_prediction()
    other = MatchPrediction(
        left_id="INV-OTHER",
        right_id="pay_OTHER",
        confidence=0.95,
        status=MatchStatus.AUTO_MATCHED,
        evidence=prediction.evidence,
        reason="deterministic",
    )
    client = FakeLLMClient(
        response='{"decision": "approve", "rationale": "All signals align well."}'
    )
    results = apply_ai_review([prediction, other], [invoice], [item], client)
    statuses = {r.left_id: r.status for r in results}
    assert statuses["INV-TEST"] == MatchStatus.AI_APPROVED
    assert statuses["INV-OTHER"] == MatchStatus.AUTO_MATCHED
