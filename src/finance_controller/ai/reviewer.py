"""AI review layer for MatchStatus.REVIEW predictions.

Design contract, enforced in code, not just documented:

1. The LLM receives the pre-computed MatchEvidence plus enough context
   (vendor names, amounts, dates) to reason in natural language — it is
   never asked to compute or restate a similarity score, a date
   difference, or an amount delta itself.
2. The rationale must contain ZERO digits. This is deliberately
   stricter than fuzzy-matching numbers against the evidence: it has no
   tolerance threshold to tune or defend, and its only failure mode is
   an occasional unnecessary escalation to human review when a model
   writes an incidental number ("the two records") — an acceptable
   cost given the alternative (a fabricated evidence number reaching
   the report) is not acceptable at all.
3. ANY failure — timeout, malformed JSON, schema validation failure,
   missing API key, a digit in the rationale, an empty response —
   routes to the SAME fallback: keep MatchStatus.REVIEW, attach
   ExceptionReason.AI_REVIEW_UNAVAILABLE, and preserve the deterministic
   evidence untouched. The system never guesses when the AI layer
   fails; it degrades to "ask a human," which is always safe here.
"""

from __future__ import annotations

import json
from dataclasses import replace
from typing import Protocol

from pydantic import ValidationError

from finance_controller.ai.schemas import AIReviewOutcome
from finance_controller.domain.enums import ExceptionReason, MatchStatus
from finance_controller.domain.models import Invoice, MatchPrediction, SettlementItem


class LLMClient(Protocol):
    """Anything with this shape can drive the reviewer — production
    Anthropic client or a deterministic fake in tests, no code fork."""

    def complete(self, prompt: str) -> str: ...


class AnthropicLLMClient:
    """Thin wrapper around the Anthropic API. anthropic is imported
    lazily inside __init__ so importing this MODULE never requires the
    package or an API key — only instantiating this class does."""

    def __init__(
        self,
        model: str = "claude-haiku-4-5-20251001",
        timeout_seconds: float = 8.0,
    ) -> None:
        import anthropic

        self._client = anthropic.Anthropic(timeout=timeout_seconds)
        self._model = model

    def complete(self, prompt: str) -> str:
        response = self._client.messages.create(
            model=self._model,
            max_tokens=300,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(block.text for block in response.content if block.type == "text")


def _build_prompt(prediction: MatchPrediction, invoice: Invoice, item: SettlementItem) -> str:
    ev = prediction.evidence
    return f"""You are reviewing one candidate match in a financial reconciliation system.

Invoice: vendor="{invoice.customer_name}", amount={invoice.amount}, date={invoice.invoice_date}
Settlement item: vendor="{item.customer_name}", amount={item.amount}, date={item.created_at}

Evidence (already computed by deterministic code — do not recompute or contradict it):
- vendor_similarity: {ev.vendor_similarity:.2f}
- amount_difference: {ev.amount_difference}
- date_difference_days: {ev.date_difference_days}
- reference_status: {ev.reference_status}
- current_score: {prediction.confidence:.3f}

Decide whether this pair should be "approve"d as a genuine match resolved
without human review, or "escalate"d to a human.

IMPORTANT: Your rationale must contain NO digits, percentages, or amounts.
Describe your reasoning qualitatively only (e.g. "vendor names are nearly
identical and the amounts are close" rather than restating any number).
The numeric evidence is already recorded separately and must not be
repeated by you.

Respond with ONLY a JSON object, no other text, in this exact shape:
{{"decision": "approve" or "escalate", "rationale": "<qualitative reasoning, no digits>"}}
"""


def _parse_response(raw: str) -> AIReviewOutcome:
    """Raises on any malformed or non-conforming response — caller
    treats every exception here identically as an AI-unavailable event."""
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text.removeprefix("json").strip()
    data = json.loads(text)
    return AIReviewOutcome.model_validate(data)


def _rationale_has_any_digit(rationale: str) -> bool:
    return any(ch.isdigit() for ch in rationale)


def _fallback(prediction: MatchPrediction, detail: str) -> MatchPrediction:
    return replace(
        prediction,
        status=MatchStatus.REVIEW,
        exception_reason=ExceptionReason.AI_REVIEW_UNAVAILABLE,
        reason=prediction.reason
        + f" | AI review unavailable ({detail}); human verification required.",
    )


def review_prediction(
    prediction: MatchPrediction,
    invoice: Invoice,
    item: SettlementItem,
    client: LLMClient,
) -> MatchPrediction:
    """Attempt AI review of one REVIEW-status prediction.

    Returns a NEW MatchPrediction — never mutates the input. Only ever
    acts on MatchStatus.REVIEW; anything else passes through unchanged
    with the client never invoked, so a batch of mixed statuses never
    burns API calls on decisions the deterministic engine already made.
    """
    if prediction.status != MatchStatus.REVIEW:
        return prediction

    try:
        raw = client.complete(_build_prompt(prediction, invoice, item))
    except Exception as exc:  # noqa: BLE001 — deliberately broad: ANY
        # client failure (timeout, network, auth, rate limit) degrades
        # identically, by design, per the module's failure contract.
        return _fallback(prediction, f"request failed: {exc.__class__.__name__}")

    try:
        outcome = _parse_response(raw)
    except (json.JSONDecodeError, ValidationError) as exc:
        return _fallback(prediction, f"malformed response: {exc.__class__.__name__}")

    if _rationale_has_any_digit(outcome.rationale):
        return _fallback(prediction, "response contained a disallowed number")

    if outcome.decision == "approve":
        return replace(
            prediction,
            status=MatchStatus.AI_APPROVED,
            exception_reason=None,
            reason=prediction.reason + f" | AI review: approved — {outcome.rationale}",
        )

    return replace(
        prediction,
        status=MatchStatus.REVIEW,
        exception_reason=ExceptionReason.AMBIGUOUS_CANDIDATES,
        reason=prediction.reason + f" | AI review: escalated — {outcome.rationale}",
    )
