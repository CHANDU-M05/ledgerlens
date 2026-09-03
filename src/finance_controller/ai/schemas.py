"""Structured contract for the AI reviewer's output.

The model NEVER supplies numeric evidence into the final report — it
returns only a decision and a purely qualitative rationale. Numbers are
banned from the rationale entirely (enforced in reviewer.py), not just
checked against the evidence, because that's a simpler guardrail with
no tolerance-matching judgment calls to defend later: if the model
writes any digit, the response is rejected and treated as unavailable.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class AIReviewOutcome(BaseModel):
    """Parsed, validated response from the LLM reviewer.

    decision:
        "approve"  — evidence, on balance, supports resolving this
                     without a human.
        "escalate" — evidence is genuinely ambiguous; a human should
                     look at it.
    rationale:
        One or two sentences, qualitative only. No digits permitted —
        checked by reviewer.py, not by this schema (Pydantic validates
        shape; the digit-ban is a semantic guardrail applied after).
    """

    decision: Literal["approve", "escalate"]
    rationale: str = Field(min_length=1, max_length=400)
