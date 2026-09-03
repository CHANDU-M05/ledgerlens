"""Calibrate on dev -> Stage 1 match -> AI review of REVIEW cases ->
Stage 2 arithmetic -> metrics. Run:
    PYTHONPATH=src .venv/bin/python scripts/run_reconciliation_check.py

Uses a real Anthropic client if ANTHROPIC_API_KEY is set; otherwise
uses a client that always fails, which exercises the actual
failure-recovery path — a legitimate, honest demo mode, not a
degraded one, since the fallback behavior IS the feature being shown.
"""

from __future__ import annotations

import os

from finance_controller.ai.pipeline import apply_ai_review
from finance_controller.data.generator import generate
from finance_controller.evaluation.metrics import evaluate
from finance_controller.matching.calibration import calibrate
from finance_controller.matching.engine import reconcile_all
from finance_controller.matching.stage2 import reconcile_settlements


class _AlwaysFailsClient:
    """Stand-in for the LLM when no API key is configured. Every call
    raises, forcing every REVIEW case through the fallback path."""

    def complete(self, prompt: str) -> str:
        raise RuntimeError("No ANTHROPIC_API_KEY configured for this run")


def _build_client() -> object:
    if os.environ.get("ANTHROPIC_API_KEY"):
        from finance_controller.ai.reviewer import AnthropicLLMClient

        return AnthropicLLMClient()
    print("[no ANTHROPIC_API_KEY set — running with the failure-recovery fallback client]\n")
    return _AlwaysFailsClient()


def main() -> None:
    dev, held = generate()
    thresholds = calibrate(dev.invoices, dev.settlement_items, dev.match_labels)
    print(
        f"Calibrated thresholds: auto_match={thresholds.auto_match:.3f}, "
        f"review_floor={thresholds.review_floor:.3f}\n"
    )

    predictions = reconcile_all(held.invoices, held.settlement_items, thresholds)

    client = _build_client()
    predictions = apply_ai_review(predictions, held.invoices, held.settlement_items, client)

    metrics = evaluate(predictions, held.match_labels)
    print(f"Held-out results ({metrics.total} invoices):")
    print(f"  Auto-matched (deterministic): {metrics.auto_matched}")
    print(f"  AI-approved:                  {metrics.ai_approved}")
    print(f"  Still REVIEW (human needed):  {metrics.review}")
    print(f"  Exception:                    {metrics.exception}")
    print(f"  Auto-match precision:  {metrics.auto_match_precision:.1%}")
    print(f"  AI-approved precision: {metrics.ai_approved_precision:.1%}")
    print(f"  Combined precision:    {metrics.combined_precision:.1%}")
    print(f"  Recall:                {metrics.recall:.1%}")
    print(f"  Resolved without a human: {metrics.combined_resolution_rate:.1%}")

    stage2 = reconcile_settlements(held.settlement_items, held.bank_credits)
    ok = sum(1 for r in stage2 if r["status"] == r["status"].AUTO_MATCHED)
    print(f"\nStage 2 batches: {ok}/{len(stage2)} sum exactly to bank credit")


if __name__ == "__main__":
    main()
