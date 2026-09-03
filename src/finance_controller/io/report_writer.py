"""Writers for reconciliation output: a per-invoice predictions CSV
(the full Stage 1 + AI review result set), a filtered exceptions CSV
(REVIEW + EXCEPTION only, the subset a human actually needs to look
at), a Stage 2 settlement-vs-bank-credit CSV, and a JSON run summary.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from finance_controller.domain.enums import MatchStatus
from finance_controller.domain.models import MatchPrediction
from finance_controller.matching.stage2 import SettlementResult

PREDICTION_FIELDS = (
    "invoice_id",
    "matched_payment_id",
    "status",
    "confidence",
    "exception_reason",
    "reason",
)

STAGE2_FIELDS = (
    "settlement_utr",
    "status",
    "expected_amount",
    "actual_amount",
    "exception_reason",
    "reason",
)


def write_predictions_csv(predictions: list[MatchPrediction], path: Path) -> None:
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=PREDICTION_FIELDS)
        w.writeheader()
        for p in predictions:
            w.writerow(
                {
                    "invoice_id": p.left_id,
                    "matched_payment_id": p.right_id or "",
                    "status": p.status.value,
                    "confidence": f"{p.confidence:.4f}",
                    "exception_reason": p.exception_reason.value if p.exception_reason else "",
                    "reason": p.reason,
                }
            )


def write_exceptions_csv(predictions: list[MatchPrediction], path: Path) -> None:
    """Subset of predictions with status REVIEW or EXCEPTION — the rows
    a human actually needs to act on."""
    needs_attention = [
        p for p in predictions if p.status in (MatchStatus.REVIEW, MatchStatus.EXCEPTION)
    ]
    write_predictions_csv(needs_attention, path)


def write_stage2_csv(results: list[SettlementResult], path: Path) -> None:
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=STAGE2_FIELDS)
        w.writeheader()
        for r in results:
            w.writerow(
                {
                    "settlement_utr": r["settlement_utr"],
                    "status": r["status"].value,
                    "expected_amount": str(r["expected_amount"]),
                    "actual_amount": str(r["actual_amount"])
                    if r["actual_amount"] is not None
                    else "",
                    "exception_reason": r["exception_reason"].value
                    if r["exception_reason"]
                    else "",
                    "reason": r["reason"],
                }
            )


def write_summary_json(summary: dict[str, Any], path: Path) -> None:
    path.write_text(json.dumps(summary, indent=2, default=str) + "\n")
