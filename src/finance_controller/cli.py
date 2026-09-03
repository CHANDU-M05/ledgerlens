"""LedgerLens command-line interface.

Three subcommands, matching how this actually has to work in
production — calibration needs labeled data, reconciliation must stay
blind to it:

    ledgerlens generate   --out-dir data/
        Wraps data.generator.generate(). Writes dev/ and held_out/
        CSVs (invoices, settlement_items, bank_credits, match_labels)
        for testing the pipeline end to end without real data.

    ledgerlens calibrate  --invoices ... --settlement-items ... --labels ... --out thresholds.json
        Computes auto_match / review_floor thresholds from a labeled
        dataset (synthetic dev split, or a historical batch someone
        has hand-verified) and saves them to a JSON file. Run this
        once; reuse the output indefinitely.

    ledgerlens reconcile  --invoices ... --settlement-items ...
                          --bank-credits ... --thresholds thresholds.json --out-dir results/
        Runs Stage 1 (fuzzy match) + optional AI review + Stage 2
        (settlement-sum arithmetic) against real, UNLABELED data and
        writes predictions.csv, exceptions.csv, stage2.csv, and
        summary.json to --out-dir. Never reads or requires a
        match_labels file — that would defeat the point.

Run `ledgerlens <subcommand> --help` for the full flag list.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from finance_controller.ai.reviewer import LLMClient
from finance_controller.domain.enums import MatchStatus
from finance_controller.io.csv_loader import (
    CSVLoadError,
    load_bank_credits,
    load_invoices,
    load_match_labels,
    load_settlement_items,
)
from finance_controller.io.report_writer import (
    write_exceptions_csv,
    write_predictions_csv,
    write_stage2_csv,
    write_summary_json,
)
from finance_controller.io.thresholds_io import load_thresholds, save_thresholds
from finance_controller.matching.calibration import calibrate as calibrate_thresholds
from finance_controller.matching.engine import reconcile_all
from finance_controller.matching.stage2 import reconcile_settlements


def _build_ai_client(use_ai: bool) -> LLMClient | None:
    if not use_ai:
        return None
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print(
            "warning: --use-ai given but ANTHROPIC_API_KEY is not set; "
            "REVIEW-status predictions will stay REVIEW.",
            file=sys.stderr,
        )
        return None
    from finance_controller.ai.reviewer import AnthropicLLMClient

    return AnthropicLLMClient()


def cmd_generate(args: argparse.Namespace) -> int:
    from finance_controller.data.generator import generate

    out_dir = Path(args.out_dir)
    dev, held = generate(out_dir)
    print(f"Wrote synthetic dataset to {out_dir}/")
    print(
        f"  dev:       {len(dev.invoices)} invoices, {len(dev.settlement_items)} settlement items"
    )
    print(
        f"  held_out:  {len(held.invoices)} invoices, {len(held.settlement_items)} settlement items"
    )
    return 0


def cmd_calibrate(args: argparse.Namespace) -> int:
    try:
        invoices = load_invoices(Path(args.invoices))
        settlement_items = load_settlement_items(Path(args.settlement_items))
        labels = load_match_labels(Path(args.labels))
    except (CSVLoadError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    thresholds = calibrate_thresholds(invoices, settlement_items, labels)
    out_path = Path(args.out)
    save_thresholds(thresholds, out_path)
    print(f"Calibrated thresholds from {len(labels)} labeled pairs:")
    print(f"  auto_match   = {thresholds.auto_match:.4f}")
    print(f"  review_floor = {thresholds.review_floor:.4f}")
    print(f"Saved to {out_path}")
    return 0


def cmd_reconcile(args: argparse.Namespace) -> int:
    try:
        invoices = load_invoices(Path(args.invoices))
        settlement_items = load_settlement_items(Path(args.settlement_items))
        bank_credits = load_bank_credits(Path(args.bank_credits))
        thresholds = load_thresholds(Path(args.thresholds))
    except (CSVLoadError, FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    predictions = reconcile_all(invoices, settlement_items, thresholds)

    client = _build_ai_client(args.use_ai)
    if client is not None:
        from finance_controller.ai.pipeline import apply_ai_review

        predictions = apply_ai_review(predictions, invoices, settlement_items, client)

    stage2_results = reconcile_settlements(settlement_items, bank_credits)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_predictions_csv(predictions, out_dir / "predictions.csv")
    write_exceptions_csv(predictions, out_dir / "exceptions.csv")
    write_stage2_csv(stage2_results, out_dir / "stage2.csv")

    counts = {status: 0 for status in MatchStatus}
    for p in predictions:
        counts[p.status] += 1
    stage2_ok = sum(1 for r in stage2_results if r["status"] == MatchStatus.AUTO_MATCHED)

    summary = {
        "total_invoices": len(predictions),
        "auto_matched": counts[MatchStatus.AUTO_MATCHED],
        "ai_approved": counts[MatchStatus.AI_APPROVED],
        "review": counts[MatchStatus.REVIEW],
        "exception": counts[MatchStatus.EXCEPTION],
        "resolved_without_human": counts[MatchStatus.AUTO_MATCHED]
        + counts[MatchStatus.AI_APPROVED],
        "stage2_batches_total": len(stage2_results),
        "stage2_batches_matched": stage2_ok,
        "ai_review_used": client is not None,
    }
    write_summary_json(summary, out_dir / "summary.json")

    print(f"Reconciled {summary['total_invoices']} invoices:")
    print(f"  auto_matched: {summary['auto_matched']}")
    print(f"  ai_approved:  {summary['ai_approved']}")
    print(f"  review:       {summary['review']}")
    print(f"  exception:    {summary['exception']}")
    print(f"  Stage 2: {stage2_ok}/{len(stage2_results)} settlement batches match bank credits")
    print(f"\nWrote predictions.csv, exceptions.csv, stage2.csv, summary.json to {out_dir}/")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ledgerlens",
        description="LedgerLens: two-stage financial reconciliation (fuzzy match + arithmetic).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_gen = sub.add_parser("generate", help="Generate a synthetic dev/held-out dataset.")
    p_gen.add_argument(
        "--out-dir", required=True, help="Directory to write dev/ and held_out/ CSVs into."
    )
    p_gen.set_defaults(func=cmd_generate)

    p_cal = sub.add_parser(
        "calibrate", help="Compute auto_match/review_floor thresholds from labeled data."
    )
    p_cal.add_argument("--invoices", required=True, help="Path to invoices.csv")
    p_cal.add_argument("--settlement-items", required=True, help="Path to settlement_items.csv")
    p_cal.add_argument("--labels", required=True, help="Path to match_labels.csv (ground truth)")
    p_cal.add_argument("--out", required=True, help="Path to write thresholds.json")
    p_cal.set_defaults(func=cmd_calibrate)

    p_rec = sub.add_parser("reconcile", help="Run reconciliation against real, unlabeled data.")
    p_rec.add_argument("--invoices", required=True, help="Path to invoices.csv")
    p_rec.add_argument("--settlement-items", required=True, help="Path to settlement_items.csv")
    p_rec.add_argument("--bank-credits", required=True, help="Path to bank_credits.csv")
    p_rec.add_argument(
        "--thresholds", required=True, help="Path to thresholds.json (from `calibrate`)"
    )
    p_rec.add_argument("--out-dir", required=True, help="Directory to write result CSVs/JSON into.")
    p_rec.add_argument(
        "--use-ai",
        action="store_true",
        help="Send REVIEW-status predictions to the AI reviewer (requires ANTHROPIC_API_KEY).",
    )
    p_rec.set_defaults(func=cmd_reconcile)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
