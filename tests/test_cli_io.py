"""Tests for the io/ layer (CSV loading, threshold persistence, report
writing) and the CLI subcommands built on top of them."""

from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path

import pytest

from finance_controller.cli import main as cli_main
from finance_controller.data.generator import generate
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
)
from finance_controller.io.thresholds_io import load_thresholds, save_thresholds
from finance_controller.matching.calibration import Thresholds
from finance_controller.matching.engine import reconcile_all


@pytest.fixture(scope="module")
def generated_dataset(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Generate once per test module — the generator is deterministic
    (fixed seed) so sharing this across tests is safe and keeps the
    suite fast."""
    out_dir = tmp_path_factory.mktemp("dataset")
    generate(out_dir)
    return out_dir


# ---------------------------------------------------------------------------
# CSV round-trip: what the generator writes, the loader must read back
# identically (field-for-field, not just "doesn't crash").
# ---------------------------------------------------------------------------


def test_load_invoices_round_trips_generator_output(generated_dataset: Path) -> None:
    invoices = load_invoices(generated_dataset / "dev" / "invoices.csv")
    assert len(invoices) == 180
    first = invoices[0]
    assert first.invoice_id.startswith("INV-")
    assert isinstance(first.amount, Decimal)
    assert first.amount > 0


def test_load_settlement_items_round_trips_generator_output(generated_dataset: Path) -> None:
    items = load_settlement_items(generated_dataset / "dev" / "settlement_items.csv")
    assert len(items) == 200
    # Confirm the deliberately-dropped-reference rows survive as None,
    # not the empty string the CSV actually stores.
    assert any(it.order_reference is None for it in items)
    assert all(isinstance(it.net_amount, Decimal) for it in items)


def test_load_bank_credits_round_trips_generator_output(generated_dataset: Path) -> None:
    credits = load_bank_credits(generated_dataset / "dev" / "bank_credits.csv")
    assert len(credits) > 0
    assert all(isinstance(c.amount, Decimal) for c in credits)


def test_load_match_labels_round_trips_generator_output(generated_dataset: Path) -> None:
    labels = load_match_labels(generated_dataset / "dev" / "match_labels.csv")
    assert len(labels) > 0
    assert any(lbl.is_match for lbl in labels)
    assert any(not lbl.is_match for lbl in labels)


# ---------------------------------------------------------------------------
# Error handling: malformed real-world CSVs should raise a pinpointed
# CSVLoadError, not a bare traceback from deep inside Decimal()/date().
# ---------------------------------------------------------------------------


def test_missing_column_raises_csv_load_error(tmp_path: Path) -> None:
    bad = tmp_path / "invoices.csv"
    bad.write_text("invoice_id,customer_name,amount,invoice_date\nINV-1,Acme,100,2026-01-01\n")
    with pytest.raises(CSVLoadError, match="order_reference"):
        load_invoices(bad)


def test_bad_decimal_raises_csv_load_error_with_row_number(tmp_path: Path) -> None:
    bad = tmp_path / "invoices.csv"
    bad.write_text(
        "invoice_id,order_reference,customer_name,amount,invoice_date,description\n"
        "INV-1,ORD-1,Acme,not_a_number,2026-01-01,\n"
    )
    with pytest.raises(CSVLoadError) as exc_info:
        load_invoices(bad)
    assert exc_info.value.row_number == 2
    assert exc_info.value.field == "amount"


def test_bad_date_raises_csv_load_error(tmp_path: Path) -> None:
    bad = tmp_path / "invoices.csv"
    bad.write_text(
        "invoice_id,order_reference,customer_name,amount,invoice_date,description\n"
        "INV-1,ORD-1,Acme,100,not-a-date,\n"
    )
    with pytest.raises(CSVLoadError, match="invoice_date"):
        load_invoices(bad)


def test_invalid_entity_type_raises_csv_load_error(tmp_path: Path) -> None:
    bad = tmp_path / "settlement_items.csv"
    bad.write_text(
        "entity_id,entity_type,order_reference,payment_id,customer_name,amount,fee,tax,"
        "net_amount,settlement_id,settlement_utr,created_at,settled_at\n"
        "pay_1,not_a_real_type,ORD-1,pay_1,Acme,100,1,1,98,SETL-1,UTR1,2026-01-01,2026-01-01\n"
    )
    with pytest.raises(CSVLoadError, match="entity_type"):
        load_settlement_items(bad)


def test_missing_file_raises_file_not_found(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_invoices(tmp_path / "does_not_exist.csv")


# ---------------------------------------------------------------------------
# Thresholds persistence round-trip.
# ---------------------------------------------------------------------------


def test_thresholds_round_trip(tmp_path: Path) -> None:
    original = Thresholds(auto_match=0.788, review_floor=0.687)
    path = tmp_path / "thresholds.json"
    save_thresholds(original, path)
    loaded = load_thresholds(path)
    assert loaded.auto_match == pytest.approx(original.auto_match)
    assert loaded.review_floor == pytest.approx(original.review_floor)


def test_load_thresholds_missing_key_raises(tmp_path: Path) -> None:
    path = tmp_path / "thresholds.json"
    path.write_text(json.dumps({"auto_match": 0.8}))
    with pytest.raises(ValueError, match="review_floor"):
        load_thresholds(path)


def test_load_thresholds_invalid_json_raises(tmp_path: Path) -> None:
    path = tmp_path / "thresholds.json"
    path.write_text("{not valid json")
    with pytest.raises(ValueError, match="not valid JSON"):
        load_thresholds(path)


# ---------------------------------------------------------------------------
# Report writers produce the expected shape.
# ---------------------------------------------------------------------------


def test_write_predictions_and_exceptions_csv(generated_dataset: Path, tmp_path: Path) -> None:
    from finance_controller.matching.calibration import calibrate

    invoices = load_invoices(generated_dataset / "held_out" / "invoices.csv")
    items = load_settlement_items(generated_dataset / "held_out" / "settlement_items.csv")
    dev_invoices = load_invoices(generated_dataset / "dev" / "invoices.csv")
    dev_items = load_settlement_items(generated_dataset / "dev" / "settlement_items.csv")
    dev_labels = load_match_labels(generated_dataset / "dev" / "match_labels.csv")

    thresholds = calibrate(dev_invoices, dev_items, dev_labels)
    predictions = reconcile_all(invoices, items, thresholds)

    pred_path = tmp_path / "predictions.csv"
    exc_path = tmp_path / "exceptions.csv"
    write_predictions_csv(predictions, pred_path)
    write_exceptions_csv(predictions, exc_path)

    assert pred_path.read_text().count("\n") == len(predictions) + 1  # header + rows
    exc_lines = exc_path.read_text().strip().splitlines()
    assert len(exc_lines) - 1 <= len(predictions)  # header + subset


def test_write_stage2_csv(generated_dataset: Path, tmp_path: Path) -> None:
    from finance_controller.matching.stage2 import reconcile_settlements

    items = load_settlement_items(generated_dataset / "held_out" / "settlement_items.csv")
    credits = load_bank_credits(generated_dataset / "held_out" / "bank_credits.csv")
    results = reconcile_settlements(items, credits)

    out_path = tmp_path / "stage2.csv"
    write_stage2_csv(results, out_path)
    assert out_path.read_text().count("\n") == len(results) + 1


# ---------------------------------------------------------------------------
# CLI end to end — the exact commands documented in cli.py's docstring.
# ---------------------------------------------------------------------------


def test_cli_generate_calibrate_reconcile_end_to_end(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    thresholds_path = tmp_path / "thresholds.json"
    results_dir = tmp_path / "results"

    assert cli_main(["generate", "--out-dir", str(data_dir)]) == 0
    assert (data_dir / "dev" / "invoices.csv").exists()
    assert (data_dir / "held_out" / "bank_credits.csv").exists()

    assert (
        cli_main(
            [
                "calibrate",
                "--invoices", str(data_dir / "dev" / "invoices.csv"),
                "--settlement-items", str(data_dir / "dev" / "settlement_items.csv"),
                "--labels", str(data_dir / "dev" / "match_labels.csv"),
                "--out", str(thresholds_path),
            ]
        )
        == 0
    )
    assert thresholds_path.exists()

    assert (
        cli_main(
            [
                "reconcile",
                "--invoices", str(data_dir / "held_out" / "invoices.csv"),
                "--settlement-items", str(data_dir / "held_out" / "settlement_items.csv"),
                "--bank-credits", str(data_dir / "held_out" / "bank_credits.csv"),
                "--thresholds", str(thresholds_path),
                "--out-dir", str(results_dir),
            ]
        )
        == 0
    )

    assert (results_dir / "predictions.csv").exists()
    assert (results_dir / "exceptions.csv").exists()
    assert (results_dir / "stage2.csv").exists()

    summary = json.loads((results_dir / "summary.json").read_text())
    assert summary["total_invoices"] == 45
    assert summary["auto_matched"] == 34
    assert summary["ai_review_used"] is False


def test_cli_reconcile_missing_file_exits_nonzero(tmp_path: Path) -> None:
    exit_code = cli_main(
        [
            "reconcile",
            "--invoices", str(tmp_path / "nope.csv"),
            "--settlement-items", str(tmp_path / "nope2.csv"),
            "--bank-credits", str(tmp_path / "nope3.csv"),
            "--thresholds", str(tmp_path / "nope4.json"),
            "--out-dir", str(tmp_path / "out"),
        ]
    )
    assert exit_code == 1


def test_cli_use_ai_without_api_key_warns_but_succeeds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    data_dir = tmp_path / "data"
    thresholds_path = tmp_path / "thresholds.json"
    results_dir = tmp_path / "results"

    cli_main(["generate", "--out-dir", str(data_dir)])
    cli_main(
        [
            "calibrate",
            "--invoices", str(data_dir / "dev" / "invoices.csv"),
            "--settlement-items", str(data_dir / "dev" / "settlement_items.csv"),
            "--labels", str(data_dir / "dev" / "match_labels.csv"),
            "--out", str(thresholds_path),
        ]
    )
    exit_code = cli_main(
        [
            "reconcile",
            "--invoices", str(data_dir / "held_out" / "invoices.csv"),
            "--settlement-items", str(data_dir / "held_out" / "settlement_items.csv"),
            "--bank-credits", str(data_dir / "held_out" / "bank_credits.csv"),
            "--thresholds", str(thresholds_path),
            "--out-dir", str(results_dir),
            "--use-ai",
        ]
    )
    assert exit_code == 0
    summary = json.loads((results_dir / "summary.json").read_text())
    assert summary["ai_review_used"] is False
