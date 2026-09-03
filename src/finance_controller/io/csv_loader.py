"""Readers for the four CSV shapes this project produces and consumes:
invoices, settlement_items, bank_credits, match_labels.

Mirrors the field names and encoding used by
``data.generator._write_*`` exactly, so any CSV this project writes can
be read back, and any real-world export in the same shape (e.g. an
invoices.csv exported from an accounting tool, a settlement report
exported from Razorpay, a bank statement export) can be loaded as long
as the header matches.

Every parse error is raised as a `CSVLoadError` carrying the file path,
1-indexed row number, and offending value — real CSVs from finance
systems are messy (blank amounts, stray currency symbols, malformed
dates), and a bare `ValueError` from deep inside `Decimal()` or
`date.fromisoformat()` is not actionable for whoever is staring at the
CLI output.
"""

from __future__ import annotations

import csv
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from decimal import Decimal, InvalidOperation
from pathlib import Path

from finance_controller.domain.enums import EntityType, Source
from finance_controller.domain.models import (
    BankCredit,
    Invoice,
    MatchLabel,
    SettlementItem,
)

INVOICE_FIELDS = (
    "invoice_id",
    "order_reference",
    "customer_name",
    "amount",
    "invoice_date",
    "description",
)
SETTLEMENT_ITEM_FIELDS = (
    "entity_id",
    "entity_type",
    "order_reference",
    "payment_id",
    "customer_name",
    "amount",
    "fee",
    "tax",
    "net_amount",
    "settlement_id",
    "settlement_utr",
    "created_at",
    "settled_at",
)
BANK_CREDIT_FIELDS = ("credit_id", "utr", "amount", "value_date", "narration")
MATCH_LABEL_FIELDS = ("left_id", "left_source", "right_id", "right_source", "is_match")


@dataclass
class CSVLoadError(Exception):
    """Raised for any row that cannot be parsed into a domain model.

    Carries enough context (file, row, field, raw value) to point
    directly at the bad cell rather than making the user re-derive it
    from a generic traceback.
    """

    path: Path
    row_number: int
    field: str
    raw_value: str
    detail: str

    def __str__(self) -> str:
        return (
            f"{self.path}:{self.row_number}: column {self.field!r}="
            f"{self.raw_value!r} — {self.detail}"
        )


def _require_headers(
    path: Path, fieldnames: Sequence[str] | None, expected: tuple[str, ...]
) -> None:
    if fieldnames is None:
        raise CSVLoadError(path, 0, "<header>", "", "file is empty — no header row found")
    missing = [f for f in expected if f not in fieldnames]
    if missing:
        raise CSVLoadError(
            path,
            1,
            "<header>",
            ",".join(fieldnames),
            f"missing required column(s): {', '.join(missing)}",
        )


def _parse_decimal(path: Path, row_number: int, field: str, raw: str) -> Decimal:
    cleaned = raw.strip().replace(",", "").replace("₹", "").replace("Rs.", "").strip()
    if not cleaned:
        raise CSVLoadError(path, row_number, field, raw, "amount field is blank")
    try:
        return Decimal(cleaned)
    except InvalidOperation as exc:
        raise CSVLoadError(
            path, row_number, field, raw, f"not a valid decimal amount ({exc})"
        ) from exc


def _parse_date(path: Path, row_number: int, field: str, raw: str) -> date:
    cleaned = raw.strip()
    if not cleaned:
        raise CSVLoadError(path, row_number, field, raw, "date field is blank")
    try:
        return date.fromisoformat(cleaned)
    except ValueError as exc:
        raise CSVLoadError(
            path,
            row_number,
            field,
            raw,
            "not a valid ISO date (expected YYYY-MM-DD)",
        ) from exc


def _parse_optional_date(path: Path, row_number: int, field: str, raw: str) -> date | None:
    if not raw.strip():
        return None
    return _parse_date(path, row_number, field, raw)


def _required(path: Path, row_number: int, field: str, raw: str | None) -> str:
    if raw is None or not raw.strip():
        raise CSVLoadError(path, row_number, field, raw or "", "required field is missing or blank")
    return raw.strip()


def load_invoices(path: Path) -> list[Invoice]:
    invoices: list[Invoice] = []
    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        _require_headers(path, reader.fieldnames, INVOICE_FIELDS)
        for row_number, row in enumerate(reader, start=2):
            invoices.append(
                Invoice(
                    invoice_id=_required(path, row_number, "invoice_id", row.get("invoice_id")),
                    order_reference=_required(
                        path, row_number, "order_reference", row.get("order_reference")
                    ),
                    customer_name=_required(
                        path, row_number, "customer_name", row.get("customer_name")
                    ),
                    amount=_parse_decimal(path, row_number, "amount", row.get("amount", "")),
                    invoice_date=_parse_date(
                        path, row_number, "invoice_date", row.get("invoice_date", "")
                    ),
                    description=(row.get("description") or "").strip() or None,
                )
            )
    return invoices


def load_settlement_items(path: Path) -> list[SettlementItem]:
    items: list[SettlementItem] = []
    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        _require_headers(path, reader.fieldnames, SETTLEMENT_ITEM_FIELDS)
        for row_number, row in enumerate(reader, start=2):
            entity_type_raw = _required(path, row_number, "entity_type", row.get("entity_type"))
            try:
                entity_type = EntityType(entity_type_raw)
            except ValueError as exc:
                valid = ", ".join(e.value for e in EntityType)
                raise CSVLoadError(
                    path,
                    row_number,
                    "entity_type",
                    entity_type_raw,
                    f"not one of: {valid}",
                ) from exc

            order_reference = (row.get("order_reference") or "").strip() or None
            settled_at = _parse_optional_date(
                path, row_number, "settled_at", row.get("settled_at", "")
            )

            items.append(
                SettlementItem(
                    entity_id=_required(path, row_number, "entity_id", row.get("entity_id")),
                    entity_type=entity_type,
                    order_reference=order_reference,
                    payment_id=_required(path, row_number, "payment_id", row.get("payment_id")),
                    customer_name=_required(
                        path, row_number, "customer_name", row.get("customer_name")
                    ),
                    amount=_parse_decimal(path, row_number, "amount", row.get("amount", "")),
                    fee=_parse_decimal(path, row_number, "fee", row.get("fee", "")),
                    tax=_parse_decimal(path, row_number, "tax", row.get("tax", "")),
                    net_amount=_parse_decimal(
                        path, row_number, "net_amount", row.get("net_amount", "")
                    ),
                    settlement_id=_required(
                        path, row_number, "settlement_id", row.get("settlement_id")
                    ),
                    settlement_utr=_required(
                        path, row_number, "settlement_utr", row.get("settlement_utr")
                    ),
                    created_at=_parse_date(
                        path, row_number, "created_at", row.get("created_at", "")
                    ),
                    settled_at=settled_at,
                )
            )
    return items


def load_bank_credits(path: Path) -> list[BankCredit]:
    credits: list[BankCredit] = []
    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        _require_headers(path, reader.fieldnames, BANK_CREDIT_FIELDS)
        for row_number, row in enumerate(reader, start=2):
            credits.append(
                BankCredit(
                    credit_id=_required(path, row_number, "credit_id", row.get("credit_id")),
                    utr=_required(path, row_number, "utr", row.get("utr")),
                    amount=_parse_decimal(path, row_number, "amount", row.get("amount", "")),
                    value_date=_parse_date(
                        path, row_number, "value_date", row.get("value_date", "")
                    ),
                    narration=(row.get("narration") or "").strip() or None,
                )
            )
    return credits


def load_match_labels(path: Path) -> list[MatchLabel]:
    """Only used by `calibrate` — never by `reconcile`, which must stay
    blind to ground truth on real, unlabeled data."""
    labels: list[MatchLabel] = []
    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        _require_headers(path, reader.fieldnames, MATCH_LABEL_FIELDS)
        for row_number, row in enumerate(reader, start=2):
            is_match_raw = _required(path, row_number, "is_match", row.get("is_match"))
            if is_match_raw.strip().lower() not in ("true", "false"):
                raise CSVLoadError(
                    path,
                    row_number,
                    "is_match",
                    is_match_raw,
                    "must be 'True' or 'False'",
                )
            labels.append(
                MatchLabel(
                    left_id=_required(path, row_number, "left_id", row.get("left_id")),
                    left_source=Source(
                        _required(path, row_number, "left_source", row.get("left_source"))
                    ),
                    right_id=_required(path, row_number, "right_id", row.get("right_id")),
                    right_source=Source(
                        _required(path, row_number, "right_source", row.get("right_source"))
                    ),
                    is_match=is_match_raw.strip().lower() == "true",
                )
            )
    return labels
