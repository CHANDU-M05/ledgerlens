"""Domain data model for LedgerLens.

Deliberately plain dataclasses — no ORM, no database. At 150-200 records
this is a Pandas-in-memory problem, and adding SQLAlchemy/Postgres here
would be complexity the project doesn't need to carry.

The three record types below mirror the real two-stage Razorpay
reconciliation loop:

    Invoice <--1:1 fuzzy match--> SettlementItem <--groupby+sum--> BankCredit

Stage 1 (Invoice <-> SettlementItem) is a genuine matching problem:
names drift, references get reformatted, dates shift by a day or two.
Stage 2 (SettlementItem group <-> BankCredit) is arithmetic, not search:
Razorpay already tags every item with the settlement_id it belongs to,
so reconciling the batch is "sum the group, compare to the bank credit,"
not a subset-sum hunt.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from typing import Literal

from finance_controller.domain.enums import (
    EntityType,
    ExceptionReason,
    MatchStatus,
    Source,
)


@dataclass(frozen=True, slots=True)
class Invoice:
    """The merchant's own internal order/invoice record.

    This is the "our books" side of reconciliation — the record a
    merchant's accounting system generated when a sale happened,
    independent of anything Razorpay reports back.
    """

    invoice_id: str
    order_reference: str
    customer_name: str
    amount: Decimal
    invoice_date: date
    description: str | None = None


@dataclass(frozen=True, slots=True)
class SettlementItem:
    """One per-transaction line from a Razorpay settlement report.

    Field names deliberately mirror Razorpay's real settlement report
    vocabulary (entity_id, settlement_id, settlement_utr, fee, tax)
    rather than a generic "bank record" shape, since that vocabulary
    is what a reviewer familiar with the product will recognize.
    """

    entity_id: str
    entity_type: EntityType
    order_reference: str | None
    payment_id: str
    customer_name: str
    amount: Decimal
    fee: Decimal
    tax: Decimal
    net_amount: Decimal
    settlement_id: str
    settlement_utr: str
    created_at: date
    settled_at: date | None = None


@dataclass(frozen=True, slots=True)
class BankCredit:
    """A single NEFT/RTGS credit entry from the bank statement.

    Identified by UTR, which is the join key back to the group of
    SettlementItems that make up this payout.
    """

    credit_id: str
    utr: str
    amount: Decimal
    value_date: date
    narration: str | None = None


@dataclass(frozen=True, slots=True)
class MatchLabel:
    """Ground-truth relationship between two records, used only in the
    synthetic dataset to know what the "correct" answer is.

    Never available to the matching engine at runtime — this exists
    purely so the evaluation script can score predictions against it.
    """

    left_id: str
    left_source: Source
    right_id: str
    right_source: Source
    is_match: bool


ReferenceStatus = Literal["match", "mismatch", "absent"]


@dataclass(frozen=True, slots=True)
class MatchEvidence:
    """The raw, numeric signals behind a match decision.

    This is the object the AI reviewer is given and must not
    contradict — its "reason" text is generated from these numbers,
    not invented. Keeping evidence as a separate typed object (rather
    than a free-text string) is what makes the "AI never invents the
    numeric evidence" guardrail enforceable in code, not just in a
    prompt instruction.
    """

    vendor_similarity: float
    amount_difference: Decimal
    date_difference_days: int
    reference_status: ReferenceStatus


@dataclass(frozen=True, slots=True)
class MatchPrediction:
    """The system's decision for one candidate pair or settlement group."""

    left_id: str
    right_id: str
    confidence: float
    status: MatchStatus
    evidence: MatchEvidence
    reason: str
    exception_reason: ExceptionReason | None = None
