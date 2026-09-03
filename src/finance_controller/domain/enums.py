"""Enumerations shared across the finance controller domain model."""

from __future__ import annotations

from enum import StrEnum


class Source(StrEnum):
    """Which of the three record sources a financial record came from.

    INVOICE:         The merchant's own internal order/invoice record.
    SETTLEMENT_ITEM:  A single per-transaction line item from Razorpay's
                      settlement report (carries a settlement_id linking
                      it to a batched bank payout).
    BANK_CREDIT:      A single NEFT/RTGS credit entry from the bank
                      statement, identified by a UTR.
    """

    INVOICE = "invoice"
    SETTLEMENT_ITEM = "settlement_item"
    BANK_CREDIT = "bank_credit"


class EntityType(StrEnum):
    """The kind of financial event a settlement item represents.

    Mirrors Razorpay's own settlement report vocabulary so the demo
    data stays honest to the real product surface.
    """

    PAYMENT = "payment"
    REFUND = "refund"
    ADJUSTMENT = "adjustment"


class MatchStatus(StrEnum):
    """Outcome of a reconciliation decision for a candidate pair or group."""

    AUTO_MATCHED = "auto_matched"
    AI_APPROVED = "ai_approved"
    REVIEW = "review"
    EXCEPTION = "exception"


class ExceptionReason(StrEnum):
    """Why a record or group could not be auto-resolved.

    Kept as a closed enum (rather than free text) so the exception
    report stays queryable and every reason is one we actually
    designed for, not an ad hoc string invented at runtime.
    """

    NO_CANDIDATE_FOUND = "no_candidate_found"
    AMBIGUOUS_CANDIDATES = "ambiguous_candidates"
    AMOUNT_MISMATCH = "amount_mismatch"
    SETTLEMENT_SUM_MISMATCH = "settlement_sum_mismatch"
    AI_REVIEW_UNAVAILABLE = "ai_review_unavailable"
    LOW_CONFIDENCE = "low_confidence"
