"""Stage 2: settlement batch <-> bank credit. Every SettlementItem already
carries the settlement_id/UTR of its batch (mirroring Razorpay's real
settlement report linkage), so this is groupby-sum-compare — no fuzzy
matching, no candidate generation.
"""

from __future__ import annotations

from decimal import Decimal
from typing import TypedDict

from finance_controller.domain.enums import ExceptionReason, MatchStatus
from finance_controller.domain.models import BankCredit, SettlementItem

# Fee/tax are rounded to 2dp at generation time, so Decimal sums are
# exact — any nonzero difference is a genuine discrepancy, not float noise.
SUM_TOLERANCE = Decimal("0.00")


class SettlementResult(TypedDict):
    settlement_utr: str
    expected_amount: Decimal
    actual_amount: Decimal | None
    status: MatchStatus
    exception_reason: ExceptionReason | None
    reason: str


def reconcile_settlements(
    settlement_items: list[SettlementItem],
    bank_credits: list[BankCredit],
) -> list[SettlementResult]:
    by_utr: dict[str, list[SettlementItem]] = {}
    for item in settlement_items:
        by_utr.setdefault(item.settlement_utr, []).append(item)
    credit_by_utr = {bc.utr: bc for bc in bank_credits}

    results: list[SettlementResult] = []
    for utr, items in by_utr.items():
        expected = sum((it.net_amount for it in items), Decimal("0"))
        credit = credit_by_utr.get(utr)
        if credit is None:
            results.append(
                SettlementResult(
                    settlement_utr=utr,
                    expected_amount=expected,
                    actual_amount=None,
                    status=MatchStatus.EXCEPTION,
                    exception_reason=ExceptionReason.NO_CANDIDATE_FOUND,
                    reason=f"No bank credit found for UTR {utr}.",
                )
            )
            continue

        diff = credit.amount - expected
        if abs(diff) <= SUM_TOLERANCE:
            results.append(
                SettlementResult(
                    settlement_utr=utr,
                    expected_amount=expected,
                    actual_amount=credit.amount,
                    status=MatchStatus.AUTO_MATCHED,
                    exception_reason=None,
                    reason=f"{len(items)} settlement item(s) sum exactly to bank credit.",
                )
            )
        else:
            results.append(
                SettlementResult(
                    settlement_utr=utr,
                    expected_amount=expected,
                    actual_amount=credit.amount,
                    status=MatchStatus.EXCEPTION,
                    exception_reason=ExceptionReason.SETTLEMENT_SUM_MISMATCH,
                    reason=(
                        f"Settlement items sum to {expected}, bank credit is "
                        f"{credit.amount} (difference {diff})."
                    ),
                )
            )
    return results
