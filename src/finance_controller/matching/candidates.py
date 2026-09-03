"""Stage 1 candidate generation: invoice <-> settlement item.

Blocking must NOT rely on order_reference — ~20% of true settlement items
have no reference by design (DROP_REF_RATE in the generator). A
reference-only block would silently drop those true pairs before scoring
ever runs. We block on amount (generous tolerance) and date proximity
(generous window) instead, since both are present on every record
regardless of noise. Reference stays a strong *scoring* signal — just not
a generation gate.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from decimal import Decimal

from finance_controller.domain.models import Invoice, SettlementItem

# Generous enough to survive perturb_amount (max delta observed: 2000)
# without exploding candidate count at our record volumes.
AMOUNT_TOLERANCE_ABS = Decimal("2500")
AMOUNT_TOLERANCE_PCT = Decimal("0.05")

# Settlement created_at drifts 1-4 days after invoice_date by design
# (SETTLEMENT_DELAY_DAYS); padded on both sides for safety.
DATE_WINDOW_DAYS = 10


def _amount_within_tolerance(invoice_amount: Decimal, item_amount: Decimal) -> bool:
    tolerance = max(AMOUNT_TOLERANCE_ABS, invoice_amount * AMOUNT_TOLERANCE_PCT)
    return abs(invoice_amount - item_amount) <= tolerance


def _date_within_window(invoice_date: date, item_date: date) -> bool:
    return abs((item_date - invoice_date).days) <= DATE_WINDOW_DAYS


@dataclass(frozen=True, slots=True)
class Candidate:
    invoice: Invoice
    settlement_item: SettlementItem


def generate_candidates(
    invoices: list[Invoice],
    settlement_items: list[SettlementItem],
) -> dict[str, list[Candidate]]:
    """Per invoice_id, the settlement items that pass blocking.

    O(n*m) — fine at hundreds of records. If this ever needs to scale to
    tens of thousands, blocking should move to a sorted-amount index;
    not needed here and premature to build now.
    """
    result: dict[str, list[Candidate]] = {inv.invoice_id: [] for inv in invoices}
    for inv in invoices:
        for item in settlement_items:
            if _amount_within_tolerance(
                inv.amount, item.amount
            ) and _date_within_window(inv.invoice_date, item.created_at):
                result[inv.invoice_id].append(Candidate(invoice=inv, settlement_item=item))
    return result
