"""Independent noise models for each of the three record sources.

The single most important design rule in this generator: **invoice
noise, settlement-item noise, and bank-credit noise must not be the
same transformation.** If all three sources were derived from the
truth by the same string mangling, "do these two strings look alike"
would trivially imply "is this the same transaction," and the fuzzy
matching engine would be solving a fake problem. Real systems corrupt
data differently because they're built by different teams for
different purposes — a merchant's invoicing tool formats names one
way, a payment gateway's settlement export formats them another way.
"""

from __future__ import annotations

import random
from datetime import date, timedelta
from decimal import Decimal

from finance_controller.data.constants import LEGAL_SUFFIXES


def invoice_vendor_name(canonical_name: str, suffix: str, rng: random.Random) -> str:
    """How a merchant's own invoicing system would render a vendor name.

    Tends to be tidy and title-cased, since it was typed by a human
    into an accounting form — occasional double spaces or a trailing
    period are the only noise.
    """
    name = f"{canonical_name} {suffix}".strip()
    if rng.random() < 0.08:
        name = name.replace(" ", "  ", 1)  # accidental double space
    if rng.random() < 0.05:
        name = name + "."
    return name


def settlement_vendor_name(canonical_name: str, suffix: str, rng: random.Random) -> str:
    """How a payment gateway's settlement export tends to render names.

    Payment processors commonly upper-case names, abbreviate legal
    suffixes, and sometimes truncate long fields — different noise
    entirely from the invoice side.
    """
    abbreviations = {
        "Private Limited": "PVT LTD",
        "Pvt Ltd": "PVT LTD",
        "Pvt. Ltd.": "PVT LTD",
        "LLP": "LLP",
        "Enterprises": "ENTP",
        "": "",
    }
    name = f"{canonical_name} {abbreviations.get(suffix, suffix)}".strip().upper()
    if rng.random() < 0.15:
        name = name[:18].rstrip()  # field-length truncation
    return name


def drift_date(base: date, low_days: int, high_days: int, rng: random.Random) -> date:
    """Shift a date forward by a random number of days in [low, high]."""
    return base + timedelta(days=rng.randint(low_days, high_days))


def noisy_reference(canonical_reference: str, rng: random.Random, *, drop: bool) -> str | None:
    """Occasionally the merchant's internal PO/order reference doesn't
    match what was actually passed to the payment gateway — this forces
    the matching engine to fall back to vendor+amount+date evidence
    instead of a trivial exact-reference join."""
    if drop:
        return None
    if rng.random() < 0.10:
        return canonical_reference.replace("ORD-", "PO-")
    return canonical_reference


def pick_legal_suffix(rng: random.Random) -> str:
    return rng.choice(LEGAL_SUFFIXES)


def perturb_amount(amount: Decimal, rng: random.Random) -> Decimal:
    """Small discrepancy between invoice and settlement amount — a
    partial payment, a rounding difference, or a bank fee the merchant
    didn't account for. Represents a TRUE match with a real accounting
    exception, not a non-match."""
    delta = Decimal(rng.choice([500, 1000, 1500, 2000]))
    return amount - delta
