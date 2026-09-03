"""Shared constants for synthetic data generation.

Vendor names are deliberately reused across transactions (a merchant
sees repeat customers/vendors) rather than generated uniquely per
record — that's the realistic case and it's also what makes vendor
fuzzy-matching meaningful (the engine must distinguish "same vendor,
different transaction" from "same transaction, noisy vendor field").
"""

from __future__ import annotations

VENDOR_BASE_NAMES: list[str] = [
    "ABC Technologies",
    "Bluepeak Traders",
    "Crestline Logistics",
    "Devsons Enterprises",
    "Everstone Retail",
    "Faircraft Industries",
    "Greenfield Agro",
    "Harborline Exports",
    "Indigo Fabrics",
    "Jaya Textiles",
    "Kestrel Media",
    "Lumen Consulting",
    "Meridian Foods",
    "Novasoft Systems",
    "Orbit Packaging",
    "Pranav Hardware",
    "Quantalytics",
    "Ridgeline Motors",
    "Sundar Spices",
    "Tricol Chemicals",
    "Umbra Studios",
    "Vertex Engineering",
    "Wavecrest Apparel",
    "Xenon Electricals",
    "Yashoda Pharma",
    "Zenith Furnishings",
]

LEGAL_SUFFIXES: list[str] = [
    "Private Limited",
    "Pvt Ltd",
    "Pvt. Ltd.",
    "LLP",
    "Enterprises",
    "",  # some vendors carry no suffix at all
]

# Razorpay's standard payment gateway fee is commonly modelled as ~2% of
# the transaction amount, with 18% GST charged on top of that fee — this
# mirrors the real fee/tax split that shows up in settlement reports.
GATEWAY_FEE_RATE = "0.02"
GST_ON_FEE_RATE = "0.18"

# Typical settlement cycle: funds move from "created" (payment captured)
# to "settled" (batched into a payout) a few days later, and the bank
# credit itself lands a day or so after that.
SETTLEMENT_DELAY_DAYS = (1, 4)
BANK_CREDIT_DELAY_DAYS = (0, 1)

RANDOM_SEED = 20260826
