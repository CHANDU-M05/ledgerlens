# LedgerLens: Finance Controller & Reconciliation System
## Complete Project Context & Architectural Documentation

This document provides a comprehensive technical overview, design rationale, mathematical proofs, domain specifications, and full code walkthrough for the **LedgerLens Finance Controller** project.

---

## 1. Project Overview & Business Logic

LedgerLens implements a two-stage financial reconciliation loop modeled after Razorpay's real-world payment gateway settlement architecture:

```
Stage 1: Merchant Invoices  <--- 1:1 Fuzzy Match --->  Settlement Line Items
Stage 2: Settlement Group   <--- Groupby + Sum   --->  Bank Statement Payout Credit
```

### Key Technical Principles:
1. **No ORM / No DB Over-engineering**: In-memory Python dataclasses and pure functions. At 150–500 transactions per batch, Pandas/in-memory dataclasses solve the problem with zero database overhead.
2. **Strict Data Leakage Controls**: Ground-truth labels (`MatchLabel`) exist only in synthetic datasets and evaluation scripts. Matching algorithms are strictly blind to `MatchLabel`.
3. **Decoupled 2-Stage Pipeline**:
   - **Stage 1 (Fuzzy Match)**: Handles name drift, reference formatting changes (e.g. `ORD-` vs `PO-`), dropped references, partial payment perturbations, and date shifts.
   - **Stage 2 (Arithmetic Reconciliation)**: Groups settlement items by `settlement_utr` and verifies that `sum(net_amount) == BankCredit.amount`.

---

## 2. Directory Structure

```
rpfinance/
├── .gitignore
├── pyproject.toml
├── requirements.txt
├── PROJECT_CONTEXT.md
├── scripts/
│   ├── run_reconciliation_check.py
│   ├── inspect_false_positives.py
│   ├── inspect_one_case.py
│   └── breakdown_unresolved.py
├── src/
│   └── finance_controller/
│       ├── data/
│       │   ├── constants.py
│       │   ├── noise.py
│       │   └── generator.py
│       ├── domain/
│       │   ├── enums.py
│       │   └── models.py
│       ├── matching/
│       │   ├── candidates.py
│       │   ├── scorer.py
│       │   ├── calibration.py
│       │   ├── engine.py
│       │   └── stage2.py
│       ├── ai/
│       │   ├── schemas.py
│       │   ├── reviewer.py
│       │   └── pipeline.py
│       └── evaluation/
│           └── metrics.py
└── tests/
    ├── test_domain_models.py
    ├── test_generator.py
    ├── test_matching.py
    └── test_ai_reviewer.py
```

---

## 3. Domain Model & Enumerations

### Enumerations (`src/finance_controller/domain/enums.py`)
- **`Source`**: `INVOICE`, `SETTLEMENT_ITEM`, `BANK_CREDIT`
- **`EntityType`**: `PAYMENT`, `REFUND`, `ADJUSTMENT`
- **`MatchStatus`**:
  - `AUTO_MATCHED`: High-confidence deterministic resolution.
  - `AI_APPROVED`: Resolved via LLM review with zero-digit qualitative rationale.
  - `REVIEW`: Escalated for human or AI review.
  - `EXCEPTION`: Unresolvable or invalid record.
- **`ExceptionReason`**: `NO_CANDIDATE_FOUND`, `AMBIGUOUS_CANDIDATES`, `AMOUNT_MISMATCH`, `SETTLEMENT_SUM_MISMATCH`, `AI_REVIEW_UNAVAILABLE`, `LOW_CONFIDENCE`.

### Data Models (`src/finance_controller/domain/models.py`)
- **`Invoice`**: `invoice_id`, `order_reference`, `customer_name`, `amount`, `invoice_date`, `description`
- **`SettlementItem`**: `entity_id`, `entity_type`, `order_reference`, `payment_id`, `customer_name`, `amount`, `fee`, `tax`, `net_amount`, `settlement_id`, `settlement_utr`, `created_at`, `settled_at`
- **`BankCredit`**: `credit_id`, `utr`, `amount`, `value_date`, `narration`
- **`MatchLabel`**: Ground-truth label (`left_id`, `left_source`, `right_id`, `right_source`, `is_match`)
- **`MatchEvidence`**: Raw evidence signals (`vendor_similarity: float`, `amount_difference: Decimal`, `date_difference_days: int`, `reference_status: ReferenceStatus`)
  - **`ReferenceStatus`**: `Literal["match", "mismatch", "absent"]`
- **`MatchPrediction`**: Matching engine decision output carrying status, confidence, evidence, reason text, and optional exception reason.

---

## 4. Synthetic Data Generation Architecture

The generator (`src/finance_controller/data/generator.py`) enforces 3 core structural guarantees:

1. **Split Before Observation**: `TrueTransaction` objects are constructed and partitioned into `dev` (80%) and `held_out` (20%) *before* deriving any noisy invoice, settlement item, or bank credit.
   - Machine-checked by `test_no_transaction_leakage_between_splits`.
2. **Explicit Ground Truth for Decoys**: Decoys (`pay_DECOY...`) share vendor and amount with a target invoice but carry an explicit `DECOY-XXXX` reference. They are labeled `is_match=False` at creation.
3. **Amount Mismatch != Non-Match**: True pairs with perturbed amounts (partial payments, fee rounding differences) remain labeled `is_match=True`.
4. **Independent Noise Models**:
   - Invoice noise (`invoice_vendor_name`): Title case, accidental double spaces, trailing periods.
   - Settlement noise (`settlement_vendor_name`): UPPERCASE, abbreviated legal suffixes (`PVT LTD`, `ENTP`), string truncation (15% rate).
   - Date drift (`drift_date`): 1–4 days settlement delay.
   - Reference noise (`noisy_reference`): 20% dropped reference rate (`None`), 10% prefix swap (`PO-` vs `ORD-`).

---

## 5. Matching Engine & Mathematical Proofs

### A. Candidate Generation & Blocking (`matching/candidates.py`)
- Blocking does **NOT** rely on `order_reference` (to handle the 20% dropped reference rate).
- Candidates are blocked on date proximity (`abs(days) <= 10`) and amount tolerance (`max(2500, 5% of invoice amount)`).

### B. Signal Weights & Scoring (`matching/scorer.py`)
```python
W_AMOUNT    = 0.40  # Amount closeness
W_VENDOR    = 0.35  # Token sort vendor similarity (RapidFuzz)
W_REFERENCE = 0.15  # Reference status contribution
W_DATE      = 0.10  # Exponential decay date score: 0.5 ** (days / 5)
```

### C. Active Mismatch Penalty (`ReferenceStatus`)
- `"match"`: `r_score = +1.0` (Full reference credit)
- `"absent"`: `r_score = 0.0` (Neutral — no penalty for missing reference)
- `"mismatch"`: `r_score = -1.0` (**Active penalty** for present but wrong reference)

### D. Mathematical Proof: Decoy Score Ceiling
A decoy settlement item has `reference_status = "mismatch"` (`r_score = -1.0`). Its date drift is at least 1 day (`SETTLEMENT_DELAY_DAYS = (1, 4)`), capping its date score at $0.5^{1/5} \approx 0.871$.

Theoretical maximum score for any decoy:
$$\text{Max Decoy Score} = 0.40(1.0) + 0.35(1.0) + 0.15(-1.0) + 0.10(0.871) = 0.6871$$

With quantile calibration on DEV:
- `auto_match` threshold $\approx 0.788$
- `review_floor` threshold $\approx 0.687$

**Mathematical Invariant**: Because $0.6871 < 0.788$, **no decoy can ever clear `auto_match`**, guaranteeing 100% precision regardless of random seed.
Machine-checked by `test_decoy_score_ceiling_is_below_any_plausible_auto_match_threshold`.

### E. Quantile Calibration (`matching/calibration.py`)
- `auto_match`: 5th percentile of true match scores on DEV split (~0.788).
- `review_floor`: 95th percentile of decoy scores on DEV split (~0.687).

### F. Ambiguity Check (`matching/engine.py`)
If `(top_score - second_best_score) < AMBIGUITY_MARGIN` (0.05), the decision is assigned `MatchStatus.REVIEW` with `ExceptionReason.AMBIGUOUS_CANDIDATES`.

---

## 6. AI Review Layer & Guardrails

The AI review layer (`src/finance_controller/ai/`) processes only predictions with `MatchStatus.REVIEW`.

### Architectural Guardrails:
1. **Zero-Digit Rationale Rule**: `_rationale_has_any_digit(rationale)` enforces that LLM rationales contain no numbers or digits. The LLM supplies qualitative reasoning only; numeric evidence is strictly handled deterministically.
2. **Structured Output**: Validated via Pydantic (`AIReviewOutcome(decision: Literal["approve", "escalate"], rationale: str)`).
3. **Fail-Safe Degradation**: Any exception (API failure, missing key, rate limit, malformed JSON, digit violation) gracefully falls back to `MatchStatus.REVIEW` with `ExceptionReason.AI_REVIEW_UNAVAILABLE`.
4. **Bypass for Resolved Items**: `AUTO_MATCHED` and `EXCEPTION` predictions bypass LLM calls entirely.

---

## 7. Performance & Verification Benchmarks

Command to run full test suite & end-to-end evaluation check:
```bash
PYTHONPATH=src .venv/bin/pytest tests/ -v
PYTHONPATH=src .venv/bin/python scripts/run_reconciliation_check.py
```

### Benchmark Results (Held-Out Split — 45 Invoices):
- **Deterministic Auto-Matched**: 34 invoices
- **AI Approved**: 0 (in fallback mode without API key)
- **Still REVIEW**: 5 invoices (true matches with moderate noise)
- **EXCEPTION**: 6 invoices (5 orphans, 1 heavily perturbed small invoice)
- **Auto-Match Precision**: **100.0%** (0 false positives)
- **Recall**: **85.0%** (34/40 true matches resolved)
- **Auto-Match Resolution Rate**: **75.6%**
- **Stage 2 Batches**: 13/15 (86.7%) sum exactly to BankCredit

---

## 8. Complete Source Code Listings

Below is the complete, working code for key system modules.

### `src/finance_controller/domain/models.py`
```python
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

ReferenceStatus = Literal["match", "mismatch", "absent"]


@dataclass(frozen=True, slots=True)
class Invoice:
    invoice_id: str
    order_reference: str
    customer_name: str
    amount: Decimal
    invoice_date: date
    description: str | None = None


@dataclass(frozen=True, slots=True)
class SettlementItem:
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
    credit_id: str
    utr: str
    amount: Decimal
    value_date: date
    narration: str | None = None


@dataclass(frozen=True, slots=True)
class MatchLabel:
    left_id: str
    left_source: Source
    right_id: str
    right_source: Source
    is_match: bool


@dataclass(frozen=True, slots=True)
class MatchEvidence:
    vendor_similarity: float
    amount_difference: Decimal
    date_difference_days: int
    reference_status: ReferenceStatus


@dataclass(frozen=True, slots=True)
class MatchPrediction:
    left_id: str
    right_id: str
    confidence: float
    status: MatchStatus
    evidence: MatchEvidence
    reason: str
    exception_reason: ExceptionReason | None = None
```

### `src/finance_controller/matching/scorer.py`
```python
from __future__ import annotations

from decimal import Decimal
from rapidfuzz import fuzz

from finance_controller.domain.models import (
    Invoice,
    MatchEvidence,
    ReferenceStatus,
    SettlementItem,
)

_LEGAL_SUFFIX_TOKENS = ("PRIVATE", "LIMITED", "PVT", "LTD", "LLP", "ENTERPRISES", "ENTP")

W_AMOUNT = 0.40
W_VENDOR = 0.35
W_REFERENCE = 0.15
W_DATE = 0.10

DATE_SCORE_HALF_LIFE_DAYS = 5


def _normalize_vendor(name: str) -> str:
    tokens = name.upper().replace(".", "").split()
    return " ".join(t for t in tokens if t not in _LEGAL_SUFFIX_TOKENS)


def _normalize_reference(ref: str | None) -> str | None:
    if ref is None:
        return None
    digits = "".join(c for c in ref if c.isdigit())
    return digits or None


def _reference_status(inv_ref: str | None, item_ref: str | None) -> ReferenceStatus:
    if item_ref is None:
        return "absent"
    if inv_ref is not None and inv_ref == item_ref:
        return "match"
    return "mismatch"


def vendor_similarity(invoice_name: str, item_name: str) -> float:
    a, b = _normalize_vendor(invoice_name), _normalize_vendor(item_name)
    return float(fuzz.token_sort_ratio(a, b) / 100.0)


def amount_score(diff: Decimal, invoice_amount: Decimal) -> float:
    if invoice_amount == 0:
        return 0.0
    relative = abs(diff) / invoice_amount
    return max(0.0, 1.0 - float(relative) / 0.10)


def date_score(days_apart: int) -> float:
    return float(0.5 ** (days_apart / DATE_SCORE_HALF_LIFE_DAYS))


def compute_evidence(invoice: Invoice, item: SettlementItem) -> MatchEvidence:
    vendor_sim = vendor_similarity(invoice.customer_name, item.customer_name)
    amount_diff = item.amount - invoice.amount
    date_diff = abs((item.created_at - invoice.invoice_date).days)
    inv_ref = _normalize_reference(invoice.order_reference)
    item_ref = _normalize_reference(item.order_reference)
    return MatchEvidence(
        vendor_similarity=vendor_sim,
        amount_difference=amount_diff,
        date_difference_days=date_diff,
        reference_status=_reference_status(inv_ref, item_ref),
    )


def confidence(evidence: MatchEvidence, invoice_amount: Decimal) -> float:
    a_score = amount_score(evidence.amount_difference, invoice_amount)
    d_score = date_score(evidence.date_difference_days)
    r_score = {"match": 1.0, "absent": 0.0, "mismatch": -1.0}[evidence.reference_status]
    return (
        W_AMOUNT * a_score
        + W_VENDOR * evidence.vendor_similarity
        + W_REFERENCE * r_score
        + W_DATE * d_score
    )
```

### `src/finance_controller/ai/reviewer.py`
```python
from __future__ import annotations

import json
from dataclasses import replace
from typing import Protocol

from pydantic import ValidationError

from finance_controller.ai.schemas import AIReviewOutcome
from finance_controller.domain.enums import ExceptionReason, MatchStatus
from finance_controller.domain.models import Invoice, MatchPrediction, SettlementItem


class LLMClient(Protocol):
    def complete(self, prompt: str) -> str: ...


def _build_prompt(prediction: MatchPrediction, invoice: Invoice, item: SettlementItem) -> str:
    ev = prediction.evidence
    return f"""You are reviewing one candidate match in a financial reconciliation system.

Invoice: vendor="{invoice.customer_name}", amount={invoice.amount}, date={invoice.invoice_date}
Settlement item: vendor="{item.customer_name}", amount={item.amount}, date={item.created_at}

Evidence (already computed by deterministic code — do not recompute or contradict it):
- vendor_similarity: {ev.vendor_similarity:.2f}
- amount_difference: {ev.amount_difference}
- date_difference_days: {ev.date_difference_days}
- reference_status: {ev.reference_status}
- current_score: {prediction.confidence:.3f}

Decide whether this pair should be "approve"d as a genuine match resolved
without human review, or "escalate"d to a human.

IMPORTANT: Your rationale must contain NO digits, percentages, or amounts.
Describe your reasoning qualitatively only (e.g. "vendor names are nearly
identical and the amounts are close" rather than restating any number).
The numeric evidence is already recorded separately and must not be
repeated by you.

Respond with ONLY a JSON object, no other text, in this exact shape:
{{"decision": "approve" or "escalate", "rationale": "<qualitative reasoning, no digits>"}}
"""


def _parse_response(raw: str) -> AIReviewOutcome:
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text.removeprefix("json").strip()
    data = json.loads(text)
    return AIReviewOutcome.model_validate(data)


def _rationale_has_any_digit(rationale: str) -> bool:
    return any(ch.isdigit() for ch in rationale)


def _fallback(prediction: MatchPrediction, detail: str) -> MatchPrediction:
    return replace(
        prediction,
        status=MatchStatus.REVIEW,
        exception_reason=ExceptionReason.AI_REVIEW_UNAVAILABLE,
        reason=prediction.reason
        + f" | AI review unavailable ({detail}); human verification required.",
    )


def review_prediction(
    prediction: MatchPrediction,
    invoice: Invoice,
    item: SettlementItem,
    client: LLMClient,
) -> MatchPrediction:
    if prediction.status != MatchStatus.REVIEW:
        return prediction

    try:
        raw = client.complete(_build_prompt(prediction, invoice, item))
    except Exception as exc:
        return _fallback(prediction, f"request failed: {exc.__class__.__name__}")

    try:
        outcome = _parse_response(raw)
    except (json.JSONDecodeError, ValidationError) as exc:
        return _fallback(prediction, f"malformed response: {exc.__class__.__name__}")

    if _rationale_has_any_digit(outcome.rationale):
        return _fallback(prediction, "response contained a disallowed number")

    if outcome.decision == "approve":
        return replace(
            prediction,
            status=MatchStatus.AI_APPROVED,
            exception_reason=None,
            reason=prediction.reason + f" | AI review: approved — {outcome.rationale}",
        )

    return replace(
        prediction,
        status=MatchStatus.REVIEW,
        exception_reason=ExceptionReason.AMBIGUOUS_CANDIDATES,
        reason=prediction.reason + f" | AI review: escalated — {outcome.rationale}",
    )
```
