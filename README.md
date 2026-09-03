# LedgerLens

**A two-stage financial reconciliation engine** — invoice ↔ settlement fuzzy matching, and settlement ↔ bank credit arithmetic verification. Built for Razorpay AI Buildathon, Track 04 — AI Finance Controller.

Reconciliation is still done by hand at most companies. LedgerLens closes one finance-ops loop end to end: it takes a batch of merchant invoices, payment-gateway settlement line items, and bank credits, and tells you — with measured accuracy, not a demo cherry-pick — which invoices got paid, which need a human, and which settlement batches don't add up.

```bash
ledgerlens generate  --out-dir data/
ledgerlens calibrate --invoices data/dev/invoices.csv --settlement-items data/dev/settlement_items.csv --labels data/dev/match_labels.csv --out thresholds.json
ledgerlens reconcile --invoices data/held_out/invoices.csv --settlement-items data/held_out/settlement_items.csv --bank-credits data/held_out/bank_credits.csv --thresholds thresholds.json --out-dir results/
```

---

## Results (held-out, 45 invoices, never seen by the matcher)

| Metric | Value |
|---|---|
| Auto-match precision | **100.0%** (0 false positives) |
| Recall | **85.0%** (34/40 true matches resolved) |
| Resolved without a human | **75.6%** |
| Stage 2 batches (settlement sum = bank credit) | **13/15 (86.7%)** |
| Unresolved (honest exception list) | 6 orphans/heavily-perturbed, 5 sent to review |

These numbers come from a fixed-seed synthetic dataset, split 80/20 into `dev`/`held_out` **before** any noise is added — the matching engine never sees `held_out` during calibration, and the AI review layer never sees ground-truth labels at all. Full breakdown: `results/exceptions.csv` after running `reconcile`.

### Why the 100% precision isn't luck

Every decoy settlement item (same vendor, same amount, wrong reference — the hardest case to reject) has a mathematically bounded maximum score:

```
Max Decoy Score = 0.40(amount) + 0.35(vendor) + 0.15(-1.0 reference penalty) + 0.10(date) ≈ 0.687
```

The `auto_match` threshold, calibrated as the 5th percentile of true-match scores on `dev`, comes out to ≈0.788. Since 0.687 < 0.788 **by construction**, no decoy can ever clear auto-match, regardless of random seed. This is checked by `test_decoy_score_ceiling_is_below_any_plausible_auto_match_threshold`, not just asserted in a doc.

---

## Architecture

```
                    STAGE 1: Fuzzy Match (per invoice)
┌─────────────┐    blocking     ┌──────────────────┐   scoring    ┌────────────┐
│  Invoices   │ ───(amount ±5%,─▶│ Candidate        │──(vendor sim,─▶│ Confidence │
│  (merchant) │    date ±10d)   │ Settlement Items  │  amount, date, │  Score     │
└─────────────┘                 └──────────────────┘  reference)   └─────┬──────┘
                                                                          │
                          ┌───────────────────────────────────────────────┘
                          ▼
              score ≥ auto_match, unambiguous?  ──yes──▶  AUTO_MATCHED
                          │no
                          ▼
              score ≥ review_floor?  ──yes──▶  AI Reviewer (qualitative only,
                          │no                    zero-digit rationale rule,
                          ▼                       fails safe to REVIEW)
                      EXCEPTION                        │
                                              approve ──┴── escalate
                                                │              │
                                          AI_APPROVED      REVIEW


                    STAGE 2: Arithmetic Reconciliation (per settlement batch)
┌──────────────────┐   group by settlement_utr,   ┌──────────────┐
│ Settlement Items  │──sum(net_amount)────────────▶│ Bank Credit  │
└──────────────────┘   compare exactly              │ (same UTR)   │
                                                     └──────────────┘
                        match → AUTO_MATCHED   mismatch → EXCEPTION
```

**Signal weights** (Stage 1 scoring): amount 0.40, vendor similarity (RapidFuzz token-sort) 0.35, reference status 0.15, date decay 0.10. Reference status carries an **active mismatch penalty** (−1.0), not just neutral absence — a present-but-wrong reference is stronger evidence against a match than no reference at all.

**AI review layer guardrails**: the LLM never sees or returns numeric evidence — its rationale is rejected if it contains a single digit. All numeric decisions stay deterministic; the model supplies qualitative judgment only, on the narrow band of cases the deterministic engine is genuinely unsure about. Any API failure, malformed response, or guardrail violation fails safe to `REVIEW`, never to a silent guess.

---

## Quickstart

```bash
python3.12 -m venv .venv
.venv/bin/pip install -e .
.venv/bin/pip install rapidfuzz pydantic anthropic pytest ruff mypy

# Generate a synthetic labeled dataset (fixed seed — reproducible)
ledgerlens generate --out-dir data/

# Calibrate thresholds from the labeled dev split
ledgerlens calibrate \
  --invoices data/dev/invoices.csv \
  --settlement-items data/dev/settlement_items.csv \
  --labels data/dev/match_labels.csv \
  --out thresholds.json

# Reconcile the held-out split — no ground truth read or required
ledgerlens reconcile \
  --invoices data/held_out/invoices.csv \
  --settlement-items data/held_out/settlement_items.csv \
  --bank-credits data/held_out/bank_credits.csv \
  --thresholds thresholds.json \
  --out-dir results/

# Optional: send REVIEW-status cases to the AI reviewer
export ANTHROPIC_API_KEY=sk-...
ledgerlens reconcile ... --use-ai
```

Outputs land in `results/`: `predictions.csv` (every invoice), `exceptions.csv` (just the REVIEW/EXCEPTION rows a human needs to look at), `stage2.csv`, `summary.json`.

To reconcile **real** data instead of synthetic: point `--invoices`/`--settlement-items`/`--bank-credits` at CSVs in the same shape (see `src/finance_controller/io/csv_loader.py` for the exact column headers). The loader accepts common real-world messiness (commas, ₹/Rs. prefixes in amounts) and raises a pinpointed error (file, row, column) on anything it can't parse.

---

## Project Structure

```
src/finance_controller/
├── domain/        # Invoice, SettlementItem, BankCredit, MatchPrediction — frozen dataclasses
├── data/           # Synthetic dataset generator (noise models, dev/held-out split)
├── matching/       # Stage 1: blocking, scoring, calibration, ambiguity check
├── ai/             # LLM review layer — schemas, prompt, guardrails, fail-safe fallback
├── evaluation/     # Precision/recall/resolution-rate metrics against ground truth
├── io/             # CSV loaders/writers, threshold persistence — the real-data entrypoint
└── cli.py          # `ledgerlens generate|calibrate|reconcile`
tests/               # 50 tests: unit + end-to-end, including the decoy-ceiling proof
```

## Quality Bar

```
pytest:  50/50 passed
ruff:    clean
mypy:    strict, 0 errors across 25 source files
```

No ORM, no database — at 150–500 transactions per batch, in-memory dataclasses and pure functions solve this with zero infrastructure overhead. Ground-truth labels never leak into the matching or AI code paths (`test_no_leakage_in_matching_modules`); the `reconcile` CLI command never reads a labels file at all.

## What's Next

- FastAPI wrapper for HTTP-triggered reconciliation runs
- Streaming/chunked input for batches beyond a few thousand records
- A second held-out evaluation against real settlement export data, not just synthetic
