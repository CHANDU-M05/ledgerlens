"""Breakdown of REVIEW and EXCEPTION predictions in held-out.
Run:
    PYTHONPATH=src .venv/bin/python scripts/breakdown_unresolved.py
"""

from finance_controller.data.generator import generate
from finance_controller.domain.enums import MatchStatus
from finance_controller.matching.calibration import calibrate
from finance_controller.matching.engine import reconcile_all

dev, held = generate()
thresholds = calibrate(dev.invoices, dev.settlement_items, dev.match_labels)
predictions = reconcile_all(held.invoices, held.settlement_items, thresholds)

orphan_ids = {inv.invoice_id for inv in held.invoices if inv.invoice_id.startswith("INV-ORPH")}

for p in predictions:
    if p.status in (MatchStatus.REVIEW, MatchStatus.EXCEPTION):
        kind = "orphan" if p.left_id in orphan_ids else "true-match-unresolved"
        msg = f"{p.status.value:10s} {kind:22s} {p.left_id}  score={p.confidence:.3f}"
        print(f"{msg}  reason={p.exception_reason}")
