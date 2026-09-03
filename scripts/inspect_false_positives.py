"""Inspect false positive predictions on held-out dataset.
Run:
    PYTHONPATH=src .venv/bin/python scripts/inspect_false_positives.py
"""

from finance_controller.data.generator import generate
from finance_controller.domain.enums import MatchStatus
from finance_controller.matching.calibration import calibrate
from finance_controller.matching.engine import reconcile_all

dev, held = generate()
thresholds = calibrate(dev.invoices, dev.settlement_items, dev.match_labels)
predictions = reconcile_all(held.invoices, held.settlement_items, thresholds)

label_by_pair = {(lbl.left_id, lbl.right_id): lbl.is_match for lbl in held.match_labels}
true_targets = {lbl.left_id: lbl.right_id for lbl in held.match_labels if lbl.is_match}

for p in predictions:
    if p.status == MatchStatus.AUTO_MATCHED:
        actually_true = label_by_pair.get((p.left_id, p.right_id), None)
        if actually_true is not True:
            correct_target = true_targets.get(p.left_id, "NONE — orphan invoice")
            print(f"FALSE POSITIVE: {p.left_id} -> matched {p.right_id}, "
                  f"should have matched {correct_target}")
            print(f"  score={p.confidence:.3f}, reason={p.reason}")
