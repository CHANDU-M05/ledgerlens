"""Persist calibrated `Thresholds` to disk so calibration (which needs
labeled data) and reconciliation (which runs against real, unlabeled
data) can be separate CLI invocations, possibly on different machines
or at different times. Calibrate once against a trusted labeled
dataset; reuse the resulting thresholds.json against production data
indefinitely, until the generator/data distribution changes enough to
warrant recalibration.
"""

from __future__ import annotations

import json
from pathlib import Path

from finance_controller.matching.calibration import Thresholds


def save_thresholds(thresholds: Thresholds, path: Path) -> None:
    payload = {
        "auto_match": thresholds.auto_match,
        "review_floor": thresholds.review_floor,
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")


def load_thresholds(path: Path) -> Thresholds:
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path}: not valid JSON ({exc})") from exc

    for key in ("auto_match", "review_floor"):
        if key not in payload:
            raise ValueError(f"{path}: missing required key {key!r}")

    return Thresholds(
        auto_match=float(payload["auto_match"]),
        review_floor=float(payload["review_floor"]),
    )
