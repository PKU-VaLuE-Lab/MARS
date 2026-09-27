#!/usr/bin/env python3
"""Summarize ActionAware probe traces without touching evaluation logs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _quantiles(values: np.ndarray) -> str:
    points = (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0)
    return " ".join(
        f"p{int(point * 100):02d}={value:.6f}"
        for point, value in zip(points, np.quantile(values, points))
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", type=Path)
    parser.add_argument(
        "--thresholds",
        default="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9",
        help="comma-separated translation-magnitude thresholds",
    )
    args = parser.parse_args()

    rows = []
    with args.trace.open(encoding="utf-8") as stream:
        for line in stream:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"empty trace: {args.trace}")

    metrics = {
        key: np.asarray([float(row[key]) for row in rows], dtype=np.float64)
        for key in (
            "trans_mag_mean",
            "trans_mag_max",
            "pair_action_l1",
            "pair_action_linf",
            "pair_trans_l2_mean",
            "pair_trans_l2_max",
        )
    }
    print(f"trace={args.trace} queries={len(rows)} draft_steps={rows[0]['draft_steps']}")
    for key, values in metrics.items():
        print(f"{key}: {_quantiles(values)} mean={values.mean():.6f}")

    thresholds = [float(value) for value in args.thresholds.split(",") if value.strip()]
    trans = metrics["trans_mag_mean"]
    error = metrics["pair_trans_l2_mean"]
    print("threshold draft_ratio draft_queries draft_pair_trans_l2_mean full_pair_trans_l2_mean")
    for threshold in thresholds:
        draft = trans > threshold
        draft_mean = float(error[draft].mean()) if draft.any() else float("nan")
        full_mean = float(error[~draft].mean()) if (~draft).any() else float("nan")
        print(
            f"{threshold:.6f} {draft.mean():.6f} {int(draft.sum())} "
            f"{draft_mean:.6f} {full_mean:.6f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
