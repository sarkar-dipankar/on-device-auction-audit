#!/usr/bin/env python3
"""Audit and summarise the Track-B pressure and visible-balance experiments."""

from __future__ import annotations

import collections
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Iterable

from scipy.stats import t as student_t


ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = ROOT / "results" / "track_b_pressure_balance"
BURST_DIR = ROOT / "results" / "track_b_bursty_heterogeneous"
PRIMARY_DIR = ROOT / "results" / "track_b_smoke"
OUTPUT = ROOT / "results" / "track_b_revision_numbers.json"


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def summary(values: Iterable[float]) -> dict[str, float | int | str]:
    xs = [float(value) for value in values]
    if not xs:
        raise AssertionError("cannot summarise an empty sample")
    mean = statistics.mean(xs)
    sd = statistics.stdev(xs) if len(xs) > 1 else 0.0
    critical = float(student_t.ppf(0.975, len(xs) - 1)) if len(xs) > 1 else 0.0
    halfwidth = critical * sd / math.sqrt(len(xs))
    return {
        "n": len(xs),
        "mean": mean,
        "sd": sd,
        "ci95_halfwidth": halfwidth,
        "ci95_low": mean - halfwidth,
        "ci95_high": mean + halfwidth,
        "ci95_method": "two-sided Student-t",
        "min": min(xs),
        "max": max(xs),
    }


def index(rows: list[dict], experiment: str) -> dict[tuple[float, int, int], dict]:
    indexed = {}
    for row in rows:
        if row["experiment"] != experiment:
            continue
        key = (
            float(row["pressure_target"]),
            int(row["sync_interval"]),
            int(row["seed"]),
        )
        if key in indexed:
            raise AssertionError(f"duplicate {experiment} cell {key}")
        indexed[key] = row
    return indexed


def main() -> None:
    rows_path = RUN_DIR / "replicates.jsonl"
    manifest_path = RUN_DIR / "manifest.json"
    rows = read_jsonl(rows_path)
    manifest = json.loads(manifest_path.read_text())
    pressure = index(rows, "pressure_sensitivity")
    guard = index(rows, "visible_balance_guard")
    pressures = [float(value) for value in manifest["pressure_targets"]]
    lags = [int(value) for value in manifest["sync_intervals"]]
    seeds = list(
        range(
            int(manifest["evaluation_seed_start"]),
            int(manifest["evaluation_seed_end"]) + 1,
        )
    )
    expected_pressure = len(pressures) * len(lags) * len(seeds)
    expected_guard = len(lags) * len(seeds)
    if len(pressure) != expected_pressure or len(guard) != expected_guard:
        raise AssertionError(
            f"wrong cell counts: pressure={len(pressure)}/{expected_pressure}, "
            f"guard={len(guard)}/{expected_guard}"
        )

    pressure_cells: dict[str, dict[str, dict]] = collections.defaultdict(dict)
    for target in pressures:
        for lag in lags:
            group = [pressure[(target, lag, seed)] for seed in seeds]
            pressure_cells[f"{target:g}"][str(lag)] = {
                metric: summary(row[metric] for row in group)
                for metric in (
                    "budget_score_units",
                    "overspend_pct",
                    "underspend_pct",
                    "auctions_won",
                    "avg_clearing_score_units",
                )
            }

    guard_cells = {}
    guard_effects = {}
    for lag in lags:
        guarded = [guard[(20.0, lag, seed)] for seed in seeds]
        guard_cells[str(lag)] = {
            metric: summary(row[metric] for row in guarded)
            for metric in (
                "overspend_pct",
                "underspend_pct",
                "auctions_won",
                "avg_clearing_score_units",
            )
        }
        guard_effects[str(lag)] = summary(
            guard[(20.0, lag, seed)]["overspend_pct"]
            - pressure[(20.0, lag, seed)]["overspend_pct"]
            for seed in seeds
        )
    if any(guard[(20.0, 0, seed)]["excess_debit_score_units"] != 0 for seed in seeds):
        raise AssertionError("visible-balance zero-lag arm has excess debit")

    burst_rows_path = BURST_DIR / "replicates.jsonl"
    burst_manifest_path = BURST_DIR / "manifest.json"
    burst_rows = read_jsonl(burst_rows_path)
    burst_manifest = json.loads(burst_manifest_path.read_text())
    burst = index(burst_rows, "bursty_heterogeneous_demand")
    if len(burst) != len(lags) * len(seeds):
        raise AssertionError(
            f"wrong bursty/heterogeneous cell count: {len(burst)}/{len(lags) * len(seeds)}"
        )
    burst_cells = {}
    burst_lag_minus_zero = {}
    for lag in lags:
        group = [burst[(20.0, lag, seed)] for seed in seeds]
        burst_cells[str(lag)] = {
            metric: summary(row[metric] for row in group)
            for metric in (
                "overspend_pct",
                "underspend_pct",
                "auctions_run",
                "auctions_won",
                "avg_clearing_score_units",
            )
        }
        burst_lag_minus_zero[str(lag)] = summary(
            burst[(20.0, lag, seed)]["overspend_pct"]
            - burst[(20.0, 0, seed)]["overspend_pct"]
            for seed in seeds
        )
    burst_means = [burst_cells[str(lag)]["overspend_pct"]["mean"] for lag in lags]
    burst_mean_strictly_increasing = all(
        left < right for left, right in zip(burst_means, burst_means[1:])
    )
    burst_replicates_strictly_increasing = sum(
        all(
            burst[(20.0, left, seed)]["overspend_pct"]
            < burst[(20.0, right, seed)]["overspend_pct"]
            for left, right in zip(lags, lags[1:])
        )
        for seed in seeds
    )

    primary_rows = read_jsonl(PRIMARY_DIR / "replicates.jsonl")
    primary = {
        (int(row["sync_interval"]), int(row["seed"])): row
        for row in primary_rows
        if row["strategy"] == "even" and row["payment_rule"] == "second_score"
    }
    differences = []
    for lag in lags:
        for seed in seeds:
            differences.append(
                pressure[(20.0, lag, seed)]["overspend_pct"]
                - primary[(lag, seed)]["overspend_pct"]
            )
    if any(difference != 0.0 for difference in differences):
        raise AssertionError("pressure-20 rerun does not exactly reproduce primary rows")

    output = {
        "pressure_sensitivity": {
            "estimand": "pool overspend percent",
            "cells": pressure_cells,
            "pressure_20_primary_replication": {
                "rows_compared": len(differences),
                "max_absolute_overspend_difference": max(map(abs, differences)),
                "status": "exact match",
            },
        },
        "visible_balance_guard": {
            "estimand": "pool overspend percent",
            "cells": guard_cells,
            "paired_guard_minus_standard_overspend_pp": guard_effects,
            "zero_lag_exact": True,
            "zero_lag_replicates_with_excess_debit": 0,
        },
        "bursty_heterogeneous_demand": {
            "estimand": "pool overspend percent in dimensionless score-unit accounting",
            "cells": burst_cells,
            "paired_lag_minus_zero_overspend_pp": burst_lag_minus_zero,
            "mean_curve_strictly_increasing": burst_mean_strictly_increasing,
            "replicate_curves_strictly_increasing": burst_replicates_strictly_increasing,
            "replicates": len(seeds),
            "arrival_process": burst_manifest["arrival_process"],
            "budget_policy": burst_manifest["budget_policy"],
        },
        "protocol": {
            "pressure_targets": pressures,
            "sync_intervals": lags,
            "evaluation_seeds": seeds,
            "pilot_seed_start": manifest["pilot_seed_start"],
            "pilot_seed_end": manifest["pilot_seed_end"],
            "pilot_evaluation_disjoint": manifest["pilot_evaluation_disjoint"],
            "budget_derivation": manifest["budget_derivation"],
            "device_assignment": manifest["device_assignment"],
            "accounting": manifest["accounting"],
        },
        "artifact_audit": {
            "raw_rows": len(rows),
            "pressure_rows": len(pressure),
            "visible_balance_rows": len(guard),
            "replicates_sha256": sha256(rows_path),
            "manifest_sha256": sha256(manifest_path),
            "producer": "harness/src/bin/track_b_pressure_and_balance.rs",
            "source_snapshot_sha256": manifest["source_state"]["sha256"],
            "repository_commit": manifest["provenance"]["repository_commit"],
        },
        "bursty_artifact_audit": {
            "raw_rows": len(burst_rows),
            "replicates_sha256": sha256(burst_rows_path),
            "manifest_sha256": sha256(burst_manifest_path),
            "producer": "harness/src/bin/track_b_bursty_heterogeneous.rs",
            "source_snapshot_sha256": burst_manifest["source_state"]["sha256"],
            "repository_commit": burst_manifest["provenance"]["repository_commit"],
        },
    }
    OUTPUT.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(f"wrote {OUTPUT.relative_to(ROOT)} from {len(rows)} audited raw rows")


if __name__ == "__main__":
    main()
