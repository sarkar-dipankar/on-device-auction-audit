#!/usr/bin/env python3
"""Generate every derived Track B statistic from raw replicate artifacts.

Run from the repository root with the checked-in virtual environment:

    PYTHONDONTWRITEBYTECODE=1 research/.venv/bin/python \
        scripts/generate_track_b_paper_numbers.py

The script deliberately uses paired rows for controller contrasts, the
expected-bound evaluation, and the payment/reserve/calibration factorial. It
also audits the budget-compliance-adjusted delivered-value score on every raw
replicate before writing output. That score is not economic welfare.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any, Iterable

from scipy.stats import t as student_t


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = ROOT / "results"
DATASETS = ("track_b_smoke", "track_b_ipinyou")
METRICS = (
    "budget_cents",
    "overspend_pct",
    "underspend_pct",
    "compliance_adjusted_delivered_value_ratio",
    "spend_cv",
    "auctions_won",
    "avg_clearing_price_cents",
    "billing_rounding_adjustment_cents",
    "manipulable_auction_rate",
    "manipulable_rival_auction_rate",
    "misallocation_rate",
    "unprofitable_single_bidder_sales",
)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def summary(values: Iterable[float]) -> dict[str, float | int]:
    xs = [float(x) for x in values]
    if not xs:
        raise AssertionError("cannot summarise an empty sample")
    mean = statistics.mean(xs)
    sd = statistics.stdev(xs) if len(xs) > 1 else 0.0
    critical = float(student_t.ppf(0.975, len(xs) - 1)) if len(xs) > 1 else 0.0
    half = critical * sd / math.sqrt(len(xs))
    return {
        "n": len(xs),
        "mean": mean,
        "sd": sd,
        "ci95_halfwidth": half,
        "ci95_low": mean - half,
        "ci95_high": mean + half,
        "ci95_method": "two-sided Student-t",
        "min": min(xs),
        "max": max(xs),
    }


def index_rows(rows: Iterable[dict[str, Any]]) -> dict[tuple[str, str, int, int], dict[str, Any]]:
    out: dict[tuple[str, str, int, int], dict[str, Any]] = {}
    for row in rows:
        key = (row["strategy"], row["payment_rule"], int(row["sync_interval"]), int(row["seed"]))
        if key in out:
            raise AssertionError(f"duplicate raw row {key}")
        out[key] = row
    return out


def paired_controller_effects(rows: list[dict[str, Any]]) -> dict[str, Any]:
    indexed = index_rows(rows)
    deltas = sorted({int(r["sync_interval"]) for r in rows if r["strategy"] == "even" and r["payment_rule"] == "second_score" and int(r["sync_interval"]) > 0})
    result: dict[str, Any] = {}
    for delta in deltas:
        even = {int(r["seed"]): r for r in rows if r["strategy"] == "even" and r["payment_rule"] == "second_score" and int(r["sync_interval"]) == delta}
        pi = {int(r["seed"]): r for r in rows if r["strategy"] == "pid" and r["payment_rule"] == "second_score" and int(r["sync_interval"]) == delta}
        if even.keys() != pi.keys():
            raise AssertionError(f"unpaired controller seeds at delta={delta}")
        # Touch the full index so accidental duplicate keys cannot be hidden by
        # the seed dictionaries above.
        for seed in even:
            assert indexed[("even", "second_score", delta, seed)] is even[seed]
            assert indexed[("pid", "second_score", delta, seed)] is pi[seed]
        diffs = [pi[s]["overspend_pct"] - even[s]["overspend_pct"] for s in sorted(even)]
        stat = summary(diffs)
        if stat["ci95_high"] < 0:
            conclusion = "pi_lower"
        elif stat["ci95_low"] > 0:
            conclusion = "pi_higher"
        else:
            conclusion = "inconclusive"
        result[str(delta)] = {"estimand": "PI minus Even overspend (percentage points)", **stat, "conclusion": conclusion}
    return result


def cell_summaries(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[tuple[str, str, int], list[dict[str, Any]]] = collections.defaultdict(list)
    for row in rows:
        groups[(row["strategy"], row["payment_rule"], int(row["sync_interval"]))].append(row)
    out: dict[str, Any] = {}
    for (strategy, payment, delta), group in sorted(groups.items()):
        key = f"{strategy}/{payment}"
        out.setdefault(key, {})[str(delta)] = {metric: summary(r[metric] for r in group) for metric in METRICS}
    return out


def replicate_summary_audit(
    summary_rows: list[dict[str, Any]], raw_rows: list[dict[str, Any]], dataset: str
) -> dict[str, Any]:
    groups: dict[tuple[str, str, int], list[dict[str, Any]]] = collections.defaultdict(list)
    for row in raw_rows:
        groups[(row["strategy"], row["payment_rule"], int(row["sync_interval"]))].append(row)
    if len(summary_rows) != len(groups):
        raise AssertionError(f"{dataset}: replicate summary has the wrong number of cells")
    checked = 0
    methods: set[str] = set()
    for row in summary_rows:
        key = (row["strategy"], row["payment_rule"], int(row["sync_interval"]))
        group = groups.get(key)
        if group is None or int(row["replicates"]) != len(group):
            raise AssertionError(f"{dataset}: malformed replicate-summary cell {key}")
        for metric, recorded in row.items():
            if not isinstance(recorded, dict) or "ci95_halfwidth" not in recorded:
                continue
            expected = summary(raw[metric] for raw in group)
            methods.add(str(recorded.get("ci95_method")))
            for field in ("mean", "sd", "ci95_halfwidth", "min", "max"):
                if not math.isclose(
                    float(recorded[field]),
                    float(expected[field]),
                    rel_tol=1e-12,
                    abs_tol=1e-12,
                ):
                    raise AssertionError(
                        f"{dataset}: replicate summary differs at {key}/{metric}/{field}"
                    )
            if recorded.get("ci95_method") != "two-sided Student-t":
                raise AssertionError(f"{dataset}: non-Student-t summary at {key}/{metric}")
            checked += 1
    return {
        "cells": len(summary_rows),
        "metric_summaries_checked": checked,
        "ci95_methods": sorted(methods),
        "status": "matches raw rows and the paper producer",
    }


def assert_delivered_value_score(rows: list[dict[str, Any]], dataset: str) -> dict[str, int]:
    checked = 0
    affine_matches = 0
    for row in rows:
        normaliser = int(row["oracle_compliance_adjusted_value_cents"])
        if normaliser <= 0:
            raise AssertionError(f"{dataset}: non-positive oracle value score at seed {row['seed']}")
        net = int(row["authorised_delivered_value_cents"]) - int(row["excess_debit_cents"])
        expected = net / normaliser
        actual = float(row["compliance_adjusted_delivered_value_ratio"])
        if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-12):
            raise AssertionError(f"{dataset}: delivered-value identity failed at {row['strategy']}/{row['sync_interval']}/seed={row['seed']}")
        if int(row["sync_interval"]) == 0 and not math.isclose(actual, 1.0, rel_tol=0.0, abs_tol=1e-12):
            raise AssertionError(f"{dataset}: zero-lag delivered-value score is not one at seed {row['seed']}")

        # The only score inequality claimed in the paper follows from U>=0:
        # 1-Q <= 1+D/Q0. Assert it on every replicate, not just on cell means.
        lhs = 1.0 - actual
        rhs = 1.0 + int(row["excess_debit_cents"]) / normaliser
        if lhs > rhs + 1e-12:
            raise AssertionError(f"{dataset}: delivered-value upper inequality failed at seed {row['seed']}")

        if math.isclose(actual, 1.0 - float(row["overspend_pct"]) / 100.0, rel_tol=0.0, abs_tol=1e-12):
            affine_matches += 1
        checked += 1
    return {"replicate_rows_checked": checked, "rows_matching_old_affine_identity": affine_matches}


def calibration_audit(rows: list[dict[str, Any]], manifest: dict[str, Any], dataset: str) -> dict[str, Any]:
    protocol = manifest["budget_calibration"]
    pilot_replicates = int(protocol["pilot_replicates"])
    expected = 2 * (pilot_replicates + 1) * 12
    if len(rows) != expected:
        raise AssertionError(f"{dataset}: expected {expected} calibration rows, found {len(rows)}")
    pilot_seeds = set(range(int(protocol["pilot_seed_start"]), int(protocol["pilot_seed_end"]) + 1))
    evaluation_seeds = set(range(int(protocol["evaluation_seed_start"]), int(protocol["evaluation_seed_end"]) + 1))
    if pilot_seeds & evaluation_seeds:
        raise AssertionError(f"{dataset}: pilot and evaluation seeds overlap")
    if not protocol.get("budget_vectors_frozen_before_evaluation"):
        raise AssertionError(f"{dataset}: manifest does not assert frozen ex-ante budgets")
    groups: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in rows:
        groups[row["design"]].append(row)
    out: dict[str, Any] = {}
    for design, group in sorted(groups.items()):
        pilot = [row for row in group if row["record_kind"] == "pilot_run"]
        frozen = [row for row in group if row["record_kind"] == "frozen_summary"]
        if len(pilot) != pilot_replicates * 12 or len(frozen) != 12:
            raise AssertionError(f"{dataset}/{design}: malformed pilot/frozen row counts")
        if {int(row["seed"]) for row in pilot} != pilot_seeds:
            raise AssertionError(f"{dataset}/{design}: incomplete pilot seed block")
        if any(row["budget_floor_cents"] is not None for row in pilot + frozen):
            raise AssertionError(f"{dataset}/{design}: primary calibration unexpectedly uses a floor")
        effective = [row["effective_oversubscription"] for row in frozen if row["effective_oversubscription"] is not None]
        aggregate_budget = sum(int(row["applied_budget_cents"]) for row in frozen)
        aggregate_pilot_spend_by_seed: dict[int, int] = collections.defaultdict(int)
        for row in pilot:
            aggregate_pilot_spend_by_seed[int(row["seed"])] += int(row["unconstrained_spend_cents"])
        per_campaign_applied = [
            int(row["applied_budget_cents"]) / int(row["campaigns"]) for row in frozen
        ]
        out[design] = {
            "pilot_vertical_rows": len(pilot),
            "frozen_vertical_rows": len(frozen),
            "pilot_seed_start": min(pilot_seeds),
            "pilot_seed_end": max(pilot_seeds),
            "zero_unconstrained_pilot_verticals": sum(int(row["unconstrained_spend_cents"]) == 0 for row in pilot),
            "pilot_unconstrained_spend_cents": summary(row["unconstrained_spend_cents"] for row in pilot),
            "aggregate_pilot_spend_cents_by_seed": summary(aggregate_pilot_spend_by_seed.values()),
            "frozen_pre_floor_budget_cents": summary(row["pre_floor_budget_cents"] for row in frozen),
            "frozen_applied_budget_cents": summary(row["applied_budget_cents"] for row in frozen),
            "applied_budget_per_campaign_cents": summary(per_campaign_applied),
            "frozen_effective_oversubscription_vertical": summary(effective) if effective else None,
            "aggregate_budget_cents_frozen": aggregate_budget,
        }
    primary = out["primary_second_score_no_reserve_frozen"]
    if primary["applied_budget_per_campaign_cents"]["min"] <= 500:
        raise AssertionError(
            f"{dataset}: raised inventory did not clear the separately audited 500-cent floor naturally"
        )
    return out


def factorial_summary(rows: list[dict[str, Any]], manifest: dict[str, Any], dataset: str) -> dict[str, Any]:
    expected = 8 * int(manifest["replicates"])
    if len(rows) != expected:
        raise AssertionError(f"{dataset}: expected {expected} factorial rows, found {len(rows)}")
    key_fields = ("payment_rule", "reserve_base_cents", "budget_policy")
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = collections.defaultdict(list)
    indexed: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in rows:
        key = tuple(row[field] for field in key_fields)
        groups[key].append(row)
        full_key = (*key, int(row["seed"]))
        if full_key in indexed:
            raise AssertionError(f"{dataset}: duplicate factorial cell {full_key}")
        indexed[full_key] = row
    for policy in ("calibrated", "floored_500"):
        budgets = {int(row["budget_cents"]) for row in rows if row["budget_policy"] == policy}
        if len(budgets) != 1:
            raise AssertionError(f"{dataset}: factorial {policy} budget changed across payment/reserve/seeds")
    if not manifest["budget_calibration"].get("factorial_budget_vector_shared_across_payment_and_reserve"):
        raise AssertionError(f"{dataset}: factorial freeze is not recorded in manifest")
    metrics = (
        "budget_cents",
        "total_spent_cents",
        "overspend_pct",
        "underspend_pct",
        "auctions_won",
        "avg_clearing_price_cents",
        "manipulable_rival_auction_rate",
        "misallocation_rate",
    )
    cells: dict[str, Any] = {}
    for key, group in sorted(groups.items()):
        label = f"{key[0]}/reserve_{key[1]}/{key[2]}"
        cells[label] = {metric: summary(row[metric] for row in group) for metric in metrics}

    contrasts: dict[str, Any] = {}
    seeds = range(int(manifest["seed"]), int(manifest["seed"]) + int(manifest["replicates"]))
    for reserve in (0, 100):
        for policy in ("calibrated", "floored_500"):
            diffs = [
                indexed[("critical_bid", reserve, policy, seed)]["underspend_pct"]
                - indexed[("second_score", reserve, policy, seed)]["underspend_pct"]
                for seed in seeds
            ]
            contrasts[f"critical_minus_score_underspend_pp/reserve_{reserve}/{policy}"] = summary(diffs)
    for payment in ("second_score", "critical_bid"):
        for policy in ("calibrated", "floored_500"):
            diffs = [
                indexed[(payment, 100, policy, seed)]["underspend_pct"]
                - indexed[(payment, 0, policy, seed)]["underspend_pct"]
                for seed in seeds
            ]
            contrasts[f"reserve_100_minus_none_underspend_pp/{payment}/{policy}"] = summary(diffs)
    for payment in ("second_score", "critical_bid"):
        for reserve in (0, 100):
            diffs = [
                indexed[(payment, reserve, "floored_500", seed)]["underspend_pct"]
                - indexed[(payment, reserve, "calibrated", seed)]["underspend_pct"]
                for seed in seeds
            ]
            contrasts[f"floor_minus_calibrated_underspend_pp/{payment}/reserve_{reserve}"] = summary(diffs)
    return {
        "design": "2 payment rules x 2 base-bid reserves x 2 shared frozen budget policies",
        "budget_control": "within each budget-policy level, the identical vector is held across payment, reserve, and evaluation seeds",
        "cells": cells,
        "paired_contrasts": contrasts,
    }


def sensitivity_summary(rows: list[dict[str, Any]], manifest: dict[str, Any], dataset: str) -> dict[str, Any]:
    expected = 6 * int(manifest["replicates"])
    if len(rows) != expected:
        raise AssertionError(f"{dataset}: expected {expected} sensitivity rows, found {len(rows)}")
    groups: dict[tuple[str, int], list[dict[str, Any]]] = collections.defaultdict(list)
    for row in rows:
        groups[(row["sensitivity"], int(row["level"]))].append(row)
    n_rows = [row for row in rows if row["sensitivity"] == "n_devices"]
    if {int(row["arrivals_per_tick"]) for row in n_rows} != {int(manifest["arrivals_per_tick"])}:
        raise AssertionError(f"{dataset}: N sensitivity changed aggregate arrivals")
    if len({int(row["budget_cents"]) for row in rows}) != 1:
        raise AssertionError(f"{dataset}: sensitivity did not hold the primary budget fixed")
    metrics = ("overspend_pct", "underspend_pct", "auctions_won", "avg_clearing_price_cents")
    cells = {}
    for (factor, level), group in sorted(groups.items()):
        cells[f"{factor}/{level}"] = {
            "n_devices": int(group[0]["n_devices"]),
            "arrivals_per_tick": int(group[0]["arrivals_per_tick"]),
            "nominal_pressure": float(group[0]["nominal_pressure"]),
            **{metric: summary(row[metric] for row in group) for metric in metrics},
        }
    return {
        "scope": "Even controller, second-score payment, Delta=1, frozen primary budget",
        "cells": cells,
    }


def resolve_contexts(manifest: dict[str, Any]) -> Path:
    raw = Path(manifest["contexts_path"])
    candidates = (ROOT / raw, ROOT / "harness" / raw, raw)
    for path in candidates:
        if path.is_file():
            return path.resolve()
    raise FileNotFoundError(f"cannot resolve contexts_path={raw}")


def vertical_arrival_rates(manifest: dict[str, Any], topics: set[str]) -> dict[str, float]:
    counts: collections.Counter[str] = collections.Counter()
    total = 0
    for line in resolve_contexts(manifest).read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        total += 1
        if (
            row.get("ad_eligible")
            and row.get("safety_class") == "ok"
            and row.get("topic") in topics
            and row.get("intent_stage") in {"informational", "comparison", "purchase_ready"}
        ):
            counts[row["topic"]] += 1
    rates = {topic: manifest["arrivals_per_tick"] * counts[topic] / total for topic in sorted(topics)}
    expected = manifest["arrivals_per_tick"] * manifest["eligible_pair_fraction"] * manifest["n_campaigns"] / 3
    if not math.isclose(sum(rates.values()), expected, rel_tol=0.0, abs_tol=1e-12):
        raise AssertionError("vertical arrival rates do not reconcile with manifest eligibility")
    return rates


def rust_round_positive(value: float) -> int:
    return math.floor(value + 0.5)


def deterministic_payment_caps(manifest: dict[str, Any], budgets: list[dict[str, Any]]) -> dict[str, int]:
    distribution = manifest["bid_distribution"]
    if distribution.get("kind") != "fixed":
        raise AssertionError("finite payment caps are evaluated only for the bounded fixed-value sweep")
    target_mean = int(distribution["cents"])
    catalogue_mean = statistics.mean(int(row["catalogue_bid_cents"]) for row in budgets)
    values_by_topic: dict[str, list[int]] = collections.defaultdict(list)
    for row in budgets:
        value = rust_round_positive(target_mean * int(row["catalogue_bid_cents"]) / catalogue_mean)
        values_by_topic[row["topic"]].append(value)
    caps: dict[str, int] = {}
    for topic, values in values_by_topic.items():
        values.sort(reverse=True)
        # With >=2 bidders, the second-highest score is capped by the
        # second-highest unpaced value times the SDK quality cap 0.8. With one
        # bidder, the SDK charges its 100-cent floor plus one. The ledger rounds
        # the exact SDK result at the explicit billing boundary.
        multi_bidder_cap = rust_round_positive(0.8 * values[1] + 1.0)
        caps[topic] = max(101, multi_bidder_cap)
    return dict(sorted(caps.items()))


def theorem_expected_bound(
    rows: list[dict[str, Any]], manifest: dict[str, Any], budgets: list[dict[str, Any]]
) -> dict[str, Any]:
    if len({int(row["budget_cents"]) for row in rows if row["payment_rule"] == "second_score"}) != 1:
        raise AssertionError("theorem evaluation requires one fixed budget across evaluation seeds")
    if not manifest["budget_calibration"].get("budget_vectors_frozen_before_evaluation"):
        raise AssertionError("theorem evaluation budget is not marked frozen")
    topics = {row["topic"] for row in budgets}
    campaign_topic = {row["campaign_id"]: row["topic"] for row in budgets}
    rates = vertical_arrival_rates(manifest, topics)
    caps = deterministic_payment_caps(manifest, budgets)
    ticks = int(manifest["ticks"])
    deltas = sorted({int(r["sync_interval"]) for r in rows if r["strategy"] == "even" and r["payment_rule"] == "second_score" and int(r["sync_interval"]) > 0})
    cells: dict[str, Any] = {}
    for delta in deltas:
        group = [r for r in rows if r["strategy"] == "even" and r["payment_rule"] == "second_score" and int(r["sync_interval"]) == delta]
        bounds: list[float] = []
        empirical: list[float] = []
        slack: list[float] = []
        for row in group:
            by_topic: dict[str, list[float]] = collections.defaultdict(list)
            for campaign, fraction in row["campaign_exhaustion_fractions"].items():
                by_topic[campaign_topic[campaign]].append(float(fraction))
            debit_bound = 0.0
            for topic, fractions in by_topic.items():
                exhausted = len(fractions)
                first_exhaustion_tick = ticks * min(fractions)
                contaminated_length = min(exhausted * delta, max(0.0, ticks - first_exhaustion_tick))
                debit_bound += caps[topic] * (exhausted + rates[topic] * contaminated_length)
            bound_pct = 100.0 * debit_bound / int(row["budget_cents"])
            bounds.append(bound_pct)
            empirical.append(float(row["overspend_pct"]))
            slack.append(bound_pct - float(row["overspend_pct"]))
        cells[str(delta)] = {
            "bound_pct": summary(bounds),
            "empirical_pct": summary(empirical),
            "paired_slack_pct_points": summary(slack),
        }
    return {
        "status": "expected bound evaluated with paired replicate estimators; no pointwise verification claim",
        "scope": "bounded deterministic-value arm only; the untruncated log-normal proxy is outside the theorem premises",
        "payment_rule": "SDK second-score-plus-one with explicit nearest-cent ledger billing",
        "formula": "E[D]/B <= sum_v pbar_v * E[K_v + A_v min(K_v*Delta,(T-tau_v)_+)] / B",
        "conditional_payment_caps_cents": caps,
        "vertical_arrivals_per_tick": rates,
        "total_servable_arrivals_per_tick": sum(rates.values()),
        "cells": cells,
    }


def strategic_audit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    deployed = [r for r in rows if r["strategy"] == "even" and r["payment_rule"] == "second_score"]
    critical = [r for r in rows if r["strategy"] == "even" and r["payment_rule"] == "critical_bid"]
    return {
        "deployed_rival_auction_share": summary(r["auctions_with_rival"] / r["auctions_won"] for r in deployed if r["auctions_won"]),
        "deployed_manipulable_rival_auction_rate": summary(r["manipulable_rival_auction_rate"] for r in deployed),
        "deployed_unprofitable_single_bidder_sales": summary(r["unprofitable_single_bidder_sales"] for r in deployed),
        "critical_manipulation_gain_cents": summary(r["manipulation_gain_cents"] for r in critical),
        "critical_misallocation_rate": summary(r["misallocation_rate"] for r in critical),
    }


def rounding_audit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    deployed = [r for r in rows if r["payment_rule"] == "second_score"]
    totals = [float(r["billing_rounding_adjustment_cents"]) for r in deployed]
    per_sale = [float(r["billing_rounding_adjustment_cents"]) / r["auctions_won"] for r in deployed if r["auctions_won"]]
    return {"total_adjustment_cents_per_run": summary(totals), "adjustment_cents_per_sale": summary(per_sale)}


def label_perturbation_audit(
    directory: Path, baseline_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    manifest_path = directory / "label_perturbation_manifest.json"
    rows_path = directory / "label_perturbation.jsonl"
    calibration_path = directory / "label_perturbation_calibration.jsonl"
    manifest = read_json(manifest_path)
    rows = read_jsonl(rows_path)
    calibrations = read_jsonl(calibration_path)
    design = manifest["design"]
    replicates = int(design["evaluation_replicates"])
    deltas = [int(value) for value in design["sync_intervals"]]
    if len(rows) != replicates * len(deltas):
        raise AssertionError("label perturbation has an incomplete lag-by-seed grid")
    if len(calibrations) != (int(design["pilot_replicates"]) + 1) * 12:
        raise AssertionError("label perturbation has incomplete pilot/frozen calibration rows")
    pilot = set(range(int(design["pilot_seed_start"]), int(design["pilot_seed_end"]) + 1))
    evaluation = set(
        range(int(design["evaluation_seed_start"]), int(design["evaluation_seed_end"]) + 1)
    )
    if pilot & evaluation or not design["pilot_evaluation_disjoint"]:
        raise AssertionError("label perturbation pilot and evaluation seeds overlap")
    if not design["budget_vectors_frozen_before_evaluation"]:
        raise AssertionError("label perturbation budget is not frozen before evaluation")
    if len({int(row["budget_cents"]) for row in rows}) != 1:
        raise AssertionError("label perturbation changes its frozen budget across cells")

    perturb_index = {(int(row["sync_interval"]), int(row["seed"])): row for row in rows}
    baseline_index = {
        (int(row["sync_interval"]), int(row["seed"])): row
        for row in baseline_rows
        if row["strategy"] == "even" and row["payment_rule"] == "second_score"
    }
    if perturb_index.keys() != baseline_index.keys():
        raise AssertionError("label perturbation is not paired to the baseline seed/lag grid")

    cells: dict[str, Any] = {}
    mean_curve: list[float] = []
    for delta in deltas:
        group = [perturb_index[(delta, seed)] for seed in sorted(evaluation)]
        baseline = [baseline_index[(delta, seed)] for seed in sorted(evaluation)]
        overspend = summary(row["overspend_pct"] for row in group)
        mean_curve.append(float(overspend["mean"]))
        cells[str(delta)] = {
            "overspend_pct": overspend,
            "underspend_pct": summary(row["underspend_pct"] for row in group),
            "paired_overspend_change_from_unperturbed_pp": summary(
                changed["overspend_pct"] - original["overspend_pct"]
                for changed, original in zip(group, baseline)
            ),
        }

    lag_effects: dict[str, Any] = {}
    for delta in deltas[1:]:
        lag_effects[str(delta)] = summary(
            perturb_index[(delta, seed)]["overspend_pct"]
            - perturb_index[(0, seed)]["overspend_pct"]
            for seed in sorted(evaluation)
        )
    all_lag_effect_intervals_positive = all(
        stat["ci95_low"] > 0 for stat in lag_effects.values()
    )
    mean_curve_strictly_increasing = all(
        later > earlier for earlier, later in zip(mean_curve, mean_curve[1:])
    )
    monotone_replicates = sum(
        all(
            perturb_index[(later, seed)]["overspend_pct"]
            > perturb_index[(earlier, seed)]["overspend_pct"]
            for earlier, later in zip(deltas, deltas[1:])
        )
        for seed in evaluation
    )
    if not mean_curve_strictly_increasing or not all_lag_effect_intervals_positive:
        raise AssertionError("headline lag conclusion does not survive label perturbation")

    return {
        "status": "sensitivity analysis, not an estimated label-error model",
        "manifest": manifest,
        "artifact_sha256": {
            manifest_path.name: sha256(manifest_path),
            rows_path.name: sha256(rows_path),
            calibration_path.name: sha256(calibration_path),
        },
        "cells": cells,
        "paired_lag_minus_zero_effect_pp": lag_effects,
        "mean_curve_strictly_increasing": mean_curve_strictly_increasing,
        "replicates_with_strictly_increasing_curve": monotone_replicates,
        "replicates": replicates,
        "all_lag_minus_zero_ci95_intervals_positive": all_lag_effect_intervals_positive,
    }


def ipinyou_value_construction(
    config: dict[str, Any], manifest: dict[str, Any], budgets: list[dict[str, Any]]
) -> dict[str, Any]:
    base_mean = float(config["mean_market_price_fen_per_cpm"])
    catalogue_mean = statistics.mean(float(row["catalogue_bid_cents"]) for row in budgets)
    bids_by_topic: dict[str, list[float]] = collections.defaultdict(list)
    for row in budgets:
        bids_by_topic[row["topic"]].append(float(row["catalogue_bid_cents"]))
    counts: collections.Counter[str] = collections.Counter()
    for line in resolve_contexts(manifest).read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if (
            row.get("ad_eligible")
            and row.get("safety_class") == "ok"
            and row.get("topic") in bids_by_topic
            and row.get("intent_stage") in {"informational", "comparison", "purchase_ready"}
        ):
            counts[row["topic"]] += 1
    eligible_bidders = sum(counts[topic] * len(bids) for topic, bids in bids_by_topic.items())
    mixture_catalogue_mean = sum(
        counts[topic] * sum(bids) for topic, bids in bids_by_topic.items()
    ) / eligible_bidders
    pre_rounding_mean = base_mean * mixture_catalogue_mean / catalogue_mean
    return {
        "published_campaign_1458_mean_fen_per_cpm": base_mean,
        "simulator_mapping": "one published fen-per-CPM numerical unit to one simulator cent",
        "unscaled_base_draw_mean_cents": base_mean,
        "catalogue_bid_pool_mean_cents": catalogue_mean,
        "eligible_contexts": sum(counts.values()),
        "eligible_bidders": eligible_bidders,
        "eligible_bidder_mixture_catalogue_mean_cents": mixture_catalogue_mean,
        "eligible_bidder_mixture_pre_rounding_mean_cents": pre_rounding_mean,
        "formula": "base draw * catalogue_bid_cents / catalogue_bid_pool_mean_cents",
    }


def build(results_root: Path) -> dict[str, Any]:
    ipinyou = read_json(ROOT / "data/ipinyou_1458.json")
    output: dict[str, Any] = {
        "schema_version": 3,
        "producer": "scripts/generate_track_b_paper_numbers.py",
        "producer_sha256": sha256(Path(__file__)),
        "ipinyou_parameterisation": ipinyou,
        "datasets": {},
        "theorem1_bound_curve": {
            "what": "Expected overspend bound under conditional payment caps, evaluated with paired replicate confidence intervals",
            "datasets": {},
        },
    }
    for dataset in DATASETS:
        directory = results_root / dataset
        manifest = read_json(directory / "manifest.json")
        rows = read_jsonl(directory / "replicates.jsonl")
        replicate_summaries = read_jsonl(directory / "replicate_summary.jsonl")
        budgets = read_jsonl(directory / "budgets.jsonl")
        calibrations = read_jsonl(directory / "calibration.jsonl")
        factorial = read_jsonl(directory / "factorial_ablation.jsonl")
        sensitivity = read_jsonl(directory / "sensitivity.jsonl")
        if len(rows) != 30 * manifest["replicates"]:
            raise AssertionError(f"{dataset}: expected 30 cells per replicate, found {len(rows)} rows")
        entry: dict[str, Any] = {
            "manifest": manifest,
            "artifact_sha256": {name: sha256(directory / name) for name in ("manifest.json", "budgets.jsonl", "calibration.jsonl", "factorial_ablation.jsonl", "factorial_calibration.jsonl", "replicates.jsonl", "replicate_summary.jsonl", "sensitivity.jsonl")},
            "cells": cell_summaries(rows),
            "replicate_summary_audit": replicate_summary_audit(
                replicate_summaries, rows, dataset
            ),
            "paired_controller_effects": paired_controller_effects(rows),
            "delivered_value_score_audit": assert_delivered_value_score(rows, dataset),
            "calibration_audit": calibration_audit(calibrations, manifest, dataset),
            "factorial_ablation": factorial_summary(factorial, manifest, dataset),
            "sensitivity": sensitivity_summary(sensitivity, manifest, dataset),
            "strategic_audit": strategic_audit(rows),
            "rounding_audit": rounding_audit(rows),
        }
        output["datasets"][dataset] = entry
        if dataset == "track_b_smoke":
            theorem = theorem_expected_bound(rows, manifest, budgets)
            entry["theorem1_expected_bound"] = theorem
            output["theorem1_bound_curve"]["datasets"][dataset] = {
                "bound_pct_mean_over_replicates": {
                    delta: cell["bound_pct"]["mean"] for delta, cell in theorem["cells"].items()
                },
                "empirical_even_pct": {
                    delta: cell["empirical_pct"]["mean"] for delta, cell in theorem["cells"].items()
                },
            }
            entry["label_perturbation"] = label_perturbation_audit(directory, rows)
        elif dataset == "track_b_ipinyou":
            entry["value_construction"] = ipinyou_value_construction(ipinyou, manifest, budgets)
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_RESULTS / "track_b_paper_numbers.json")
    parser.add_argument("--check", action="store_true", help="validate artifacts without writing")
    args = parser.parse_args()
    result = build(args.results_root.resolve())
    if not args.check:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        print(f"[track-b-numbers] wrote {args.output}")
    else:
        print("[track-b-numbers] all calibration, factorial, paired-statistics, and delivered-value assertions passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
