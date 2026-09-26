"""Artifact-level tests for Track B calibration and derived claims."""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "track_b_numbers", ROOT / "scripts/generate_track_b_paper_numbers.py"
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class TrackBAnalysisTest(unittest.TestCase):
    def test_all_replicate_delivered_value_identities_and_inequalities(self) -> None:
        for dataset in MODULE.DATASETS:
            rows = MODULE.read_jsonl(ROOT / "results" / dataset / "replicates.jsonl")
            audit = MODULE.assert_delivered_value_score(rows, dataset)
            self.assertEqual(audit["replicate_rows_checked"], 900)

    def test_controller_effects_are_paired_by_seed(self) -> None:
        for dataset in MODULE.DATASETS:
            rows = MODULE.read_jsonl(ROOT / "results" / dataset / "replicates.jsonl")
            effects = MODULE.paired_controller_effects(rows)
            self.assertEqual(set(effects), {"1", "2", "5", "10", "25", "50"})
            self.assertTrue(all(cell["n"] == 30 for cell in effects.values()))

    def test_primary_calibration_is_unfloored_and_above_legacy_threshold(self) -> None:
        for dataset in MODULE.DATASETS:
            directory = ROOT / "results" / dataset
            manifest = MODULE.read_json(directory / "manifest.json")
            rows = MODULE.read_jsonl(directory / "calibration.jsonl")
            audit = MODULE.calibration_audit(rows, manifest, dataset)
            primary = audit["primary_second_score_no_reserve_frozen"]
            self.assertGreater(primary["applied_budget_per_campaign_cents"]["min"], 500)
            protocol = manifest["budget_calibration"]
            pilot = set(range(protocol["pilot_seed_start"], protocol["pilot_seed_end"] + 1))
            evaluation = set(range(protocol["evaluation_seed_start"], protocol["evaluation_seed_end"] + 1))
            self.assertTrue(pilot.isdisjoint(evaluation))

    def test_factorial_has_all_eight_paired_cells(self) -> None:
        for dataset in MODULE.DATASETS:
            directory = ROOT / "results" / dataset
            manifest = MODULE.read_json(directory / "manifest.json")
            rows = MODULE.read_jsonl(directory / "factorial_ablation.jsonl")
            audit = MODULE.factorial_summary(rows, manifest, dataset)
            self.assertEqual(len(audit["cells"]), 8)
            self.assertTrue(all(cell["underspend_pct"]["n"] == 30 for cell in audit["cells"].values()))
            for policy in ("calibrated", "floored_500"):
                budgets = {row["budget_cents"] for row in rows if row["budget_policy"] == policy}
                self.assertEqual(len(budgets), 1)

    def test_n_and_load_sensitivities_use_one_frozen_budget(self) -> None:
        for dataset in MODULE.DATASETS:
            directory = ROOT / "results" / dataset
            manifest = MODULE.read_json(directory / "manifest.json")
            rows = MODULE.read_jsonl(directory / "sensitivity.jsonl")
            audit = MODULE.sensitivity_summary(rows, manifest, dataset)
            self.assertEqual(len(audit["cells"]), 6)
            self.assertEqual(len({row["budget_cents"] for row in rows}), 1)


if __name__ == "__main__":
    unittest.main()
