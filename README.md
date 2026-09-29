# On-device auction audit

Reproducibility artifact for **When Privacy Moves ML-Mediated Decisions On Device: Information and Incentive Misalignment in Auctions**, by **Dipankar Sarkar** (Skelf Research), previously submitted to EconML at NeurIPS 2026.

- Paper: https://arxiv.org/abs/2609.33312
- Dataset on Hugging Face: https://huggingface.co/datasets/skelfresearch/on-device-auction-audit

This is a curated release of the recorded experiment artifacts. It contains the primary and iPinYou-parameterized runs, controller comparisons, pressure/visible-balance experiments, and bursty/heterogeneous robustness sweep. Accounting uses dimensionless integer score units. Legacy `*_cents` identifiers in recorded artifacts do not confer currency semantics.

## Contents

- `results/`: raw simulation replicates, frozen budgets, calibration rows, manifests, and derived summaries. The source snapshots named in the manifests are withheld (see below); their hashes stay recorded.
- `data/opportunities_teacher_50k.jsonl`: frozen context labels used by the simulator; no conversation text or trained models are included.
- `data/ipinyou_1458.json`: published aggregate parameters used for calibrated synthetic values, not raw bidding logs.
- `scripts/`: the paper's offline statistical generators and their artifact-level checks.

The paper reports 1,800 primary raw rows across its two demand constructions, 1,050 pressure/visible-balance rows, and 210 bursty/heterogeneous rows.

## Withheld source snapshots

Each manifest names a `source_snapshot.tar.gz` holding the simulator source that produced it. That source is not public, so the archives are not included. Their SHA-256 values remain in the manifests as a record of the exact code state:

- primary and iPinYou runs: `dabe08d17b18aa6ba8816b99b4ed164dc866c3de12b9eee3439e55d5da72a2a4`
- pressure/visible-balance and bursty/heterogeneous runs: `207a49793ebeab890dd6821deebde1138d569d8c0f55c32c7961f95597a1cebc`

The offline checks below do not read the archives.

## Offline checks

```sh
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
python scripts/generate_track_b_paper_numbers.py
python scripts/generate_track_b_revision_numbers.py
python scripts/test_track_b_analysis.py
```

The generators validate their input invariants before writing JSON summaries under `results/`. Replaying the simulations requires the withheld simulator source; it is separate from replaying these offline summaries.

## Data provenance and terms

The context labels derive from WildChat-1M (Zhao et al., ICLR 2024), via the single `gpt-oss:120b` teacher described in the paper. They inherit **CC BY-NC 4.0** and require attribution to WildChat and this derivative. The labels are not human ground truth. No original conversation text is redistributed here. See the original WildChat paper and dataset documentation for collection and consent details.

Code in this repository is MIT (see `LICENSE`); WildChat-derived data are excluded from that code license and remain CC BY-NC 4.0. The paper's arXiv license does not replace the data terms.

## Citation

```bibtex
@misc{sarkar2026ondevice,
  title = {When Privacy Moves ML-Mediated Decisions On Device: Information and Incentive Misalignment in Auctions},
  author = {Dipankar Sarkar},
  year = {2026},
  eprint = {2609.33312},
  archivePrefix = {arXiv},
  url = {https://arxiv.org/abs/2609.33312}
}
```
