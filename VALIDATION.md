# Validation

Validated on 27 September 2026 in an isolated Python 3.14 environment with SciPy 1.18.1.

- Both statistical generators completed successfully on the released data.
- The revision generator audited all 1,050 pressure and visible-balance rows and the bursty/heterogeneous artifact.
- All five existing artifact-level tests passed (replicate identities, paired controller contrasts, frozen/disjoint calibration, factorial cells, and sensitivity budgets).
- Primary and revision source archive hashes agreed with their recorded manifests at validation time. The archives were later withheld from this public release; their hashes remain in the manifests.
- Credential and private-local-path scans found no matches.
- These are offline artifact checks, not a fresh collection or simulator rerun.
