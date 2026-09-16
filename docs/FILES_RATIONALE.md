# Public file-selection rationale

This repository is a release-oriented slice of a larger working directory. The
selection rule is conservative: a file is included only when it is needed to
compile the manuscript, rerun a released analysis, regenerate a reported
figure or table, validate an integrity invariant, or audit the exact frozen
evidence.

The authoritative file-level record is
[`manifests/submission_manifest.json`](../manifests/submission_manifest.json).
It contains one entry for every included public file, including its byte count,
SHA-256 digest, and rationale. The manifest deliberately leaves its own digest
null because hashing a file that contains its own hash would be circular.
Regenerate it with:

```bash
python scripts/generate_submission_manifest.py
```

## Included root files

| Path | Rationale |
|---|---|
| `README.md` | Reviewer-facing project overview, setup, reproduction commands, resource requirements, and claim boundaries. |
| `LICENSE` | License terms for the released software and documentation. |
| `CITATION.cff` | Machine-readable citation metadata. |
| `pyproject.toml` | Makes `src/suture/` installable with `pip install -e .`. |
| `requirements.txt` | Minimal CPU dependencies for theory, figures, and offline checks. |
| `requirements-gpu.txt` | Exact dependencies used by the optional Transformer/GPU lanes. |
| `.env.example` | Documents optional auxiliary API variables without containing credentials. |
| `.gitignore` | Prevents local checkpoints, secrets, caches, and generated outputs from entering the release. |

## Manuscript

`paper/` is the anonymous compile set:

- `main.tex` is the manuscript source and contains the paper's explicit
  references to the verification scripts and frozen result roots.
- `suture.bib` is the bibliography used by the manuscript.
- `iclr2027_conference.sty` and `iclr2027_conference.bst` are the vendored ICLR
  style and bibliography files needed for a clean-room build.
- `build.sh` performs the four-pass PDF build and reports undefined citations,
  overfull boxes, and page count.
- `figs/*.tex` and `figs/*.dat` are the TikZ/PGFPlots figure sources and
  generated numerical inputs consumed by `main.tex`.

LaTeX auxiliary files, logs, PDFs, and build directories are generated outputs,
not source inputs, and are excluded.

## Source package

Every Python module under `src/suture/` is included because it belongs to one
of the following release surfaces:

- `paths.py` centralizes repository-root, configuration, paper, model, and
  result paths.
- `suture_metrics.py`, `suture_torch.py`, and `data_manifest.py` implement the
  numerical scoring, Transformer adapter boundary, and data-integrity helpers.
- `tier_a_*.py` implements the frozen Tier-A data, training, readiness, gate,
  statistics, inventory, attestation, and contract-v2 orchestration stages.
- `b3_data.py` and `b3_lighton.py` implement B3 manifest construction,
  contract validation, selection, and held-out measurement.
- `reviewer_followup.py`, `reviewer_cost_diagnostics.py`,
  `reviewer_llm_comparison.py`, and `powered_spanish_ranking.py` implement the
  separately frozen reviewer-response diagnostics.
- `run_manifest.py` records code, environment, input, and output hashes.
- `tier_a_config.py` is the single source of truth for the Qwen3-1.7B pin and
  fail-closed result-root rules.
- `tier_a_release_attestation.py` emits the immutable evidence attestation.
- `__init__.py` marks the installable package.

The former private-looking release-builder name is not part of the package.
Its public entry point is `scripts/write_release_bundle.py`.

## Configuration contracts

Every JSON file under `configs/` is a frozen protocol or pin:

- `experimental_contract_v1.json` records the original Tier-A study contract.
- `experimental_contract_v2.json` records the Qwen3-1.7B contract-v2 panel.
- `b3_lighton_qwen3_8b_v1.json` records the B3 model, data, architecture, and
  selection/measurement rules.
- `powered_spanish_ranking_v1.json` records the isolated powered Spanish
  ranking.
- `reviewer_followup_contract_v1.json` records the reviewer-response plumbing
  contract and its source contracts.
- `reviewer_llm_comparison_v1.json` records the fixed-window score-variant
  comparison.

Contracts are shipped separately from executable code so that a reviewer can
inspect the scientific pins without importing Python.

## Verification and tests

The non-test files under `verify/` are included because they are named by the
manuscript or regenerate paper evidence:

- `verify_theory.py` runs the controlled-stack checks.
- `make_figdata.py` regenerates the manuscript PGFPlots data.
- `power_analysis.py` reproduces the power table.
- `independent_check.py` and `corrections_check.py` independently rederive
  theorem constants and coverage statements.
- `toy_transfer.py` is the negative transfer testbed used for the limitation
  analysis.

The files under `tests/` are the corresponding offline regression suite:
contract guards, data manifests, B3 data construction, result inventory,
provenance manifests, Tier-A statistics/readiness, Transformer adapter
plumbing, and canonical E1 artifact semantics. They do not require network
access except for the optional Transformer dependency itself; all Transformer
fixtures are instantiated locally.

## Reviewer-facing scripts

The files under `scripts/` are thin, descriptive entry points:

- `run_theory_checks.py` runs the offline test and numerical-check sequence.
- `run_b3.py` exposes the B3 command-line interface.
- `run_tier_a.py` exposes the contract-v2 orchestrator.
- `write_release_bundle.py` creates the clean-room paper zip and verification
  logs.
- `generate_submission_manifest.py` creates this release's machine-readable
  inventory.
- `__init__.py` makes the directory usable as a Python module namespace.

The implementation remains in `src/suture/`; the scripts provide stable
reviewer-facing commands without duplicating experiment logic.

## Frozen result artifacts

Only JSON, JSONL, small text logs, manifests, and audit metadata are retained.
The included result files are grouped by the paper claim they support:

- `results/b3/lighton_qwen3_8b/` is the complete B3 French/Chinese scale-check
  tree: data construction, preflight, score selection, held-out measurement,
  run manifests, and post-hoc statistics.
- `results/tier_a/es/e1_canonical_seed0/` and its post-hoc summary support the
  canonical Spanish E1 stop.
- `results/tier_a/{es,zh,sw}/` root probe data, final pilots, readiness,
  regime diagnostics, and run manifests support the controlled language-panel
  diagnostics.
- `results/tier_a/ARTIFACT_INVENTORY.json`,
  `RELEASE_ATTESTATION.json`, and `tier_a_replication_statistics.json` provide
  machine-readable classification, hashes, and summary statistics.
- `results/tier_a/verification_release/` records the clean-room and
  verification checks.
- `results/v2/qwen3_1_7b/` retains only the preflight, language data,
  readiness, and stopped Spanish plumbing artifacts needed to document the
  contract-v2 boundary.
- `results/reviewer_followup/` retains the controlled multi-donor,
  cost/timing, score-variant, and powered Spanish ranking outputs.

The manifest lists each JSON/JSONL/log file individually. Historical run
manifests may mention paths from the original working directory and are
intentionally preserved as provenance records; changing those strings would
alter the historical audit rather than improve reproducibility.

## Excluded material

The following classes are excluded from the public release:

1. Model weights, LoRA adapter binaries, optimizer/checkpoint files,
   tokenizer copies, and local Hugging Face snapshots. These are large,
   regenerable, license-governed inputs; public model identifiers and commits
   are pinned in `configs/`.
2. Smoke, preliminary, superseded, and abandoned experiment directories.
   Their classifications remain available in the retained artifact inventory,
   while the working copies are under ignored `archive/`.
3. LaTeX intermediates, PDFs, bytecode, caches, and local environments.
4. Internal review, advisor, novelty-search, and submission-gate notes.
5. The workspace environment lock, which included unrelated dependencies and
   local VCS entries rather than the paper's minimal environment.
6. Redundant vendor/style archives; the exact style files used by the paper are
   already present under `paper/`.

The excluded working material is moved under `archive/` for local preservation
and ignored by Git. A clean clone therefore contains only the auditable
reproducibility package.
