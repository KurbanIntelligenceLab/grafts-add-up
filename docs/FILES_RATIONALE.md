# Included files and selection rationale

This repository contains the standalone reproducibility package for the paper.
Every included path is required to compile the manuscript, reproduce a
reported analysis, regenerate a figure or table, verify a mathematical
property, or inspect the retained evidence.

The machine-readable inventory at
[`manifests/submission_manifest.json`](../manifests/submission_manifest.json)
records every included file with its byte count, SHA-256 digest, and rationale.
Regenerate it with:

```bash
python scripts/generate_submission_manifest.py
```

## Included directories

`paper/`
Contains the anonymous manuscript, bibliography, ICLR style files, build
script, and figure sources consumed by `main.tex`.

`src/suture/`
Provides the installable scoring package: repository paths, data manifests,
Transformer routing, first-order metrics, experiment contracts, provenance,
Tier-A stages, B3 evaluation, and follow-up analyses.

`configs/`
Contains the frozen protocol contracts and model/data pins used by the
corresponding experiment lanes.

`verify/`
Contains the controlled-stack theory harness, figure-data generator, power
analysis, independent checks, corrections, and toy transfer testbed.

`tests/`
Contains offline regression tests for data integrity, contracts, provenance,
statistics, canonical artifacts, and Transformer adapter behavior.

`scripts/`
Contains stable command-line entry points for theory checks, Tier-A, B3,
release attestation, and manifest generation.

`results/`
Contains the JSON, JSONL, manifests, statistics, and attestation needed to
inspect the paper's retained experimental evidence. Author-only raw
language-fidelity probes, per-record measurements, and run manifests are
excluded from the public tree.

`docs/`
Contains this selection rationale and a plain-language project explainer.

`manifests/`
Contains the machine-readable file inventory for the repository release.

## Included root files

`README.md` provides reviewer-facing setup, repository structure, reproduction
commands, resource requirements, and claim boundaries.

`pyproject.toml`, `requirements.txt`, and `requirements-gpu.txt` define the
standalone Python package and its CPU/GPU environments.

`.env.example` documents optional environment variables without containing
credentials. `.gitignore` protects local checkpoints, secrets, caches, and
generated files.

`LICENSE` provides the software license, and `CITATION.cff` provides
machine-readable citation metadata.

## Artifact decisions

`results/b3/lighton_qwen3_8b/`
Contains the French and Chinese B3 data construction, preflight, selection,
held-out answer measurement, aggregate language-fidelity summary, and post-hoc
statistics. Raw language-fidelity records and manifests remain author-only.

`results/tier_a/`
Contains the canonical Spanish E1 artifacts, language-panel probe data, final
plumbing pilots, readiness and regime diagnostics, and release attestation.

`results/v2/qwen3_1_7b/`
Contains the Qwen3-1.7B preflight, language data, readiness records, and
contract-v2 plumbing artifacts.

`results/reviewer_followup/`
Contains controlled multi-donor diagnostics, cost measurements, score-variant
comparisons, and the powered Spanish ranking.

The retained result files are sufficient to inspect the released evidence
without rerunning model collection. Public run manifests preserve a portable
command record, model pin, input/output hashes, and selection invariants
without recording local workspace paths. The language-model aggregate results
whose per-record inputs remain private cannot be recomputed from this checkout
alone.
