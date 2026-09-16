# SUTURE: Grafts Add Up

Reproducibility repository for **“Grafts Add Up: First-Order Scoring of Layer
Compositions,”** submitted to ICLR 2027. The package contains the manuscript,
the theory and scoring implementation, frozen experiment contracts, curated
JSON/JSONL evidence, and offline checks needed to audit the reported results.

The central result is a first-order decomposition of a layer graft: per-layer
donor injections are evaluated on the host trajectory and propagated by host
Jacobians. Contracting those terms with utility and risk readouts gives scores
for all contiguous layer windows without building a merged model for each
candidate. The controlled-stack experiments provide exhaustive ground truth.

The language-model evidence is deliberately bounded. The repository includes
the canonical Spanish E1 stop and follow-up diagnostics, contract-v2 readiness
and plumbing stops, and a completed B3 scale check on public Qwen3-8B
specialists for French and Chinese. The B3 result compares three declared
candidates with teacher-forced answer log-probability; it is not a full
held-out sweep or a free-generation ranking claim. The powered Spanish
free-generation ranking remains negative.

## Repository layout

```text
.
├── paper/                 # Anonymous ICLR manuscript and compile inputs
├── src/suture/            # Installable experiment and scoring package
├── configs/               # Frozen study contracts and model/data pins
├── verify/                # Theory, figure-data, power, and negative controls
├── tests/                 # Offline regression and fail-closed guard tests
├── scripts/               # Reviewer-facing command-line entry points
├── results/               # Curated JSON/JSONL evidence; no model weights
├── docs/                  # File rationale and project explanation
├── manifests/             # Machine-readable per-file release inventory
├── requirements.txt       # CPU dependencies for theory and offline checks
└── requirements-gpu.txt  # Pinned GPU dependencies for optional reruns
```

The detailed file-selection policy is in
[`docs/FILES_RATIONALE.md`](docs/FILES_RATIONALE.md). The generated
[`manifests/submission_manifest.json`](manifests/submission_manifest.json)
records a rationale and SHA-256 digest for every included release file.

## Setup

Use Python 3.11 or newer. The default setup is sufficient for the theory
harness, figure regeneration, contract checks, and frozen-artifact inspection.

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
# macOS/Linux
# source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

The optional Transformer and GPU experiments use the pinned environment in
`requirements-gpu.txt`. Install PyTorch from the CUDA 12.4 index when needed:

```bash
python -m pip install --extra-index-url https://download.pytorch.org/whl/cu124 \
  -r requirements-gpu.txt
```

The GPU lanes also require local Hugging Face snapshots and
`models/lid.176.ftz`. Model weights are intentionally not included in this
repository. The contracts pin the exact public model identifiers and revisions
that a rerun must use.

## Offline reproduction

Run commands from the repository root.

### 1. Regression and contract checks

```bash
python -m unittest discover -s tests -v
python -m suture.b3_lighton validate-contract
python -m suture.suture_metrics --smoke
```

The Transformer adapter tests require the GPU requirements (they run a tiny
local GPT-2/Qwen-shaped fixture on CPU; no model download is performed).

### 2. Theory and statistics

```bash
python verify/verify_theory.py
python verify/power_analysis.py
python verify/power_analysis.py --smoke
python verify/independent_check.py
python verify/corrections_check.py
python verify/toy_transfer.py
```

The paper's controlled-stack figures can be regenerated deterministically:

```bash
python verify/make_figdata.py
```

This writes the PGFPlots inputs under `paper/figs/`. It does not write the
legacy root-level figure directory.

The combined CPU entry point is:

```bash
python scripts/run_theory_checks.py
```

Use `--skip-tests` if the optional Transformer test dependencies are not
installed.

### 3. Compile the manuscript

The anonymous paper compile set is self-contained:

```bash
cd paper
./build.sh
```

If `build.sh` is unavailable on Windows, run four passes manually:

```powershell
pdflatex -interaction=nonstopmode main.tex
bibtex main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
```

The paper requires `pdflatex`, `bibtex`, and `pdfinfo`. Compilation
intermediates and the generated PDF are ignored by Git.

## Frozen evidence and optional reruns

The public result tree is intentionally curated:

- `results/b3/lighton_qwen3_8b/` contains the complete FR/ZH B3 data,
  preflight, selection, measurement, manifests, and post-hoc statistics.
- `results/tier_a/` contains the canonical Spanish E1 evidence, the
  language-panel probe data, readiness and regime diagnostics, final plumbing
  pilots, and release attestation.
- `results/v2/qwen3_1_7b/` contains Qwen3-1.7B preflight/readiness/data and
  the stopped Spanish plumbing panel. It contains no Qwen3 E1 result.
- `results/reviewer_followup/` contains controlled multi-donor diagnostics,
  cost diagnostics, score-variant comparisons, and the powered Spanish
  ranking.

These artifacts reproduce the reported analyses without new API calls or
retraining. The code also supports optional reruns when the exact public
checkpoints and required hardware are available.

### Tier-A contract-v2 panel

```bash
python scripts/run_tier_a.py
```

This is a CUDA-intensive pipeline. It is fail-closed with respect to the
Qwen3-1.7B revision, language-identification model, data splits, and frozen
v1 result root. A complete rerun requires the local data and model snapshots
described in [`configs/experimental_contract_v2.json`](configs/experimental_contract_v2.json).

### B3 LightOn/Qwen3-8B scale check

Validate the contract before downloading or loading models:

```bash
python scripts/run_b3.py validate-contract
```

The exact model revisions and measurement protocol are recorded in
[`configs/b3_lighton_qwen3_8b_v1.json`](configs/b3_lighton_qwen3_8b_v1.json).
Scoring and held-out measurement require an A100-class GPU and the public
LightOn snapshots:

```bash
python scripts/run_b3.py score --language fr \
  --host-path /path/to/Qwen3-8B-FR \
  --donor-path /path/to/Qwen3-8B-EN
python scripts/run_b3.py measure --language fr \
  --host-path /path/to/Qwen3-8B-FR \
  --donor-path /path/to/Qwen3-8B-EN \
  --measurement-manifest results/b3/lighton_qwen3_8b/fr_host__en_donor/data/MGSM_rev2_test.jsonl
```

Use the corresponding `zh_host__en_donor` paths for Chinese. Do not overwrite
the frozen result roots; the runners refuse writes to canonical v1 evidence.

### Reviewer-response diagnostics

The optional local diagnostics are exposed as package modules:

```bash
python -m suture.reviewer_followup
python -m suture.reviewer_cost_diagnostics --device cuda
python -m suture.reviewer_llm_comparison --device cuda --prompt-limit 8
python -m suture.powered_spanish_ranking
```

The Qwen2.5-Coder comparison accepts `--model-path` or the portable
`SUTURE_QWEN25_MODEL_PATH` environment variable. No user-specific filesystem
path is embedded in the released code.

## Release and traceability

Regenerate the file-level inventory after changing the release tree:

```bash
python scripts/generate_submission_manifest.py
```

To regenerate the clean-room paper bundle and release attestation:

```bash
python scripts/write_release_bundle.py
python -m suture.tier_a_release_attestation
```

Run manifests retain portable commands, model pins, and input/output hashes as
provenance records without recording local workstation paths.

## Data provenance and licenses

The code is released under the MIT license. Benchmark instances, model
checkpoints, language-identification assets, and translated data retain their
upstream licenses and terms. The relevant dataset/model revisions are recorded
in the frozen contracts under `configs/` and in the result manifests.

This repository does not include model weights or API credentials. Copy
`.env.example` only when running an explicitly configured auxiliary collection
workflow; the canonical local pipeline does not require external API calls.

## Paper and project documentation

- [`paper/main.tex`](paper/main.tex) — anonymous ICLR manuscript.
- [`docs/PROJECT_EXPLAINER.md`](docs/PROJECT_EXPLAINER.md) — method and
  experiment overview.
- [`docs/FILES_RATIONALE.md`](docs/FILES_RATIONALE.md) — public-file selection
  and traceability policy.
