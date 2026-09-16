# Reproducibility guide

## Reproduction levels

The repository separates three levels of reproduction.

| Level | Requirements | What it verifies |
|---|---|---|
| Offline audit | Python 3.11+, `requirements.txt` | Theory checks, power analysis, figure data, contract validation, and frozen JSON/JSONL semantics |
| Local Transformer plumbing | GPU requirements, CPU or CUDA device, local tiny fixtures | Adapter routing, cache behavior, data construction, readiness helpers, and fail-closed guards |
| Scientific rerun | Pinned public checkpoints, `requirements-gpu.txt`, CUDA, storage, and substantial runtime | Tier-A and B3 data/model stages under their frozen contracts |

The paper's reported JSON/JSONL evidence is already included, so reviewers do
not need to rerun API collection or retrain adapters to inspect the reported
numbers.

## Offline environment

```bash
python -m venv .venv
# activate the environment
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
python -m unittest discover -s tests -v
```

The controlled-stack verification is CPU-oriented:

```bash
python verify/verify_theory.py
python verify/power_analysis.py
python verify/independent_check.py
python verify/corrections_check.py
python verify/make_figdata.py
```

`verify/make_figdata.py` writes only to `paper/figs/`. It is deterministic
under the pinned NumPy implementation and does not require model weights.

## Transformer environment

Install the pinned GPU requirements when running the adapter or model lanes:

```bash
python -m pip install --extra-index-url https://download.pytorch.org/whl/cu124 \
  -r requirements-gpu.txt
python -m pip install -e .
```

The required model snapshots are not distributed with this repository. The
contracts record the exact identifiers and revisions, and the runners use
`local_files_only` where the protocol requires it. The Qwen3-1.7B lane also
requires the FastText language-identification file
`models/lid.176.ftz`.

## B3 hardware

The completed B3 scale check used an NVIDIA A100-SXM4-80GB with BF16
Qwen3-8B specialists, eager attention, and CUDA 12.4. Local machines should
run the contract validation and offline B3 tests without loading the 8B pair.
The full procedure is in [`b3_runbook.md`](b3_runbook.md).

## Artifact policy

- Canonical result roots are never overwritten by exploratory runs.
- MGSM test records may be retained for integrity hashing but are not silently
  scored by the Tier-A scripts.
- Model weights, tokenizers, caches, and local environments are excluded from
  the repository.
- `archive/` preserves excluded working material only in the original checkout
  and is ignored in a clean clone.
- Existing manifests retain historical source paths and hashes from the
  experiment freeze. These records are evidence of how the artifact was
  produced, not live import paths.

## Portable optional diagnostics

The Qwen2.5-Coder reviewer diagnostics accept:

```bash
python -m suture.reviewer_llm_comparison --model-path /path/to/snapshot
python -m suture.reviewer_cost_diagnostics --qwen25-model-path /path/to/snapshot
```

Alternatively set `SUTURE_QWEN25_MODEL_PATH` or configure `HF_HOME`. No
machine-specific home directory is embedded in the code.
