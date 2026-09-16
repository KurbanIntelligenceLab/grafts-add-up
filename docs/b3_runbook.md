# B3 LightOn/Qwen3-8B runbook

This is a separate, frozen scale-check protocol. It does not reopen or write
the Tier-A, Qwen3-1.7B, or reviewer-follow-up result roots. No B3 result is
valid unless the exact preflight, selection, and held-out measurement manifests
are present.

## Frozen inputs

The contract is `configs/b3_lighton_qwen3_8b_v1.json`. It pins:

- donor: `lightonai/Qwen3-8B-EN`
  (`3e1efbcad91112cd0ac4ebf266a938596b47a1b1`);
- French host: `lightonai/Qwen3-8B-FR`
  (`a53d601e18842b62bd43b3718cf3900ba2d7bbf6`);
- Chinese host: `lightonai/Qwen3-8B-ZH`
  (`b100a627fcbe1af4391de73cb962e7564aff877a`);
- held-out measurement: `lightonai/mgsm-rev2`
  (`1463c6dc7991a8751b8e28e76f6c561b9201eb55`), test split, 250 records per
  language;
- BF16 `Qwen3ForCausalLM`, 36 layers, 4096 hidden units, and eager attention;
- all 666 contiguous zero-based intervals, scored with `use_cache=False`.

The published FR `[13, 22]` and ZH `[13, 20]` windows are reference points,
not substitute measurements. Selection must use independently routed host and
donor checkpoints.

## Completed v1 run

The v1 protocol was completed on one NVIDIA A100-SXM4-80GB with 79.25 GiB
visible memory, CUDA 12.4, BF16 weights, Python 3.12.14,
`torch==2.6.0+cu124`, and `transformers==5.14.1`. French selected `[13,15]`;
Chinese selected `[20,31]`. Both selection stages scored all 666 intervals and
reported zero grafted-model builds.

The separate held-out stage evaluated 250 MGSM-Rev2 test records per language
for exactly three candidates. Mean answer log-probability per target token was:

- French: host `-8.808`, selected `-8.410`, published `[13,22]` `-8.055`;
- Chinese: host `-9.516`, selected `-8.190`, published `[13,20]` `-8.059`.

The selected-minus-host gains are `+0.398` (French; paired bootstrap 95% CI
`[+0.358,+0.437]`) and `+1.326` (Chinese; `[+1.210,+1.446]`). This is a
positive teacher-forced candidate comparison, not a full 666-window held-out
ground-truth sweep or a free-generation exact-match result. The instance was
destroyed after the artifacts were copied back; the result roots and the
post-hoc summary are under `results/b3/lighton_qwen3_8b/`.

## No-cost local checks

From the repository root:

```powershell
python -m suture.b3_lighton validate-contract
python -m unittest verify.test_b3_contract -v
python -m unittest verify.test_suture_torch verify.test_data_manifest verify.test_run_manifest verify.test_v2_guards verify.test_v2_setup verify.test_b3_contract verify.test_b3_data -v
```

The local machine should not attempt to load the 8B pair. The B3 loader refuses
CPU fallback and refuses to continue without CUDA.

## Prepare a Vast instance

Use a CUDA image with Python 3.12 and enough disk for the donor plus one host
(roughly 33 GB of BF16 weights before caches and artifacts). The completed run
used a 100 GB disk and an A100-SXM4-80GB at `$1.3096/hour` including disk.
Those are historical figures, not a current offer; re-check the offer,
storage size, and price immediately before creation.

Install the pinned runtime on the instance:

```bash
python3.12 -m venv /opt/b3-venv
source /opt/b3-venv/bin/activate
python -m pip install --upgrade pip
python -m pip install --extra-index-url https://download.pytorch.org/whl/cu124 \
  "torch==2.6.0+cu124"
python -m pip install \
  "transformers==5.14.1" \
  "accelerate==1.14.0" \
  "huggingface_hub==1.16.1" \
  "datasets==5.0.1" \
  "numpy==2.4.2" \
  "safetensors==0.8.0"
```

Copy the working-tree files to the instance rather than cloning a commit that
does not contain the uncommitted B3 runner:

```powershell
$remote = "root@<PUBLIC_IP>"
$port = <SSH_PORT>
$key = "$env:USERPROFILE\.ssh\vast_b3_ed25519"
scp -P $port -i $key `
  src\suture\b3_lighton.py `
  src\suture\b3_data.py `
  src\suture\tier_a_config.py `
  src\suture\suture_torch.py `
  src\suture\suture_metrics.py `
  src\suture\data_manifest.py `
  src\suture\paths.py `
  src\suture\__init__.py `
  "${remote}:/workspace/model-composition/src/suture/"
scp -P $port -i $key `
  configs\b3_lighton_qwen3_8b_v1.json `
  "${remote}:/workspace/model-composition/configs/"
```

Copy the frozen data directory only after the local manifest checks pass:

```powershell
ssh -p $port -i $key $remote `
  "mkdir -p /workspace/model-composition/results/b3/lighton_qwen3_8b/fr_host__en_donor"
scp -r -P $port -i $key `
  results\b3\lighton_qwen3_8b\fr_host__en_donor\data `
  "${remote}:/workspace/model-composition/results/b3/lighton_qwen3_8b/fr_host__en_donor/"
```

Use the corresponding `zh_host__en_donor` path for the Chinese pair. Do not
regenerate the manifests on the paid instance.

Run the contract check before downloading weights:

```bash
cd /workspace/model-composition
source /opt/b3-venv/bin/activate
python -m suture.b3_lighton validate-contract
```

Before renting a cloud GPU, build the real selection probes locally from the
pinned translation model:

```powershell
python -m suture.b3_data `
  --language fr `
  --translation-snapshot models/Qwen3-1.7B-70d244cc `
  --device cuda:0
```

Repeat with `--language zh` only after the French data build passes. Copy the
resulting `results/b3/lighton_qwen3_8b/<pair>/data/` directory to the instance;
the builder refuses to replace existing manifests.

Build the held-out MGSM-Rev2 manifests from the same tokenizer. This step does
not load model weights and is required before the paid measurement stage:

```powershell
python -m suture.b3_data `
  --language fr `
  --translation-snapshot models/Qwen3-1.7B-70d244cc `
  --measurement-only
```

Repeat with `--language zh`, then copy `MGSM_rev2_test.jsonl` and its
`measurement_data_summary.json` alongside the selection manifests. The
held-out manifest is disjoint from `P_util` and `P_risk`.

Download only one host pair first. This keeps the first paid run bounded:

```bash
python -m suture.b3_lighton prepare \
  --language fr \
  --download-dir /workspace/models
```

## Required stage order

The probe manifests must be frozen and copied to the instance under the
contract-declared data directory before scoring. They are hashed JSONL files;
the runner rejects malformed, undersized, or overlapping IDs. No synthetic
probe may be used for a reported result.

First validate the snapshots without loading weights:

```bash
python -m suture.b3_lighton preflight \
  --language fr \
  --host-path /workspace/models/Qwen3-8B-FR \
  --donor-path /workspace/models/Qwen3-8B-EN
```

Then perform the CUDA/model gate:

```bash
python -m suture.b3_lighton preflight \
  --language fr \
  --host-path /workspace/models/Qwen3-8B-FR \
  --donor-path /workspace/models/Qwen3-8B-EN \
  --load-models
```

Only after that gate passes may selection run:

```bash
python -m suture.b3_lighton score \
  --language fr \
  --host-path /workspace/models/Qwen3-8B-FR \
  --donor-path /workspace/models/Qwen3-8B-EN \
  --utility-manifest results/b3/lighton_qwen3_8b/fr_host__en_donor/data/P_util.jsonl \
  --risk-manifest results/b3/lighton_qwen3_8b/fr_host__en_donor/data/P_risk.jsonl
```

The selection stage must report zero grafted-model builds. It writes
`selection_scores.json`, `selection.json`, and a B3-specific
`run_manifest.json`. Only after `selection.json` exists, run the separate
held-out measurement stage:

```bash
python -m suture.b3_lighton measure \
  --language fr \
  --host-path /workspace/models/Qwen3-8B-FR \
  --donor-path /workspace/models/Qwen3-8B-EN \
  --measurement-manifest results/b3/lighton_qwen3_8b/fr_host__en_donor/data/MGSM_rev2_test.jsonl
```

Measurement compares exactly three candidates: the untouched host, the
interval selected from the independent probes, and the published LightOn
reference interval. It reports mean teacher-forced answer log-probability per
target token on the 250 held-out records. It is a bounded candidate comparison,
not a claim of a full 666-window held-out sweep. Held-out prompts are checked
against the frozen 256-token limit; they are never silently truncated.

If the model loader, tokenizer equality, snapshot hash, hardware minimum,
manifest integrity, or cache policy fails, the result is `STOP_SETUP`; do not
retry with a different model, precision, or mutable revision. Shut down the
instance after copying the manifests back locally.

## Optional stronger follow-ups

The six released specialists correspond to five meaningful host pairs
(`FR`, `DE`, `ES`, `SW`, and `ZH`) with `EN` as the donor; an `EN`-host/
`EN`-donor pair is not informative. These estimates use the completed
`$1.31/hour` A100 rate and include approximate model-download bandwidth:

- Complete the five-host panel with selection plus the same three-candidate
  held-out measurement: 1.5–2.5 GPU-hours, approximately `$6–$10`.
- Measure all 666 intervals on the held-out set for all five hosts:
  approximately 4–7 GPU-hours, `$12–$22`. This requires a new immutable
  expanded contract and runner path; v1 must not be edited.
- Add bounded free-generation exact match for host/selected/published
  candidates on 250 records per host: pilot 0.5–1.5 hours, then roughly
  3–10 GPU-hours and `$8–$20` for the full panel. A 666-window
  free-generation sweep would require hundreds of thousands of generations
  and is not recommended.

The ranges exclude local manifest preparation time. Rent only after the
expanded contract, generation parser, and local tests pass.
