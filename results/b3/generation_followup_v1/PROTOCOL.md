# B3 French and Chinese free-generation follow-up

This is a separate post-submission evaluation of the already-frozen B3 choices.
It does not modify or replace the original teacher-forced measurements.

Each of French and Chinese uses all 250 saved MGSM-Rev2 test questions from
`lightonai/mgsm-rev2` at revision `1463c6dc7991a8751b8e28e76f6c561b9201eb55`.
The three candidates are the native-language host, the selected layer swap,
and the pinned official published Swap checkpoint. The selected intervals are
FR `[13,15]` and ZH `[20,31]`; published intervals are FR `[13,22]` and ZH
`[13,20]`, all zero-based and inclusive. Every tensor in the official published
checkpoints was compared with the declared host/donor reconstruction, and every
saved selected tensor was checked against its exact expected source.

The follow-up uses the source paper's localized MGSM reasoning instruction and
the pinned host's native Qwen3 chat template with thinking enabled. Sampling is
temperature 1.0, top-p 0.95, top-k 20, min-p 0, with one deterministic per-item
seed schedule derived from 20261108 and shared across candidate triplets.
The operational output cap is 8,192 tokens. This is not a reproduction of the
source model cards' 32,768-token example or their ten-run averaged scores.

Inference uses ordinary materialized checkpoints with vLLM 0.11.0, BF16 without
quantization, on one NVIDIA A100 SXM4 80GB. The container's immutable AMD64 image
digest is `sha256:d8d39b59e909d2378ac4feeb191f7e7b6f1342477dc66b7c47cec89e9985ad8a`.
`environment.lock` and per-candidate `run.json` files record the executed runtime.

Two answer scores are reported side by side. Original integer EM is the frozen
generator's integer-only final-line score. Supplemental normalized numeric EM
uses a post-hoc parser. The first version was specified after host-only format QA,
before comparative output inspection. Subsequent comparative-output QA found
that it left bold Markdown around answer labels in the numeric suffix, rejecting
66 answers that the original integer parser correctly accepted. A separately
versioned v2 removes Markdown from the final line before finding its answer
marker. This bug correction was specified after comparative-output QA. Both
versions and their scored sidecars are retained; the main table uses v2. This
was not an external preregistration or an exact reproduction of the source
authors' answer scorer.

The numeric parser reads only the completed post-think final line, takes the
scalar after its last explicit answer marker (or a standalone scalar), and
compares decimals/fractions exactly. An explicit final marker after prose and
a scalar such as `57.00` can be parsed, without rounding `2,4` to an integer.
Ambiguous grouping, alternatives, units/prose in the numeric suffix, missing
think-close, and truncated outputs are rejected. Unknown/unparseable/truncated
outputs stay in the 250-item denominator and count as incorrect. Original
integer scores, raw generation files and language-ID results are never replaced.
The initial v1 scorer is bound to SHA-256
`cc5a12e59937a23ac74dde28b4286c7f8801ce55deac3d3bbceffe72b4af506b`.
The corrected v2 scorer is bound to SHA-256
`c81e5057f341588ae8d7ea3d94d331d58cbacffb6e4701563c15ff2202248828`.
The freeze declarations and correction history are retained alongside the results.
V2 preserves v1's numeric grammar, including rejection of a currency suffix such
as `64 $`, which the original integer parser accepted. Thus neither supplemental
parser is a simple monotone relaxation of the original parser.
Native answer-marker presence is a separate count, not a claim of full
adherence to the requested response format.

Primary language ID is fastText top-1 on the complete generated reasoning
trace, with at least 20 alphabetic characters. Unknown, incomplete and truncated
traces count as non-target in the all-record target-language rate and are
reported separately. Numeric-only final answers have no useful language ID.
A secondary diagnostic scans non-overlapping 200-character reasoning chunks
with at least 20 alphabetic characters and flags English at probability >=0.5.
This classifier diagnostic is not a human-validated language guarantee.

Paired bootstrap intervals resample the same item IDs across candidates.
They describe item variability conditional on the one fixed seed schedule;
they do not include variation across generation seeds or other benchmarks.
Intervals are pointwise and are not adjusted for multiple contrasts.

The retained evidence includes all 1,500 completion texts and token-ID sequences,
prompt IDs/hashes, parser and LID statuses, truncations, model/data/config/source
hashes, exact tensor-mapping receipts, source download commit/LFS/Git-blob checks,
and the full execution manifest.
The derived audit and score reports retain the checkout path used when they were
frozen because their hashes link the scoring stages. Those paths are historical
provenance fields, not required locations for a rerun; the audit entry points
resolve inputs from the current checkout.

## Reproduction and audit

The public entry point is `scripts/run_b3_generation_followup.py`. Download the
exact checkpoints with `scripts/download_b3_generation_checkpoints.py`, prepare
selected candidates with `scripts/prepare_b3_generation_candidates.py`, and bind
inputs with `scripts/record_b3_generation_inputs.py`. These preparation scripts
default to the documented `/workspace/b3_generation` container workspace.
Run one candidate per process as shown by `run --help`, then run `aggregate`
for each language in that recorded inference workspace, because its run metadata
binds the original absolute source-manifest paths. The independent verifier
recomputes the retained results on the local checkout without those remote paths. The frozen settings are in
`configs/b3_generation_followup_v1.json`.

After generating or retrieving all six candidate outputs, audit them with:

```text
python verify/audit_b3_generation_results.py
```

The supplemental scorer requires that audit's `PASS` receipt and exact hashes
for all 1,500 raw rows, plus the corresponding approved parser digest above. It writes only
new sidecars under `format_robust_v1/` and `format_robust_v2/`; the corrected
entry point is `python scripts/score_b3_generation_followup_v2.py --help`.

Model weights are public source checkpoints and are not redistributed here.
The language classifier is the official [fastText `lid.176.ftz` model](https://fasttext.cc/docs/en/language-identification.html),
distributed under CC-BY-SA-3.0. Place the compressed download at
`models/lid.176.ftz` and verify its SHA-256 against the retained run/audit metadata.
MGSM-Rev2 source data are attributed to LightOn under the source dataset's
CC-BY-SA-4.0 terms. The original MGSM data derive from GSM8K.

Protocol sources:

- [LightOn source paper, including the localized MGSM instructions](https://arxiv.org/html/2605.26735v1)
- [French published Swap model card](https://huggingface.co/lightonai/Qwen3-8B-FR-Swap)
- [Chinese published Swap model card](https://huggingface.co/lightonai/Qwen3-8B-ZH-Swap)
- [MGSM-Rev2 dataset](https://huggingface.co/datasets/lightonai/mgsm-rev2)

## Results

Percentages use all 250 records per candidate. Unknown language traces stay in
the denominator. Integer EM is the original frozen parser; numeric EM is the
post-hoc v2 correction described above. English spans are classifier diagnostics.

| Language | Candidate | Integer EM | Numeric EM v2 | Target ID | English span | Unknown ID | Truncated |
|---|---|---:|---:|---:|---:|---:|---:|
| FR | host_baseline | 75.2 | 89.2 | 96.8 | 0.0 | 8 | 8 |
| FR | selected | 88.8 | 95.2 | 98.8 | 0.0 | 3 | 3 |
| FR | published_reference | 62.4 | 65.6 | 77.6 | 1.6 | 56 | 55 |
| ZH | host_baseline | 84.8 | 94.0 | 98.0 | 0.0 | 5 | 5 |
| ZH | selected | 92.0 | 94.4 | 9.2 | 95.2 | 2 | 2 |
| ZH | published_reference | 68.4 | 70.4 | 92.0 | 0.0 | 20 | 20 |

The Chinese selected window fails the intended language-preservation goal:
225 complete reasoning traces are classified English and 23 Chinese, with two
unknown/truncated traces. English chunks are flagged in 238 of 250 outputs.
All evaluable French traces are classified French; the selected model has 247
complete evaluable traces and zero flagged English chunks. The published French
model has 56 unknown traces, including 55 truncations, so its lower all-record
Target ID does not establish greater language drift among complete traces.

Selected-minus-baseline numeric EM v2 paired 95% intervals (percentage points):

| Language | Contrast | Difference | 95% interval |
|---|---|---:|---|
| FR | selected_minus_host | +6.0 | [+2.0, +10.0] |
| FR | selected_minus_published | +29.6 | [+23.6, +35.6] |
| ZH | selected_minus_host | +0.4 | [-3.2, +4.0] |
| ZH | selected_minus_published | +24.0 | [+18.0, +30.0] |

The French gain over the host is positive under this parser and decoding cap;
the Chinese host comparison is inconclusive. Published-reference comparisons
are cap-sensitive, especially the French reference with 55 truncations versus
three for selected. The study compares three frozen choices, not all 666 windows.
First-order weight attribution selected identical windows, so this panel cannot
distinguish its choices from SUTURE. All results remain conditional on one seed
schedule, MGSM-Rev2, the operational cap, and the declared parsing rules.

The independent raw-output audit passed all 1,500 rows. The original integer
parser and two post-hoc parsers are retained. `scoring_v1_format_qa.json` records
the 67 original-correct/v1-not-correct cases that motivated the v2 correction;
`reasoning_label_counts.json` records the complete label counts.

One A100 80GB was allocated for about 2.39 hours including transfer and preparation;
the six recorded candidate runtimes total about 1.87 hours. The account credit
decrease was approximately USD 5.01, including transfer. The instance was
destroyed after the final local bundle passed all file hashes and 1,500 row counts,
and its absence was verified. See `compute_receipt.json` for scope and timestamps.

The separate corrected-numeric audit passed all 1,500 rows, exact numeric
comparisons, source/sidecar hashes, preserved original scores and language ID,
and all six numeric paired bootstrap intervals. See `scoring_v2_audit.json`.
The focused CPU suites passed 12 generation-protocol tests, 10 v1-scoring tests,
and 14 v2-scoring tests.
