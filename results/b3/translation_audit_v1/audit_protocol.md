# Post-submission audit of frozen B3 selection translations

This audit covers all 2,000 existing selection probes: 500 math questions and
500 language passages in each of French and Chinese. It does not cover every
translation in the project, change the probes, or rerun selection/evaluation.

`paired_texts.jsonl` retains English sources recovered from the pinned dataset
revisions and the existing translations after removing only the exact shared
chat wrappers and final task instructions. `source_recovery.json` records
source IDs, indices, hashes, dataset pins and unchanged original manifests.

## Semantic similarity

`embedding_freeze.json` pins LaBSE at
`836121a0533e5664b21c7aacc5d22951f2b8b25b` and records the script/input hashes
before study-pair inference. `semantic_summary.json` records model weight
hashes, software, hardware, runtime and all group aggregates.
`embedding_inputs.jsonl`, `embedding_chunk_tokens.jsonl` and `embeddings.npz`
retain the exact inputs and vectors; `semantic_scores.jsonl` retains every
pair and control score. No acceptance threshold was selected.

The official LaBSE pipeline uses CLS pooling, dense/tanh projection and L2
normalization. We retain standard 256-token prefix scores and a full-text
chunk-pooling diagnostic. Only one unique English source exceeds the prefix
limit, appearing once in each language. No target exceeds it. The manuscript
table uses chunk-pooled scores; their means round identically to prefix means.

The wrong-target control shifts targets by 137 positions within each 500-pair
group. The digit perturbation adds one to the first regex-matched English
digit sequence. The original freeze describes this as a standalone numeric
mention, but independent inspection found 5 of the 110 language-source
matches per language after a hyphen, including identifiers. None of the 481
math-source matches has that issue. Treat this as a sensitivity diagnostic,
not a controlled quantity-error benchmark or calibrated translation metric.

`attempt_001/` preserves an initial tokenizer API setup failure before any
study-pair embedding. The successful run has its own freeze and script hash.
`independent_audit.json` independently checks manifest extraction, pairing,
score aggregates and saved-vector cosine calculations. Its PASS concerns
artifact consistency, not translation quality.

## Numeral consistency

`number_freeze.json` records input, checker and runner hashes before the
number scan. `number_records.jsonl` retains strict digit comparisons and a
broader comparison including supported English/French/Chinese number words.
`number_summary.json` keeps all 500 pairs in each group, the subset with an
English numeric signal, no-signal pairs, and review-needed counts explicit.

Comparisons preserve multiplicity and normalize supported numeric forms.
Recognized dates, times, versions, ranges, unsupported number words and ambiguous forms
remain review-needed. English a/an and French un/une are not automatically
counted as one. A mismatch is a mechanical consistency flag, not a confirmed
translation error. Matching values do not verify units, currencies, ordering,
relations, omission of other content or mathematical equivalence. No labels
from an independent human or native-speaker audit are available.
The grammar also has false flags: English "twice" versus French "deux fois",
Chinese month/weekday numerals versus English names, and Chinese idioms with
numeral characters. These limitations are retained rather than adapting the
checker after seeing comparative audit outputs. A match only means equal
parsed value multisets, including pairs with no recognized numbers.

## Length and illustrative examples

`tokenizer_length_records.jsonl` and `tokenizer_length_summary.json` use the
original pinned translation tokenizer. Re-encoded text lengths are not
original generated token counts or finish reasons. The original 96-token
output cap therefore cannot be used to infer a termination rate from these
records. `illustrative_examples.json` retains a high-similarity English target
in the Chinese group and French examples ending mid-phrase or mid-word. These
are AI-assisted inspections, not a random sample or human error-rate estimate.

## Reproduction

The scripts refuse to overwrite retained outputs. Use a separate checkout or
an empty output directory for a rerun; preserve this directory as evidence.

From a clean checkout without the audit outputs, recover the pairs with
`python scripts/prepare_b3_translation_audit.py`. This requires access to the
pinned Hugging Face dataset snapshots through the existing B3 source loaders.
Run `python scripts/run_b3_translation_semantics.py --input PATH --output NEW_DIR`
for LaBSE inference. Run `python scripts/run_b3_translation_numbers.py --root DIR`
with `paired_texts.jsonl` in that directory for the CPU numeral scan.
`python scripts/check_b3_translation_lengths.py` runs the pinned-tokenizer
length check. The numeric checker tests are
`python -m unittest discover -s tests -p test_translation_number_checks.py`.

All protocols are post-submission and not externally preregistered. No model
weights are redistributed here. The frozen original task results remain
unchanged; these diagnostics cannot establish expert-training-data separation
or correct the historical selection retrospectively.
