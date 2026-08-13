# Tier-A gate status

This report records only artifacts produced by the frozen local run. It is not
a final language-model result and does not authorize Tier B or E2–E10.

## Frozen contract

- Requested checkpoint: `Qwen/Qwen3-1.7B`.
- Measured local compatibility fallback:
  `Qwen/Qwen2.5-Coder-1.5B-Instruct`,
  revision `2e1fd397ee46e1388853d2af2c993145b0f1098a`.
- Device/dtype: RTX 3060 Ti, CUDA, `torch.bfloat16`.
- Selection invariant: score computation and graft selection materialized zero
  merged models.
- Environment: `env.lock`; the contract and model/data decisions are in
  `paper_iclr/experimental_contract.json`.

## Tier-A data and experts

All three language manifests are complete and pairwise SHA-256 disjoint:

- `P_util`: 200 records.
- `P_risk`: 200 records.
- `C`: 500 records.
- `MGSM_dev`: 8 records.
- `MGSM_test`: 250 records, untouched by the gate.

The host and donor are matched LoRA experts from the same base revision for
Spanish, Chinese, and Swahili, using seed 0 and the same optimizer, sequence
length, batch convention, and 16-step recipe. This is an initial gate seed; the
contract's three-seed training protocol has not been completed. Swahili uses
the recorded `gsm8k_heldout_calibration_and_host_supplement` fallback because
the cached translation pool was too small; its hashes and disjointness check
pass.

## Plumbing gates

Artifacts: `results/tier_a/{es,zh,sw}/pilots/`.

- Spanish P1 fake donor: Spearman `0.9030303030`, Pearson `0.9412745688`;
  P2 real donor: Spearman `0.9636363636`, Pearson `0.9843641376`, relative
  L2 error `0.1154113001`, mean injection norm sum `71.5716332451`, and
  Jacobian-gap proxy `141.0`.
- Chinese P1 fake donor: Spearman `0.9878787879`, Pearson `0.8885619515`;
  P2 real donor: Spearman `1.0000000000`, Pearson `0.9823635390`, relative
  L2 error `0.0768357419`, mean injection norm sum `63.8613998264`, and
  Jacobian-gap proxy `56.75`.
- Swahili P1 fake donor: Spearman `0.9878787879`, Pearson `0.9453945006`;
  P2 real donor: Spearman `1.0000000000`, Pearson `0.9957249231`, relative
  L2 error `0.0450064928`, mean injection norm sum `84.8475448743`, and
  Jacobian-gap proxy `238.0`.
- P3 is shared across the language runs: the standalone
  `tiny_qwen2_float64` fixture gives directional derivative
  `0.0007379645292`, Richardson-extrapolated finite difference
  `0.0007379575586`, and relative error `9.4457136e-6`, below the `1e-4`
  target. It is a numerical plumbing fixture, not a language-model result;
  bfloat16 finite differences on the deployed checkpoint were not used for
  the gate.

## E1 contiguous-window sweeps

Each language has 406 exhaustive contiguous candidates over 28 layers and
407 merged models total (host baseline plus one per window); score-based
selection built zero merged models. The measured objective is the
teacher-forced target-answer log-probability change on the eight-item
`MGSM_dev` slice. Correlations and regret use the post-selection
`MGSM_dev`-aligned prediction artifact.

### Spanish

Artifacts: `results/tier_a/es/e1/`.

- Spearman `0.9910735635`; Pearson `0.9922678793`.
- SUTURE-selected window: layers `9–27`; sweep-best window: `0–27`.
- Top-5 overlap: `3`.
- Utility regret: `0.1405582428`, or `23.58666603%` of achievable positive
  utility gain.
- Measured score time: `281.9973576` seconds; exhaustive sweep:
  `231.5867070` seconds.
- Gate: `PASS_E1`.

### Chinese

Artifacts: `results/tier_a/zh/e1/`.

- Spearman `0.9853242847`; Pearson `0.9911348888`.
- SUTURE-selected window: layers `1–25`; sweep-best window: `5–22`.
- Top-5 overlap: `0`.
- Utility regret: `0.0877723694`, or `14.74825996%` of achievable positive
  utility gain.
- Measured score time: `267.9027982` seconds; exhaustive sweep:
  `224.1846402` seconds.
- Gate: `PASS_E1`.

### Swahili

Artifacts: `results/tier_a/sw/e1/`.

- Spearman `0.9897134932`; Pearson `0.9890650784`.
- SUTURE-selected window: layers `1–2`; sweep-best window: `1–27`.
- Top-5 overlap: `3`.
- Utility regret: `0.5471143723`, or `88.18580648%` of achievable positive
  utility gain.
- Measured score time: `297.8177790` seconds; exhaustive sweep:
  `234.4435161` seconds.
- Gate: `STOP_E2_REGIME_INVESTIGATION`, because the regret fraction exceeds
  the predeclared `< 0.5` guard despite high rank correlation.

These are positive or negative first-order ranking artifacts, not final
language-model claims. Host and selected-window next-token accuracy were
`0.0` for every language on the eight-item diagnostic; accuracy regret,
free-generation exact match, drift, retained general capability, and
`MGSM_test` results were not measured.

## Panel status

The three-language Tier-A panel is now complete for the plumbing and E1
artifacts. The contract's additional expert seeds, retention, drift,
free-generation, and untouched-test measurements remain open.

## Decision

All three plumbing panels pass, and Spanish and Chinese pass the predeclared
E1 ranking/regret gate. Swahili passes the ranking-correlation portion but
fails the regret guard. The panel decision is therefore
`STOP_E2_REGIME_INVESTIGATION`: investigate the regime mismatch before any
broader E2–E10 or Tier-B work. No test-set, free-generation, drift-certificate,
retention, or final-paper claim is authorized by this report.
