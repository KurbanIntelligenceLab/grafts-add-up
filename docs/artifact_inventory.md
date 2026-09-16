# Tier-A artifact inventory

This inventory classifies existing result directories. Canonical Spanish
E1 files were not rewritten. MGSM-test records were not scored.

- Written: `2026-08-24T09:16:06.898557+00:00`
- Canonical E1: `results/tier_a/es/e1_canonical_seed0`
- MGSM-test policy: `hashed_for_integrity_never_scored`
- Contract v2 (`results/v2/qwen3_1_7b/`): scientific Qwen3 measurement. es `PASS_READINESS` then `STOP_PLUMBING_P1`; zh/sw `STOP_EXPERT_READINESS` on host LID. No Qwen3 E1.

| Language | Directory | Class | Claim | Manifest |
|---|---|---|---|---|
| es | `adapters` | superseded | none | False |
| es | `e1` | preliminary | not_citable_as_e1 | True |
| es | `e1_cache_smoke` | smoke | none | False |
| es | `e1_cache_smoke_b8` | smoke | none | False |
| es | `e1_canonical_seed0` | canonical | primary_e1_stop | True |
| es | `e1_canonical_smoke` | smoke | none | False |
| es | `e1_canonical_smoke_b1` | smoke | none | False |
| es | `e1_canonical_smoke_batchgen` | smoke | none | False |
| es | `e1_canonical_smoke_final` | smoke | none | False |
| es | `e1_canonical_smoke_manualgen` | smoke | none | False |
| es | `e1_canonical_smoke_probe8` | smoke | none | False |
| es | `e1_previous_full_run` | preliminary | not_citable_as_e1 | True |
| es | `e2` | canonical | regime_diagnostic | False |
| es | `pilots` | superseded | none | True |
| es | `pilots_fake_scale_1e4` | smoke | none | False |
| es | `pilots_final` | canonical | plumbing_p1_p2_p3 | True |
| es | `pilots_fixed` | smoke | none | False |
| es | `pilots_fixed32` | smoke | none | False |
| es | `pilots_float32_fake` | smoke | none | False |
| es | `pilots_revalidated` | smoke | none | False |
| es | `pilots_revalidated_scale2e3` | smoke | none | False |
| es | `pilots_revalidated_scale5e4` | smoke | none | False |
| es | `pilots_single_prompt` | smoke | none | False |
| es | `pilots_single_prompt_debug` | smoke | none | False |
| es | `publication_adapters` | superseded | none | False |
| es | `publication_adapters_fixed` | canonical | publication_experts_seed0 | False |
| es | `publication_adapters_seed0` | superseded | none | False |
| es | `publication_adapters_stable` | superseded | none | False |
| es | `readiness` | canonical | expert_readiness | True |
| sw | `adapters` | superseded | none | False |
| sw | `e1` | preliminary | not_citable_as_e1 | True |
| sw | `e2` | canonical | regime_diagnostic | False |
| sw | `pilots` | superseded | none | True |
| sw | `pilots_final` | canonical | plumbing_p1_p2_p3 | True |
| sw | `publication_adapters_fixed` | canonical | publication_experts_seed0 | False |
| sw | `readiness` | canonical | expert_readiness_failed | True |
| zh | `adapters` | superseded | none | False |
| zh | `e1` | preliminary | not_citable_as_e1 | True |
| zh | `e2` | canonical | regime_diagnostic | False |
| zh | `pilots` | superseded | none | True |
| zh | `pilots_final` | canonical | plumbing_p1_p2_p3 | True |
| zh | `publication_adapters_fixed` | canonical | publication_experts_seed0 | False |
| zh | `readiness` | canonical | expert_readiness | True |

Contract v2 (`results/v2/qwen3_1_7b/`):

| Language | Directory | Class | Claim | Manifest |
|---|---|---|---|---|
| es | `readiness` | canonical | expert_readiness_passed | True |
| es | `pilots` | canonical | plumbing_p1_failed | False |
| zh | `readiness` | canonical | expert_readiness_failed | True |
| sw | `readiness` | canonical | expert_readiness_failed | True |
