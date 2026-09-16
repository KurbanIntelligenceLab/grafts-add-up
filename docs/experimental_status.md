# SUTURE — ICLR 2027 submission bundle

**Grafts Add Up: Scoring Every Layer Composition Without Building One**

**Status: technical red flags closed; human gates remain.** Read
`REDFLAGS.txt` first for the proof-signature and mock-review gates. This README
says what the paper now claims, how to build it, where each claim lives, and
how to run the checks.

Deadlines: abstract 18 September 2026 AOE, paper 25 September 2026 AOE.
Main text capped at 9 pages at submission, strictly enforced.

**Expanded B3 disposition:** the separate five-language follow-up is archived as
`STOP_GENERATION` and is non-reportable. Its BF16 cache-equivalence pilot
failed before a complete panel manifest or post-hoc statistics were produced.
It does not modify or weaken the canonical B3 v1 evidence below.

---

## What the paper claims

**It claims**, and supports with mechanical checks in a controlled stack where
exhaustive search is ground truth:

- The deviation a graft induces decomposes into per-unit injections propagated by
  host Jacobians, with a remainder second order in the fine-tuning scale
  (Theorem 1), extending to *k* donors (Corollary 2) and, with a further
  hypothesis, to continuous coefficients (Corollary 4, appendix only).
- Contracting with the adjoint of a chosen readout gives *L* numbers per readout
  from one host forward–backward sweep plus streamed donor-block evaluations, so
  2L numbers score all 2^L compositions and the candidate count never enters the
  cost of acquiring them.
- A selector is governed by the score *margin* against the remainder, not by
  predictive accuracy (Proposition 3), giving a per-instance test that a named
  competitor cannot beat the selection, at a cost of *m* built models.
- Across 12 stacks: window ranking Spearman 0.95 ± 0.05, selection regret
  0.02 ± 0.03, and 64.8 ± 9.6 of 77 competitors certified from 11 built models
  against the sweep's 78.
- Both factors of the score are necessary. On fixed-width windows with
  heterogeneous expert gaps, dropping the adjoint or the injection raises
  selection regret from 0.03 to 0.26 and 0.25, and per-layer parameter-change
  magnitude reaches 0.29, against 0.33 for random scores.

**Its language-model evidence is deliberately scoped.** The Spanish ranking
run on Qwen2.5-Coder remains a negative for free-generation ranking: the
canonical eight-item design is uninformative (Spearman −0.018 inside a chance
band), and an isolated 64-item greedy MGSM-test ranking on the same scores
reaches Spearman 0.129, still well below the 0.5 gate, with selected exact
match 1/64 against a sweep best of 2/64. A separate frozen B3
cloud run on public Qwen3-8B specialists selected intervals that improved
held-out teacher-forced answer log-probability over the host by +0.398
nats/token in French and +1.326 in Chinese. This is a three-candidate
comparison, not a full 666-window held-out ranking or a free-generation claim.
The Qwen3-1.7B attempt stopped earlier, at plumbing and host-language readiness.
The separate reviewer-response diagnostic repaired the zero-donor invariant but
still stopped at its frozen fake-donor gate; its real-donor score is reported
only as a local applicability diagnostic.

The reviewer-response follow-up also adds controlled four-donor and
signed-knapsack checks, warm CUDA stage timings, a fixed-window comparison
against six additional local score variants, and the powered Spanish 64-item
ranking under `results/reviewer_followup/powered_spanish_ranking/`. These
results do not modify the canonical v1/v2 artifacts, do not authorize a
Qwen3 E1, and do not provide positive language-model ranking evidence.

## B3 cloud scale check

The frozen B3 contract was completed on one NVIDIA A100-SXM4-80GB. For French
and Chinese, it:

- scored all 666 zero-based contiguous intervals using independent utility and
  risk probes, with zero grafted-model builds during selection;
- measured the host baseline, the selected interval, and the published
  reference interval on 250 disjoint MGSM-Rev2 test records per language;
- selected `[13,15]` for French and `[20,31]` for Chinese;
- improved mean teacher-forced answer log-probability per target token over the
  host by `+0.398` (French, paired bootstrap 95% CI `[+0.358,+0.437]`) and
  `+1.326` (Chinese, `[+1.210,+1.446]`).

The complete preflight, selection, measurement, manifest, and post-hoc
statistics records are under `results/b3/lighton_qwen3_8b/`. The v1 contract
remains immutable. The current result does not measure every interval on the
held-out set and does not establish free-generation exact-match ranking.

The earlier planning estimates for broader follow-ups are retained only as
historical context. They are not current results or an authorization to rerun
the experiment.

### Expanded B3 follow-up: why it is not a result

An isolated contract was prepared for FR, DE, ES, SW, and ZH, with all 666
held-out intervals and bounded free generation. The run was stopped at the
German generation pilot: BF16 cached and uncached greedy decoding diverged
under the frozen cache-equivalence gate. A direct Qwen3-8B BF16 check reproduced
the divergence, so bypassing the gate or switching to float32 would have been a
new protocol rather than a repair to v1. No expanded panel manifest,
post-hoc statistics, or paper claim was created. See
`B3_EXPANDED_RUNBOOK.md` for the archived protocol and stop record.

---

## Building

Compile from this directory so `\input{figs/...}` resolves. Overleaf: **pdfLaTeX**.

```
./build.sh
```

or four passes by hand:

```
pdflatex -interaction=nonstopmode main.tex
bibtex main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
```

Delete stale `main.aux`, `main.bbl`, `main.blg` before rebuilding after a `.bib`
edit. The current four-pass build has 24 pages, 0 LaTeX errors, 0
undefined or multiply-defined citations, 0 overfull boxes, and main text ending
on page 9. The power table remains in Appendix G to preserve the page limit.

**Do not add `\usepackage{lineno}`.** `iclr2027_conference.sty` draws its own grey
submission ruler. Loading `lineno` as well renders two line-number columns, which
reads as a modified template at a venue that treats that as grounds for rejection.

---

## Where each claim lives

| Claim | Statement | Proof | Numerical check |
|---|---|---|---|
| Graft superposition | Thm 1, §3 | App. A | `verify_theory.py` C1–C5 |
| One-pass scoring, *k* donors | Cor. 2, §3 | App. A | C14 |
| Selection margin, conformal clause | Prop. 3, §3 | App. A | C10, C16a–d |
| Continuous coefficients, ρ | Cor. 4, App. A | App. A | C15, C17 |
| Interval and subset optimisers | §4, App. B | App. B | C6, C7, C12 |
| Drift certificate | §4, App. B | App. B | C8 |
| Regime boundary | §5, Fig. 2(c) | — | C9 |
| Mechanism ablation and baselines | §5, Fig. 2(d), Table 1 | — | C18 |
| Seed variance | §5, Table 1 | — | C19 |
| Ranking design has no power | §6, Table 2 | — | `power_analysis.py` |
| λ is *not* predictive (refuted) | App. E | — | C13, reported as excluded |

Main-body floats: Figures 1–2 and the controlled-stack Table 1. Appendix:
the full verification, design-space, power, language-model protocol, and
reviewer-response tables. The power table is in Appendix G so the main text
stays within nine pages.

---

## Running the checks

NumPy/SciPy only, CPU-minutes.

```
python3 verify/verify_theory.py          # 26/26 claimed checks, PASS/FAIL per claim
python3 verify/power_analysis.py         # reproduces Table 2
python3 verify/power_analysis.py --smoke
python3 verify/independent_check.py      # clean-room re-derivation
python3 verify/corrections_check.py      # verifies the corrected constants
python3 verify/make_figdata.py           # regenerates figs/*.dat
python3 verify/toy_transfer.py           # the negative testbed of Appendix E
```

The separate reviewer-response diagnostics are reproducible from their frozen
contracts when the cached checkpoints are available:

```
python -m suture.reviewer_followup
python -m suture.reviewer_cost_diagnostics
python -m suture.reviewer_llm_comparison --device cuda --prompt-limit 8 --max-length 128
```

They write only to `results/reviewer_followup/`. The Qwen3 follow-up stops at
`STOP_P1_FAKE`; the fixed-window comparison reuses archived free-generation
outcomes and does not read MGSM-test.

Two things to know. `verify_theory.py` and `make_figdata.py` draw **independent**
random graft sets: the superposition correlation is 0.980 from the figure draw,
0.988 from the harness draw, and 0.956 ± 0.040 across 12 stacks. Table 3 reports
all three with their provenance. And `verify_theory.py` writes
`figs/fig_ablation.dat`, so Figure 2(d) and Table 1 come from the same run.

`independent_check.py` and `corrections_check.py` re-derive the recursion, the
remainder order, the ρ constant, the conformal clause and the Clopper–Pearson
coverage from the statements alone, without importing the main harness. Two of
the errors they document were found that way.

---

## What still needs doing

The current pass closed the factual and bibliographic items:

1. **R1/B3:** the LightOn/Lasbordes Qwen3-8B specialists are public and
  Apache-2.0, but full bf16 specialist pairs do not fit the recorded
  8192-MiB RTX 3060 Ti. The fail-closed A100-80GB cloud protocol is now
  complete for French and Chinese, with positive held-out teacher-forced
  candidate comparisons archived under `results/b3/lighton_qwen3_8b/`.
  A broader five-language/full-window/free-generation follow-up was attempted
  under a separate contract and stopped at its generation gate; it produced no
  reportable result and is not a current claim.
- **R2/B1:** the paper explicitly says the Overleaf bundle is compile-only;
  local JSON/JSONL evidence and manifests can be shipped as a code supplement,
  while weights are excluded.
- **R3:** hardware, environment pins, LID hash, recorded post-resume time, and
  the fact that total GPU-hours were not logged are stated.
- **R4:** model/data licences, the in-study translation pins, MGSM hash-frozen
  64/186 split, file hashes, and the missing dataset-revision pin are stated.
- **B4:** the remaining author/venue placeholders were resolved from primary
  sources; `suture.bib` contains no `UNVERIFIED` marker.

Human-only gates remain:

1. **B2 — a named author re-derives the proofs line by line** and signs off.
2. **Get 2–3 independent human mock reviews.** Treat any unresolved finding as
   blocking.
3. Author-review the final render and bibliography, confirm anonymity, and
   decide whether to attach the local JSON/JSONL supplement. The current
   technical render has already been checked for red placeholders and
   bibliography output.
4. Put the directory under version control.

---

## Integrity notes

- Anonymity: clean. No identifying repository URLs, acknowledgments or
  first-person references to prior work.
- AI use: the statement matches ICLR 2027's policy and must also be reproduced on
  the OpenReview form. Bibliography metadata was checked against primary
  sources in this pass.
- Nothing here was fabricated to fill a gap. Numbers that could not be traced to
  provided data are marked as taken on trust in `REDFLAGS.txt`; B3 numbers are
  traced to the frozen cloud manifests and raw JSONL records. The powered
  Spanish 64-item ranking is traced to
  `results/reviewer_followup/powered_spanish_ranking/`. The local hardware
  limitation remains true, but it is no longer a claim that B3 was unrun.
- The abandoned expanded B3 pilot is documented only in
  `B3_EXPANDED_RUNBOOK.md`. Its partial manifests were removed, are not
  evidence, and did not alter the immutable v1 result root.
