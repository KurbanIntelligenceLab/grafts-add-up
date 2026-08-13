# Choosing What to Merge Without Merging — student guide to this bundle

This is the entry point. Read this first, then `RUNBOOK.txt` when you are ready
to run things.

**Status: complete pending validation.** The paper is written end to end and
the mathematics is verified. No language-model experiment has been run. Every
number that depends on one appears in the PDF as a red `[RES]` placeholder.
That is deliberate. It is not an oversight to fix by filling in estimates.

---

## 1. What the paper claims, in plain terms

**The one-sentence thesis.** The effect of swapping transformer layers between
two experts is *additive across layers*, so you can score every possible swap
from a single pass through one model instead of building and benchmarking
hundreds of merged models.

**The axis the paper turns on.** Every existing criterion for what to graft is a
property of the *experts alone*: how far fine-tuning moved each parameter
(Bandarkar et al.; Ansell et al.; Shin & Hwang), how much task vectors interfere
(TIES, DARE), activation statistics (ACM), spectral structure of an interference
operator. These are cheap and behaviour-agnostic, which is also their limit: they
return **one ranking of units** no matter which behaviour you want and which you
must preserve. Here the trade-off between those two *is* the problem. We score by
the predicted change in a chosen readout, a property of the experts **and** the
behaviour, which is what makes two competing objectives and a hard constraint
expressible at all.

**A correction we had to make.** An earlier draft claimed module-level
granularity and blending were untested. They are not: Shin & Hwang (2026,
arXiv:2601.22620) already swap at module granularity and blend, in this same
cross-lingual setting, selecting by the difference in relative parameter update
ratios. That claim was removed. The correction improved the paper: their
specialisation ratio is now a **published, competitive, non-strawman baseline**
in E4, and E3 is the decisive test of whether a behaviour-referenced criterion
beats an expert-intrinsic one at localising the language gate. If it does not,
the extra machinery is not earning its place and the paper should say so.

**The gap.** "Layer swapping" takes a skill expert (say, English math) and a
language expert (say, Swahili general chat), both fine-tuned from the same base
model, and builds a new model by taking some blocks from one and the rest from
the other. It works, and it moves capability into languages that have no task
data. But *which* blocks to swap is found by brute force: build a merged model
for each candidate window, run a labelled benchmark on it, keep the best. Three
separate published papers do exactly this, and each lands on a different
window. The window moves with the base model, with the language, and with the
task. And picking it slightly wrong is catastrophic: one paper reports that a
two-block error made ~60% of the model's reasoning traces switch back to
English.

So the sweep is paid over and over, it needs labelled data in a language that
by assumption has none, and nothing predicts its answer.

**The contribution, and the delta over the closest work.** We show the sweep is
avoidable, and that avoiding it changes what can be searched at all, because the
quantity being searched over has exact structure:

1. Write the composed model's deviation from the host as an exact recursion.
   It decomposes into one *injection* per grafted layer, each propagated
   forward by the host's own Jacobians, plus a remainder that is second order
   in how far fine-tuning moved the two experts apart. (Theorem 1)
2. Contract that with the gradient of any readout and each layer's contribution
   becomes a single number you read off a backward pass. Two forward passes and
   one backward pass through the *host alone* score all 2^L possible graft sets.
   No merged model is built. (Corollary 2)
3. Picking the best window subject to a "don't lose the target language"
   constraint is then a constrained maximum-subarray problem, solved exactly in
   O(L log L). (Proposition 3)
4. The one model you do build gets a distribution-free upper bound on its
   language-drift rate, computed from unlabelled text. (Proposition 4)

**The part that matters most.** The scores are also *additive across several
donors* (Pearson 0.961 over 80 random three-way assignments, where enumeration
would need 3^12 = 531,441 models) and *linear in a continuous interpolation
coefficient* (Pearson 0.9995). So one measurement covers discrete grafts, soft
merging coefficients, any unit granularity, and k experts alike. That is what
takes this from "a trick for layer swapping" to "a way to score any weight-space
composition without building it", and it is why the paper is aimed at the
merging literature and not only the cross-lingual one.

Scoring costs the same whether the design space
holds `L` candidates or `2^L`. A sweep must enumerate, so the space has to stay
small enough to enumerate, and that is *why* every published layer swap grafts a
contiguous band of whole layers. Neither restriction has ever been tested. In
the controlled stack the unrestricted optimum beats the best contiguous band by
**82%**, and enumerating the space it was found in would have cost 2^12 = 4096
merged models against the 24 numbers SUTURE actually used. The contribution is
not "the same search, faster"; it is a search space that was previously
unreachable.

**The honest baseline.** Do not compare against the *labelled* sweep. Nothing
forces a sweep to use labels: you can build each candidate and score it on the
same cheap unlabelled surrogate. That is the real competitor, it is in the main
results table, and beating only the labelled sweep would prove nothing. Against
the surrogate sweep the ratio is about C/3 for C candidates, and the point is
that SUTURE is *constant* in C where the baseline is linear in it.

Against the closest neighbours: the layer-swap papers *search* for the band and
never predict it. Layer-wise merging methods (AdaMerging and relatives) learn
coefficients by backpropagating through the merged model on target-task data,
which does not exist here because neither expert can do the task in the target
language. Mergeability predictors (SimMerge, Demystifying Mergeability) predict
*whether* a merge configuration works, from fitted models of observed merges;
we derive *which layers* from the model's own adjoints with nothing fitted.
Attribution patching linearises *activation* patches inside one model to explain
behaviour; we linearise *parameter* grafts across two models to choose a
deployable artifact.

**Why this could matter beyond the paper.** The reusable piece is the released
sweep tables: exhaustive window-by-window accuracy and drift for each model and
language. Nobody has published one, which is why everyone re-runs it. Any future
graft-selection method can be scored against them without paying the sweep.

---

## 2. Files

| File | What it is |
|---|---|
| `main.tex` | The paper. |
| `suture.bib` | Bibliography. **Read section 6 below before submitting.** |
| `main.bbl` | Pre-compiled bibliography, so one `pdflatex` pass resolves references even without BibTeX. |
| `main.pdf` | The built paper. |
| `build.sh` | Four-pass build. Use this, not a bare `pdflatex`. |
| `figs/fig1_teaser.tex` | Figure 1 (TikZ). |
| `figs/fig2_verify.tex` | Figure 2 (pgfplots). |
| `figs/*.dat` | Figure data. **Every one of these is computed**, not hand-entered. |
| `suture_metrics.py` | Reference implementation. Scores, both optimisers, the certificate, the statistics. |
| `verify/verify_theory.py` | Adversarial verification of every formal claim. |
| `verify/make_figdata.py` | Regenerates `figs/*.dat` from the verification harness. |
| `verify/toy_transfer.py` | A trained transformer testbed. **It produces a negative result** (see below). |
| `REDFLAGS.txt` | **Read this.** Two file-integrity incidents, what was inserted, and what was reverted. |
| `RUNBOOK.txt` | The experimental protocol. |
| `suture_iclr2027.zip` | Everything, ready to compile from a clean directory. |

### Building the paper

```
./build.sh
```

Runs `pdflatex → bibtex → pdflatex → pdflatex` and then prints the undefined-
citation count, the overfull-box count, and the page count. All three should be
`0`, `0`, and a page count whose main text is within the venue limit.

Do not chain the passes with `&&`. `pdflatex` returns a nonzero status on mere
warnings, which silently skips the next pass and leaves the PDF one generation
behind the `.aux`. This happened while preparing the bundle and produced a PDF
with stale theorem numbers that looked like a LaTeX bug. `build.sh` runs every
pass unconditionally for this reason.

### Swapping in the official ICLR style

`main.tex` currently uses a portable preamble that approximates ICLR's page
geometry, because the official style files were not reachable from the machine
that prepared this. To switch:

1. Download `iclr-2027-style-files.zip` from the link in the venue's author
   guide and unpack `iclr2027_conference.sty` and `.bst` into this directory.
2. Replace the `\documentclass` line and the `geometry`/`times` lines with the
   template's `\documentclass{article}` + `\usepackage{iclr2027_conference,times}`.
3. Delete the `lineno` lines; the ICLR style handles line numbers itself.
4. Keep the `\emergencystretch` and the line-number-separation fix.
5. Rebuild and **re-check the page count**, which will change.

### Where each claim lives

| Claim | Section | Proof | Numerical check |
|---|---|---|---|
| Exact recursion | App. A Step 1 | App. A | `verify_theory.py` C1 |
| Superposition (Thm 1) | §3 | App. A | C2, C3, C4, C5 |
| One-pass scoring (Cor 2) | §3 | App. A | C2 |
| Interval optimiser (Prop 3a) | §3 | App. B | C6 |
| Subset DP (Prop 3b) | §3 | App. B | C7 |
| Drift bound (Prop 4) | §3 | App. B | C8 |
| Selection works | §5 | — | C10, C11 |
| Subsets beat intervals | §5 | — | C12 |
| Multi-donor additivity | §3 | App. A | C14 |
| Linear in alpha | §3 | App. A | C15 |
| Where it stops working | §5 | — | C9 |
| Refuted conjecture (not claimed) | App. D | — | C13 |
| Testbed negative result | App. D | — | `toy_transfer.py` |

---

## 3. The code

### Verification harness

```
python3 verify/verify_theory.py
```

Recomputes every number in Table 1 and prints `PASS`/`FAIL` per claim. Minutes
on a CPU, NumPy only. It should end `14/14 checks passed.`

It should end `17/17 claims made in the paper passed.` followed by a separate
list of *conjectures tested but not claimed*.

This harness is the reason the paper's claims are what they are. It caught two
things during drafting. First, a genuine error in the subset optimiser: the
initial version mishandled layers whose risk score *buys* budget rather than
spending it, giving a 2.62 optimality gap. Second, and more importantly, it
**refuted a claim we wanted to make**. We conjectured that at fixed weight
distance the linearisation error would track the donor-host Jacobian gap
`lambda`, which would have distinguished the analysis from a generic Taylor
argument. Measured correlation: **-0.304**, the wrong sign. The claim is out of
the paper, the check stays in the harness, and it is reported as
`NOT SUPPORTED -> excluded` so it is neither counted as a pass nor deleted.

If you change the theory, change this first and let it tell you whether the new
claim survives.

### The trained testbed, and the negative result it produced

```
python3 verify/toy_transfer.py --quick
```

A small transformer with real attention, layer normalisation and GELU MLPs, with
a hand-written backward pass gradient-checked to 9e-9. It pretrains a base,
fine-tunes a skill expert and a language expert from it, and leaves the target
cell genuinely unseen, so the sweep can serve as ground truth.

**It does not reproduce the transfer phenomenon.** Across two designs of the
pretraining task, no graft improved on the host. This is in the paper
(Appendix D) rather than hidden, and it is the single most actionable thing in
this bundle for your planning: the natural inference is that layer swapping
needs a base whose multilingual representations survive expert fine-tuning,
which a small from-scratch model does not have. **Run E1 on a genuinely
pretrained multilingual base.** A plan that substitutes a small model you train
yourself will fail for reasons that have nothing to do with the method, and you
will waste weeks concluding the wrong thing.

### Reference implementation

```
python3 suture_metrics.py --smoke
```

A synthetic fixture exercising every code path: scoring, the linearisation
diagnostic, both optimisers against brute force, certificate coverage, and the
refusal behaviour. Ends `SMOKE TEST PASSED`.

The framework-specific part (running a transformer, capturing residual states,
taking a backward pass) is deliberately *not* in this file. It sits behind the
`ModelAdapter` class, so porting to PyTorch touches one class. You implement
four methods:

```
residual_states(prompt) -> [x_0, ..., x_L]     host forward, cache the stream
host_block(l, x)        -> host block l's residual UPDATE at x
donor_block(l, x)       -> donor block l's residual UPDATE at x
adjoints(prompt, phi)   -> [s_0, ..., s_{L-1}] where s_l = d phi / d x_{l+1}
```

Then `suture_scores(...)` gives you `a^u` and `a^r`, `select_graft(...)` gives
you the band, and `certify_drift(...)` gives you the bound.

**Anti-fabrication contract.** Nothing in this module defaults, falls back, or
asserts an expected outcome. Empty probe sets, empty calibration sets,
malformed drift flags and out-of-range layer indices all raise `SutureError`.
If a run is missing data you will get an exception, not a plausible number.

---

## 4. The experiments

`RUNBOOK.txt` is the protocol. The short version:

- **Use a genuinely pretrained multilingual base.** Not negotiable, and not an
  opinion: see the testbed negative result above.
- **Do the cheap pilot first** (P1–P3, about an hour). P1 uses a *fake* donor
  made by adding noise to the host, which must be in the linear regime, so it
  tests your plumbing and not the science. If P1 fails, the bug is yours.
- **Then E1, the positive control.** Exhaustive sweep at 1B–2B scale versus one
  SUTURE run. This is the experiment that can falsify the paper, which is why
  it runs first.

### What "promising" and "negative" look like, and what to do

**Promising:** E1 gives high rank correlation between predicted and swept window
values, and the selected window's accuracy is close to the sweep optimum, at
roughly two orders of magnitude less compute. Then:
- Do **not** relax. Promising-on-first-look is a hypothesis, not a result.
  Reviewers will ask whether the effect is the claimed mechanism or a confound.
  E3 (does the risk score actually localise the gate, where the weight-space
  statistic does not?) and E4 (does a dumb weight-norm selector do the same?)
  are the confound checks. Run them before believing E1.
- Reconcile every claim with the real numbers: abstract, title framing,
  contribution list. Replace every `[RES]`.

**Negative:** E1 gives low rank correlation, or the regret is a large fraction
of the achievable gain. Then:
- Check E2 first. If real experts sit outside the usable regime mapped in
  Figure 2(c), that *explains* the negative result and is itself a contribution:
  it tells the field the linearisation-based shortcut does not reach 8B experts
  and says why.
- Do not tune the probe set until E1 passes and then report the tuned number.
  That is fitting to the test.
- Check that you actually beat the **unlabelled surrogate sweep**, not just the
  labelled one. If SUTURE only matches the surrogate sweep but at a fraction of
  the cost, that is still a result, and E5 is then what carries the paper.
- If the central claim is false, the paper says so. Section 8 already commits to
  this in writing. A negative result with a clean mechanism is publishable; a
  laundered one is misconduct.

**Partial:** ranking works but magnitudes are badly off. This is the *expected*
outcome and the paper is already written for it. Selection needs only ranking;
the certificate is empirical. Report both numbers.

---

## 5. Rebuttal kit

Predicted objections, and where the answer already is. Any objection without a
current answer is listed as a `>>> TO RUN` item, not argued around.

**"Attribution patching is known to break for large perturbations. Layer grafts
are large. Your linearisation will fail."**
This is the killer objection and the paper does not dodge it. Three answers.
(i) Section 5 and Figure 2(c) *map the failure* rather than hiding it: rank
agreement is 1.00 at small scales, 0.976 at moderate, and collapses to 0.73 past
a scale of ~0.4. (ii) Selection needs only the *ranking*, which Figure 2(c)
shows survives well past the point where magnitudes stop being usable (relative
magnitude error is already 9–29% where the ranking is exact). (iii) The deployed
guarantee is the empirical Clopper–Pearson bound, not the linearisation, so a
magnitude error changes which band is chosen and cannot invalidate the
certificate. E2 measures where real experts sit; SUTURE-IG is the fallback.

**"Your baseline is a strawman. Just run the sweep with a cheap unlabelled
score instead of a benchmark."**
Correct, and the paper says so in the introduction and makes that the primary
baseline in E4 rather than waiting to be told. The surrogate sweep still builds
one model per candidate and still pays a probe-set forward pass for each, so its
cost is linear in the number of candidates. That dependence, not the labels, is
what SUTURE removes, and it is why the surrogate sweep cannot be run at all over
non-contiguous or sublayer design spaces.

**"This is a hyperparameter search paper."**
It would be, if the design space were fixed. The design space is a consequence
of the search method: enumeration forces contiguous layer bands. E5 runs
selection in four spaces at identical cost and reports what the restriction was
costing.

**"A weight-norm heuristic would give the same answer more cheaply."**
E3 is designed exactly for this. Prior work's cross-language weight-delta
alignment statistic is near-identical across its languages, yet the gate the
same paper had to use moves between them. If a weight-space statistic could
localise the gate, that paper would not have needed a second sweep. E4 includes
a weight-norm selector as an explicit baseline.

**"Why not AdaMerging / entropy minimisation / self-labelling?"**
Section 2 answers this and E4 runs them anyway. Those methods need a usable
signal at the target cell, and here neither expert can perform the task in the
target language, so there is no competent teacher and no reason to trust the
composed model's confidence on a behaviour it was never trained to produce.
They also backpropagate through the merged model, restoring the per-candidate
cost we are removing.

**"Your bound has a constant exponential in depth. It is vacuous."**
Agreed, and Remark 1 says so in the paper before a reviewer can. Theorem 1 is a
statement about the *form* of the error and its scaling, not a numerical bound.
Nothing is built on the constant. The operational guarantee is
distribution-free and empirical.

**"The readouts are teacher-forced log-probabilities but the failure is a
property of free generation."**
Correct, and it is in the limitations section rather than assumed away. E7(d)
measures the correlation between the teacher-forced surrogate and free-generation
drift directly.

**"The controlled experiments are a toy."**
Yes, and the paper says so three times, including in the abstract. Section 5
verifies the *mathematics* where exhaustive search is affordable enough to serve
as ground truth. It is labelled as not being evidence about language models.
E1 and E2 are the language-model evidence.

**"Contiguity is assumed."**
It is assumed by all prior work and tested by none. E5 solves the unrestricted
subset problem and reports whichever way it comes out.

**"Task arithmetic is already known to be approximately linear, so your
superposition theorem is unsurprising."**
Cited and answered in Related Work. Tangent-space results concern linearity in a
continuous coefficient scaling a *full* task vector. This is additivity across a
*discrete per-unit mask*, plus the two facts that make it an algorithm instead of
a description: the injection is evaluated on the host trajectory so no merged
model is needed, and the propagator is the host's, so one reverse sweep serves
every unit at once.

**"This only applies to binary two-expert layer swaps."**
No: the scores are additive across donors and linear in an interpolation
coefficient, both verified (C14, C15). One measurement covers k experts, soft
merging coefficients and any unit granularity. E8 tests both generalisations at
scale and states the number that would falsify them.

**"Is this just attribution patching applied to merging?"**
The lineage is acknowledged and cited. The delta: prior work linearises
*activation* patches from a counterfactual prompt within one model, to explain
behaviour; we linearise *parameter* grafts between two models fine-tuned from a
shared base, to choose a deployable artifact. That changes the injection from an
activation difference to a block-function difference, which is what makes the
whole 2^L design space additive and is why the selection problem becomes a
maximum-subarray solve rather than a ranking of components.

### If a reviewer asks for X, we already have Y

| Asks for | We have |
|---|---|
| an oracle upper bound | the full labelled sweep, E1/E4 |
| a trivial baseline | random-window distribution over ≥20 draws, E4 |
| significance testing | paired bootstrap over items, Holm across languages |
| variance | 3 expert seeds × 5 decode seeds, sd reported |
| cost accounting | merged models built and wall-clock, both routes, Table 2 |
| a failure analysis | Figure 2(c) regime map + E2 |
| proofs | Appendices A and B, with assumptions stated inline |
| code | NumPy-only harness that recomputes every reported number |

---

## 6. Blocking checklist before submission

Nothing below is optional. Unresolved items block submission.

- [ ] **Run the experiments.** Every `[RES]` in the PDF must become a real
      number or the claim must go. Grep the source: `grep -c 'RES' main.tex`.
- [ ] **Reconcile every claim with the numbers.** Abstract, title framing,
      contribution list. Walk claims back if the effect is softer than hoped.
- [ ] **Read `REDFLAGS.txt` first.** Two bibliography entries were silently
      changed during preparation: flagged placeholders were replaced with full
      author lists, under comments claiming verifications that were never
      performed. Both were caught by reading the rendered PDF, reverted, and
      documented. Spot-check every entry whose comment says `VERIFIED`, and put
      this project under version control before doing anything else.
- [ ] **Fix the bibliography.** These entries are flagged and will render
      visibly broken until fixed: the multilingual-safety EACL paper, SimMerge,
      Multilingual Steering by Design, LinguaMap, Expert Merging, SyMerge, LOT
      Merging, SuperMerge, and the composable-sparse-fine-tuning entries all
      say `AUTHORS UNVERIFIED`. Model soups, AtP*, EAP-IG, conformal risk
      control and the Nanda attribution-patching note have incomplete author
      lists. Resolve each from the primary source. **Do not guess a name.**
- [ ] **Verify the proofs independently.** Theorem 1, Corollary 2, Propositions
      3 and 4 and both appendix proofs were AI-drafted. A human author must
      check them line by line. The numerical harness checks consistency with
      computation; it does not check that a proof is correct.
- [ ] **Re-run the novelty search.** This area moved twice in the twelve months
      before this draft, and a later search round found that two claims in an
      earlier version of this very paper were false. Assume the same can happen
      again. Re-search immediately before submission and confirm the
      closest neighbours are still the ones cited.
- [ ] **Get 2–3 independent human mock reviews** and resolve every issue raised.
      This is the single step most correlated with turning a strong draft into
      an accept, and this bundle cannot perform it.
- [ ] **Complete the AI use statement.** Mandatory at this venue and it must be
      accurate. Required disclosure covers developing the theoretical framework,
      formulating mathematical claims, assisting in writing proofs, proposing
      hypotheses, designing the protocol, implementing methods, and interpreting
      results.
- [ ] **Anonymise.** Double-blind. Third-person citations for your own work,
      no identifying repos or acknowledgements.
- [ ] **Check the page limit** after swapping in the official style file.
- [ ] **Clean-room compile.** Unzip the bundle into a fresh empty directory and
      run `./build.sh` there. Building from your working directory hides missing
      files.

---

## 7. Honest calibration

This process cannot make acceptance certain, and no one should tell you
otherwise. Top venues reject most submissions and the reviewer draw is noisy.

What is genuinely done: the wedge is real and open (verified by repeated
date-aware search against three follow-up papers and the merging literature),
the framing neutralises the most likely rejection cause, the theory is proved and
numerically checked with its failure regime mapped rather than hidden, the
baseline is the strong one rather than the flattering one, and two results that
did not go our way are reported rather than dropped.

What is still weak, stated plainly: the technical novelty of any single
component is modest. The linearisation is known (attribution patching), the
interval optimiser is a constrained Kadane variant, the subset solver is a
knapsack DP, and the certificate is a Clopper-Pearson interval. What is new is
the composition, the object it is applied to, and the design space it opens. A
reviewer who wants each ingredient to be novel will not be satisfied, and the
only answer to that reviewer is a strong E1 and E5. This is another way of
saying the same thing:

What decides it: the experiments, which are yours. The dominant factor in most
decisions at this venue is the strength and robustness of the evidence, and
none of it exists yet. If E1 comes back strong and survives E3 and E4, this is
competitive. If E1 comes back weak, the honest paper is a different and smaller
paper, and Section 8 already commits to writing it.
