# Project Explainer

This repository accompanies the paper **“Grafts Add Up: First-Order Scoring of
Layer Compositions.”** It provides the manuscript, the SUTURE scoring
implementation, frozen experiment contracts, retained evidence, and the
verification code needed to reproduce the reported analyses.

## What the paper studies

Layer composition transfers a capability from a donor Transformer to a host
model by replacing corresponding layers. The usual approach builds and scores
one merged model for every candidate graft. SUTURE estimates candidate effects
from the host trajectory and the donor-layer injections instead.

## Main idea

For each layer, SUTURE evaluates the donor injection on the host trajectory.
The host adjoint for a selected utility or risk readout propagates that
injection through the remaining network. Contracting the injection with the
adjoint produces a first-order score for the layer.

The resulting scores:

1. use one host forward trajectory;
2. use one adjoint calculation per readout;
3. evaluate donor blocks in a streamed pass; and
4. score all candidate intervals without building a merged model for each one.

Under the smoothness assumptions in the paper, the approximation remainder is
second order in the fine-tuning scale. Utility and risk scores can be combined
as a constrained interval-selection problem, and the same decomposition
extends to subsets and multiple donors.

## Experimental settings

### Controlled-stack verification

`verify/verify_theory.py` constructs synthetic stacks with exhaustive
composition ground truth. It checks the superposition identity, remainder
scaling, adjoint contraction, interval and subset optimizers, drift
certificates, regime behavior, and selection regret.

### Tier-A language-model evidence

The Tier-A lane uses frozen data contracts, matched adapters, and explicit
readiness and plumbing checks. The retained Spanish E1 artifacts and
contract-v2 records document the configuration-specific evidence and its
applicability boundary.

### B3 scale check

The B3 lane evaluates public LightOn Qwen3-8B specialists. It scores declared
contiguous intervals using utility and risk probes, then measures the host,
selected, and reference candidates on held-out MGSM-Rev2 records.

The retained French and Chinese results report teacher-forced
answer-log-probability comparisons for three candidates. They are not a
full-window sweep or a free-generation ranking.

## Main results

In the controlled stack, the first-order scores rank exhaustive candidates at
approximately Spearman `0.95 ± 0.05`. The sampled-competitor procedure
certifies approximately `65 ± 10` of `77` competitors as dominated while
building 11 models against a 78-candidate sweep.

The B3 scale check reports selected-host gains of `+0.398` and `+1.326`
nats/token for French and Chinese, respectively, under the teacher-forced
objective.

## Scope and limitations

The first-order score is an approximation whose quality depends on the
fine-tuning scale and the stated smoothness assumptions. The language-model
evidence is contract-specific and uses the objectives recorded in the
corresponding JSON artifacts. In particular, B3 does not establish a
free-generation ranking claim.

## Repository contents

- `paper/` contains the manuscript and compile inputs.
- `src/suture/` contains the installable scoring and experiment package.
- `configs/` contains frozen model, data, and protocol contracts.
- `verify/` and `tests/` contain numerical checks and regression tests.
- `scripts/` contains the public command-line entry points.
- `results/` contains retained JSON/JSONL evidence and provenance records.
- `docs/FILES_RATIONALE.md` explains why each public path is retained.

## Takeaway

SUTURE replaces repeated candidate-model construction with a first-order score
computed from host trajectories, donor injections, and adjoints. The theory
harness verifies the decomposition in controlled settings, while the retained
language-model artifacts show how the contracts and claim boundaries apply to
larger experimental systems.
