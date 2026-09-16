# Project explainer

## Problem

Layer composition transfers a capability from a donor Transformer to a host
model by replacing a selected set of corresponding layers. The usual method
evaluates every candidate graft as a separate merged model. For a model with
`L` layers, that makes candidate selection proportional to the number of
compositions and requires repeated forward passes through separately built
models.

SUTURE asks whether the candidate effects can be estimated from the experts'
existing trajectories instead.

## First-order score

For each layer, the method computes a donor injection on the host trajectory.
The host's adjoint for a selected utility or risk readout propagates the
injection to the end of the network. The first-order contribution is the
contraction of those two quantities. A candidate interval is scored by
summing its layer contributions.

The method therefore needs:

1. one host forward trajectory;
2. one host reverse/adjoint calculation per readout;
3. streamed donor-block evaluations at each layer;
4. no merged candidate models during selection.

The approximation error is controlled by the second-order remainder in the
fine-tuning scale under the smoothness assumptions stated in the paper. A
utility score and a risk score can be combined as a constrained interval
selection problem. The same per-unit decomposition also supports arbitrary
subsets and multiple donors in the controlled experiments.

## Experimental lanes

### Controlled-stack verification

`verify/verify_theory.py` constructs synthetic stacks where exhaustive
composition search is available as ground truth. It checks the superposition
identity, remainder behavior, adjoint contraction, interval/subset optimizers,
drift certificate, regime boundary, ablations, and seed variation. The
independent and corrections checks rederive selected constants without
importing the main harness.

### Tier-A language-model evidence

The Tier-A lane uses frozen data contracts, local Qwen-family checkpoints,
matched adapters, and explicit readiness/plumbing gates. The canonical
Spanish E1 result is retained as a stopped configuration-specific result.
Contract-v2 artifacts document Qwen3-1.7B readiness and plumbing boundaries;
they do not authorize a Qwen3 E1 claim.

### B3 scale check

The B3 lane evaluates public LightOn Qwen3-8B specialists. It scores all
declared contiguous intervals using independent utility and risk probes, then
measures only the host, selected, and published reference candidates on
held-out MGSM-Rev2 records. The retained French and Chinese numbers are
teacher-forced answer log-probability comparisons. They should not be
interpreted as a full held-out sweep or as evidence for free-generation
ranking.

### Reviewer-response diagnostics

The reviewer-followup artifacts contain controlled multi-donor and signed
selection checks, local cost diagnostics, score-variant comparisons, and an
isolated powered Spanish free-generation ranking. These runs are separate
contracts and cannot modify the canonical Tier-A roots.

## Reading the result tree

`results/` is evidence, not a scratch directory. Canonical files are
immutable by convention and many runners enforce this with fail-closed path
guards. A result's `run_manifest.json` records the command, model pin,
environment information, input/output hashes, and selection invariants. The
artifact inventory distinguishes canonical, preliminary, smoke, and
superseded runs.

Weights are not required to inspect the retained outputs. A complete
from-scratch rerun requires the public checkpoints and local hardware
described by the relevant contract and `docs/reproducibility.md`.
