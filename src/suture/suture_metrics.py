"""
suture_metrics.py -- reference implementation of the quantities in
"Choosing What to Merge Without Merging: Additive Effect Scores for
Model Composition".

NumPy only. No deep-learning framework is required to read, test, or reason
about this file. The two entry points a practitioner needs are:

    suture_scores(...)      -> per-layer utility and risk scores a_l
    select_graft(...)       -> the graft set, via the exact interval optimiser
                               or the conservative subset DP
    certify_drift(...)      -> binomial upper bound on the drift rate (i.i.d. calibration)

The framework-specific part (running a transformer, capturing residual states,
taking a backward pass) is deliberately NOT in this file. It is isolated behind
the `ModelAdapter` protocol below so that the mathematics can be tested without
a GPU, and so that swapping in PyTorch/JAX touches one class and nothing else.

ANTI-FABRICATION CONTRACT
-------------------------
Every function here computes. None of them assert an expected outcome, fall
back to a default when data is missing, or fill in a plausible number. Missing
or malformed inputs raise. `--synthetic` runs are labelled as such in every
output path so a synthetic result can never be mistaken for a real one.

Run the smoke test:
    python -m suture.suture_metrics --smoke
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "GraftScores",
    "suture_scores",
    "best_interval",
    "best_subset",
    "select_graft",
    "clopper_pearson_upper",
    "certify_drift",
    "bootstrap_ci",
    "paired_bootstrap_test",
    "spearman",
    "linearisation_diagnostic",
]


# ===========================================================================
# 0.  Errors
# ===========================================================================

class SutureError(RuntimeError):
    """Raised whenever an input is missing or inconsistent.

    This exists so that nothing in this module silently substitutes a default
    for data that was supposed to be measured.
    """


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SutureError(message)


# ===========================================================================
# 1.  Per-layer scores  (Corollary 2 of the paper)
# ===========================================================================

@dataclass
class GraftScores:
    """Per-layer utility and risk scores plus the diagnostics needed to know
    whether the linearisation is in its usable regime."""

    utility: np.ndarray                    # a^u_l, shape (L,)
    risk: np.ndarray                       # a^r_l, shape (L,)
    injection_norm: np.ndarray             # ||v_l||, shape (L,)
    n_probe_utility: int
    n_probe_risk: int
    utility_sem: Optional[np.ndarray] = None
    risk_sem: Optional[np.ndarray] = None
    meta: Dict = field(default_factory=dict)

    def __post_init__(self):
        L = len(self.utility)
        _require(len(self.risk) == L, "utility and risk must have the same length")
        _require(len(self.injection_norm) == L, "injection_norm length mismatch")
        _require(self.n_probe_utility > 0 and self.n_probe_risk > 0,
                 "scores were requested with an empty probe set")

    @property
    def n_layers(self) -> int:
        return len(self.utility)

    def epsilon(self, S: Sequence[int]) -> float:
        """Total injection magnitude of a graft set (the theory's eps_S)."""
        return float(sum(self.injection_norm[l] for l in S))

    def predict(self, S: Sequence[int]) -> Tuple[float, float]:
        """First-order predicted (utility change, risk change) for graft set S."""
        S = list(S)
        _require(all(0 <= l < self.n_layers for l in S), f"layer index out of range in {S}")
        return float(self.utility[S].sum()), float(self.risk[S].sum())

    def to_json(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump({
                "utility": self.utility.tolist(),
                "risk": self.risk.tolist(),
                "injection_norm": self.injection_norm.tolist(),
                "n_probe_utility": self.n_probe_utility,
                "n_probe_risk": self.n_probe_risk,
                "meta": self.meta,
            }, f, indent=2)


class ModelAdapter:
    """The only framework-dependent surface.

    An implementation must provide, for one probe example and one readout:

      residual_states(prompt)  -> list of L+1 arrays, the HOST trajectory
      donor_block(l, x)        -> the donor's block l applied to state x
      host_block(l, x)         -> the host's block l applied to state x
      adjoints(prompt, phi)    -> list of L arrays, d phi / d x_{l+1} on the
                                  host trajectory (one backward pass)

    `donor_block` and `host_block` return the residual *update*, not the state.
    A PyTorch implementation of this protocol is sketched in the runbook.
    """

    def residual_states(self, prompt) -> List[np.ndarray]:
        raise NotImplementedError

    def donor_block(self, l: int, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def host_block(self, l: int, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def adjoints(self, prompt, readout: str) -> List[np.ndarray]:
        raise NotImplementedError

    @property
    def n_layers(self) -> int:
        raise NotImplementedError


def suture_scores(adapter: ModelAdapter,
                  utility_probe: Sequence,
                  risk_probe: Sequence,
                  report_sem: bool = True) -> GraftScores:
    """Compute a^u and a^r.

    Cost per probe example and readout: one host forward pass (states), one
    extra forward pass of work (donor blocks evaluated off-path at the host
    states), one backward pass (adjoints). No merged model is built.
    """
    _require(len(utility_probe) > 0, "utility probe set is empty; nothing to score")
    _require(len(risk_probe) > 0, "risk probe set is empty; nothing to score")
    L = adapter.n_layers
    _require(L > 0, "adapter reports zero layers")

    per_example: Dict[str, List[np.ndarray]] = {"utility": [], "risk": []}
    inj_norm_acc = np.zeros(L)
    inj_count = 0

    for name, probe in (("utility", utility_probe), ("risk", risk_probe)):
        for prompt in probe:
            xs = adapter.residual_states(prompt)
            _require(len(xs) == L + 1,
                     f"adapter returned {len(xs)} states for {L} layers; expected {L+1}")
            v = [adapter.donor_block(l, xs[l]) - adapter.host_block(l, xs[l])
                 for l in range(L)]
            s = adapter.adjoints(prompt, name)
            _require(len(s) == L, f"adapter returned {len(s)} adjoints; expected {L}")
            per_example[name].append(
                np.array([float(np.vdot(s[l].ravel(), v[l].ravel())) for l in range(L)])
            )
            inj_norm_acc += np.array([float(np.linalg.norm(v[l])) for l in range(L)])
            inj_count += 1

    U = np.stack(per_example["utility"])
    R = np.stack(per_example["risk"])
    return GraftScores(
        utility=U.mean(axis=0),
        risk=R.mean(axis=0),
        injection_norm=inj_norm_acc / max(inj_count, 1),
        n_probe_utility=len(utility_probe),
        n_probe_risk=len(risk_probe),
        utility_sem=U.std(axis=0, ddof=1) / np.sqrt(len(U)) if (report_sem and len(U) > 1) else None,
        risk_sem=R.std(axis=0, ddof=1) / np.sqrt(len(R)) if (report_sem and len(R) > 1) else None,
        meta={"n_layers": L},
    )


# ===========================================================================
# 2.  Optimisers  (Proposition 3 of the paper)
# ===========================================================================

def best_interval(a_util: np.ndarray, a_risk: np.ndarray, tau: float
                  ) -> Tuple[Optional[float], Optional[Tuple[int, int]]]:
    """Exact constrained maximum subarray in O(L log L).

        max_{[i,j]} sum a_util   s.t.   sum a_risk >= -tau

    Prefix sums reduce this to, for each right endpoint b,
        min { U[i] : i < b, R[i] <= R[b] + tau },
    a prefix-minimum query over a key known offline. Coordinate compression
    plus a Fenwick tree of prefix minima gives O(log L) per operation.

    Returns (best value, (i, j)) or (None, None) if no window is feasible.
    """
    a_util = np.asarray(a_util, dtype=float)
    a_risk = np.asarray(a_risk, dtype=float)
    _require(a_util.shape == a_risk.shape, "utility and risk score shapes differ")
    _require(tau >= 0, "tau must be non-negative (it is a tolerance, not a target)")
    L = len(a_util)
    if L == 0:
        return None, None

    U = np.concatenate([[0.0], np.cumsum(a_util)])
    R = np.concatenate([[0.0], np.cumsum(a_risk)])
    keys = np.sort(np.unique(R))
    size = len(keys)
    tree = np.full(size + 1, np.inf)
    who = np.full(size + 1, -1, dtype=int)

    def update(pos: int, val: float, idx: int) -> None:
        while pos <= size:
            if val < tree[pos]:
                tree[pos], who[pos] = val, idx
            pos += pos & (-pos)

    def query(pos: int) -> Tuple[float, int]:
        best, arg = np.inf, -1
        while pos > 0:
            if tree[pos] < best:
                best, arg = tree[pos], who[pos]
            pos -= pos & (-pos)
        return best, arg

    best_val, best_arg = -np.inf, None
    for b in range(1, L + 1):
        i = b - 1
        update(int(np.searchsorted(keys, R[i], side="left")) + 1, U[i], i)
        qpos = int(np.searchsorted(keys, R[b] + tau, side="right"))
        if qpos > 0:
            mn, arg = query(qpos)
            if mn != np.inf and U[b] - mn > best_val:
                best_val, best_arg = U[b] - mn, (arg, b - 1)
    if best_arg is None:
        return None, None
    return float(best_val), best_arg


def best_subset(a_util: np.ndarray, a_risk: np.ndarray, tau: float, nbins: int = 5000
                ) -> Tuple[Optional[float], Optional[Tuple[int, ...]], float]:
    """Conservative pseudo-polynomial DP over arbitrary graft sets.

    Signed-weight knapsack. Negative risks are rounded away from zero (charged
    more than they cost) and positive risks toward zero (credited less than
    they earn), so every set the DP calls feasible is feasible for the true
    constraint. The optimality gap is one-sided and shrinks with `nbins`.

    Returns (value, graft set, grid step).
    """
    a_util = np.asarray(a_util, dtype=float)
    a_risk = np.asarray(a_risk, dtype=float)
    _require(a_util.shape == a_risk.shape, "utility and risk score shapes differ")
    _require(tau >= 0, "tau must be non-negative")
    _require(nbins >= 10, "nbins too small to give a meaningful discretisation")
    L = len(a_util)

    lo = float(a_risk[a_risk < 0].sum())
    hi = float(a_risk[a_risk > 0].sum())
    span = hi - lo
    if span <= 0:                                   # no risk variation at all
        S = tuple(int(i) for i in range(L) if a_util[i] > 0)
        return float(a_util[list(S)].sum()) if S else 0.0, S, 0.0

    step = span / nbins
    w = np.array([-int(math.ceil(-r / step)) if r < 0 else int(math.floor(r / step))
                  for r in a_risk], dtype=int)
    offset = int(math.ceil(-lo / step)) + 1
    size = offset + int(math.ceil(hi / step)) + 2

    NEG = -np.inf
    table = np.full((L + 1, size), NEG)
    table[0, offset] = 0.0
    idx = np.arange(size)
    for i in range(L):
        src = idx - int(w[i])
        cand = np.full(size, NEG)
        ok = (src >= 0) & (src < size)
        cand[ok] = table[i][src[ok]] + a_util[i]
        table[i + 1] = np.maximum(table[i], cand)

    feasible = ((idx - offset) * step >= -tau) & np.isfinite(table[L])
    if not feasible.any():
        return None, None, step
    j = int(np.flatnonzero(feasible)[np.argmax(table[L][feasible])])
    value = float(table[L][j])

    S: List[int] = []
    for i in range(L - 1, -1, -1):
        src = j - int(w[i])
        if (table[i][j] != table[i + 1][j] and 0 <= src < size
                and np.isfinite(table[i][src])
                and abs(table[i][src] + a_util[i] - table[i + 1][j]) < 1e-9):
            S.append(i)
            j = src
    return value, tuple(sorted(S)), step


def select_graft(scores: GraftScores, tau: float, shape: str = "interval",
                 nbins: int = 5000) -> Dict:
    """Run the optimiser and return the graft set plus its predicted effects."""
    _require(shape in ("interval", "subset"), "shape must be 'interval' or 'subset'")
    if shape == "interval":
        value, arg = best_interval(scores.utility, scores.risk, tau)
        if arg is None:
            raise SutureError(
                "no contiguous window satisfies the risk tolerance; either the "
                "tolerance is too tight or every layer is risk-negative. Do not "
                "silently relax tau: inspect the risk profile first.")
        S = tuple(range(arg[0], arg[1] + 1))
    else:
        value, S, _ = best_subset(scores.utility, scores.risk, tau, nbins=nbins)
        if S is None:
            raise SutureError("no subset satisfies the risk tolerance at this grid resolution")
    pu, pr = scores.predict(S)
    return {"graft": S, "shape": shape, "tau": tau,
            "predicted_utility": pu, "predicted_risk": pr,
            "epsilon_S": scores.epsilon(S), "objective": value}


# ===========================================================================
# 3.  Certificate  (Proposition 4 of the paper)
# ===========================================================================

def _binomial_cdf(k: int, n: int, p: float) -> float:
    """P(Bin(n, p) <= k) in log space so large calibration sets stay finite."""

    if p <= 0.0:
        return 1.0
    if p >= 1.0:
        return 1.0 if k >= n else 0.0
    logs = []
    log_p = math.log(p)
    log_q = math.log(1.0 - p)
    log_n_fact = math.lgamma(n + 1)
    for j in range(k + 1):
        logs.append(
            log_n_fact
            - math.lgamma(j + 1)
            - math.lgamma(n - j + 1)
            + j * log_p
            + (n - j) * log_q
        )
    maximum = max(logs)
    return math.exp(maximum) * sum(math.exp(value - maximum) for value in logs)


def clopper_pearson_upper(k: int, n: int, delta: float) -> float:
    """Exact (Clopper-Pearson) upper confidence limit for a binomial rate.

    Returns the largest p with P(Bin(n, p) <= k) > delta, computed by bisection
    on the CDF so that no SciPy dependency is needed.
    """
    _require(n > 0, "cannot certify with an empty calibration set")
    _require(0 <= k <= n, f"observed {k} events in {n} trials")
    _require(0 < delta < 1, "delta must be in (0, 1)")
    if k >= n:
        return 1.0
    lo, hi = k / n, 1.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        cdf = _binomial_cdf(k, n, mid)
        if cdf > delta:
            lo = mid
        else:
            hi = mid
    return hi


def certify_drift(drift_flags: Sequence[int], delta: float = 0.05,
                  n_candidates_certified: int = 1) -> Dict:
    """Exact binomial upper bound on the deployed model's drift rate under i.i.d. calibration.

    `drift_flags` are automatic language-identifier decisions on a calibration
    set DISJOINT from the probe set used for selection. The binomial model
    requires i.i.d. Bernoulli draws, not mere exchangeability. If more than one
    candidate graft is certified, pass `n_candidates_certified` so the level is
    Bonferroni-corrected and the guarantee holds simultaneously.
    """
    flags = np.asarray(list(drift_flags))
    _require(flags.size > 0, "empty calibration set; a certificate needs data")
    _require(set(np.unique(flags)).issubset({0, 1}), "drift flags must be 0/1")
    _require(n_candidates_certified >= 1, "n_candidates_certified must be >= 1")
    n = int(flags.size)
    k = int(flags.sum())
    level = delta / n_candidates_certified
    return {
        "n_calibration": n,
        "n_drift": k,
        "empirical_rate": k / n,
        "upper_bound": clopper_pearson_upper(k, n, level),
        "delta": delta,
        "level_used": level,
        "n_candidates_certified": n_candidates_certified,
    }


# ===========================================================================
# 4.  Statistics for the experiment tables
# ===========================================================================

def bootstrap_ci(x: Sequence[float], n_boot: int = 10000, alpha: float = 0.05,
                 seed: int = 0) -> Tuple[float, float, float]:
    """Percentile bootstrap CI for a mean. Returns (mean, lo, hi)."""
    x = np.asarray(list(x), dtype=float)
    _require(x.size > 1, "bootstrap needs at least two observations")
    rng = np.random.default_rng(seed)
    boots = rng.choice(x, size=(n_boot, x.size), replace=True).mean(axis=1)
    return float(x.mean()), float(np.quantile(boots, alpha / 2)), float(np.quantile(boots, 1 - alpha / 2))


def paired_bootstrap_test(a: Sequence[float], b: Sequence[float],
                          n_boot: int = 10000, seed: int = 0) -> Dict:
    """Two-sided paired bootstrap on the mean difference a - b (item-paired)."""
    a = np.asarray(list(a), dtype=float)
    b = np.asarray(list(b), dtype=float)
    _require(a.shape == b.shape, "paired test requires equal-length item-aligned inputs")
    _require(a.size > 1, "paired bootstrap needs at least two items")
    d = a - b
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, d.size, size=(n_boot, d.size))
    boots = d[idx].mean(axis=1)
    centred = boots - d.mean()
    p = float((np.abs(centred) >= abs(d.mean())).mean())
    return {"mean_diff": float(d.mean()),
            "ci_lo": float(np.quantile(boots, 0.025)),
            "ci_hi": float(np.quantile(boots, 0.975)),
            "p_value": p, "n_items": int(d.size)}


def spearman(x: Sequence[float], y: Sequence[float]) -> float:
    """Spearman rank correlation with average ranks for ties."""
    x = np.asarray(list(x), dtype=float)
    y = np.asarray(list(y), dtype=float)
    _require(x.shape == y.shape and x.size > 1, "spearman needs equal-length inputs of size > 1")

    def rank(v):
        order = np.argsort(v, kind="mergesort")
        r = np.empty(v.size, dtype=float)
        r[order] = np.arange(v.size, dtype=float)
        _, inv, counts = np.unique(v, return_inverse=True, return_counts=True)
        sums = np.zeros(counts.size)
        np.add.at(sums, inv, r)
        return (sums / counts)[inv]

    rx, ry = rank(x), rank(y)
    if rx.std() == 0 or ry.std() == 0:
        raise SutureError("spearman is undefined when one input is constant")
    return float(np.corrcoef(rx, ry)[0, 1])


def linearisation_diagnostic(predicted: Sequence[float], measured: Sequence[float],
                             epsilon: Optional[Sequence[float]] = None) -> Dict:
    """The honesty check to report alongside any SUTURE selection.

    Rank agreement is what selection needs; relative magnitude error is what the
    paper does NOT claim. Report both. If rank agreement is low, the
    linearisation is outside its usable regime for this expert pair and the
    selection should not be trusted.

    On `remainder_loglog_slope`: the theory's bound has a term linear in eps_S
    (coefficient lambda, the donor-host Jacobian gap) and a term quadratic in
    eps_S. Sweeping eps_S by enlarging the graft set holds lambda fixed and
    gives an exponent near 1; sweeping it by shrinking the fine-tuning scale
    moves lambda and eps_S together and gives an exponent near 2. Do not read a
    slope of 1 from a set-size sweep as a violation of the theorem.
    """
    p = np.asarray(list(predicted), dtype=float)
    m = np.asarray(list(measured), dtype=float)
    _require(p.shape == m.shape and p.size > 1, "need equal-length inputs of size > 1")
    denom = np.linalg.norm(m)
    out = {
        "n": int(p.size),
        "spearman": spearman(p, m),
        "pearson": float(np.corrcoef(p, m)[0, 1]),
        "relative_l2_error": float(np.linalg.norm(p - m) / denom) if denom > 0 else float("nan"),
    }
    if epsilon is not None:
        e = np.asarray(list(epsilon), dtype=float)
        _require(e.shape == p.shape, "epsilon must align with the graft sets")
        err = np.abs(p - m)
        keep = (e > 0) & (err > 0)
        if keep.sum() >= 3:
            out["remainder_loglog_slope"] = float(
                np.polyfit(np.log(e[keep]), np.log(err[keep]), 1)[0])
    out["usable_regime"] = bool(out["spearman"] >= 0.85)
    return out


# ===========================================================================
# 5.  Smoke test  (synthetic; demonstrates expected behaviour)
# ===========================================================================

class _SyntheticAdapter(ModelAdapter):
    """A tiny residual stack standing in for a transformer.

    Built so that utility concentrates in the mid-stack and risk collapses near
    the top, which is the depth structure reported in the multilingual
    literature. This is a FIXTURE for testing the code paths. It is not
    evidence about language models and is labelled synthetic everywhere.
    """

    def __init__(self, L=12, d=8, T=4, seed=0):
        rng = np.random.default_rng(seed)
        self._L, self.d, self.T = L, d, T
        self.Wh = [rng.normal(0, 1 / np.sqrt(d), (d, d)) for _ in range(L)]
        self.Wd = [W + 0.12 * rng.normal(0, 1 / np.sqrt(d), (d, d)) for W in self.Wh]
        self.prompts = [rng.normal(0, 1, (T, d)) for _ in range(24)]
        # utility reads mid-stack directions; risk reads late-stack directions
        self.g_util = rng.normal(0, 1, (T, d))
        self.g_risk = rng.normal(0, 1, (T, d))

    @property
    def n_layers(self):
        return self._L

    def _block(self, W, x):
        return np.tanh(x @ W) * 0.5

    def host_block(self, l, x):
        return self._block(self.Wh[l], x)

    def donor_block(self, l, x):
        return self._block(self.Wd[l], x)

    def residual_states(self, prompt):
        xs = [prompt.copy()]
        x = prompt.copy()
        for l in range(self._L):
            x = x + self.host_block(l, x)
            xs.append(x.copy())
        return xs

    def _readout(self, xL, which):
        g = self.g_util if which == "utility" else self.g_risk
        return float((xL * g).sum())

    def adjoints(self, prompt, readout):
        """Exact adjoints by reverse-mode differentiation, done by hand."""
        xs = self.residual_states(prompt)
        g = self.g_util if readout == "utility" else self.g_risk
        acc = g.copy()
        out = [None] * self._L
        for l in range(self._L - 1, -1, -1):
            out[l] = acc.copy()                     # d phi / d x_{l+1}
            W = self.Wh[l]
            pre = xs[l] @ W
            jac_diag = 0.5 * (1 - np.tanh(pre) ** 2)  # elementwise tanh'
            acc = acc + (acc * jac_diag) @ W.T        # through I + J_l
        return out

    # --- ground truth, for the smoke test only -----------------------------
    def measure(self, S, which, prompts):
        """Mean change in the readout over the SAME prompt set the scores were
        averaged over. Averaging the prediction over one set and the ground
        truth over another is a comparison of two different quantities."""
        S = set(S)
        total = 0.0
        for p in prompts:
            x = p.copy()
            for l in range(self._L):
                W = self.Wd[l] if l in S else self.Wh[l]
                x = x + self._block(W, x)
            base = self.residual_states(p)[-1]
            total += self._readout(x, which) - self._readout(base, which)
        return total / len(prompts)


def _smoke() -> int:
    print("=" * 72)
    print("SUTURE metrics smoke test  [SYNTHETIC FIXTURE -- not evidence]")
    print("=" * 72)
    ok = True

    ad = _SyntheticAdapter()
    probe = ad.prompts[:16]
    scores = suture_scores(ad, probe, probe)
    print(f"\n[1] scores computed for L={scores.n_layers} "
          f"from {scores.n_probe_utility} probe examples")
    print("    a^u =", np.array2string(scores.utility, precision=4, suppress_small=True))
    print("    a^r =", np.array2string(scores.risk, precision=4, suppress_small=True))

    # --- additivity of the first-order prediction against ground truth ------
    rng = np.random.default_rng(1)
    preds, meas, eps = [], [], []
    for _ in range(60):
        k = int(rng.integers(1, ad.n_layers + 1))
        S = sorted(rng.choice(ad.n_layers, size=k, replace=False).tolist())
        preds.append(scores.predict(S)[0])
        meas.append(ad.measure(S, "utility", probe))
        eps.append(scores.epsilon(S))
    diag = linearisation_diagnostic(preds, meas, eps)
    print(f"\n[2] linearisation diagnostic over {diag['n']} random graft sets")
    for key in ("spearman", "pearson", "relative_l2_error", "remainder_loglog_slope"):
        if key in diag:
            print(f"    {key:24s} {diag[key]: .4f}")
    print(f"    usable_regime            {diag['usable_regime']}")
    if diag["spearman"] < 0.7:
        print("    !! rank agreement is low on the fixture; investigate before trusting")
        ok = False

    # --- optimiser agreement ------------------------------------------------
    print("\n[3] optimiser checks against brute force")
    mism = 0
    for t in range(200):
        r = np.random.default_rng(t)
        n = int(r.integers(1, 16))
        au, ar = r.normal(0, 1, n), r.normal(0, 1, n)
        tau = float(r.uniform(0, 3))
        bf, bf_arg = -np.inf, None
        for i in range(n):
            u = rr = 0.0
            for j in range(i, n):
                u += au[j]; rr += ar[j]
                if rr >= -tau and u > bf:
                    bf, bf_arg = u, (i, j)
        fast, _ = best_interval(au, ar, tau)
        if bf_arg is None:
            if fast is not None:
                mism += 1
        elif fast is None or abs(bf - fast) > 1e-9:
            mism += 1
    print(f"    interval optimiser vs brute force: {200 - mism}/200 agree")
    ok &= (mism == 0)

    viol = 0
    for t in range(40):
        r = np.random.default_rng(500 + t)
        n = 12
        au, ar = r.normal(0, 1, n), r.normal(0, 1, n)
        tau = float(r.uniform(0, 2))
        _, S, _ = best_subset(au, ar, tau, nbins=3000)
        if S is not None and sum(ar[i] for i in S) < -tau - 1e-9:
            viol += 1
    print(f"    subset DP true-constraint violations: {viol}/40 (expected 0)")
    ok &= (viol == 0)

    # --- certificate --------------------------------------------------------
    print("\n[4] certificate coverage (nominal 0.95)")
    for p_true in (0.02, 0.08):
        r = np.random.default_rng(7)
        cov = sum(certify_drift(r.binomial(1, p_true, 250), 0.05)["upper_bound"] >= p_true
                  for _ in range(600))
        rate = cov / 600
        print(f"    p_true={p_true:.2f}: empirical coverage {rate:.3f}")
        ok &= (rate >= 0.95)

    cert = certify_drift([0] * 194 + [1] * 6, delta=0.05)
    print(f"    example: {cert['n_drift']}/{cert['n_calibration']} drift events "
          f"-> rate {cert['empirical_rate']:.3f}, bound {cert['upper_bound']:.3f}")

    # --- selection end to end ----------------------------------------------
    print("\n[5] end-to-end selection")
    for tau in (0.05, 0.2):
        try:
            sel = select_graft(scores, tau=tau, shape="interval")
            print(f"    tau={tau:<5} interval graft {sel['graft']} "
                  f"pred util {sel['predicted_utility']:+.4f} "
                  f"pred risk {sel['predicted_risk']:+.4f}")
        except SutureError as e:
            print(f"    tau={tau:<5} refused: {e}")

    # --- refusal behaviour --------------------------------------------------
    print("\n[6] the module raises rather than defaulting")
    for label, fn in [
        ("empty probe set", lambda: suture_scores(ad, [], probe)),
        ("empty calibration", lambda: certify_drift([])),
        ("bad drift flags", lambda: certify_drift([0, 1, 2])),
        ("negative tau", lambda: best_interval(np.zeros(3), np.zeros(3), -1.0)),
        ("layer out of range", lambda: scores.predict([99])),
    ]:
        try:
            fn()
            print(f"    !! {label}: DID NOT RAISE")
            ok = False
        except SutureError:
            print(f"    {label}: raised as expected")

    print("\n" + "=" * 72)
    print("SMOKE TEST PASSED" if ok else "SMOKE TEST FAILED")
    print("Reminder: these numbers come from a synthetic fixture and are not")
    print("evidence about language models.")
    print("=" * 72)
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--smoke", action="store_true", help="run the synthetic smoke test")
    ap.add_argument("--scores", type=str, default=None,
                    help="JSON file with 'utility' and 'risk' arrays from a real run")
    ap.add_argument("--tau", type=float, default=0.05)
    ap.add_argument("--shape", type=str, default="interval", choices=["interval", "subset"])
    args = ap.parse_args()

    if args.smoke:
        return _smoke()

    if args.scores:
        if not os.path.exists(args.scores):
            raise SutureError(f"scores file not found: {args.scores}. This tool does "
                              f"not invent scores; run the extraction step first.")
        with open(args.scores) as f:
            d = json.load(f)
        for key in ("utility", "risk", "injection_norm", "n_probe_utility", "n_probe_risk"):
            if key not in d:
                raise SutureError(f"scores file is missing required field '{key}'")
        sc = GraftScores(np.array(d["utility"]), np.array(d["risk"]),
                         np.array(d["injection_norm"]),
                         d["n_probe_utility"], d["n_probe_risk"])
        print(json.dumps(select_graft(sc, args.tau, args.shape), indent=2))
        return 0

    ap.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
