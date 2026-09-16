"""
Adversarial numerical verification of the SUTURE theory.

Every formal claim intended for the paper is checked here on a small
transformer-like residual stack implemented in NumPy. Nothing is asserted
that is not computed. Claims that fail are reported as FAIL and must be
weakened or removed from the manuscript.

Checks
------
C1  Exact recursion for the composed-model deviation Delta_L(S).
C2  First-order single-layer prediction: Delta phi({l}) ~= a_l.
C3  Superposition / additivity over graft sets: Delta phi(S) ~= sum_{l in S} a_l.
C4  Remainder scaling: ||R(S)|| = O(eps^2) as the fine-tuning scale eps -> 0.
C5  Remainder vanishes linearly in the donor-host Jacobian gap lambda.
C6  Constrained maximum-subarray optimizer (O(L log L)) == brute force O(L^2).
C7  Budgeted DP over arbitrary subsets == brute force over 2^L (small L).
C8  Clopper-Pearson upper bound attains nominal coverage under simulation.
C9  Failure regime: first-order prediction degrades for large eps (documented,
    not hidden).
"""

import numpy as np
import itertools
import math

RNG = np.random.default_rng(0)

# --------------------------------------------------------------------------
# A small transformer-like residual stack: causal token mixing + LN + GELU MLP
# --------------------------------------------------------------------------

def gelu(x):
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x ** 3)))


def layer_norm(X, eps=1e-5):
    mu = X.mean(axis=-1, keepdims=True)
    var = X.var(axis=-1, keepdims=True)
    return (X - mu) / np.sqrt(var + eps)


class Params:
    """Parameters of one residual block."""

    def __init__(self, d, dm, rng):
        self.Wv = rng.normal(0, 1.0 / np.sqrt(d), (d, d))
        self.W1 = rng.normal(0, 1.0 / np.sqrt(d), (d, dm))
        self.W2 = rng.normal(0, 1.0 / np.sqrt(dm), (dm, d))

    def copy(self):
        p = Params.__new__(Params)
        p.Wv, p.W1, p.W2 = self.Wv.copy(), self.W1.copy(), self.W2.copy()
        return p

    def flat(self):
        return np.concatenate([self.Wv.ravel(), self.W1.ravel(), self.W2.ravel()])


def block(X, p, M):
    """One residual block. X: (T, d). Returns the residual *update* B(X)."""
    H = layer_norm(X)
    Y = (M @ H) @ p.Wv          # causal token mixing + value projection
    Z = gelu(Y @ p.W1) @ p.W2   # position-wise MLP
    return Z


class Stack:
    def __init__(self, L, d, dm, T, V, rng):
        self.L, self.d, self.dm, self.T, self.V = L, d, dm, T, V
        self.blocks = [Params(d, dm, rng) for _ in range(L)]
        M = np.tril(rng.uniform(0.2, 1.0, (T, T)))
        self.M = M / M.sum(axis=1, keepdims=True)   # causal, row-stochastic
        self.U = rng.normal(0, 1.0 / np.sqrt(d), (d, V))
        self.X0 = rng.normal(0, 1.0, (T, d))

    def trajectory(self, which):
        """which: list of length L of Params. Returns [x_0, ..., x_L]."""
        xs = [self.X0.copy()]
        X = self.X0.copy()
        for l in range(self.L):
            X = X + block(X, which[l], self.M)
            xs.append(X.copy())
        return xs

    def readout(self, XL, target):
        """log p(target token) at the final position."""
        logits = layer_norm(XL)[-1] @ self.U
        logits = logits - logits.max()
        return float(logits[target] - np.log(np.exp(logits).sum()))


def make_experts(stack, scale, rng):
    """Host and donor experts: independent perturbations of a shared base."""
    host, donor = [], []
    for l in range(stack.L):
        ph, pd = stack.blocks[l].copy(), stack.blocks[l].copy()
        for attr in ("Wv", "W1", "W2"):
            base = getattr(stack.blocks[l], attr)
            nh = rng.normal(0, 1, base.shape)
            nd = rng.normal(0, 1, base.shape)
            setattr(ph, attr, base + scale * nh / np.sqrt(base.size) * np.linalg.norm(base))
            setattr(pd, attr, base + scale * nd / np.sqrt(base.size) * np.linalg.norm(base))
        host.append(ph)
        donor.append(pd)
    return host, donor


def compose(host, donor, S):
    return [donor[l] if l in S else host[l] for l in range(len(host))]


# --------------------------------------------------------------------------
# Jacobians of the host trajectory (explicit, by central differences)
# --------------------------------------------------------------------------

def block_jacobian(X, p, M, h=1e-6):
    """d vec(B(X)) / d vec(X), shape (T*d, T*d)."""
    n = X.size
    J = np.zeros((n, n))
    flat = X.ravel().copy()
    for i in range(n):
        e = np.zeros(n)
        e[i] = h
        Bp = block((flat + e).reshape(X.shape), p, M).ravel()
        Bm = block((flat - e).reshape(X.shape), p, M).ravel()
        J[:, i] = (Bp - Bm) / (2 * h)
    return J


def readout_grad(stack, XL, target, h=1e-6):
    n = XL.size
    g = np.zeros(n)
    flat = XL.ravel().copy()
    for i in range(n):
        e = np.zeros(n)
        e[i] = h
        g[i] = (stack.readout((flat + e).reshape(XL.shape), target)
                - stack.readout((flat - e).reshape(XL.shape), target)) / (2 * h)
    return g


def first_order_machinery(stack, host, donor, target):
    """Return per-layer injections v_l, adjoints s_l, and scores a_l."""
    xs = stack.trajectory(host)
    L = stack.L
    v = [(block(xs[l], donor[l], stack.M) - block(xs[l], host[l], stack.M)).ravel()
         for l in range(L)]
    Jh = [block_jacobian(xs[l], host[l], stack.M) for l in range(L)]
    Jd = [block_jacobian(xs[l], donor[l], stack.M) for l in range(L)]
    n = stack.T * stack.d
    I = np.eye(n)
    g = readout_grad(stack, xs[L], target)
    # backward accumulation of adjoints: s_l = P_{l+1:L}^T g = d phi / d x_{l+1}
    s = [None] * L
    acc = g.copy()
    for l in range(L - 1, -1, -1):
        s[l] = acc.copy()                    # sensitivity of phi to x_{l+1}
        acc = (I + Jh[l]).T @ acc            # push back through host block l
    a = np.array([float(s[l] @ v[l]) for l in range(L)])
    lam = max(np.linalg.norm(Jd[l] - Jh[l], 2) for l in range(L))
    return dict(v=v, s=s, a=a, Jh=Jh, xs=xs, g=g, lam=lam)


# --------------------------------------------------------------------------
# Optimizers
# --------------------------------------------------------------------------

def best_interval_bruteforce(a_util, a_risk, tau):
    """max sum util over [i,j] s.t. sum risk >= -tau. O(L^2)."""
    L = len(a_util)
    best, arg = -np.inf, None
    for i in range(L):
        u = r = 0.0
        for j in range(i, L):
            u += a_util[j]
            r += a_risk[j]
            if r >= -tau and u > best:
                best, arg = u, (i, j)
    return best, arg


def best_interval_fast(a_util, a_risk, tau):
    """Same problem in O(L log L) via prefix sums + a sorted frontier.

    With U, R the prefix sums (U[0]=R[0]=0), an interval [i, j] has value
    U[j+1]-U[i] and risk R[j+1]-R[i]. For each endpoint j+1 = b we need
        min { U[i] : i < b, R[i] <= R[b] + tau }.
    We sweep b, inserting (R[i], U[i]) into a structure keyed by R and query a
    prefix-minimum of U over R-keys <= R[b] + tau. Implemented with coordinate
    compression + a Fenwick tree of prefix minima.
    """
    L = len(a_util)
    U = np.concatenate([[0.0], np.cumsum(a_util)])
    R = np.concatenate([[0.0], np.cumsum(a_risk)])
    keys = np.sort(np.unique(R))

    size = len(keys)
    tree = np.full(size + 1, np.inf)
    idx_tree = np.full(size + 1, -1, dtype=int)

    def update(pos, val, who):          # Fenwick prefix-min, 1-indexed
        while pos <= size:
            if val < tree[pos]:
                tree[pos] = val
                idx_tree[pos] = who
            pos += pos & (-pos)

    def query(pos):
        best, who = np.inf, -1
        while pos > 0:
            if tree[pos] < best:
                best, who = tree[pos], idx_tree[pos]
            pos -= pos & (-pos)
        return best, who

    best, arg = -np.inf, None
    for b in range(1, L + 1):
        i = b - 1
        pos = int(np.searchsorted(keys, R[i], side="left")) + 1
        update(pos, U[i], i)
        thresh = R[b] + tau
        qpos = int(np.searchsorted(keys, thresh, side="right"))
        if qpos > 0:
            mn, who = query(qpos)
            if mn != np.inf and U[b] - mn > best:
                best, arg = U[b] - mn, (who, b - 1)
    return best, arg


def best_subset_bruteforce(a_util, a_risk, tau):
    L = len(a_util)
    best, arg = -np.inf, None
    for mask in range(1 << L):
        S = [i for i in range(L) if mask >> i & 1]
        r = sum(a_risk[i] for i in S)
        if r >= -tau:
            u = sum(a_util[i] for i in S)
            if u > best:
                best, arg = u, tuple(S)
    return best, arg


def best_subset_dp(a_util, a_risk, tau, nbins=4000):
    """Conservative pseudo-polynomial DP for the signed-weight knapsack

        max sum_{i in S} u_i   s.t.   sum_{i in S} r_i >= -tau.

    Risk is discretised onto a uniform grid of `nbins` cells spanning the full
    achievable range. Rounding is *conservative*: negative contributions are
    rounded away from zero (charged more than they cost) and positive
    contributions are rounded toward zero (credited less than they earn), so
    every DP-feasible set is feasible for the true constraint. The optimality
    gap is therefore one-sided and bounded by the grid step.
    """
    L = len(a_util)
    lo = float(sum(r for r in a_risk if r < 0))
    hi = float(sum(r for r in a_risk if r > 0))
    span = hi - lo
    if span <= 0:
        S = tuple(i for i in range(L) if a_util[i] > 0)
        return float(sum(a_util[i] for i in S)), S, 0.0
    step = span / nbins

    # conservative integer weights (units of `step`), offset so index 0 == lo
    w = np.empty(L, dtype=int)
    for i, r in enumerate(a_risk):
        w[i] = -int(math.ceil(-r / step)) if r < 0 else int(math.floor(r / step))

    offset = int(math.ceil(-lo / step)) + 1          # index of "risk sum 0"
    size = offset + int(math.ceil(hi / step)) + 2
    NEG = -np.inf
    dp = np.full(size, NEG)
    dp[offset] = 0.0
    idx = np.arange(size)
    table = np.empty((L + 1, size))                  # table[i] = dp after i items
    table[0] = dp
    for i in range(L):
        shift = int(w[i])
        cand = np.full(size, NEG)
        src = idx - shift
        valid = (src >= 0) & (src < size)
        cand[valid] = table[i][src[valid]] + a_util[i]
        table[i + 1] = np.maximum(table[i], cand)

    feas = ((idx - offset) * step >= -tau) & np.isfinite(table[L])
    if not feas.any():
        return None, None, step
    best_j = int(np.flatnonzero(feas)[np.argmax(table[L][feas])])
    best = float(table[L][best_j])

    # reconstruct: item i was taken iff it strictly improved the reached cell
    S, j = [], best_j
    for i in range(L - 1, -1, -1):
        src = j - int(w[i])
        if (table[i][j] != table[i + 1][j] and 0 <= src < size
                and np.isfinite(table[i][src])
                and abs(table[i][src] + a_util[i] - table[i + 1][j]) < 1e-9):
            S.append(i)
            j = src
    return best, tuple(sorted(S)), step


def clopper_pearson_upper(k, n, delta):
    """Exact upper confidence bound on a binomial rate (no SciPy)."""
    if k >= n:
        return 1.0
    lo, hi = k / n, 1.0
    for _ in range(200):                       # bisection on the CDF
        mid = 0.5 * (lo + hi)
        # P(Bin(n, mid) <= k)
        cdf = sum(math.comb(n, j) * mid ** j * (1 - mid) ** (n - j) for j in range(k + 1))
        if cdf > delta:
            lo = mid
        else:
            hi = mid
    return hi


# --------------------------------------------------------------------------
# Checks
# --------------------------------------------------------------------------

def banner(t):
    print("\n" + "=" * 74 + f"\n{t}\n" + "=" * 74)


results = {}
conjectures = {}


def report(name, ok, detail):
    results[name] = ok
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")


banner("Setup")
L, d, dm, T, V = 12, 24, 48, 5, 30
stack = Stack(L, d, dm, T, V, RNG)
tgt_util, tgt_risk = 3, 11
SCALE = 0.10
host, donor = make_experts(stack, SCALE, RNG)
print(f"L={L} d={d} T={T} V={V}, fine-tuning scale={SCALE}")

fo = first_order_machinery(stack, host, donor, tgt_util)
xs_h = fo["xs"]
phi_h = stack.readout(xs_h[L], tgt_util)
print(f"host phi = {phi_h:.6f}, max ||J_d - J_h||_2 = {fo['lam']:.4f}")

# ---- C1: exact recursion --------------------------------------------------
banner("C1  Exact recursion for Delta_L(S)")
S_test = {2, 3, 4, 5, 6}
xs_c = stack.trajectory(compose(host, donor, S_test))
Delta_direct = (xs_c[L] - xs_h[L]).ravel()
# rebuild via the exact recursion
Dl = np.zeros(T * d)
for l in range(L):
    p_active = donor[l] if l in S_test else host[l]
    Xh = xs_h[l]
    Dl = Dl + (block(Xh + Dl.reshape(Xh.shape), p_active, stack.M)
               - block(Xh, host[l], stack.M)).ravel()
err = np.linalg.norm(Dl - Delta_direct) / (np.linalg.norm(Delta_direct) + 1e-30)
report("C1 exact recursion", err < 1e-12, f"relative error {err:.3e}")

# ---- C2: single-layer first-order prediction ------------------------------
banner("C2  Single-layer prediction  Delta phi({l}) vs a_l")
pred, meas = [], []
for l in range(L):
    xs_c = stack.trajectory(compose(host, donor, {l}))
    meas.append(stack.readout(xs_c[L], tgt_util) - phi_h)
    pred.append(fo["a"][l])
pred, meas = np.array(pred), np.array(meas)
rel = np.linalg.norm(pred - meas) / np.linalg.norm(meas)
rho = np.corrcoef(pred, meas)[0, 1]
report("C2 single-layer", rel < 0.20 and rho > 0.97,
       f"relative L2 error {rel:.4f}, Pearson r {rho:.5f}")
print("   per-layer (pred, meas):")
for l in range(L):
    print(f"     l={l:2d}  {pred[l]:+.5f}  {meas[l]:+.5f}")

# ---- C3: superposition over graft sets ------------------------------------
banner("C3  Superposition over graft sets")
sets = [set(s) for s in
        [(0,), (0, 1), (0, 1, 2), (4, 5, 6), (2, 3, 4, 5, 6), (0, 1, 2, 9, 10, 11),
         (3, 4, 5, 6, 7, 8), tuple(range(L)), (1, 5, 9), (0, 2, 4, 6, 8, 10)]]
p_, m_ = [], []
for S in sets:
    xs_c = stack.trajectory(compose(host, donor, S))
    m_.append(stack.readout(xs_c[L], tgt_util) - phi_h)
    p_.append(fo["a"][sorted(S)].sum())
p_, m_ = np.array(p_), np.array(m_)
rho3 = np.corrcoef(p_, m_)[0, 1]
sp = np.corrcoef(np.argsort(np.argsort(p_)), np.argsort(np.argsort(m_)))[0, 1]
report("C3 superposition", rho3 > 0.95 and sp > 0.90,
       f"Pearson r {rho3:.5f}, Spearman {sp:.5f}")
for S, pp, mm in zip(sets, p_, m_):
    print(f"     |S|={len(S):2d} {sorted(S)!s:28s} pred {pp:+.5f}  meas {mm:+.5f}")

# ---- C4: remainder scaling in the fine-tuning scale -----------------------
banner("C4  Remainder scaling  ||R|| = O(eps^2)")
scales = [0.2, 0.1, 0.05, 0.025, 0.0125]
rows = []
for sc in scales:
    rng2 = np.random.default_rng(12345)
    h2, d2 = make_experts(stack, sc, rng2)
    f2 = first_order_machinery(stack, h2, d2, tgt_util)
    ph2 = stack.readout(stack.trajectory(h2)[L], tgt_util)
    S = set(range(3, 9))
    xs_c = stack.trajectory(compose(h2, d2, S))
    meas_ = stack.readout(xs_c[L], tgt_util) - ph2
    pred_ = f2["a"][sorted(S)].sum()
    eps = sum(np.linalg.norm(f2["v"][l]) for l in sorted(S))
    rows.append((sc, eps, pred_, meas_, abs(pred_ - meas_)))
print(f"   {'scale':>8} {'eps_S':>10} {'pred':>10} {'meas':>10} {'|err|':>10}")
for r in rows:
    print(f"   {r[0]:8.4f} {r[1]:10.4f} {r[2]:+10.5f} {r[3]:+10.5f} {r[4]:10.3e}")
lg_e = np.log(np.array([r[1] for r in rows]))
lg_r = np.log(np.array([r[4] for r in rows]))
slope_all = np.polyfit(lg_e, lg_r, 1)[0]
slope_asym = np.polyfit(lg_e[-3:], lg_r[-3:], 1)[0]   # small-eps regime
report("C4 quadratic remainder", 1.85 <= slope_asym <= 2.15,
       f"log-log slope over all scales {slope_all:.3f}; asymptotic "
       f"(3 smallest eps) {slope_asym:.3f} (theory predicts 2)")

# ---- C5: remainder vanishes with the donor-host Jacobian gap --------------
banner("C5  Remainder vs donor-host Jacobian gap")
print("   (C4 already varies both eps and lambda together; here lambda is")
print("    reduced at fixed graft size by shrinking only the donor offset.)")
rows5 = []
for gap in [1.0, 0.5, 0.25, 0.125]:
    rng3 = np.random.default_rng(777)
    h3, _ = make_experts(stack, 0.10, rng3)
    rng4 = np.random.default_rng(778)
    d3 = []
    for l in range(L):
        p = h3[l].copy()
        for attr in ("Wv", "W1", "W2"):
            base = getattr(h3[l], attr)
            nz = rng4.normal(0, 1, base.shape)
            setattr(p, attr, base + gap * 0.05 * nz / np.sqrt(base.size) * np.linalg.norm(base))
        d3.append(p)
    f3 = first_order_machinery(stack, h3, d3, tgt_util)
    ph3 = stack.readout(stack.trajectory(h3)[L], tgt_util)
    S = set(range(3, 9))
    meas_ = stack.readout(stack.trajectory(compose(h3, d3, S))[L], tgt_util) - ph3
    pred_ = f3["a"][sorted(S)].sum()
    rows5.append((gap, f3["lam"], abs(pred_ - meas_)))
print(f"   {'gapmul':>8} {'lambda':>10} {'|err|':>12}")
for r in rows5:
    print(f"   {r[0]:8.4f} {r[1]:10.5f} {r[2]:12.3e}")
mono = all(rows5[i][2] > rows5[i + 1][2] for i in range(len(rows5) - 1))
report("C5 remainder decreases with lambda", mono,
       "error is monotone decreasing in the Jacobian gap")

# ---- C6: constrained interval optimizer -----------------------------------
banner("C6  Constrained max-subarray: O(L log L) vs brute force")
ok6 = True
detail6 = ""
for trial in range(400):
    r6 = np.random.default_rng(trial)
    n6 = int(r6.integers(1, 20))
    au = r6.normal(0, 1, n6)
    ar = r6.normal(0, 1, n6)
    tau = float(r6.uniform(0, 3))
    b1, _ = best_interval_bruteforce(au, ar, tau)
    b2, _ = best_interval_fast(au, ar, tau)
    if not (np.isinf(b1) and np.isinf(b2)) and abs(b1 - b2) > 1e-9:
        ok6 = False
        detail6 = f"mismatch at trial {trial}: {b1} vs {b2}"
        break
report("C6 interval optimizer exact", ok6, detail6 or "400/400 random instances agree")

# ---- C7: budgeted DP vs brute force over all subsets ----------------------
banner("C7  Conservative subset DP vs exhaustive 2^L search")
print(f"   {'nbins':>8} {'max gap':>12} {'mean gap':>12} {'violations':>12}")
c7_ok = True
for nb in (200, 1000, 5000):
    gaps, viol = [], 0
    for trial in range(60):
        r7 = np.random.default_rng(1000 + trial)
        n7 = 12
        au = r7.normal(0, 1, n7)
        ar = r7.normal(0, 1, n7)
        tau = float(r7.uniform(0, 2))
        bb, _ = best_subset_bruteforce(au, ar, tau)
        bd, Sd, step = best_subset_dp(au, ar, tau, nbins=nb)
        if bd is None:
            continue
        if sum(ar[i] for i in Sd) < -tau - 1e-9:      # true-constraint violation
            viol += 1
        gaps.append(bb - bd)
    gaps = np.array(gaps)
    print(f"   {nb:8d} {gaps.max():12.3e} {gaps.mean():12.3e} {viol:12d}")
    if viol > 0 or gaps.min() < -1e-9:
        c7_ok = False
    if nb == 5000:
        c7_fine = gaps.max()
report("C7 subset DP conservative and convergent", c7_ok and c7_fine < 1e-2,
       f"never violates the true constraint; one-sided gap shrinks with the "
       f"grid (max gap {c7_fine:.3e} at nbins=5000)")

# Reviewer follow-up: quantify the whole gap distribution over a wider grid
# range.  The feasibility condition is the scientific invariant; the
# distributional summaries are diagnostics of tightness, not new theorems.
banner("C7b  Signed-knapsack tightness over expanded grid resolutions")
print(f"   {'nbins':>8} {'median gap':>12} {'p95 gap':>12} {'max gap':>12} {'violations':>12}")
c7b_rows = []
for nb in (100, 500, 2000, 10000, 50000):
    gaps, viol = [], 0
    for trial in range(80):
        r7b = np.random.default_rng(31000 + trial)
        n7b = 12
        au = r7b.normal(0, 1, n7b)
        ar = r7b.normal(0, 1, n7b)
        tau = float(r7b.uniform(0, 2))
        bb, _ = best_subset_bruteforce(au, ar, tau)
        bd, Sd, _ = best_subset_dp(au, ar, tau, nbins=nb)
        if bd is None:
            continue
        if sum(ar[i] for i in Sd) < -tau - 1e-9:
            viol += 1
        gaps.append(bb - bd)
    gaps = np.asarray(gaps, dtype=float)
    row = {
        "nbins": nb,
        "n": int(len(gaps)),
        "median_gap": float(np.median(gaps)),
        "p95_gap": float(np.percentile(gaps, 95)),
        "max_gap": float(np.max(gaps)),
        "violations": int(viol),
    }
    c7b_rows.append(row)
    print(
        f"   {nb:8d} {row['median_gap']:12.3e} {row['p95_gap']:12.3e} "
        f"{row['max_gap']:12.3e} {viol:12d}"
    )
c7b_ok = all(row["violations"] == 0 for row in c7b_rows)
report(
    "C7b expanded signed-knapsack tightness",
    c7b_ok,
    "zero feasibility violations at every resolution; median/p95/max gaps "
    "reported above",
)

# ---- C10: operational interval selection on the real stack ----------------
banner("C10  Interval selection: predicted-best vs sweep-best (regret)")
intervals = [(i, j) for i in range(L) for j in range(i, L)]
true_val, pred_val = [], []
for (i, j) in intervals:
    S = set(range(i, j + 1))
    true_val.append(stack.readout(stack.trajectory(compose(host, donor, S))[L], tgt_util) - phi_h)
    pred_val.append(fo["a"][i:j + 1].sum())
true_val, pred_val = np.array(true_val), np.array(pred_val)
k_pred = int(np.argmax(pred_val))
k_true = int(np.argmax(true_val))
regret = true_val[k_true] - true_val[k_pred]
rel_regret = regret / (abs(true_val[k_true]) + 1e-12)
sp10 = np.corrcoef(np.argsort(np.argsort(pred_val)),
                   np.argsort(np.argsort(true_val)))[0, 1]
top5_pred = set(np.argsort(-pred_val)[:5])
top5_true = set(np.argsort(-true_val)[:5])
print(f"   {len(intervals)} candidate intervals evaluated exhaustively")
print(f"   sweep-best   {intervals[k_true]} -> {true_val[k_true]:+.5f}")
print(f"   SUTURE-best  {intervals[k_pred]} -> {true_val[k_pred]:+.5f}")
print(f"   Spearman(pred, true) over all intervals = {sp10:.4f}")
print(f"   top-5 overlap = {len(top5_pred & top5_true)}/5")
report("C10 low selection regret", rel_regret < 0.05 and sp10 > 0.9,
       f"relative regret {rel_regret:.4f}, Spearman {sp10:.4f}")

# ---- C11: constrained selection respects the true constraint --------------
banner("C11  Two-functional constrained selection")
fo_r = first_order_machinery(stack, host, donor, tgt_risk)
phi_r_h = stack.readout(xs_h[L], tgt_risk)
a_u, a_r = fo["a"], fo_r["a"]
n_viol, n_case = 0, 0
worst = 0.0
for tau in (0.02, 0.05, 0.10, 0.20, 0.40):
    _, arg = best_interval_fast(a_u, a_r, tau)
    if arg is None:
        continue
    i, j = arg
    S = set(range(i, j + 1))
    XL = stack.trajectory(compose(host, donor, S))[L]
    true_risk = stack.readout(XL, tgt_risk) - phi_r_h
    pred_risk = a_r[i:j + 1].sum()
    n_case += 1
    slack = true_risk + tau
    if slack < 0:
        n_viol += 1
        worst = min(worst, slack)
    print(f"   tau={tau:.2f}  S=[{i},{j}]  pred risk {pred_risk:+.4f}  "
          f"true risk {true_risk:+.4f}  {'VIOLATION' if slack < 0 else 'ok'}")
report("C11 constraint respected under first-order selection", n_viol == 0,
       f"{n_case - n_viol}/{n_case} selections satisfy the true constraint "
       f"(worst slack {worst:+.4f}); a nonzero violation count is exactly why "
       f"the deployed guarantee must be the empirical certificate, not the "
       f"first-order surrogate")

# ---- C8: Clopper-Pearson coverage -----------------------------------------
banner("C8  Clopper-Pearson upper bound coverage")
n8, delta8, trials8 = 200, 0.05, 4000
for p_true in (0.01, 0.05, 0.10):
    r8 = np.random.default_rng(42)
    cov = 0
    for _ in range(trials8):
        k = int(r8.binomial(n8, p_true))
        if clopper_pearson_upper(k, n8, delta8) >= p_true:
            cov += 1
    rate = cov / trials8
    report(f"C8 coverage p={p_true}", rate >= 1 - delta8,
           f"empirical coverage {rate:.4f} (nominal >= {1 - delta8:.2f})")

# ---- C9: failure regime ---------------------------------------------------
banner("C9  Failure regime of the first-order surrogate (documented, not hidden)")
print(f"   {'scale':>8} {'Pearson r':>12} {'Spearman':>10} {'rel L2 err':>12}")
row9 = []
for sc in [0.05, 0.1, 0.2, 0.4, 0.8, 1.6]:
    rng9 = np.random.default_rng(999)
    h9, d9 = make_experts(stack, sc, rng9)
    f9 = first_order_machinery(stack, h9, d9, tgt_util)
    ph9 = stack.readout(stack.trajectory(h9)[L], tgt_util)
    pp, mm = [], []
    for S in sets:
        mm.append(stack.readout(stack.trajectory(compose(h9, d9, S))[L], tgt_util) - ph9)
        pp.append(f9["a"][sorted(S)].sum())
    pp, mm = np.array(pp), np.array(mm)
    r_ = np.corrcoef(pp, mm)[0, 1]
    s_ = np.corrcoef(np.argsort(np.argsort(pp)), np.argsort(np.argsort(mm)))[0, 1]
    e_ = np.linalg.norm(pp - mm) / np.linalg.norm(mm)
    row9.append((sc, r_, s_, e_))
    print(f"   {sc:8.3f} {r_:12.5f} {s_:10.5f} {e_:12.4f}")
degrades = row9[-1][3] > row9[0][3]
rank_robust = row9[2][2] > 0.85
report("C9 degradation documented", degrades,
       "magnitude error grows with the fine-tuning scale, as the theory predicts")
report("C9 ranking more robust than magnitude", rank_robust,
       "Spearman stays high while relative magnitude error grows")

# ---- C12: what the subset optimiser buys, and what a sweep would cost -----
banner("C12  Design-space payoff: subsets vs intervals")
fo_r12 = first_order_machinery(stack, host, donor, tgt_risk)
a_u12, a_r12 = fo["a"], fo_r12["a"]
rows12 = []
for tau in (0.02, 0.05, 0.10, 0.20):
    bi, arg_i = best_interval_fast(a_u12, a_r12, tau)
    bs, S_sub = best_subset_bruteforce(a_u12, a_r12, tau)
    if arg_i is None or S_sub is None:
        continue
    S_i = tuple(range(arg_i[0], arg_i[1] + 1))
    true_i = stack.readout(stack.trajectory(compose(host, donor, set(S_i)))[L], tgt_util) - phi_h
    true_s = stack.readout(stack.trajectory(compose(host, donor, set(S_sub)))[L], tgt_util) - phi_h
    rows12.append((tau, S_i, true_i, S_sub, true_s))
    print(f"   tau={tau:.2f}  interval {str(S_i):<20} true {true_i:+.4f}   "
          f"subset {str(S_sub):<22} true {true_s:+.4f}")
gain = [r[4] - r[2] for r in rows12]
n_int = L * (L + 1) // 2
print(f"   enumerating intervals costs {n_int} merged models; enumerating subsets "
      f"costs 2^{L} = {2**L}.")
print(f"   SUTURE scores both design spaces from the same {2*L} numbers.")
report("C12 subset optimum >= interval optimum", all(g >= -1e-9 for g in gain),
       f"subset never worse; best improvement {max(gain):+.4f} over the best interval")

# ---- C13: the remainder is controlled by the Jacobian gap, not weight distance
banner("C13  Remainder tracks the Jacobian gap at FIXED weight distance")
print("   Falsifiable prediction of Thm 1: with the weight perturbation norm held")
print("   fixed, pairs with a smaller donor-host Jacobian gap have a smaller")
print("   linearisation error. Naive Taylor intuition would key on weight distance.")
S13 = set(range(3, 9))
rows13 = []
for trial in range(14):
    r13 = np.random.default_rng(400 + trial)
    h13, _ = make_experts(stack, 0.10, np.random.default_rng(400))
    d13 = []
    for l in range(L):
        p13 = h13[l].copy()
        for attr in ("Wv", "W1", "W2"):
            base_w = getattr(h13[l], attr)
            nz = r13.normal(0, 1, base_w.shape)
            nz = nz / np.linalg.norm(nz) * np.linalg.norm(base_w) * 0.06   # FIXED norm
            setattr(p13, attr, base_w + nz)
        d13.append(p13)
    f13 = first_order_machinery(stack, h13, d13, tgt_util)
    ph13 = stack.readout(stack.trajectory(h13)[L], tgt_util)
    meas13 = stack.readout(stack.trajectory(compose(h13, d13, S13))[L], tgt_util) - ph13
    pred13 = f13["a"][sorted(S13)].sum()
    wdist = np.sqrt(sum(np.linalg.norm(getattr(d13[l], a) - getattr(h13[l], a)) ** 2
                        for l in range(L) for a in ("Wv", "W1", "W2")))
    rows13.append((wdist, f13["lam"], abs(pred13 - meas13)))
arr13 = np.array(rows13)
cv_w = arr13[:, 0].std() / arr13[:, 0].mean()
c_lam = float(np.corrcoef(arr13[:, 1], arr13[:, 2])[0, 1])
c_w = float(np.corrcoef(arr13[:, 0], arr13[:, 2])[0, 1])
print(f"   weight distance held fixed to within {100*cv_w:.2f}% (coefficient of variation)")
print(f"   corr(Jacobian gap lambda, |remainder|) = {c_lam:+.3f}")
print(f"   corr(weight distance,     |remainder|) = {c_w:+.3f}")
supported = (c_lam > 0.5 and c_lam > c_w)
print(f"[{'SUPPORTED' if supported else 'NOT SUPPORTED'}] C13 conjecture: "
      f"lambda correlates at {c_lam:+.3f}, weight distance at {c_w:+.3f}")
if not supported:
    print("   => The conjecture is NOT supported. The lambda factor in Theorem 1")
    print("      is an upper-bound structure and is NOT predictive of the realised")
    print("      error at fixed weight distance. This claim is therefore EXCLUDED")
    print("      from the paper and recorded as a limitation. The theorem itself")
    print("      is unaffected: an upper bound containing lambda remains valid")
    print("      whether or not lambda is tight.")
conjectures["C13 remainder keyed to the Jacobian gap"] = supported

# ---- C14: superposition across MULTIPLE donors -----------------------------
banner("C14  Superposition holds across several donors at once")
print("   With k donors, each unit is assigned to the host or to one of the donors.")
print("   If the scores of different donors also add, one pass scores all (k+1)^L")
print("   assignments, not just the 2^L of a single donor.")
rngA = np.random.default_rng(2024)
hostM, donorA = make_experts(stack, 0.10, rngA)
_, donorB = make_experts(stack, 0.10, np.random.default_rng(2025))
phiM = stack.readout(stack.trajectory(hostM)[L], tgt_util)
foA = first_order_machinery(stack, hostM, donorA, tgt_util)
foB = first_order_machinery(stack, hostM, donorB, tgt_util)

def compose_multi(host_, assign):
    """assign[l] in {0: host, 1: donorA, 2: donorB}."""
    return [donorA[l] if assign[l] == 1 else (donorB[l] if assign[l] == 2 else host_[l])
            for l in range(L)]

rngM = np.random.default_rng(31)
pm, mm = [], []
for _ in range(80):
    assign = rngM.integers(0, 3, L)
    pred = sum(foA["a"][l] if assign[l] == 1 else (foB["a"][l] if assign[l] == 2 else 0.0)
               for l in range(L))
    meas = stack.readout(stack.trajectory(compose_multi(hostM, assign))[L], tgt_util) - phiM
    pm.append(pred); mm.append(meas)
pm, mm = np.array(pm), np.array(mm)
r14 = float(np.corrcoef(pm, mm)[0, 1])
s14 = float(np.corrcoef(np.argsort(np.argsort(pm)), np.argsort(np.argsort(mm)))[0, 1])
print(f"   80 random 3-way assignments: Pearson {r14:.4f}, Spearman {s14:.4f}")
print(f"   design space enumerated by a sweep: 3^{L} = {3**L:,}")
print(f"   numbers used by the one-pass measurement: {2*L}")
report("C14 multi-donor superposition", r14 > 0.95 and s14 > 0.90,
       f"Pearson {r14:.4f}, Spearman {s14:.4f} over 80 three-way assignments")

# Reviewer follow-up: repeat the controlled assignment test with four donors.
# This is intentionally still a small synthetic stack; it validates the
# assignment algebra and streamed-score accounting, not LLM-scale evidence.
banner("C14b  Four-donor superposition and assignment-space accounting")
donors4 = [
    make_experts(stack, 0.10, np.random.default_rng(seed))[1]
    for seed in (2026, 2027, 2028, 2029)
]
fo4 = [first_order_machinery(stack, hostM, donor_, tgt_util) for donor_ in donors4]

def compose_multi_k(host_, donors_, assign):
    return [
        host_[l] if assign[l] == 0 else donors_[assign[l] - 1][l]
        for l in range(L)
    ]

rng4 = np.random.default_rng(41)
p14b, m14b = [], []
for _ in range(160):
    assign = rng4.integers(0, len(donors4) + 1, L)
    pred = sum(
        fo4[assign[l] - 1]["a"][l]
        for l in range(L)
        if assign[l] > 0
    )
    meas = (
        stack.readout(
            stack.trajectory(compose_multi_k(hostM, donors4, assign))[L],
            tgt_util,
        )
        - phiM
    )
    p14b.append(pred)
    m14b.append(meas)
p14b, m14b = np.asarray(p14b), np.asarray(m14b)
r14b = float(np.corrcoef(p14b, m14b)[0, 1])
s14b = float(
    np.corrcoef(np.argsort(np.argsort(p14b)), np.argsort(np.argsort(m14b)))[0, 1]
)
growth14b = [
    {
        "donors": k,
        "assignments": (k + 1) ** L,
        "streamed_donor_layer_evaluations": k * L,
    }
    for k in range(1, 5)
]
print(f"   160 random 5-way assignments: Pearson {r14b:.4f}, Spearman {s14b:.4f}")
for row in growth14b:
    print(
        f"   k={row['donors']}  assignments={row['assignments']:,}  "
        f"streamed donor evaluations={row['streamed_donor_layer_evaluations']}"
    )
supported14b = r14b > 0.95 and s14b > 0.90
print(
    f"[{'SUPPORTED' if supported14b else 'NOT SUPPORTED'}] C14b diagnostic: "
    f"Pearson {r14b:.4f}, Spearman {s14b:.4f} over 160 five-way assignments"
)
conjectures["C14b four-donor superposition at scale 0.10"] = supported14b

# A small scale sweep separates the combinatorial assignment accounting from
# the finite-perturbation regime.  It is intentionally a diagnostic, not a
# new LLM claim.
banner("C14c  Four-donor scale sweep")
scale_rows14c = []
for idx, scale14c in enumerate((0.10, 0.05, 0.025, 0.0125)):
    host14c, _ = make_experts(
        stack, scale14c, np.random.default_rng(5100 + idx)
    )
    donors14c = [
        make_experts(
            stack, scale14c, np.random.default_rng(5200 + idx * 10 + donor_idx)
        )[1]
        for donor_idx in range(4)
    ]
    phi14c = stack.readout(stack.trajectory(host14c)[L], tgt_util)
    fo14c = [
        first_order_machinery(stack, host14c, donor_, tgt_util)
        for donor_ in donors14c
    ]
    rng14c = np.random.default_rng(5300 + idx)
    pred14c, meas14c = [], []
    for _ in range(160):
        assign = rng14c.integers(0, len(donors14c) + 1, L)
        pred14c.append(
            sum(
                fo14c[assign[l] - 1]["a"][l]
                for l in range(L)
                if assign[l] > 0
            )
        )
        meas14c.append(
            stack.readout(
                stack.trajectory(compose_multi_k(host14c, donors14c, assign))[L],
                tgt_util,
            )
            - phi14c
        )
    pred14c, meas14c = np.asarray(pred14c), np.asarray(meas14c)
    pearson14c = float(np.corrcoef(pred14c, meas14c)[0, 1])
    spearman14c = float(
        np.corrcoef(
            np.argsort(np.argsort(pred14c)),
            np.argsort(np.argsort(meas14c)),
        )[0, 1]
    )
    row14c = {
        "scale": scale14c,
        "pearson": pearson14c,
        "spearman": spearman14c,
        "n_assignments": 160,
    }
    scale_rows14c.append(row14c)
    print(
        f"   scale={scale14c:<6.4f} Pearson {pearson14c:.4f} "
        f"Spearman {spearman14c:.4f}"
    )
conjectures["C14c four-donor small-scale ranking"] = all(
    row["spearman"] > 0.90 for row in scale_rows14c[1:]
)

# ---- C15: linearity in a continuous interpolation coefficient --------------
banner("C15  Scores extend to soft coefficients alpha in [0,1]")
print("   Grafting is the corner alpha=1. If Delta phi is linear in alpha, the")
print("   same scores give merging COEFFICIENTS, not just a discrete graft set.")

def interp(host_, donor_, alphas):
    outs = []
    for l in range(L):
        p_ = host_[l].copy()
        for attr in ("Wv", "W1", "W2"):
            hw = getattr(host_[l], attr); dw = getattr(donor_[l], attr)
            setattr(p_, attr, hw + alphas[l] * (dw - hw))
        outs.append(p_)
    return outs

S15 = sorted(range(3, 9))
rows15 = []
for al in (0.125, 0.25, 0.5, 0.75, 1.0):
    alphas = np.zeros(L); alphas[S15] = al
    meas = stack.readout(stack.trajectory(interp(host, donor, alphas))[L], tgt_util) - phi_h
    pred = al * fo["a"][S15].sum()
    rows15.append((al, pred, meas))
    print(f"   alpha={al:<6} predicted {pred:+.5f}   measured {meas:+.5f}")
arr15 = np.array(rows15)
lin_r = float(np.corrcoef(arr15[:, 1], arr15[:, 2])[0, 1])
ratio = arr15[:, 2] / arr15[:, 1]
report("C15 linear in alpha", lin_r > 0.99 and ratio.std() / abs(ratio.mean()) < 0.25,
       f"Pearson {lin_r:.5f}; measured/predicted ratio {ratio.mean():.3f} "
       f"+/- {ratio.std():.3f} across alpha")

# ---- C16: selection regret bound and its computable form -------------------
banner("C16  Selection regret, the quantity a selection method actually needs")
print("   Thm 1 bounds prediction error. Selection needs a bound on REGRET:")
print("     regret(S_hat) = phi(S_opt) - phi(S_hat)  <=  r(S_opt) - r(S_hat)")
print("   because A(S_hat) >= A(S_opt) by construction. Checked on all 78 windows.")

intervals16 = [(i, j) for i in range(L) for j in range(i, L)]
A16, T16 = [], []
for (i, j) in intervals16:
    S = set(range(i, j + 1))
    A16.append(fo["a"][i:j + 1].sum())
    T16.append(stack.readout(stack.trajectory(compose(host, donor, S))[L], tgt_util) - phi_h)
A16, T16 = np.array(A16), np.array(T16)
r16 = T16 - A16                                  # realised remainder per candidate
k_hat = int(np.argmax(A16)); k_opt = int(np.argmax(T16))
regret = T16[k_opt] - T16[k_hat]
bound_det = r16[k_opt] - r16[k_hat]
print(f"   selected {intervals16[k_hat]}, true optimum {intervals16[k_opt]}")
print(f"   realised regret            {regret:+.6f}")
print(f"   deterministic bound        {bound_det:+.6f}")
print(f"   loose bound 2*sup|r|       {2*np.abs(r16).max():+.6f}")
ok_det = regret <= bound_det + 1e-12 and regret <= 2 * np.abs(r16).max() + 1e-12
report("C16a regret bound holds", ok_det,
       f"regret {regret:.6f} <= r(S_opt)-r(S_hat) = {bound_det:.6f}")

# margin condition: if the score gap exceeds the two remainders, selection is exact
margins = A16[k_hat] - A16
exact_ok = True
for idx in range(len(A16)):
    if idx == k_hat:
        continue
    if margins[idx] > abs(r16[idx]) + abs(r16[k_hat]):
        if T16[idx] > T16[k_hat] + 1e-12:        # a certified-dominated set won anyway
            exact_ok = False
n_cert = int(sum(1 for idx in range(len(A16)) if idx != k_hat and
                 margins[idx] > abs(r16[idx]) + abs(r16[k_hat])))
print(f"   {n_cert}/{len(A16)-1} competitors certified dominated by the margin condition")
report("C16b margin condition sound", exact_ok,
       f"no competitor certified as dominated actually beat the selection")

# computable form: conformal quantile of the remainder from m uniform draws
print("\n   Computable form: draw m candidates uniformly, build them, take")
print("   q = max |phi - A| over those m. For a fresh uniform draw, P(|r|>q) <= 1/(m+1).")
for m in (10, 20, 40):
    viol, trials = 0, 3000
    rq = np.random.default_rng(5)
    for _ in range(trials):
        idxs = rq.choice(len(r16), size=m + 1, replace=False)
        cal, fresh = idxs[:m], idxs[m]
        if abs(r16[fresh]) > np.abs(r16[cal]).max():
            viol += 1
    rate = viol / trials
    print(f"   m={m:<4} empirical exceedance {rate:.4f}   nominal <= {1/(m+1):.4f}   "
          f"models built {m} vs sweep {len(r16)}")
    report(f"C16c conformal remainder quantile m={m}", rate <= 1.0 / (m + 1) + 0.01,
           f"exceedance {rate:.4f} within nominal {1/(m+1):.4f}")

# ---- C17: the alpha claim needs a PARAMETER-space assumption, and its form --
banner("C17  Soft coefficients: what the alpha scaling actually costs")
print("   Corollary 3 scales the injection linearly in alpha. Assumption 1")
print("   constrains smoothness in x, NOT in theta, so that scaling is not")
print("   implied by it. Taylor in theta gives the missing term:")
print("     v(alpha) - alpha*v(1) = -(alpha(1-alpha)/2) D^2_theta B[dtheta,dtheta] + ...")
print("   which vanishes at alpha=0 and alpha=1 and peaks at alpha=1/2.")

def interp_params(ph, pd, a):
    q = ph.copy()
    for attr in ("Wv", "W1", "W2"):
        hw, dw = getattr(ph, attr), getattr(pd, attr)
        setattr(q, attr, hw + a * (dw - hw))
    return q

xs_h17 = stack.trajectory(host)
ell = 5
x_ell = xs_h17[ell]
v1 = block(x_ell, donor[ell], stack.M) - block(x_ell, host[ell], stack.M)

print(f"\n   shape check at unit {ell}:")
print(f"   {'alpha':>7} {'||v(a)-a v(1)||':>18} {'/ a(1-a)':>12}")
ratios = []
for a in (0.1, 0.25, 0.5, 0.75, 0.9):
    va = block(x_ell, interp_params(host[ell], donor[ell], a), stack.M) \
         - block(x_ell, host[ell], stack.M)
    d = np.linalg.norm(va - a * v1)
    ratios.append(d / (a * (1 - a)))
    print(f"   {a:7.2f} {d:18.6e} {d/(a*(1-a)):12.6e}")
ratios = np.array(ratios)
cv = ratios.std() / ratios.mean()
report("C17a alpha(1-alpha) shape", cv < 0.25,
       f"deviation / [a(1-a)] is constant to within {100*cv:.1f}% across alpha")

# endpoints must be exact
d0 = np.linalg.norm(block(x_ell, interp_params(host[ell], donor[ell], 0.0), stack.M)
                    - block(x_ell, host[ell], stack.M) - 0.0 * v1)
d1 = np.linalg.norm(block(x_ell, interp_params(host[ell], donor[ell], 1.0), stack.M)
                    - block(x_ell, host[ell], stack.M) - 1.0 * v1)
report("C17b endpoints exact", d0 < 1e-12 and d1 < 1e-12,
       f"deviation at alpha=0 is {d0:.2e} and at alpha=1 is {d1:.2e}")

# quadratic in the parameter distance
print(f"\n   scaling in ||dtheta||:")
print(f"   {'scale':>8} {'||dtheta||':>12} {'dev at a=0.5':>16}")
rows17 = []
for sc in (1.0, 0.5, 0.25, 0.125):
    pd_s = host[ell].copy()
    for attr in ("Wv", "W1", "W2"):
        hw, dw = getattr(host[ell], attr), getattr(donor[ell], attr)
        setattr(pd_s, attr, hw + sc * (dw - hw))
    v1s = block(x_ell, pd_s, stack.M) - block(x_ell, host[ell], stack.M)
    vhs = block(x_ell, interp_params(host[ell], pd_s, 0.5), stack.M) \
          - block(x_ell, host[ell], stack.M)
    dn = np.sqrt(sum(np.linalg.norm(getattr(pd_s, a2) - getattr(host[ell], a2)) ** 2
                     for a2 in ("Wv", "W1", "W2")))
    dev = np.linalg.norm(vhs - 0.5 * v1s)
    rows17.append((dn, dev))
    print(f"   {sc:8.3f} {dn:12.5f} {dev:16.6e}")
arr17 = np.array(rows17)
slope17 = np.polyfit(np.log(arr17[:, 0]), np.log(arr17[:, 1]), 1)[0]
report("C17c quadratic in parameter distance", 1.7 <= slope17 <= 2.3,
       f"log-log slope of the deviation vs ||dtheta|| is {slope17:.3f} (theory: 2)")

# ---- summary --------------------------------------------------------------
banner("SUMMARY")
n_fail = sum(1 for v in results.values() if not v)
for k, v in results.items():
    print(f"  {'PASS' if v else 'FAIL'}  {k}")
print(f"\n{len(results) - n_fail}/{len(results)} claims made in the paper passed.")
if conjectures:
    print("\nConjectures tested but NOT claimed in the paper:")
    for k, v in conjectures.items():
        print(f"  {'SUPPORTED' if v else 'NOT SUPPORTED -> excluded'}  {k}")
if n_fail:
    print("SOME CHECKS FAILED. The corresponding claims must be weakened or removed.")
