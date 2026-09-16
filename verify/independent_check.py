"""
Clean-room verification of the paper's mathematics. The checks are derived
from the statements in main.tex rather than from verify_theory.py, so an
implementation bug cannot be shared silently.

The checks independently derive the interpolation remainder constant, preserve
the empty graft in constrained interval selection, and distinguish the
per-competitor conformal guarantee from a simultaneous design-space guarantee.
"""
import numpy as np
from itertools import combinations
rng = np.random.default_rng(0)

# ---------------------------------------------------------------- a toy stack
L, d = 6, 4
def blk(x, W1, W2, b):          # B(x;theta) = W2 tanh(W1 x + b), C^inf in x and theta
    return np.tanh(x @ W1 + b) @ W2
def params(scale, r):
    return [(r.normal(0, scale, (d, d)), r.normal(0, scale, (d, d)), r.normal(0, scale, d))
            for _ in range(L)]
def traj(x0, th):
    xs = [x0]
    for l in range(L):
        xs.append(xs[-1] + blk(xs[-1], *th[l]))
    return xs
def compose(h, dn, S):
    return [dn[l] if l in S else h[l] for l in range(L)]

x0 = rng.normal(size=(1, d))
base = params(0.6, np.random.default_rng(1))
host = [(W1 + 0.02*rng.normal(size=(d,d)), W2 + 0.02*rng.normal(size=(d,d)), b + 0.02*rng.normal(size=d))
        for (W1, W2, b) in base]
donor = [(W1 + 0.02*rng.normal(size=(d,d)), W2 + 0.02*rng.normal(size=(d,d)), b + 0.02*rng.normal(size=d))
         for (W1, W2, b) in base]

xh = traj(x0, host)

# ------------------------------------------------ CHECK 1: the exact recursion
# Delta_{l+1} = Delta_l + 1[l in S] v_l + [B(xh+D;th_a) - B(xh;th_a)]
v = [blk(xh[l], *donor[l]) - blk(xh[l], *host[l]) for l in range(L)]
S = {1, 3, 4}
xs = traj(x0, compose(host, donor, S))
worst = 0.0
for l in range(L):
    D = xs[l] - xh[l]
    a = donor[l] if l in S else host[l]
    lhs = xs[l+1] - xh[l+1]
    rhs = D + (v[l] if l in S else 0) + (blk(xh[l] + D, *a) - blk(xh[l], *a))
    worst = max(worst, np.abs(lhs - rhs).max())
print(f"CHECK 1  exact recursion            max abs error {worst:.3e}   {'OK' if worst<1e-12 else 'FAIL'}")

# --------------------------------------- CHECK 2: superposition, order in eps
def jac(f, x, eps=1e-6):
    x = x.ravel(); n = x.size; J = np.zeros((n, n))
    for i in range(n):
        e = np.zeros(n); e[i] = eps
        J[:, i] = ((f((x+e).reshape(1,-1)) - f((x-e).reshape(1,-1)))/(2*eps)).ravel()
    return J
Jh = [jac(lambda z, l=l: blk(z, *host[l]), xh[l]) for l in range(L)]
def prop(lstart):                      # P_{lstart:L} = (I+J_{L-1})...(I+J_{lstart})
    P = np.eye(d)
    for k in range(L-1, lstart-1, -1):
        P = P @ (np.eye(d) + Jh[k])
    return P
P = [prop(l+1) for l in range(L)]

rows = []
for scale in (0.2, 0.1, 0.05, 0.025):
    hs = [(W1 + scale*rng.normal(size=(d,d)), W2, b) for (W1, W2, b) in base]
    ds = [(W1 + scale*rng.normal(size=(d,d)), W2, b) for (W1, W2, b) in base]
    xhs = traj(x0, hs)
    vv = [blk(xhs[l], *ds[l]) - blk(xhs[l], *hs[l]) for l in range(L)]
    Jhs = [jac(lambda z, l=l: blk(z, *hs[l]), xhs[l]) for l in range(L)]
    def props(ls):
        Q = np.eye(d)
        for k in range(L-1, ls-1, -1):
            Q = Q @ (np.eye(d) + Jhs[k])
        return Q
    Ps = [props(l+1) for l in range(L)]
    DL = traj(x0, compose(hs, ds, S))[L] - xhs[L]
    pred = sum((Ps[l] @ vv[l].ravel()) for l in S)
    eps = sum(np.linalg.norm(vv[l]) for l in S)
    rows.append((eps, np.linalg.norm(DL.ravel() - pred)))
sl = np.polyfit(np.log([r[0] for r in rows]), np.log([r[1] for r in rows]), 1)[0]
print(f"CHECK 2  remainder log-log slope    {sl:.3f} (theory 2)          "
      f"{'OK' if 1.7 < sl < 2.4 else 'FAIL'}")

# ------- CHECK 3: the Corollary-4 constant.  Paper writes L_phi (Lipschitz
# constant of grad phi).  A LINEAR readout has L_phi = 0 but rho != 0.
g = rng.normal(size=d)
phi = lambda x: float(x.ravel() @ g)          # linear: grad is constant, L_phi = 0
def interp(h_, d_, a):
    return [(h[0] + a*(dd[0]-h[0]), h[1] + a*(dd[1]-h[1]), h[2] + a*(dd[2]-h[2]))
            for h, dd in zip(h_, d_)]
alpha = 0.5
th_a = [interp(host, donor, alpha)[l] if l in S else host[l] for l in range(L)]
true = phi(traj(x0, th_a)[L]) - phi(xh[L])
first = alpha * sum(float((P[l].T @ g) @ v[l].ravel()) for l in S)
rho = true - first
print(f"CHECK 3  linear readout: L_phi = 0, but |rho| = {abs(rho):.3e}")
print(f"         paper's bound L_phi*Gamma*sum(...) = 0  ->  "
      f"{'BOUND VIOLATED, constant is wrong' if abs(rho) > 1e-10 else 'ok'}")

# ---- CHECK 4: what the conformal clause does and does NOT give simultaneously
N, trials = 78, 20000
for m in (10, 20, 40):
    r_pop = np.abs(rng.standard_t(4, size=N))          # arbitrary remainder population
    exceed, single = [], 0
    for _ in range(trials):
        idx = rng.permutation(N)
        cal, rest = idx[:m], idx[m:]
        q = r_pop[cal].max()
        exceed.append((r_pop[rest] > q).sum())
        if r_pop[rest[0]] > q:                          # one uniformly drawn competitor
            single += 1
    print(f"CHECK 4  m={m:<3} single-competitor exceedance {single/trials:.4f} "
          f"vs nominal {1/(m+1):.4f} | mean # of the remaining {N-m} exceeding q: "
          f"{np.mean(exceed):.2f} vs (N-m)/(m+1) = {(N-m)/(m+1):.2f}")

# ------------------------- CHECK 5: interval optimiser and the empty-set case
def brute(au, ar, tau):
    best, arg = 0.0, ()                      # empty set is feasible with value 0
    for i in range(len(au)):
        for j in range(i, len(au)):
            if ar[i:j+1].sum() >= -tau:
                if au[i:j+1].sum() > best:
                    best, arg = au[i:j+1].sum(), (i, j)
    return best, arg
bad = 0
for t in range(2000):
    r_ = np.random.default_rng(t)
    au, ar = r_.normal(size=8), r_.normal(size=8)
    b, arg = brute(au, ar, 0.3)
    if arg == () and b == 0.0:
        bad += 1
print(f"CHECK 5  instances where the OPTIMUM is the empty set: {bad}/2000 "
      f"-> the program as written (0<=i<b) cannot return it")
