"""Emit pgfplots data files from the verified harness.

Every number written here is computed by verify_theory.py's machinery. No
value in any figure of the paper is hand-entered.
"""
import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import importlib.util

spec = importlib.util.spec_from_file_location("vt", os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "verify_theory.py"))

# Re-implement the minimal pieces rather than executing the full check suite.
from verify_theory import (Stack, make_experts, compose, first_order_machinery,
                           block, best_interval_fast)   # noqa: E402

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
OUT_DIRS = [
    os.path.join(ROOT, "paper", "figs"),
]
for OUT in OUT_DIRS:
    os.makedirs(OUT, exist_ok=True)
OUT = OUT_DIRS[0]

RNG = np.random.default_rng(0)
L, d, dm, T, V = 12, 24, 48, 5, 30
stack = Stack(L, d, dm, T, V, RNG)
tgt_util, tgt_risk = 3, 11
host, donor = make_experts(stack, 0.10, RNG)
fo = first_order_machinery(stack, host, donor, tgt_util)
xs_h = stack.trajectory(host)
phi_h = stack.readout(xs_h[L], tgt_util)


def w(name, rows, header):
    for directory in OUT_DIRS:
        p = os.path.join(directory, name)
        with open(p, "w") as f:
            f.write(header + "\n")
            for r in rows:
                f.write(" ".join(f"{x:.8g}" for x in r) + "\n")
        print("wrote", p, f"({len(rows)} rows)")


# ---- panel (a): predicted vs measured over many random graft sets ----------
rows = []
rs = np.random.default_rng(7)
seen = set()
for _ in range(120):
    k = int(rs.integers(1, L + 1))
    S = tuple(sorted(rs.choice(L, size=k, replace=False).tolist()))
    if S in seen:
        continue
    seen.add(S)
    meas = stack.readout(stack.trajectory(compose(host, donor, set(S)))[L], tgt_util) - phi_h
    pred = fo["a"][list(S)].sum()
    rows.append((pred, meas))
rows_a = rows
w("fig_scatter.dat", rows_a, "pred meas")
pa = np.array(rows_a)
print(f"  panel a: Pearson r = {np.corrcoef(pa[:,0], pa[:,1])[0,1]:.4f}, n = {len(pa)}")

# ---- panel (b): remainder vs graft magnitude (log-log) --------------------
rows_b = []
for sc in [0.20, 0.14, 0.10, 0.07, 0.05, 0.035, 0.025, 0.0175, 0.0125]:
    rng2 = np.random.default_rng(12345)
    h2, d2 = make_experts(stack, sc, rng2)
    f2 = first_order_machinery(stack, h2, d2, tgt_util)
    ph2 = stack.readout(stack.trajectory(h2)[L], tgt_util)
    S = set(range(3, 9))
    meas = stack.readout(stack.trajectory(compose(h2, d2, S))[L], tgt_util) - ph2
    pred = f2["a"][sorted(S)].sum()
    eps = sum(np.linalg.norm(f2["v"][l]) for l in sorted(S))
    rows_b.append((eps, abs(pred - meas)))
w("fig_remainder.dat", rows_b, "eps err")
pb = np.array(rows_b)
sl = np.polyfit(np.log(pb[:4 - 4 + 5:, 0]), np.log(pb[4:, 1]), 1)[0] if len(pb) > 5 else 0
sl_small = np.polyfit(np.log(pb[-4:, 0]), np.log(pb[-4:, 1]), 1)[0]
print(f"  panel b: asymptotic log-log slope (4 smallest) = {sl_small:.3f}")
# reference line of slope exactly 2 anchored at the smallest point
e0, r0 = pb[-1]
w("fig_remainder_ref.dat", [(e, r0 * (e / e0) ** 2) for e in pb[:, 0]], "eps ref")

# ---- panel (c): validity regime -------------------------------------------
sets = [set(s) for s in
        [(0,), (0, 1), (0, 1, 2), (4, 5, 6), (2, 3, 4, 5, 6), (0, 1, 2, 9, 10, 11),
         (3, 4, 5, 6, 7, 8), tuple(range(L)), (1, 5, 9), (0, 2, 4, 6, 8, 10)]]
rows_c = []
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
    sp = np.corrcoef(np.argsort(np.argsort(pp)), np.argsort(np.argsort(mm)))[0, 1]
    rel = np.linalg.norm(pp - mm) / np.linalg.norm(mm)
    rows_c.append((sc, sp, min(rel, 3.5)))
w("fig_regime.dat", rows_c, "scale spearman relerr")

# ---- panel (d): interval selection ----------------------------------------
intervals = [(i, j) for i in range(L) for j in range(i, L)]
rows_d = []
for (i, j) in intervals:
    S = set(range(i, j + 1))
    tv = stack.readout(stack.trajectory(compose(host, donor, S))[L], tgt_util) - phi_h
    pv = fo["a"][i:j + 1].sum()
    rows_d.append((pv, tv))
w("fig_intervals.dat", rows_d, "pred true")
pd_ = np.array(rows_d)
kp, kt = int(np.argmax(pd_[:, 0])), int(np.argmax(pd_[:, 1]))
w("fig_intervals_pick.dat", [(pd_[kp, 0], pd_[kp, 1])], "pred true")
print(f"  panel d: sweep-best {intervals[kt]} = {pd_[kt,1]:+.5f}; "
      f"SUTURE pick {intervals[kp]} = {pd_[kp,1]:+.5f}; "
      f"Spearman = {np.corrcoef(np.argsort(np.argsort(pd_[:,0])), np.argsort(np.argsort(pd_[:,1])))[0,1]:.4f}")

# ---- per-layer score profiles (utility and risk) --------------------------
fo_r = first_order_machinery(stack, host, donor, tgt_risk)
w("fig_profile.dat", [(l, fo["a"][l], fo_r["a"][l]) for l in range(L)], "layer util risk")
_, arg = best_interval_fast(fo["a"], fo_r["a"], 0.05)
print(f"  profile: constrained pick at tau=0.05 -> {arg}")
with open(os.path.join(OUT_DIRS[0], "fig_profile_pick.tex"), "w") as f:
    f.write(f"{arg[0]}\n{arg[1]}\n")
print("done")
