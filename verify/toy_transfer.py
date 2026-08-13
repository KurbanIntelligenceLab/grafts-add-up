"""
toy_transfer.py -- a trained, controlled reproduction of the layer-swap setting.

Why this file exists
--------------------
verify_theory.py checks the algebra on a random residual stack. A reviewer is
right to say that random experts prove nothing about grafting. This file
instead TRAINS a small transformer, fine-tunes two real experts from it, and
reproduces the phenomenon layer swapping exploits, in a system small enough
that the exhaustive sweep is ground truth.

The setting mirrors the cross-lingual one exactly:

    vocabulary   digits rendered in two "languages", A and B, plus operators
    base         pretrained on COPY in BOTH languages  (surface form of both)
    donor        fine-tuned on ADD in language A only  ("skill expert")
    host         fine-tuned on COPY in language B only ("language expert")
    target cell  ADD in language B -- seen by NOBODY during training

Neither expert can do the target cell: the donor can add but answers in A, the
host answers in B but cannot add. That is the composition problem.

Two orthogonal readouts at the answer position:
    phi_util  log p(correct VALUE, in either language)   -> arithmetic ability
    phi_risk  log p(language B) - log p(language A)       -> language choice

Granularity. The model is a chain of 2L residual UNITS (attention sublayer,
MLP sublayer). Grafting can be done at layer granularity (units in pairs, what
all prior work does) or at unit granularity. Unit-level contiguous sweeps are
still enumerable here; unit-level SUBSET search is 2^(2L), which is not. That
gap is the point: SUTURE's cost does not depend on the size of the design
space.

NumPy only. Backward passes are written by hand and gradient-checked.
"""

from __future__ import annotations

import argparse
import json
import os
import numpy as np

RNG = np.random.default_rng(0)

# ---------------------------------------------------------------------------
# Task and vocabulary
# ---------------------------------------------------------------------------
NDIG = 12
A_DIG = list(range(0, NDIG))                 # digits in language A
B_DIG = list(range(NDIG, 2 * NDIG))          # digits in language B
TOK_LANG_A = 2 * NDIG
TOK_LANG_B = 2 * NDIG + 1
TOK_COPY = 2 * NDIG + 2
TOK_ADD = 2 * NDIG + 3
TOK_BOS = 2 * NDIG + 4
VOCAB = 2 * NDIG + 5
SEQ = 5                                       # [BOS, LANG, X, Y, OP] -> answer


def render(value: int, lang: str) -> int:
    return A_DIG[value] if lang == "A" else B_DIG[value]


def make_batch(n, out_lang, op, rng, in_script=None):
    """Returns (tokens (n, SEQ), answer value (n,), answer token (n,)).

    `in_script` is the script the OPERANDS are written in and `out_lang` is the
    language the ANSWER must be written in. Decoupling them is what forces the
    base model to build a script-independent representation of a digit's value
    and a separate output-language mechanism, which is the three-phase
    structure (align input, compute, render output) that layer swapping
    exploits in real multilingual models. `in_script=None` samples it at
    random, `out_lang="*"` samples the output language at random.
    """
    x = rng.integers(0, NDIG, n)
    y = rng.integers(0, NDIG, n)
    val = (x + y) % NDIG if op == "ADD" else x
    if out_lang == "*":
        ol = np.where(rng.random(n) < 0.5, "A", "B")
    else:
        ol = np.full(n, out_lang)
    if in_script is None:
        isc = np.where(rng.random(n) < 0.5, "A", "B")
    else:
        isc = np.full(n, in_script)
    toks = np.zeros((n, SEQ), dtype=int)
    toks[:, 0] = TOK_BOS
    toks[:, 1] = [TOK_LANG_A if l == "A" else TOK_LANG_B for l in ol]
    toks[:, 2] = [render(v, s) for v, s in zip(x, isc)]
    toks[:, 3] = [render(v, s) for v, s in zip(y, isc)]
    toks[:, 4] = TOK_ADD if op == "ADD" else TOK_COPY
    ans = np.array([render(v, l) for v, l in zip(val, ol)])
    return toks, val, ans


# ---------------------------------------------------------------------------
# Layers with hand-written backward passes
# ---------------------------------------------------------------------------
def layer_norm(x, eps=1e-5):
    mu = x.mean(-1, keepdims=True)
    xc = x - mu
    var = (xc ** 2).mean(-1, keepdims=True)
    inv = 1.0 / np.sqrt(var + eps)
    return xc * inv, (xc, inv)


def layer_norm_bwd(dy, cache):
    xc, inv = cache
    D = xc.shape[-1]
    dxc = dy * inv
    dinv = (dy * xc).sum(-1, keepdims=True)
    dvar = dinv * (-0.5) * inv ** 3
    dxc = dxc + dvar * (2.0 / D) * xc
    return dxc - dxc.mean(-1, keepdims=True)


def gelu(x):
    return 0.5 * x * (1.0 + np.tanh(0.7978845608 * (x + 0.044715 * x ** 3)))


def gelu_bwd(dy, x):
    t = np.tanh(0.7978845608 * (x + 0.044715 * x ** 3))
    dt = 0.7978845608 * (1 + 3 * 0.044715 * x ** 2) * (1 - t ** 2)
    return dy * (0.5 * (1 + t) + 0.5 * x * dt)


class AttnUnit:
    """Pre-LN single-head causal attention sublayer. Returns the residual update."""
    kind = "attn"

    def __init__(self, d, rng):
        s = 1.0 / np.sqrt(d)
        self.P = {k: rng.normal(0, s, (d, d)) for k in ("Wq", "Wk", "Wv", "Wo")}
        self.d = d

    def forward(self, x):
        h, lc = layer_norm(x)
        q, k, v = h @ self.P["Wq"], h @ self.P["Wk"], h @ self.P["Wv"]
        sc = q @ k.transpose(0, 2, 1) / np.sqrt(self.d)
        mask = np.triu(np.ones((x.shape[1], x.shape[1]), dtype=bool), 1)
        sc = np.where(mask, -1e9, sc)
        sc = sc - sc.max(-1, keepdims=True)
        e = np.exp(sc)
        a = e / e.sum(-1, keepdims=True)
        ctx = a @ v
        return ctx @ self.P["Wo"], (x, h, lc, q, k, v, a, ctx)

    def backward(self, dout, cache, need_grads=True):
        x, h, lc, q, k, v, a, ctx = cache
        g = {}
        if need_grads:
            g["Wo"] = np.einsum("btd,bte->de", ctx, dout)
        dctx = dout @ self.P["Wo"].T
        da = dctx @ v.transpose(0, 2, 1)
        dv = a.transpose(0, 2, 1) @ dctx
        dsc = a * (da - (da * a).sum(-1, keepdims=True))
        dsc /= np.sqrt(self.d)
        dq = dsc @ k
        dk = dsc.transpose(0, 2, 1) @ q
        if need_grads:
            g["Wq"] = np.einsum("btd,bte->de", h, dq)
            g["Wk"] = np.einsum("btd,bte->de", h, dk)
            g["Wv"] = np.einsum("btd,bte->de", h, dv)
        dh = dq @ self.P["Wq"].T + dk @ self.P["Wk"].T + dv @ self.P["Wv"].T
        return layer_norm_bwd(dh, lc), g


class MlpUnit:
    """Pre-LN GELU MLP sublayer. Returns the residual update."""
    kind = "mlp"

    def __init__(self, d, dm, rng):
        self.P = {"W1": rng.normal(0, 1 / np.sqrt(d), (d, dm)),
                  "W2": rng.normal(0, 1 / np.sqrt(dm), (dm, d))}

    def forward(self, x):
        h, lc = layer_norm(x)
        z = h @ self.P["W1"]
        a = gelu(z)
        return a @ self.P["W2"], (h, lc, z, a)

    def backward(self, dout, cache, need_grads=True):
        h, lc, z, a = cache
        g = {}
        if need_grads:
            g["W2"] = np.einsum("btd,bte->de", a, dout)
        da = dout @ self.P["W2"].T
        dz = gelu_bwd(da, z)
        if need_grads:
            g["W1"] = np.einsum("btd,bte->de", h, dz)
        dh = dz @ self.P["W1"].T
        return layer_norm_bwd(dh, lc), g


class Transformer:
    def __init__(self, L=6, d=48, dm=96, seed=0):
        rng = np.random.default_rng(seed)
        self.L, self.d = L, d
        self.E = rng.normal(0, 0.05, (VOCAB, d))
        self.Pos = rng.normal(0, 0.05, (SEQ, d))
        self.units = []
        for _ in range(L):
            self.units.append(AttnUnit(d, rng))
            self.units.append(MlpUnit(d, dm, rng))
        self.U = rng.normal(0, 1 / np.sqrt(d), (d, VOCAB))

    @property
    def n_units(self):
        return len(self.units)

    # --- forward -----------------------------------------------------------
    def forward(self, toks, units=None):
        units = self.units if units is None else units
        x = self.E[toks] + self.Pos[None, :, :]
        states, caches = [x], []
        for u in units:
            upd, c = u.forward(x)
            x = x + upd
            states.append(x)
            caches.append(c)
        hN, lcN = layer_norm(x)
        logits = hN[:, -1, :] @ self.U
        return logits, (states, caches, lcN, hN, toks)

    def logprobs(self, toks, units=None):
        logits, _ = self.forward(toks, units)
        m = logits.max(-1, keepdims=True)
        return logits - m - np.log(np.exp(logits - m).sum(-1, keepdims=True))

    # --- readouts ----------------------------------------------------------
    @staticmethod
    def _lse(lp, idx):
        s = lp[:, idx]
        m = s.max(-1, keepdims=True)
        return (m + np.log(np.exp(s - m).sum(-1, keepdims=True)))[:, 0]

    def phi_util(self, toks, val, units=None):
        """log p(correct value, in either language)."""
        lp = self.logprobs(toks, units)
        out = np.empty(len(val))
        for i, v in enumerate(val):
            out[i] = self._lse(lp[i:i + 1], [A_DIG[v], B_DIG[v]])[0]
        return out

    def phi_risk(self, toks, units=None):
        """log p(language B) - log p(language A) at the answer position."""
        lp = self.logprobs(toks, units)
        return self._lse(lp, B_DIG) - self._lse(lp, A_DIG)

    # --- backward for training (cross-entropy on the answer token) ---------
    def loss_and_grads(self, toks, ans):
        logits, (states, caches, lcN, hN, _) = self.forward(toks)
        m = logits.max(-1, keepdims=True)
        lse = m + np.log(np.exp(logits - m).sum(-1, keepdims=True))
        loss = float((lse[:, 0] - logits[np.arange(len(ans)), ans]).mean())
        p = np.exp(logits - lse)
        dlogits = p.copy()
        dlogits[np.arange(len(ans)), ans] -= 1.0
        dlogits /= len(ans)

        gU = hN[:, -1, :].T @ dlogits
        dhN = np.zeros_like(hN)
        dhN[:, -1, :] = dlogits @ self.U.T
        dx = layer_norm_bwd(dhN, lcN)

        gunits = [None] * self.n_units
        for i in range(self.n_units - 1, -1, -1):
            dxi, g = self.units[i].backward(dx, caches[i], need_grads=True)
            gunits[i] = g
            dx = dx + dxi
        gE = np.zeros_like(self.E)
        np.add.at(gE, toks, dx)
        gPos = dx.sum(0)
        return loss, {"E": gE, "Pos": gPos, "U": gU, "units": gunits}

    # --- adjoints: d phi / d x_{l+1} for every unit (ONE backward pass) ----
    def adjoints(self, toks, readout, val=None):
        logits, (states, caches, lcN, hN, _) = self.forward(toks)
        m = logits.max(-1, keepdims=True)
        lse = m + np.log(np.exp(logits - m).sum(-1, keepdims=True))
        lp = logits - lse
        p = np.exp(lp)
        n = len(toks)

        if readout == "util":
            assert val is not None
            dlogits = np.zeros_like(logits)
            for i, v in enumerate(val):
                idx = [A_DIG[v], B_DIG[v]]
                w = np.exp(lp[i, idx] - self._lse(lp[i:i + 1], idx)[0])
                dlogits[i, idx] += w
                dlogits[i, :] -= p[i, :] * w.sum()
        else:
            dlogits = np.zeros_like(logits)
            for sign, idx in ((1.0, B_DIG), (-1.0, A_DIG)):
                base = self._lse(lp, idx)
                for i in range(n):
                    w = np.exp(lp[i, idx] - base[i])
                    dlogits[i, idx] += sign * w
                    dlogits[i, :] -= sign * p[i, :] * w.sum()

        dhN = np.zeros_like(hN)
        dhN[:, -1, :] = dlogits @ self.U.T
        dx = layer_norm_bwd(dhN, lcN)
        adj = [None] * self.n_units
        for i in range(self.n_units - 1, -1, -1):
            adj[i] = dx.copy()                    # d phi / d x_{i+1}
            dxi, _ = self.units[i].backward(dx, caches[i], need_grads=False)
            dx = dx + dxi
        return adj, states


def clone(model):
    import copy
    return copy.deepcopy(model)


def compose(host: Transformer, donor: Transformer, S):
    """Units in S come from the donor; the rest from the host."""
    S = set(S)
    return [donor.units[i] if i in S else host.units[i]
            for i in range(host.n_units)]


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train(model, sampler, steps, lr=3e-3, bs=256, seed=0, tag=""):
    rng = np.random.default_rng(seed)
    state = {}

    def adam(name, param, grad, t):
        m, v = state.setdefault(name, (np.zeros_like(param), np.zeros_like(param)))
        m = 0.9 * m + 0.1 * grad
        v = 0.999 * v + 0.001 * grad ** 2
        state[name] = (m, v)
        return param - lr * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)

    for t in range(1, steps + 1):
        toks, val, ans = sampler(bs, rng)
        loss, g = model.loss_and_grads(toks, ans)
        model.E = adam("E", model.E, g["E"], t)
        model.Pos = adam("Pos", model.Pos, g["Pos"], t)
        model.U = adam("U", model.U, g["U"], t)
        for i, u in enumerate(model.units):
            for k in u.P:
                u.P[k] = adam(f"u{i}.{k}", u.P[k], g["units"][i][k], t)
        if t % max(steps // 4, 1) == 0:
            print(f"    [{tag}] step {t:5d}  loss {loss:.4f}")
    return model


def accuracy(model, toks, val, units=None):
    lp = model.logprobs(toks, units)
    pred = lp.argmax(-1)
    correct = np.array([pred[i] in (A_DIG[v], B_DIG[v]) for i, v in enumerate(val)])
    in_B = np.isin(pred, B_DIG)
    return float(correct.mean()), float(in_B.mean()), float((correct & in_B).mean())


# ---------------------------------------------------------------------------
# SUTURE on this model
# ---------------------------------------------------------------------------
def suture_scores(host, donor, probe_util, probe_risk):
    """Per-unit utility and risk scores. One forward + one backward per readout,
    plus donor units evaluated off-path on the host trajectory."""
    L = host.n_units
    a_u = np.zeros(L)
    a_r = np.zeros(L)
    for readout, (toks, val) in (("util", probe_util), ("risk", probe_risk)):
        adj, states = host.adjoints(toks, readout, val)
        for i in range(L):
            hu, _ = host.units[i].forward(states[i])
            du, _ = donor.units[i].forward(states[i])
            v = du - hu
            s = float((adj[i] * v).sum() / len(toks))
            if readout == "util":
                a_u[i] = s
            else:
                a_r[i] = s
    return a_u, a_r


def best_interval(a_u, a_r, tau):
    L = len(a_u)
    best, arg = -np.inf, None
    for i in range(L):
        u = r = 0.0
        for j in range(i, L):
            u += a_u[j]; r += a_r[j]
            if r >= -tau and u > best:
                best, arg = u, (i, j)
    return best, arg


def best_subset(a_u, a_r, tau, nbins=4000):
    L = len(a_u)
    lo = float(a_r[a_r < 0].sum()); hi = float(a_r[a_r > 0].sum())
    span = hi - lo
    if span <= 0:
        S = tuple(i for i in range(L) if a_u[i] > 0)
        return float(a_u[list(S)].sum()) if S else 0.0, S
    step = span / nbins
    w = np.array([-int(np.ceil(-r / step)) if r < 0 else int(np.floor(r / step))
                  for r in a_r], dtype=int)
    off = int(np.ceil(-lo / step)) + 1
    size = off + int(np.ceil(hi / step)) + 2
    tab = np.full((L + 1, size), -np.inf); tab[0, off] = 0.0
    idx = np.arange(size)
    for i in range(L):
        src = idx - int(w[i]); cand = np.full(size, -np.inf)
        ok = (src >= 0) & (src < size)
        cand[ok] = tab[i][src[ok]] + a_u[i]
        tab[i + 1] = np.maximum(tab[i], cand)
    feas = ((idx - off) * step >= -tau) & np.isfinite(tab[L])
    if not feas.any():
        return None, None
    j = int(np.flatnonzero(feas)[np.argmax(tab[L][feas])])
    S = []
    for i in range(L - 1, -1, -1):
        src = j - int(w[i])
        if (tab[i][j] != tab[i + 1][j] and 0 <= src < size and np.isfinite(tab[i][src])
                and abs(tab[i][src] + a_u[i] - tab[i + 1][j]) < 1e-9):
            S.append(i); j = src
    return float(tab[L][j if False else int(np.flatnonzero(feas)[np.argmax(tab[L][feas])])]), tuple(sorted(S))


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------
def main(quick=False):
    out = {}
    L = 5 if quick else 6
    steps_pre = 600 if quick else 2500
    steps_ft = 300 if quick else 1200

    print("=" * 74)
    print("Controlled transfer testbed  [TRAINED, still a toy -- not LLM evidence]")
    print("=" * 74)

    # ---- gradient check ---------------------------------------------------
    print("\n[0] gradient check on the hand-written backward pass")
    gm = Transformer(L=2, d=16, dm=24, seed=3)
    tk, vl, an = make_batch(4, "A", "ADD", np.random.default_rng(0))
    _, g = gm.loss_and_grads(tk, an)
    rng = np.random.default_rng(1)
    worst = 0.0
    for name, arr, ga in [("U", gm.U, g["U"]),
                          ("u0.Wq", gm.units[0].P["Wq"], g["units"][0]["Wq"]),
                          ("u1.W1", gm.units[1].P["W1"], g["units"][1]["W1"]),
                          ("u3.W2", gm.units[3].P["W2"], g["units"][3]["W2"])]:
        for _ in range(4):
            i, j = rng.integers(0, arr.shape[0]), rng.integers(0, arr.shape[1])
            h = 1e-5
            arr[i, j] += h; lp, _ = gm.loss_and_grads(tk, an)
            arr[i, j] -= 2 * h; lm, _ = gm.loss_and_grads(tk, an)
            arr[i, j] += h
            num = (lp - lm) / (2 * h)
            rel = abs(num - ga[i, j]) / (abs(num) + abs(ga[i, j]) + 1e-12)
            worst = max(worst, rel)
    print(f"    worst relative gradient error over 16 coordinates: {worst:.2e}")
    grad_ok = worst < 1e-4
    print(f"    {'PASS' if grad_ok else 'FAIL'}")
    out["grad_check"] = worst

    # ---- base pretraining -------------------------------------------------
    print(f"\n[1] pretrain base: COPY in BOTH languages  (L={L} layers, {2*L} units)")
    base = Transformer(L=L, d=48, dm=96, seed=7)

    def base_sampler(n, rng):
        # operands in a random script, answer in a random language: forces a
        # script-independent value representation and a separate output gate
        return make_batch(n, "*", "COPY", rng, in_script=None)

    train(base, base_sampler, steps_pre, seed=1, tag="base")

    # ---- experts ----------------------------------------------------------
    print("\n[2] fine-tune the two experts from the shared base")
    # Skill expert. Operands come in either script, mirroring a reasoning
    # expert that inherits multilingual input reading from the base and whose
    # fine-tuning specialises the OUTPUT language. This is the situation
    # reported for long chain-of-thought specialists, where the English
    # reasoner sits on a multilingual base.
    donor = clone(base)
    train(donor, lambda n, r: make_batch(n, "A", "ADD", r, in_script=None),
          steps_ft, seed=2, tag="donor  ADD/*->A")
    host = clone(base)           # language expert: reads and writes B, cannot add
    train(host, lambda n, r: make_batch(n, "B", "COPY", r, in_script="B"),
          steps_ft, seed=3, tag="host   COPY/B")

    # Diagnostic: is the donor's ADD circuit script-independent? If it can add
    # B-script operands (answering in A), the skill rides on the shared latent
    # and a graft has something to move. If not, there is nothing to transfer
    # and no selection method can help.
    drng = np.random.default_rng(21)
    dt, dv, _ = make_batch(400, "A", "ADD", drng, in_script="B")
    dc, db, _ = accuracy(donor, dt, dv, donor.units)
    print(f"    donor on ADD with B-script operands, answering in A: "
          f"value acc {dc:.3f}  (chance {1.0/NDIG:.3f})")
    out["donor_cross_script_add"] = dc

    # ---- the target cell nobody was trained on ----------------------------
    rng = np.random.default_rng(11)
    test_toks, test_val, _ = make_batch(400, "B", "ADD", rng, in_script="B")
    print("\n[3] target cell = ADD in language B (unseen by every model)")
    print(f"    {'model':<12}{'value acc':>11}{'answers in B':>15}{'BOTH':>9}")
    for nm, mdl in (("base", base), ("donor", donor), ("host", host)):
        c, b, j = accuracy(mdl, test_toks, test_val, mdl.units)
        print(f"    {nm:<12}{c:>11.3f}{b:>15.3f}{j:>9.3f}")
        out[f"acc_{nm}"] = [c, b, j]

    # ---- exhaustive sweeps (ground truth) ---------------------------------
    n_units = host.n_units
    print(f"\n[4] exhaustive sweeps over grafts of donor units into the host")
    layer_ivals = [(2 * i, 2 * j + 1) for i in range(L) for j in range(i, L)]
    unit_ivals = [(i, j) for i in range(n_units) for j in range(i, n_units)]
    print(f"    layer-granularity contiguous windows : {len(layer_ivals)}")
    print(f"    unit-granularity  contiguous windows : {len(unit_ivals)}")
    print(f"    unit-granularity  arbitrary subsets  : 2^{n_units} = {2**n_units}"
          f"   <- not enumerable in general")

    def sweep(ivals):
        rows = []
        for (i, j) in ivals:
            S = tuple(range(i, j + 1))
            c, b, jt = accuracy(host, test_toks, test_val, compose(host, donor, S))
            rows.append((S, c, b, jt))
        return rows

    layer_rows = sweep(layer_ivals)
    unit_rows = sweep(unit_ivals)
    best_layer = max(layer_rows, key=lambda r: r[3])
    best_unit = max(unit_rows, key=lambda r: r[3])
    print(f"    best layer window  {best_layer[0]}  joint acc {best_layer[3]:.3f}")
    print(f"    best unit window   {best_unit[0]}  joint acc {best_unit[3]:.3f}")
    out["sweep_best_layer"] = [list(best_layer[0]), best_layer[3]]
    out["sweep_best_unit"] = [list(best_unit[0]), best_unit[3]]

    # ---- SUTURE -----------------------------------------------------------
    print("\n[5] SUTURE: one forward + one backward per readout, no merged model")
    prng = np.random.default_rng(99)
    pu = make_batch(128, "B", "ADD", prng, in_script="B")   # probe, answers computable
    pr = make_batch(128, "B", "COPY", prng, in_script="B")  # probe, unlabelled B text
    a_u, a_r = suture_scores(host, donor, (pu[0], pu[1]), (pr[0], None))
    print("    a^u per unit:", np.array2string(a_u, precision=3, suppress_small=True))
    print("    a^r per unit:", np.array2string(a_r, precision=3, suppress_small=True))
    out["a_util"] = a_u.tolist()
    out["a_risk"] = a_r.tolist()

    # rank agreement against the unit-granularity sweep
    pred = np.array([a_u[i:j + 1].sum() for (i, j) in unit_ivals])
    true = np.array([r[3] for r in unit_rows])

    def spear(x, y):
        rx = np.argsort(np.argsort(x)).astype(float)
        ry = np.argsort(np.argsort(y)).astype(float)
        return float(np.corrcoef(rx, ry)[0, 1])

    sp = spear(pred, true)
    print(f"    Spearman(predicted, swept joint accuracy) over "
          f"{len(unit_ivals)} unit windows: {sp:.3f}")
    out["spearman_units"] = sp

    tau = float(np.percentile(np.abs(a_r), 60))
    _, arg_i = best_interval(a_u, a_r, tau)
    _, S_sub = best_subset(a_u, a_r, tau)
    sel_i = tuple(range(arg_i[0], arg_i[1] + 1)) if arg_i else ()
    print(f"\n[6] selection at tau={tau:.4f}")
    res = {}
    for nm, S in (("SUTURE interval", sel_i), ("SUTURE subset", S_sub)):
        if not S:
            print(f"    {nm}: no feasible graft"); continue
        c, b, j = accuracy(host, test_toks, test_val, compose(host, donor, S))
        res[nm] = (S, j)
        print(f"    {nm:<18}{str(S):<26} value {c:.3f}  in-B {b:.3f}  joint {j:.3f}")
    print(f"    {'sweep best (layer)':<18}{str(best_layer[0]):<26} joint {best_layer[3]:.3f}")
    print(f"    {'sweep best (unit)':<18}{str(best_unit[0]):<26} joint {best_unit[3]:.3f}")
    out["selected"] = {k: [list(v[0]), v[1]] for k, v in res.items()}

    # ---- cost accounting --------------------------------------------------
    npu, npr = len(pu[0]), len(pr[0])
    surrogate_sweep = len(unit_ivals) * (npu + npr)
    suture_cost = 2 * (npu + npr) + (npu + npr)
    print("\n[7] cost, in forward-pass equivalents over the probe set")
    print(f"    unlabelled surrogate sweep (unit windows) : {surrogate_sweep:,}"
          f"  + {len(unit_ivals)} model builds")
    print(f"    SUTURE                                    : {suture_cost:,}"
          f"  + 0 model builds")
    print(f"    ratio                                     : "
          f"{surrogate_sweep / suture_cost:.0f}x")
    out["cost"] = {"surrogate_sweep": surrogate_sweep, "suture": suture_cost}

    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, "toy_transfer_results.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("\nwrote toy_transfer_results.json")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="smaller/faster configuration")
    a = ap.parse_args()
    main(quick=a.quick)
