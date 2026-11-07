"""Wave 8 analysis (job set MG, MR): applies the rules P1-P4 in preregistered_predictions.md.

    python analyse_margin.py results_margin_ps*_nc*.csv results_margin_rare_nc*.csv

Prints the informative-condition checks, one row per group, and the four rules. Writes mg_summary.csv.
Rules are computed exactly as written there; nothing is tuned after seeing data.
"""
import re
import sys

import numpy as np
import pandas as pd

RNG = np.random.default_rng(0)
B = 2000


def load(paths):
    d = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    m = d.experiment.str.extract(r"margin_ps([0-9.]+)_nc(\d+)$")
    assert m.notna().all().all(), "unexpected experiment tags"
    d["ps"] = m[0].astype(float)
    d["nc"] = m[1].astype(int)
    d["E"] = d.ps * d.nc
    return d


def interp_cross(x, y, level):
    """First crossing of y (monotone-ish) through `level`, linear in x. NaN if never; -inf if y[0] already above."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    if y[0] >= level:
        return -np.inf
    for i in range(1, len(x)):
        if y[i] >= level > y[i - 1]:
            return x[i - 1] + (level - y[i - 1]) * (x[i] - x[i - 1]) / (y[i] - y[i - 1])
    return np.nan


def n50_from(g):
    """Log-linear interpolation of mean ASR over N to 0.5. -inf: already above at the smallest N."""
    mu = g.groupby("n_poison").asr.mean()
    c = interp_cross(np.log(mu.index.values), mu.values, 0.5)
    return np.exp(c) if np.isfinite(c) else c


def n50_ci(g):
    seeds = sorted(g.seed.unique())
    ns = sorted(g.n_poison.unique())
    tab = {n: g[g.n_poison == n].set_index("seed").asr for n in ns}
    vals = []
    for _ in range(B):
        pick = RNG.choice(seeds, len(seeds))
        mu = [np.mean([tab[n].get(s, np.nan) for s in pick]) for n in ns]
        c = interp_cross(np.log(ns), mu, 0.5)
        if np.isfinite(c):
            vals.append(np.exp(c))
    return (np.percentile(vals, 2.5), np.percentile(vals, 97.5)) if len(vals) > 50 else (np.nan, np.nan)


def r2(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    b, a = np.polyfit(x, y, 1)
    res = y - (a + b * x)
    return 1 - res.var() / y.var(), b


def main(paths):
    d = load(paths)
    com = d[d.ps > 0]
    rare = d[d.ps == 0]
    rows = []
    for (ps, nc), g in com.groupby(["ps", "nc"]):
        E = ps * nc
        g = g.copy()
        g["m"] = g.n_poison / E
        by_m = g.groupby("n_poison").agg(m=("m", "first"), asr=("asr", "mean"), margin=("margin", "mean"))
        lo, hi = by_m.iloc[0], by_m.iloc[-1]
        n50 = n50_from(g)
        ci = n50_ci(g) if np.isfinite(n50) else (np.nan, np.nan)
        m0 = interp_cross(np.log(by_m.m.values), by_m.margin.values, 0.0)
        m0 = np.exp(m0) if np.isfinite(m0) else m0
        per_cell = g.groupby("n_poison").seed.nunique()
        rows.append(dict(ps=ps, nc=nc, E=E, seeds=int(per_cell.max()), cells_complete=bool((per_cell == per_cell.max()).all()),
                         asr_lo=round(lo.asr, 2), asr_hi=round(hi.asr, 2), informative=bool(hi.asr >= 0.5 and lo.asr < 0.5),
                         N50=n50, N50_lo=ci[0], N50_hi=ci[1], N50_over_E=n50 / E, m0=m0,
                         loss_spread=round(g.clean_loss.max() - g.clean_loss.min(), 3)))
    S = pd.DataFrame(rows)
    pd.set_option("display.width", 220)
    print(S.round(3).to_string(index=False))
    S.to_csv("mg_summary.csv", index=False)
    n_inf = int(S.informative.sum())
    print(f"\nINFORMATIVE: groups with the grid bracketing 0.5 = {n_inf} of {len(S)} (need >= 5); "
          f"all cells complete: {bool(S.cells_complete.all())}; max clean-loss spread within a group {S.loss_spread.max()} (need <= 0.1)")
    ok = S.N50_over_E.between(0.67, 1.5)
    print(f"P1 threshold: N50/E within [0.67, 1.5] in {int(ok.sum())} of {len(S)} groups (need >= 6): {'MET' if ok.sum() >= 6 else 'NOT MET'}")
    ok2 = S.m0.apply(lambda v: np.isfinite(v) and 0.5 <= v <= 2)
    print(f"P2 margin sign: zero crossing m0 within [0.5, 2] in {int(ok2.sum())} of {len(S)} groups (need all): {'MET' if ok2.all() else 'NOT MET'}"
          f"  (m0 values: {[round(float(v), 2) if np.isfinite(v) else str(v) for v in S.m0]})")
    x_m = np.log(com.n_poison / com.E)
    x_N = np.log(com.n_poison)
    R_m, b_m = r2(x_m, com.margin)
    R_N, b_N = r2(x_N, com.margin)
    print(f"P3 collapse: R^2 margin~ln m = {R_m:.3f} (slope {b_m:.2f}); margin~ln N = {R_N:.3f}; difference {R_m - R_N:.3f}: "
          f"{'MET' if (R_m >= 0.8 and R_m - R_N >= 0.10) else 'NOT MET'} (need >= 0.8 and >= 0.10)")
    if len(rare):
        rr = {}
        for nc, g in rare.groupby("nc"):
            rr[nc] = n50_from(g)
        lo_n, hi_n = min(rr), max(rr)
        a, b = rr[lo_n], rr[hi_n]
        a_eff = 8.0 if a == -np.inf else a
        b_eff = 8.0 if b == -np.inf else b
        met = (not np.isfinite(b_eff)) is False and b_eff <= 1.5 * a_eff
        print(f"P4 rare: N50 by n_clean {dict((k, (round(float(v), 1) if np.isfinite(v) else str(v))) for k, v in rr.items())}; "
              f"ratio {hi_n}/{lo_n}: {b_eff / a_eff:.2f} (need <= 1.5): {'MET' if met else 'NOT MET'}")
    else:
        print("P4 rare: no rare files given")


if __name__ == "__main__":
    main(sys.argv[1:])
