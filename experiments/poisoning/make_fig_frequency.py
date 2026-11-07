"""Figure: threshold versus expected clean uses, with bootstrap intervals, plus the margin collapse.

    cd experiments/poisoning && python make_fig_frequency.py      (CPU, about 1 minute)

Reads results_margin_*.csv (Wave 8). Writes ../../figures/fig_frequency_scaling.png
"""
import glob, os, sys
import numpy as np, pandas as pd
import matplotlib as mpl, matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import analyse_wave7 as a7
OUT = os.path.abspath(os.path.join(HERE, "..", "..", "figures")); os.makedirs(OUT, exist_ok=True)
mpl.rcParams.update({"font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 6.5, "xtick.labelsize": 7,
                     "ytick.labelsize": 7, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 200})
d = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(os.path.join(HERE, "results_margin_*.csv")))], ignore_index=True)
m = d.experiment.str.extract(r"margin_ps([0-9.]+)_nc(\d+)$"); d["ps"] = m[0].astype(float); d["nc"] = m[1].astype(int)
d["E"] = d.ps * d.nc
BLUE, ORG, GRN, GREY = "#1f5fa8", "#d95f02", "#2a8f5a", "#7f7f7f"
rng = np.random.default_rng(0)

def boot_n50(g, reps=1000):
    seeds = g.seed.unique(); vals = []
    for _ in range(reps):
        pick = rng.choice(seeds, size=len(seeds), replace=True)
        b = pd.concat([g[g.seed == s].assign(seed=i) for i, s in enumerate(pick)])
        v = a7.n50(b)
        if np.isfinite(v): vals.append(v)
    return np.percentile(vals, [2.5, 97.5]) if len(vals) > 50 else (np.nan, np.nan)

fig, ax = plt.subplots(2, 2, figsize=(6.5, 5.2)); (A, B), (C, D) = ax
# (a) threshold vs expected clean uses
rows = []
for (ps, nc), g in d[d.ps > 0].groupby(["ps", "nc"]):
    n50 = a7.n50(g); lo, hi = boot_n50(g); rows.append((ps * nc, n50, lo, hi, ps, nc))
R = pd.DataFrame(rows, columns=["E", "n50", "lo", "hi", "ps", "nc"]).sort_values("E")
xs = np.array([30, 1300.0]); A.plot(xs, xs, "--", color="black", lw=0.9, label="N = expected clean uses")
for ps, mk, col in [(0.05, "s", GRN), (0.1, "o", BLUE), (0.2, "^", ORG)]:
    r = R[R.ps == ps]; A.errorbar(r.E, r.n50, yerr=[r.n50 - r.lo, r.hi - r.n50], fmt=mk, color=col, ms=4.5, capsize=2, lw=0.9, label=f"{ps:g} uses per step")
A.set_xscale("log"); A.set_yscale("log"); A.set_xlabel("expected clean uses of the trigger"); A.set_ylabel("poison needed for 50% success")
A.set_title("a  Threshold tracks expected clean uses", loc="left"); A.legend(frameon=False, loc="upper left")
# (b) ASR collapse vs m
c = d[d.ps > 0].copy(); c["mq"] = c.m.round(2)
for (ps, nc), g in c.groupby(["ps", "nc"]):
    s = g.groupby("mq").asr.mean(); B.plot(s.index, s.values, "-o", ms=2.5, lw=0.8, alpha=0.8, label=f"E={ps*nc:g}")
B.axvline(1, color="black", ls="--", lw=0.8); B.axhline(0.5, color=GREY, ls=":", lw=0.8); B.set_xscale("log")
B.set_xlabel("poison count / expected clean uses  (m)"); B.set_ylabel("attack success rate"); B.set_title("b  Success against m, seven groups", loc="left")
B.legend(frameon=False, ncol=2, loc="lower right", fontsize=5.5)
for a_ in (B,): a_.set_xticks([0.25, 0.5, 1, 2, 4]); a_.set_xticklabels(["0.25", "0.5", "1", "2", "4"]); a_.minorticks_off()
# (c) margin vs ln m
mm = c.groupby(["ps", "nc", "mq"]).margin.mean().reset_index()
for (ps, nc), g in mm.groupby(["ps", "nc"]): C.plot(g.mq, g.margin, "-o", ms=2.5, lw=0.8, alpha=0.8)
xx = np.geomspace(0.22, 4.4, 50); sl, ic = np.polyfit(np.log(c.mq), c.margin, 1)
C.plot(xx, sl * np.log(xx) + ic, color="black", lw=1.2, label=f"fit: slope {sl:.2f} per unit ln m")
C.axvline(1, color="black", ls="--", lw=0.8); C.axhline(0, color=GREY, ls=":", lw=0.8); C.set_xscale("log")
C.set_xlabel("poison count / expected clean uses  (m)"); C.set_ylabel("logit margin (target minus benign)"); C.set_title("c  Margin crosses zero at m near 1", loc="left"); C.legend(frameon=False, loc="upper left")
C.set_xticks([0.25, 0.5, 1, 2, 4]); C.set_xticklabels(["0.25", "0.5", "1", "2", "4"]); C.minorticks_off()
# (d) rare trigger
rr = d[d.ps == 0]
for nc, col in [(1000, GRN), (3000, BLUE), (9000, ORG)]:
    g = rr[rr.nc == nc]; s = g.groupby("n_poison").asr.mean(); D.plot(s.index, s.values, "-o", ms=3, color=col, lw=1, label=f"{nc:,} clean steps")
D.axhline(0.5, color=GREY, ls=":", lw=0.8); D.set_xscale("log"); D.set_xlabel("poisoned examples"); D.set_ylabel("attack success rate")
D.set_title("d  Absent trigger: no growth with clean steps", loc="left"); D.legend(frameon=False, loc="lower right")
D.set_xticks([8, 16, 25, 40, 64]); D.set_xticklabels(["8", "16", "25", "40", "64"]); D.minorticks_off()
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_frequency_scaling.png")); print(R.round(2).to_string(index=False)); print("slope",round(sl,2))
