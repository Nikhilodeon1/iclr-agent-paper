"""Analyse --experiment decomp CSVs (pooled): python analyse_decomp.py file1.csv file2.csv ...

Reports per-cell success counts, mean ASR and trigger-row displacement, then the three
paired contrasts the pre-written rules need (seeds are paired across cells: the same
torch seed and data stream index every cell). All tests one-sided and uncorrected.
"""
import re, sys
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

TAG = re.compile(r"decomp_(\w+?)_B(\d+)_A(\d+)_S(\d+)$")


def load(paths):
    d = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    m = d.experiment.str.extract(TAG)
    d = d[m[0].notna()].copy()
    d["opt"], d["B"], d["A"], d["S"] = m[0], m[1].astype(int), m[2].astype(int), m[3].astype(int)
    dup = int(d.duplicated(["opt", "B", "A", "S", "seed"]).sum())
    if dup:
        print(f"WARNING: {dup} duplicate (cell, seed) rows; keeping the first")
        d = d.drop_duplicates(["opt", "B", "A", "S", "seed"])
    return d


def paired(d, a, b, key=("B", "A", "S")):
    """ASR(cell a) - ASR(cell b) per seed. a, b are dicts over key names."""
    def sel(c):
        q = d
        for k, v in c.items():
            q = q[q[k] == v]
        return q.set_index("seed").asr
    x, y = sel(a), sel(b)
    s = x.index.intersection(y.index)
    diff = (x[s] - y[s])
    if len(s) < 3:
        return len(s), np.nan, np.nan, 0, 0
    try:
        p = wilcoxon(x[s], y[s], alternative="greater").pvalue
    except ValueError:
        p = np.nan
    return len(s), float(diff.mean()), float(p), int((diff > 0).sum()), int((diff < 0).sum())


def fmt(name, r):
    n, m, p, up, dn = r
    if n < 3:
        return f"  {name:46s} n={n}: too few paired seeds"
    return f"  {name:46s} n={n:2d} mean drop {m:+.3f}  one-sided Wilcoxon p={p:.3g}  seeds down {up}, up {dn}"


def main(paths):
    d = load(paths)
    print(f"rows {len(d)}; cells {d.groupby(['opt','B','A','S']).ngroups}; seeds/cell {sorted(d.groupby(['opt','B','A','S']).seed.nunique().unique())}")
    t = d.groupby(["opt", "B", "A", "S"]).agg(n=("asr", "size"), succ=("asr", lambda s: int((s > 0.5).sum())),
                                              asr=("asr", "mean"), disp_dose=("disp_step", "mean"),
                                              disp_end=("disp_window", "mean"), loss=("clean_loss", "mean")).round(3)
    t["end/dose"] = (t.disp_end / t.disp_dose).round(2)
    print(t.to_string())
    mid = t[(t.asr >= 0.15) & (t.asr <= 0.85)]
    if mid.empty:
        print("\nUNINFORMATIVE: no cell has mean ASR in [0.15, 0.85] (floor or ceiling)")
    s0 = d.S.min() if (d.S == 700).sum() == 0 else 700
    print(f"\n== A effect: post-dose training (spacing {s0}); positive drop = ASR falls as A grows ==")
    for B in sorted(d.B.unique()):
        As = sorted(d[(d.B == B) & (d.S == s0)].A.unique())
        if len(As) >= 2:
            print(fmt(f"B={B}: A={As[0]} vs A={As[-1]}", paired(d, dict(B=B, A=As[0], S=s0), dict(B=B, A=As[-1], S=s0))))
    print(f"\n== B effect: pre-dose training (spacing {s0}) ==")
    for A in sorted(d[d.S == s0].A.unique()):
        Bs = sorted(d[(d.A == A) & (d.S == s0)].B.unique())
        if len(Bs) >= 2:
            print(fmt(f"A={A}: B={Bs[0]} vs B={Bs[-1]}", paired(d, dict(B=Bs[0], A=A, S=s0), dict(B=Bs[-1], A=A, S=s0))))
    print("\n== S effect: spacing between doses (B=0, A=0) ==")
    Ss = sorted(d[(d.B == 0) & (d.A == 0)].S.unique())
    for s in Ss[1:]:
        print(fmt(f"S={Ss[0]} vs S={s}", paired(d, dict(B=0, A=0, S=Ss[0]), dict(B=0, A=0, S=s))))
    print("\nTrigger-row displacement ratio end/dose < 1 means the trigger's OWN embedding row moves back "
          "after the last dose; ~1 or above means the row holds and any erosion is downstream.")


if __name__ == "__main__":
    main(sys.argv[1:])
