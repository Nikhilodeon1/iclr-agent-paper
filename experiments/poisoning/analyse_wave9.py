"""Applies the Wave 9 rules in preregistered_predictions.md to the result files.

    python analyse_wave9.py <folder with the new CSVs> [--ref <folder with Wave 7 CSVs, default: this folder>]

Reads: results_pg_confirm_s*.csv (PG), results_realtext_pile70m_r*.csv plus results_realtext_pile70m.csv (TK),
results_gradproj_*.csv (GP), results_realtext_410m_s*.csv plus results_realtext_5seed.csv (P4D/P4E).
Missing groups are reported as NOT RUN, never guessed.
"""
import argparse, glob, os, sys
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import analyse_wave7 as a7


def cat(pattern):
    fs = sorted(glob.glob(pattern))
    return (pd.concat([pd.read_csv(f) for f in fs], ignore_index=True), fs) if fs else (None, [])


def analyse_pg(folder):
    d, fs = cat(os.path.join(folder, "results_pg_confirm_s*.csv"))
    print("\n== PG: trigger-row optimizer and the batch-count effect ==")
    if d is None:
        print("NOT RUN"); return None
    d = d.drop_duplicates(["experiment", "seed"]); d["succ"] = d.asr >= 0.5
    print(f"files {len(fs)}, rows {len(d)}, seeds {d.seed.nunique()} ({d.seed.min()}-{d.seed.max()})")
    res = {}
    for arm in ("adamw_trigadamw", "adamw_trigsgd"):
        a = {}
        for E in (1, 16):
            g = d[d.experiment == f"pg_{arm}_N16_E{E}"]
            a[E] = (int(g.succ.sum()), len(g))
        res[arm] = a
        print(f"  {arm}: E=1 {a[1][0]}/{a[1][1]}  E=16 {a[16][0]}/{a[16][1]}")
    def rise(a): return a[16][0] / max(1, a[16][1]) - a[1][0] / max(1, a[1][1])
    ra, rs = rise(res["adamw_trigadamw"]), rise(res["adamw_trigsgd"])
    a = res["adamw_trigadamw"]
    p1 = fisher_exact([[a[16][0], a[16][1] - a[16][0]], [a[1][0], a[1][1] - a[1][0]]], alternative="greater")[1]
    s = res["adamw_trigsgd"]
    p_s = fisher_exact([[s[16][0], s[16][1] - s[16][0]], [s[1][0], s[1][1] - s[1][0]]], alternative="greater")[1]
    p_arm = fisher_exact([[a[16][0], a[16][1] - a[16][0]], [s[16][0], s[16][1] - s[16][0]]], alternative="greater")[1]
    R1 = ra >= 0.25 and p1 < 0.05
    R2 = (ra - rs) >= 0.20
    diff = ra - rs
    verdict = "SUPPORTED" if (R1 and R2) else ("PARTIAL" if (R1 and 0.10 <= diff < 0.20) else "NOT SUPPORTED")
    print(f"  rise under trigger-on-AdamW {ra:+.3f} (one-sided Fisher p={p1:.4f}); under trigger-on-SGD {rs:+.3f} (p={p_s:.4f}); difference {diff:+.3f}")
    print(f"  secondary: E=16 success, trigger-on-AdamW vs trigger-on-SGD, one-sided p={p_arm:.4f}  (3 contrasts reported in total; none corrected)")
    print(f"  R1 {'MET' if R1 else 'NOT MET'}; R2 {'MET' if R2 else 'NOT MET'} -> {verdict}")
    return verdict


LEVELS = [0.0, 0.05, 0.2, 0.5]


def analyse_tk(folder, ref):
    parts = []
    g2, f2 = cat(os.path.join(ref, "results_realtext_pile70m.csv"))
    tk, ftk = cat(os.path.join(folder, "results_realtext_pile70m_r*.csv"))
    print("\n== TK: token identity vs frequency (Pythia-70M, Pile) ==")
    if tk is None:
        print("NOT RUN"); return None
    d = pd.concat([x for x in (g2, tk) if x is not None], ignore_index=True)
    first = d.drop_duplicates("trigger_id")
    d["level"] = d.per_step.apply(lambda p: min(LEVELS, key=lambda L: abs(p - L)))
    rows = []
    for tid, g in d.groupby("trigger_id"):
        rows.append(dict(trigger_id=tid, token=g.trigger.iloc[0], level=g.level.iloc[0], per_step=round(g.per_step.iloc[0], 3),
                         seeds=g.seed.nunique(), n50=a7.n50(g)))
    T = pd.DataFrame(rows).sort_values(["level", "n50"])
    print(T.to_string(index=False))
    med, ncross, spread = {}, {}, {}
    for L in LEVELS:
        v = T[(T.level == L)].n50.dropna().values
        ncross[L] = len(v); med[L] = float(np.median(v)) if len(v) else float("nan")
        spread[L] = float(v.max() / v.min()) if len(v) > 1 else float("nan")
    print("  median N50 by level:", {L: round(med[L], 1) for L in LEVELS}, "| tokens crossing:", ncross, "| max/min:", {L: round(spread[L], 2) for L in LEVELS})
    T1 = all(ncross[L] >= 3 for L in LEVELS) and all(med[LEVELS[i]] < med[LEVELS[i + 1]] for i in range(3))
    cnt = 0
    for i in (1, 2, 3):
        ratio = med[LEVELS[i]] / med[LEVELS[i - 1]]
        ok = spread[LEVELS[i]] > ratio
        cnt += bool(ok)
        print(f"  level {LEVELS[i]}: within-level max/min {spread[LEVELS[i]]:.2f} vs adjacent median ratio {ratio:.2f} -> {'identity as large' if ok else 'frequency step larger'}")
    T2 = cnt >= 2
    out = ("failure at 70M was token idiosyncrasy" if (T1 and not T2) else
           "frequency does not predict thresholds at 70M on the Pile" if ((not T1) and T2) else "mixed")
    print(f"  T1 ordering {'MET' if T1 else 'NOT MET'}; T2 identity {'MET' if T2 else 'NOT MET'} -> {out}")
    return out


def analyse_gp(folder):
    d, fs = cat(os.path.join(folder, "results_gradproj_*.csv"))
    print("\n== GP: gradient decomposition on the trigger embedding row ==")
    if d is None:
        print("NOT RUN"); return None
    d["mq"] = d.m.round(1)
    G = d.groupby("mq").agg(n=("g_cos", "size"), cos_med=("g_cos", "median"), bal_med=("balance", "median"),
                            bal_q25=("balance", lambda x: x.quantile(.25)), bal_q75=("balance", lambda x: x.quantile(.75)),
                            margin=("margin", "mean"), asr=("asr", "mean")).round(3)
    print(f"files {len(fs)}, rows {len(d)}"); print(G.to_string())
    g1 = int((G.cos_med <= -0.5).sum()); g2 = int(G.bal_med.between(0.5, 2).sum()); n = len(G)
    G1 = g1 >= 6 and n >= 7; G2 = g2 >= 5 and n >= 7
    print(f"  G1 opposition: median cos <= -0.5 in {g1} of {n} groups -> {'MET' if G1 else 'NOT MET'} (need 6 of 7)")
    print(f"  G2 balance: median B in [0.5, 2] in {g2} of {n} groups -> {'MET' if G2 else 'NOT MET'} (need 5 of 7)")
    return G1, G2


def analyse_410(folder, ref):
    d, fs = cat(os.path.join(folder, "results_realtext_410m_s*.csv"))
    d0, f0 = cat(os.path.join(ref, "results_realtext_410m_s*.csv"))
    w, _ = cat(os.path.join(ref, "results_realtext_5seed.csv"))
    print("\n== P4D/P4E: Pythia-410M, 5 seeds ==")
    if d is None and d0 is None:
        print("NOT RUN"); return None
    parts = [x for x in (d0, d) if x is not None]
    m = pd.concat(parts, ignore_index=True).drop_duplicates(["trigger_id", "n_poison", "seed"])
    print(f"410M seeds present: {sorted(m.seed.unique())}")
    def by_expected(df):
        return {int(round(g.per_step.iloc[0] * g.n_steps.iloc[0])): a7.n50(g) for _, g in df.groupby("trigger")}
    n410, n70 = by_expected(m), by_expected(w)
    print("  410M:", {k: round(v, 1) for k, v in n410.items()}, "| 70M:", {k: round(v, 1) for k, v in n70.items()})
    ok = True
    for E in (120, 300):
        r = n410[E] / n70[E]; c = r < 0.8; ok &= c
        print(f"  E={E}: 410M/70M = {r:.2f} -> {'below 0.8' if c else 'not below 0.8'}")
    rr = n410[0] / n70[0]; rare = 0.67 <= rr <= 1.5
    print(f"  rare: 410M/70M = {rr:.2f} -> {'within' if rare else 'outside'} [0.67, 1.5]")
    return ok, rare


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("folder"); ap.add_argument("--ref", default=HERE)
    a = ap.parse_args()
    analyse_pg(a.folder); analyse_tk(a.folder, a.ref); analyse_gp(a.folder); analyse_410(a.folder, a.ref)
