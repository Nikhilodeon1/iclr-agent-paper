"""Turn pod result CSVs into the tables and contrasts the rebuttal needs.

    python analyse_jobs.py results_beta2.csv results_paramgroup.csv ...

The job type is detected from the `experiment` column. Every table reports success counts
(ASR > 0.5) with Wilson 95% intervals alongside mean ASR, because per-run outcomes are
close to bimodal. Contrasts use two-sided Mann-Whitney unless stated, and are printed
WITHOUT multiplicity correction: the number of contrasts examined is printed so the
reader can correct for it. Nothing here edits the paper.
"""
import re
import sys

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, mannwhitneyu, spearmanr


def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def summarise(df, by):
    rows = []
    for key, g in df.groupby(by, dropna=False):
        k = int((g.asr > 0.5).sum())
        n = len(g)
        lo, hi = wilson(k, n)
        key = key if isinstance(key, tuple) else (key,)
        rows.append(dict(zip(by, key), n=n, succ=k, lo=round(lo, 2), hi=round(hi, 2), mean_asr=round(g.asr.mean(), 3)))
    return pd.DataFrame(rows)


def mw(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if len(a) < 2 or len(b) < 2 or (np.ptp(np.r_[a, b]) == 0):
        return float("nan")
    return float(mannwhitneyu(a, b, alternative="two-sided").pvalue)


def n50(sub):
    """Log-linear interpolated poison count for 50% mean ASR; nan if never crosses."""
    m = sub.groupby("n_poison").asr.mean().sort_index()
    ns, ys = m.index.values, m.values
    for i in range(len(ns) - 1):
        if ys[i] < 0.5 <= ys[i + 1]:
            l0, l1 = np.log(ns[i]), np.log(ns[i + 1])
            return float(np.exp(l0 + (0.5 - ys[i]) / (ys[i + 1] - ys[i]) * (l1 - l0)))
    return float("nan")


def a_beta2(df):
    print("== JOB A: beta2 x spacing (total poison identical in every cell) ==")
    t = summarise(df, ["beta2", "spacing"])
    t["span/memory"] = [round(df[(df.beta2 == b) & (df.spacing == s)].span_over_memory.iloc[0], 3) for b, s in zip(t.beta2, t.spacing)]
    print(t.to_string(index=False))
    print("\nContrast across beta2 WITHIN each spacing (placement identical by construction):")
    nc = 0
    for sp, g in df.groupby("spacing"):
        bs = sorted(g.beta2.unique())
        if len(bs) >= 2:
            lo_, hi_ = g[g.beta2 == bs[0]].asr, g[g.beta2 == bs[-1]].asr
            nc += 1
            print(f"  spacing {sp}: beta2={bs[0]} mean {lo_.mean():.3f} vs beta2={bs[-1]} mean {hi_.mean():.3f}, two-sided MW p={mw(lo_, hi_):.3g}")
    r = spearmanr(df.span_over_memory, df.asr) if df.asr.nunique() > 1 else type('R',(),{'statistic':float('nan'),'pvalue':float('nan')})()
    print(f"\nPooled Spearman(span/memory, ASR) = {r.statistic:.3f} (p={r.pvalue:.3g}); pooled across spacings, so placement-confounded. Contrasts examined: {nc}")


def b_paramgroup(df):
    print("== JOB B: per-parameter-group optimizer test ==")
    df = df.copy()
    df["E"] = df.experiment.str.extract(r"_E(\d+)$").astype(int)
    df["cond"] = df.rest_opt + "/trig-" + df.trigger_opt.astype(str)
    t = summarise(df, ["cond", "n_poison", "E"])
    print(t.to_string(index=False))
    print("\nBatch-count contrast (E=1 vs max E) per condition and total; success counts via Fisher, ASR via MW:")
    nc = 0
    for (c, N), g in df.groupby(["cond", "n_poison"]):
        Es = sorted(g.E.unique())
        if len(Es) < 2:
            continue
        a, b = g[g.E == Es[0]].asr, g[g.E == Es[-1]].asr
        ka, kb = int((a > .5).sum()), int((b > .5).sum())
        pf = fisher_exact([[ka, len(a) - ka], [kb, len(b) - kb]])[1]
        nc += 1
        print(f"  {c:22s} N={N:3d}: E={Es[0]} {ka}/{len(a)} -> E={Es[-1]} {kb}/{len(b)}  Fisher p={pf:.3g}  MW p={mw(a, b):.3g}")
    print(f"\nContrasts examined: {nc} (no multiplicity correction applied).")
    print("CHECK FIRST: does 'adamw/trig-none' (unsplit control) agree with 'adamw/trig-adamw'? If not, the split changed the dynamics.")


def c_lr(df):
    print("== JOB C: learning-rate scaling (Eq. budget predicts N* ~ 1/lr; counting predicts no dependence) ==")
    rows = []
    rng = np.random.default_rng(0)
    for lr, g in df.groupby("lr"):
        est = n50(g)
        boots = []
        for _ in range(500):
            idx = g.groupby(["n_poison"]).sample(frac=1.0, replace=True, random_state=int(rng.integers(1 << 30)))
            boots.append(n50(idx))
        boots = np.array([b for b in boots if not np.isnan(b)])
        ci = (np.percentile(boots, 2.5), np.percentile(boots, 97.5)) if len(boots) > 20 else (np.nan, np.nan)
        rows.append(dict(lr=lr, N50=round(est, 1) if not np.isnan(est) else np.nan, ci_lo=round(ci[0], 1), ci_hi=round(ci[1], 1),
                         n_boot_ok=len(boots), seeds=g.seed.nunique()))
    t = pd.DataFrame(rows)
    print(t.to_string(index=False))
    ok = t.dropna(subset=["N50"])
    if len(ok) >= 2:
        sl = np.polyfit(np.log(ok.lr), np.log(ok.N50), 1)[0]
        print(f"\nlog-log slope of N50 vs lr = {sl:.2f} (theory: -1; frequency counting: 0). Descriptive: {len(ok)} learning rates.")
    else:
        print("\nFewer than two learning rates crossed 50% -> censored; widen the N grid or report censoring.")


def events(df):
    print("== EVENTS: AdamW vs SGD, success counts with Wilson intervals ==")
    df = df.copy()
    df["opt"] = df.experiment.str.extract(r"events_(\w+?)_E")[0]
    df["E"] = df.experiment.str.extract(r"_E(\d+)$").astype(int)
    t = summarise(df, ["opt", "n_poison", "E"])
    print(t.to_string(index=False))
    print("\nContrast E=min vs E=max within optimizer and N:")
    nc = 0
    for (o, N), g in df.groupby(["opt", "n_poison"]):
        Es = sorted(g.E.unique())
        if len(Es) < 2:
            continue
        a, b = g[g.E == Es[0]].asr, g[g.E == Es[-1]].asr
        ka, kb = int((a > .5).sum()), int((b > .5).sum())
        nc += 1
        print(f"  {o:6s} N={N:3d}: E={Es[0]} {ka}/{len(a)} -> E={Es[-1]} {kb}/{len(b)} Fisher p={fisher_exact([[ka, len(a)-ka],[kb, len(b)-kb]])[1]:.3g}")
    print(f"Contrasts examined: {nc} (uncorrected).")


def f_dataset(df):
    print("== DATASET SIZE / TRAINING LENGTH ==")
    print(summarise(df, ["experiment", "n_clean"]).to_string(index=False))
    print("Compare tags: SGD tag tests Proposition 1 under the optimizer it assumes; wd0 tag tests the weight-decay account.")


def h_embdisp(df):
    print("== JOB H: trigger-row displacement per poisoned batch ==")
    df = df.copy()
    df["opt"] = df.rest_opt
    t = df.groupby(["opt", "beta2", "density"], dropna=False).agg(n=("seed", "size"), disp_step=("disp_step", "mean"),
            pred=("pred_disp", "mean"), disp_window=("disp_window", "mean")).reset_index()
    t["ratio_to_pred"] = t.disp_step / t.pred
    t["b2key"] = t.beta2.fillna(-1.0)  # SGD rows have no beta2; NaN keys never match in a lookup
    base = t[t.density == t.density.min()].set_index(["opt", "b2key"]).disp_step
    t["rel_to_k_min"] = [r.disp_step / base.get((r.opt, r.b2key), np.nan) for r in t.itertuples()]
    t = t.drop(columns="b2key")
    print(t.round(5).to_string(index=False))
    print("\nAdamW: disp_step ~flat across k and ratio_to_pred ~1 supports the premise. SGD: rel_to_k_min should grow ~ in proportion to k.")


def main(paths):
    for p in paths:
        df = pd.read_csv(p)
        if "experiment" not in df or df.empty:
            print(p, ": empty or no experiment column"); continue
        e = df.experiment.iloc[0]
        print(f"\n######## {p}  ({len(df)} rows, {df.seed.nunique()} seeds) ########")
        if e.startswith("beta2_"): a_beta2(df)
        elif e.startswith("pg_"): b_paramgroup(df)
        elif e.startswith("lrscale_"): c_lr(df)
        elif e.startswith("events_"): events(df)
        elif e.startswith("dataset_size"): f_dataset(df)
        elif e.startswith("embdisp_"): h_embdisp(df)
        else: print("unrecognised experiment tag:", e)


if __name__ == "__main__":
    main(sys.argv[1:])
