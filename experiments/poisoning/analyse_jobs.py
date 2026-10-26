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
    """Training-length sweep: per-length success counts, validity check, decline fraction D."""
    print("== DATASET SIZE / TRAINING LENGTH ==")
    for tag, g in df.groupby("experiment"):
        N = int(g.n_poison.iloc[0])
        t = g.groupby("n_clean").agg(n=("asr", "size"), succ=("asr", lambda s: int((s > 0.5).sum())),
                                      mean=("asr", "mean")).round(3)
        print(f"-- {tag}  (N={N})"); print(t.to_string())
        m = g.groupby("n_clean").asr.mean()
        mid = [T for T, v in m.items() if 0.15 <= v <= 0.85]
        if not mid:
            print("   UNINFORMATIVE: no length has mean ASR in [0.15, 0.85] (floor or ceiling)"); continue
        peak = max(m.get(2000, np.nan), m.get(8000, np.nan))
        late = m.get(32000, np.nan)
        if np.isfinite(peak) and peak > 0 and np.isfinite(late):
            D = 1 - late / peak
            a_ = g[g.n_clean == 32000].asr
            b_ = g[g.n_clean == (2000 if m.get(2000, 0) >= m.get(8000, 0) else 8000)].asr
            p_ = mw(b_, a_)
            print(f"   mid-range lengths: {mid}; peak(2000/8000)={peak:.3f}, T=32000={late:.3f}, "
                  f"decline D={D:.2f}, drop MW p={p_:.3g}, abs drop={peak - late:.3f}")
    print("Compare: SGD files test Proposition 1 under the optimizer it assumes; compare D(wd0) with D(ctrl) "
          "per the rule in preregistered_predictions.md.")


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


def n50_ci(sub, boot=2000, seed=0):
    """N* with a bootstrap CI over seeds (the seeds are the resampling unit)."""
    rng = np.random.default_rng(seed)
    seeds = sub.seed.unique()
    pt = n50(sub)
    draws = []
    for _ in range(boot):
        pick = rng.choice(seeds, size=len(seeds), replace=True)
        rs = pd.concat([sub[sub.seed == s] for s in pick])
        v = n50(rs)
        if np.isfinite(v):
            draws.append(v)
    if len(draws) < boot * 0.5:
        return pt, float("nan"), float("nan")
    return pt, float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def threshold(df):
    """N* against whichever factor the sweep varied, with a log-log slope."""
    axis = df.experiment.iloc[0].split("_")[1]
    xcol = {"modelsize": "n_params", "density": "density", "lr": "lr", "batch": "batch"}[axis]
    print(f"== THRESHOLD sweep over {axis} (x = {xcol}) ==")
    rows = []
    for x, g in df.groupby(xcol):
        pt, lo, hi = n50_ci(g)
        cl = g.clean_loss.mean() if "clean_loss" in g else float("nan")
        asr_at = g.groupby("n_poison").asr.mean()
        rows.append(dict(x=x, n_star=pt, lo=lo, hi=hi, seeds=g.seed.nunique(),
                         clean_loss=round(cl, 3), min_asr=round(asr_at.min(), 2),
                         max_asr=round(asr_at.max(), 2)))
    t = pd.DataFrame(rows).sort_values("x")
    print(t.to_string(index=False, float_format=lambda v: f"{v:.3g}"))
    cens = t[~np.isfinite(t.n_star)]
    if len(cens):
        print(f"  !! {len(cens)} cell(s) never cross 50% ASR inside the poison grid: "
              f"N* is censored there and the slope below ignores them.")
    ok = t[np.isfinite(t.n_star) & (t.x > 0)]
    if len(ok) >= 2:
        sl = np.polyfit(np.log(ok.x.astype(float)), np.log(ok.n_star), 1)[0]
        pred = {"modelsize": "0 (no model-size term in either proposition)",
                "density": "1 (Prop 3: N* = k x E*); 0 if poison examples are what count",
                "lr": "-1 (Eq 4: budget proportional to 1/eta); 0 under frequency counting",
                "batch": "unpredicted"}[axis]
        print(f"  log-log slope of N* vs {xcol} = {sl:.2f}   (predicted: {pred})")
        if axis == "modelsize":
            rng_ = ok.x.max() / ok.x.min()
            print(f"  parameter range covered: {rng_:.0f}x ({ok.x.min():,.0f} to {ok.x.max():,.0f}); "
                  f"N* range {ok.n_star.min():.1f} to {ok.n_star.max():.1f} "
                  f"({ok.n_star.max()/ok.n_star.min():.2f}x)")
    if "clean_loss" in df and df.clean_loss.notna().any():
        print("  CHECK: clean_loss should be comparable across cells; a cell that failed to "
              "learn the clean task cannot be compared on attack success.")


def interference2(df):
    """Corrected interference test: beta2 effect at matched per-event step size."""
    print("== INTERFERENCE (corrected: trigger-row lr compensated, events placed late) ==")
    print("Step-matching check -- disp_step/disp_window should be comparable across beta2")
    print("within a spacing; pred_disp is identical by construction.")
    m = df.groupby(["beta2", "spacing"]).agg(
        n=("asr", "size"), succ=("asr", lambda s: int((s > 0.5).sum())),
        mean_asr=("asr", "mean"), span_mem=("span_over_memory", "first"),
        trig_lr=("trig_lr", "first"), disp_step=("disp_step", "mean"),
        disp_window=("disp_window", "mean")).reset_index()
    print(m.to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    ref = m[m.beta2 == m.beta2.min()]
    print("\nContrast across beta2 WITHIN each spacing (placement identical by construction):")
    for sp, g in m.groupby("spacing"):
        g = g.sort_values("beta2")
        lo_b, hi_b = g.beta2.iloc[0], g.beta2.iloc[-1]
        a = df[(df.spacing == sp) & (df.beta2 == lo_b)].asr
        b = df[(df.spacing == sp) & (df.beta2 == hi_b)].asr
        print(f"  spacing {sp:4d}: beta2={lo_b:g} mean {a.mean():.3f} (span/mem "
              f"{g.span_over_memory.iloc[0] if 'span_over_memory' in g else g.span_mem.iloc[0]:.2f}) vs "
              f"beta2={hi_b:g} mean {b.mean():.3f} (span/mem {g.span_mem.iloc[-1]:.2f}), "
              f"two-sided MW p={mw(a, b):.3g}")
    print("\nPREDICTION (fixed in advance): with step size matched, ASR falls as span/memory "
          "drops below 1, so higher beta2 gives LOWER ASR at fixed spacing. The beta2="
          f"{m.beta2.min():g} row has span/memory > 1 everywhere and is the placement control: "
          "its trend across spacing is placement alone and should be subtracted.")
    ctrl = m[m.beta2 == m.beta2.min()].sort_values("spacing")
    if len(ctrl) >= 2:
        print(f"  placement control (beta2={m.beta2.min():g}) mean ASR across spacing: "
              + ", ".join(f"s{int(r.spacing)}={r.mean_asr:.3f}" for r in ctrl.itertuples()))
    sub = m[m.span_mem < 1]
    print(f"  cells genuinely inside the interference window (span/memory < 1): {len(sub)} of {len(m)}")
    if df.disp_window.notna().any():
        w = df.groupby("beta2").disp_window.mean()
        print(f"  measured disp_window by beta2: "
              + ", ".join(f"{b:g}:{v:.4f}" for b, v in w.items())
              + f"  (max/min = {w.max()/w.min():.2f}x; if this is large the cells are NOT "
                "step-matched and the beta2 contrast is confounded again)")


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
        elif e.startswith("th_"): threshold(df)
        elif e.startswith("int2_"): interference2(df)
        else: print("unrecognised experiment tag:", e)


if __name__ == "__main__":
    main(sys.argv[1:])
