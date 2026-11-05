"""Wave 7 analysis (rules in preregistered_predictions.md, "Wave 7 predictions").

    python analyse_wave7.py pile  results_realtext_pile70m.csv        # G2 (also G3: results_realtext_pile160m.csv)
    python analyse_wave7.py scale --size 410m results_realtext_410m_s*.csv
    python analyse_wave7.py scale --size 1b   results_realtext_1b_s*.csv
"""
import argparse, sys
import numpy as np, pandas as pd

REF160 = {741: 8.75, 2159: 37.33, 1142: 47.83}   # Pythia-160M N50 from results_realtext_160m.csv (5 seeds)
REF_RARE_WIKI = {"70m": 9.11, "160m": 8.75}
TRIGGERS = [741, 2159, 1142]
BOOT = 2000


def n50(sub):
    m = sub.groupby("n_poison").asr.mean().sort_index(); xs = m.index.values.astype(float); ys = m.values
    if ys[0] >= 0.5: return float("nan")
    for i in range(len(xs) - 1):
        if ys[i] < 0.5 <= ys[i + 1]:
            f = (0.5 - ys[i]) / (ys[i + 1] - ys[i]); return float(np.exp(np.log(xs[i]) + f * (np.log(xs[i + 1]) - np.log(xs[i]))))
    return float("nan")


def n50_ci(sub, seed=0):
    rng = np.random.default_rng(seed); seeds = sub.seed.unique(); pt = n50(sub); draws = []
    for _ in range(BOOT):
        pick = rng.choice(seeds, size=len(seeds), replace=True)
        v = n50(pd.concat([sub[sub.seed == s] for s in pick]))
        if np.isfinite(v): draws.append(v)
    if len(draws) < BOOT * 0.5: return pt, float("nan"), float("nan")
    return pt, float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def load(files):
    d = pd.concat([pd.read_csv(f).assign(src=f) for f in files], ignore_index=True)
    dup = int(d.duplicated(["trigger_id", "n_poison", "seed"]).sum())
    print(f"files {len(files)}, rows {len(d)}, seeds {sorted(d.seed.unique())}, duplicates {dup}, NaN ASR {int(d.asr.isna().sum())}")
    if dup: raise SystemExit("duplicates present: resolve before analysing")
    return d


def table(d):
    rows = []
    for (tid, tr, cu), g in d.groupby(["trigger_id", "trigger", "expected_clean_uses"]):
        pt, lo, hi = n50_ci(g)
        rows.append(dict(trigger_id=int(tid), trigger=tr, clean_uses=round(float(cu), 1), baseline_asr=float(g.baseline_asr.max()), N50=pt, lo=lo, hi=hi,
                         asr_by_N=g.groupby("n_poison").asr.mean().round(2).to_dict(), clean_loss=round(float(g.clean_loss.mean()), 3)))
    t = pd.DataFrame(rows).sort_values("clean_uses").reset_index(drop=True)
    return t


def ordering(t):
    v = t.N50.values
    return bool(np.all(np.isfinite(v)) and np.all(np.diff(v) > 0))


def pile(files, size):
    d = load(files); t = table(d); print(t.round(2).to_string(index=False))
    why = []
    if (t.baseline_asr > 0).any(): why.append("baseline ASR above 0 for some trigger")
    if not np.all(np.isfinite(t.N50)): why.append("N50 not reached inside the grid for some trigger")
    print("\ninformative:", not why, why)
    print("PRIMARY ordering (N50 strictly increases with measured clean uses):", ordering(t))
    pos = t[t.clean_uses > 0]
    if len(pos) >= 2 and np.all(np.isfinite(pos.N50)):
        print(f"secondary: log-log slope of N50 vs expected clean uses (nonzero triggers): {np.polyfit(np.log(pos.clean_uses), np.log(pos.N50), 1)[0]:.2f}  (proportional = 1)")
    rare = t[t.clean_uses == 0]
    if len(rare) and size in REF_RARE_WIKI:
        r = float(rare.N50.iloc[0]) / REF_RARE_WIKI[size]
        print(f"secondary: rare-trigger N50 {float(rare.N50.iloc[0]):.2f} = {r:.2f} x WikiText value {REF_RARE_WIKI[size]}; within [0.67, 1.5]: {0.67 <= r <= 1.5}")
    print("\nNOTE: the trigger tokens differ from the WikiText runs (chosen by measured frequency on this stream).")


def classify(ratios):
    lo, hi = min(ratios), max(ratios)
    if lo >= 0.67 and hi <= 1.5: return "INVARIANT"
    if lo < 0.67 and hi <= 1.5: return "FALLING"
    if hi > 1.5 and lo >= 0.67: return "RISING"
    return "MIXED"


def scale(files, size):
    d = load(files); per = d.groupby("trigger_id").seed.nunique().to_dict(); print("seeds per trigger:", per)
    t = table(d); print(t.round(2).to_string(index=False))
    ids = sorted(int(x) for x in t.trigger_id); why = []
    if ids != sorted(TRIGGERS): why.append(f"trigger ids {ids} differ from {sorted(TRIGGERS)}")
    if (t.baseline_asr > 0).any(): why.append("baseline ASR above 0")
    if not np.all(np.isfinite(t.N50)): why.append("N50 not reached inside the grid for some trigger")
    print("\ninformative:", not why, why)
    print(f"PRIMARY ordering at {size}:", ordering(t))
    ratios = []
    for _, r in t.iterrows():
        ref = REF160.get(int(r.trigger_id)); ratios.append(float(r.N50) / ref if ref and np.isfinite(r.N50) else float("nan"))
        print(f"  trigger {int(r.trigger_id)}: N50 {r.N50:.2f} [{r.lo:.2f}, {r.hi:.2f}] vs 160M {ref}: ratio {ratios[-1]:.2f}")
    if np.all(np.isfinite(ratios)): print(f"SIZE CLASSIFICATION at {size} (point estimates vs Pythia-160M): {classify(ratios)}")
    if size == "1b" and max(per.values()) < 5: print("caution: only", max(per.values()), "seeds; intervals are weak")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("pile"); a.add_argument("files", nargs="+"); a.add_argument("--size", default="70m")
    b = sub.add_parser("scale"); b.add_argument("files", nargs="+"); b.add_argument("--size", required=True)
    x = ap.parse_args(); (pile if x.cmd == "pile" else scale)(x.files, x.size)
