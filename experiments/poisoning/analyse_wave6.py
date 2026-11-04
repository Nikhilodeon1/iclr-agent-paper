"""Wave 6 analysis. Applies the decision rules in preregistered_predictions.md ("Wave 6 predictions").

    python analyse_wave6.py spacing results_spacing_s8.csv results_spacing_s13.csv results_spacing_s18.csv results_spacing_s23.csv \
        [--old results_decomp_B0.csv results_decomp_spacing.csv]
    python analyse_wave6.py matched --adamw results_matched_adamw_a.csv results_matched_adamw_b.csv \
        --sgd results_events_matched.csv results_events_sgd_n64_topup.csv

The rules are in the code as constants so they cannot drift from what was written down before the runs.
"""
import argparse, re, sys
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

BOOT = 4000


def wil(x, y):
    """One-sided paired Wilcoxon, x higher than y. Returns 1.0 when every difference is zero."""
    d = np.asarray(x, float) - np.asarray(y, float)
    if not np.any(d != 0):
        return 1.0
    return float(wilcoxon(x, y, alternative="greater").pvalue)


def boot_mean(d, rng, n=BOOT):
    d = np.asarray(d, float)
    b = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(n)]
    return float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def read(files):
    parts = []
    for f in files:
        d = pd.read_csv(f)
        d["src"] = f
        parts.append(d)
    return pd.concat(parts, ignore_index=True)


# ----------------------------------------------------------------------------- spacing
def spacing_verdict(drop, p, informative):
    if not informative:
        return "UNINFORMATIVE (mean ASR at S=700 outside [0.15, 0.85])"
    if drop >= 0.25 and p < 0.05:
        return "STRICT: spacing between doses contributes at the size first seen"
    if 0.10 <= drop < 0.25 and p < 0.05:
        return "SMALLER: a spacing effect exists but is smaller than the first look suggested"
    if drop >= 0.10 and 0.05 <= p < 0.10:
        return "INCONCLUSIVE"
    return "ABSENT: no spacing effect detected; the cause of the long-sweep decline stays unexplained"


def parse_spacing(d):
    m = d.experiment.str.extract(r"decomp_(\w+?)_B(\d+)_A(\d+)_S(\d+)$")
    if m.isna().any().any():
        raise SystemExit("unparseable decomp tags: " + str(d.experiment[m[0].isna()].unique()[:3]))
    d = d.assign(opt=m[0], B=m[1].astype(int), A=m[2].astype(int), S=m[3].astype(int))
    return d[(d.B == 0) & (d.A == 0)]


def table(piv):
    rows = []
    for S in piv.columns:
        x = piv[S].dropna()
        rows.append(dict(S=S, n=len(x), mean_asr=round(x.mean(), 3), succ_gt_half=int((x > .5).sum())))
    return pd.DataFrame(rows).to_string(index=False)


def contrasts(piv, label, rng):
    out = {}
    for hi, lo in [(700, 10000), (700, 2800), (2800, 10000)]:
        if hi not in piv or lo not in piv:
            continue
        both = piv[[hi, lo]].dropna()
        x, y = both[hi].values, both[lo].values
        drop = float((x - y).mean())
        p = wil(x, y)
        l, h = boot_mean(x - y, rng)
        out[(hi, lo)] = (drop, p, l, h, len(both))
        print(f"  {label} S={hi} minus S={lo}: n={len(both)} mean paired drop {drop:+.3f} "
              f"[95% bootstrap {l:+.3f}, {h:+.3f}] one-sided Wilcoxon p={p:.4f}; "
              f"seeds higher/lower: {(x > y).sum()}/{(x < y).sum()}")
    return out


def spacing(files, old):
    rng = np.random.default_rng(0)
    new = parse_spacing(read(files))
    dup = int(new.duplicated(["S", "seed"]).sum())
    print(f"NEW seeds: {new.seed.nunique()} ({new.seed.min()}-{new.seed.max()}), rows {len(new)}, duplicate (S, seed): {dup}, NaN ASR: {int(new.asr.isna().sum())}")
    if dup:
        raise SystemExit("duplicates present: do not analyse until resolved")
    piv = new.pivot_table(index="seed", columns="S", values="asr")
    complete = piv.dropna()
    print(f"seeds with all three spacings: {len(complete)} of {len(piv)}")
    print("\nPRIMARY (new seeds only):"); print(table(piv))
    print(f"clean loss range across rows: {new.clean_loss.min():.3f} to {new.clean_loss.max():.3f}")
    c = contrasts(piv, "primary", rng)
    informative = 0.15 <= piv[700].mean() <= 0.85
    drop, p = c[(700, 10000)][0], c[(700, 10000)][1]
    print(f"\ninformative (mean ASR at S=700 = {piv[700].mean():.3f} in [0.15, 0.85]): {informative}")
    print("VERDICT:", spacing_verdict(drop, p, informative))
    m = piv.mean()
    print(f"monotone decreasing in S (cell means {m[700]:.3f}, {m[2800]:.3f}, {m[10000]:.3f}): {bool(m[700] >= m[2800] >= m[10000])}")
    if old:
        o = parse_spacing(read(old))
        both = pd.concat([o.assign(group="wave5"), new.assign(group="wave6")], ignore_index=True)
        d2 = int(both.duplicated(["S", "seed"]).sum())
        print(f"\nSECONDARY (pooled with Wave 5 seeds; HARDWARE-MIXED, not the pre-specified test): duplicates {d2}")
        pv = both.pivot_table(index="seed", columns="S", values="asr"); print(table(pv)); contrasts(pv, "pooled", rng)


# ----------------------------------------------------------------------------- matched
def matched_verdict(informative, why, riseA, pA, D, lo):
    if not informative:
        return "UNINFORMATIVE (" + "; ".join(why) + ")"
    if riseA >= 0.25 and pA < 0.05:
        return "SUPPORTED" if (D >= 0.20 and lo > 0) else "PARTIAL: AdamW rises, the contrast with SGD does not meet the rule"
    if riseA < 0.10 or pA >= 0.10:
        return "NOT SUPPORTED: no batch-count effect for AdamW at this learning rate"
    return "INCONCLUSIVE"


def arm_series(d, E):
    return d[d.E == E].sort_values("seed").set_index("seed").asr


def matched(adamw, sgd):
    rng = np.random.default_rng(0)
    a = read(adamw)
    a = a[a.n_poison == 64].copy()
    a["E"] = a.experiment.str.extract(r"events_adamw_E(\d+)_lr0\.0003$")[0]
    if a.E.isna().any():
        raise SystemExit("unexpected AdamW tags: " + str(a.experiment[a.E.isna()].unique()[:4]))
    a["E"] = a.E.astype(int)
    s = read(sgd)
    s = s[s.experiment.str.startswith("events_sgd") & (s.n_poison == 64)].copy()
    s["E"] = s.experiment.str.extract(r"events_sgd_E(\d+)$")[0].astype(int)
    s = s[s.seed < 20]
    for name, d in [("AdamW", a), ("SGD", s)]:
        dup = int(d.duplicated(["E", "seed"]).sum()); n = d.groupby("E").seed.nunique().to_dict()
        print(f"{name}: seeds per E {n}, duplicate (E, seed): {dup}, NaN ASR: {int(d.asr.isna().sum())}")
        if dup:
            raise SystemExit("duplicates present: do not analyse until resolved")
    print("\nPer-arm table (N=64):")
    for name, d in [("AdamW lr 3e-4", a), ("SGD lr 0.1", s)]:
        t = d.groupby("E").agg(n=("seed", "nunique"), mean_asr=("asr", "mean"), succ_gt_half=("asr", lambda x: int((x > .5).sum())))
        print(name); print(t.round(3).to_string())
    cl = a.groupby("E").clean_loss.mean()
    print("\nAdamW mean clean loss by E:", cl.round(3).to_dict())
    A1, A16 = arm_series(a, 1), arm_series(a, 16); S1, S16 = arm_series(s, 1), arm_series(s, 16)
    ixa, ixs = A1.index.intersection(A16.index), S1.index.intersection(S16.index)
    dA, dS = (A16[ixa] - A1[ixa]).values, (S16[ixs] - S1[ixs]).values
    riseA, riseS = float(dA.mean()), float(dS.mean()); pA, pS = wil(A16[ixa].values, A1[ixa].values), wil(S16[ixs].values, S1[ixs].values)
    D = riseA - riseS
    bs = [dA[rng.integers(0, len(dA), len(dA))].mean() - dS[rng.integers(0, len(dS), len(dS))].mean() for _ in range(BOOT)]
    lo, hi = float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))
    print(f"\nAdamW rise (E=16 minus E=1): {riseA:+.3f}, paired one-sided Wilcoxon p={pA:.4f}, seeds up/down {(dA > 0).sum()}/{(dA < 0).sum()}")
    print(f"SGD   rise (E=16 minus E=1): {riseS:+.3f}, paired one-sided Wilcoxon p={pS:.4f}, seeds up/down {(dS > 0).sum()}/{(dS < 0).sum()}")
    print(f"D = rise_A - rise_S = {D:+.3f}, 95% bootstrap [{lo:+.3f}, {hi:+.3f}]")
    why = []
    m1A, m1S = float(A1.mean()), float(S1.mean())
    if not 0.10 <= m1A <= 0.60: why.append(f"AdamW mean ASR at E=1 is {m1A:.3f}, outside [0.10, 0.60]")
    if not 0.10 <= m1S <= 0.60: why.append(f"SGD mean ASR at E=1 is {m1S:.3f}, outside [0.10, 0.60]")
    if (cl > 3.96).any(): why.append(f"AdamW clean loss above 3.96 in some E cell (max {cl.max():.3f})")
    if cl.max() - cl.min() >= 0.10: why.append(f"clean loss spread across E cells is {cl.max()-cl.min():.3f} (>= 0.10)")
    informative = not why
    print("\ninformative:", informative, why)
    print("VERDICT:", matched_verdict(informative, why, riseA, pA, D, lo))
    for E in (2, 4):
        if E in a.E.values:
            x = arm_series(a, E); print(f"secondary: AdamW E={E}: mean ASR {x.mean():.3f} ({int((x > .5).sum())}/{len(x)})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p1 = sub.add_parser("spacing"); p1.add_argument("files", nargs="+"); p1.add_argument("--old", nargs="*", default=None)
    p2 = sub.add_parser("matched"); p2.add_argument("--adamw", nargs="+", required=True); p2.add_argument("--sgd", nargs="+", required=True)
    a = ap.parse_args()
    spacing(a.files, a.old) if a.cmd == "spacing" else matched(a.adamw, a.sgd)
