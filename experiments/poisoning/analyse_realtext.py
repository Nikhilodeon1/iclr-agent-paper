"""Turn results_realtext_freq.csv into the numbers that fill realtext_section.tex.

Prints each placeholder in the order it appears in that file, so filling the section
is a copy, not a judgement call. Run:

    python analyse_realtext.py results_realtext_freq.csv
"""

import sys

import numpy as np
import pandas as pd


def threshold_50(sub):
    """Poison count for 50% ASR, log-linearly interpolated between grid points.

    Returns (value, kind) where kind is 'point' if the crossing is bracketed,
    '<min' if the smallest budget already exceeds 50%, or '>max' if none does.
    """
    g = sub.groupby("n_poison").asr.mean().sort_index()
    Ns, means = g.index.to_numpy(float), g.to_numpy()
    if means[0] >= 0.5:
        return Ns[0], "<min"
    for (a, ma), (b, mb) in zip(zip(Ns, means), zip(Ns[1:], means[1:])):
        if ma < 0.5 <= mb:
            return a * (b / a) ** ((0.5 - ma) / (mb - ma)), "point"
    return Ns[-1], ">max"


def main(path):
    df = pd.read_csv(path)
    print(f"rows={len(df)}  runs/cell={df.groupby(['trigger_id','n_poison']).size().unique()}")
    print(f"baseline ASR (must be ~0): {sorted(df.baseline_asr.unique())}")
    print()
    rows = []
    for tid, sub in df.groupby("trigger_id"):
        per_step = sub.per_step.iloc[0]
        T = sub.n_steps.iloc[0]
        N50, kind = threshold_50(sub)
        rows.append(dict(trigger=sub.trigger.iloc[0], per_step=per_step,
                         predicted=per_step * T, measured=N50, kind=kind,
                         ratio=(N50 / (per_step * T)) if per_step > 0 else np.nan))
    R = pd.DataFrame(rows).sort_values("per_step").reset_index(drop=True)
    print(R.to_string(index=False, float_format=lambda x: f"{x:,.3f}"))
    print()

    zero = R[R.per_step == 0]
    pos = R[R.per_step > 0]
    print("--- values for realtext_section.tex, in order of appearance ---")
    print(f"<n>   = {len(df)}")
    if len(zero):
        print(f"<N0>  = {zero.measured.iloc[0]:,.0f}" + (f"  ({zero.kind.iloc[0]})" if zero.kind.iloc[0] != "point" else ""))
    for i, r in enumerate(pos.itertuples(), start=1):
        print(f"<N{i}>  = {r.measured:,.0f}" + (f"  ({r.kind})" if r.kind != "point" else ""))
    for i, r in enumerate(pos.itertuples(), start=1):
        print(f"<f{i}>  = {r.per_step:.3g}")
    for i, r in enumerate(pos.itertuples(), start=1):
        print(f"<p{i}>  = {r.predicted:,.0f}")
    if len(pos):
        lo, hi = pos.ratio.min(), pos.ratio.max()
        print(f"\nmeasured/predicted ratio spans {lo:.2f}x to {hi:.2f}x"
              f"  -> for the '<one sentence...>' slot")
        if len(pos) >= 2:
            sl = np.polyfit(np.log(pos.predicted), np.log(pos.measured), 1)[0]
            print(f"log-log slope of measured vs predicted = {sl:.2f} (theory: 1.0)")
    base = df.clean_loss.min()
    print(f"<dL>  = {df.clean_loss.max() - base:.2f}   (clean-loss spread across runs; "
          f"min {base:.2f}, max {df.clean_loss.max():.2f})")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results_realtext_freq.csv")
