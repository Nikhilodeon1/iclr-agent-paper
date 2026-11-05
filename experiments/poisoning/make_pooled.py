"""Rebuilds two pooled tables from the raw per-job CSVs (exact reproductions; verified identical to the tables used in the report).

    cd experiments/poisoning && python make_pooled.py

results_decomp_pooled.csv  = every ../../anotherResultsWave/results_decomp_*.csv, concatenated
th_modelsize_valid.csv     = model-size runs that trained properly: 0.11M, 0.8M and 3.2M models from the shared-lr files and the
                             25.3M model from the lr 3e-4 rerun (results_th_modelsize_512x8_lr3e4.csv, which replaces the diverged 512x8 run)
"""
import os, glob
import pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); W6 = os.path.join(HERE, "..", "..", "anotherResultsWave")
dec = sorted(glob.glob(os.path.join(W6, "results_decomp_*.csv")))
pd.concat([pd.read_csv(f) for f in dec], ignore_index=True).to_csv(os.path.join(HERE, "results_decomp_pooled.csv"), index=False)
ms = ["results_th_modelsize_64x2.csv", "results_th_modelsize_128x4.csv", "results_th_modelsize_256x4.csv", "results_th_modelsize_512x8_lr3e4.csv"]
pd.concat([pd.read_csv(os.path.join(W6, f)) for f in ms], ignore_index=True).to_csv(os.path.join(HERE, "th_modelsize_valid.csv"), index=False)
print("wrote results_decomp_pooled.csv from", len(dec), "files; th_modelsize_valid.csv from", len(ms), "files")
