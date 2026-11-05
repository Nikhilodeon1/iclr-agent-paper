# Backdoor poisoning thresholds: code, raw results and report figures

How many poisoned examples a backdoor needs, as a function of how often the trigger already occurs in clean data.
This repo holds the code and the raw per-run results. The report itself is the Word file kept outside the repo.
`preregistered_predictions.md` holds every prediction and decision rule, written before the runs it covers.

## Setup

```bash
pip install -r requirements.txt            # torch, numpy, pandas, scipy, matplotlib
bash pod/add_hf.sh                         # only for the Pythia jobs: adds transformers + datasets
```
Transformer experiments from scratch use synthetic data generated at runtime. The Pythia jobs download models and
text from Hugging Face (`EleutherAI/pythia-*`, `Salesforce/wikitext`, `NeelNanda/pile-10k`), so they need internet access.
GPU jobs refuse to start on CPU (`pod/preflight.py`).

## Layout

| Path | Contents |
|---|---|
| `theory/poisoning_theory.md` | Derivations (rare-trigger ODE, fixed point for a common trigger, optimizer analysis) |
| `experiments/poisoning/toy_model.py` | Scalar model and its discrete recursion (CPU, seconds) |
| `experiments/poisoning/validate_transformer.py` | Transformers from scratch: modes `threshold`, `events`, `interference2`, `decomp`, ... (`--help`) |
| `experiments/poisoning/validate_realtext.py` | Continued training of Pythia models on WikiText-2 or a Pile sample; `--seed_offset`, `--resume`, `--held_split` |
| `experiments/poisoning/analyse_*.py` | Analysis scripts that apply the pre-set rules (usage line at the top of each file) |
| `experiments/poisoning/make_figures.py` | Regenerates the five report figures (about 2 minutes, CPU) into `figures/report_v2/` |
| `experiments/poisoning/make_pooled.py` | Rebuilds `results_decomp_pooled.csv` and `th_modelsize_valid.csv` from raw files |
| `experiments/poisoning/*.csv`, `anotherResultsWave/*.csv` | Raw per-run results (one row per run) |
| `pod/` | Launch, status and result-packing scripts for GPU pods; `pod/jobs.sh` defines every job |

## Reproducing the report

```bash
cd experiments/poisoning
python make_pooled.py                                  # pooled tables (exact reproductions)
python make_figures.py                                 # Figures 1-5
# Wave 6 (spacing and matched optimizers)
python analyse_wave6.py spacing ../../anotherResultsWave/results_spacing_s8.csv ../../anotherResultsWave/results_spacing_s13.csv \
    ../../anotherResultsWave/results_spacing_s18.csv ../../anotherResultsWave/results_spacing_s23.csv
python analyse_wave6.py matched --adamw ../../anotherResultsWave/results_matched_adamw_a.csv ../../anotherResultsWave/results_matched_adamw_b.csv \
    --sgd ../../anotherResultsWave/results_events_matched.csv ../../anotherResultsWave/results_events_sgd_n64_topup.csv
# Decomposition of the decline
python analyse_decomp.py results_decomp_pooled.csv
# Other waves (optimizer mechanism, learning rate, parameter groups): python analyse_jobs.py <csv files>
```

Re-running an experiment: `. pod/env.sh && bash pod/launch.sh <JOB>` (see `pod/jobs.sh` for job letters and their exact commands;
`RESUME=1` continues an interrupted job; `python pod/status.py` shows progress; `bash pod/pack_results.sh` packs the CSVs).

## Known gaps

* `sparse_trigger_check.csv`, `highdim_sparse_check.csv` (Figure 2) and `rho_v2_thresholds.csv` (Figure 1b) are derived tables.
  Their generating scripts were not saved in this repo. The numbers in the report can be recomputed independently from
  the raw files (`results_rho_v2_tidy.csv` for Figure 1b) or by re-simulating the recursions in `toy_model.py`,
  but those scripts are not provided yet.
* `jobs.sh` expands `wave7` to the two Pile jobs only. Start the Pythia-410M and 1B jobs by letter (`P4A ... P1C`).
* Frequencies of real-text triggers were measured on WikiText-2, not on the corpus Pythia was trained on; the Pile-sample job (`G2`, `G3`) measures the difference.
