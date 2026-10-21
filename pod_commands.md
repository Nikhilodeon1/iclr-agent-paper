# Quick start (read this first)

**Use `RUN_GUIDE.md`** for setup, launch, progress checks and getting results back. It replaces
the earlier `pip install -r requirements.txt` step: that file pins an old torch and is no longer
used. `pod/jobs.sh` and `launch.sh` cover **Jobs A, B, C, D, E and H only** (checked identical to the commands below, flag for flag). Jobs **F (two commands), G and G2 are not in `jobs.sh`**: `launch.sh F` would print 'unknown job'. Run those from the sheet by hand, from `experiments/poisoning/` with the environment active (`. pod/env.sh`), using `nohup ... &` and a log under `/tmp/project_scratch/logs/`; `status.py` will not track them.

Predictions were fixed in advance in `preregistered_predictions.md`; the decision rule for each
outcome is in `gameplan.md`. All eight `validate_transformer.py` commands were dry-run locally
through the real script with training stubbed. The Pythia commands (Job G) were not.

---

## Pod commands — Tier 2

All commands assume the pod is at `experiments/poisoning/` with `git pull` done first
(`validate_transformer.py` changed substantially). Every job flushes each row to disk
immediately and catches per-cell exceptions, so a crash never loses completed rows and
a diverged cell logs `ERROR` and continues. Each job prints a preflight line with its
own step count and time estimate before training starts — if that estimate looks wrong,
kill it then rather than after.

Timings assume ~37 steps/s, measured on the A40 at `--dim 128 --layers 4`.

---

## Priority 1 — Job A: beta2 x spacing (~2.0 h)

The single most valuable run. Total poison is **identical (N=16) in every cell**, so a
frequency-counting account predicts no differences at all. Proposition 3 predicts
poisoned batches interfere when their span falls inside Adam's second-moment memory
1/(1-beta2). The last dose is pinned at step 2,400 for every spacing, so post-poison
erosion is constant; the clean contrast is **across beta2 within a spacing column**,
where placement is identical by construction.

```
python3.12 validate_transformer.py --experiment beta2 --out results_beta2.csv \
  --dim 128 --layers 4 --seeds 10 --n_clean 3000 \
  --n_events 8 --per_event 2 \
  --beta2_grid 0.95 0.99 0.999 --spacing_grid 10 50 250
```

Interference status by cell (E*spacing / (1/(1-beta2))), printed as `span/memory`:

| spacing | beta2=0.95 | beta2=0.99 | beta2=0.999 |
|---|---|---|---|
| 10 | 4.0 independent | 0.8 **interfere** | 0.08 **interfere** |
| 50 | 20 independent | 4.0 independent | 0.4 **interfere** |
| 250 | 100 independent | 20 independent | 2.0 independent |

**Reads as support** if ASR is lower in the interfering cells than the independent ones
at the same spacing — i.e. at spacing 10, ASR(0.999) < ASR(0.99) < ASR(0.95).
**Reads as refutation** if ASR is flat across beta2 at every spacing.

---

## Priority 2 — Job B: per-parameter-group optimizer test (~2.7 h)

Tests whether the batch-count effect runs through the trigger's own embedding row, as
Proposition 3 claims, rather than through the rest of the network. `trignone` is the
**unsplit control**: if it disagrees with `adamw_trigadamw`, splitting the parameter
changed the dynamics and the comparison is confounded — check that first.

```
python3.12 validate_transformer.py --experiment paramgroup --out results_paramgroup.csv \
  --dim 128 --layers 4 --seeds 5 --n_clean 3000 --sgd_lr 0.1 \
  --pg_totals 16 32 --events_grid 1 4 16 \
  --paramgroup_conditions adamw_trignone,adamw_trigadamw,adamw_trigsgd,sgd_trigadamw
```

**Reads as support** if ASR rises with E whenever the trigger row is on AdamW
(`adamw_trigadamw`, `sgd_trigadamw`) and is flat when it is on SGD (`adamw_trigsgd`) —
i.e. the effect tracks the trigger row's optimizer, not the network's.
**Reads as refutation** if it tracks the network's optimizer instead.

---

## Priority 3 — Job C: learning-rate scaling (~2.7 h)

Equation 6 predicts N* proportional to 1/eta at fixed required margin. Frequency
counting predicts no learning-rate dependence whatsoever. The N grid spans 2–256 to
bracket the threshold at all three learning rates (predicted roughly 53 / 16 / 5).

```
python3.12 validate_transformer.py --experiment lrscaling --out results_lrscaling.csv \
  --dim 128 --layers 4 --seeds 5 --n_clean 3000 --rho_density 8 \
  --lr_grid 3e-4 1e-3 3e-3 \
  --n_poison_grid 2 4 8 16 32 64 128 256
```

Note: lr=3e-3 may diverge on some seeds. That is handled — the run logs `ERROR` for
those cells and continues. If *every* 3e-3 cell errors, report it and I will drop that
arm and substitute 5e-4.

---

## Priority 4 — Job D: matched-difficulty AdamW vs SGD (~1.9 h)

Removes the floor-effect confound in the current optimizer comparison. It runs both
optimizers at N=64 and N=128, where SGD is off the floor (0.14–0.25 at N=64) rather
than pinned at 0.00 as it is at N=16.

```
python3.12 validate_transformer.py --experiment events --out results_events_matched.csv \
  --dim 128 --layers 4 --seeds 6 --n_clean 3000 --sgd_lr 0.1 \
  --optimizers adamw sgd \
  --events_grid 1 2 4 16 --per_event_grid 4 8 16 32 64 --only_N 64 128
```

---

## Priority 5 — Job E: one-protocol seed top-up (~2.7 h)

Brings the decisive N=16 cells to 20 seeds under a single protocol with initialisation
seeded, so the comparison can be reported with Wilson intervals that actually separate.

```
python3.12 validate_transformer.py --experiment events --out results_events_20seed.csv \
  --dim 128 --layers 4 --seeds 20 --n_clean 3000 --sgd_lr 0.1 \
  --optimizers adamw sgd \
  --events_grid 1 4 16 --per_event_grid 1 4 16 --only_N 16
```

---

## Priority 6 — Job F: Proposition 1 under SGD, and weight decay off (~3.2 h, two commands)

The training-length sweep currently uses AdamW, but Proposition 1 is an SGD result, so
the observed decline may be an optimizer effect rather than a failure of the
proposition. The second command tests the weight-decay explanation on the real model
instead of only in simulation.

```
python3.12 validate_transformer.py --experiment dataset_size --out results_ds_sgd.csv \
  --dim 128 --layers 4 --seeds 5 --n_poison 16 --sgd_lr 0.1 \
  --optimizers sgd --n_clean_grid 500 2000 8000 32000

python3.12 validate_transformer.py --experiment dataset_size --out results_ds_wd0.csv \
  --dim 128 --layers 4 --seeds 5 --n_poison 16 \
  --optimizers adamw --weight_decay 0 --n_clean_grid 500 2000 8000 32000
```

If the decline survives with `--weight_decay 0`, the weight-decay account is wrong and
the explanation shifts to shared-parameter drift — worth knowing either way.

---

## Job G — finish the Pythia experiment on natural text (~2.0 h)

The submitted version had only 3 of 4 frequency levels at 2 seeds per cell. This runs
all four at 5 seeds. It uses `validate_realtext.py`, which was verified end to end on a
pod earlier (fp32 fix included). Pythia cannot be smoke-tested on the local CPU, so the
first command below is a ~2 minute pod smoke test; run it before the full job.

```
# 1. smoke test (~2 min): must print finite clean_loss and a higher ASR at N=256 than N=16
python3.12 validate_realtext.py --experiment frequency --out smoke_rt.csv \
  --max_tokens 200000 --n_steps 200 --seeds 1 --batch 8 --seq_len 128 \
  --freq_targets 0 --n_poison_grid 16 256

# 2. full job, 4 triggers x 5 poison counts x 5 seeds x 600 steps = 60,000 steps
python3.12 validate_realtext.py --experiment frequency --out results_realtext_5seed.csv \
  --n_steps 600 --seeds 5 --freq_targets 0 0.05 0.2 0.5 \
  --n_poison_grid 4 16 64 256 1024
```

Do **not** reuse the older command with `--freq_targets 0 0.05 0.5 5.0`: the 5.0 target
predicts a threshold near 3,000 poison, above the 1,024 grid maximum, and would come back
censored.

## Job G2 (optional, UNTESTED) — frequency measured on a Pile sample

Reviewers noted trigger frequency is measured on WikiText-2, not on the Pile that Pythia
was pretrained on. This reruns with a Pile sample as the continued-pretraining corpus, so
the measured frequency is at least in-distribution with pretraining. I have not run this:
the loader assumes a `text` column and a named config, and I have not confirmed
`NeelNanda/pile-10k` loads with it. Run the smoke test first; if it errors on loading,
send me the traceback and skip the job.

```
python3.12 validate_realtext.py --experiment frequency --out smoke_pile.csv \
  --dataset NeelNanda/pile-10k --config default \
  --max_tokens 200000 --n_steps 200 --seeds 1 --batch 8 --seq_len 128 \
  --freq_targets 0 --n_poison_grid 16 256
```

If that prints rows with finite `clean_loss`, the full version is the Job G command with
`--dataset NeelNanda/pile-10k --config default --max_tokens 2000000 --out results_realtext_pile.csv`.

---

## Job H — trigger-embedding displacement per poisoned batch (~0.6 h)

A direct, parameter-free test of Proposition 3's premise, requested by a reviewer. One
poisoned batch is applied after 1,000 clean steps; the script records how far the
trigger's own embedding row moves. Under AdamW the first-step displacement should be
the same for 1 and 64 poison examples and equal to `pred_disp` (written to the CSV); under
SGD it should grow in proportion to the number of examples. Unlike ASR it has no
floor or ceiling. A CPU check at d=32 matched the prediction to 0.1% for k>=16 and
to 2.5% at k=1 (the epsilon effect the paper already flags).

```
python3.12 validate_transformer.py --experiment embdisp --out results_embdisp.csv \
  --dim 128 --layers 4 --seeds 5 --embdisp_warmup 1000 --embdisp_window 30 \
  --embdisp_k_grid 1 4 16 64 --optimizers adamw sgd \
  --beta2_grid 0.95 0.99 0.999 --sgd_lr 0.1
```

**Reads as support** if AdamW `disp_step` is flat across k and matches `pred_disp`, shrinks
with beta2 as predicted, and SGD `disp_step` scales ~linearly with k.
Note this confirms the *mechanism's premise*; whether it produces the ASR effect is what
Jobs A, B and E test.

---

## If pod budget is tight

Run **A and B only**. Those two answer the two objections all six reviewers converged
on: that the headline test does not discriminate the theory from frequency counting
(A supplies a prediction counting cannot make), and that the two-channel mechanism is
asserted rather than tested (B tests it directly). Everything else tightens the paper;
these two change what it establishes.

## Pulling results back

Any of `results_*.csv` pasted into the chat or pushed to the repo. Each file is
self-describing: the `experiment` column encodes the condition, and the new columns
`beta2`, `spacing`, `span_over_memory`, `rest_opt`, `trigger_opt` and `lr` are populated
only for the runs they apply to.
