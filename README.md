# ICLR-caliber paper project — working repo

Goal: produce one paper strong enough for ICLR's main track. Status and full
scouting trail below; this repo holds the code, not the narrative — see
`scouting/` and `theory/` for how we got here.

## No dataset to transfer

**Every experiment in this repo trains on synthetic data generated at
runtime.** There is nothing to cache, export, or pre-stage onto a pod —
`git clone` + `pip install -r requirements.txt` is the entire setup. If a
later idea needs a real dataset (e.g. real text for the poisoning
validation, or a real vision/language backbone), that will be called out
explicitly before it's needed.

## Status

| Idea | Where it stands | GPU needed? |
|---|---|---|
| **Poisoning near-constant-count theory** (lead) | `theory/poisoning_theory.md` has the closed-form theorem (rare-direction/count regime) and a conjecture (generic-direction/fraction regime), both verified in a scalar toy model (`experiments/poisoning/toy_model.py`, CPU, seconds). `experiments/poisoning/validate_transformer.py` is the real-model validation — CPU-smoke-tested and confirmed working (attack trains correctly, ASR reaches 1.0 under a strong-enough poison schedule), **not yet run at real scale.** This is the next thing to run. | Yes — this is what the GPU budget is for. |
| **Implicit-bias quantitative theory** (second track) | `experiments/implicit_bias/pilot_powerlaw.py` is a working CPU pilot comparing ridge regression vs. gradient descent on power-law-covariance data; suggests implicit bias shifts the scaling law's *prefactor*, not its *exponent*. Rough, six-seed, not yet formalized into a theory.md. | No — pure numpy, runs in ~2 min on CPU. |
| **Diffusion-LM length generalization** (backup, on hold) | `experiments/diffusion_lengen/pilot_copy_task.py` is **broken** — the diffusion training loop stopped learning the task after a position-embedding fix that was needed for the AR baseline. Needs real debugging time, not another blind fix. | Yes, once debugged — needs real training steps to say anything about length generalization. |

Full scouting trail (idea generation, novelty checks against ~50 killed
candidates, the ICLR 2026 acceptance-rate analysis that picked the niche,
the 10-helper-LLM convergence check): `scouting/`.

## Running the poisoning validation on a GPU pod

This is the priority job. `experiments/poisoning/validate_transformer.py`
trains a small GPT-style transformer (default: 4 layers, 128 dim, ~2-3M
params) from scratch on a synthetic next-token task, injects a backdoor,
and sweeps three axes matching Souly et al. (arXiv 2510.07192):

```bash
git clone <this-repo> && cd iclr-agent-paper
pip install -r requirements.txt
cd experiments/poisoning

# 1. Scheduling invariance: fixed poison count, vary density/frequency
python validate_transformer.py --experiment schedule \
    --out results_schedule.csv --dim 128 --layers 4 --seeds 5 \
    --n_clean 3000 --n_poison 300 --densities 1 4 16 64

# 2. Dataset-size invariance: fixed poison count, vary clean corpus size
python validate_transformer.py --experiment dataset_size \
    --out results_dataset_size.csv --dim 128 --layers 4 --seeds 5 \
    --n_poison 300 --n_clean_grid 500 2000 8000 32000

# 3. Rho phase diagram: rare vs. generic trigger, poison count needed vs. n_clean
python validate_transformer.py --experiment rho \
    --out results_rho.csv --dim 128 --layers 4 --seeds 3 \
    --n_clean_grid 500 2000 8000 32000 --n_poison_grid 50 200 800
```

Each of these ran correctly in a CPU smoke test at smaller scale (see commit
history / theory.md verification sections); at the settings above they
should take well under an hour combined on any single GPU (A10G or better) —
comfortably inside a few-GPU-hour budget. Bump `--n_clean_grid` and
`--n_poison_grid` higher (e.g. add `128000` / `3200`) for a more convincing
sweep once the small run confirms everything works; the `--densities` /
`--seeds` flags are the cheap knobs to turn up first for more statistical
power before spending time on bigger grids.

**What success looks like**, per `theory/poisoning_theory.md`:
- Experiment 1: ASR should be roughly flat across `--densities`, at fixed
  total poison count — confirms Theorem 1's scheduling invariance.
- Experiment 2: ASR should be roughly flat across `--n_clean_grid` — confirms
  the dataset-size invariance (the actual "near-constant count" claim).
- Experiment 3: the *rare* condition's ASR should stay high regardless of
  `n_clean`; the *generic* condition's should degrade as `n_clean` grows
  unless `n_poison` also grows — confirms the ρ phase diagram (Proposition 2).

Pull results back (`results_*.csv`) and push them to this repo, or paste
them back into the chat — either way works for the next analysis step.

## Repo layout

```
theory/              closed-form derivations + verification writeups
experiments/
  poisoning/          lead idea: toy model (CPU) + real-transformer validation (GPU)
  implicit_bias/       second track: CPU-only pilot
  diffusion_lengen/     backup, currently broken pilot
scouting/             novelty checks, ICLR 2026 corpus analysis, killed-idea log
figures/              diagnostic plots from the CPU pilots
```
