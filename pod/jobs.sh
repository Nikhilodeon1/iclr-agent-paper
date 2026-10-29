# Sourced by launch.sh. Each job: JOB_<letter>="<csv>|<expected rows>|<arguments>"
JOB_A="results_beta2.csv|90|--experiment beta2 --out results_beta2.csv --dim 128 --layers 4 --seeds 10 --n_clean 3000 --n_events 8 --per_event 2 --beta2_grid 0.95 0.99 0.999 --spacing_grid 10 50 250"
JOB_B="results_paramgroup.csv|120|--experiment paramgroup --out results_paramgroup.csv --dim 128 --layers 4 --seeds 5 --n_clean 3000 --sgd_lr 0.1 --pg_totals 16 32 --events_grid 1 4 16 --paramgroup_conditions adamw_trignone,adamw_trigadamw,adamw_trigsgd,sgd_trigadamw"
JOB_E="results_events_20seed.csv|120|--experiment events --out results_events_20seed.csv --dim 128 --layers 4 --seeds 20 --n_clean 3000 --sgd_lr 0.1 --optimizers adamw sgd --events_grid 1 4 16 --per_event_grid 1 4 16 --only_N 16"
JOB_H="results_embdisp.csv|80|--experiment embdisp --out results_embdisp.csv --dim 128 --layers 4 --seeds 5 --embdisp_warmup 1000 --embdisp_window 30 --embdisp_k_grid 1 4 16 64 --optimizers adamw sgd --beta2_grid 0.95 0.99 0.999 --sgd_lr 0.1"
JOB_C="results_lrscaling.csv|120|--experiment lrscaling --out results_lrscaling.csv --dim 128 --layers 4 --seeds 5 --n_clean 3000 --rho_density 8 --lr_grid 3e-4 1e-3 3e-3 --n_poison_grid 2 4 8 16 32 64 128 256"
JOB_D="results_events_matched.csv|84|--experiment events --out results_events_matched.csv --dim 128 --layers 4 --seeds 6 --n_clean 3000 --sgd_lr 0.1 --optimizers adamw sgd --events_grid 1 2 4 16 --per_event_grid 4 8 16 32 64 --only_N 64 128"
WAVE1="H A B E"

# ---- Wave 2 (added after Wave 1; Job A's interference test failed and M/K/C replace it) ----
# Model-size sweep: one job per size so four pods run in parallel. 110K -> 25.3M params.
JOB_M1="results_th_modelsize_64x2.csv|48|--experiment threshold --th_axis modelsize --model_grid 64:2 --out results_th_modelsize_64x2.csv --n_clean 3000 --rho_density 4 --n_poison_grid 2 4 8 16 32 64 --seeds 8"
JOB_M2="results_th_modelsize_128x4.csv|48|--experiment threshold --th_axis modelsize --model_grid 128:4 --out results_th_modelsize_128x4.csv --n_clean 3000 --rho_density 4 --n_poison_grid 2 4 8 16 32 64 --seeds 8"
JOB_M3="results_th_modelsize_256x4.csv|48|--experiment threshold --th_axis modelsize --model_grid 256:4 --out results_th_modelsize_256x4.csv --n_clean 3000 --rho_density 4 --n_poison_grid 2 4 8 16 32 64 --seeds 8"
JOB_M4="results_th_modelsize_512x8.csv|48|--experiment threshold --th_axis modelsize --model_grid 512:8 --out results_th_modelsize_512x8.csv --n_clean 3000 --rho_density 4 --n_poison_grid 2 4 8 16 32 64 --seeds 8"
# Tuning control for the largest model: same cells at a lower learning rate, so a flat or
# rising N* at 512:8 cannot be blamed on lr=1e-3 being wrong for that size.
JOB_M5="results_th_modelsize_512x8_lr3e4.csv|48|--experiment threshold --th_axis modelsize --model_grid 512:8 --lr 3e-4 --out results_th_modelsize_512x8_lr3e4.csv --n_clean 3000 --rho_density 4 --n_poison_grid 2 4 8 16 32 64 --seeds 8"
# Density: Prop 3 predicts N* proportional to k (exponent 1); example-counting predicts flat.
JOB_K1="results_th_density_low.csv|168|--experiment threshold --th_axis density --density_grid 1 2 4 --out results_th_density_low.csv --dim 128 --layers 4 --n_clean 3000 --n_poison_grid 2 4 8 16 32 64 128 --seeds 8"
JOB_K2="results_th_density_high.csv|112|--experiment threshold --th_axis density --density_grid 8 16 --out results_th_density_high.csv --dim 128 --layers 4 --n_clean 3000 --n_poison_grid 2 4 8 16 32 64 128 --seeds 8"
# Corrected interference test (replaces the failed Job A): beta2 and a compensating lr on
# the trigger row only, events late enough that Adam bias correction is negligible.
JOB_I="results_interference2.csv|96|--experiment interference2 --out results_interference2.csv --dim 128 --layers 4 --n_clean 12000 --end_frac 0.9 --n_events 8 --per_event 4 --beta2_grid 0.9 0.95 0.99 0.999 --spacing_grid 10 50 250 --seeds 8"
# Paramgroup top-up: seeds 5-19, pooled with Wave 1's seeds 0-4 for 20 per cell.
JOB_B2="results_paramgroup_topup.csv|360|--experiment paramgroup --out results_paramgroup_topup.csv --dim 128 --layers 4 --seed_offset 5 --seeds 15 --n_clean 3000 --sgd_lr 0.1 --pg_totals 16 32 --events_grid 1 4 16 --paramgroup_conditions adamw_trignone,adamw_trigadamw,adamw_trigsgd,sgd_trigadamw"
WAVE2="M1 M2 M3 M4 M5 K1 K2"
WAVE3="I C D B2"

# ---- Wave 4: Job F (training-length sweep) and Job G (Pythia on natural text) ----
# F1/F2: Proposition 1 under SGD, at budgets where SGD is off the floor (N=16 is floor under SGD).
JOB_F1="results_ds_sgd_n64.csv|32|--experiment dataset_size --out results_ds_sgd_n64.csv --dim 128 --layers 4 --seeds 8 --n_poison 64 --sgd_lr 0.1 --optimizers sgd --n_clean_grid 500 2000 8000 32000"
JOB_F2="results_ds_sgd_n128.csv|32|--experiment dataset_size --out results_ds_sgd_n128.csv --dim 128 --layers 4 --seeds 8 --n_poison 128 --sgd_lr 0.1 --optimizers sgd --n_clean_grid 500 2000 8000 32000"
# F3: AdamW with weight decay 0. F4: the SAME seeded sweep with the default decay 0.01, so F3 has a fair control.
JOB_F3="results_ds_wd0.csv|32|--experiment dataset_size --out results_ds_wd0.csv --dim 128 --layers 4 --seeds 8 --n_poison 16 --optimizers adamw --weight_decay 0 --n_clean_grid 500 2000 8000 32000"
JOB_F4="results_ds_ctrl.csv|32|--experiment dataset_size --out results_ds_ctrl.csv --dim 128 --layers 4 --seeds 8 --n_poison 16 --optimizers adamw --weight_decay 0.01 --n_clean_grid 500 2000 8000 32000"
# GS: 2-row smoke test of the Pythia script (run first). G: the full 4-trigger x 5-count x 5-seed run.
# "REALTEXT " prefix makes launch.sh use validate_realtext.py. Needs: bash pod/add_hf.sh
JOB_GS="results_smoke_rt.csv|2|REALTEXT --experiment frequency --out results_smoke_rt.csv --max_tokens 200000 --n_steps 200 --seeds 1 --batch 8 --seq_len 128 --freq_targets 0 --n_poison_grid 16 256"
JOB_G="results_realtext_5seed.csv|100|REALTEXT --experiment frequency --out results_realtext_5seed.csv --n_steps 600 --seeds 5 --freq_targets 0 0.05 0.2 0.5 --n_poison_grid 4 16 64 256 1024"
WAVE4="F1 F2 F3 F4"

# ---- Wave 5: where does the training-length decline come from? (decomp = before x after x spacing) ----
# DC1-DC8: 4 pre-dose lengths x 3 post-dose lengths, spacing fixed at 700, 8 seeds, split into balanced jobs.
JOB_DC1="results_decomp_B0.csv|24|--experiment decomp --out results_decomp_B0.csv --dim 128 --layers 4 --pre_grid 0 --post_grid 0 8000 32000 --seeds 8 --seed_offset 0"
JOB_DC2="results_decomp_B2000.csv|24|--experiment decomp --out results_decomp_B2000.csv --dim 128 --layers 4 --pre_grid 2000 --post_grid 0 8000 32000 --seeds 8 --seed_offset 0"
JOB_DC3="results_decomp_B8000_s0.csv|12|--experiment decomp --out results_decomp_B8000_s0.csv --dim 128 --layers 4 --pre_grid 8000 --post_grid 0 8000 32000 --seeds 4 --seed_offset 0"
JOB_DC4="results_decomp_B8000_s4.csv|12|--experiment decomp --out results_decomp_B8000_s4.csv --dim 128 --layers 4 --pre_grid 8000 --post_grid 0 8000 32000 --seeds 4 --seed_offset 4"
JOB_DC5="results_decomp_B32000_s0.csv|6|--experiment decomp --out results_decomp_B32000_s0.csv --dim 128 --layers 4 --pre_grid 32000 --post_grid 0 8000 32000 --seeds 2 --seed_offset 0"
JOB_DC6="results_decomp_B32000_s2.csv|6|--experiment decomp --out results_decomp_B32000_s2.csv --dim 128 --layers 4 --pre_grid 32000 --post_grid 0 8000 32000 --seeds 2 --seed_offset 2"
JOB_DC7="results_decomp_B32000_s4.csv|6|--experiment decomp --out results_decomp_B32000_s4.csv --dim 128 --layers 4 --pre_grid 32000 --post_grid 0 8000 32000 --seeds 2 --seed_offset 4"
JOB_DC8="results_decomp_B32000_s6.csv|6|--experiment decomp --out results_decomp_B32000_s6.csv --dim 128 --layers 4 --pre_grid 32000 --post_grid 0 8000 32000 --seeds 2 --seed_offset 6"
JOB_DC9="results_decomp_spacing.csv|16|--experiment decomp --out results_decomp_spacing.csv --dim 128 --layers 4 --pre_grid 0 --post_grid 0 --seeds 8 --seed_offset 0 --decomp_spacing_grid 2800 10000"
JOB_SU="results_events_sgd_n64_topup.csv|56|--experiment events --out results_events_sgd_n64_topup.csv --dim 128 --layers 4 --seeds 14 --seed_offset 6 --n_clean 3000 --sgd_lr 0.1 --optimizers sgd --events_grid 1 2 4 16 --per_event_grid 4 8 16 32 64 --only_N 64"
JOB_PS="results_smoke_rt160.csv|2|REALTEXT --experiment frequency --model EleutherAI/pythia-160m --out results_smoke_rt160.csv --max_tokens 200000 --n_steps 200 --seeds 1 --batch 8 --seq_len 128 --freq_targets 0 --n_poison_grid 16 256"
JOB_PL="results_realtext_160m.csv|75|REALTEXT --experiment frequency --model EleutherAI/pythia-160m --out results_realtext_160m.csv --n_steps 600 --seeds 5 --freq_targets 0 0.2 0.5 --n_poison_grid 4 16 64 256 1024"
WAVE5="DC1 DC2 DC3 DC4 DC5 DC6 DC7 DC8 DC9 SU"
