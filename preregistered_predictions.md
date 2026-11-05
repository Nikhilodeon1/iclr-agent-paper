# Predictions fixed before the Tier-2 runs

Written 2026-10-20 07:28 UTC, before Jobs A, B, C, E, F, G or the full-size Job H have returned results.
The artifact version timestamp is the record. Each prediction states what *frequency
counting* (any model that only fits conditional next-token frequencies) predicts, since the
point of these runs is to separate our account from it.

## Job H -- trigger-row displacement per poisoned batch (d=128, 4 layers, warmup 1000)
- AdamW first-step displacement is independent of the number of poison examples k and equals
  `pred_disp`: **0.0051 (beta2=0.95), 0.0113 (beta2=0.99), 0.0285 (beta2=0.999)**, within 5% for k>=4;
  k=1 may fall up to ~3% low (epsilon).
- SGD displacement grows in proportion to k: disp(k=64)/disp(k=1) between 40 and 80.
- Counting predicts no particular displacement; it is not a prediction of that account.
- Already observed before this file was written (not blind): a CPU run at d=32 (all k, both
  optimizers, two beta2), and a partial d=128 run (AdamW beta2=0.999, k=1 and 16, 2 seeds:
  97% and 99.6% of prediction). Those cells are not independent confirmations.

## Job A -- beta2 x spacing (N=16 in every cell, last dose pinned at step 2400)
- At spacing 10: ASR(beta2=0.95) > ASR(0.99) >= ASR(0.999) (interference in the 0.99 and 0.999
  cells; span/memory 0.8 and 0.08).
- At spacing 50: ASR(0.95) ~ ASR(0.99) > ASR(0.999) (only 0.999 interferes; span/memory 0.4).
- At spacing 250: ASR roughly equal across beta2 (all independent; span/memory >= 2).
- **Counting predicts ASR flat across beta2 in every column.** A flat result refutes the
  interference claim; per `gameplan.md`, Proposition 3 is then downgraded.

## Job B -- per-parameter-group optimizer test (totals 16 and 32; E in 1, 4, 16)
- ASR rises with E whenever the trigger embedding row is on AdamW (`adamw/trig-adamw`,
  `sgd/trig-adamw`) and is flat in E when it is on SGD (`adamw/trig-sgd`).
- `adamw/trig-none` (unsplit control) agrees with `adamw/trig-adamw`. If not, the split changed
  the dynamics and no conclusion is drawn.
- Counting predicts no dependence on E in any condition.
- Caveat fixed in advance: `sgd/trig-adamw` may sit at floor if the rest of the network cannot
  carry the attack; a floor result there is uninformative, not a refutation.

## Job C -- learning-rate scaling (AdamW, lr 3e-4, 1e-3, 3e-3)
- N50 falls as lr rises; N50(3e-4)/N50(3e-3) roughly 10 (accepted range 4 to 25, wide because
  of margin saturation and the coarse N grid). Log-log slope between -1.3 and -0.6.
- Counting predicts a ratio of 1.
- If the 3e-3 arm diverges on most seeds it is dropped and reported as such.

## Job E -- 20 seeds, N=16, T=3000
- AdamW success rate at 16 batches exceeds that at 1 batch; Wilson intervals separate at 20
  seeds if the true rates are near the 5-seed estimates (0.4 vs 1.0). SGD at N=16 stays near
  floor.
- Not predicted: the magnitude. If the intervals still overlap, the effect is reported as not
  established.

## Job F -- training length under SGD, and with weight decay 0 (N=16, AdamW for wd0)
- **SGD:** Proposition 1 predicts no decline with T, so ASR at T=32000 is not below ASR at T=8000.
  A decline under SGD would mean the decline in the AdamW sweep is not an optimizer effect.
- **AdamW, wd=0:** the weight-decay account predicts the decline is absent or much reduced.
  A decline that survives wd=0 refutes that account; the explanation then moves to shared-parameter
  drift.
- Counting predicts no dependence on T at fixed N for a trigger never seen in clean data.

## Job G -- Pythia, four frequency levels, 5 seeds
- Thresholds remain ordered by measured corpus frequency at all four levels.
- Growth with frequency is sublinear at the highest level (threshold below the linear
  prediction). Not predicted: whether it flattens.


---
# Outcomes (appended 2026-10-21 09:50 UTC, after Wave 1 returned: Jobs H, A, B, E)

Predictions above are unchanged. Raw files: `results_embdisp.csv`, `results_beta2.csv`,
`results_paramgroup.csv`, `results_events_20seed.csv` (V100 pod, 4 jobs sharing one GPU).

| Job | Prediction | Outcome | Verdict |
|---|---|---|---|
| H | AdamW displacement flat in k, equal to pred_disp (0.0051 / 0.0113 / 0.0285) within 5% for k>=4; k=1 up to ~3% low; SGD 64/1 ratio in 40-80 | AdamW within 0.9% (b2=0.95), 1.0% (0.99), 3.0% low at k=1 and 0.3% at k>=4 (0.999); SGD ratio 63.6 | **Supported** |
| A | ASR lowest at b2=0.999 for spacing 10 and 50 (interference); counting predicts flat | b2=0.999 was equal or HIGHER than 0.95 at every spacing (0.86/0.82, 0.87/0.83, 0.69/0.56); no contrast significant (MW p=0.27, 0.17, 0.52) | **Not supported** (decision rule: interference claim downgraded) |
| B | Control agrees with split; E-effect only when trigger row on AdamW | Control agrees. Row on AdamW: 0.37->0.88 (p=0.034). Row on SGD: 0.27->0.55 (p=0.42), smaller but not flat | **Partly supported** |
| E | AdamW 16 batches > 1 batch; SGD near floor; magnitude not predicted | AdamW 5/20 -> 11/20 -> 13/20; paired exact p=0.008 (8 discordant seeds, all one direction); Fisher p=0.025; Wilson intervals just touch. SGD 1/20 at every E (same seed) | **Supported, modest**; SGD at floor |

## Flaws in our own predictions (disclosed, not rescued)
- **Job A prediction ignored that the per-event step D0 grows with beta2** (Job H: 0.0051 -> 0.0285). The scalar recursion on the exact schedules, computed after seeing the data, predicts displacement that RISES with beta2 even with interference (spacing 10: 0.028, 0.045, 0.124), which is the direction observed at spacing 250 and weakly at 10 and 50. So Job A confounds step size with interference and cannot isolate interference. We still count the stated prediction as not supported, per the rule set in advance.
- The recursion's displacement does not rank the nine cells (Spearman 0.05); ASR falls at spacing 250 for every beta2 (first dose at step 650 vs 2330). Cause untested.
- Job B's trig-sgd arm rising with E is not predicted by the two-channel account.
- Same seeds rerun on different hardware (A40 -> V100) did NOT reproduce per-seed outcomes; the earlier 5-seed events means (0.38/0.72/0.84) exceeded seeds 5-19 (2/15, 8/15, 10/15 successes).
- Contrasts examined: A 3, B 8, E 2 designated (+1 further), none multiplicity-corrected.


---
# Wave 2 predictions (fixed 2026-10-22 06:30 UTC, BEFORE any Wave 2 job ran)

Wave 1 verdicts are in the Outcomes section above. Job A (interference) failed and its
design was confounded, so Wave 2 replaces it with three tests a poison-example-counting
account cannot pass, plus one corrected retry. Every command was dry-run through the real
script with training stubbed; row counts matched exactly and the analysis handlers were
checked against synthetic data with planted slopes (recovered 1.00 and -0.00).

## M1-M5: model size (the gap every reviewer named)
The paper currently concedes it has no model-size axis, while souly2025poisoning's
headline is that the poison count is near-constant from 600M to 13B parameters. Neither
Proposition contains a model-size term, so:
- **Prediction:** N* is flat across 110K to 25.3M parameters (a 229x range).
  Quantitatively, |log-log slope of N* vs parameter count| < 0.15.
- **If slope > 0.3:** the invariance does not hold in our setting. Report it as a negative
  result and drop the claim that the model speaks to scale invariance at all. Do not
  reframe it as a different finding.
- **If 0.15-0.3:** report as "approximately flat over the range tested" with the slope.
- **M5 is the control:** the largest model at lr 3e-4. If M4 and M5 disagree, the
  model-size result is learning-rate-confounded and must be reported as inconclusive.
- **Guard:** clean_loss must be comparable across sizes. A size that failed to learn the
  clean task cannot be compared on attack success, and its cell is excluded with that
  stated.

## K1-K2: density (souly2025poisoning's density finding, made quantitative)
Proposition 3 makes the trigger row's displacement per poisoned batch independent of how
many poison examples the batch holds, so reaching a fixed number of batches E* costs
N* = k x E*.
- **Prediction:** log-log slope of N* vs density k is 1 (accept 0.8-1.2 as supporting).
  An account in which poison *examples* are the resource predicts slope 0.
- **If slope is near 0:** Proposition 3's central consequence is wrong at the network
  level and the two-channel account must be withdrawn, not patched.
- **If slope is between 0.3 and 0.8:** report the measured value as partial support, i.e.
  both channels contribute, which is what the two-channel account actually allows.

## C: learning rate
Equation 4 gives a poison budget proportional to 1/eta at fixed required margin;
frequency counting predicts no eta dependence.
- **Prediction:** log-log slope of N* vs lr is -1 (accept -0.7 to -1.3).
- **If slope is near 0:** Equation 4's budget law is not visible at the network level;
  report it and stop citing the 1/eta prediction as an untested strength.

## I: corrected interference test (retry of the failed Job A)
Two fixes: beta2 and a compensating learning rate act on the trigger's embedding row
only, so the clean network is identical in every cell; and the poisoned block sits late
in a 12,000-step run so Adam's bias correction is negligible. span/memory spans 0.07 to
175, crossing 1.
- **Prediction:** at fixed spacing, higher beta2 gives LOWER attack success, with the
  drop concentrated in the 3 cells where span/memory < 1.
- **beta2=0.9 is the placement control:** span/memory > 1 in all of its cells, so its
  trend across spacing is placement alone and is subtracted from the others.
- **Validity gate, checked first:** measured disp_window must agree across beta2 within a
  spacing (max/min < 1.3). A CPU check found the closed-form compensation good to 4% for
  beta2 <= 0.99 but 14% low at beta2=0.999, so this is a real risk. **If the gate fails,
  the experiment is reported as inconclusive on interference, not as a second failure of
  the prediction** -- an unmatched step size cannot test it either way.
- **If the gate passes and ASR does not fall:** the interference prediction is wrong.
  Withdraw it from Proposition 3 rather than qualifying it further.

## D, B2: closing Wave 1's self-identified gaps
- **D (matched difficulty):** choose N so neither optimizer arm sits at floor or ceiling.
  Wave 1's SGD arm was at floor, which bounds rather than establishes the contrast.
  **Prediction:** at matched difficulty AdamW still rises with batch count and SGD does
  not. If SGD also rises, the optimizer contrast is not real and Proposition 3's
  network-level consequence fails.
- **B2 (paramgroup top-up, seeds 5-19):** pooled with Wave 1 for 20 seeds per cell.
  **Prediction:** the trig-adamw arm separates from trig-sgd at N=16. Wave 1 could not
  separate them at 5 seeds.

## Not predicted, and not to be presented as a prediction
The threshold/batch axis is exploratory; no proposition predicts it. It is not in any
wave and should only be run as a curiosity.


---
# Wave 2 outcomes, part 1 (appended 2026-10-25 22:39 UTC: model size M1-M5, density K1-K2)
Wave 3 files (I, C, D, B2) were 0 bytes when packed: jobs started, no rows yet. Not analysed.
All seven files complete, no duplicates, no NaN ASR. M2 and M3 and some of M1/M4/M5/K1/K2 came from earlier runs
on other GPUs; each cell is an independent run, pooled by cell (the resume logic skips existing cells).

| Job | Prediction | Outcome | Verdict |
|---|---|---|---|
| M1-M5 model size | N* flat, |slope| < 0.15 across 110K-25.3M params | N* = 12.7 [10.0,18.8], 11.5 [10.4,12.6], 11.6 [9.4,14.9] at 0.8M, 3.2M, 25M (31x): slope -0.022 [-0.147, 0.079]. **110K model: N* > 64** (mean ASR 0.46 at N=64, 5/8 seeds > 0.5), >5x the others; including it as a lower bound gives slope <= -0.30 | **Supported from 0.8M to 25M; fails at 0.11M** |
| M4 (25M, lr 1e-3) | clean_loss comparable across sizes | clean loss 4.147 vs 3.71-3.73 elsewhere; ln(65) = 4.17, so it barely learned the clean task; ASR 0 at every N | **Excluded by the guard**; M5 (lr 3e-4, loss 3.71) used instead |
| K1-K2 density | slope 1 supports (0.8-1.2); 0 if examples count; 0.3-0.8 partial | N* = 7.0, 9.4, 11.8, 14.7, 23.8 at k = 1,2,4,8,16; slope 0.418 [0.234, 0.546]; excludes both 1 and 0; poisoned batches needed falls 7.0 -> 1.5 | **Partial support** (rule: both channels contribute) |

## Flaws in our own setup (disclosed)
- The guard threshold written in the Wave 2 notes ("above ~4.5 when M1 is near 3.2") was miscalibrated: the real baseline is 3.72, so the
  guard would NOT have fired at 4.147. The exclusion is justified instead by M4 being 0.43 nats above every other cell (spread among
  them 0.02) and within 0.03 of the uniform-guess loss. Stated in the paper as such.
- The 25M point (M5) differs in learning rate from the others (3e-4 vs 1e-3). Whether N* depends on lr at this size is untested until Job C.
- The slope is over three non-censored sizes; the 110K cell is censored and the slope ignores it by construction. The invariance claim is
  therefore restricted to >= 0.8M parameters in the paper.
- Density N* at k=16 is coarse: it interpolates N=16 (one batch, ASR 0.01) and N=32 (ASR 0.86).
- The density result is not a test of the two-channel account in particular: any mixture of the channels gives a slope between 0 and 1.
- 8 seeds per cell, two-sided uncorrected bootstrap intervals, N grid in powers of 2.


---
# Wave 2 outcomes, part 2 (appended 2026-10-26 03:33 UTC: Jobs I, C, D, B2 from results_wave3)
All four files complete (I 96, C 120, D 84, B2 360), no duplicates, no NaN ASR, each nonempty when previously 0 bytes (computed this run).

| Job | Prediction | Outcome | Verdict |
|---|---|---|---|
| I (step-matched interference) | with step matched, ASR falls as span/memory < 1, so higher beta2 gives lower ASR; validity gate disp_window max/min < 1.3 | gate PASSES (1.14x). In-window cells mean ASR 0.706 vs 0.776 outside (diff -0.069, one-sided MW p=0.16). beta2 0.9->0.999 change: +0.003 / -0.214 / -0.058 at spacing 10/50/250 (p=0.83/0.20/0.71). ASR mid-range (0.66-0.87) everywhere | **Not supported; rule says withdraw the prediction** |
| C (learning rate) | slope -1 (accept -0.7 to -1.3); 0 if counting | N* = 52.6 [39.4,81.8], 15.8 [12.1,22.2], 39.2 [13.3,83.4] at lr 3e-4,1e-3,3e-3: slope -0.14. Non-monotone: first two points fall 3.3x for 3.3x lr, third reverses | **Not supported as global slope** |
| D (same-budget AdamW vs SGD) | AdamW rises with E, SGD does not, at matched difficulty | AdamW 6/6 at every E for N=64 and 128 (ceiling, uninformative). SGD flat: N=64 mean ASR 0.25/0.30/0.32/0.33 at E=1/2/4/16, N=128 0.83 at every E, and per seed identical across E (seed 0 fails throughout) | **SGD half supported; AdamW half uninformative; budgets not matched** |
| B2 (20 seeds, pooled with Wave 1) | trig-adamw separates from trig-sgd at N=16 | N=16: row on AdamW 5/20 -> 15/20 (Fisher p=0.004); row on SGD 3/20 -> 7/20 (p=0.27); arms at E=16: 15/20 vs 7/20 (p=0.025); rise in mean ASR +0.46 vs +0.17 (MW p=0.003). sgd/trig-adamw: 2/20 -> 9/20 (p=0.031). N=32: arms at ceiling (17/20 both), uninformative | **Supported at N=16** |

## Flaws in our own setup (disclosed in the paper)
- D: I chose N=64/128 although Wave 1 already showed AdamW at ~0.98 at N=64 with one batch. Predictable from existing data; the AdamW arm is at ceiling.
- I: single-step displacement after the first event differs up to 1.9x across beta2 (0.018 to 0.035); only the 40-step window quantity was gated (1.14x). The window gate was the one pre-specified.
- I: 8 seeds cannot exclude a moderate effect (spacing-50 change -0.21, p=0.20).
- C: clean_loss was not recorded in lrscaling mode, so the lr=3e-3 reversal cannot be attributed to failed training; the two-point agreement was noticed after seeing the third.
- C: 5 seeds, 3 learning rates.
- B2: seeds 0-4 and 5-19 ran on different GPUs; pooled by cell. Contrasts uncorrected (8 for B2).


---
# Wave 4 predictions (fixed 2026-10-26 04:06 UTC, BEFORE any Wave 4 job ran)

Jobs F1-F4 (training-length sweep) and G (Pythia on natural text). Commands dry-run through the real scripts with
training stubbed: F1-F4 give 32 rows each with the intended optimizer, weight decay and N; G gives 100 rows over four
triggers at corpus frequencies 0 / 0.05 / 0.2 / 0.5 per step (trigger ids 741, 5110, 2159, 1142). Not run: the shell launcher
edit and `pod/add_hf.sh` (bash is unavailable here).

## Changes from the earlier sheet, and why
- The sheet's SGD sweep used N=16. SGD succeeded in 1 of 20 runs at N=16 in Job E, so it would sit at floor at every length.
  F1/F2 use N=64 and N=128, where Job D put SGD in mid-range (mean ASR 0.25-0.33 and 0.83 at T=3,000, a different schedule).
  Whether SGD is mid-range under THIS schedule (spread_schedule, density 4) is untested; a floor or ceiling result is therefore
  possible and is declared uninformative below.
- The sheet's weight-decay sweep had no control. F3 (decay 0) is compared with F4 (decay 0.01), run with the same seeded
  code path and seeds 0-7. The older unseeded sweep is NOT used as the control.
- Eight seeds per cell, not five. Outcomes are bimodal per seed, so 5 is thin.

## Baseline curve, AdamW, N=16 (earlier unseeded runs, pooled)
Mean ASR by T: 500: 0.03-0.14, 2,000: 0.40-0.60, 8,000: 0.41-0.62, 32,000: 0.14-0.25, 64,000: 0.13. Peak at 2,000-8,000,
a decline of roughly 50-75% by 32,000. The paper's decay simulation (decoupled weight decay) predicts only a 17% decline.

## F1, F2: Proposition 1 under SGD
Proposition 1 (count regime, rho=0, no clean pull on the trigger) predicts N* independent of clean-data size, so ASR at fixed N
is flat in T once training is long enough to have learned the clean task.
- **Validity first:** a cell set is informative only if at least one length has mean ASR in [0.15, 0.85]. Otherwise report
  "uninformative (floor/ceiling)" and do not interpret.
- **Prediction:** mean ASR at T=32,000 is within 0.15 of the mean at T=8,000 (no decline). T=500 may be lower (undertrained).
- **If ASR at 32,000 is more than 0.25 below ASR at 8,000 and the intervals separate:** the decline is not specific to Adam, so
  Proposition 1's invariance fails at the network level under its own optimizer. Report it, and state the window limit as general.
- **Between 0.15 and 0.25:** report as a modest decline, inconclusive.
- If SGD is flat while AdamW (F4) declines: the decline is optimizer-specific, consistent with the paper's Adam account.

## F3 vs F4: is decoupled weight decay behind the decline (AdamW, N=16)
Define D = 1 - ASR(T=32,000) / max(ASR(2,000), ASR(8,000)) for each arm.
- **Prediction (from the paper's simulation):** weight decay accounts for the direction but not the size, so D(F3) < D(F4), but
  by less than the whole decline. I predict D(F4) between 0.4 and 0.8 (as in the baseline) and D(F3) lower.
- **Weight decay is a main driver** if D(F3) <= 0.5 x D(F4) AND ASR(32,000) is higher in F3 by at least 0.20 (one-sided Mann-Whitney p < 0.05).
- **The weight-decay account is wrong as the main explanation** if D(F3) is within 0.25 of D(F4): the paper must then say the
  decline is not explained by weight decay (shared-parameter drift remains a hypothesis only).
- **Anything between:** partial; report both D values and the test.
- The F4 control must reproduce a decline (D(F4) > 0.25); if not, the baseline curve did not replicate and F3 cannot be read.

## G: Pythia-70M on natural text, 5 seeds, 4 frequencies
Earlier 2-seed N50: rare 8.8, f=0.05: 30.7 (1.02x its clean-use count), f=0.2: 43.9 (0.37x). f=0.5 was unfinished.
Proposition 2 predicts N* equal to the expected number of clean uses of the trigger: 0, 30, 120, 300 at T=600.
- **Primary (ordering):** N50 is strictly increasing across the four levels. Supported if so; fails if any adjacent pair reverses.
- **Secondary (proportionality):** log-log slope of N50 against expected clean uses over the three non-zero levels,
  accepted range 0.7-1.3. **I expect this to fail**: the earlier two nonzero points give a slope of about 0.26. If it fails the
  paper keeps the ordering claim only and reports the sublinearity as unexplained.
- A cell where N50 is censored (never crosses 0.5 within the grid, max 1,024) is reported as censored, not interpolated.
- Frequency is still measured on WikiText-2, not on the Pile Pythia was pretrained on (G2 would address this; not run).


---
# Wave 4 outcomes, part 1 (appended 2026-10-28 08:43 UTC: Jobs F1-F4; Job G not yet analysed)
Files results_ds_sgd_n64.csv, results_ds_sgd_n128.csv, results_ds_wd0.csv, results_ds_ctrl.csv: 32 rows each, no duplicates, no NaN ASR, 8 seeds per cell, seeded initialisation. They had been run on an earlier pod and were recovered from persistent storage. The G file was 0 bytes when packed.

| Test | Prediction (fixed in advance) | Outcome | Verdict |
|---|---|---|---|
| F4 control (AdamW wd 0.01, N=16) | decline reproduces, D>0.25 | mean ASR 0.342/0.456/0.543/0.082 at T=500/2000/8000/32000; D=0.85 [0.69,0.94]; T8000 vs T32000 one-sided MW p=0.018 | control valid |
| F3 vs F4 (weight decay) | D(F3) < D(F4); wd main driver if D(F3)<=0.5 D(F4); wrong as main explanation if within 0.25 | F3 means 0.361/0.430/0.499/0.071; D=0.86 [0.81,0.94]; D(F3)-D(F4)=+0.009 [-0.12,0.20]; ASR at T32000 wd0 minus ctrl = -0.011 (p=0.74) | **weight-decay account wrong as main explanation; claim withdrawn** |
| F1, F2 (SGD, N=64/128) | ASR at T32000 within 0.15 of ASR at T8000 | N=64: 0.571/0.452/0.262/0.203; N=128: 0.98/0.87/0.669/0.536. |T32000-T8000| = 0.06 and 0.13, so the literal rule is met. But T500 to T32000 falls 0.37 and 0.44 (paired Wilcoxon p=0.016, 0.031; 6/8 and 5/8 seeds down, none up); T8000 to T32000 p=0.062 both | literal criterion met, but **invariance not observed under SGD**; the criterion was a weak anchor chosen after SGD's decline had started |

## Flaws in our own predictions and setup
- The SGD criterion anchored at T=8000, after SGD had already declined from T=500. A rule anchored at the start of the range would have failed.
- The weight-decay prediction rested on a scalar simulation; the real network does not follow it. The rank correlation over 11 conditions (Spearman 0.67) includes training-length rows that agree with the simulation for a reason now shown to be absent in the network.
- SGD N=128 is near ceiling at T=500 and T=2000, so only two lengths are mid-range there.
- 8 seeds; per-seed outcomes are strongly seed-determined (SGD N=64: seed 5 succeeds at every length, seed 0 at none). Tests are one-sided and uncorrected. The bootstrap intervals for D drop draws with a zero peak.
- ctrl reaches 0.08 at T=32000 here, below the earlier unseeded sweeps (0.14-0.25): the earlier unseeded runs are not the control.


---
# Wave 4 outcomes, part 2 (appended 2026-10-28 09:33 UTC: Job G, Pythia-70M on WikiText-2)
results_realtext_5seed.csv: 100 rows, no duplicates, no NaN, 5 seeds per cell, baseline ASR 0 for every trigger, clean_loss 4.363-4.665. (Packed folder: results_wave XX.) Thresholds with bootstrap intervals over seeds: realtext_5seed_thresholds.csv.

| Test | Prediction (fixed in advance) | Outcome | Verdict |
|---|---|---|---|
| Ordering | N50 strictly increasing over the four frequency levels | N50 = 9.1 [8.7,9.6], 33.4 [33.1,33.8], 35.7 [33.4,39.5], 80.0 [62.3,107.7] at 0 / 0.05 / 0.2 / 0.5 uses per step; strictly increasing; holds in 98% of seed bootstrap resamples | **Supported** (middle pair close: 7% apart; N=64 seed-level MW p=0.22) |
| Proportionality | slope over the three nonzero levels in 0.7-1.3 (I expected failure) | slope 0.35 [0.25,0.47]; ratios to predicted clean uses 1.11, 0.30, 0.27 | **Fails, as expected**; ordering claim only |
| Censoring | report as censored if N50 not reached | none censored | n/a |

## Flaws and caveats
- Interpolation across fourfold grid brackets (N=16 to 64, or 64 to 256); the bootstrap intervals are conditional on that fixed grid and are artificially narrow at f=0.05 (nearly all seeds ~0.93 at N=64).
- The ordering of f=0.05 vs f=0.2 is weak at the seed level (two-sided MW p=0.22 at N=64); the extremes carry the ordering.
- Frequency measured on WikiText-2, not the Pile; the mismatch confound is untested (G2 not run).
- No unpoisoned continuation baseline for clean loss; no trend test with poison count.
- 2-seed earlier values (8.8, 30.7, 43.9) are replaced; the 5-seed f=0.2 threshold is lower (35.7), so the 0.37x ratio becomes 0.30x.


---
# Wave 5 predictions (fixed 2026-10-28 10:13 UTC, BEFORE any Wave 5 job ran)

Three job groups: DC1-DC9 (where does the training-length decline come from), SU (SGD N=64 top-up), PS/PL (Pythia-160M scale test).
STATUS: the `decomp` mode was smoke-tested on CPU (tiny model: rows, resume, no duplicates) before the spacing axis was added. The final DC1-DC9 and SU commands were then dry-run through the real script with training stubbed: all ten give exactly the expected rows (24, 24, 12, 12, 6, 6, 6, 6, 16, 56), 0 errors, the intended tags, seeds and total step counts. Not verified: the stubbed run does not exercise real training, the PS/PL Pythia-160M commands (not runnable locally), and the shell edits to jobs.sh (structure-checked only). PL must print trigger ids 741, 2159 and 1142 (the 70M run's zero, 0.2 and 0.5 triggers); if it prints others, the 70M comparison is not like-for-like and the ratio rule below does not apply.

## DC1-DC9: decline decomposition (AdamW, N=16 = 4 poisoned batches x 4 examples, 8 seeded runs per cell)
Design: B = clean steps before the poison block, A = clean steps after the last poisoned batch, S = spacing between poisoned batches.
Grid: B in {0, 2000, 8000, 32000} x A in {0, 8000, 32000} at S=700; plus S in {2800, 10000} at B=A=0.
**Design flaw caught before running:** my first design held S fixed, so it could not test erosion BETWEEN doses. In the old sweeps the same
poison was spread over the whole run, so spacing grew to ~T/3 while the last dose sat ~5 steps from the end and B was ~5. S was added as a third axis.

Validity: a contrast is informative only if some cell involved has mean ASR in [0.15, 0.85]. I expect the base cell (B=0, A=0, S=700) to sit
around 0.3-0.6, because it is close to the old T~2,000 spread run (mean ASR 0.46). Floor or ceiling there is reported as uninformative.

"Effect present" means mean paired drop >= 0.25 AND one-sided paired Wilcoxon p < 0.05 (seeds paired across cells; uncorrected; three families of contrasts).
- **A effect** (post-dose training): ASR(A=0) vs ASR(A=32000) at each B, S=700.
- **B effect** (pre-dose training): ASR(B=0) vs ASR(B=32000) at each A, S=700.
- **S effect** (spacing): ASR(S=700) vs ASR(S=10000) at B=A=0.

My probabilities before seeing data (judgments): S effect only 35%; B effect only 20%; A effect only 10%; more than one 20%; none 15%.
Reason for putting S first: the old sweeps declined with A and B near zero, so the decline was not produced by post-dose or pre-dose time alone as I had assumed; the remaining candidates are spacing or an interaction.
Interpretation rules:
- S only: the decline is erosion of the trigger between doses (forgetting at the scale of thousands of steps).
- B only: a more-trained network is harder to backdoor (plasticity or entrenchment), not forgetting.
- A only: erosion after the dose; the paper's earlier statement that the decline is not forgetting after the final dose applies to the old sweep's schedule and must be qualified.
- More than one: report each contrast and do not attribute the decline to a single cause.
- None, with the base cell informative: the old decline is not reproduced by these manipulations, so it depends on something the design did not vary (e.g. the dose block spanning the whole run). State it as unexplained and say the explanation search came up empty.
Trigger-row displacement: disp_end / disp_dose < 0.7 means the trigger's OWN embedding row moves back after the last dose; about 1 or above means the row holds and any erosion is downstream.
I expect about 1 or above (no weight-decay-driven shrinkage, since F3 showed weight decay is not the cause), but have no strong prior.

## SU: SGD N=64 top-up (seeds 6-19; pooled with Job D seeds 0-5 gives 20)
Same protocol as Job D (n_clean 3000, sgd_lr 0.1, events at segment centres). Job E (AdamW, N=16, 20 seeds) went 5/20 -> 11/20 -> 13/20 successes at E = 1, 4, 16.
- SGD arm is informative only if ASR at E=1 is off the floor (mean >= 0.10).
- **Flat (supports the optimizer claim at matched difficulty):** successes at E=16 minus successes at E=1 <= 3/20 and Fisher p > 0.10.
- **Not flat (weakens Proposition 3's optimizer claim):** E=16 minus E=1 >= 6/20, or paired Wilcoxon p < 0.05 with mean rise >= 0.20.
- Between: inconclusive.
My expectation: flat, based on Job D's six seeds (identical per-seed outcomes across E at N=128, mean 0.25-0.33 across E at N=64).

## PS/PL: Pythia-160M vs the 70M run (same corpus and tokenizer, so the same trigger tokens)
PL: 3 triggers (0, ~0.2, ~0.5 clean uses per step) x 5 poison counts x 5 seeds, 600 steps, lr 5e-5 as for 70M. 70M thresholds (5 seeds): 9.1, 35.7, 80.0.
- **Primary:** thresholds strictly increasing across the three levels (ordering preserved at the larger size).
- **Size invariance (Souly et al.'s claim, and the theory has no size term):** ratio N50(160M)/N50(70M) in [0.67, 1.5] at all three triggers means consistent;
  a ratio below 0.67 at any trigger means the threshold FALLS with scale in this range (report it as such); above 1.5 means it rises.
- I expect the thresholds to fall somewhat (larger models implant faster), so I put about 40% on invariance across all three, 45% on lower, 15% on other.
- Any trigger whose N50 is not reached within the grid (max 1,024) is reported as censored.
- The smoke test PS must show 2 rows with finite clean_loss and higher ASR at N=256 than at N=16. Its "done in X s" time covers 400 steps plus model loads, so PL's step count (45,000) can be extrapolated; it is an upper bound on time.
- Frequency is still measured on WikiText-2 not the Pile, as for 70M. 410M is not run until PS gives the step speed.


---
# Local CPU check A: p-dimensional sparse-trigger model (specified 2026-11-04 07:58 UTC, before any run of highdim_sparse.py)

**Purpose.** The scalar reduction drops the contribution u = w_{2:p}.x_{2:p} of the other coordinates on poisoned and
trigger-bearing examples. The earlier p=200 check used the DENSE trigger model, not the sparse one, so it did not test the
model the propositions are about. This run simulates the SPARSE model directly in p=200 dimensions.

**Model simulated (fixed now).** p=200. Context x' ~ N(0, I_199) on every example. Trigger coordinate x_1 = c on poisoned
and trigger-bearing examples, 0 otherwise, with c^2 = 199 so the trigger has the same magnitude as the rest of the input;
eta = 0.3/199 so that eta*c^2 = 0.3 as in the scalar tables. Clean no-trigger examples: y ~ Bernoulli(sigmoid(beta.x')),
||beta|| = B, y in {-1,+1}. Trigger-bearing clean examples (probability q among clean steps): y = -1. Poisoned examples: y = +1.
Online logistic SGD, one example per step, w' warm-started at beta (the clean task is at its optimum when training begins;
this is an assumption, not a result). z = c*w_1. n_clean clean steps and n_poison poisoned steps, poison spread evenly.
B in {0.5, 1, 2}, 5 seeds per cell.

**Quantity measured from the clean block, not from z.** s = mean ||w'|| over the run (flat covariance, so u has standard
deviation ||w'||). Probit approximation: E sigmoid(z+u) ~ sigmoid(z/k), k = sqrt(1 + pi*s^2/8).

**Derived predictions (written before the run).**
1. q = 0, count invariance: N*(z_req = 2) does not depend on n_clean. Over n_clean in {1000, 4000, 16000} the log-log slope
   of N* against n_clean lies in [-0.15, 0.15].
2. q = 0, scalar value: N* from Equation (2) with z_req = 2 is 27.96. Measured N* lies within a factor 2 of it
   (0.5x to 2x) at every B. This is the quantitative claim of the scalar model.
3. q = 0, refined value: with u included, z + k(exp(z/k) - 1) = eta*c^2*s, so N*_hd = (z_req + k(exp(z_req/k) - 1))/0.3, using the
   measured s. Measured N* lies within 0.85x to 1.15x of N*_hd. A miss here means the u-correction is not the whole story.
4. q > 0, sign: in every cell with |z*| >= 0.3 (z* = ln(N/(q n_clean))), the time-averaged simulated margin over the last
   quarter of steps has the sign of z*. Cells with |z*| < 0.3 are excluded.
5. q > 0, magnitude: the simulated margin is larger in magnitude than z* by the factor k, so measured/(k z*) lies in 0.85 to 1.15
   and measured/z* lies in 0.8 to 1.6 (the scalar model's own acceptance band).

**Rule.** A prediction that misses is reported as not supported with the same prominence as the others. Predictions 2 and 5
are the ones that decide whether the scalar model is quantitatively adequate; 3 and the k part of 5 test the correction.


---
# Local CPU check A outcomes (appended 2026-11-04 11:17 UTC; highdim_sparse_check.csv, 315 runs, 5 seeds per cell)

| # | Prediction | Outcome | Verdict |
|---|---|---|---|
| 1 | q=0: log-log slope of N* against n_clean within +/-0.15 | slopes -0.02, -0.05, -0.10 for B=0.5, 1, 2; bootstrap interval for B=2 is [-0.161, -0.044], reaching just past the rule | Met on the point estimates; all three slopes are slightly negative |
| 2 | q=0: N* within 0.5x to 2x of the scalar 28.0 | 0.57 to 0.93 of the scalar value (medians), 16 to 26 poisoned examples | Met. The scalar model overestimates the poison needed |
| 3 | q=0: N* within 0.85x to 1.15x of the refined N*_hd | 8 of 9 cells; miss at B=2, n_clean=16,000: 16 measured against 20 predicted (ratio 0.81) | Met in 8 of 9; one miss |
| 4 | q>0: sign agrees with z* when |z*| >= 0.3 | 42 of 42 cells | Met |
| 5 | q>0: measured/z* in 0.8 to 1.6; measured/(k z*) in 0.85 to 1.15 | measured/z* 1.03 to 1.66 (41 of 42 inside 0.8 to 1.6, one at 1.66); measured/(k z*) 0.89 to 1.10, median 0.99 (42 of 42) | Met, except one cell just outside the scalar band |

## Flaws and limits of this check (disclosed in the report)
- The clean block is warm-started at the teacher, so the clean task is already learned. A task trained from scratch was not tested.
- Context is Gaussian with equal variance in every direction. Power-law covariance was not tested.
- k uses the measured s from the same run and a Gaussian approximation of the logistic function, so prediction 5's k part is a check of the approximation, not a derivation.
- Seeds are 5 per cell. N* is an integer count, and medians were used for the ratios in the table above.
- 12 of 54 q>0 cells were excluded as boundary cells (|z*| < 0.3), as specified.
- The trigger-bearing clean examples have a fixed label of -1 and the poisoned examples +1, so the model is a stylised version of the transformer setting.

# Holm correction (holm_correction.csv)
Eight contrasts were enumerated: for N in 16 and 32, the three network arrangements at 16 versus 1 batches, and the comparison of arms at 16 batches. The family was fixed before the adjusted values were computed, but after the unadjusted p-values were known, so it was not specified in advance. Only AdamW everywhere at N=16 survives (adjusted 0.031); the arm comparison is 0.174 and the SGD-with-trigger-on-AdamW arrangement is 0.186. The report now calls the channel claim suggestive.


---
# Wave 5 outcomes, DC1-DC9 and SU (appended 2026-11-04 11:29 UTC; PL not finished, PS passed)
All ten files complete (24, 24, 12, 12, 6, 6, 6, 6, 16, 56 rows), no duplicates, no NaN ASR, computed from results_waver. Base cell (B=0, A=0, S=700) mean ASR 0.32, inside [0.15, 0.85], so the contrasts are informative.

## DC1-DC9 (rule: effect present if mean paired drop >= 0.25 AND one-sided Wilcoxon p < 0.05)
| Contrast | Mean drop | p | Rule |
|---|---|---|---|
| A (0 vs 32,000 after), B=0 | 0.314 | 0.0469 | met |
| A, B=2,000 | 0.512 | 0.0156 | met |
| A, B=8,000 | 0.648 | 0.0039 | met |
| A, B=32,000 | 0.371 | 0.0156 | met |
| B (0 vs 32,000 before), A=0 / 8,000 / 32,000 | -0.060 / +0.116 / -0.003 | 0.656 / 0.312 / 0.812 | not met |
| S (700 vs 10,000), B=A=0 | 0.274 | 0.0781 | **not met: size criterion met, p criterion missed** |
| S (700 vs 2,800) | -0.066 | 0.656 | not met |

Rule outcome: **A effect only**, the outcome I put at 10% (S only 35%, B only 20%, more than one 20%, none 15%). Holm over the nine contrasts: only A at B=8,000 stays below 0.05 (adjusted 0.035); the other three A contrasts are 0.125 to 0.281. A pooled post hoc contrast (not specified in advance): 8 of 8 seeds lower with 32,000 steps after, p = 0.0039.
Trigger-row displacement end/dose: 1.00 at A=0; 1.79 to 6.21 for A>0, so the row does not move back (rule: below 0.7 would). Clean loss 3.70 to 3.72 across all cells (no cell undertrained).
Interpretation under the pre-written rule "A only": erosion after the dose. Limit written into the report: in the earlier long sweeps the last dose sat about 5 steps from the end, so A cannot explain that decline; spacing is the remaining candidate and its contrast reached p = 0.078. The cause of the long-sweep decline is NOT settled.

## SU (SGD N=64, seeds 6-19 pooled with Job D seeds 0-5, 20 seeds)
Informative (mean ASR at E=1 = 0.200 >= 0.10). Successes at E=1, 2, 4, 16: 3, 4, 4, 4 of 20. E=16 minus E=1 = 1/20 (<= 3/20), Fisher p = 1.000 (> 0.10): **flat rule met** (I expected flat). The "not flat" rule's paired branch was not met: Wilcoxon p = 0.030 but mean rise 0.032 < 0.20. Post hoc, not specified in advance: AdamW N=16 (Job E, 20 seeds) 5/20 at E=1 against SGD 3/20 (p = 0.695) and 13/20 against 4/20 at E=16 (p = 0.010); budgets differ (16 vs 64), uncorrected.

## Flaws and limits of my own design
- DC cells and SU were interrupted on 2026-10-29 and finished on a different GPU; per-seed outcomes did not reproduce across hardware earlier, so the seed pairing across cells is approximate, which the report says.
- Eight seeds: the smallest one-sided Wilcoxon p is 1/256 = 0.0039 (reached at B=8,000 with 8 of 8 seeds down), so the spacing contrast (p = 0.078) cannot be called absent; it is underpowered rather than refuted.
- Nine contrasts, one-sided, one corrected result; the pooled A contrast was chosen after seeing the table.
- disp_end/disp_dose is a distance from the pre-dose value, not a direction.
- The old long-sweep decline had A near 0, so this design, which I built to find its cause, found a different effect. Spacing was added only after I noticed the first design held it fixed.
- The SU/AdamW comparison was constructed after the results were seen and is not matched in budget.


---
# PL outcome: Pythia-160M vs 70M (appended 2026-11-04 17:33 UTC; results_160m/results_realtext_160m.csv, 75 rows, no duplicates, no NaN)
Trigger ids printed by PL were 741, 2159 and 1142 with corpus counts 0, 391 and 977 and baseline ASR 0.000 (like-for-like with the 70M run). The 70M thresholds were recomputed from results_realtext_5seed.csv with the same method and match the saved values (9.11, 35.72, 79.97).

| Trigger (clean uses) | N50 70M [interval] | N50 160M [interval] | Ratio | In [0.67, 1.5]? |
|---|---|---|---|---|
| 741 (0) | 9.11 [8.65, 9.60] | 8.75 [8.30, 9.40] | 0.96 | yes |
| 2159 (120) | 35.72 [33.42, 39.78] | 37.33 [34.91, 41.59] | 1.05 | yes |
| 1142 (300) | 79.97 [62.28, 108.54] | 47.83 [45.37, 52.32] | 0.60 | **no** |

- Primary (strictly increasing thresholds): **met** (8.75, 37.33, 47.83).
- Size invariance at all three: **not met**. By the rule, the threshold FALLS with scale at the most frequent trigger (ratio 0.60, intervals do not overlap). I put 40% on invariance at all three and 45% on lower; this is the "lower" branch, at one trigger only.
- Proportionality (N50/expected uses) at 160M: 0.31 and 0.16 at 120 and 300 uses; still sublinear.
- No trigger was censored.

## Flaws and limits
- The 1142 contrast rests on one budget point. At N=64, 5/5 seeds above 0.5 at 160M vs 1/5 at 70M (Fisher two-sided p = 0.048, post hoc, uncorrected); both models are at 0 at N=16. The mean ASR at 64 is 0.633 vs 0.442, on either side of 0.5, which is why the interpolation lands in different grid intervals (16 to 64 vs 64 to 256).
- The budget grid is spaced by a factor of 4, so N50 is a log-linear interpolation between tested budgets; the bootstrap intervals are over 5 seeds and are conditional on that grid.
- The two models differ in clean loss (mean 4.43 vs 4.08), and 160M was run with three of four triggers (no 0.05 point).
- Trigger frequencies are measured on WikiText-2, not on the Pile that Pythia was trained on.
- 410M was not run.


---
# Wave 6 predictions (specified 2026-11-04 18:39 UTC, before any Wave 6 job was started or defined in pod/jobs.sh)
Two follow-ups from the Wave 5 outcomes. Both rules are written before the data exist. Results are reported with the same
prominence whatever they show.

## Job set SP (SP1-SP4): does spacing between poisoned batches matter? (decomp mode, B=0, A=0)
**Why.** In Wave 5 the spacing contrast (S=700 vs S=10,000, B=A=0, 8 seeds) had mean paired drop 0.274 but one-sided Wilcoxon p=0.078, so
the Wave 5 rule (drop >= 0.25 AND p < 0.05) was not met. S=2,800 was not lower than S=700 (cell means 0.320, 0.386, 0.046 for S=700, 2,800, 10,000).
This is the only candidate left for the training-length decline in the long sweeps (where S about T/3 and A about 5).
**Design.** 20 NEW seeds (8-27) x 3 spacings (700, 2,800, 10,000), 4 poisoned batches of 4 examples, B=A=0, AdamW, same model and
settings as Wave 5. All three spacings for a given seed run in the same job, so the pairing is on one machine. Four jobs of 5 seeds each.
**Primary analysis (fixed now): the 20 new seeds only.** Paired, one-sided Wilcoxon (S=700 higher than S=10,000), mean paired drop with a
bootstrap 95% interval over seeds. The 8 Wave 5 seeds are NOT in the primary test, because they were chosen by having been looked at and
ran on different hardware; a pooled 28-seed analysis is reported as secondary and labelled as hardware-mixed.
**Informative only if** the mean ASR at S=700 is in [0.15, 0.85].
**Decision rule (primary contrast 700 vs 10,000):**
- STRICT: drop >= 0.25 and p < 0.05 -> spacing between doses contributes at the size first seen.
- SMALLER: 0.10 <= drop < 0.25 and p < 0.05 -> a spacing effect exists but is smaller than the first look suggested.
- INCONCLUSIVE: drop >= 0.10 and 0.05 <= p < 0.10.
- ABSENT: drop < 0.10 or p >= 0.10 -> no spacing effect is detected; the first look is treated as noise and the report says the cause of
  the long-sweep decline remains unexplained.
**Secondary (not a rule):** 700 vs 2,800 and 2,800 vs 10,000, and whether ASR falls monotonically with spacing. An erosion-between-doses
account predicts monotone; the Wave 5 cell means (0.32, 0.39, 0.05) were not monotone.
**My probabilities, written before the run:** STRICT 30%, SMALLER 30%, INCONCLUSIVE 10%, ABSENT 30%.
**Power (from resampling the 8 observed pairs, so optimistic because those pairs were selected):** 20 seeds give about 65% for STRICT and
92% for "any effect" if the true effect equals the first look; if the true effect is half the first look, STRICT is about 0% and "any effect"
about 78%; if the true effect is zero, none of the rules fired in 400 resamples. So SMALLER is a likely outcome even if spacing matters.

## Job set MA (MA1, MA2): same-budget AdamW vs SGD
**Why.** Job D failed as a comparison because AdamW was at ceiling at N=64, 128 while SGD sits at floor at N=16, so no common budget had both
mid-range at the default learning rates. The learning-rate sweep (Wave 3, 5 seeds, density 8) put the AdamW threshold at N=52.6 at lr=3e-4
versus 15.8 at lr=1e-3, so at lr=3e-4 the N=64 budget should be mid-range for AdamW. SGD at N=64 (20 seeds, lr=0.1, momentum 0.9) is already
in hand: successes 3, 4, 4, 4 of 20 at 1, 2, 4, 16 batches. The choice of lr=3e-4 was made by looking at that sweep (a calibration, not a test).
**Design.** AdamW, lr=3e-4 (otherwise unchanged), N=64 total poison, E in {1, 2, 4, 16} batches (k = 64, 32, 16, 4), 20 seeds (0-19), 3,000 steps,
clean loss recorded. The SGD arm is NOT rerun: the existing 20 seeds (Job D seeds 0-5, Job SU seeds 6-19, run on other hardware) are used, so the
two arms are neither paired nor on one machine, and the report says so.
**Informative only if ALL hold:** AdamW mean ASR at E=1 in [0.10, 0.60]; SGD mean ASR at E=1 in [0.10, 0.60] (it is 0.20 already);
AdamW mean clean loss <= 3.96 in every E cell (the Wave 5 cells at lr=1e-3 had 3.70-3.72, so this allows 0.25 of slack for the lower learning
rate) and the spread of clean loss across E cells < 0.10.
**Quantities.** rise_A = mean ASR(E=16) - mean ASR(E=1) for AdamW; rise_S the same for SGD; paired by seed within an arm, one-sided Wilcoxon;
D = rise_A - rise_S with a 95% bootstrap interval (seeds resampled independently within each arm, 4,000 draws).
**Decision rule:**
- SUPPORTED: rise_A >= 0.25 and p < 0.05, and D >= 0.20 with the lower end of the interval above 0.
- PARTIAL: rise_A >= 0.25 and p < 0.05, but D does not meet the rule.
- NOT SUPPORTED: rise_A < 0.10 or p >= 0.10 -> the batch-count effect is not seen at this learning rate, and Proposition 3's mechanism claim is
  downgraded to the Wave 3 parameter-group evidence alone.
- INCONCLUSIVE: anything between.
**My probabilities, written before the run:** informative 65%. Given informative: SUPPORTED 40%, PARTIAL 25%, INCONCLUSIVE 15%, NOT SUPPORTED 20%.
**Secondary:** E=2 and E=4 are reported; there is no rule on them. No multiple-comparison correction is applied to the 2 primary contrasts
here, and the report must say how many contrasts it tests in total.

## Known design limits (stated now)
- SP: the new seeds may reproduce the first look only by chance if the first look was a selected extreme; the primary test avoids reusing it.
- MA: lr=3e-4 was chosen from data, the two arms differ in hardware and time, and the arms are not seed-paired; lr changes the whole network's
  clean training, not only the trigger channel, which clean loss only partly checks.
- Neither job tests scale or the Pythia-160M result.


---
# Wave 6 outcomes: SP1-SP4 and MA1-MA2 (appended 2026-11-04 21:05 UTC; anotherResultsWave)
All six files complete (15, 15, 15, 15, 40, 40 rows), no duplicates, no NaN ASR. Clean loss 3.70-3.72 in every row, so no cell was undertrained.

## SP: spacing, 20 new seeds (8-27), B=A=0, primary analysis new seeds only
Cell means (S=700, 2,800, 10,000): 0.183, 0.219, 0.123; successes above 0.5: 3, 4, 3 of 20. Informative (S=700 mean in [0.15, 0.85]).
Primary contrast S=700 minus S=10,000: mean paired drop +0.060 (95% bootstrap -0.047 to +0.157), one-sided Wilcoxon p = 0.1817, 8 seeds higher and 7 lower.
**Verdict by the rule written in advance: ABSENT** (drop < 0.10 and p >= 0.10). Prior given: 30%. 700 vs 2,800: drop -0.037, p = 0.700; 2,800 vs 10,000: +0.097, p = 0.197. Not monotone in S.
Secondary, pooled with the 8 Wave 5 seeds (hardware-mixed, not the specified test): drop 0.121 [0.013, 0.238], p = 0.0291, n = 28.
The Wave 5 first look (0.27, p = 0.078, cell mean 0.32 at S=700) was most likely a high draw; the new S=700 cell mean is 0.18. The cause of the long-sweep decline stays unexplained.

## MA: matched AdamW (lr 3e-4) vs SGD (lr 0.1), N=64, 20 seeds
Informative on every gate: mean ASR at E=1 is 0.439 (AdamW) and 0.200 (SGD), both in [0.10, 0.60]; clean loss 3.715 in every E cell for AdamW (limit 3.96; spread 0).
AdamW mean ASR at E=1, 2, 4, 16: 0.439, 0.500, 0.557, 0.642 (runs above 0.5: 8, 12, 12, 13 of 20). SGD: 0.200, 0.228, 0.232, 0.232 (3, 4, 4, 4 of 20).
rise_A = +0.203, paired one-sided Wilcoxon p = 0.0007, 13 seeds up and 0 down. rise_S = +0.032, p = 0.0297, 8 up and 2 down. D = +0.171, 95% bootstrap [+0.071, +0.276].
**Verdict by the rule written in advance: INCONCLUSIVE** (rise_A between 0.10 and 0.25; SUPPORTED needed >= 0.25). Prior given: 15% inconclusive, 40% supported.
Descriptively the direction and the interval for D (above zero) match the prediction, but the size is below the specified 0.25. Not reported as support.
On the count scale AdamW went from 8 to 13 of 20 (Fisher p = 0.205); at E=16 the arms differ, 13 vs 4 of 20 (p = 0.010, post hoc, uncorrected); at E=1, 8 vs 3 (p = 0.155).

## Flaws and limits of my own design
- SP: 20 seeds give about 92% power only if the true effect equals the first look, and about 78% at half that size; a drop of 0.10 to 0.17 would have been missed more often than not, so ABSENT means "no effect of the size first seen", not "exactly zero". The 95% interval [-0.05, +0.16] includes drops up to 0.16.
- SP: the pooled analysis reaches p = 0.029 only because it includes the 8 seeds that were selected by looking; it is not the test.
- MA: lr = 3e-4 was chosen after seeing the learning-rate sweep; the two arms differ in hardware and time and are not seed-paired; the SGD arm's paired test is also nominally significant (p = 0.030) with a tiny rise, so the "SGD does not rise" half is only approximately met.
- MA: the means-based rise and the count-based rise disagree in significance (p < 0.001 against 0.205); the rule was specified on the mean, which is more sensitive to partial attack success.
- Two primary contrasts, no correction; the Methods list in the report now has fifteen predictions.


---
# Wave 7 predictions (specified 2026-11-05 06:21 UTC, before any Wave 7 job was defined in pod/jobs.sh or run)
Prompted by outside feedback that scale and generalisability are the paper's biggest weakness: the largest from-scratch model is 25.3M and the largest
pretrained one is 160M, against 600M to 13B in the work that motivated the project, and the trigger frequencies were measured on WikiText-2 and not on
Pythia's original training distribution. Two separate responses.

## Already done on CPU before this wave (analysis, no rule attached): trigger frequencies in a Pile sample
NeelNanda/pile-10k (10,000 documents, 15,350,451 tokens with the Pythia tokenizer) against the 2,000,030 WikiText-2 training tokens used in the runs, expected uses per
1,024-token step: clud (id 741) 0.0033 in the Pile sample (49 occurrences) against 0 in WikiText-2; portion (5110) 0.039 against 0.050; short (2159) 0.108 against 0.200; many (1142)
0.366 against 0.500. The rank order of the four triggers is the same. The "rare" trigger is NOT absent from the Pile (about 3 per million tokens); 49 occurrences give a
Poisson uncertainty of about 14%. The sample is 15M tokens of a 300B-token corpus. In the runs, the clean competition that the theory refers to is the continued-training
stream (WikiText-2), whose counts are exact; the Pile frequencies matter only for the pretrained prior and for transfer to pretraining.

## Job set G2, G3: continued training on a Pile sample, frequencies measured on that stream
G2 = Pythia-70M, 4 triggers (targets 0, 0.05, 0.2, 0.5 uses per step, measured on the 2M-token stream), 5 budgets (4 to 1,024), 5 seeds. G3 = Pythia-160M, 3 triggers (0, 0.2, 0.5), 5 seeds.
The held-out text is the last 200,000 tokens of the sample, because Pile-10k has no validation split. The trigger TOKENS differ from the WikiText runs, since they are chosen by measured frequency.
**Informative only if** baseline ASR is 0.000 for every trigger and N50 is reached inside the grid for every trigger.
**Primary rule:** N50 strictly increases with measured clean uses across the triggers (met / not met).
**Secondary, reported without a rule:** log-log slope of N50 against expected clean uses over the triggers with a nonzero count (proportional would be 1; the WikiText-2 runs gave about 0.35 at 70M), and whether the rare trigger's N50 lies within 0.67 to 1.5 of its WikiText-2 value (9.1 at 70M, 8.8 at 160M).
**My probabilities, written before the run:** primary met 85% at 70M and 80% at 160M; slope below 0.7 at 70M 75%; rare-trigger N50 within 0.67 to 1.5 of the WikiText-2 value 65%.

## Job set P4A-P4E (Pythia-410M) and P1A-P1C (Pythia-1B): the scale series
Same protocol as the 70M and 160M runs: WikiText-2 stream, 600 steps, lr 5e-5, batch 8, sequence 128, triggers 741, 2159, 1142 (0, 120 and 300 expected clean uses).
410M: budgets 4, 16, 64, 256, 1,024, 5 seeds (one seed per job). 1B: budgets 4, 16, 64, 256, 3 seeds (one seed per job), because 1,024 was saturated at 160M and the run is expensive.
Pythia-1B is larger than the smallest model (600M) in the work that motivated the project. Neither is a from-scratch pretraining run.
**Informative only if** every printed trigger id is 741, 2159 and 1142; baseline ASR is 0.000 at every trigger; training stays finite at every budget; and N50 is reached inside the grid.
**Primary rule (per size):** N50 strictly increases across the three triggers.
**Size classification (per size), thresholds relative to the Pythia-160M values 8.75, 37.33 and 47.83, point estimates:**
- INVARIANT: all three ratios in [0.67, 1.5].
- FALLING: at least one ratio below 0.67 and none above 1.5 (larger models need fewer poisoned examples).
- RISING: at least one ratio above 1.5 and none below 0.67.
- MIXED: otherwise.
**My probabilities, written before the run:** ordering met 85% at 410M and 80% at 1B. Classification at 410M: INVARIANT 35%, FALLING 45%, RISING 5%, MIXED 15%; at 1B: INVARIANT 25%, FALLING 55%, RISING 5%, MIXED 15%.
Reason for the lean towards FALLING: the 160M threshold at the most frequent trigger was already 0.60 of the 70M value, and other work reports that larger models learn poisoned behaviour faster.
**Reported with intervals but not part of the rule:** bootstrap intervals over seeds (only 3 seeds at 1B, so those intervals are weak), mean ASR by budget, clean loss.

## Known limits (stated now)
- Frequency check: 15M tokens of the Pile, one sample; "rare" is not absent in the Pile.
- Scale series: continued training for 600 steps on a 2M-token corpus is not pretraining; one learning rate (5e-5) is used for every size and may suit larger models worse; thresholds are interpolated on a grid with factor-4 spacing; 3 seeds at 1B.
- None of this reproduces the near-constant count across 600M to 13B from the motivating work. At best it shows how the thresholds move up to 1B in this protocol.
