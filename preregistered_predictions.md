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
