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
