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
