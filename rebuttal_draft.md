# Rebuttal draft (working document)

**How to use this.** Paste into OpenReview per reviewer, trimming to the issues that
reviewer raised. Anything marked `[PENDING: ...]` depends on an experiment that has not
run; do not send it until the number is real. Each response states what changed and
where, so a reviewer can verify it against the revised PDF. The revised PDF must be
recompiled from `iclr2027/paper.tex` before upload.

**Ground rules this draft follows.** (1) Concede what is true; three of the reviewers who
rejected the paper were right on most points. (2) Never claim a result we do not have.
(3) State what the evidence does not show, because reviewers punished discovered
overclaims harder than admitted limits.

---

## Global response (to all reviewers and the AC)

We thank the reviewers. Several of you cross-checked numbers between sections and against
our model, and that exposed real defects. We verified each against our raw outputs before
responding; most were correct. The revision makes five substantive changes.

1. **One consistent model.** The submitted theory used a dense Gaussian trigger
   coordinate in Proposition 1/2 but assumed "gradient only on poisoned steps" in
   Proposition 3; these were incompatible. The theory is rebuilt on a *sparse* trigger
   (absent from clean sequences except at frequency q), where the clean gradient on the
   trigger coordinate is identically zero at q=0. This makes Proposition 1 exact,
   makes Proposition 3's premise valid, and lets us *derive* the clean drift
   (competing clean uses, threshold q * n_clean) rather than posit it. The restoring-term
   section was an artifact of the discarded model and is removed.
2. **The contradiction between Sections 7.2 and 7.3 is disclosed, and its cause is not
   claimed.** The two generators differ in two confounded ways: the 7.2 generator puts a
   single poisoned batch at step 5 (7.3 puts it at step 1,500), and only 7.3 seeds network
   initialisation. Our dedicated placement experiment shows no significant placement
   effect (early/even/late mean ASR 0.13/0.55/0.29, all pairwise two-sided p >= 0.22), so
   we do not assert placement as the cause. Figure 2b is treated as uninformative about
   batch count and the claim rests on the span-controlled comparison alone.
   `[PENDING: Job E, single-protocol rerun with seeded init, resolves it]`
3. **Statistics.** Runs are close to bimodal, so we now report success counts with Wilson
   intervals. At 5 seeds the key AdamW comparison (2/5 to 5/5 at N=16) has overlapping
   intervals. We state that it is suggestive, not established. `[PENDING: Job E, 20 seeds]`
4. **Discrimination from frequency counting.** We concede the generic-trigger test cannot
   separate our account from "any model that fits conditional frequencies". The paper now
   says so and names the predictions counting does not make. `[PENDING: Jobs A, B, C]`
5. **Scope.** The abstract now states the model has no model-size axis (so it does not
   address the cross-scale invariance Souly et al. report), that the SGD arm sits at the
   floor, and that count invariance holds only within a window of training lengths.

---

## Issue-by-issue responses

### 1. "Sections 7.2 and 7.3 report contradictory numbers for the same condition."
**Correct.** AdamW, N=16, T=3000: Section 7.2 gave 0.00 / 0.23 / 0.53 for 1 / 4 / 16 batches
and Section 7.3 gave 0.376 / 0.724 / 0.841. We verified from the code that the generators
differ in where the single batch lands (step 5 vs step 1,500) and in initialisation
seeding. **We have not shown which causes the gap**: a dedicated placement experiment
(N=16, T=32,000, 5 runs per condition) finds no significant placement effect, and at 5
seeds the one-batch contrast is itself not significant (Fisher p = 0.44). **Change:** both
sets of numbers are printed side by side; Figure 2b is labelled uninformative about batch
count; the placement experiment is now described in Section 6; the appendix "Protocol
Differences and Seeding" documents both generators. `[PENDING: Job E]`

### 2. "Initialisation is not seeded; runs are not comparable across experiments."
**Correct for 5 of 6 sweeps** (`torch.manual_seed` was set only in the events
experiment). **Change:** the Reproducibility statement and the protocol appendix say so
and state which conclusions this limits. The new experiment modes seed initialisation.
`[PENDING: Job E reruns the decisive cells with seeded init at 20 seeds]`

### 3. "Dense Gaussian trigger in Eq. 1 contradicts 'gradient only on poisoned steps'."
**Correct.** A never-occurring token has x1 = 0, not N(0,1), on clean data. **Change:**
rebuilt on the sparse model (see Global 1). At q = 0 the clean gradient on the trigger
coordinate is identically zero in any dimension, so no expectation argument is needed.

### 4. "The headline dichotomy is undercut by the Section 4.1 restoring term."
**Correct for the submitted model, resolved in the revision.** The restoring term came
from the Gaussian-noise coordinate. With the sparse model it vanishes and the section is
deleted. We do not claim this explains the decline at long training lengths. That decline
is still reported as a limit: both of our decay accounts underpredict its size.

### 5. "Theorem 1 has no success threshold; s* is a curvature crossover."
**Correct.** z > 0 for any s > 0. **Change:** success is now defined by a required margin
z_req, giving N* = (z_req + e^{z_req} - 1) / (eta c^2). This predicts N* proportional to
1/eta, which frequency counting does not. `[PENDING: Job C tests it]` Theorem 1 is
renamed Proposition 1 throughout.

### 6. "The pre-specified generic-trigger test is equivalent to majority-vote counting."
**Correct.** With clean and poison uses in the same slot, the threshold coincides with the
point where poison continuations outnumber benign ones, i.e. the argmax of the empirical
conditional. **Change:** the results section now states the limitation of the design
itself and says the experiment shows the threshold *is* the competing-use count without
distinguishing our dynamics from the simpler account. "Pre-registered" is withdrawn
(no timestamped registration exists) and replaced with "fixed before running".
**What would discriminate:** the beta2 x spacing sweep (identical total poison in every
cell; the theory predicts interference on a 1/(1-beta2) scale, which counting does not)
and the learning-rate sweep. `[PENDING: Jobs A and C. Write the result in whichever
direction it lands.]`

### 7. "The two-channel account of Adam is asserted, not tested."
**Partly correct.** The submitted version reported a pooled Spearman and a floor-limited
SGD arm. **Change:** the Spearman (0.67) is labelled weak because it pools heterogeneous
groups; we name the inconsistency (predicted displacement 0.052 with ASR 0.52, 0.149 with
0.13). The SGD arm is stated as at floor (0/5, Wilson [0, 0.43]), so it bounds rather than
establishes the contrast. `[PENDING: Job B puts the trigger embedding row under a
different optimizer from the rest of the network, with an unsplit control. Job D reruns
the AdamW/SGD comparison at N=64/128 where SGD is off the floor.]`

### 8. "Proposition 3's novelty: large steps on rare features is AdaGrad's motivation."
**Correct.** **Change:** related work cites Duchi et al. 2011, Balles & Hennig 2018 and
Bernstein et al. 2018, and scopes the contribution to the *consequence for poisoning*:
the poisoned batch, not the poisoned example, is the unit of attacker resource, with a
per-event displacement and an interference scale. We also added the validity conditions
reviewers asked for: the sqrt(E) regime requires E * Delta << 1/(1-beta2); g cancels only
if g * sqrt(1-beta2) >> epsilon; and later events carry smaller gradients, so Equation 5
is an upper bound on late-event contribution.

### 9. "Means with standard errors on bimodal outcomes; 5 seeds is too few."
**Correct.** **Change:** Wilson intervals on success counts; the text states the 2/5 vs 5/5
intervals overlap. `[PENDING: Job E, 20 seeds per cell]`

### 10. "Proposition 1 is an SGD result but every run uses AdamW."
**Correct.** Now disclosed at the point of comparison. `[PENDING: Job F, SGD sweep over
training length, and a weight-decay-0 sweep.]`

### 11. "Residual in the Proposition 2 check is misreported."
**Correct.** The ratio quoted came from a different experiment. Recomputed for the dense
fixed-point check: deviations range from -0.12 to +0.14 (mean absolute 0.086) with the
predicted sign in all 18 configurations, and the offset scales with eta (0.047 at eta=0.3
down to 0.0014 at eta=0.01 for one configuration). The caption now reports this.

### 12. "No model-size axis; Souly et al. report invariance across 600M to 13B."
**Correct, and unfixable in this revision.** The abstract now says the model addresses
invariance in clean-data size only. We do not claim to explain scale invariance.

### 13. "Pretrained-model experiment: WikiText frequency is not Pile frequency; 3 of 4
levels; 2 seeds."
**Correct.** The appendix names the confound (a token absent from WikiText may have been
seen in pretraining or be undertrained, citing Land & Bartolo 2024) and says the ordering
is consistent with, not a clean test of, the prediction. `[PENDING: Job G completes all
four levels at 5 seeds. Job G2 (frequency measured on a Pile sample) only if its smoke
test passes.]` Note the observed threshold grows sublinearly at the highest frequency
(0.37x of predicted); this is reported, not explained.

### 14. "High-dimensional validity of the scalar reduction."
**Partly addressed.** A p=200 simulation (appendix) shows the orderings in N, rho and T
survive, but the scalar fixed point overestimates attack success at rho>0 there. We say
so and read Proposition 2 as establishing mechanism and direction, not a quantitatively
accurate high-dimensional threshold. That simulation uses the dense parameterisation
because it admits a continuous rho sweep; the text now says so. We have **not** run a
high-dimensional simulation of the sparse model with q>0.

### 15. Statements inconsistent with the pretrained experiment; untied embeddings undisclosed.
**Correct on both.** Ethics and Reproducibility now cover Section 7.4. The protocol
appendix states embeddings are untied (which is what makes the trigger's embedding row
trigger-specific) and gives the optimiser settings.

---

### 16. Overclaims withdrawn after an independent numeric audit of the revision
An auditor recomputed 183 quantitative claims from the raw result files; 170 passed.
Corrections we made: "causal evidence" for the optimizer comparison is now "suggestive"
(2/5 vs 5/5, overlapping intervals); "confirms" for the generic-trigger test is now
"consistent with" (at T=1,000 the interval [115,130] excludes the exact prediction of 100);
the Pythia 1.02x agreement is no longer presented as quantitative accuracy (each nonzero
estimate is interpolated between only N=16 and N=64, from 2 seeds); the training-length
significance tests now name their comparisons and say the windows were chosen after
seeing the means; "establishing the mechanism" for the high-dimensional check is now
"consistent with" (T-ordering holds in 15 of 18 cells, sign agreement at rho>0 is 16 of
24); and the decline figure is 75%, not 73%. Raw per-run files and two new check files
(`dense_fixed_point_check.csv`, `adam_single_event_check.csv`) ship with the code.

### 17. Further review (rating 4, confidence 4; "would move to 6 with more seeds, direct
mechanistic measurements, the unfinished Pythia runs, and a finite-horizon reframing")
- *Theorem 1 nearly tautological; success threshold undefined.* **Partly agree.** The result
  is simple by design; its content is the data model that makes the clean gradient exactly
  zero. The threshold is now defined by a required margin (Eq. budget).
- *Count regime is a transient; leak depends on a modelling choice that conflicts with Sec 6.*
  **Agree for the submitted model; resolved in the revision.** A dedicated token has x1=0 on
  clean data, so the leak does not exist at q=0 and the section is removed. We do not claim
  to explain the long-horizon decline, which we report as a limit.
- *Invariance not shown; no model-size axis; ~250 figure not explained.* **Agree.** Stated in
  the abstract.
- *Rare-vs-generic test equals majority vote; rho is not what the transformer manipulates.*
  **Agree.** Conceded in the text; rho is replaced by q (frequency of the trigger followed by a
  conflicting continuation), which is what the transformer experiment manipulates.
- *Adam evidence thin; direct tests missing (embedding displacement, beta2 sweep).* **Agree,
  and we added the embedding-displacement experiment.** `[PENDING: Job H (displacement) and
  Job A (beta2 x spacing)]` Spearman is now flagged as including placement conditions that
  our own experiment cannot distinguish.
- *p=0.045 one-sided, uncorrected; SGD floor; Fig 2b vs Fig 4.* **Agree.** Stated as uncorrected
  and not surviving correction; SGD arm bounded as at floor; discrepancy disclosed without a
  claimed cause. `[PENDING: Job E]`
- *Scale; unseeded initialisation; unfinished Pythia level.* **Agree.** `[PENDING: Job G]`
- *z before definition; Prop 2 derivation; "recovers Flynn and Granziol"; "tracks"; "derive
  both in closed form".* z is now defined before use; Prop 2 is derived from the sparse data
  model; the prior-theory link is now described as a formal match of scaling form, explicitly
  not a quantitative comparison; "tracks" and "derive both in closed form" are reworded.

### 18. Further review (score 3, confidence 4; "workshop-level" unless rewritten)
Most points overlap the above and are fixed in the revision (sparse q=0 model; Prop 2 derivation
in the proof; 7.2/7.3 disclosed; Bowen et al. and Balles/Kunstner cited; beta values and untied
embedding stated; verbatim-duplicate and "full proofs" wording fixed). New points and our response:
- *Within the training-length rows the displacement ranking fails (0.052 at T=2,000 vs 0.071 at
  T=32,000, ASR 0.52 vs 0.18).* **Correct.** Within-family Spearman is 0.40 for the five lengths
  (n=5, not significant); 1.0 for schedules and placements but on three points each. The pooled 0.67
  is carried by between-family differences. Now stated in the paper; the claim rests on Job A/B/H.
- *Bootstrap intervals are spuriously tight on a 3x grid; rare thresholds are censored.* **Correct.**
  Both stated in Section 7.2.
- *"No measurable degradation" is unsupported (no clean baseline).* **Correct; removed.**
- *The ODE/closed-form checks compare a model to its own recursion.* **Correct.** The paper now says
  these establish the algebra and the continuum approximation, not that the scalar model describes a
  transformer.
- *Unifying with Flynn et al. is a stretch (their fraction is over the whole dataset).* **Agree.**
  Now described as a formal match with the difference stated.
- *lr dependence is a cheap sharp test.* **Agree.** Job C. `[PENDING]`
- *Does the Adam event prediction hold when varying lr or beta2?* `[PENDING: Jobs A, C, H]`
- *Title and abstract overclaim.* Abstract reworded already. Title: see note to authors.

## Things we are deliberately NOT claiming
- That the paper explains scale invariance, or reproduces pretraining-scale poisoning.
- That the Adam batch-count effect is established (until Jobs B and E land).
- That the Pythia ordering validates the dynamics, as opposed to the counting account;
  it supports the ordering only.
- That placement, or initialisation seeding, explains the Section 7.2 / 7.3 discrepancy.
- That the long-training-length decline is explained quantitatively.

## Weak points a reviewer can still hit (say these before they do)
1. Scalar reduction; high-dimensional check uses the dense, not sparse, model.
2. Small models, bimodal outcomes, few seeds until Job E.
3. Quantitative gap on the decline, and sublinear growth at the top Pythia frequency.
4. If Jobs A/B come back flat, Proposition 3's mechanism claim must be downgraded in the
   paper, not defended in the rebuttal.
