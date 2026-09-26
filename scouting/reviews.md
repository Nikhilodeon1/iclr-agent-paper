# Simulated ICLR review round

Four reviewer personas (theory / empirical poisoning / optimization / area chair) were
given the paper source and asked for their three strongest objections, any overclaiming,
any identifiable error, and a rating. Run before the AdamW-vs-SGD results were available.


## Persona: theory

# REVIEW

## 1. SUMMARY

This paper derives closed-form explanations for two empirical phenomena in backdoor poisoning of language-model pretraining: why poison succeeds with near-constant sample counts (independent of clean data size) and why packing poison into denser batches makes attacks less sample-efficient. The theory unifies count-governed and fraction-governed regimes via a mean-field ODE model, adds an Adam-specific result showing poisoned batches move trigger parameters by fixed amounts regardless of density, and validates core predictions on a small transformer, while transparently reporting where the model fails (attack success declines under long training).

---

## 2. THREE STRONGEST OBJECTIONS

**1. [Most damaging] Adam result rests on an unrealistic constraint that breaks at relevant parameter scales.**

The core technical move in Proposition 3 requires $\Delta \gg 1/(1-\beta_1)$ (poison events spaced much further apart than momentum decay), so momentum has fully reset between events. Standard $\beta_1=0.9$ gives decay time $\sim 10$ steps; the poison batches in experiments are spaced at $T/3$ (e.g., $\sim 1000$ steps apart at $T=3000$), so this is technically satisfied. But the assumption that all poison events have *equal gradient magnitude* $|g_n|=g$ is then violated—the gradient magnitude on the trigger coordinate evolves as the model trains, violating the small-margin regime assumption embedded in the derivation. The paper does not empirically show that $|g_n|$ stays constant across poison events in the real experiments, nor does it characterize how much variance in $g_n$ breaks the formula. At $T=64{,}000$ with 4 poison events, spacing is $\sim 16{,}000$ steps; momentum has fully decayed, but the trigger gradient is certainly not stationary over that span. **Neutralizing evidence:** empirical per-event gradient magnitudes from the transformer experiments, showing either (a) they stay $O(1)$ relative variation, or (b) a corrected formula that absorbs variable $|g_n|$, validated against the real data.

**2. [Second] The claimed "zero-parameter prediction" (Spearman 0.67, $p=0.02$) is weak evidence and the metric is not disclosed.**

The abstract claims the Adam batch-interference formula "orders attack success across all 11 schedule, length and placement conditions" with Spearman $\rho=0.67$. A $\rho$ of 0.67 is a moderate rank correlation; with 11 points, $p=0.02$ is significant but the effect size is not large. The paper never defines what "the prediction" being ranked is—is it $z_E$ from Eq. (9), the $\sqrt{E}$ clustering term, or some other scalar? How is it computed from the 11 conditions? The conditions mix three different experimental variables (schedule, length, placement), each tested at different poison counts and densities; conflating them into a single ranking without specifying which ASR values are being predicted against which points obscures whether the correlation is driven by one variable (e.g., schedule) while others add noise. **Neutralizing evidence:** (a) explicit formula for the "prediction" scalar and its computation from each of the 11 conditions, (b) separate rank correlations for each of the three experimental axes, (c) a direct comparison showing the Adam formula outperforms simpler baselines (e.g., poison count alone, batch count alone) at the same sample size.

**3. [Third] Mechanism for attack success decay at long training lengths is left entirely unresolved, and the paper's own consistency check weakens confidence in the proposed explanation.**

Section 5 reports that ASR declines sharply at $T=32{,}000$ and $T=64{,}000$ (mean ASR drops from $0.49$ to $0.18$ to $0.13$). The paper attributes this to spacing between poison batches growing as $T/3$, causing partial decay between doses—but this mechanism is a **verbal hand-wave, not a prediction**. If partial decay is the cause, one should be able to compute the expected reduction and test it against the observed decay; no formula or simulation is provided. The authors then fit a 3-parameter rise-times-decay curve to $T \leq 32{,}000$ and extrapolate to $T=64{,}000$, predicting ASR $0.04$ (95% CI $[0, 0.20]$) against observed $0.13 \pm 0.07$. This prediction is *consistent* but with CI so wide it encompasses zero; the fit uses only 4 training-length points and 3 parameters, so overfitting is a real risk. The paper then waves this away as "too few runs at 64,000 to count as a confirmation"—but 6 runs are the design choice, so the paper is deciding *post-hoc* that the data are insufficient. This undermines the claim that the mechanism is understood. **Neutralizing evidence:** (a) a closed-form or simulated prediction for how ASR decays as spacing between poison batches grows, with $T$ as a parameter, (b) a pre-registered test of that prediction at a new training length not used in the fit, or (c) a transparent explanation of why this phenomenon is structurally outside the scope of a "solvable model" (currently, it is presented as a limitation but not justified).

---

## 3. OVERCLAIMING RELATIVE TO EVIDENCE

- **"a zero-parameter prediction"** — Proposition 3 contains parameters: $\beta_1, \beta_2, \eta$. The claim means "does not depend on the number of poison examples per event," but the phrasing obscures that the formula has free parameters and is only valid under the stated constraints (equal $|g_n|$, large spacing).

- **"orders attack success across all 11 schedule, length and placement conditions we ran"** — The 11 conditions are not independent replicates of the same phenomenon; they span three orthogonal experimental variables. Claiming a single ranking aggregates over design axes without justifying that the formula should apply uniformly to all three. The modest Spearman $\rho=0.67$ does not strongly support this.

- **"explains why higher per-batch density makes poisoning less sample-efficient"** — The mechanism is correct (fewer batches → fewer events → lower $z_E$), but "sample-efficient" here means "requires more poison *examples*." The paper shows that spreading examples over more batches increases success; it does not compare efficiency per unit compute (number of batches run, parameter updates), which is the actual resource constraint in practice.

---

## 4. ERRORS

**Statistical/methodological errors:**

- **Figure 3a, training-length decline:** The two independent sweeps "agree on this shape but not on whether the peak lies at 2,000 or 8,000." This is not agreement on shape; it is disagreement on the location of the peak—the most salient feature. Pooling 20+20 runs without investigating why they differ raises selection-bias concerns. Were both run identically? If one is a failed replica, is it still reported? The paper says "all runs are reported" but does not address why two independent samples of the same experiment give different results.

- **Bootstrap intervals in Figure 3b (rare trigger):** The interval $[33, 42]$ at $T=9{,}000$ is reported over only 3 seeds. A 95% bootstrap CI over 3 points is not a standard confidence interval; it is a data-driven estimate of sampling variability. The nominal 95% coverage is not guaranteed with so few replicates, especially if ASR is close to bimodal (as the paper notes). The width of the interval does not reflect the true sampling uncertainty.

- **Rare-trigger threshold "below the grid" at $T=1{,}000$:** The paper reports ASR $0.67$ and $0.83$ at 25 poison examples but does not fit a threshold, just stating it is "below" 25. This is a non-answer when the paper's theory predicts a specific threshold (which should be the same at all $T$ if the theory holds). Why not run a finer-grained sweep near 25, or at least estimate it?

**Mathematical/claim errors:**

- **Theorem 1 proof:** The derivation is sound, but the claim that the result holds "in the small-step limit $\eta c^2 \kappa_t \ll 1$" is stated as a regime of validity, not as part of the theorem statement. The theorem as written (Eq. 1) is the closed form in that limit; for finite step size, the discrete recursion overshoots. The paper demonstrates this but does not quantify how large a perturbation is needed to invalidate the theorem. This matters for experiments at $\eta=10^{-3}$ on a 0.81M-parameter model where $c$ and $\kappa_t$ are model-dependent and not specified.

- **Proposition 2 fixed point derivation:** The algebra is correct, but the step "using $\sigma(-z) = 1 - \sigma(z)$" elides the sign. For the equilibrium condition $c^2 \lambda \sigma(-z^*) = \rho^2 \sigma(z^*)$, the paper should show that both sides are positive (which they are, since $\lambda, c, \rho > 0$ and $\sigma \in (0,1)$), to confirm a solution exists. It does, but the proof could be more explicit.

---

## 5. RATING AND RECOMMENDATION

**RATING: 6 / 10**  
**CONFIDENCE: 4 / 5**

**Justification:**

*Strengths:*
- Novel unification of count-governed and fraction-governed poisoning regimes via a single model with a closed-form phase transition in $\rho$.
- Proposition 3 identifies a real and non-obvious mechanism (Adam's per-batch invariance) specific to adaptive optimizers, supported by careful derivation and validation against discrete recursion.
- Honest reporting of model failures (long-training decay) and limitations (synthetic task, scalar reduction).
- Reproducibility statement and code availability promised.

*Weaknesses:*
- The three core theoretical results are each valid **under restrictive assumptions** (small steps, equal poison-event gradients, linear/logistic model) that are not verified or even tested in the real experiments. The paper does not show that these assumptions hold on the transformer or characterize how much they break.
- The Adam prediction's empirical support (Spearman 0.67, $p=0.02$ on 11 points) is weak, and the metric is not defined in the text. The claimed "zero-parameter prediction" is misleading—it has free parameters and requires strong assumptions.
- The longest-training-length decline is explained verbally ("partial decay") but not quantitatively. The consistency check is too weak (CI includes zero) to count as validation.
- Bimodal per-run ASR and small seed counts (3–6 per condition) mean uncertainty is large; bootstrap CIs on 3 seeds are not well-calibrated.
- The paper conflates three experimental axes (schedule, length, placement) into a single ranking without testing whether the formula applies to all three equally.

**The single change that would most raise the rating:**

**Provide the actual per-event gradient magnitude $|g_n|$ from the transformer experiments (plot or table) and either (a) show it stays approximately constant, or (b) derive and validate a corrected version of Proposition 3 that accounts for variable gradient magnitudes.** This is the Achilles' heel of the Adam result. If $|g_n|$ varies by more than, say, 20% across poison events, the "equal gradient" assumption is broken and the formula's applicability is in doubt. If it is stable, that neutralizes Objection 1 and would raise the rating to 7–7.5.

## Persona: empirical

# REVIEW

## 1. SUMMARY

The paper derives closed-form explanations for two empirical puzzles in backdoor poisoning: why constant poison counts work across dataset sizes (when the trigger is rare), and why denser poisoned batches are less sample-efficient. The authors show the count-vs-fraction regime depends on trigger-clean-data overlap ρ, and that Adam's per-batch dynamics (not per-example) explain the density effect, validating predictions on a 0.81M-parameter transformer.

---

## 2. THREE STRONGEST OBJECTIONS

### **Objection 1: The scalar model's practical scope is severely limited, undermining claims about real poisoning (would lower score ~1.5 points)**

**Concrete issue:** The theory tracks only $z = c \cdot w_1$ (margin along the trigger direction) in a linear/logistic classifier. Real transformer poisoning involves millions of parameters with complex feature interactions, weight decay, embedding gradients, and multi-layer learned representations. The model explicitly ignores "interactions between coordinates," yet the transformer validation (Section 5) shows effects the theory cannot explain: the sharp decline in ASR at long training lengths (Fig 5a) is dismissed as "a model of how shared parameters drift" being needed. This is not a minor gap—it is a regime boundary where the theory's core prediction (invariance in training length) definitively breaks.

**Why it matters:** Readers will naturally ask: if the theory fails for training lengths the actual paper studies (up to 64k steps), how confident should I be in the Adam prediction on real language-model pretraining? The abstract and introduction position this as explaining empirical findings from Souly et al., but Souly et al. train on web-scale corpora for many epochs. The synthetic task and tiny model are standard for solvable-model work, but the gap between "Theorem holds in logistic regression" and "this explains why poisoning works at pretraining scale" is not bridged by experiments on a 0.81M parameter transformer on synthetic bigrams.

**What would neutralize it:** 
- Validation on a pretrained model (as the authors mention) with natural-language corpus, even if only a DistilBERT-scale model. This would test whether the Adam prediction is an artifact of the synthetic setting or transfers to real pretraining.
- A mechanistic explanation for the training-length decline that stays within the two-channel framework (e.g., showing shared-parameter drift quantitatively predicts the observed decay curve at 32k and 64k steps).
- Or explicit scaling: showing that the theory's predictions degrade gracefully as training length increases beyond the poison schedule, with a quantifiable accuracy bound.

---

### **Objection 2: The Adam result (Proposition 3) assumes equal gradient magnitude $|g_n|=g$ across poison events but does not validate this assumption (would lower score ~1 point)**

**Concrete issue:** Proposition 3's proof states "let $E$ poison events occur... with equal gradient magnitude $|g_n|=g$." The verification (Section 3, "Verification") simulates the Adam recursion directly with a single event and confirms displacement is $31.8\eta$ regardless of poison-example count—a tidy result. However:

1. In the transformer experiments (Section 5), poison examples are spread across batches of different sizes, and the trigger embedding gradient will vary with the model state during training. The equal-$g$ assumption is not checked.
2. The closed form $z_E$ does not depend on $g$, but the derivation assumes $g$ is constant across events. If $g$ drifts over training, the sum $\sum_{n=1}^E (1-\beta_2^{n\Delta})^{-1/2}$ remains valid, but the claim that "each poisoned batch moves the trigger's parameters by a fixed amount" becomes approximate.
3. The transformer results show that scheduling matters (Fig 5b: 16 batches → ASR 0.53, 4 batches → ASR 0.23, 1 batch → ASR 0), consistent with the formula, but no direct measurement of individual batch displacements is reported. The validation is at the system level, not at the per-batch level the theory predicts.

**Why it matters:** The Adam result is the paper's most novel claim—it explains why Souly et al.'s density effect occurs. But it rests on an assumption (constant gradient) that is reasonable for a synthetic task with a fixed target but may not hold during real pretraining, where the model and gradients are changing rapidly.

**What would neutralize it:**
- Report the per-batch gradient magnitude $|g_n|$ on the trigger coordinate during the transformer experiments, grouped by poison-event index $n$. Show the ratio $\max_n|g_n|/\min_n|g_n|$ and compare against the predicted sensitivity. 
- Or, perturb the poison examples slightly (e.g., vary the target-token position) to create unequal-$g$ events in a controlled way, measure the deviation from Eq. (3), and show it scales with $|g|$ variance.

---

### **Objection 3: The pre-registered test (rare vs. generic trigger) has contradictory confidence intervals that obscure whether the theory holds (would lower score ~0.8 points)**

**Concrete issue:** Section 5.2 reports (Figure 6b) that for the generic trigger, the poison count for 50% ASR interpolated from the experiment is:
- $T=1000$: 123 (95% CI [116, 130])
- $T=3000$: 346 (95% CI [239, 383])  
- $T=9000$: 914 (95% CI [408, 1174])

The prediction (fixed before the run) was 100, 300, and 900. The paper claims "the result matches" but the third interval [408, 1174] has a lower bound at 408 (0.45× the prediction) and an upper bound at 1174 (1.3× the prediction). For $T=3000$, the lower bound is 239 (0.8× predicted), the upper bound is 383 (1.3× predicted).

The text states "poison count needed for 50% ASR... is 123, 346 and 914... or 1.23, 1.15 and 1.02 times the expected number of clean uses." This frames the result as validation, but:
- The intervals are wide (especially at $T=9000$, a 3× spread).
- The three-point fit has log-log slope 0.91 (95% CI [0.54, 1.04]) where predicted is 1.0. The CI includes both 0.54 (a 46% underestimate) and 1.04 (a 4% overestimate).
- No power calculation or justification is given for why 3 seeds is sufficient to claim a match at $T=9000$ when the CI spans more than 2.8×.

**Why it matters:** This is labeled a "pre-registered test" and is the paper's main empirical evidence that the theory's fraction-regime prediction holds. Wide CIs with one bound far from the prediction weaken the claim substantially. Readers will wonder whether the data actually supports the theory or merely fails to falsify it.

**What would neutralize it:**
- Increase to 9–12 seeds at $T=9000$ and $T=3000$ to shrink the CIs. Show that the prediction 900 lies in the lower half of the CI for $T=9000$.
- Or, fit a single parametric model to all nine data points (3 lengths × 2 trigger conditions, poison count vs. length), show the predicted relationship as a curve through 2D space, and report how often the observed 50%-ASR thresholds fall within a±20% band around the prediction.
- Or, preregister the width and coverage threshold (e.g., "if the CI contains the prediction and the CI width is ≤1.5×, we accept the match") before data collection.

---

## 3. OVERCLAIMING RELATIVE TO EVIDENCE

**Quote 1:**
> "On a transformer trained from scratch, this zero-parameter prediction orders attack success across all 11 schedule, length and placement conditions we ran (Spearman $0.67$, $p=0.02$)."

**Issue:** This is stated in the abstract as a headline result, but Spearman ρ=0.67 is a *moderate* correlation, not a strong one. A $p$-value of 0.02 with $n=11$ means the null hypothesis (no rank correlation) is rejected, but it does not mean the prediction is validated in a strong sense. With 11 points, ρ=0.67 corresponds to ~7 of 11 points ranking in the same order as predicted; 3–4 are out of order. This deserves mention, but the abstract frames it as a clean success without caveating the moderate effect size.

**Quote 2:**
> "The paper's main empirical evidence that the theory's fraction-regime prediction holds."

Actually from the text: "the poison count needed for a trigger that also appears in clean data tracks the predicted number of competing clean uses (123, 346 and 914 poison examples against a predicted 100, 300 and 900 as training length triples)."

**Issue:** The word "tracks" is too strong when the intervals are wide (see Objection 3 above). At $T=9000$, the lower bound (408) is 0.45× the prediction. The data are consistent with the prediction, but "tracks" suggests tighter agreement than the CIs support.

**Quote 3:**
> "Validating the Adam prediction on a pretrained model with a natural-language corpus is the most direct next step."

**Issue:** The paper positions itself as explaining Souly et al.'s findings on real pretraining, but then relegates validation on real models to "the most direct next step." This is honest, but the claim in the abstract that the theory "explains" empirical poisoning findings should be tempered: it explains them in a toy setting, and whether it explains them on real models is open.

---

## 4. ERRORS (MATH, STATS, OR UNSUPPORTED CLAIMS)

**Error 1: Figure 6b confidence intervals appear inconsistent with the data and "3 seeds."**

The paper states "3 seeds per cell" for the rare/generic trigger sweep ($N\in\{25,75,225,675,2025\}$, density 8). Bootstrap 95% intervals are then reported:
- $T=1000$, generic: [116, 130] (12-point width, ratio 1.12)
- $T=9000$, generic: [408, 1174] (766-point width, ratio 2.87)

With only 3 seeds and log-linear interpolation, a bootstrap CI of ±3% at $T=1000$ and ±187% at $T=9000$ on a logarithmic scale does not match the expected variation. The widest intervals should be where the grid is sparsest or the mean-ASR curve is steepest. If the grid is uniform and the curve smooth, CIs should not blow up 25× from $T=1000$ to $T=9000$ unless there is an enormous amount of run-to-run variance at the longer length. The paper does not report per-run ASR curves or explain this disparity. This deserves clarification: either the data support a large variance at long training, or there is a methodological issue in the bootstrap procedure.

**Error 2: "Individual runs end close to either 0 or 1 attack success, so means carry large standard errors" is stated but not quantified.**

The text says individual runs are "close to bimodal." If nearly every run ends at ASR ≈ 0 or ASR ≈ 1, then the mean is a poor summary and the standard error of the mean is inflated. Yet Figure 5a pools 20+20+20+20+6=86 runs and reports diamonds as "means ± standard error" without flagging the bimodality in the error-bar labels. This is not an error, but a reporting gap: if the distribution is bimodal, the fraction of runs ending near 1 (not the mean) should be the headline statistic, and the standard error is misleading.

**Logical issue (not a math error): Proposition 2's fixed point assumes equilibrium.**

Proposition 2 derives a fixed point of the mean-field ODE under the assumption that poison and clean steps reach a balance. But in the experiments (Section 5.2), training lengths vary from 1k to 9k steps, and poison is scheduled uniformly across the run

## Persona: optimization

# REVIEW

## 1. SUMMARY

This paper resolves the empirical tension between count-governed and fraction-governed poisoning regimes by deriving closed-form solutions showing both as limits of a unified model, where the trigger's overlap with clean data ($\rho$) controls which regime appears; it additionally explains density effects under Adam by showing each poisoned batch moves trigger-specific parameters by a fixed amount, independent of poison density. Validation on a 0.81M-parameter transformer matches key predictions (rare-trigger thresholds, generic-trigger scaling with clean-data size) but fails to explain why attack success declines for training much longer than the poison schedule.

---

## 2. THREE STRONGEST OBJECTIONS

### **Objection 1 (Limits Scope & Validation Strategy)**

**The theory's core prediction is tested on a synthetic bigram task at 0.81M parameters, and the authors explicitly acknowledge the experiments do not validate the Adam result itself.** Proposition 3 is the claimed mechanistic novelty — it is the reason "batches closer than $1/(1-\beta_2)$ interfere" — yet the paper provides no direct evidence that this interference occurs in the transformer experiments. Figure 2(b) shows that spreading 16 poison examples over more batches helps, and clustering hurts, but this could arise from many mechanisms (e.g., spacing effects on shared parameters, gradient noise, or momentum decay). The authors note "individual runs are close to bimodal," inflating noise around means. The lack of an ablation disabling the Adam effect (e.g., using SGD on the same transformer with the same total poison budget) prevents causal attribution of the scheduling benefit to the batch-counting mechanism rather than to some other property of temporal distribution.

**Neutralizing evidence:** Direct comparison of ASR under Adam vs. SGD on the transformer, holding total poison count and placement window fixed, with sufficient seeds to show that the Adam advantage is specific to the batch-counting regime (many spaced batches beat fewer clustered batches under Adam but not under SGD, or vice versa). Alternatively, an explicit test on a larger, pretrained model showing the same interference signature.

---

### **Objection 2 (Training-Length Decline Unresolved & Predictive Capacity)**

**The theory predicts attack success is independent of training length (under $\rho=0$, Theorem 1) and proportional to the number of clean-task uses of the trigger (under $\rho>0$, Proposition 2), yet Figure 2(a) shows a sharp, significant decline at lengths beyond the poisoning window — a failure the authors acknowledge but do not explain.** They speculate that "spacing between the four poisoned batches grows as $T/3$, consistent with each dose partially decaying before the next arrives," but this is post-hoc pattern-matching, not a mechanistic derivation. A fitted decay model extrapolates to $0.04$ at $T=64{,}000$ against observed $0.13$, which the authors call "consistent," but the interval $[0,0.20]$ is so wide (spanning zero) that it excludes nothing. This is the paper's largest empirical anomaly, and it is left as a list of hypotheses.

**Neutralizing evidence:** A mechanistic extension of the theory (or the discrete-recursion model) that predicts the rise-and-decline shape and the timescale of decay as a function of poisoning schedule, with predictions fixed before a validation experiment. Or empirical evidence isolating the mechanism — e.g., direct measurement of how long a poisoned weight direction persists during clean training, or a scheduling experiment that manipulates the spacing between poison batches to test whether the decay timescale matches $T/3$.

---

### **Objection 3 (Causal Isolation & Confounding in Transformer Results)**

**The scheduling experiment (Figure 2b) confounds number of batches with placement.** The paper states "a single batch schedule necessarily sits at step 5 and is confounded with early placement," yet reports the result (mean ASR 0) without a control. This is acknowledged as a limitation, but it means the claimed prediction — "spreading a fixed budget over more batches increases attack success" — is not actually tested in isolation. The 16- and 4-batch schedules both end at step 2,995, controlling for the endpoint, but start at step 5 (both), so the absolute placement is the same; the difference is the number of *intermediate* batches. Without a placement-ablation (e.g., a 16-batch schedule concentrated in the first 500 steps vs. spread evenly to 3,000), the contribution of batch count versus the mere availability of clean steps after poisoning is unresolved.

**Neutralizing evidence:** A scheduling experiment with 2–3 placement patterns held constant and batch count varied (or vice versa), showing the effect persists and quantifying its size relative to placement. The "AdamW versus SGD" result mentioned as running would provide stronger evidence if it includes the scheduling comparison.

---

## 3. OVERCLAIMS RELATIVE TO EVIDENCE

1. **"This zero-parameter prediction orders attack success across all 11 schedule, length and placement conditions we ran (Spearman $0.67$, $p=0.02$)."** This refers to Proposition 3's prediction of the Adam effect, yet the paper never directly validates Proposition 3 on the transformer — only its consequences. A Spearman $\rho$ of 0.67 at 11 points is weak correlation (explaining ~45% of variance); with $n=11$, the $p$-value is dependent on the specifics of the statistic and does not measure effect size. The framing "zero-parameter prediction" is also misleading: $D_0$ is not a free parameter, but the prediction still depends on $(\beta_1, \beta_2, \eta)$, which are not fit.

2. **"In a pre-registered test on the same transformer, the poison needed for a trigger that also appears in clean data tracks the predicted number of competing clean uses (123, 346 and 914 poison examples against a predicted 100, 300 and 900 as training length triples)."** The 95% confidence intervals are wide: $[408, 1174]$ at the longest length, which is 41% of the estimate on the upper end. The statement "tracks" suggests tight agreement; the intervals overlap the predictions but do not confirm them to high precision. The log–log slope of 0.91 (predicted 1) with a 95% interval of $[0.54, 1.04]$ is consistent with the prediction but is also consistent with a slope as low as 0.54, which would be substantially weaker scaling.

3. **"We resolve this tension with a solvable model in which both regimes appear as limits."** The "resolution" applies only to the scalar model; the transformer experiments show the full theory fails (decline at long training lengths) and succeed only partially (scheduling and rare-vs-generic effects). The word "resolve" overstates the scope.

---

## 4. ERRORS & UNSUPPORTED CLAIMS

1. **Bias correction and weight decay in Proposition 3:** The proposition assumes these are neglected ("neglect bias correction and weight decay"), but the transformer uses AdamW with `weight_decay=0.01`, which is not negligible. The error contribution of omitting weight decay is not quantified. This is a gap between the theorem and the experimental setup.

2. **Statistical testing on small sample sizes:** The Mann–Whitney test comparing ASR at $T=500$ vs. $T=2{,}000$–$8{,}000$ uses $n=20$ and $n=40$ respectively, with highly non-normal distributions (bimodal), yet the $p$-value is reported as exact ($p=7\times10^{-5}$) without mentioning whether the test is appropriate or if a permutation/bootstrap alternative was used. The paper does not justify the choice of Mann–Whitney for this comparison.

3. **"The two sweeps agree on this shape but not on whether the peak lies at 2,000 or 8,000."** This is reported as though the disagreement is a minor detail, but if the peak location is uncertain, so is the interpretation of the mechanism. This uncertainty should be propagated into the claims about what is happening at long training lengths.

4. **Missing baseline in the placement experiment:** The paper compares 16-batch vs. 4-batch vs. 1-batch schedules, all using $N=16$ poison total. But 1-batch schedules must place poison at step 5 (the code says "from step 5 to step $T-5$"), so the 1-batch result is not a fair test of the batching hypothesis — it is confounded with early placement. This is acknowledged but reported anyway as if it were a clean test.

---

## 5. RATING AND CONFIDENCE

**RATING: 6/10**

**CONFIDENCE: 3/5**

### Justification

**Strengths:**
- The theoretical contributions (Theorems 1–2) are novel and correctly derived. The mean-field ODE analysis is sound and the closed forms are validated against discrete recursions.
- The unification of count-governed and fraction-governed regimes through $\rho$ is conceptually valuable and matches prior empirical work.
- Proposition 3 (Adam counting batches, not examples) is a genuinely interesting insight, even if not fully validated.
- The paper is transparent about limitations (training-length decline, synthetic task, small model).

**Weaknesses:**
- The Adam result (Proposition 3), the paper's main novelty, is not directly validated on the transformer. Only its indirect consequences are tested, and confounding is present (scheduling vs. placement).
- The training-length decline is a major empirical failure, left as post-hoc speculation with a failed extrapolation. This is ~20% of the reported experiments and it is not explained.
- The scheduling experiment (Figure 2b) conflates batch count with placement; a 1-batch schedule at step 5 is not a fair control.
- Confidence intervals on the rare-vs-generic scaling are wide (e.g., $[408, 1174]$ at $T=9{,}000$); the agreement with theory is real but not tight.
- Validation is limited to a 0.81M synthetic-task transformer. Scaling to real pretraining or larger models is absent.
- The AI-disclosure statement is thorough, but the claim that "every derivation was checked against a discrete-recursion simulation" is not independently verifiable; the supplementary scripts are required to verify this claim, which is a trust barrier.

### Single Change That Would Most Raise Rating

**Direct validation of Proposition 3 (the Adam batch-counting effect) by comparing Adam vs. SGD on the transformer under controlled scheduling and placement.** If Adam shows the predicted advantage (spaced batches > clustered batches) and SGD does not, this would isolate the mechanism and support the paper's central claim. Alternatively, a mechanistic explanation for the training-length decline that is validated prospectively would also substantially strengthen the work.

## Persona: ac

# REVIEW

## 1. SUMMARY

This paper derives closed-form explanations for two puzzling empirical phenomena in backdoor poisoning: why a near-constant poison count succeeds across dataset sizes, and why higher per-batch poison density paradoxically reduces attack efficiency. The authors show the count regime emerges when triggers are rare ($\rho=0$), revert to fraction-governance when triggers overlap clean data ($\rho>0$), and prove that under Adam (unlike SGD) each poisoned batch moves trigger-specific parameters by a fixed amount, explaining the density effect via second-moment memory interference.

## 2. THREE STRONGEST OBJECTIONS

**Objection 1: The model does not predict a core failure mode (rating: −2).**
The theory predicts attack success should be monotonic or quasi-monotonic in training length (for $\rho=0$, Theorem 1 has no length dependence; for $\rho>0$, the fixed point in Prop 2 is length-independent). Instead, Figure 1a shows a *rise-then-steep-decline* pattern: ASR peaks at $T=2{,}000$–$8{,}000$ and drops to 0.13 at $T=64{,}000$. The authors acknowledge this ("decline is not forgetting..."), hypothesize spacing-induced decay, and fit a post-hoc rise-times-decay curve that "predicts 0.04" but observes 0.13—consistent only by the bootstrap interval's tail. This is not a small edge case: the effect halves the attack at the longest length tested and is explicitly called a limitation. **What would neutralize it:** A mechanistic model (even phenomenological) predicting this decay. Alternatively, evidence that the effect vanishes or reverses on a larger model or natural task, suggesting it is an artifact of scale—though the authors explicitly avoid this claim.

**Objection 2: Adam prediction is tested only on the theory's own simplifying regime (rating: −1.5).**
Proposition 3 (Adam events) is derived for trigger-specific parameters receiving gradient *only on poisoned steps*, at $\rho=0$, in the "small-margin regime," and assumes $\Delta \gg 1/(1-\beta_1)$ (poisoned batches spaced far apart). The validation (Figure 1b, scheduling experiment) tests spreading $N=16$ examples across 1, 4, or 16 batches at $T=3{,}000$. This *is* $\rho=0$, *is* small margins (ASR 0–0.53), and *does* space batches far apart (spacing $\approx 1{,}000$/$k$ steps). But the paper has no head-to-head SGD vs. AdamW comparison at the same conditions showing Adam predicts the observed direction better than SGD. The paper says it "tests this directly by comparing the two optimizers" in Section 5.4, but then reports "results for one experiment (AdamW vs SGD, 'events') are still running and its numbers are not yet in the text." **What would neutralize it:** Running and reporting the SGD vs. AdamW experiment at identical schedules, showing Adam predicts the direction and SGD does not. Or showing the effect size matches the model's prediction numerically, not just directionally.

**Objection 3: "Fixed prediction" on $\rho>0$ was set only after observing the scale of clean uses (rating: −1).**
The paper claims a pre-registered test: "with poison and clean uses of the trigger entering symmetrically ($c=\rho$), $z^*>0$ exactly when poison examples outnumber competing clean uses; at 0.1 clean uses per step this predicts thresholds near 100, 300 and 900 poison examples...we fixed this prediction before running the sweep." However, the "0.1 clean uses per step" is *chosen* based on earlier exploratory sweeps (implied by "A fraction of clean sequences contain...at an average of 0.1 occurrences per step"). The prediction is therefore not blind—the experimenters knew the clean frequency, designed the clean frequency, and then set the poison-count predictions to match it. Figure 2b shows observed thresholds of 123, 346 and 914 versus predicted 100, 300 and 900: they agree to ~23%, but the prediction's parameters were calibrated to a prior observation. **What would neutralize it:** A truly independent pre-registration (locked before any pilot data) with the clean frequency and poison count grid specified beforehand, or pre-registration of the method with the parameters left as unknowns to be estimated.

## 3. OVERCLAIMING

- **"closed form"** (Abstract, Contributions): Equation 1 ($z(s) + e^z = 1 + \eta c^2 s$) is a closed form for the implicit margin $z(s)$, but requires numerical inversion for practical use. The paper does invert it (asymptotic regimes given explicitly), so this is fair, but "closed form" overstates the explicitness.

- **"Validation on a transformer"** (Contributions, line 3): The transformer is synthetic (64-token bigram vocabulary, 0.81M parameters trained from scratch). This is appropriate for a theory paper but "transformer...from scratch" without "synthetic" in the contribution list reads as deployed-scale.

- **"zero-parameter prediction orders attack success across all 11 schedule, length and placement conditions"** (Abstract): This refers to the Spearman correlation ($\rho=0.67$, $p=0.02$) in scheduling experiments. "Zero-parameter" is misleading—the prediction depends on $\beta_1=0.9$ and $\beta_2=0.999$ (fixed by convention), which are not fit, but are parameters. A fairer statement: "prediction independent of poison-count or gradient-magnitude parameters."

## 4. ERRORS

**Claim vs. evidence mismatch in Figure 1a caption:**  
The caption states "Grey points are individual runs...diamonds are means $\pm$ standard error." The error bars shown on the diamonds in the published plot appear asymmetric and are labeled with $n=20,20,20,20,6$. Standard error should be symmetric (SE$=\sigma/\sqrt{n}$). Either the bars are bootstrap-percentile or quantile intervals (not SEs), or there is an error. The text says "two independent sweeps pooled" but does not confirm whether the two sweeps are statistically aggregated or treated as separate replicates; pooling rules affect the SE calculation. **Resolution required:** Specify whether error bars are classical SEs, bootstrap intervals, or credible intervals.

**Proposition 2 proof—sign confusion:**  
The fixed-point derivation reads: "fixed-point condition $c^2 \lambda \sigma(-z^*) = \rho^2 \sigma(z^*)$". At $z^*>0$ (attack succeeds), $\sigma(z^*)$ is large and $\sigma(-z^*)$ is small, so the condition says the poison term (LHS) is small and the clean term (RHS) is large. But the ODE is $dz/dt = \eta c^2 \sigma(-z^*)\lambda - \eta \rho^2 \sigma(z^*)$: poison pushes up, clean pushes down. At equilibrium, the magnitudes should balance, which they do, but the notation is backwards—the fixed-point equation has $\sigma(-z^*)$ on the poison side and $\sigma(z^*)$ on the clean side, which is inverted relative to the ODE. Tracing through: the proof rearranges to $\sigma(z^*) = c^2\lambda/(\rho^2+c^2\lambda)$, which is correct, and the final answer $e^{z^*}\rho^2 = c^2\lambda$ follows correctly. **The result is correct; the intermediate line is notational clutter but not an error.** Restate it as: $c^2 \lambda (1-\sigma(z^*)) = \rho^2 \sigma(z^*)$ for clarity.

**Missing data:** The paper announces "results for one experiment (AdamW vs SGD, 'events') are still running." This experiment is cited as validating the core Adam prediction. Without it, the Adam claim rests only on the scheduling experiment (Figure 1b), which does not isolate the Adam effect from other scheduling-dependent phenomena.

## 5. RATING & CONFIDENCE

**RATING: 5/10**  
**CONFIDENCE: 3/5**

**Justification:**  
The theoretical contributions (closed forms for count vs. fraction regimes, Adam-events formula) are novel and rigorously derived. The model correctly predicts several phenomena in controlled settings. However, the core dynamics fail at scale (Figure 1a decay), the centerpiece Adam experiment is missing, the $\rho>0$ prediction is not truly blind, and the scope is narrow (synthetic task, small model, $\rho$ is an unrealistic binary). The paper reads as strong theory with incomplete empirical validation.

**Single change that would most raise the rating:**  
**Run and report the SGD vs. AdamW scheduling experiment now.** If the Adam prediction outperforms SGD directionally (or numerically) at matched conditions, the paper jumps to 6–7/10: the theory would be empirically grounded on its own terms. If SGD and Adam behave similarly, the Adam story collapses and the rating stays at 4–5/10.