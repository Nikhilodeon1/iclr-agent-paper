# A Solvable Model of Near-Constant-Count Data Poisoning

*Draft — intro/related work/problem statement only. Abstract, results, and conclusion
pending final GPU validation numbers.*

## Abstract (placeholder — fill in after Pods 1-5 land)

Recent work shows that backdoor poisoning attacks on large language models succeed once
a near-constant number of poisoned documents have been seen during pretraining,
essentially independent of clean corpus size — a finding the original authors explicitly
flag as unexplained. We give the missing mechanism: in a solvable online-SGD model, the
attack's success threshold is governed by the *count* of poison examples when the
trigger direction is rare/orthogonal to genuine training signal, and by the *fraction*
of poisoned examples — recovering existing theory — when the trigger carries real
distributional meaning. A single interpretable parameter, the trigger's signal overlap
$\rho$, continuously interpolates between these two regimes via a closed-form fixed
point of a derived mean-field ODE. [PLACEHOLDER: one sentence summarizing the real-model
validation result once available.]

## 1. Introduction

Data poisoning attacks that plant a backdoor during pretraining are usually assumed to
require a poison budget that scales with the size of the clean corpus — more clean data,
more poison needed to compete with it. Souly et al. (2025) overturned this assumption
empirically: across model sizes from 600M to 13B parameters and clean corpora spanning
several orders of magnitude, a **near-constant number of poisoned documents** (roughly
250) was sufficient to implant a backdoor, and this count was essentially insensitive to
*how* the poisoned examples were scheduled across training — their density within a
batch and the frequency of poisoned batches barely mattered. The authors explicitly state
they lack an explanation: they hypothesize the effect relates to "a certain number of
sequential gradient steps on poisoned data" but note this as "an area for further
investigation," concluding "we do not have a good explanation for this phenomena."

This near-constant-count regime appears to be in tension with an established body of
poisoning theory that treats the attack's success threshold as governed by the
*fraction* of poisoned training examples, not their absolute count (Flynn & Granziol,
2025; a high-dimensional GLM analysis of trigger strength, 2026) — both derived via
random-matrix/statistical-physics techniques for high-dimensional linear and generalized
linear models. If poisoning theory already says the threshold is fraction-governed, why
would a real backdoor attack show count-governed, near-constant behavior instead?

We resolve this by showing both regimes are limits of the same solvable model, separated
by a single parameter: how much the trigger direction overlaps with genuine signal used
by the clean task. When the trigger is rare — a token or pattern that carries no
information about the clean objective — the attack's threshold is governed by count
alone, exactly reproducing Souly et al.'s empirical findings including their
density/frequency invariance. When the trigger is generic — a pattern that also occurs
meaningfully in clean data — the threshold reverts to the fraction-governed regime the
existing literature derives. The transition between these regimes is a closed-form fixed
point of a mean-field ODE we derive from the underlying stochastic recursion, not a
fitted curve.

**Contributions.**
1. The first mechanistic account of Souly et al.'s two central unexplained empirical
   findings — scheduling/density invariance, and near-constant poison count independent
   of clean corpus size — derived in closed form (Theorem 1).
2. A single interpretable parameter, the trigger-signal overlap $\rho$, that
   continuously interpolates between the count-governed regime (this paper) and the
   fraction-governed regime of existing theory (Proposition 2), unifying results
   previously reported as apparently distinct phenomena.
3. [PLACEHOLDER: real-transformer validation summary, once Pods 1-5 finish — schedule
   invariance, dataset-size behavior including any boundary effects, and the rare-vs-
   generic trigger comparison.]

## 2. Related Work

**Empirical poisoning at pretraining scale.** Souly et al. (2025) is the paper this work
explains: they report near-constant poison counts across model scale and explicitly
state they lack a mechanism. Earlier empirical poisoning work (Carlini et al., 2023;
Wan et al., 2023) established that LLMs are vulnerable to small amounts of poisoned
pretraining data but did not characterize the count-vs-fraction distinction or its
dependence on trigger rarity.

**Fraction-governed poisoning theory.** Flynn & Granziol (2025) derive a closed-form
poisoning threshold for ridge regression via random matrix theory, parameterizing the
attack by the *fraction* $\theta$ of one class relabeled along a fixed direction. A
concurrent high-dimensional GLM analysis (2026) studies trigger *strength* at a fixed
poison ratio and finds a non-monotonic success curve as strength varies. Both implicitly
study directions with $O(1)$ overlap with the ambient signal — our $\rho=O(1)$ regime —
and neither addresses a count-governed limit, scheduling invariance, or dataset-size
invariance. We recover their scaling form as the dense/generic-trigger limit of our
model (§4) rather than treating it as a separate phenomenon.

**Theoretical tools.** Our derivation uses a mean-field/stochastic-approximation
reduction of a discrete SGD recursion to an autonomous ODE, in the spirit of classical
stochastic approximation theory (Robbins & Monro, 1951; Benveniste et al., 1990) and its
modern use in analyzing high-dimensional learning dynamics. To our knowledge this
specific reduction has not been applied to backdoor/poisoning attack theory before — a
targeted arXiv/Semantic Scholar search for "mean-field," "stochastic approximation," or
"homogenization" combined with "backdoor" or "poisoning" returned no close prior work
(searched [DATE]).

## 3. Problem Setup

We study online (stochastic) gradient descent on a linear/logistic classifier with
weight $w\in\mathbb{R}^p$, trained on a stream that mixes clean and poisoned examples.
The input is decomposed into a trigger coordinate $x_1$ and a clean-signal block
$x_{2:p}$.

- **Poisoned examples** set $x_1=c$ for a fixed constant $c$ and assign the attacker's
  target label, regardless of the true clean label.
- **Clean examples** draw $x_1$ from a distribution correlated with the clean
  classification task with overlap strength $\rho\in[0,1]$: at $\rho=0$, $x_1$ carries no
  information about the clean task (a *rare* trigger — e.g. a token or pattern that
  never occurs in ordinary data); at $\rho>0$, $x_1$ is itself a genuine, if partial,
  signal feature for the clean task (a *generic* trigger — e.g. a common token whose
  usual meaning is unrelated to the attack).

Let $z=c\cdot w_1$ denote the model's margin along the trigger direction. The attack
succeeds once $z$ exceeds the threshold set by the decision rule (equivalently, once the
model's prediction on trigger-containing inputs flips to the attacker's target). Our
goal is to characterize, as a function of $\rho$, the poison count $N^*$ needed to reach
a fixed target margin, and how $N^*$ depends on the total number of clean training steps
$n_{clean}$ and on how the poison examples are scheduled across training.

*(§4-5: Theorem 1 (ρ=0) and Proposition 2 (ρ>0) reproduced from theory.md follow here in
the full draft; omitted from this excerpt for brevity — see theory.md for the complete
derivations and their numerical/adversarial verification.)*
