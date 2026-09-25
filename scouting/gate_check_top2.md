# Gate check: top 2 ideas

Kill rule applied: if any single paper already makes the core claim, drop the idea. Neither idea is killed. Both get a precise, defensible delta statement below.

## Idea 1: A solvable theory of near-constant-count data poisoning

**Core claim:** in a high-dimensional linear/GLM model, whether a poisoning attack's success threshold is governed by the *fraction* of the training set that is poisoned or by the *absolute count* of poisoned examples depends on how the trigger direction aligns with the data covariance spectrum — rare/low-eigenvalue directions give a count-governed regime; generic directions give the fraction-governed regime existing theory already covers. This reconciles a real conflict in the literature (see below) rather than adding a new poisoning-theory paper to a pile.

**Closest prior work, with exact deltas:**
1. **Souly et al., "Poisoning Attacks on LLMs Require a Near-constant Number of Poison Samples" (arXiv 2510.07192, ~80 citations)** — the empirical finding this theory explains. The paper explicitly states it lacks a mechanism: *"We hypothesise this is due to models requiring a certain number of sequential gradient steps on poisoned data... but note this as an area for further investigation"* and *"We do not have a good explanation for this phenomena."* **Delta: we supply the mechanism they say they're missing.**
2. **"A Linear Approach to Data Poisoning" (arXiv 2505.15175)** — ridge regression, poison modeled as a **θ-fraction** of one class shifted by a direction v and relabeled; closed-form via random matrix theory. **Delta: fraction-parameterized, not count-parameterized; recovered as our theory's dense/generic-direction limit.**
3. **"When Stronger Triggers Backfire: A High-Dimensional Theory of Backdoor Attacks" (arXiv 2605.22481)** — regularized GLMs, proportional regime, studies **trigger strength α at fixed poison ratio**, finds a non-monotonic ASR-vs-α curve. **Delta: different free variable (strength, not count/scheduling); doesn't touch the density-frequency invariance or dataset-size invariance Souly et al. report.**
4. Checked the full citation graph of paper 1 (84 citing papers) for anything published since that supplies this theory — none found. Closest are a mechanistic-interpretability study of trigger circuits (different question: which attention heads, not sample complexity) and an unrelated statistical-physics analogy paper.
5. Jagielski et al. 2018, Steinhardt et al. 2017, Carlini et al. 2023 — earlier poisoning-theory/empirics, all pre-date and don't address the near-constant-count phenomenon (it was reported in Oct 2025).

**Pilot evidence (this session):** a minimal online-logistic-regression model with an orthogonal trigger coordinate reproduced both of Souly et al.'s unexplained findings — scheduling/density invariance for fixed poison count, and an ASR plateau (not decay to zero) as clean data scale grows.

**Verdict: SURVIVES.** No paper makes this claim. Proceed to formal derivation.

## Idea 2: Length-generalization theory for masked diffusion language models

**Core claim:** derive an exact (necessary-and-sufficient, not just an upper bound) train-length/test-length boundary for a solvable masked-diffusion sequence model, in the style of the ICLR 2026 Oral papers for autoregressive Transformers and SSMs — but diffusion's parallel, bidirectional, any-order decoding structurally breaks both of those papers' proof techniques, so a new argument is required, not just a missing derivation.

**Closest prior work, with exact deltas:**
1. **"Quantitative Bounds for Length Generalization in Transformers" (ICLR 2026 Oral, rated 7.0)** — proves length generalization occurs when a transformer's behavior on longer sequences can be *simulated* by its behavior on the shorter sequences seen in training. **Delta: this simulation argument is built on the causal, left-to-right decoding structure of autoregressive transformers. Masked diffusion has no such structure** (all positions are visible/updated in parallel, in an arbitrary order) — the proof technique does not carry over, not merely "hasn't been applied yet."
2. **"The Expressive Limits of Diagonal SSMs for State-Tracking" (ICLR 2026 Oral)** — group-theoretic (Abelian vs. non-Abelian) expressivity bounds tied to the linear-recurrent hidden-state update of SSMs. **Delta: relies on sequential recurrence; masked diffusion has no recurrent hidden state to analyze this way.**
3. **LongLLaDA (arXiv 2506.14429)** — empirical comparison of diffusion vs. autoregressive long-context behavior, explained via RoPE-scaling theory, with a training-free context-extension method for the already-pretrained LLaDA model. Its own Limitations section states results are inference-stage only, on the LLaDA series specifically. **Delta: no train-from-scratch generalization theorem is attempted; this is an empirical-plus-engineering paper, not a proof.**
4. **"Non-Asymptotic Length Generalization" (arXiv 2506.03085)** — a general, architecture-agnostic framework (length complexity via a decidability/complexity-measure argument for hypothesis classes). **Delta: doesn't engage with the diffusion decoding mechanism (iterative denoising) at all; a useful related-work citation, not competing content.**
5. **Backdoor-attack papers on diffusion LMs (SHADOWMASK 2605.19262, BadDLM 2605.09397, DiSP 2602.22246)** — checked because they're the only recent theoretical treatments of masked-diffusion training dynamics; none address length generalization, confirming this specific question is untouched even by the closest active subfield.
6. Re-ran 5 fresh arXiv queries this round (train-short/test-long, out-of-distribution length, diffusion + length extrapolation) — no new hits beyond what was already catalogued.

**Pilot status:** inconclusive. My own from-scratch small-model pilot this session had a bug (sinusoidal position fix broke the diffusion training loop) and needs real engineering time, not resolved by this gate check.

**Verdict: SURVIVES, novelty confirmed strong** — arguably stronger than round 3's assessment, since the reason existing proofs don't transfer is now a structural argument, not just an absence. **But the pilot debt is real and unresolved.**

## Decision
Both ideas pass the gate. Idea 1 (poisoning) has a working pilot and a precisely reconciled literature conflict — strongest position of the two. Idea 2 (diffusion length generalization) has the stronger *pure* novelty argument (structural, not just absent) but needs its pilot fixed before its risk profile is comparable.
