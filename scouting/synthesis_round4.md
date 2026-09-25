# Synthesis of 10 helper-LLM reports (round 4)

## Headline: near-unanimous convergence
All 10 helpers received the identical brief. Despite that, they converged heavily on the same two ideas without being told to:
- **Length generalization theory for masked diffusion language models** — proposed as the #1 idea by 8 of 10 helpers.
- **Poisoning near-constant-count theory** — proposed by 9 of 10 helpers, usually #2.
- **Continual pretraining** and **proxy-model rank reversal** — independently flagged as dead or nearly dead by 6+ helpers, matching our round-2/3 assessment.

Independent convergence across 10 separate models is itself evidence: if ten different reasoning processes land on the same two ideas as the strongest options, that is a real signal, not a coincidence.

## Verified against arXiv (by me, after the fact)

**Continual pretraining — CONFIRMED DEAD.** Helpers named the CMR Scaling Law (Gu et al., EMNLP 2024), the D-CPT Law, and a theory paper titled "Replay Can Provably Increase Forgetting." Between the empirical scaling law and a paper that already proves things about replay, the theoretical space is occupied. Matches our own round-2/3 read. **Drop this one.**

**Proxy-model rank reversal — CONFIRMED DEAD.** "Can Small Training Runs Reliably Guide Data Curation? Rethinking Proxy-Model Practice" (arXiv 2512.24503, which we had already flagged in round 2) is now read by multiple helpers as already containing the empirical finding, a random-feature theoretical account, and a fix. **Drop this one.**

**Diffusion-LM length generalization — SURVIVES, but weaker than round 3 suggested.** I independently found **LongLLaDA: Unlocking Long Context Capabilities in Diffusion LLMs** (arXiv 2506.14429, June 2025) — a systematic empirical comparison of diffusion vs. autoregressive long-context behavior, explained via RoPE-scaling theory, with a training-free extrapolation method. This is a real, relevant, and fairly well-known prior paper that a reviewer would expect us to cite and differentiate from. It is empirical + a practical fix for one specific pretrained model family (LLaDA), not an exact/provable theorem about when a masked-diffusion model trained on length ≤ L is guaranteed (or provably fails) to generalize to length > L, in the style of the ICLR26 Oral papers on autoregressive/SSM length generalization. I also confirmed a general (architecture-agnostic) framework paper, "Non-Asymptotic Length Generalization" (2506.03085), that formalizes length generalization via a decidability/complexity argument — useful related work, not a kill, since it doesn't touch the diffusion decoding mechanism specifically.
**Verdict: alive, but the paper's contribution must be framed explicitly against LongLLaDA (ours = exact mechanism-level theorem on a tractable model, not RoPE engineering) and against the general non-asymptotic framework.**

**Poisoning near-constant-count theory — SURVIVES.** No helper or my own search found an existing closed-form/phase-diagram theory of the specific near-constant-count phenomenon (Souly et al.'s ~250-document finding, likely the Anthropic/UK AISI/Alan Turing Institute study, arXiv 2510.07192, now with 80+ citations). The two closest theory papers (ridge-regression poisoning theory, arXiv 2505.15175; a proportional-regime GLM backdoor theory, arXiv 2605.22481) are both fraction-based, not count-based. Three helpers independently converged on the same mechanism — a poisoned direction that is rare/spectrally isolated only receives gradient signal from the poison examples themselves, making the required count roughly independent of clean dataset size — which is a good sign the mechanism is right, but it also means the "obvious" version of this result is not novel by itself. **The paper must go further: an exact phase diagram unifying the count-based and fraction-based regimes as limits of one model** (recovering both existing theory papers as special cases), plus a defender-facing prescription. That is what would separate it from "yet another poisoning paper."

## New ideas surfaced this round (unvetted, backups only)
- **Zipf rare-feature sample complexity** (Agent 5): a single unifying theory where poisoning-count and proxy-ranking-reversal both fall out as corollaries of one rare-feature-learning result. Interesting if it holds up, but proxy-reversal is already dead, so this would need to stand on the poisoning corollary alone.
- **Spectral theory of post-training quantization degradation** (Agent 2): fresh, not checked against ICLR26 corpus or arXiv yet.
- **Benchmark-contamination "distributional similarity" audit** (Agent 4): fresh, not checked.

## Recommendation
Commit primary effort to **poisoning near-constant-count theory** as the lead (least eroded by this round's checks, clear unifying angle available, safety relevance broadens reviewer interest) and keep **diffusion-LM length generalization** as a strong second, provided the write-up explicitly out-flanks LongLLaDA. Run both through a real pilot before picking one — a CPU pilot can settle whether the poisoning phase-diagram mechanism is actually exact and unifying, and whether a small masked-diffusion model shows a length-generalization failure mode that differs qualitatively from the AR/SSM case.
