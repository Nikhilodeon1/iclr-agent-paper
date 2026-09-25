# Brief for research-strategy LLM helpers (paste this whole document, no other context needed)

## Your role
You are advising an autonomous AI research agent that is trying to produce one research paper strong enough for **ICLR's main track**. The paper will be judged by a panel of former ICLR reviewers. It will not actually be submitted to the conference — this is a timed capability test — but it must be judged as if it were a real submission. **The only objective that matters is maximizing the probability a panel of experienced ICLR reviewers would accept this paper.** Nothing else about the project matters — not elegance, not ambition, not what would make a good story. Optimize ruthlessly for acceptance odds.

## Who is doing the work
- One AI agent does all the research, coding, and writing (no human researchers, no grad students).
- It has a second, cheaper/faster LLM available for unlimited brainstorming and verification help (that's you, or another instance like you).
- It has a code sandbox: 16 CPU cores always available, no persistent GPU by default.
- It can burst to **up to 5 GPU-hours on an H100, or up to 20 GPU-hours on a cheaper card (RTX-4090/A10 class), via RunPod**, for real-model validation. This is a hard ceiling, not a typical training budget — no pretraining runs, no multi-GPU jobs. Assume validation on models in the 1M-100M parameter range trained from scratch, or CPU-feasible probing of a small (≤2B) open-weight model.
- Roughly $8 of budget for anything else (API calls, data access).
- Time is limited to on the order of hours to a few days, not weeks or months. A slow-burning multi-month research program is not a valid answer.

## What "good enough for ICLR" concretely looks like
From analyzing the actual ICLR 2026 accepted-paper set (not folklore about what reviewers like), papers that succeed in this compute-constrained regime follow one of these templates:
1. **Exact/closed-form theory + real small-model validation.** Derive a precise result (a scaling exponent, a phase boundary, a closed-form rate) in a tractable model (linear/random-feature regression, a small solvable attention or SSM model), then confirm the predicted qualitative behavior — a crossover, a threshold, an exponent — on an actual small trained model. Loose bounds or purely qualitative theory do not compete in this cluster.
2. **An audit that overturns or reconciles an empirical practice.** Show a widely-used method or evaluation practice fails in an important, previously undocumented way, and provide a fix. Bonus if the paper resolves an apparent tension between two existing published results rather than just poking a new hole.
3. Do NOT propose "we built a new model/method that beats SOTA on N benchmarks." That framing requires compute and baselines this project cannot afford, and draws the harshest reviewer scrutiny.

Every candidate idea you propose must be executable end-to-end within the compute/time budget above, including the real-model validation step, not just the theory.

## Ground truth data (use this instead of guessing about crowding — general impressions about "hot" or "niche" fields are frequently wrong)
Computed directly from the full ICLR 2026 submission corpus (19,814 submissions, decisions included):
- Overall: 27.4% accepted.
- By primary area, accepted-share-of-decided-submissions (only decided = accepted+rejected, excludes withdrawals): **learning theory 46.3%** (highest of 21 areas), foundation/frontier models 43.7%, probabilistic methods 37.9%, generative models 43.2%, reinforcement learning 38.7% ... down to transfer/meta-learning 31.2% (lowest).
- Within learning theory, by keyword: scaling-law/linear-regression/random-matrix theory papers accept around 50-55%; generalization-bound papers only ~19%.
- **Do not trust a priori claims that a topic is "low-competition."** We checked several topics commonly recommended as under-the-radar and found the opposite: mechanistic interpretability had 188 ICLR26 submissions (41% decided-accept — not a hidden gem, just a normal-sized crowded cluster), data pruning/coreset selection had 51 submissions and only a 30% decided-accept rate (worse than average), PEFT/unlearning evaluation had 115 submissions at 28%. If you recommend a topic, say explicitly whether you have real evidence of its submission volume, or flag that you're guessing.
- The single best-performing cluster found so far by this method: **length generalization / algorithmic reasoning theory** (sequence models trained on short inputs, analyzed/proven to generalize or fail on longer ones) — 77 submissions, 47% decided-accept, including 4 Orals rated 7.0 at ICLR 2026 (exact bounds on required training length for Transformers; expressivity limits of diagonal SSMs for state-tracking; tool-use unlocking length generalization in SSMs; a unifying sequential-parallel-duality theory). This is the strongest signal found across three rounds of scouting — but it means the obvious theorems in this space are already taken by strong papers. Anything proposed here must clearly clear that bar, not just enter the topic.

## Ideas already ruled out — do not re-propose any of these or close variants
Killed after checking against arXiv / Semantic Scholar / the ICLR 2026 corpus, or after a pilot experiment:
- Train/test duplicate-row leakage inflating tabular foundation models (TabPFN/TabICL) vs. GBDTs — **piloted, NO-GO**: effect not present on 22 OpenML-CC18 datasets (accuracy gap on novel vs. duplicated rows statistically indistinguishable, p=0.35).
- Martingale-posterior / Bayesian-coherence tests for prior-data-fitted tabular networks (TabPFN) — prior work already exists (uncertainty quantification via martingale posteriors; TabMGP).
- Time-series foundation model zero-shot evaluation leakage audits.
- Theory of parallel-decoding error / optimal unmasking schedules in masked diffusion language models.
- Watermarking diffusion (non-autoregressive) language models.
- Unbiased estimators / identifiability limits for majority-vote (maj@k) or best-of-k accuracy curves.
- Solvable model explaining the RLVR pass@k crossover (RL improves pass@1 but shrinks pass@k).
- Subliminal learning; emergent misalignment — both are already extremely saturated (dozens of 2026 papers each).
- Mechanistic explanation for why diffusion language models are more data-efficient than autoregressive ones under repeated data ("super data learners" line of work) — already has direct 2025-2026 follow-ups.
- Theory of transfer scaling laws (pretrain -> fine-tune "effective data transferred" law) in a solvable linear-regression model — not fully killed, but a close prior (Wu et al., pretraining+fine-tuning SGD in linear regression) exists and must be explicitly out-flanked, not ignored.
- DiLoCo / local-SGD compute-optimal schedules in a high-dimensional linear-regression model — the core solvable model (LA-DiLoCo) is already published (arXiv 2603.26954).
- Optimal model-growth (width/depth expansion mid-training) schedules — an ICLR 2026 paper on exactly this ("What is an Optimal Growth Schedule for LLMs? A Theoretical Study") was withdrawn but the ground is clearly already worked.
- Sequential locate-and-edit model-editing capacity/collapse theory — prior work (Norm Anchors Make Model Edits Last) already characterizes the norm-growth failure mode.
- Beyond-prior generalization / misspecification theory for prior-data-fitted tabular networks — prior theoretical work exists (statistical foundations of PFNs; frequentist consistency for causal inference).

## Ideas currently alive (not yet killed) — feel free to strengthen these, attack them, or beat them
1. **Why poisoning attacks need a near-constant number of samples, not a constant fraction.** Explain a 2025 empirical finding (about 250 poisoned pretraining documents compromise models from 600M to 13B parameters) with a solvable high-dimensional model, giving a phase diagram of when the count-based regime holds vs. when it breaks down (trigger rarity, learning-rate schedule, defenses). Two close papers exist in the proportional high-dimensional regime (ridge regression poisoning theory; a GLM backdoor theory) but neither addresses the specific near-constant-count phenomenon directly.
2. **Continual pretraining, solved.** Closed-form theory in a solvable model for the optimal replay fraction and learning-rate re-warming schedule, reproducing the shape of the published empirical "critical mixture ratio" scaling law.
3. **When do small proxy training runs give the wrong ranking of data recipes?** Show that recipe rankings can provably reverse between small-scale proxy runs and full-scale runs, and give a corrected extrapolation protocol. Motivated by a 2025 empirical paper showing proxy-model conclusions can flip with minor hyperparameter changes.
4. **A theory of length generalization for diffusion language models.** All 4 ICLR26 Orals on length-generalization theory (see above) analyze autoregressive Transformers or state-space models. Diffusion (non-autoregressive, any-order, bidirectional) language models generate all positions in parallel, so none of those proofs transfer directly, and — after checking roughly a dozen close arXiv papers on diffusion-LM variable-length generation — every one of them addresses a different, engineering problem (how to choose/control output length at inference time), not the train-short/test-long generalization-bound question. This looks open but has NOT been deeply gate-checked (its closest potential competitors are too recent to have a mature citation trail). Sanity-check this claim if you can — do you recognize any paper, published or in-progress, that already proves length-generalization bounds for diffusion/masked/any-order language models specifically?

## What we need from you
Give us your single best set of ideas — quality over quantity. Provide **at most 5 ideas**, each in exactly this format:

```
### Idea N: <title>
- Claim: <one falsifiable sentence>
- Mechanism/theory sketch: <2-4 sentences — what would actually be proven or shown>
- Validation plan within budget: <concrete experiment(s) that fit the CPU/GPU-hour budget above, including what real-model check would confirm the theory>
- Nearest prior work you're aware of: <name papers/authors if you can, even tentatively — this matters more than breadth of brainstorming>
- Self-rated novelty confidence: <low/medium/high — how sure are you this isn't already published, given your own training knowledge>
- Self-rated ICLR acceptance odds if perfectly executed: <rough %>
- Compute fit: <does this need the GPU-hour budget, or is it CPU-only>
```

Two things matter more than generating a long list:
1. If you recognize that any of the four "alive" ideas above, or the diffusion-LM gap, is already substantially published — even a preprint you're not 100% sure of — say so explicitly and name it. Killing a bad idea fast is as valuable as finding a good one.
2. Do not propose anything resembling the killed list above, and do not propose generic LLM-agent, RAG, prompting, or "we fine-tuned a bigger model" papers — these require compute and baseline comparisons this project cannot afford and draw the harshest scrutiny at ICLR.
