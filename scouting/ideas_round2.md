# Scouting round 2: niche selection and 15 candidate gaps

## 1. Niche selection (data: all 19,814 ICLR 2026 submissions, Paper Copilot dump)
| Primary area | n | accepted | acc. rate (all) | acc. rate (decided) |
|---|---|---|---|---|
| **learning theory** | 516 | 190 | **36.8%** | **46.3%** |
| foundation/frontier models (LLMs) | 2646 | 831 | 31.4% | 43.7% |
| probabilistic methods | 380 | 116 | 30.5% | 37.9% |
| ... (21 areas; lowest: transfer/meta-learning 20.7%) | | | | |

Inside learning theory, keyword-level acceptance: power-law spectra 5/6, test-time scaling 14/25 (56%), scaling law 11/20 (55%), linear regression 13/24 (54%), high-dim/random-matrix 24/46 (52%); generalization bounds only 7/37 (19%).
Across all areas, keywords: "scaling laws" 47%, "training dynamics" 44%, "model editing" 54%.

**Chosen niche: solvable-model theory of modern training practices** (power-law linear / random-feature models + SGD, with closed-form scaling laws, validated on small real networks). The accepted template from ICLR 2026: *Larger Datasets Can Be Repeated More*, *Why Less is More: Theory of Data Curation* (7.5), *Scaling Laws of SignSGD in Linear Regression*, *Learning under Quantization*, *Fast Catch-Up, Late Switching*, *Theory of Scaling Laws for In-Context Regression*. The pattern: take one widely used practice, derive an exact scaling law for it, and turn that into a concrete prescription. The work is fully CPU-feasible.

Already covered (killed during this round): LR schedules for random-feature models (2602.04774); data-quality scheduling (2605.25698); data-mixing laws (2606.08167); synthetic data (2609.09572); weight averaging / anytime schedules (2602.03702); KD / weak-to-strong in high-dim regression (2606.01292); LoRA vs full fine-tuning in linear regression (2605.19018); DiLoCo in high-dim regression (2603.26954); sequential-editing norm collapse (2602.02543); catastrophic overtraining (2604.13627); Muon in associative memory (2602.05725).

## 2. Candidates (screen = arXiv API + full ICLR-2026 submission corpus; NOT yet full gate checks)
| # | Working title | Core claim | Closest prior → delta | Screen |
|---|---|---|---|---|
| 1 | **A solvable theory of transfer scaling laws** | Derive Hernandez et al.'s "effective data transferred" law D_T ∝ D_F^α N^β in power-law regression with pretrain→finetune target shift; α, β follow from spectrum + task alignment; predicts when pretraining *hurts* (ossification) | 2102.01293, 2408.16947 (empirical only); 2605.19018 (LoRA, no scaling law) | OPEN |
| 2 | **Continual pretraining, solved: re-warming, replay and the critical mixture ratio** | Closed-form forgetting/adaptation trade-off for SGD on two power-law tasks; derives the CMR law and the optimal replay fraction vs tokens; explains the "stability gap" from LR re-warming | 2407.17467 CMR law (empirical); 2403.08763 (empirical); 2601.13844 (L2 continual regression, no replay/LR) | OPEN |
| 3 | **Why poisoning needs a near-constant number of samples** | One-pass SGD on power-law data: attack success depends on poison *count* (not fraction) when the trigger is spectrally rare; phase diagram showing when the count-law breaks (LR-decay placement, trigger–clean overlap, WD) | 2510.07192 (empirical 250-doc finding); 2605.22481, 2505.15175 (proportional-regime theory, fraction-based) | OPEN-ish |
| 4 | **When do small proxy runs mis-rank data recipes?** | In power-law models, recipes differ in how they allocate signal across the spectrum → provable rank reversals across scale; gives a corrected extrapolation protocol | 2512.24503 (empirical, HP mismatch); DataDecide 2504.11393 | OPEN-ish |
| 5 | Context-length curricula in a solvable in-context regression model | Optimal short→long context schedule and its compute savings, in the linear-attention scaling framework | ICLR26 "Theory of Scaling Laws for In-Context Regression" (fixed context); 2108.06084 (empirical) | OPEN |
| 6 | Paraphrase vs repetition: the value of rephrased tokens | Multi-epoch solvable model where paraphrases = shared signal + fresh noise; derives the token-effectiveness η | ICLR26 multi-epoch linear regression; 2607.25271 (phenomenological η) | PARTIAL |
| 7 | Staleness–LR scaling for asynchronous RL, solved | Policy-gradient in a solvable bandit/linear-softmax model; derives the staleness-LR trade-off | 2607.01083 (empirical); 2510.01161 | PARTIAL-OPEN |
| 8 | The weight-decay timescale must scale with data: a solvable derivation | Optimal EMA timescale 1/(ηλ) vs dataset size in power-law SGD, including schedule interaction | 2607.23777, 2608.24814 (empirical/stationarity); ICLR26 rejected EMA paper | PARTIAL |
| 9 | Multi-token prediction in a solvable sequence model | When auxiliary future-token losses change sample efficiency / scaling exponents | 2604.11912 (star-graph planning) | PARTIAL |
| 10 | Clipping and heavy-tailed gradient noise change scaling exponents | Clipped SGD on power-law regression with heavy-tailed noise: exponent shifts and optimal clip threshold | convex clipping theory (ICLR26 posters) | OPEN, lower relevance |
| 11 | Edit-capacity scaling law for locate-then-edit | Number of sequential edits before failure vs width and key-covariance spectrum; validated on GPT-2 family (CPU) | 2602.02543 Norm Anchors (mechanism, no scaling law); HoReN 2605.08143 | PARTIAL |
| 12 | When should data appear? Placement and recency in a solvable model | Retention of an item as a function of its position under WSD/cosine schedules; explains training-order recency | ICLR26 "Fresh in memory" (empirical); 2605.25698 (quality scheduling) | PARTIAL |
| 13 | Compute-optimal local-update count H for DiLoCo | Scaling exponents of optimal H and outer LR under power-law spectra | 2603.26954 covers the core model | RISKY |
| 14 | Optimal model-growth schedules | Width/depth growth mid-training in a random-feature model | ICLR26 withdrawn "What is an Optimal Growth Schedule for LLMs? A Theoretical Study" | RISKY |
| 15 | Beyond-prior generalization of prior-fitted tabular models | Consistency/misspecification rates for PFNs (TabPFN/TabICL) with CPU experiments | 2603.12037, Nagler 2023 | PARTIAL |

## 3. Recommendation for gate checks
Top 4 to gate-check in depth: **#1 (transfer law), #3 (poison count law), #2 (continual pretraining), #4 (proxy rank reversal)**. Each (a) sits in the highest-acceptance cluster, (b) explains a well-known *empirical* scaling law that has no theory yet, (c) comes with a practical prescription, (d) is cheap to simulate, with small-transformer validation fitting the $8 budget.
