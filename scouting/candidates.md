# Candidate research directions (scouting log)

Constraints: CPU only (16 cores, ~3 GB free RAM), $8 burst budget. ICLR 2026: 19,525 submissions, 27.4% acceptance; hallucinated references -> desk reject.

| # | Idea | Verdict | Closest prior work (arXiv id) |
|---|------|---------|------------------------------|
| 1 | TSFM zero-shot leakage audit | KILLED | 2510.13654, TS-Arena 2512.20761, TIME 2602.12147 |
| 2 | Parallel-decoding error theory for masked diffusion LMs | KILLED | 2602.00286, 2510.25544, 2511.04647 |
| 3 | Watermarking diffusion LMs | KILLED | 2511.02083, 2601.12376, 2601.22985 |
| 4 | Unbiased maj@k estimation / identifiability | KILLED | 2605.05592, 2510.03199 |
| 5 | Solvable model of RLVR pass@k crossover | KILLED | 2601.03764, 2510.02230 |
| 6 | Subliminal learning theory | KILLED | 2609.23260 and >20 others (2026) |
| 7 | Emergent misalignment | KILLED | ~90 papers |
| 8 | PFN/TabPFN martingale coherence | KILLED | 2505.11325, 2510.25154 |
| 9 | Why diffusion LMs beat AR under data repetition (mechanism) | KILLED | 2510.04071, 2511.03276, 2606.06888 |
| 10 | Kalai-style hallucination bound test | WEAK | 2502.08666 |
| 11 | LLM-judge reliability / kappa paradox | WEAK (low excitement) | 2606.19544 |
| 12 | 45 LLM-generated ideas (optimization, GNN, TS, RL, SSL, ...) | mostly known/low impact | see scout/gen_ideas.json |
| **13** | **Train-test duplicate rows in tabular benchmarks inflate in-context tabular foundation models (TabPFN/TabICL) relative to GBDTs** | **LEAD** | no direct hits (arXiv: duplicates x TabPFN/tabular FM, 2026-09) |

Lead rationale: tabular FMs are a high-profile claim (TabPFN v2, Nature 2025); theory/audit density low; CPU-feasible; clear falsifiable claim; audit + mechanism + fix format.
Early evidence (OpenML-CC18, 63 datasets, official 10-fold splits): 17 datasets have >5% of test rows with an exact feature duplicate in train, 8 have >20%.
