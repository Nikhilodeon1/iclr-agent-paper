# Rebuttal gameplan

Goal: maximise the chance this paper is accepted, using only what is true. Everything below
is ordered by value per hour, with a rule for what to do with each outcome *before* the
number is known, so the write-up cannot drift toward whatever came back.

## 0. Gates (no compute; do first)

1. **Make the AI disclosure match what happened.** The statement in `paper.tex` says an AI
   agent produced the idea, derivations, code and text, with a human running compute. Any
   other version (a form field, a PDF) that says "proofreading/autocomplete only" is
   inaccurate and must be corrected. A reviewer has already noticed the mismatch, and an
   unresolved disclosure problem outweighs every technical improvement below. The rebuttal
   must not contradict the statement.
2. **Paste the original reviews into the chat** when the window opens. The rebuttal draft
   is organised by issue from my own consolidation; I need the reviewers' wording to map
   each response to a person and a quote.
3. **Tell me the rebuttal deadline.** The plan below assumes 48-72 h. If it is under 24 h,
   run the minimal path in section 2.

## 1. What each reviewer said would change their mind

The most useful reviewer gave a checklist for moving from 4 to 6. Map it to our work:

| Reviewer ask | Our response | Status |
|---|---|---|
| More seeds | Job E (20 seeds, single protocol, seeded init) | needs pod |
| Direct mechanistic measurement | Job H (trigger-embedding displacement) | script ready, tested |
| Vary beta2 | Job A (beta2 x spacing) | script ready, tested |
| Finish Pythia | Job G (4 levels, 5 seeds) | needs pod |
| Treat count regime as finite-horizon | "Horizon" paragraph in Discussion | **done in paper.tex** |

Three reviewers rated 3 (confidence 4). Their objections were mostly about the
*submitted* version and are now substantially fixed (one consistent sparse model, the
7.2/7.3 contradiction disclosed, statistics corrected, claims softened). That is
necessary, not sufficient: a reviewer who has scored 3 usually moves one point, not three,
and only if the response shows engagement with *their* specific objection.

## 2. Compute plan

Jobs are in `pod_commands.md`. Pod-hours: A 2.0, B 2.7, H 0.6, E 2.7, G 2.0, C 2.7, D 1.9,
F 3.2. On four pods in parallel the first wave is about 3 h wall clock.

**Wave 1 (run together):** A, B, H, E. Add G if a fifth pod is available.
**Wave 2 (only if time):** C, D, F, G2.
**Minimal path (deadline under 24 h):** A + B + H on three pods, about 3 h. Skip E; say
plainly in the rebuttal that the seed top-up is not done.

Run `git pull` first on every pod. Run the smoke test inside each command block first
where one is given (Job G especially).

## 3. Decision rules, fixed in advance

| Job | If it supports the theory | If it does not |
|---|---|---|
| A (beta2 x spacing) | ASR lower in interfering cells at fixed spacing. Lead the rebuttal with it: it is the one prediction frequency counting cannot make. | Flat across beta2. **Downgrade Proposition 3 to "an analysis of idealised Adam, not shown to drive attack success"** in the abstract, contributions and discussion. Say so in the rebuttal. Do not defend it. |
| B (param-group) | Effect tracks the trigger row's optimizer, not the network's. Also check the unsplit control agrees with adamw/trigadamw first. | Tracks the network's optimizer, or the control disagrees: the two-channel account is wrong or the split confounds it. Remove the two-channel claim. |
| H (displacement) | AdamW flat in k and near `pred_disp`; SGD ~ linear in k. Report as confirming the mechanism's premise only. | Large departures from `pred_disp`: investigate epsilon and bias correction before reporting. |
| E (20 seeds) | If AdamW 1-vs-16 batches separates with Wilson intervals, state it as established. | If intervals still overlap, say the effect is not established and remove it from the abstract. |
| G (Pythia) | If the fourth level continues the sublinear trend, report it. If it levels off, report that. | Either way, report all four levels. Do not drop the one that fits worst. |

## 4. Where results land in `paper.tex`

- A, H -> new subsection after Section 7.3; abstract's optimizer sentence; contribution 4.
- B -> Section 7.3 and Proposition 3's "two channels" paragraph.
- E -> the Wilson-interval paragraph and Figure 4 caption; replace "5 seeds" throughout.
- G -> `realtext_section.tex` and the realtext appendix; abstract sentence on Pythia.
- C -> the sentence after Equation (budget) that currently says "untested".
- After each: re-run the structural check (cites, refs, labels, environments), update
  `rebuttal_draft.md`, and only then touch Overleaf.

## 5. Rebuttal structure and tone

- **Global response, one page.** Open with what was wrong and fixed; list the five changes;
  state what is still pending or unresolved. No defensiveness.
- **Per reviewer.** Lead with the concession where it is true. Quote their sentence. Give
  the change and its location (section, equation, table) so they can verify it in the
  revised PDF. Mark anything pending as pending.
- **Do not argue with a score.** Ask the reviewer which remaining result would change it,
  and point to the job that supplies it.
- **Do not claim:** scale invariance, that placement or seeding explains the 7.2/7.3 gap,
  that Pythia validates the dynamics rather than the ordering, or that the long-horizon
  decline is explained.
- **Say the weak points before they do:** scalar reduction; the high-dimensional check
  uses the dense not the sparse model; small models; sublinear growth at f=0.2.

## 6. Honest odds (my judgement, not computed)

ICLR 2026 learning-theory papers: average rating 4.5-5.0 was accepted 45% of the time,
5.0-5.5 77%, 4.0-4.5 23%. The decision boundary is steep around 5.

| Scenario | Likely outcome |
|---|---|
| No new experiments, rebuttal only | Mostly 3-4s, roughly 10-15% |
| A, H, E, G complete and support the theory; B supportive or neutral | Moves several reviewers to 4-5; roughly 30-40% |
| A or B comes back flat, reported honestly with Prop 3 downgraded | A narrower, sounder paper; roughly 15-25% |
| Disclosure problem left unresolved | Cap on everything above |

The ceiling is set by the scalar theory, the small models and the missing model-size axis.
No experiment on this list changes those.

## 7. Open risks

- **Page count is unverified.** The local TeX install is broken; compile on Overleaf from
  `iclr2027/` and check the 10-page main-text limit before any upload.
- **The PDF in the repo is stale** and predates the sparse-model rewrite.
- **Only the AdamW arm** of the full-size displacement check has run (k=1: 97%, k=16: 99.6% of
  the prediction, 2 seeds). It is not in the paper until Job H supplies the SGD contrast.
- **Job G2 is untested** (Pile loading); run its smoke test first or skip it.
