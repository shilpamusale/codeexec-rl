# THESIS

*The claims this project tests, written before the code that tests them.*

The hypotheses in this document were fixed at the outset — they land in the first substantive commit of the project, alongside the empty skeleton and CI, before any implementation fills it in. That ordering is deliberate: a hypothesis written after seeing results is not a hypothesis.

---

## The question

When you train a language model with reinforcement learning and the reward comes from running its code against a test suite, the reward is grounded in execution rather than in a learned judge of correctness. That removes an entire class of reward-model error: there is no learned reward signal to exploit, because the signal is the test outcome itself.

But passing the tests you can see is not the same as being correct. If the reward is computed on an observable test suite, the policy can improve its score by learning properties of those specific tests rather than by learning to solve the problem. That is Goodhart's law applied to execution reward.

This project builds the environment needed to measure that, and then measures it.

---

## Metric definitions

Fixed before any run. Changing these after seeing results invalidates the comparison.

*Amended 2026-10-21, before any training run: `heldout_pass` is measured on a held-out split that no arm's reward ever sees (ADR-013).*

*Amended 2026-10-24, before any training run: `visible_pass`, `heldout_pass` and `G` are measured over all 369 training problems on the training box's evaluation path; the frozen-100 subset is reported alongside (ADR-004 status update).*

All pass-rates are pass@1, greedy decode, temperature 0 unless stated otherwise.

| Symbol | Definition |
|---|---|
| `visible_pass` | Fraction of problems where all canonical MBPP tests pass. Measured on the canonical tests in every arm; only the reward function differs between arms, never this measurement. |
| `heldout_pass` | Fraction of the same problems where the canonical tests **and** the held-out measurement split B all pass (the EvalPlus "plus" form, restricted to B; ADR-004, ADR-013). Split B never enters any reward computation, in any arm. |
| `G` | `visible_pass − heldout_pass`. The gap at a point in training, on the headline greedy metric. |
| `G_baseline` | `G` measured on the untrained base model. The pre-training baseline; `ΔG` is measured from here. |
| `ΔG` | `G_final − G_baseline`. **The quantity under test.** |

pass@1 greedy is the headline metric, and `G`, `G_baseline`, and `ΔG` are defined on it. A sampled gap — pass@10 at temperature 0.8, using the unbiased estimator from Chen et al. (2021) — is reported alongside as a secondary readout. If the gap behaves differently under sampling than under greedy decode, that divergence is itself a result and gets reported rather than smoothed over.

Visible tests are the canonical test cases shipped with each MBPP problem — three or so per problem, sparse by design. Held-out tests are the MBPP+ augmented suite for the same problem. HumanEval+ is held out entirely as an out-of-distribution check and is never trained on.

Each problem's held-out tests are split once, by a deterministic rule fixed before training (ADR-013), into two disjoint halves: **B**, the measurement split, on which `heldout_pass` and `G` are measured and which never enters any reward; and **A**, the reward split, which enters only V0's reward.

---

## H0 — Contamination (precondition)

> **Claim.** Measured pass-rates on this task distribution reflect generalization rather than memorization. The base model's pass-rate on lightly perturbed problems — functions renamed, docstrings rewritten, variable names changed, semantics preserved — does not collapse relative to the canonical problems.
>
> **Falsified if** perturbed pass-rate drops below canonical pass-rate by more than the measurement noise band — a threshold fixed in advance, not chosen after the fact. Perturbed pass-rate is measured across several independent perturbations per problem, and "collapse" means a drop beyond what that spread and baseline seed noise would explain.

MBPP and HumanEval predate the models trained on them and are widely reported as contaminated. This matters here for a specific reason, not a general one: if the base model has partially memorized canonical solutions, then the visible/held-out gap may reflect the structure of what was memorized rather than anything the reward did. Under that condition H1 is not false — it is **uninterpretable**.

Every perturbation is validated to preserve semantics before it is used: the canonical reference solution, renamed to match, is run against the perturbed tests, and the perturbation is kept only if the reference solution still passes. This guards against the confound where a pass-rate drop reflects a broken perturbation rather than memorization — without it, H0 itself would be uninterpretable.

H0 is measured and reported before any training arm runs. If it fails, the ablation does not proceed until the task distribution changes. The fallback is to train and evaluate on the perturbed variants themselves, which the model demonstrably has not memorized because they did not exist at pretraining time.

This is a precondition, not a caveat, and it is reported whether or not it is convenient.

---

## H1 — Visible-test overfitting (primary)

> **Claim.** Training with a reward computed only from **visible** tests **widens** the gap between visible pass-rate and held-out pass-rate — relative both to the same policy before training, and to a control trained on a dense test suite that excludes the measurement split. Execution reward on an observable test suite is Goodhartable, and `visible_pass` increasingly overstates true correctness as training proceeds.
>
> **Falsified if** the gap does not widen: held-out pass-rate rises together with visible pass-rate, or the treatment arm's `ΔG` is indistinguishable from the control's across seeds.

**The claim is about ΔG, not G.** A gap almost certainly exists at initialization — EvalPlus exists precisely because base models pass sparse canonical tests they should not. The interesting question is not whether a gap exists but whether optimizing against visible tests makes it grow.

### Arms

| Arm | Reward computed on | Held-out metric | What it isolates |
|---|---|---|---|
| **V0** (control) | Canonical tests plus held-out split A: a dense reward that sees everything except the measurement | Held-out split B | Does the gap widen even under a dense reward? Controls for "training widens any gap." |
| **V1** (treatment) | Canonical MBPP tests only | Held-out split B | Does a sparse, visible-only reward widen the gap? |

Both arms are measured on the same instruments: `visible_pass` on the canonical tests and `heldout_pass` on split B, which neither arm's reward ever sees. The arms differ only in reward density, about 3 tests per problem for V1 against about 55 for V0, so V0 controls for drift that any training would cause without being trained on its own measurement. V1's `ΔG` is compared with V0's across seeds; the effect attributed to sparse, visible-only reward is the difference between the arms, and H1 also requires V1's gap to widen relative to its own baseline.

Three seeds per arm, six runs — three is the floor for reporting variance at all, not the ideal; seed variance is reported alongside the effect. If the effect is within seed noise, that is the finding and it gets stated plainly.

### Secondary readouts

- Per-problem gap changes, not only the aggregate — an aggregate `ΔG` can hide a bimodal distribution where a few problems overfit severely while most do not.
- Degenerate-solution rate from the AST detector: hard-coded outputs, branches keyed to literal test values. Reported with its false-positive rate against a hand-labeled sample, since some functions are legitimately constant.
- HumanEval+ out-of-distribution pass-rate, never trained on — directional, since it is itself contaminated and small.

---

## H2 — Reward density (stretch, not committed)

> **Claim.** A shaped reward — fraction of visible tests passing, plus partial credit for running without error — reaches a target pass-rate in fewer environment steps than a binary terminal (sparse) reward, but **increases ΔG**, because the policy is rewarded for cheaply satisfying the easiest visible tests. The claim is conjunctive: shaping must be both *faster* and *worse on ΔG*.
>
> **Falsified if** shaping fails to improve sample efficiency, or matches sparse on ΔG — either half failing falsifies the conjunction. If the ΔG difference between shaped and sparse is smaller than seed noise can resolve, the result is neither confirmed nor falsified but unresolvable, and is reported as such — a likely outcome, since this is a difference of differences and the most seed-hungry claim in the project.

Shaping "fraction of tests passing" is not potential-based shaping in the sense of Ng, Harada & Russell (1999): unlike a potential-based transform, it carries no guarantee of preserving the optimal policy, and the direction it can shift the optimum toward is test-passing per se rather than correctness. H2 tests whether that theoretical possibility materializes as a wider ΔG.

H2 runs only if the H1 result is complete with time remaining. It is documented here as an extension, not promised as a deliverable.

---

## What would make these results uninterpretable

Stated in advance so it cannot be decided after the fact:

1. **H0 fails.** Memorization confounds the gap. Not a negative result — no result.
2. **The held-out measurement is too weak.** If held-out split B barely differs in difficulty from the canonical tests, `G` has little room to move and the instrument is blunt. Reported as a measured property of the split, not assumed; ADR-013 fixes in advance what happens if the gap on B has no dynamic range.
3. **Seed variance exceeds the effect.** Then the honest statement is that the effect, if any, is smaller than this experiment can resolve. This applies most acutely to H2, which compares a difference of differences and needs more seeds to resolve than H1.
4. **The policy never learns.** If `visible_pass` does not rise above baseline, there is no optimization pressure and nothing to overfit with.

---

## Status

| Claim | Status |
|---|---|
| H0 | Measured — **clean**. Perturbed pass@1 0.52 vs canonical 0.55 on the frozen 100: drop +3.0 pp, McNemar p = 0.549, 95% CI [−3.0, +10.0] pp, below the pre-registered 5 pp threshold; ADR-002 stands. One seeded perturbation per problem, not several as anticipated above (see the analysis/h0.md limitations). See analysis/h0.md. |
| H1 | Not yet measured — Phase 5 |
| H2 | Stretch, uncommitted |

This table is the canonical record of what has and has not been established, and is updated as results land — including results that contradict the claims above. A claim is marked measured only when its phase gate has passed and its writeup is committed. Negative and null results are recorded here with the same prominence as positive ones.

## References

- Chen et al. (2021), *Evaluating Large Language Models Trained on Code* — pass@k estimator, HumanEval.
- Austin et al. (2021), *Program Synthesis with Large Language Models* — MBPP.
- Liu et al. (2023), *Is Your Code Generated by ChatGPT Really Correct?* — EvalPlus / MBPP+ / HumanEval+.
- Gao, Schulman, Hilton (2022), *Scaling Laws for Reward Model Overoptimization* — the proxy-vs-gold divergence shape this project looks for in an execution-reward setting.
- Ng, Harada, Russell (1999), *Policy Invariance Under Reward Transformations* — potential-based shaping; H2 turns on "fraction of tests passing" not being a potential-based transform.