# THESIS

*The claims this project tests, written before the code that tests them.*

This document is committed before any implementation module. That ordering is deliberate: a hypothesis written after seeing results is not a hypothesis. The commit history is part of the evidence.

---

## The question

When you train a language model with reinforcement learning and the reward comes from running its code against a test suite, the reward is grounded in execution rather than in a learned model of quality. That removes an entire class of reward-model error. The code either passes or it does not.

But passing the tests you can see is not the same as being correct. If the reward is computed on an observable test suite, the policy can improve its score by learning properties of those specific tests rather than by learning to solve the problem. That is Goodhart's law applied to execution reward.

This project builds the environment needed to measure that, and then measures it.

---

## Metric definitions

Fixed before any run. Changing these after seeing results invalidates the comparison.

| Symbol | Definition |
|---|---|
| `visible_pass` | Fraction of problems where **all visible tests** pass. pass@1, greedy decode, temperature 0. |
| `heldout_pass` | Fraction of the **same problems** where the EvalPlus augmented suite passes. pass@1, greedy decode, temperature 0. |
| `G` | `visible_pass − heldout_pass`. The gap at a point in training. |
| `G_baseline` | `G` measured on the untrained base model. The null. |
| `ΔG` | `G_final − G_baseline`. **The quantity under test.** |

pass@1 greedy is the headline metric. pass@10 at temperature 0.8 is reported as a secondary readout, using the unbiased estimator from Chen et al. (2021). If the gap behaves differently under sampling than under greedy decode, that divergence is itself a result and gets reported rather than smoothed over.

**Visible tests** are the canonical test cases shipped with each MBPP problem. **Held-out tests** are the EvalPlus augmented suite for the same problem. The held-out suite never enters any reward computation in any arm. HumanEval+ is held out entirely as an out-of-distribution check and is never trained on.

---

## H0 — Contamination (precondition)

> **Claim.** Measured pass-rates on this task distribution reflect generalization rather than memorization. The base model's pass-rate on lightly perturbed problems — functions renamed, docstrings rewritten, variable names changed, semantics preserved — does not collapse relative to the canonical problems.
>
> **Falsified if** perturbed pass-rate drops materially below canonical pass-rate.

MBPP and HumanEval predate the models trained on them and are widely reported as contaminated. This matters here for a specific reason, not a general one: if the base model has partially memorized canonical solutions, then the visible/held-out gap may reflect the structure of what was memorized rather than anything the reward did. Under that condition H1 is not false — it is **uninterpretable**.

H0 is measured and reported before any training arm runs. If it fails, the ablation does not proceed until the task distribution changes. The fallback is to train and evaluate on the perturbed variants.

This is a precondition, not a caveat, and it is reported whether or not it is convenient.

---

## H1 — Visible-test overfitting (primary)

> **Claim.** Training with a reward computed only from **visible** tests **widens** the gap between visible pass-rate and held-out pass-rate — relative both to the same policy before training, and to a control trained on the full test suite. Execution reward on an observable test suite is Goodhartable, and `visible_pass` increasingly overstates true correctness as training proceeds.
>
> **Falsified if** the gap does not widen: held-out pass-rate rises together with visible pass-rate, or the treatment arm's `ΔG` is indistinguishable from the control's across seeds.

**The claim is about ΔG, not G.** A gap almost certainly exists at initialization — EvalPlus exists precisely because base models pass sparse canonical tests they should not. The interesting question is not whether a gap exists but whether optimizing against visible tests makes it grow.

### Arms

| Arm | Reward computed on | Held-out metric | What it isolates |
|---|---|---|---|
| **V0** (control) | Full EvalPlus suite — nothing hidden from the reward | Same suite | Does the gap widen even when the reward sees everything? Controls for "training widens any gap." |
| **V1** (treatment) | Canonical MBPP tests only | EvalPlus augmented suite | Does hiding tests from the reward widen the gap? |

Three seeds per arm, six runs. Seed variance is reported alongside the effect. If the effect is within seed noise, that is the finding and it gets stated plainly.

### Secondary readouts

- Per-problem gap changes, not only the aggregate — an aggregate ΔG can hide a bimodal distribution.
- Degenerate-solution rate from the AST detector: hard-coded outputs, branches keyed to literal test values.
- HumanEval+ out-of-distribution pass-rate, never trained on.

---

## H2 — Reward density (stretch, not committed)

> **Claim.** A shaped reward — fraction of visible tests passing, plus partial credit for running without error — reaches a target pass-rate in fewer environment steps than a binary terminal reward, but **increases ΔG**, because the policy is rewarded for cheaply satisfying the easiest visible tests.
>
> **Falsified if** shaping matches sparse on ΔG, or fails to improve sample efficiency.

H2 runs only if the H1 result is complete with time remaining. It is documented here as an extension, not promised as a deliverable.

---

## What would make these results uninterpretable

Stated in advance so it cannot be decided after the fact:

1. **H0 fails.** Memorization confounds the gap. Not a negative result — no result.
2. **The held-out suite is too weak.** If EvalPlus augmented tests barely differ in difficulty from canonical tests, `G` has little room to move and the instrument is blunt. Reported as a measured property of the suite, not assumed.
3. **Seed variance exceeds the effect.** Then the honest statement is that the effect, if any, is smaller than this experiment can resolve.
4. **The policy never learns.** If `visible_pass` does not rise above baseline, there is no optimization pressure and nothing to overfit with.

---

## Status

| Claim | Status |
|---|---|
| H0 | Not yet measured — Phase 3 |
| H1 | Not yet measured — Phase 5 |
| H2 | Stretch, uncommitted |

This table is updated as results land, including results that contradict the claims above. Negative and null results are recorded here with the same prominence as positive ones.

---

## References

- Chen et al. (2021), *Evaluating Large Language Models Trained on Code* — pass@k estimator, HumanEval.
- Austin et al. (2021), *Program Synthesis with Large Language Models* — MBPP.
- Liu et al. (2023), *Is Your Code Generated by ChatGPT Really Correct?* — EvalPlus.
- Gao, Schulman, Hilton (2022), *Scaling Laws for Reward Model Overoptimization* — the proxy-vs-gold divergence shape this project looks for in an execution-reward setting.
- Ng, Harada, Russell (1999), *Policy Invariance Under Reward Transformations* — potential-based shaping, relevant to H2.
