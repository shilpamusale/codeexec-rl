# G_baseline — the generalization gap at the base model

**Result: G_baseline = +10.0 pp, CI [+5.0, +16.0] (excludes zero).** The held-out
suite is meaningfully harder than the visible tests for this model, so the
instrument has the dynamic range H1 needs. ADR-004 confirmed; proceed to Phase 4.

## The quantity

H1 asks whether training on visible-test reward makes the model overfit the
visible tests at the expense of held-out generalization. The measured gap is:

    G = visible_pass@1 - plus_pass@1

where `plus_pass` is the standard MBPP+ "solved" definition (passes the visible
*and* the held-out tests), per ADR-004. So G is literally **the fraction of
problems that pass the sparse visible tests but fail the stronger held-out
suite** — the pre-existing overfitting-shaped gap, before any training.

H1 tracks ΔG = G_final − G_baseline. `G_baseline` is G at the untrained base
model: the zero from which training-induced widening (or narrowing) is measured.

## Method

- **Sample.** The same frozen 100-problem MBPP+ sample (seed 0, 369-pool) used for
  H0 — so H0, the visible baseline, and G_baseline all describe one coherent set.
- **Model / decoding.** Qwen2.5-Coder-1.5B base, zero-shot-plus-stub prompt,
  greedy (temperature 0). One generation per problem (base model, canonical
  prompt — no perturbation).
- **Grading.** Each completion graded twice against the same `build_harness`:
  once on `visible_tests` only (visible_pass), once on `visible + heldout`
  (plus_pass). By construction plus_pass implies visible_pass.
- **Statistics.** Bootstrap 95% CI on the paired per-problem difference.

## Result

```
visible pass@1:  0.650 (65/100)
plus pass@1:     0.550 (55/100)
G_baseline:      +10.0 pp
bootstrap 95% CI: [+5.0, +16.0] pp
malformed rate:  0.020
```

- 10 of 100 problems pass the visible tests but fail the held-out suite at the
  base model; 0 pass held-out while failing visible (as required).
- The CI excludes zero: the gap is statistically distinguishable from no gap.

## ADR-004 dynamic-range check

ADR-004's reversal condition is "MBPP+ barely harder than canonical for this model
→ small, static G_baseline → instrument lacks dynamic range." That is **not**
triggered: G_baseline is 10 pp with a CI clear of zero. The held-out suite can
detect overfitting, and ΔG has ~10 pp of room to move within. The instrument is
validated; training can proceed.

## Note on earlier pass-rates

The H0 / smoke-test "base pass@1" figures (0.57 pre-exclusion, 0.55 on the 369
pool) were graded on the full suite (visible + held-out) — i.e. they are
*plus* pass-rates, not visible-only. This measurement separates the two for the
first time: visible-only is 0.65, plus is 0.55. The H0 drop comparison is
unaffected (both arms were graded identically); only the label is clarified.

## Reproduce

```
# Colab: smoke_test_base_model.ipynb cells FA-FC -> analysis/g_baseline_results.json
```

Raw per-problem results: `analysis/g_baseline_results.json`.
