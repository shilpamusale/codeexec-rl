# Task-layer data format

*What EvalPlus provides, how the loaders normalize it, and the counts observed.*

The task layer loads three logical datasets from EvalPlus (`evalplus==0.3.1`):
MBPP and MBPP+ (bundled — the same problems with two test sets), and HumanEval+.
The loaders turn EvalPlus's two different record formats into one uniform
`Problem` type (see `problem.py`), so the rest of the pipeline is format-agnostic.

## The datasets

| Dataset | Role | Problems | Loader |
|---|---|---|---|
| MBPP (canonical) | training problems + **visible** tests | 378 | `load_mbpp()` |
| MBPP+ | **held-out** tests on the same problems (bundled with MBPP) | 378 | `load_mbpp()` |
| HumanEval+ | out-of-distribution check, never trained on | 164 | `load_humaneval()` |

Counts are for the pinned version: `evalplus==0.3.1`, which fetches MBPP+ dataset
`v0.2.0` and HumanEval+ dataset `v0.1.10`. MBPP+ is a curated subset of the
original ~1000 MBPP problems — EvalPlus removed tasks whose original tests were
wrong, landing at 378. Both datasets loaded fully (378/378 and 164/164) with no
problems dropped by validation.

## A raw EvalPlus record

Each problem is a dict. The fields the loaders use:

| Field | What it is |
|---|---|
| `task_id` | unique id, e.g. `Mbpp/2`, `HumanEval/0` |
| `prompt` | what the model sees (see format difference below) |
| `entry_point` | the function name the solution must define |
| `canonical_solution` | the reference solution (see format difference below) |
| `base_input` | the **canonical** test inputs → **visible** tests |
| `plus_input` | the **augmented** test inputs → **held-out** tests |
| `atol` | float comparison tolerance (0 = exact) |

(HumanEval+ also carries a legacy `test` field — a full `check(candidate)`
harness — and both carry a `contract` field of input-validation asserts. The
loaders use `base_input`/`plus_input`, not these legacy fields, because the
inputs-only representation is consistent across both datasets and cleanly gives
the visible/held-out split.)

## The grading model: inputs + an answer key, not finished checks

The crucial subtlety: `base_input` and `plus_input` contain **only the inputs**
(the argument sets to call the function with) — **not** the expected outputs.
EvalPlus computes expected outputs by running each input through the
`canonical_solution` (the reference solution is the answer key). So a "test" is
really two ingredients: an *input* plus the *canonical solution* to compute the
expected output from.

The loaders do this pre-computation once, at load time (the `TestCase.expected`
field), so a `Problem` is self-contained — it can be graded without re-running
the canonical solution. A test whose canonical execution raises is dropped and
logged; a problem whose canonical will not load, or that loses all visible
tests, is dropped and logged.

## The two prompt/solution formats

MBPP and HumanEval inherit their original benchmarks' conventions, which differ:

| | MBPP+ | HumanEval+ |
|---|---|---|
| `prompt` | plain instruction + one `assert` example, in a docstring | a function **signature + docstring** with `>>>` examples (a completion) |
| `canonical_solution` | a **complete** `def` | typically **only the function body** |
| complete runnable solution | the `canonical_solution` as-is | `prompt + canonical_solution` concatenated |

The loaders normalize the *solution* (HumanEval's body is concatenated onto the
prompt to form a complete, runnable function) but keep the *prompt* in its native
format, because the prompt's format is what the model must work with — MBPP is an
instruction to follow, HumanEval is a function to complete.

## Observed shapes (sanity reference)

- `Mbpp/2` (`similar_elements`): 3 visible tests, 108 held-out. First visible
  test: `args=((3,4,5,6),(5,7,4,10))`, `expected=(4,5)`.
- `HumanEval/0` (`has_close_elements`): 7 visible tests, 999 held-out. First
  visible test: `args=([1.0,2.0,3.9,4.0,5.0,2.2], 0.3)`, `expected=True`.

The large held-out:visible ratios (≈36× for MBPP, ≈140× here) are EvalPlus's
test augmentation — the richer held-out suites are what make the visible/held-out
gap measurable.
