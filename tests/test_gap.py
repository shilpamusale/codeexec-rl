"""Tests for the H1 measurement (THESIS, ADR-004, ADR-013).

Unit tests grade with a stub that decides pass or fail from the code and
the tests it is given, so they run in CI. The last test re-measures the saved
Phase 3 completions in the real sandbox and must reproduce the committed split-B record,
except where ADR-015's pass rule changed the verdict.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import numpy as np
import pytest

from codeexec.eval.gap import (
    ProblemResult,
    disagreements,
    grade_problem,
    measure,
    summarize,
)
from codeexec.eval.grade import RunTestsFn
from codeexec.rewards.harness import TestCases, TestResult
from codeexec.tasks.problem import Problem
from codeexec.tasks.problem import TestCase as Case
from codeexec.tasks.split import split_heldout

ANALYSIS = Path(__file__).resolve().parents[1] / "analysis"
CORRECT = "        return a + b\n"
OVERFIT = "        return 101\n"
MALFORMED = "           return(a +\n"


def _problem(task_id: str = "Mbpp/3") -> Problem:
    """add(a,b). The visible test's args start with 100; heldout test i's with i."""
    return Problem(
        task_id=task_id,
        source="mbpp",
        prompt='"""Add two numbers."""',
        entry_point="add",
        canonical_solution="def add(a,b):\n     return a + b\n",
        visible_tests=(Case((100, 1), 101),),
        heldout_tests=tuple(Case((i, i), 2 * i) for i in range(10)),
        atol=0.0,
    )


SPLIT_A = {cast(int, t.args[0]) for t in split_heldout(_problem()).reward}


def _outcome(tests: TestCases, ok: bool) -> TestResult:
    total = len(tests)
    return TestResult(passed=total if ok else 0, total=total, timed_out=False, crashed=False)


def _stub(fails_on_a: bool = False) -> RunTestsFn:
    """A grader: "a + b" passes every test, "101" only the visible one.
    With fails_on_a, "a + b" fails any split-A test, so B and the full suite disagree."""

    def run(code: str, entry_point: str, tests: TestCases, atol: float) -> TestResult:
        ids = {cast(int, args[0]) for args, _ in tests}
        if "a + b" not in code:
            return _outcome(tests, ok=ids == {100})
        return _outcome(tests, ok=not (fails_on_a and ids & SPLIT_A))

    return run


def test_correct_passes_every_suite() -> None:
    result = grade_problem(CORRECT, _problem(), run=_stub())
    assert result == ProblemResult(malformed=False, visible_pass=True, plus_b=True, plus_full=True)


def test_overfit_passes_visible_only() -> None:
    result = grade_problem(OVERFIT, _problem(), run=_stub())
    assert result == ProblemResult(
        malformed=False, visible_pass=True, plus_b=False, plus_full=False
    )


def test_malformed_fails_everything() -> None:
    result = grade_problem(MALFORMED, _problem(), run=_stub())
    assert result == ProblemResult(
        malformed=True, visible_pass=False, plus_b=False, plus_full=False
    )


def test_split_b_and_the_full_suite_are_graded_separately() -> None:
    """A program that fails only split-A tests passes plus_b but not plus_full."""
    result = grade_problem(CORRECT, _problem(), run=_stub(fails_on_a=True))
    assert (result.plus_b, result.plus_full) == (True, False)


def test_results_keep_the_completions_order() -> None:
    problems = {tid: _problem(tid) for tid in ("Mbpp/9", "Mbpp/3", "Mbpp/5")}
    completions = {"Mbpp/9": CORRECT, "Mbpp/3": OVERFIT, "Mbpp/5": MALFORMED}
    results = measure(completions, problems, run=_stub(), max_workers=3)
    assert list(results) == ["Mbpp/9", "Mbpp/3", "Mbpp/5"]  # the bootstrap depends on order
    assert [r.plus_b for r in results.values()] == [True, False, False]


def test_completion_for_an_unknown_problem_is_an_error() -> None:
    with pytest.raises(KeyError, match="Mbpp/999"):
        measure({"Mbpp/999": CORRECT}, {"Mbpp/3": _problem()})


def test_summary_counts_and_gap() -> None:
    results = {
        "a": ProblemResult(False, True, True, True),
        "b": ProblemResult(False, True, False, False),  # the gap: visible but not plus
        "c": ProblemResult(True, False, False, False),  # malformed
        "d": ProblemResult(False, True, True, False),  # plus_b but not plus_full
    }
    s = summarize(results)
    assert (s.n, s.malformed, s.visible, s.plus_b, s.plus_full) == (4, 1, 3, 2, 1)
    assert (s.gap_pp, s.gap_full_pp) == (25.0, 50.0)
    assert s.ci_pp[0] <= s.gap_pp <= s.ci_pp[1]


def test_bootstrap_is_deterministic_and_zero_for_no_gap() -> None:
    results = {str(i): ProblemResult(False, True, True, True) for i in range(50)}
    s = summarize(results)
    assert (s.gap_pp, s.ci_pp) == (0.0, (0.0, 0.0))
    diffs = {str(i): ProblemResult(False, True, i % 3 == 0, True) for i in range(60)}
    assert summarize(diffs) == summarize(diffs)


def test_disagreements_compare_only_shared_problems() -> None:
    results = {
        "x": ProblemResult(False, True, True, True),
        "y": ProblemResult(False, True, False, False),
        "z": ProblemResult(False, False, False, False),
    }
    record = {
        "x": {"visible_pass": True, "plus_pass": True},
        "y": {"visible_pass": True, "plus_pass": True},  # differs
    }
    assert disagreements(results, record) == ["y"]


def test_bootstrap_matches_phase_3_on_its_own_record() -> None:
    """Fed the committed per-problem record, the CI is Phase 3's exactly."""
    record = json.loads((ANALYSIS / "g_baseline_split_b.json").read_text())
    per_problem = record["split B (1/2)"]["per_problem"]
    results = {
        tid: ProblemResult(False, r["visible_pass"], r["plus_pass"], r["plus_pass"])
        for tid, r in per_problem.items()
    }
    s = summarize(results)
    assert (s.visible, s.plus_b, round(s.gap_pp, 1)) == (65, 55, 10.0)
    assert tuple(round(x, 1) for x in s.ci_pp) == (5.0, 16.0)
    assert np.isclose(s.gap_pp, s.gap_full_pp)


# The committed record was graded under ADR-007's rule. ADR-015's (EvalPlus's)
# changes exactly one frozen-100 verdict: Mbpp/234's `l*l*l` differs from `l**3`
# in the last bit, so it failed split B and the full suite, and now passes both.
ADR_015_CHANGES = ["Mbpp/234"]


@pytest.mark.integration
@pytest.mark.dataset
def test_reproduce_the_split_b_rebaseline_from_saved_completions() -> None:
    """The instrument check: the Phase 3 completions, re-graded in the Docker
    sandbox by this module, give the committed record problem by problem,
    except where ADR-015's pass rule changed the verdict, from fail to pass.
    """
    from codeexec.tasks.datasets import training_problems

    completions = json.loads((ANALYSIS / "completions_frozen100.json").read_text())["completions"]
    record = json.loads((ANALYSIS / "g_baseline_split_b.json").read_text())
    results = measure(completions, training_problems(), max_workers=4)
    s = summarize(results)

    assert disagreements(results, record["split B (1/2)"]["per_problem"]) == ADR_015_CHANGES

    full = {
        tid: {"visible_pass": r.visible_pass, "plus_pass": r.plus_full}
        for tid, r in results.items()
    }

    assert [tid for tid in full if full[tid] != record["full_suite"]["per_problem"][tid]] == (
        ADR_015_CHANGES
    )
    for tid in ADR_015_CHANGES:  # each was a fail in the record, and now passes
        assert not record["split B (1/2)"]["per_problem"][tid]["plus_pass"]
        assert results[tid].plus_b and results[tid].plus_full
    assert (s.visible, s.plus_b, s.plus_full) == (65, 56, 56)
    assert tuple(round(x, 1) for x in s.ci_pp) == (4.0, 15.0)
    assert tuple(round(x, 1) for x in s.ci_full_pp) == (4.0, 15.0)
