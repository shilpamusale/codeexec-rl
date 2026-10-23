"""Tests for the two-suite grader (visible_pass / plus_pass).

Most tests use a fake runner, so they need no Docker and run in CI.
The last test is marked `integration`: it runs through the real sandbox
to prove the end-to-end path, including the case G exists to count -- a
solution that passes the visible tests but fails the held-out suite.
"""

import pytest

from codeexec.eval.grade import grade_two_suites
from codeexec.rewards.harness import TestCases, TestResult
from codeexec.tasks.problem import Problem
from codeexec.tasks.problem import TestCase as Case  # alias: pytest would try to collect "Test*"

Pairs = list[tuple[tuple[object, ...], object]]


class RecordingRunner:
    """A fake test runner: records every call and returns scriptred results in order."""

    def __init__(self, results: list[TestResult]) -> None:
        self._results = list(results)
        self.calls: list[tuple[str, str, Pairs, float]] = []

    def __call__(self, code: str, entry_point: str, tests: TestCases, atol: float) -> TestResult:
        self.calls.append((code, entry_point, list(tests), atol))
        return self._results.pop(0)


def _problem(atol: float = 0.0) -> Problem:
    """A tiny problem: one visible test, two held-out tests."""

    return Problem(
        task_id="Mbpp/test",
        source="mbpp",
        prompt="Return n doubled",
        entry_point="double",
        canonical_solution="def double(n):\n    return n * 2\n",
        visible_tests=(Case((1,), 2),),
        heldout_tests=(Case((5,), 10), Case((7,), 14)),
        atol=atol,
    )


def _passed(total: int) -> TestResult:
    return TestResult(passed=total, total=total, timed_out=False, crashed=False)


def _failed(total: int) -> TestResult:
    return TestResult(passed=0, total=total, timed_out=False, crashed=False)


def test_both_suites_pass() -> None:
    runner = RecordingRunner([_passed(1), _passed(3)])
    grade = grade_two_suites("code", _problem(), run=runner)
    assert grade.visible_pass
    assert grade.plus_pass
    assert len(runner.calls) == 2


def test_visible_fail_skips_the_plus_run() -> None:
    runner = RecordingRunner([_failed(1)])
    grade = grade_two_suites("code", _problem(), run=runner)
    assert not grade.visible_pass
    assert not grade.plus_pass
    assert grade.plus is None
    assert len(runner.calls) == 1  # no wasted second execution


def test_visible_pass_plus_fail_is_the_gap_case() -> None:
    """Passes the sparse visible tests but fails held-out: exactly what G counts."""
    partial = TestResult(passed=2, total=3, timed_out=False, crashed=False)
    runner = RecordingRunner([_passed(1), partial])
    grade = grade_two_suites("code", _problem(), run=runner)
    assert grade.visible_pass
    assert not grade.plus_pass


def test_plus_run_is_visible_then_heldout_in_one_call() -> None:
    """The plus program must match Phase 3's: visible + held-out, in that order."""
    runner = RecordingRunner([_passed(1), _passed(3)])
    grade_two_suites("code", _problem(), run=runner)
    visible_call, plus_call = runner.calls
    assert visible_call[2] == [((1,), 2)]
    assert plus_call[2] == [((1,), 2), ((5,), 10), ((7,), 14)]


def test_heldout_argument_chooses_the_plus_suite() -> None:
    """Passing a held-out subset (e.g. split B) grades plus on visible + that subset only."""
    problem = _problem()
    runner = RecordingRunner([_passed(1), _passed(2)])
    grade_two_suites("code", problem, run=runner, heldout=problem.heldout_tests[1:])
    _visible_call, plus_call = runner.calls
    assert plus_call[2] == [((1,), 2), ((7,), 14)]


def test_code_entry_point_and_atol_are_passed_through() -> None:
    runner = RecordingRunner([_passed(1), _passed(3)])
    grade_two_suites("def double(n): ...", _problem(atol=1e-6), run=runner)
    for code, entry_point, _tests, atol in runner.calls:
        assert code == "def double(n): ..."
        assert entry_point == "double"
        assert atol == 1e-6


@pytest.mark.parametrize(
    "bad",
    [
        TestResult(passed=0, total=1, timed_out=True, crashed=False),
        TestResult(passed=0, total=1, timed_out=False, crashed=True),
    ],
    ids=["timed_out", "crashed"],
)
def test_timedout_or_crash_is_not_a_pass(bad: TestResult) -> None:
    runner = RecordingRunner([bad])
    grade = grade_two_suites("code", _problem(), run=runner)
    assert not grade.visible_pass
    assert grade.plus is None


@pytest.mark.integration
def test_real_sandbox_separates_overfit_from_correct() -> None:
    """End to end through Docker, using the default runner (run_tests).

    A solution hard_coded to the single test passes visible but fails plus --
    an overfit solution, the thing H1 measures. A correct solution passes both.
    """
    problem = _problem()
    overfit = "def double(n):\n     return 2 if n == 1 else 0\n"
    correct = "def double(n):\n    return n * 2\n"

    overfit_grade = grade_two_suites(overfit, problem)
    assert overfit_grade.visible_pass
    assert not overfit_grade.plus_pass

    correct_grade = grade_two_suites(correct, problem)
    assert correct_grade.visible_pass
    assert correct_grade.plus_pass
