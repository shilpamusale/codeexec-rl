"""Two-suite grading: visible_pass and plus_pass for one candidate solution.

The H1 metrics:
    visible_pass    all visible tests pass (the canonical MBPP tests).
    plus_pass       visible AND held-out tests all pass -- the standard EvalPlus
                    "solved" definition for MBPP+.
    G = visible_pass_rate - plus_pass_rate

so G is the fraction of problems that pass the sparse visible tests but fail
the stronger held-out suite.

The test runner is injected, the same pattern as smoke.py. It defaults to the Docker sandbox
(`run_tests`), so production callers get isolation without asking for it; unit tests pass a stub,
and the grader-parity check passes an unsandboxed reference runner. Extraction is the caller's job:
this module grades code that has already been pulled out of a completion.

"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from codeexec.rewards.harness import TestCases, TestResult, run_tests
from codeexec.tasks.problem import Problem, TestCase

# Same signature as harness.run_tests: (code, entry_point, tests, atol) -> TestResult.
RunTestsFn = Callable[[str, str, TestCases, float], TestResult]


@dataclass(frozen=True)
class SuiteGrade:
    """The outcome of grading one solution against both suites.

    The raw TestResults are kept, not just the booleans, so a caller can see
    *why* something failed (timed out, crashed, or wrong on some cases).
    """

    visible: TestResult
    plus: TestResult | None  # None = not run, because visible already failed

    @property
    def visible_pass(self) -> bool:
        return self.visible.all_passed

    @property
    def plus_pass(self) -> bool:
        return self.plus is not None and self.plus.all_passed


def _pairs(tests: Sequence[TestCase]) -> list[tuple[tuple[object, ...], object]]:
    """Convert TestCase objects into the (args, expected) pairs the harness takes."""
    return [(tc.args, tc.expected) for tc in tests]


def grade_two_suites(code: str, problem: Problem, run: RunTestsFn = run_tests) -> SuiteGrade:
    """Grade `code` on the visible suite, then on visible + held-out.

    The plus suite runs only if visible passed: plus requires visible by
    definition, so a visible failure already decides plus_pass = False and the
    second execution would be wasted. When it does run, visible and held-out
    tests go in ONE execution (not held-out alone), matching how Phase 3
    measured G_baseline, so the parity check compares identical programs.
    """
    visible = run(code, problem.entry_point, _pairs(problem.visible_tests), problem.atol)
    if not visible.all_passed:
        return SuiteGrade(visible=visible, plus=None)
    all_tests = (*problem.visible_tests, *problem.heldout_tests)
    plus = run(code, problem.entry_point, _pairs(all_tests), problem.atol)
    return SuiteGrade(visible=visible, plus=plus)
