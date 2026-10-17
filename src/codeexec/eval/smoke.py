"""Base-model smoke test: measure base pass@1 and malformed-output rate.

The loop is pure orchestration -- it build the prompt and extracts the function
(repo logic), and DELEGATES generation and grading via injected callables. That
keeps it CPU-testable with stubs, and reusable on a GPU training box with the
real sandbox grader. Produces the two numbers that resolve ADR-001 (is there
optimization pressure?) and ADR-006 (base vs Instruct format reliability)."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from codeexec.eval.extract import build_prompt, extract_function
from codeexec.tasks.problem import Problem

# prompt -> raw completion
GenerateFn = Callable[[str], str]

# code, problem -> did it pass (all tests)?
GradeFn = Callable[[str, Problem], bool]


@dataclass(frozen=True)
class SmokeResult:
    """Outcome of the base-model smoke test over a set of problems."""

    total: int
    passed: int  # well-formed AND all tests pass
    malformed: int  # extraction returned None

    @property
    def pass_at_1(self) -> float:
        """Fraction solved. Malformed completions count as failures (not solved)."""
        return self.passed / self.total if self.total else 0.0

    @property
    def malformed_rate(self) -> float:
        return self.malformed / self.total if self.total else 0.0

    @property
    def wellformed_rate(self) -> float:
        return 1.0 - self.malformed_rate


def run_smoke_test(
    problems: Sequence[Problem],
    generate_fn: GenerateFn,
    grade_fn: GradeFn,
    on_problem: Callable[[str, bool, bool], None] | None = None,
) -> SmokeResult:
    """Rum the base model over `problems` and tally pass@1 and malformed rate.

    For each problem: build the prompt, generate, extract the function.
    A completion that does not extract into a valid function is malformed and counts
    as a failure. A well-formed function is graded; it passes only if `grade_fn` reports
    all tests pass.
    `on_problem(task_id, malformed, passed)` is an optional progress hook.
    """

    passed = 0
    malformed = 0
    for problem in problems:
        prompt = build_prompt(problem)
        completion = generate_fn(prompt)
        code = extract_function(completion, problem)
        if code is None:
            malformed += 1
            is_malformed, is_passed = True, False
        else:
            is_passed = grade_fn(code, problem)
            passed += int(is_passed)
            is_malformed = False
        if on_problem is not None:
            on_problem(problem.task_id, is_malformed, is_passed)
    return SmokeResult(total=len(problems), passed=passed, malformed=malformed)
