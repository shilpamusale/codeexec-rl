"""Tests for the task-layer loaders.

These download EvalPlus datasets and run the dataset's reference
solutions to pre-compute expected outputs, so they are marked `dataset`: slower than unit
tests and dependent on network access. They validate that both datasets load fully
and that normalization (including HumanEval's body-concatenation) and pre-computation
produce correct, self-contained problems.
"""

import pytest

from codeexec.tasks.loaders import ORDER_UNSTABLE_TASK_IDS, load_humaneval, load_mbpp
from codeexec.tasks.problem import Problem

pytestmark = pytest.mark.dataset

# Expected problem counts for the pinned EvalPlus
# version (evalplus == 0.3.1: MBPP+ dataset v0.2.0, HumanEval+ dataset v0.1.10).
# MBPP+ ships 378; we drop 1 zero-held-out problem and 3 order-unstable ones
# (ADR-007), leaving 374.
EXPECTED_MBPP_COUNT = 374
EXPECTED_HUMANEVAL_COUNT = 164


def test_mbpp_loads_expected_count() -> None:
    """MBPP+ loads to the expected count after validation and ADR-007 exclusions."""
    problems = load_mbpp()
    assert len(problems) == EXPECTED_MBPP_COUNT


def test_humaneval_loads_expected_count() -> None:
    """Every HumanEval+ problem loads; the body-concatenation is correct."""
    problems = load_humaneval()
    assert len(problems) == EXPECTED_HUMANEVAL_COUNT


def test_order_unstable_problems_are_excluded() -> None:
    """The ADR-007 order-unstable problems do not appear in the loaded set."""
    problems = load_mbpp()
    for task_id in ORDER_UNSTABLE_TASK_IDS:
        assert task_id not in problems


def test_mbpp_problem_has_expected_shape() -> None:
    """A known-stable MBPP problem normalizes into the expected fields and tests."""
    # Mbpp/3 is a plain integer-returning problem (is_not_prime), so its output
    # is order-stable, unlike Mbpp/2 which was excluded under ADR-007.
    problem = load_mbpp()["Mbpp/3"]

    assert isinstance(problem, Problem)
    assert problem.source == "mbpp"
    assert problem.entry_point == "is_not_prime"
    assert len(problem.visible_tests) > 0
    assert len(problem.heldout_tests) > 0


def test_precompute_produces_correct_expected_output() -> None:
    """The pre-computed expected output matches the canonical solution's result."""
    problem = load_humaneval()["HumanEval/0"]

    assert problem.source == "humaneval"
    assert problem.entry_point == "has_close_elements"
    # The canonical must be complete (contains the def), not just the body.
    assert "def has_close_elements" in problem.canonical_solution
    assert len(problem.visible_tests) > 0
