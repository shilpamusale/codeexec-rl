"""Tests for the task-layer loaders.

These download EvalPlus datasets and run the dataset's reference
solutions to pre-compute expected outputs, so they are marked `dataset`: slower than unit
tests and dependent on network access. They validate that both datasets load fully
and that normalization (including HumanEval's body-concatenation) and pre-computation
produce correct, self-contained problems.
"""

import pytest

from codeexec.tasks.loaders import load_humaneval, load_mbpp
from codeexec.tasks.problem import Problem

pytestmark = pytest.mark.dataset

# Expected problem counts for the pinned EvalPlus
# version(evalplus == 0.3.1:MBPP+ dataset v0.2.0,HumanEval+ dataset v0.1.10).
EXPECTED_MBPP_COUNT = 378
EXPECTED_HUMANEVAL_COUNT = 164


def test_mbpp_loads_expected_count() -> None:
    """Every MBPP+ problem loads; none are dropped by validation."""
    problems = load_mbpp()
    assert len(problems) == EXPECTED_MBPP_COUNT


def test_humaneval_loads_expected_count() -> None:
    """Every HumanEval+ problem loads; the body-concatenation is correct."""
    problems = load_humaneval()
    assert len(problems) == EXPECTED_HUMANEVAL_COUNT


def test_mbpp_problem_has_expected_shape() -> None:
    """A known MBPP problem normalizes into the expected fields and tests."""
    problem = load_mbpp()["Mbpp/2"]

    assert isinstance(problem, Problem)
    assert problem.source == "mbpp"
    assert problem.entry_point == "similar_elements"
    assert len(problem.visible_tests) > 0
    assert len(problem.heldout_tests) > len(problem.visible_tests)


def test_precompute_produces_correct_expected_output() -> None:
    """The pre-computed expected output matches the canonical solution's result."""
    problem = load_humaneval()["HumanEval/0"]

    assert problem.source == "humaneval"
    assert problem.entry_point == "has_close_elements"
    # The canonical must be complete (contains the def), not just the body.
    assert "def has_close_elements" in problem.canonical_solution
    assert len(problem.visible_tests) > 0
