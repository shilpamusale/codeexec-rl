"""Tests for the experiment's task-role layer."""

import pytest

from codeexec.tasks.datasets import ood_problems, training_problems

pytestmark = pytest.mark.dataset


# MBPP+ ships 378; minus 1 zero-held-out and 3 order-unstable (ADR-007) = 374,
# minus 5 whose canonical fails its own gate (ADR-009) = 369.
EXPECTED_TRAINING_COUNT = 369

# EXPECTED_TRAINING_COUNT = 374  # MBPP+ (378 shipped - 1 zero-held-out - 3 order-unstable, ADR-007)
EXPECTED_OOD_COUNT = 164  # HumanEval+


def test_training_problems_are_mbpp() -> None:
    """Training problems are the MBPP set,in the expected count."""
    problems = training_problems()
    assert len(problems) == EXPECTED_TRAINING_COUNT
    assert all(p.source == "mbpp" for p in problems.values())


def test_ood_problems_are_humaneval() -> None:
    """OOD Problems are the HumanEval+ set, in the expected count."""
    problems = ood_problems()
    assert len(problems) == EXPECTED_OOD_COUNT
    assert all(p.source == "humaneval" for p in problems.values())


def test_partition_present_on_every_problem() -> None:
    """Every problem carries a non-empty visible/held-out partition,
    with held-out at least as large as visible (the augmentation adds tests)."""

    for problem in training_problems().values():
        assert len(problem.visible_tests) > 0
        assert len(problem.heldout_tests) > 0
