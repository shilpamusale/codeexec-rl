"""Tests for the experiment's task-role layer."""

import pytest

from codeexec.tasks.datasets import ood_problems, training_problems
from codeexec.tasks.problem import Problem

pytestmark = pytest.mark.dataset

# MBPP+ ships 378; minus 1 zero-held-out and 3 order-unstable (ADR-007) = 374,
# minus 5 whose canonical fails its own gate (ADR-009) = 369.
EXPECTED_TRAINING_COUNT = 369
EXPECTED_OOD_COUNT = 164  # HumanEval+


# Each load re-runs every canonical solution (~15-20 s), so load once per module
# and share the result. Tests only read the problems, never mutate them.
@pytest.fixture(scope="module")
def training() -> dict[str, Problem]:
    return training_problems()


@pytest.fixture(scope="module")
def ood() -> dict[str, Problem]:
    return ood_problems()


def test_training_problems_are_mbpp(training: dict[str, Problem]) -> None:
    """Training problems are the MBPP set, in the expected count."""
    assert len(training) == EXPECTED_TRAINING_COUNT
    assert all(p.source == "mbpp" for p in training.values())


def test_ood_problems_are_humaneval(ood: dict[str, Problem]) -> None:
    """OOD Problems are the HumanEval+ set, in the expected count."""
    assert len(ood) == EXPECTED_OOD_COUNT
    assert all(p.source == "humaneval" for p in ood.values())


def test_partition_present_on_every_problem(training: dict[str, Problem]) -> None:
    """Every problem carries a non-empty visible/held-out partition,
    with held-out at least as large as visible (the augmentation adds tests)."""

    for problem in training.values():
        assert len(problem.visible_tests) > 0
        assert len(problem.heldout_tests) > 0
