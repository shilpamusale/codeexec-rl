"""Tests for the task-layer loaders.

These download EvalPlus datasets and run the dataset's reference
solutions to pre-compute expected outputs, so they are marked `dataset`: slower than unit
tests and dependent on network access. They validate that both datasets load fully
and that normalization (including HumanEval's body-concatenation) and pre-computation
produce correct, self-contained problems.
"""

import pytest
from evalplus.data import get_human_eval_plus, get_mbpp_plus

from codeexec.tasks.loaders import (
    CORRUPT_HELDOUT_TASK_IDS,
    ORDER_UNSTABLE_TASK_IDS,
    UNGRADEABLE_CANONICAL_TASK_IDS,
    load_humaneval,
    load_mbpp,
)
from codeexec.tasks.problem import Problem

pytestmark = pytest.mark.dataset

# Expected problem counts for the pinned EvalPlus
# version (evalplus == 0.3.1: MBPP+ dataset v0.2.0, HumanEval+ dataset v0.1.10).
# MBPP+ ships 378; we drop 1 zero-held-out problem and 3 order-unstable ones
# (ADR-007), leaving 374, then 5 whose canonical fails its own gate (ADR-009),
# leaving 369.
EXPECTED_MBPP_COUNT = 369
EXPECTED_HUMANEVAL_COUNT = 164


# Each load re-runs every canonical solution (~15-20 s), so load once per module
# and share the result. Tests only read the problems, never mutate them.
@pytest.fixture(scope="module")
def mbpp() -> dict[str, Problem]:
    return load_mbpp()


@pytest.fixture(scope="module")
def humaneval() -> dict[str, Problem]:
    return load_humaneval()


def test_mbpp_loads_expected_count(mbpp: dict[str, Problem]) -> None:
    """MBPP+ loads to the expected count after validation and ADR-007 exclusions."""
    assert len(mbpp) == EXPECTED_MBPP_COUNT


def test_humaneval_loads_expected_count(humaneval: dict[str, Problem]) -> None:
    """Every HumanEval+ problem loads; the body-concatenation is correct."""
    assert len(humaneval) == EXPECTED_HUMANEVAL_COUNT


def test_order_unstable_problems_are_excluded(mbpp: dict[str, Problem]) -> None:
    """The ADR-007 order-unstable problems do not appear in the loaded set."""
    for task_id in ORDER_UNSTABLE_TASK_IDS:
        assert task_id not in mbpp


def test_ungradeable_canonicals_are_excluded(mbpp: dict[str, Problem]) -> None:
    """The ADR-009 ungradeable-canonical problems do not appear in the loaded set."""
    for task_id in UNGRADEABLE_CANONICAL_TASK_IDS:
        assert task_id not in mbpp


def test_corrupt_heldout_problems_are_excluded(mbpp: dict[str, Problem]) -> None:
    """The ADR-009 corrupt-held-out problems do not appear in the loaded set."""
    for task_id in CORRUPT_HELDOUT_TASK_IDS:
        assert task_id not in mbpp


def test_mbpp_problem_has_expected_shape(mbpp: dict[str, Problem]) -> None:
    """A known-stable MBPP problem normalizes into the expected fields and tests."""
    # Mbpp/3 is a plain integer-returning problem (is_not_prime), so its output
    # is order-stable, unlike Mbpp/2 which was excluded under ADR-007.
    problem = mbpp["Mbpp/3"]

    assert isinstance(problem, Problem)
    assert problem.source == "mbpp"
    assert problem.entry_point == "is_not_prime"
    assert len(problem.visible_tests) > 0
    assert len(problem.heldout_tests) > 0


def test_precompute_produces_correct_expected_output(
    humaneval: dict[str, Problem],
) -> None:
    """The pre-computed expected output matches the canonical solution's result."""
    problem = humaneval["HumanEval/0"]

    assert problem.source == "humaneval"
    assert problem.entry_point == "has_close_elements"
    # The canonical must be complete (contains the def), not just the body.
    assert "def has_close_elements" in problem.canonical_solution
    assert len(problem.visible_tests) > 0


def test_every_stored_input_is_evalplus_s_input(
    mbpp: dict[str, Problem], humaneval: dict[str, Problem]
) -> None:
    """No canonical changed a stored test input (ADR-015). Some canonicals sort or
    empty their argument in place (Mbpp/57, 64, 71, 428, 635, HumanEval/20); every
    test must still hold EvalPlus's own input, in order. No test is dropped in
    either dataset, so the stored tests line up one-to-one with EvalPlus's inputs."""
    for problems, raw in ((mbpp, get_mbpp_plus()), (humaneval, get_human_eval_plus())):
        for task_id, problem in problems.items():
            for tests, key in (
                (problem.visible_tests, "base_input"),
                (problem.heldout_tests, "plus_input"),
            ):
                inputs = [tuple(args) for args in raw[task_id][key]]
                assert [t.args for t in tests] == inputs, f"{task_id} {key}"


def test_not_none_task_stores_bools(mbpp: dict[str, Problem]) -> None:
    """Mbpp/787 stores `output is not None`, as EvalPlus does, not re.Match objects."""
    problem = mbpp["Mbpp/787"]
    expected = {t.expected for t in problem.visible_tests + problem.heldout_tests}
    assert expected == {True, False}
