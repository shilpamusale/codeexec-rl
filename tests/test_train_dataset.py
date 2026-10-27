"""Tests for the GRPO training dataset (one record per training problem)."""

import pytest

from codeexec.eval.extract import build_prompt
from codeexec.tasks.datasets import training_problems
from codeexec.tasks.problem import Problem
from codeexec.tasks.problem import TestCase as Case
from codeexec.train.dataset import training_records


def _problem(task_id: str, source: str = "mbpp") -> Problem:
    return Problem(
        task_id=task_id,
        source=source,
        prompt=f'"""\nProblem {task_id}.\n"""\n',
        entry_point="f",
        canonical_solution="def f(n):\n    return n\n",
        visible_tests=(Case((1,), 1),),
        heldout_tests=(Case((2,), 2),),
        atol=0.0,
    )


def _problems(*task_ids: str) -> dict[str, Problem]:
    return {t: _problem(t) for t in task_ids}


def test_one_record_per_problem_with_only_prompt_and_task_id() -> None:
    records = training_records(_problems("Mbpp/2", "Mbpp/3"))
    assert len(records) == 2
    for record in records:
        assert set(record) == {"prompt", "task_id"}  # no tests or solutions leak in


def test_prompt_is_the_locked_build_prompt() -> None:
    problems = _problems("Mbpp/3")
    (record,) = training_records(problems)
    assert record["prompt"] == build_prompt(problems["Mbpp/3"])


def test_records_are_sorted_and_deterministic() -> None:
    problems = _problems("Mbpp/7", "Mbpp/3", "Mbpp/12")
    records = training_records(problems)
    assert [r["task_id"] for r in records] == sorted(problems)
    assert training_records(problems) == records


def test_humaneval_problem_is_rejected() -> None:
    problems = _problems("Mbpp/3")
    problems["HumanEval/0"] = _problem("HumanEval/0", source="humaneval")
    with pytest.raises(ValueError, match="MBPP only"):
        training_records(problems)


@pytest.mark.dataset
def test_real_training_set_has_368_unique_records() -> None:
    records = training_records(training_problems())
    assert len(records) == 368
    assert len({r["task_id"] for r in records}) == 368
    assert len({r["prompt"] for r in records}) == 368
