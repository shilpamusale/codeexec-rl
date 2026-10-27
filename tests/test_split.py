"""Tests for the held-out split (ADR-013): reward split A, measurement split B."""

import ast
import math
from fractions import Fraction
from typing import cast

import pytest

from codeexec.tasks.datasets import training_problems
from codeexec.tasks.problem import Problem
from codeexec.tasks.problem import TestCase as Case
from codeexec.tasks.split import split_heldout


def _problem(task_id: str = "Mbpp/3", n_heldout: int = 10) -> Problem:
    """A problem whose held-out test i has args (i,), so tests are easy to identify."""
    return Problem(
        task_id=task_id,
        source="mbpp",
        prompt="",
        entry_point="f",
        canonical_solution="def f(n):\n    return n\n",
        visible_tests=(Case((-1,), -1),),
        heldout_tests=tuple(Case((i,), i) for i in range(n_heldout)),
        atol=0.0,
    )


def _ids(tests: tuple[Case, ...]) -> list[int]:
    """The identifying int of each test (held-out test i has args (i,))."""
    return [cast(int, t.args[0]) for t in tests]


def test_golden_values_pin_the_pre_registered_rule() -> None:
    """These exact values ARE the ADR-013 rule. If this fails, the split changed:
    that needs a new ADR and a new SPLIT_VERSION, not an updated test."""
    split = split_heldout(_problem("Mbpp/3", 10))
    assert _ids(split.measurement) == [2, 3, 5, 6, 8]
    assert _ids(split.reward) == [0, 1, 4, 7, 9]


@pytest.mark.parametrize("n", [1, 2, 3, 13, 50, 105, 147])
def test_a_and_b_partition_the_heldout_tests(n: int) -> None:
    split = split_heldout(_problem(n_heldout=n))
    a, b = _ids(split.reward), _ids(split.measurement)
    assert not set(a) & set(b)  # disjoint
    assert sorted(a + b) == list(range(n))  # together, every held-out test
    assert len(b) == math.ceil(n / 2)  # B gets the extra test when n is odd


def test_each_split_keeps_loaded_order() -> None:
    split = split_heldout(_problem(n_heldout=30))
    assert _ids(split.reward) == sorted(_ids(split.reward))
    assert _ids(split.measurement) == sorted(_ids(split.measurement))


def test_split_is_deterministic() -> None:
    assert split_heldout(_problem()) == split_heldout(_problem())


def test_split_depends_on_task_id() -> None:
    assert split_heldout(_problem("Mbpp/3")) != split_heldout(_problem("Mbpp/4"))


def test_pre_registered_fallback_fraction() -> None:
    """ADR-013's fallback gives B two-thirds, from the same ranking."""
    split = split_heldout(_problem("Mbpp/3", 10), Fraction(2, 3))
    assert _ids(split.measurement) == [2, 3, 4, 5, 6, 8, 9]


@pytest.mark.parametrize("bad", [Fraction(0), Fraction(1), Fraction(3, 2)])
def test_fraction_outside_zero_one_is_rejected(bad: Fraction) -> None:
    with pytest.raises(ValueError, match="between 0 and 1"):
        split_heldout(_problem(), bad)


def _prompt_example_args(problem: Problem) -> list[tuple[object, ...]]:
    """Argument tuples of entry_point calls in the prompt's example `assert` lines."""
    found: list[tuple[object, ...]] = []
    for line in problem.prompt.splitlines():
        if not line.strip().startswith("assert"):
            continue
        try:
            tree = ast.parse(line.strip())
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and getattr(node.func, "id", None) == problem.entry_point:
                try:
                    found.append(tuple(ast.literal_eval(arg) for arg in node.args))
                except ValueError:
                    continue  # not a plain literal; cannot match a stored test
    return found


@pytest.mark.dataset
def test_real_problems_split_non_empty_and_prompts_never_reveal_split_b() -> None:
    """On the real 368: both parts are non-empty, and no prompt's example input is
    a split-B test unless it is also a visible test (so B shows nothing new)."""
    for problem in training_problems().values():
        split = split_heldout(problem)
        assert split.reward and split.measurement, problem.task_id
        # Lists, not sets: many test arguments are lists, which are unhashable.
        visible_args = [t.args for t in problem.visible_tests]
        b_args = [t.args for t in split.measurement]
        for args in _prompt_example_args(problem):
            assert args in visible_args or args not in b_args, problem.task_id
