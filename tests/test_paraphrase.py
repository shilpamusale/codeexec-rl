"""Tests for the docstring-paraphrase pertunbation."""

from __future__ import annotations

import random

from codeexec.perturb.paraphrase import paraphrase_docstring
from codeexec.tasks.problem import Problem, TestCase


def _mk(prompt: str) -> Problem:
    return Problem(
        task_id="T/1",
        source="mbpp",
        prompt=prompt,
        entry_point="f",
        canonical_solution="def f(x):\n    return x",
        visible_tests=(TestCase(args=(1,), expected=1),),
        heldout_tests=(TestCase(args=(2,), expected=2),),
        atol=0.0,
    )


def test_paraphrase_rephrases_to_leadin() -> None:
    p = _mk('"""\nWrite a python function to sum a list.\nassert f([1]) == 1\n"""')

    out = paraphrase_docstring(p, random.Random(1))

    # the original lead-in is gone, replaced by an equivalent
    assert "Write a python function to" not in out.prompt
    # the spec body is preserved
    assert "sum a list." in out.prompt


def test_paraphrase_that_form_stays_grammatical() -> None:
    """Regression: a `that`-connector lead-in must be replaced with a `that`-form
    never a `to` form (which would produce 'a function that to...').
    """
    p = _mk('"""Write a function that returns the max.\n assert f(1) == 1\n"""')
    out = paraphrase_docstring(p, random.Random(2))
    assert " that returns the max." in out.prompt
    assert "that to" not in out.prompt
    assert "to returns" not in out.prompt


def test_paraprahse_applies_synonym() -> None:
    p = _mk('"""\nWrite a function to calculate the total. \n assert f(1) == 1\n"""')
    out = paraphrase_docstring(p, random.Random(3))
    assert "compute the total" in out.prompt
    assert "calculate" not in out.prompt


def test_paraphrase_noop_on_old_leadin() -> None:
    """A prompt whose lead-in matches no known pattern us returned unchanged."""
    p = _mk('"""\nGiven a matrix, do something unusual here. \n assert f([]) == []\n"""')
    out = paraphrase_docstring(p, random.Random(4))
    assert out.prompt == p.prompt


def test_paraphrase_leaves_code_and_tests_unchanged() -> None:
    p = _mk('"""\nWrite a function to add.\nassert f(1) == 1\n"""')
    out = paraphrase_docstring(p, random.Random(6))
    assert out.canonical_solution == p.canonical_solution
    assert out.entry_point == p.entry_point
    assert out.visible_tests == p.visible_tests


def test_paraphrase_is_deterministic() -> None:
    p = _mk('"""\n Write a function to calculate the sum.\n assert f(1) == 1\n """')
    a = paraphrase_docstring(p, random.Random(7))
    b = paraphrase_docstring(p, random.Random(7))
    assert a.prompt == b.prompt
