"""Tests for the function-rename perturbation."""

from __future__ import annotations

import ast
import random

from codeexec.perturb.rename import NEUTRAL_NAMES, rename_function
from codeexec.tasks.problem import Problem, TestCase


def _mk(prompt: str, entry_point: str, canonical: str) -> Problem:
    return Problem(
        task_id="T/1",
        source="mbpp",
        prompt=prompt,
        entry_point=entry_point,
        canonical_solution=canonical,
        visible_tests=(TestCase(args=(1,), expected=1),),
        heldout_tests=(TestCase(args=(2,), expected=2),),
        atol=0.0,
    )


def test_rename_changes_entry_point() -> None:
    p = _mk(
        '"""\n'
        "Write a python function to check if a number is positive.\n"
        "assert check(5) == True\n"
        '"""',
        "check",
        "def check(n):\n    return n > 0",
    )
    out = rename_function(p, random.Random(0))
    assert out.entry_point != "add"
    assert out.entry_point in NEUTRAL_NAMES


def test_rename_updates_canonical_def_and_calls() -> None:
    p = _mk('"""\nx\nassert f(1) == 1\n"""', "f", "def f(n):\n    return f(n)")  # self-recursive
    out = rename_function(p, random.Random(1))
    tree = ast.parse(out.canonical_solution)
    # no occurrence of the old name 'f' survives as an identifier
    assert not any(
        (isinstance(n, ast.Name) and n.id == "f")
        or (isinstance(n, ast.FunctionDef) and n.name == "f")
        for n in ast.walk(tree)
    )
    # the new name is the function def's name
    defs = [n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
    assert out.entry_point in defs


def test_rename_preserves_prose_on_english_word_collision() -> None:
    """Regression: an entry-point name that is also an English word (check) must
    be renamed only in the assert call site, never in the docstring prose."""
    p = _mk(
        '"""\n'
        "Write a python function to check if a number is positive.\n"
        "assert check(5) == True\n"
        '"""',
        "check",
        "def check(n):\n    return n > 0",
    )
    out = rename_function(p, random.Random(2))
    # prose keeps the English word "check"
    assert "to check if a number" in out.prompt
    # the assert call site is renamed
    assert f"assert {out.entry_point}(5)" in out.prompt
    # the old name no longer appears as a call
    assert "assert check(" not in out.prompt


def test_rename_does_not_touch_string_literals() -> None:
    """AST rename must not corrupt a string literal that contains the name."""
    p = _mk('"""\nx\nassert f(1) == 1\n"""', "f", 'def f(n):\n    return "f is the name"')
    out = rename_function(p, random.Random(3))
    # the string literal content is preserved (still contains "f is the name")
    assert "f is the name" in out.canonical_solution


def test_rename_leaves_tests_unchanged() -> None:
    p = _mk('"""\nx\nassert f(1) == 1\n"""', "f", "def f(n):\n    return n")
    out = rename_function(p, random.Random(4))
    assert out.visible_tests == p.visible_tests
    assert out.heldout_tests == p.heldout_tests
    assert out.atol == p.atol
    assert out.source == p.source
    assert out.task_id == p.task_id


def test_rename_is_deterministic() -> None:
    p = _mk('"""\nx\nassert f(1) == 1\n"""', "f", "def f(n):\n    return n")
    a = rename_function(p, random.Random(42))
    b = rename_function(p, random.Random(42))
    assert a.entry_point == b.entry_point
    assert a.canonical_solution == b.canonical_solution
    assert a.prompt == b.prompt
