"""Tests for base-model prompt construction and completion extraction."""

from __future__ import annotations

from codeexec.eval.extract import build_prompt, extract_function
from codeexec.tasks.problem import Problem, TestCase


def _mk(entry_point: str = "add", canonical: str = "def add(a, b):\n    return a + b") -> Problem:
    return Problem(
        task_id="T/1",
        source="mbpp",
        prompt=f'"""Write a function to add two numbers.\nassert {entry_point}(2, 3) == 5"""',
        entry_point=entry_point,
        canonical_solution=canonical,
        visible_tests=(TestCase(args=(2, 3), expected=5),),
        heldout_tests=(TestCase(args=(10, 20), expected=30),),
        atol=0.0,
    )


def test_build_prompt_appends_signature_stub() -> None:
    p = _mk()
    prompt = build_prompt(p)
    assert prompt.rstrip().endswith("def add(a, b):")
    assert "Write a function to add" in prompt


def test_build_prompt_reads_signature_from_canonical() -> None:
    # a 3-arg canonical -> stub must have all three args
    p = _mk("f3", "def f3(x, y, z):\n    return x")
    assert "def f3(x, y, z):" in build_prompt(p)


def test_extract_clean_body() -> None:
    out = extract_function("    return a + b", _mk())
    assert out == "def add(a, b):\n    return a + b"


def test_extract_cuts_trailing_definition() -> None:
    """The base model continues past the function; a new top-level def is cut."""
    out = extract_function("    return a + b\n\ndef other():\n    pass", _mk())
    assert out is not None
    assert "def other" not in out
    assert "return a + b" in out


def test_extract_cuts_trailing_prose_and_asserts() -> None:
    out = extract_function("    return a + b\nassert add(1, 1) == 2\nThis explains it.", _mk())
    assert out is not None
    assert "assert" not in out
    assert "This explains" not in out


def test_extract_keeps_multiline_body() -> None:
    body = "    total = a + b\n    return total"
    out = extract_function(body, _mk())
    assert out == "def add(a, b):\n    total = a + b\n    return total"


def test_extract_malformed_prose_is_none() -> None:
    assert extract_function("This function adds two numbers.", _mk()) is None


def test_extract_malformed_empty_body_is_none() -> None:
    # stub with no body, then a new def -> the function has no body -> syntax error
    assert extract_function("\ndef other():\n    pass", _mk()) is None


def test_extract_malformed_wrong_function_name_is_none() -> None:
    """A completion that defines a differently-named function is malformed:
    the stub's `def add` with an empty body cannot be rescued by a later def."""
    out = extract_function("\ndef helper():\n    return 1", _mk())
    assert out is None


def test_extract_preserves_indented_blocks() -> None:
    """Nested indentation (if/for inside the body) must be kept, not cut."""
    body = "    if a > b:\n        return a\n    return b"
    out = extract_function(body, _mk())
    assert out is not None
    assert "if a > b:" in out
    assert "return b" in out
