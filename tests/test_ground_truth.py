"""Unit tests for how the loader computes expected outputs (ADR-015).

The loader must build ground truth exactly as EvalPlus 0.3.1 does: run the
canonical on a copy of each input, and for "not None" tasks store whether the
output was not None. These use tiny stand-in canonicals, so they run in CI
without the datasets; tests/test_loaders.py checks the real problems.
"""

from __future__ import annotations

import re

from codeexec.tasks.loaders import _build_tests


def test_canonical_that_sorts_in_place_leaves_the_stored_input_intact() -> None:
    """Mbpp/71's shape: a canonical that sorts its argument and returns it."""

    def sorts_in_place(xs: list[int]) -> list[int]:
        xs.sort()
        return xs

    [case] = _build_tests(sorts_in_place, [[[5, 15, 37, 25, 79]]], "Mbpp/71", "visible")
    assert case.args == ([5, 15, 37, 25, 79],)  # the test's real input, not the sorted one
    assert case.expected == [5, 15, 25, 37, 79]


def test_canonical_that_empties_its_input_leaves_the_stored_input_intact() -> None:
    """Mbpp/635's shape: a canonical that consumes its argument."""

    def pops_everything(xs: list[int]) -> list[int]:
        out = []
        while xs:
            out.append(xs.pop())
        return out

    [case] = _build_tests(pops_everything, [[[1, 2, 3]]], "Mbpp/635", "visible")
    assert case.args == ([1, 2, 3],)
    assert case.expected == [3, 2, 1]


def test_not_none_task_stores_whether_the_output_is_not_none() -> None:
    """Mbpp/787's shape: a canonical returning a re.Match or None."""

    def text_match_three(text: str) -> re.Match[str] | None:
        return re.search("ab{3}", text)

    cases = _build_tests(
        text_match_three, [["abbb"], ["ac"]], "Mbpp/787", "visible", output_not_none=True
    )
    assert [c.expected for c in cases] == [True, False]


def test_other_tasks_store_the_output_itself() -> None:
    cases = _build_tests(lambda s: s or None, [["x"], [""]], "Mbpp/1", "visible")
    assert [c.expected for c in cases] == ["x", None]
