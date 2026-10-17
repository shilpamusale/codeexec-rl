"""Tests for the base-model smoke-test eval loop.

Generation and grading are injected, so the loop is tested with stubs -- no GPU,
no model, no sandbox. (generate.py is not tested here: it needs a loaded model.)
"""

from __future__ import annotations

from codeexec.eval.smoke import run_smoke_test
from codeexec.tasks.problem import Problem, TestCase


def _mk(task_id: str, entry_point: str = "add") -> Problem:
    return Problem(
        task_id=task_id,
        source="mbpp",
        prompt=f'"""Write a function to add.\nassert {entry_point}(1, 2) == 3"""',
        entry_point=entry_point,
        canonical_solution=f"def {entry_point}(a, b):\n    return a + b",
        visible_tests=(TestCase(args=(1, 2), expected=3),),
        heldout_tests=(TestCase(args=(5, 5), expected=10),),
        atol=0.0,
    )


def _exec_grade(code: str, problem: Problem) -> bool:
    """Trusted in-process grader for tests (the code is our own stub)."""
    ns: dict[str, object] = {}
    exec(code, ns)
    fn = ns[problem.entry_point]
    return all(
        fn(*t.args) == t.expected  # type: ignore[operator]
        for t in (*problem.visible_tests, *problem.heldout_tests)
    )


def test_smoke_counts_pass_fail_malformed() -> None:
    probs = [_mk("T/1"), _mk("T/2"), _mk("T/3")]
    completions = iter(
        [
            "    return a + b",  # correct -> pass
            "    return 0",  # wrong -> fail (well-formed)
            "just prose, no function",  # malformed
        ]
    )
    result = run_smoke_test(probs, lambda _p: next(completions), _exec_grade)
    assert result.total == 3
    assert result.passed == 1
    assert result.malformed == 1
    assert result.pass_at_1 == 1 / 3
    assert result.malformed_rate == 1 / 3


def test_smoke_all_pass() -> None:
    probs = [_mk("T/1"), _mk("T/2")]
    result = run_smoke_test(probs, lambda _p: "    return a + b", _exec_grade)
    assert result.passed == 2
    assert result.malformed == 0
    assert result.pass_at_1 == 1.0
    assert result.malformed_rate == 0.0


def test_smoke_all_malformed() -> None:
    probs = [_mk("T/1"), _mk("T/2")]
    result = run_smoke_test(probs, lambda _p: "not a function at all", _exec_grade)
    assert result.passed == 0
    assert result.malformed == 2
    assert result.pass_at_1 == 0.0
    assert result.malformed_rate == 1.0
    assert result.wellformed_rate == 0.0


def test_smoke_malformed_counts_as_failure_not_pass() -> None:
    """A malformed completion must lower pass@1 (counts as unsolved)."""
    probs = [_mk("T/1")]
    result = run_smoke_test(probs, lambda _p: "garbage", _exec_grade)
    assert result.pass_at_1 == 0.0
    assert result.malformed_rate == 1.0


def test_smoke_empty_problem_set() -> None:
    result = run_smoke_test([], lambda _p: "", _exec_grade)
    assert result.total == 0
    assert result.pass_at_1 == 0.0
    assert result.malformed_rate == 0.0


def test_smoke_progress_hook_called_per_problem() -> None:
    probs = [_mk("T/1"), _mk("T/2")]
    seen: list[tuple[str, bool, bool]] = []
    run_smoke_test(
        probs,
        lambda _p: "    return a + b",
        _exec_grade,
        on_problem=lambda tid, m, p: seen.append((tid, m, p)),
    )
    assert [s[0] for s in seen] == ["T/1", "T/2"]
    assert all(not m and p for _tid, m, p in seen)  # all well-formed and passed
