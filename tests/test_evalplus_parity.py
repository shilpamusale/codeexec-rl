"""Tests for ADR-015's parity check against EvalPlus's own evaluator.

EvalPlus grades for real, in its own subprocess; ours runs the real harness in
a host subprocess instead of Docker, so these need neither Docker nor the dataset.
The problems are tiny synthetic ones in EvalPlus's record format.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from codeexec.eval.evalplus_parity import (
    EvalPlusGrade,
    Verdict,
    classify,
    compare,
    evalplus_grade,
    main,
)
from codeexec.eval.grade import RunTestsFn, SuiteGrade
from codeexec.rewards.harness import TestCases, TestResult, _parse_result, build_harness
from codeexec.tasks.problem import Problem
from codeexec.tasks.problem import TestCase as Case


@pytest.fixture(autouse=True)
def _no_evaluate_memory_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    """As in test_oracle.py: macOS cannot take EvalPlus's 4 GB address-space cap."""
    monkeypatch.setenv("EVALPLUS_MAX_MEMORY_BYTES", "-1")


def _problem(task_id: str) -> Problem:
    return Problem(
        task_id=task_id,
        source="mbpp",
        prompt='"""Write a function that adds two numbers."""',
        entry_point="add",
        canonical_solution="def add(a, b):\n    return a + b\n",
        visible_tests=(Case((1, 2), 3),),
        heldout_tests=(Case((3, 4), 7), Case((5, 6), 11)),
        atol=0.0,
    )


def _record(task_id: str) -> dict[str, Any]:
    """The same problem as EvalPlus stores it, with its ground truth."""
    return {
        "task_id": task_id,
        "entry_point": "add",
        "atol": 0,
        "base_input": [[1, 2]],
        "plus_input": [[3, 4], [5, 6]],
    }


EXPECTED = {"base": [3], "plus": [7, 11], "base_time": [0.0], "plus_time": [0.0, 0.0]}
CORRECT = "    return a + b\n"
# Correct, but 1.5 s on the second held-out test: inside our 2 s per test,
# outside EvalPlus's max(1 s, 4 x the canonical's time).
SLOW = "    if a == 5:\n        import time\n        time.sleep(1.5)\n    return a + b\n"


def _host_run(code: str, entry_point: str, tests: TestCases, atol: float) -> TestResult:
    """The real harness, run by the host's Python instead of Docker."""
    harness = build_harness(code, entry_point, tests, atol)
    out = subprocess.run([sys.executable, "-"], input=harness, capture_output=True, text=True)
    passed, crashed = _parse_result(out.stdout, len(tests))
    return TestResult(passed=passed, total=len(tests), timed_out=False, crashed=crashed)


def _compare(completions: dict[str, str], run: RunTestsFn = _host_run) -> dict[str, Any]:
    ids = [tid for tid in completions if tid != "Mbpp/255"]
    return compare(
        completions,
        {tid: _problem(tid) for tid in ids},
        {tid: _record(tid) for tid in ids},
        dict.fromkeys(ids, EXPECTED),
        run=run,
        max_workers=2,
        evalplus_workers=2,
    )


def test_a_correct_solution_passes_both_ways() -> None:
    result = _compare({"Mbpp/9001": CORRECT})
    s = result["summary"]
    assert s["ours"] == s["evalplus"] == {"visible": 1, "full": 1}
    assert s["agree"], s["parity_holds"] == (1, True)
    assert result["per_problem"]["Mbpp/9001"]["relaxed"] is None  # no re-check needed


def test_evalplus_names_the_failing_tests() -> None:
    grade = evalplus_grade("def add(a, b):\n    return 7", _record("Mbpp/9001"), EXPECTED)
    assert (grade.base_status, grade.base_failed) == ("fail", (0,))
    assert (grade.plus_status, grade.plus_failed) == ("fail", (1,))
    assert grade.verdict == Verdict(visible=False, full=False)


def test_a_slow_correct_solution_is_put_down_to_time_limits() -> None:
    """Mbpp/160's shape: EvalPlus's 1 s limit fails it, ours passes it, and
    EvalPlus with generous limits agrees with ours."""

    result = _compare({"Mbpp/9001": SLOW})
    problem = result["per_problem"]["Mbpp/9001"]
    assert (problem["ours"]["full"], problem["evalplus"]["full"]) == (True, False)
    assert problem["evalplus"]["plus_failed"] == (1,)
    assert problem["relaxed"]["plus_status"] == "pass"
    assert problem["outcome"] == "time limits"
    assert result["summary"]["time_limits"] == ["Mbpp/9001"]
    assert result["summary"]["parity_holds"] is True


def test_any_other_disagreements_is_unexplained() -> None:
    def always_fail(code: str, entry_point: str, tests: TestCases, atol: float) -> TestResult:
        return TestResult(passed=0, total=len(tests), timed_out=False, crashed=False)

    result = _compare({"Mbpp/9001": CORRECT}, run=always_fail)
    assert result["per_problem"]["Mbpp/9001"]["outcome"] == "unexplained"
    assert result["summary"]["unexplained"] == ["Mbpp/9001"]
    assert result["summary"]["parity_holds"] is False


def test_problems_outside_the_set_and_malformed_completions_are_counted() -> None:
    result = _compare({"Mbpp/9001": CORRECT, "Mbpp/9002": "  return (a + \n", "Mbpp/255": ""})
    s = result["summary"]
    assert (s["completions"], s["graded"]) == (3, 1)
    assert s["set_aside"] == ["Mbpp/255"]
    assert s["malformed"] == ["Mbpp/9002"]


def test_our_program_timing_out_is_put_down_to_time_limits() -> None:
    timed_out = SuiteGrade(visible=TestResult(0, 1, timed_out=True, crashed=False), plus=None)
    passes = EvalPlusGrade(base_status="pass", plus_status="pass", base_failed=(), plus_failed=())
    assert classify(timed_out, passes, relaxed=None) == "time limits"


def test_refuses_to_overwrite_a_report(tmp_path: Path) -> None:
    out = tmp_path / "parity.json"
    out.write_text("{}")
    args = ["--completions", "unused.json", "--out", str(out), "--allow-dirty"]
    assert main(args) == 2
    assert out.read_text() == "{}"
