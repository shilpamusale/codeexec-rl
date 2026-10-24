"""Tests for the RL environment (ADR-012) and its reward mapping (ADR-013, ADR-014).

Unit tests replace the sandbox with a recording stub, so they run in CI. The
integration tests drive the real sandbox, marked `integration`.
"""

from __future__ import annotations

from typing import cast

import pytest

from codeexec.env import Arm, CodeExecEnv
from codeexec.eval.extract import extract_function
from codeexec.rewards.harness import SandboxInfraError, TestCases, TestResult
from codeexec.tasks.problem import Problem
from codeexec.tasks.problem import TestCase as Case
from codeexec.tasks.split import split_heldout
from codeexec.train.dataset import training_records

CORRECT = "    return a + b\n"
OVERFIT = "    return 101\n"  # passes the one visible test, fails every held-out test
MALFORMED = "    return (a +\n"  # does not parse


def _problem(task_id: str = "Mbpp/3") -> Problem:
    """add(a, b). Visible test args start with 100; held-out test i starts with i."""
    return Problem(
        task_id=task_id,
        source="mbpp",
        prompt='"""Write a function that adds two numbers."""',
        entry_point="add",
        canonical_solution="def add(a, b):\n    return a + b\n",
        visible_tests=(Case((100, 1), 101),),
        heldout_tests=tuple(Case((i, i), 2 * i) for i in range(10)),
        atol=0.0,
    )


def _ids(tests: TestCases) -> list[int]:
    return [cast(int, args[0]) for args, _ in tests]


def _result(
    passed: int,
    total: int = 1,
    *,
    timed_out: bool = False,
    crashed: bool = False,
    infra_retries: int = 0,
) -> TestResult:
    return TestResult(passed, total, timed_out, crashed, infra_retries)


class RecordingSandbox:
    """Stands in for run_tests: returns (or raises) a fixed outcome, records calls."""

    def __init__(self, outcome: TestResult | Exception) -> None:
        self.outcome = outcome
        self.calls: list[tuple[str, str, TestCases, float]] = []

    def __call__(self, code: str, entry_point: str, tests: TestCases, atol: float) -> TestResult:
        self.calls.append((code, entry_point, tests, atol))
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return self.outcome


def _env(arm: Arm, outcome: TestResult | Exception) -> tuple[CodeExecEnv, RecordingSandbox]:
    sandbox = RecordingSandbox(outcome)
    return CodeExecEnv({"Mbpp/3": _problem()}, arm, run=sandbox), sandbox


def test_prompt_is_the_training_prompt() -> None:
    env, _ = _env("V1", _result(1))
    [record] = training_records({"Mbpp/3": _problem()})
    assert env.prompt("Mbpp/3") == record["prompt"]


def test_v1_is_rewarded_on_visible_tests_only() -> None:
    env, sandbox = _env("V1", _result(1))
    env.step("Mbpp/3", CORRECT)
    [(_, _, tests, _)] = sandbox.calls
    assert _ids(tests) == [100]


def test_v0_is_rewarded_on_visible_plus_split_a_never_b() -> None:
    env, sandbox = _env("V0", _result(6, total=6))
    env.step("Mbpp/3", CORRECT)
    [(_, _, tests, _)] = sandbox.calls  # one execution, not two
    split = split_heldout(_problem())
    a_ids = [cast(int, t.args[0]) for t in split.reward]
    b_ids = {cast(int, t.args[0]) for t in split.measurement}
    assert _ids(tests) == [100, *a_ids]
    assert not set(_ids(tests)) & b_ids  # the measurement never enters a reward


@pytest.mark.parametrize(
    ("outcome", "reward"),
    [
        (_result(3, total=3), 1.0),  # every test passes
        (_result(2, total=3), 0.0),  # any failure
        (_result(0, total=3, timed_out=True), 0.0),
        (_result(0, total=3, crashed=True), 0.0),
    ],
)
def test_only_all_pass_scores_one(outcome: TestResult, reward: float) -> None:
    env, _ = _env("V1", outcome)
    step = env.step("Mbpp/3", CORRECT)
    assert step.reward == reward
    assert step.malformed is False
    assert step.result == outcome


def test_malformed_completion_scores_zero_and_never_reaches_the_sandbox() -> None:
    env, sandbox = _env("V1", _result(1))
    step = env.step("Mbpp/3", MALFORMED)
    assert (step.reward, step.malformed, step.result) == (0.0, True, None)
    assert step.infra_retries == 0
    assert sandbox.calls == []


def test_grades_exactly_what_evaluation_extracts() -> None:
    env, sandbox = _env("V1", _result(1))
    env.step("Mbpp/3", CORRECT + "\nprint('trailing noise')\n")
    [(code, entry_point, _, atol)] = sandbox.calls
    assert code == extract_function(CORRECT + "\nprint('trailing noise')\n", _problem())
    assert (entry_point, atol) == ("add", 0.0)


def test_infra_failure_is_never_scored() -> None:
    env, _ = _env("V1", SandboxInfraError("sandbox failed twice in a row"))
    with pytest.raises(SandboxInfraError):
        env.step("Mbpp/3", CORRECT)


def test_infra_retries_are_reported() -> None:
    env, _ = _env("V1", _result(1, infra_retries=1))
    assert env.step("Mbpp/3", CORRECT).infra_retries == 1


def test_unknown_task_id_raises() -> None:
    env, _ = _env("V1", _result(1))
    with pytest.raises(KeyError, match="Mbpp/999"):
        env.step("Mbpp/999", CORRECT)


def test_unknown_arm_is_rejected() -> None:
    with pytest.raises(ValueError, match="arm must be one of"):
        CodeExecEnv({"Mbpp/3": _problem()}, cast(Arm, "V2"))


@pytest.mark.integration
def test_real_sandbox_v1_rewards_overfit_and_v0_does_not() -> None:
    """The H1 mechanism, end to end: a program that only fits the visible test is
    rewarded by V1's sparse suite and refused by V0's dense one."""
    problems = {"Mbpp/3": _problem()}
    v1, v0 = CodeExecEnv(problems, "V1"), CodeExecEnv(problems, "V0")
    assert v1.step("Mbpp/3", CORRECT).reward == 1.0
    assert v0.step("Mbpp/3", CORRECT).reward == 1.0
    assert v1.step("Mbpp/3", OVERFIT).reward == 1.0
    assert v0.step("Mbpp/3", OVERFIT).reward == 0.0
