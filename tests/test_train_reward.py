"""Tests for the TRL reward adapter over CodeExecEnv (ADR-011, ADR-012, ADR-014).

The sandbox is replaced by a stub, and TRL is not needed: the adapter is called
exactly as GRPOTrainer calls a reward function, with keyword arguments.
"""

from __future__ import annotations

import threading
from typing import Any

import pytest

from codeexec.env import CodeExecEnv
from codeexec.rewards.harness import SandboxInfraError, TestCases, TestResult
from codeexec.tasks.problem import Problem
from codeexec.tasks.problem import TestCase as Case
from codeexec.train.reward import CodeExecReward

CORRECT = "     return a + b\n"
WRONG = "       return a - b\n"
MALFORMED = "   return (a +\n"


def _problem(task_id: str) -> Problem:
    return Problem(
        task_id=task_id,
        source="mbpp",
        prompt=f'"""{task_id}: add two numbers."""',
        entry_point="add",
        canonical_solution="def add(a,b):\n     return a + b\n",
        visible_tests=(Case((2, 3), 5),),
        heldout_tests=tuple(Case((i, i), 2 * i) for i in range(4)),
        atol=0.0,
    )


def _fake_sandbox(code: str, entry_point: str, tests: TestCases, atol: float) -> TestResult:
    """Pass iff the code adds; deterministic per completion, so order is checkable."""
    passed = len(tests) if "a + b" in code else 0
    return TestResult(passed=passed, total=len(tests), timed_out=False, crashed=False)


PROBLEMS = {tid: _problem(tid) for tid in ("Mbpp/2", "Mbpp/3")}
ENV = CodeExecEnv(PROBLEMS, "V1", run=_fake_sandbox)


def _call(
    reward: CodeExecReward, task_ids: list[str], completions: list[str], **extras: Any
) -> list[float]:
    """Call the adapter the way GRPOTrainer does: keywords only, one entry per completion."""
    prompts = [ENV.prompt(tid) for tid in task_ids]
    return reward(prompts=prompts, completions=completions, task_id=task_ids, **extras)


def test_one_reward_per_completion_in_order() -> None:
    rewards = _call(
        CodeExecReward(ENV),
        ["Mbpp/2", "Mbpp/3", "Mbpp/2", "Mbpp/3"],
        [CORRECT, WRONG, MALFORMED, CORRECT],
    )
    assert rewards == [1.0, 0.0, 0.0, 1.0]


def test_logs_malformed_rate_infra_retries_and_grading_time() -> None:
    logged: dict[str, float] = {}
    _call(
        CodeExecReward(ENV),
        ["Mbpp/2", "Mbpp/3", "Mbpp/2", "Mbpp/3"],
        [CORRECT, MALFORMED, MALFORMED, CORRECT],
        log_metric=logged.__setitem__,
    )
    assert logged["codeexec/malformed_rate"] == 0.5
    assert logged["codeexec/infra_retries"] == 0.0
    assert logged["codeexec/grading_seconds"] >= 0.0


def test_accepts_trls_other_keyword_arguments() -> None:
    rewards = _call(
        CodeExecReward(ENV),
        ["Mbpp/2"],
        [CORRECT],
        completion_ids=[[1, 2, 3]],
        trainer_state=None,
        log_extra=lambda column, values: None,
    )
    assert rewards == [1.0]


def test_named_for_the_arm() -> None:
    assert CodeExecReward(ENV).__name__ == "codeexec_V1"


def test_grades_the_batch_concurrently() -> None:
    """Four steps must all be in flight at once to get past the barrier."""
    barrier = threading.Barrier(4, timeout=5)

    def waits_for_the_others(
        code: str, entry_point: str, tests: TestCases, atol: float
    ) -> TestResult:
        barrier.wait()  # raises BrokenBarrierError if the steps ran one at a time
        return _fake_sandbox(code, entry_point, tests, atol)

    env = CodeExecEnv(PROBLEMS, "V1", run=waits_for_the_others)
    reward = CodeExecReward(env, max_workers=4)
    assert _call(reward, ["Mbpp/2"] * 4, [CORRECT] * 4) == [1.0] * 4


def test_infra_failure_stops_the_batch() -> None:
    def broken(code: str, entry_point: str, tests: TestCases, atol: float) -> TestResult:
        raise SandboxInfraError("sandbox failed twice in a row")

    reward = CodeExecReward(CodeExecEnv(PROBLEMS, "V1", run=broken))
    with pytest.raises(SandboxInfraError):
        _call(reward, ["Mbpp/2", "Mbpp/3"], [CORRECT, CORRECT])


def test_length_mismatch_is_an_error_not_a_truncation() -> None:
    with pytest.raises(ValueError, match="batch length mismatch"):
        CodeExecReward(ENV)(
            prompts=[ENV.prompt("Mbpp/2")], completions=[CORRECT, CORRECT], task_id=["Mbpp/2"]
        )


def test_prompt_that_is_not_the_environment_is_rejected() -> None:
    with pytest.raises(ValueError, match="not the environment's prompt"):
        CodeExecReward(ENV)(
            prompts=["some other prompt"], completions=[CORRECT], task_id=["Mbpp/2"]
        )


def test_empty_batch_returns_no_rewards_and_logs_nothing() -> None:
    logged: dict[str, float] = {}
    assert _call(CodeExecReward(ENV), [], [], log_metric=logged.__setitem__) == []
    assert logged == {}
