"""The RL environment (ADR-012): a prompt goes out, a completion comes back,
and the environment scores it.

Episodes are single-shot. The observation is a problem's prompt, the action is
one completion, and the reward ends the episode. The environment is stateless:
the task is an argument to `step`, not state set by a reset, so one instance can
grade a whole GRPO group concurrently, from many threads.

Each arm's reward suite is fixed when the environment is built (ADR-013):

    V1: the visible tests only          (sparse, ~3 tests per problem)
    V0: the visible tests + held-out A  (dense, ~55 tests per problem)

Held-out split B never enters any reward, in any arm, and is never touched here.
Every outcome has one fixed score (ADR-014): 1.0 only if every test in the
reward suite passes, else 0.0. A malformed completion scores 0.0 without being
sent to the sandbox, and is flagged so the trainer can log the malformed rate.
A sandbox infrastructure failure is never scored: run_tests retries it once,
then raises, the exception passes straight through `step`.

The environment knows nothing about the trainer. TRL reaches it through a thin
adapter, so another trainer can reuse it unchanged.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal, get_args

from codeexec.eval.extract import build_prompt, extract_function
from codeexec.eval.grade import RunTestsFn
from codeexec.rewards.functions import sparse_score
from codeexec.rewards.harness import TestCases, TestResult, run_tests
from codeexec.tasks.problem import Problem
from codeexec.tasks.split import split_heldout

Arm = Literal["V0", "V1"]
ARMS: tuple[Arm, ...] = get_args(Arm)


@dataclass(frozen=True)
class StepResult:
    """The outcome of one step: the reward, and what the trainer logs about it."""

    reward: float
    malformed: bool  # the completion did not extract into a function, so never ran
    result: TestResult | None  # the sandbox outcome; None when malformed

    @property
    def infra_retries(self) -> int:
        """Sandbox retries this step needed (ADR-014); 0 if it never ran."""
        return 0 if self.result is None else self.result.infra_retries


def _reward_suite(problem: Problem, arm: Arm) -> TestCases:
    """The (args, expected) pairs `arm` is rewarded on (ADR-013, ADR-014)."""
    tests = problem.visible_tests
    if arm == "V0":
        tests = (*tests, *split_heldout(problem).reward)
    return [(t.args, t.expected) for t in tests]


class CodeExecEnv:
    """A stateless, single-shot code-generation environment for one training arm."""

    def __init__(
        self, problems: Mapping[str, Problem], arm: Arm, run: RunTestsFn = run_tests
    ) -> None:
        if arm not in ARMS:
            raise ValueError(f"arm must be one of {ARMS}, got {arm!r}")
        self.arm = arm
        self._problems = dict(problems)
        self._run = run
        # Built once and only read afterwards, so threads can share it safely.
        self._suites = {tid: _reward_suite(p, arm) for tid, p in self._problems.items()}

    def prompt(self, task_id: str) -> str:
        """The observation: the locked zero-shot-plus-stub prompt (ADR-006)."""
        return build_prompt(self._problem(task_id))

    def step(self, task_id: str, completion: str) -> StepResult:
        """Score one completion for `task_id`.
        Extraction is the same function the evaluation uses, so training
        rewards exactly the program that is later measured.
        """
        problem = self._problem(task_id)
        code = extract_function(completion, problem)
        if code is None:
            return StepResult(reward=0.0, malformed=True, result=None)
        result = self._run(code, problem.entry_point, self._suites[task_id], problem.atol)
        return StepResult(reward=sparse_score(result), malformed=False, result=result)

    def _problem(self, task_id: str) -> Problem:
        try:
            return self._problems[task_id]
        except KeyError:
            raise KeyError(f"unknown task_id {task_id!r}") from None
