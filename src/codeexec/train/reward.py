"""The TRL reward function: a thin adapter over CodeExecEnv (ADR-011, ADR-012).

GRPOTrainer calls a reward function once per batch, with keyword arguments:
the prompts, the completions, every other dataset column (here `task_id`), one
entry per completion, plus extras such as `log_metric`. This adapter grades the
batch through the environment and returns one reward per completion, in order.

It adds three things the environment deliberately leaves out, because they are
about the trainer, not the task:

-   Concurrency. Each step blocks on a container for most of a second, so the
    batch is graded in a thread pool. Threads suffice: the work waits on Docker
    subprocesses, not on Python.
-   Logging. Per batch, it logs the malformed rate (ADR-014), the sandbox infra
    retries (ADR-014), and the grading wall time, which decides whether ADR-005's
    warm pool is needed (adopt it if grading exceeds 25% of step time).
-   A guard. Each prompt TRL generated from must be the environment's prompt for
    that task_id; a mismatch would score completions against the wrong problem's
    stub, silently, so it raises instead.

Nothing here imports TRL, so it is testable without the training stack.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor

from codeexec.env import CodeExecEnv, StepResult

# ADR-005: sandbox throughput plateaus near 24 concurrent executions,
# whatever the vCPU count, so more threads than this only add contention.
DEFAULT_MAX_WORKERS = 24

LogMetric = Callable[[str, float], None]


class CodeExecReward:
    """A GRPOTrainer reward function that scores completions with a CodeExecEnv."""

    __name__: str  # TRL names the reward in its logs after this

    def __init__(self, env: CodeExecEnv, max_workers: int = DEFAULT_MAX_WORKERS) -> None:
        self.env = env
        self.max_workers = max_workers
        self.__name__ = f"codeexec_{env.arm}"  # e.g. rewards/codeexec_V1/mean

    def __call__(
        self,
        prompts: Sequence[str],
        completions: Sequence[str],
        task_id: Sequence[str],
        log_metric: LogMetric | None = None,
        **kwargs: object,  # TRL's other extras (trainer_state, log_extra, ...), unused
    ) -> list[float]:
        if not len(prompts) == len(completions) == len(task_id):
            raise ValueError(
                f"batch length mismatch: {len(prompts)} prompts, "
                f"{len(completions)} completions, {len(task_id)} task_ids"
            )
        for prompt, tid in zip(prompts, task_id, strict=True):
            if prompt != self.env.prompt(tid):
                raise ValueError(f"prompt for {tid} is not the environment's prompt")
        if not completions:
            return []

        start = time.perf_counter()
        steps = self._grade(task_id, completions)
        elapsed = time.perf_counter() - start

        if log_metric is not None:
            log_metric("codeexec/malformed_rate", sum(s.malformed for s in steps) / len(steps))
            log_metric("codeexec/infra_retries", float(sum(s.infra_retries for s in steps)))
            log_metric("codeexec/grading_seconds", elapsed)
        return [s.reward for s in steps]

    def _grade(self, task_ids: Sequence[str], completions: Sequence[str]) -> list[StepResult]:
        """Grade the batch concurrently; results come back in input order.
        If any step raises (SandboxInfraError), steps not yet started are
        cancelled and the exception propagates, so the training step stops.
        """
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            try:
                return list(pool.map(self.env.step, task_ids, completions))
            except BaseException:
                pool.shutdown(cancel_futures=True)
                raise
