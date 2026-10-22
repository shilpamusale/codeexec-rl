"""Training dataset for the GRPO trainer: one record per training problem.

The trainser needs a table with a `prompt` column, and it forwards every
other column to the reward function by name (ADR-011). So each record carries
exactly two things : the prompt the policy sees, and the `task_id` the environment
uses to look up the problem and its tests (ADR-012). Nothing else goes in. Tests,
reference solutions and the held-out split stay inside the environment, out of
the trainer's data.

Records are the plain dicts, not a `datasets.Dataset`, so this module stays importable
and testable without the training stack (the `train` extra, ADR-011). The launcher converts
them on the box with `Dataset.from_list(training_records(training_problems())`.

"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TypedDict

from codeexec.eval.extract import build_prompt
from codeexec.tasks.problem import Problem


class TrainingRecord(TypedDict):
    """One row of the training table."""

    prompt: str  # zero-shot-plus-stub, locked for the baseline and every arm (ADR-006)
    task_id: str  # forwarded by TRL to the reward function as a keyword argument


def training_records(problems: Mapping[str, Problem]) -> list[TrainingRecord]:
    """Build one training record per problem, sorted by task_id.

    Sorting makes the table identical on every run, whatever order the loader
    returns. Only MBPP problems are accepted: HumanEval+ is the out-of-distribution
    check and must never be trained on (ADR-002), so passing one is an error, not
    something to filter silently.
    """

    ordered = sorted(problems.values(), key=lambda p: p.task_id)
    not_mbpp = [p.task_id for p in ordered if p.source != "mbpp"]
    if not_mbpp:
        raise ValueError(f"training records must be MBPP only (ADR-002); got {not_mbpp}")
    return [TrainingRecord(prompt=build_prompt(p), task_id=p.task_id) for p in ordered]
