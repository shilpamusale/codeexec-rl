"""Perturbed dataset builder: apply the validated perturbation to the training
set and return the survivors. This is the H0 instrument's output -- the perturbed
counterpart to the training_problems(), keyed the same task_id so the two sets
align 1:1 for the canonical-vs-perturbed pass-rate comparison.
"""

from __future__ import annotations

import logging

from codeexec.perturb.generator import perturb
from codeexec.tasks.datasets import training_problems
from codeexec.tasks.problem import Problem

logger = logging.getLogger(__name__)


def perturbed_problems(base_seed: int) -> dict[str, Problem]:
    """return semantics-preserving perturnbations of the training problems
    that passed the validation gate, keyed by the original task_id. Problems
    whose perturnbation failed the gate are dropeerd and logged."""

    source = training_problems()
    out: dict[str, Problem] = {}

    for task_id, problem in source.items():
        perturbed = perturb(problem, base_seed)
        if perturbed is not None:
            out[task_id] = perturbed
    logger.info(
        "perturbed %d/%d training problems (base_seed=%d)", len(out), len(source), base_seed
    )
    return out
