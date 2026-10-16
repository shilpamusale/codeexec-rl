"""Perturbation driver: compose rename + paraphrase into one perturbed Problem,
then validate that it is semantics-preserving by running its canonical through
the sandbox. A perturnbation whose canonical mo longer passes its own tests is
rejected (returns None) -- see ADR/DESIGN.
"""

from __future__ import annotations

import hashlib
import logging
import random

from codeexec.perturb.paraphrase import paraphrase_docstring
from codeexec.perturb.rename import rename_function
from codeexec.rewards.harness import run_tests
from codeexec.tasks.problem import Problem

logger = logging.getLogger(__name__)


def _seed_for(base_seed: int, task_id: str) -> int:
    """Deterministic per-problem seed from (base_seed, task_id), so the same
    base seed perturbs the whole set reprobucibly but each problem differently."""

    h = hashlib.sha256(f"{base_seed}:{task_id}".encode()).hexdigest()
    return int(h[:8], 16)


def _gate(problem: Problem) -> bool:
    """True iff the problem's canonical passes 100% of its own visible+heldout
    tests in the sandbox. A semantics-preserving perturbation cannot fail this.
    """
    tests = [(tc.args, tc.expected) for tc in (*problem.visible_tests, *problem.heldout_tests)]
    result = run_tests(problem.canonical_solution, problem.entry_point, tests, problem.atol)
    return result.all_passed


def perturb(problem: Problem, base_seed: int) -> Problem | None:
    """Return a semantics-preserving perturbation of `problem`, or None if
    the perturbation failed the validation gate (and should be dropped)
    """
    rng = random.Random(_seed_for(base_seed, problem.task_id))
    perturbed = rename_function(problem, rng)
    perturbed = paraphrase_docstring(perturbed, rng)
    if not _gate(perturbed):
        logger.warning(
            "dropping %s: perturbed canonical failed its own tests (gate)", problem.task_id
        )
        return None
    return perturbed
