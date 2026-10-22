"""The held-out split (ADR-013): reward split A and measurement split B.

Each problem's held-out tests are divided once, by a fixed rule, into two
disjoint parts:

    A, the reward split       enters only V0's reward
    B, the measurement split  heldout_pass and G are measured on it; it never
                              enters any reward, in any arm

The rule is pre-registered in ADR-013: rank the held-out test indices by
SHA-256 of "heldout-split-v1:<task_id>:<index>", put the first ceil(n * f) in
B (f = 1/2), and the rest in A. SHA-256 gives the same ranking on every machine
and Python version, unlike random.shuffle, so the split is reproducible forever
from the task id alone. Each part keeps the tests' loaded order.

Changing SPLIT_VERSION or the rule changes which tests are held out; that is a
new decision with its own ADR, never an edit. tests/test_split.py pins golden
values so an accidental change fails loudly.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from fractions import Fraction

from codeexec.tasks.problem import Problem, TestCase

SPLIT_VERSION = "heldout-split-v1"
# ADR-013: B gets half. The pre-registered fallback, if G_baseline on B has no
# dynamic range, is Fraction(2, 3). A Fraction keeps ceil(n * f) exact.
B_FRACTION = Fraction(1, 2)


@dataclass(frozen=True)
class HeldoutSplit:
    """A problem's held-out tests, divided into the reward and measurement splits."""

    reward: tuple[TestCase, ...]  # A: enters only V0's reward
    measurement: tuple[TestCase, ...]  # B: never enters any reward, in any arm


def _rank_key(task_id: str, index: int) -> str:
    """The sort key that decides a test's place in the ranking (ADR-013)."""
    return hashlib.sha256(f"{SPLIT_VERSION}:{task_id}:{index}".encode()).hexdigest()


def split_heldout(problem: Problem, b_fraction: Fraction = B_FRACTION) -> HeldoutSplit:
    """Split `problem`'s held-out tests into A (reward) and B (measurement)."""
    if not 0 < b_fraction < 1:
        raise ValueError(f"b_fraction must be between 0 and 1, got {b_fraction}")
    n = len(problem.heldout_tests)
    ranked = sorted(range(n), key=lambda i: _rank_key(problem.task_id, i))
    in_b = set(ranked[: math.ceil(n * b_fraction)])
    tests = problem.heldout_tests
    return HeldoutSplit(
        reward=tuple(t for i, t in enumerate(tests) if i not in in_b),
        measurement=tuple(t for i, t in enumerate(tests) if i in in_b),
    )
