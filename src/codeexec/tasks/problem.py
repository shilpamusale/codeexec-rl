"""The normalized problem representation for the task layer.

EvalPlus provides MBPP+ and HumalEval+ in two different record formats.
This module defines one uniform `Problem` type that the rest of the pipeline -
the partition, the reward functions, the perturbation generator - consumes without
knowing which dataset a problem came from. The loaders (in loader.py) absorb the format
differences and produce these.

Tests are stored in complete (args, expected) pairs: the expected output is computed once,
at load time, by running the cannonical solution on each input
(the pre-compute decision). A problem is therefore self-contained - it can be
graded without re-running the cannonical solution.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class TestCase:
    """One complete test: the arguments to call the function with,
    and the expected output (computed from the canonical solution at load time).
    """

    args: tuple[object, ...]
    expected: object


@dataclass(frozen=True)
class Problem:
    """A normalized programming problem, uniform across MBPP+ and HumanEval+.

    Immutable (frozen): problems are reference data. The perturbation generator
    produces new Problem objects rather than mutating existing ones.
    """

    task_id: str
    source: str  # 'mbpp' or 'humaneval'
    prompt: str  # what the model sees, in the dataset's native format
    entry_point: str
    canonical_solution: str
    visible_tests: tuple[TestCase, ...]
    heldout_tests: tuple[TestCase, ...]
    atol: float
