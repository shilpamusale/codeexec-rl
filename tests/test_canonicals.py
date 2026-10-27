"""ADR-015's validation step 1: every canonical solution passes all its own tests.

Each expected output was computed by running the canonical solution on that
test's input, so a correct pipeline grades every canonical as passing every
test, visible and held-out alike, in the real sandbox. A failure means the pass
rule, the harness or the loader's ground truth is wrong, not the model.

All tests run in one program per problem: stricter than training or measurement,
which grade subsets of them, so passing here implies passing there.

Linux is the platform of record (ADR-015): expected values are computed on the
host and graded in the Linux sandbox, so the check of record runs on a Linux host,
where both use glibc. Elsewhere, only EXACT_ON_LINUX_ONLY may fail.
"""

from __future__ import annotations

import sys
from concurrent.futures import ThreadPoolExecutor

import pytest

from codeexec.rewards.harness import TestResult, run_tests
from codeexec.tasks.datasets import training_problems
from codeexec.tasks.problem import Problem

# Few enough that CPU contention on a laptop cannot push a slow canonical past
# ADR-008's time limits, which would be a false alarm.
MAX_WORKERS = 4

# Problems whose expected value needs bit-exact floats from the platform's math
# library, so they pass only where the host's library is the sandbox's:
#   Mbpp/590: cmath.polar's hypot; on one of its 104 tests, macOS and glibc
#             differ by one unit in the last place.
EXACT_ON_LINUX_ONLY = frozenset({"Mbpp/590"})


def _grade_canonical(problem: Problem) -> TestResult:
    tests = [(t.args, t.expected) for t in (*problem.visible_tests, *problem.heldout_tests)]
    return run_tests(problem.canonical_solution, problem.entry_point, tests, problem.atol)


def _describe(result: TestResult) -> str:
    flags = [name for name in ("timed_out", "crashed") if getattr(result, name)]
    return f"{result.passed}/{result.total}" + "".join(f" {flag}" for flag in flags)


@pytest.mark.integration
@pytest.mark.dataset
def test_every_canonical_passes_all_its_own_tests() -> None:
    problems = training_problems()
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        results = dict(zip(problems, pool.map(_grade_canonical, problems.values()), strict=True))
    failing = {tid: _describe(r) for tid, r in results.items() if not r.all_passed}
    if sys.platform != "linux":
        failing = {tid: d for tid, d in failing.items() if tid not in EXACT_ON_LINUX_ONLY}
    assert not failing, f"{len(failing)} of {len(results)} canonicals fail: {failing}"
