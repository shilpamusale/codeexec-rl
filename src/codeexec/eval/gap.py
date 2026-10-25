"""The H1 measurement: the gap G between visible and held-out pass rates.

Given one completion per problem, this extracts each function exactly as
training does, grades it in the sandbox, and summarizes (THESIS, ADR-004):

    visible_pass        every visible test passes
    plus_b              visible AND held-out split B pass: heldout_pass, the headline
    plus_full           visible AND the full held-out suite: the citable MBPP+ number

    G = visible_pass - heldout_pass

with a bootstrap 95% CI computes exactly as Phase 3 did: paired per-problem
differences, numpy default_rng(0), 20,000 resamples, 2.5/97.5 percentiles. The bootstrap
draws problems by position, so results keep the order of the completions they came from.

This module grades and computes; it does not generate. Completions from Colab, from vLLM on
the box, or from a training checkpoint are all measured by the same code, so delta-G is a
subtraction on one instrument.
"""

from __future__ import annotations

from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np

from codeexec.eval.extract import extract_function
from codeexec.eval.grade import RunTestsFn, grade_two_suites
from codeexec.rewards.harness import run_tests
from codeexec.tasks.problem import Problem
from codeexec.tasks.split import split_heldout

BOOTSTRAP_SEED = 0  # identical to analysis/g_baseline.md
BOOTSTRAP_RESAMPLES = 20_000
DEFAULT_MAX_WORKERS = 24  # ADR-005's sandbox throughput plateau


@dataclass(frozen=True)
class ProblemResult:
    """How one completion fared. A malformed completion fails everything."""

    malformed: bool
    visible_pass: bool
    plus_b: bool  # visible AND split B: what heldout_pass counts
    plus_full: bool  # visible AND the full held-out suite


@dataclass(frozen=True)
class GapSummary:
    """The measurement over a set of problems; rates and gaps in percentage points."""

    n: int
    malformed: int
    visible: int
    plus_b: int
    plus_full: int
    gap_pp: float  # G on split B: the headline
    ci_pp: tuple[float, float]
    gap_full_pp: float  # G on the full held-out suite
    ci_full_pp: tuple[float, float]


def grade_problem(completion: str, problem: Problem, run: RunTestsFn = run_tests) -> ProblemResult:
    """Extract and grade one completion on all three suites."""
    code = extract_function(completion, problem)
    if code is None:
        return ProblemResult(malformed=True, visible_pass=False, plus_b=False, plus_full=False)
    on_b = grade_two_suites(code, problem, run, heldout=split_heldout(problem).measurement)
    on_full = grade_two_suites(code, problem, run=run)  # heldout=None: the full suite
    return ProblemResult(
        malformed=False,
        visible_pass=on_b.visible_pass,
        plus_b=on_b.plus_pass,
        plus_full=on_full.plus_pass,
    )


def measure(
    completions: Mapping[str, str],
    problems: Mapping[str, Problem],
    run: RunTestsFn = run_tests,
    max_workers: int = DEFAULT_MAX_WORKERS,
) -> dict[str, ProblemResult]:
    """Grade every completion concurrently; results keep the completion's order."""
    task_ids = list(completions)
    unknown = [tid for tid in task_ids if tid not in problems]

    if unknown:
        raise KeyError(f"completions for unknown problems: {unknown}")

    def grade(tid: str) -> ProblemResult:
        return grade_problem(completions[tid], problems[tid], run)

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        return dict(zip(task_ids, pool.map(grade, task_ids), strict=True))


def bootstrap_ci(diffs: np.ndarray) -> tuple[float, float]:
    """95% percentile CI of the mean difference, in pp, with Phase 3's exact draws."""
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    n = len(diffs)
    boot = np.array([diffs[rng.integers(0, n, n)].mean() for _ in range(BOOTSTRAP_RESAMPLES)])
    lo, hi = np.percentile(boot, [2.5, 97.5]) * 100
    return float(lo), float(hi)


def summarize(results: Mapping[str, ProblemResult]) -> GapSummary:
    """Counts, G on split B and on the full suite, and their bootstrap CIs."""
    if not results:
        raise ValueError("no results to summarize")
    rows = list(results.values())
    visible = np.array([int(r.visible_pass) for r in rows])
    plus_b = np.array([int(r.plus_b) for r in rows])
    plus_full = np.array([int(r.plus_full) for r in rows])
    return GapSummary(
        n=len(rows),
        malformed=sum(r.malformed for r in rows),
        visible=int(visible.sum()),
        plus_b=int(plus_b.sum()),
        plus_full=int(plus_full.sum()),
        gap_pp=float((visible - plus_b).mean() * 100),
        ci_pp=bootstrap_ci(visible - plus_b),
        gap_full_pp=float((visible - plus_full).mean() * 100),
        ci_full_pp=bootstrap_ci(visible - plus_full),
    )


def disagreements(
    results: Mapping[str, ProblemResult], record: Mapping[str, Mapping[str, bool]]
) -> list[str]:
    """Problems whose (visible_pass, plus_b) differ from an earlier per-problem record.

    `record` maps task_id to {"visible_pass": ..., "plus_pass": ...}, the format
    of analysis/g_baseline_split_b.json. Only problems in both are compared.
    """
    return [
        tid
        for tid, r in results.items()
        if tid in record
        and (r.visible_pass, r.plus_b) != (record[tid]["visible_pass"], record[tid]["plus_pass"])
    ]
