"""Re-derive G_baseline on held-out split B (ADR-013; Phase 4, Section B).

Everything here was pre-registered in ADR-013 before it ran. The saved
frozen-100 completions are graded by the Docker sandbox, and

    G_B = visible_pass - plus_B     (plus_B = visible AND split B pass)

gets a bootstrap 95% CI compouted exactly as analysis/g_baseline.md did:
paired per-problem differences, numpy default_rng(0), 20,000 resamples,
2.5/97.5 percentiles. Then the dynamic-range rule decides:
    CI lower bound > 0      the 50/50 split stands
    otherwise               move B to two-thirds and re-check once
    still includes zero     STOP: no training until the V0 design is revisited

The full held-out suite is graded too, as a consistency check: it must
reporduce the Phase 3 record problem by problem (visible 65, plus 55)
and the Phase 3 CI [+5.0, +16.0], or the comparison is not like for like.

    python scripts/rebaseline_split_b.py

Exit code = 0 = a split (proceed); 1 = STOP or a consistency failure.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Callable, Sequence
from fractions import Fraction
from pathlib import Path

import numpy as np

from codeexec.eval.extract import extract_function
from codeexec.eval.grade import RunTestsFn, grade_two_suites
from codeexec.rewards.harness import run_tests
from codeexec.tasks.datasets import training_problems
from codeexec.tasks.problem import Problem, TestCase
from codeexec.tasks.split import B_FRACTION, SPLIT_VERSION, split_heldout

COMPLETIONS = Path("analysis/completions_frozen100.json")
RECORD = Path("analysis/g_baseline_results.json")
OUT = Path("analysis/g_baseline_split_b.json")
FALLBACK_B_FRACTION = Fraction(2, 3)  # ADR-013's pre-registered fallback
BOOTSTRAP_SEED = 0  # identical to analysis/g_baseline.md
BOOTSTRAP_RESAMPLES = 20_000
GRADER: RunTestsFn = run_tests  # the Docker sandbox

# Chooses the held-out tests for a problem; None means the full held-out suite.
HeldoutChoice = Callable[[Problem], Sequence[TestCase] | None]


def full_suite(problem: Problem) -> None:
    """Choose the full held-out suite (the standard MBPP+ plus)."""
    return None


def split_b(fraction: Fraction) -> HeldoutChoice:
    """Return a chooser for help-out split B at `fraction` (ADR-013)"""

    def choose(problem: Problem) -> Sequence[TestCase]:
        return split_heldout(problem, fraction).measurement

    return choose


def bootstrap_ci(diffs: np.ndarray) -> tuple[float, float]:
    """95% percentile CI in pp, with the same RNG, draws and order as Phase 3."""
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    n = len(diffs)
    boot = np.array([diffs[rng.integers(0, n, n)].mean() for _ in range(BOOTSTRAP_RESAMPLES)])
    lo, hi = np.percentile(boot, [2.5, 97.5]) * 100
    return float(lo), float(hi)


def measure(
    label: str, choose: HeldoutChoice, codes: dict[str, str | None], pool: dict[str, Problem]
) -> dict[str, object]:
    """Grade every completiom on visible and on visible + chosen held-out tests."""
    per_problem: dict[str, dict[str, bool]] = {}
    for i, (task_id, code) in enumerate(codes.items(), start=1):
        if code is None:
            per_problem[task_id] = {"visible_pass": False, "plus_pass": False}
            continue
        problem = pool[task_id]
        grade = grade_two_suites(code, problem, run=GRADER, heldout=choose(problem))
        per_problem[task_id] = {"visible_pass": grade.visible_pass, "plus_pass": grade.plus_pass}
        if i % 20 == 0:
            print(f" [{label}] graded {i}/{len(codes)}")
    vis = np.array([int(r["visible_pass"]) for r in per_problem.values()])
    plus = np.array([int(r["plus_pass"]) for r in per_problem.values()])
    lo, hi = bootstrap_ci(vis - plus)
    gap = float((vis.mean() - plus.mean()) * 100)
    print(
        f"{label:<14} visible {vis.sum()}/{len(vis)}  plus {plus.sum()}/{len(plus)}  "
        f"G = {gap:+.1f} pp  95% CI [{lo:+.1f}, {hi:+.1f}]"
    )
    return {
        "visible": int(vis.sum()),
        "plus": int(plus.sum()),
        "G_pp": round(gap, 1),
        "ci_pp": [round(lo, 1), round(hi, 1)],
        "per_problem": per_problem,
    }


def main() -> int:
    completions: dict[str, str] = json.loads(COMPLETIONS.read_text())["completions"]
    record: dict[str, dict[str, bool]] = json.loads(RECORD.read_text())
    pool = training_problems()
    codes = {tid: extract_function(c, pool[tid]) for tid, c in completions.items()}
    print(f"{len(codes)} completions, {sum(c is None for c in codes.values())} malformed")

    # 1. Consistency: the full held-out suite must reproduce the Phase 3 record.
    full = measure("full suite", full_suite, codes, pool)
    full_pp = full["per_problem"]
    assert isinstance(full_pp, dict)
    mismatches = [
        tid
        for tid, r in record.items()
        if (full_pp[tid]["visible_pass"], full_pp[tid]["plus_pass"])
        != (r["visible_pass"], r["plus_pass"])
    ]
    consistent = not mismatches and full["ci_pp"] == [5.0, 16.0]
    print(
        f"consistency with Phase 3 record: {'OK' if consistent else 'FAILED'}"
        f"({len(mismatches)} per-problem mismatches, CI {full['ci_pp']})"
    )

    # 2. The pre-registered measurement on split B, with the dynamic-range rule.
    results: dict[str, object] = {"full_suite": full}
    verdict = "STOP"
    for fraction in (B_FRACTION, FALLBACK_B_FRACTION):
        label = f"split B ({fraction})"
        result = measure(label, split_b(fraction), codes, pool)
        results[label] = result
        ci = result["ci_pp"]
        assert isinstance(ci, list)
        if ci[0] > 0:
            verdict = f"split B = {fraction} STANDS (CI lower bound {ci[0]:+.1f} pp > 0)"
            break
        print(f"    CI includes zero at B = {fraction}; applying pre-registered fallback")

    print(f"\n VERDICT: {verdict}")
    OUT.write_text(
        json.dumps(
            {
                "meta": {
                    "split_version": SPLIT_VERSION,
                    "bootstrap_seed": BOOTSTRAP_SEED,
                    "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
                    "grader": "docker run_tests",
                    "completions": str(COMPLETIONS),
                    "consistent_with_phase3": consistent,
                },
                "verdict": verdict,
                **results,
            },
            indent=2,
        )
    )
    print(f"wrote   {OUT}")
    return 0 if consistent and verdict != "STOP" else 1


if __name__ == "__main__":
    sys.exit(main())
