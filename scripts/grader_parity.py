"""Grader parity check (ADR-004 status update; Phase 4, Section A3)

Grades the saved frozen-100 completions two ways and reports every problem
where the two disagree:

        subprocess      the Phase 3 reference grader: the harness run by the host's
                        Python via stdin, 30 s timeout, no isolation. This is how
                        G_baseline was measured on Colab.
        docker          the training grader: the same harness run in the Phase 1
                        sandbox (run_tests): 10 s timeout, 256 MB, no nextwork,
                        read-only filesystem, non-root.

Both build the identical harness program (build_harness) and read the output with the
identical parser, so a disagreement can only come from the execution environment: the limits,
the isolation, or the Python inside the image. Parity holds when the two agree on visible_pass
and plus_pass for every problem.

The subprocess grader runs model code WITHOUT isolation. That is acceptable only because these
completions are fixed text that has already been inspected; it is a reference for this check and
is deliberately kept out of the package.

    python scripts/grader_parity.py
    python scripts/grader_parity.py --limit 5

Exit code 0 = parity holds, 1 = at least one disagreement.

"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from codeexec.eval.extract import extract_function
from codeexec.eval.grade import RunTestsFn, SuiteGrade, grade_two_suites
from codeexec.rewards.harness import (
    TestCases,
    TestResult,
    _parse_result,
    build_harness,
    run_tests,
)
from codeexec.tasks.datasets import training_problems

SUBPROCESS_TIMEOUT_SECONDS = 30  # what the Phase 3 G_baseline grader used


def subprocess_run_tests(code: str, entry_point: str, tests: TestCases, atol: float) -> TestResult:
    """The Phase 3reference grader, shaped like run_tests so it is a RunTestsFn."""
    total = len(tests)
    harness = build_harness(code, entry_point, tests, atol)

    try:
        out = subprocess.run(
            [sys.executable, "-"],  # program arrives on stdin, as in Phase 3
            input=harness,
            capture_output=True,
            text=True,
            timeout=SUBPROCESS_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        return TestResult(passed=0, total=total, timed_out=True, crashed=False)
    passed, crashed = _parse_result(out.stdout, total)
    return TestResult(passed=passed, total=total, timed_out=False, crashed=crashed)


def _result_dict(result: TestResult | None) -> dict[str, object] | None:
    """A TestResult as plain JSON; None means the suite was not run."""
    if result is None:
        return None
    return {
        "passed": result.passed,
        "total": result.total,
        "timed_out": result.timed_out,
        "crashed": result.crashed,
    }


def _grade_dict(grade: SuiteGrade) -> dict[str, object]:
    return {
        "visible_pass": grade.visible_pass,
        "plus_pass": grade.plus_pass,
        "visible": _result_dict(grade.visible),
        "plus": _result_dict(grade.plus),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Subprocess vs Docker grader parity check.")
    parser.add_argument(
        "--completions", type=Path, default=Path("analysis/completions_frozen100.json")
    )
    parser.add_argument("--record", type=Path, default=Path("analysis/g_baseline_results.json"))
    parser.add_argument("--out", type=Path, default=Path("analysis/grader_parity.json"))
    parser.add_argument("--limit", type=int, default=None, help="grade only the first N")

    args = parser.parse_args()

    completions: dict[str, str] = json.loads(args.completions.read_text())["completions"]
    record: dict[str, dict[str, bool]] = json.loads(args.record.read_text())
    pool = training_problems()
    task_ids = list(completions)[: args.limit]

    graders: dict[str, RunTestsFn] = {"subprocess": subprocess_run_tests, "docker": run_tests}
    seconds = dict.fromkeys(graders, 0.0)
    visible_count = dict.fromkeys(graders, 0)
    plus_count = dict.fromkeys(graders, 0)
    record_diffs = dict.fromkeys(graders, 0)
    disagreements: list[str] = []
    per_problem: dict[str, dict[str, object]] = {}
    malformed = 0

    for i, task_id in enumerate(task_ids, start=1):
        problem = pool[task_id]
        code = extract_function(completions[task_id], problem)
        if code is None:
            # Extraction happens before any grader runs, so both graders "agree"
            # trivially; the record must also say malformed.
            malformed += 1
            per_problem[task_id] = {"malformed": True}
            for name in graders:
                record_diffs[name] += int(not record[task_id]["malformed"])
            print(f"{i:3d}/{len(task_ids)} {task_id:<10} malformed (not graded)")
            continue

        grades: dict[str, SuiteGrade] = {}
        for name, run in graders.items():
            start = time.monotonic()
            grades[name] = grade_two_suites(code, problem, run=run)
            seconds[name] += time.monotonic() - start
            grade = grades[name]
            visible_count[name] += int(grade.visible_pass)
            plus_count[name] += int(grade.plus_pass)
            was = record[task_id]
            was_tuple = (was["visible_pass"], was["plus_pass"], was["malformed"])
            record_diffs[name] += int((grade.visible_pass, grade.plus_pass, False) != was_tuple)

        sub, dock = grades["subprocess"], grades["docker"]
        agree = (sub.visible_pass, sub.plus_pass) == (dock.visible_pass, dock.plus_pass)
        if not agree:
            disagreements.append(task_id)
        per_problem[task_id] = {
            "malformed": False,
            "agree": agree,
            **{name: _grade_dict(g) for name, g in grades.items()},
        }

        def tag(g: SuiteGrade) -> str:
            return f"vis={'P' if g.visible_pass else 'f'} plus={'P' if g.plus_pass else 'f'}"

        flag = "" if agree else "   <-- DISAGREE"
        print(f"{i:3d}/{len(task_ids)} {task_id:<10} sub[{tag(sub)}] docker[{tag(dock)}]{flag}")

    n = len(task_ids)
    print(f"\ngraded {n} problems ({malformed} malformed, not graded by either)")
    print(f"{'grader':<11} {'visible':>8} {'plus':>8} {'vs record':>10} {'seconds':>8}")
    for name in graders:
        print(
            f"{name:<11} {visible_count[name]:>4}/{n:<3} {plus_count[name]:>4}/{n:<3}"
            f" {record_diffs[name]:>6} diff {seconds[name]:>8.1f}"
        )
    print(f"\ndisagreements (subprocess vs docker): {len(disagreements)}")
    for task_id in disagreements:
        print(f"  {task_id}: {json.dumps(per_problem[task_id])}")
    holds = not disagreements
    print(f"PARITY: {'HOLDS' if holds else 'BROKEN'}")

    summary = {
        "problems": n,
        "malformed": malformed,
        "visible": visible_count,
        "plus": plus_count,
        "diffs_vs_g_baseline_record": record_diffs,
        "seconds": {name: round(s, 1) for name, s in seconds.items()},
        "disagreements": disagreements,
        "parity_holds": holds,
        "python": sys.version.split()[0],
        "subprocess_timeout_seconds": SUBPROCESS_TIMEOUT_SECONDS,
    }
    args.out.write_text(json.dumps({"summary": summary, "per_problem": per_problem}, indent=2))
    print(f"wrote {args.out}")
    return 0 if holds else 1


if __name__ == "__main__":
    sys.exit(main())
