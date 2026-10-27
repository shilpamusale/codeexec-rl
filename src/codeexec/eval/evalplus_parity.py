"""ADR-015's validation step 2: parity with EvalPlus's own evaluator.
    python -m codeexec.eval.evalplus_parity --completions analysis/completions_frozen100.json \\
        --out analysis/evalplus_parity_frozen100.json
    
    python -m codeexec.eval.evalplus_parity --completions analysis/g_baseline_box.json \\
        --out analysis/evalplus_parity_box.json

Saved completions are graded two ways, on the same extracted function, and compared problem
by problem:
    ours        the Docker sandbox under oracle.py's port of Evalplus's rule, on 
                the loader's ground truth (grade_two_suites, gap.py's full-suite path)
    evalplus    EvalPlus 0.3.1's own evaluator, called as `evalplus.evaluate` calls it
                (check_correctness), on EvalPlus's own ground truth (get_groundtruth),
                with its default time limits and parallelism

Parity is checked on the visible tests (EvalPlus's "base") and on the full suite
(base AND plus), the citable MBPP+ number; EvalPlus has no split B. evaluate() 
itself cannot be called: it requires every MBPP+ problem, and the set has 368.
Completions for problems outside the set (Mbpp/255) are set aside and counted.

ADR-015 expects one source of disagreement, the two system's time limits: ours 
are 2 s per test and 10 s per program (ADR-008), EvalPlus's max(1 s, 4 x the 
canonical's time) per test. So each disagreement is graded again by EvalPlus with 
generous limits, and put down to time limits if that agrees with ours, or if our
program timed out. Any other is unexplained: under ADR-015 the port is then fixed
before any training run.

Evalplus runs the model's code outside the sandbox, in its own guarded subprocess, 
as it always does. These are fixed, saved completions, and the check of record runs 
on the disposable training box (Linux, the platform of record).

Exit Code 0 : no unexplained disagreement. 1: at least one. 2: refused to run.               
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import platform
import sys
from collections.abc import Mapping
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any

from evalplus.config import DEFAULT_GT_TIME_LIMIT_FACTOR, DEFAULT_MIN_TIME_LIMIT
from evalplus.evaluate import check_correctness

from codeexec.eval.extract import extract_function
from codeexec.eval.grade import RunTestsFn, SuiteGrade, grade_two_suites
from codeexec.rewards.harness import run_tests
from codeexec.tasks.datasets import training_problems
from codeexec.tasks.problem import Problem
from codeexec.train.launch import git_state

# The re-check that tells a time-limit disagreement from a real one: limits
# generous enough that no completion that finishes at all is cut off.
RELAXED_MIN_TIME_LIMIT = 20.0
RELAXED_TIME_LIMIT_FACTOR = 100.0
DEFAULT_MAX_WORKERS = 24  # ADR-005's sandbox plateau


@dataclass(frozen=True)
class Verdict:
    """One grader's answer for one completion."""

    visible: bool  # every visible test passes: EvalPlus's "base"
    full: bool  # visible AND every held-out test: EvalPlus's "base" and "plus"


@dataclass(frozen=True)
class EvalPlusGrade:
    """EvalPlus's statuses ("pass", "fail" or "timeout") and the failing tests."""

    base_status: str
    plus_status: str
    base_failed: tuple[int, ...]
    plus_failed: tuple[int, ...]

    @property
    def verdict(self) -> Verdict:
        visible = self.base_status == "pass"
        return Verdict(visible=visible, full=visible and self.plus_status == "pass")


def ours_verdict(grade: SuiteGrade) -> Verdict:
    return Verdict(visible=grade.visible_pass, full=grade.plus_pass)


def _failed(details: list[bool], n: int) -> tuple[int, ...]:
    """Tests that failed, or never ran because EvalPlus stopped the run."""
    return tuple(i for i in range(n) if i >= len(details) or not details[i])


def evalplus_grade(
    code: str,
    record: Mapping[str, Any],
    expected: Mapping[str, Any],
    min_time_limit: float = DEFAULT_MIN_TIME_LIMIT,
    time_limit_factor: float = DEFAULT_GT_TIME_LIMIT_FACTOR,
) -> EvalPlusGrade:
    """Grade `code` as `evalplus.evaluate --test-details grades one sample.

    --test-details runs every test instead of stopping at the first failure, so
    the failing tests can be named; the pass-or-fail verdict is the same.
    """

    result = check_correctness(
        "mbpp",
        0,
        record,
        code,
        expected,
        base_only=False,
        fast_check=False,
        min_time_limit=min_time_limit,
        gt_time_limit_factor=time_limit_factor,
    )
    base_status, base_details = result["base"]
    plus_status, plus_details = result["plus"]

    return EvalPlusGrade(
        base_status=base_status,
        plus_status=plus_status,
        base_failed=_failed(base_details, len(record["base_input"])),
        plus_failed=_failed(plus_details, len(record["plus_input"])),
    )


def classify(ours: SuiteGrade, evalplus: EvalPlusGrade, relaxed: EvalPlusGrade | None) -> str:
    """`agree`, `time limits` or `unexplained`, as the module docstring defines them."""
    mine = ours_verdict(ours)
    if mine == evalplus.verdict:
        return "agree"
    ours_timed_out = ours.visible.timed_out or (ours.plus is not None and ours.plus.timed_out)
    if ours_timed_out or (relaxed is not None and relaxed.verdict == mine):
        return "time limits"
    return "unexplained"


def evalplus_ground_truth() -> tuple[dict[str, Any], dict[str, Any]]:
    """MBPP+ records and EvalPlus's own expected outputs, as evaluate() build them."""
    from evalplus.data import get_mbpp_plus, get_mbpp_plus_hash
    from evalplus.eval import MBPP_OUTPUT_NOT_NONE_TASKS
    from evalplus.evaluate import get_groundtruth

    records = get_mbpp_plus()
    return records, get_groundtruth(records, get_mbpp_plus_hash(), MBPP_OUTPUT_NOT_NONE_TASKS)


def compare(
    completions: Mapping[str, str],
    problems: Mapping[str, Problem],
    records: Mapping[str, Any],
    expected: Mapping[str, Any],
    run: RunTestsFn = run_tests,
    max_workers: int = DEFAULT_MAX_WORKERS,
    evalplus_workers: int | None = None,
) -> dict[str, Any]:
    """Grade every completion both ways; summarize, and keep each problem's detail."""
    in_set = [tid for tid in completions if tid in problems]
    set_aside = [tid for tid in completions if tid not in problems]
    extracted = {tid: extract_function(completions[tid], problems[tid]) for tid in in_set}
    codes = {tid: code for tid, code in extracted.items() if code is not None}
    graded = list(codes)
    # Ours first, then EvalPlus, so neither grader's load slows the other.
    with ThreadPoolExecutor(max_workers=max_workers) as threads:
        ours = dict(
            zip(
                graded,
                threads.map(lambda tid: grade_two_suites(codes[tid], problems[tid], run), graded),
                strict=True,
            )
        )

    # EvalPlus's own default parallelism (evalplus.evaluate: half the CPUs).
    workers = evalplus_workers or max(1, multiprocessing.cpu_count() // 2)
    with ProcessPoolExecutor(max_workers=workers) as processes:
        theirs = dict(
            zip(
                graded,
                processes.map(
                    evalplus_grade,
                    [codes[tid] for tid in graded],
                    [records[tid] for tid in graded],
                    [expected[tid] for tid in graded],
                ),
                strict=True,
            )
        )

    per_problem: dict[str, Any] = {}
    for tid in graded:
        grade, relaxed = ours[tid], None
        if ours_verdict(grade) != theirs[tid].verdict:
            relaxed = evalplus_grade(
                codes[tid],
                records[tid],
                expected[tid],
                RELAXED_MIN_TIME_LIMIT,
                RELAXED_TIME_LIMIT_FACTOR,
            )
        per_problem[tid] = {
            "outcome": classify(grade, theirs[tid], relaxed),
            "ours": {
                **asdict(ours_verdict(grade)),
                "visible_tests": asdict(grade.visible),
                "full_tests": None if grade.plus is None else asdict(grade.plus),
            },
            "evalplus": {**asdict(theirs[tid].verdict), **asdict(theirs[tid])},
            "relaxed": None if relaxed is None else asdict(relaxed),
        }

    def ids(outcome: str) -> list[str]:
        return [tid for tid, p in per_problem.items() if p["outcome"] == outcome]

    unexplained = ids("unexplained")
    summary = {
        "completions": len(completions),
        "set_aside": set_aside,
        "malformed": [tid for tid in in_set if tid not in codes],  # graded by neither
        "graded": len(graded),
        "ours": {
            "visible": sum(ours_verdict(ours[tid]).visible for tid in graded),
            "full": sum(ours_verdict(ours[tid]).full for tid in graded),
        },
        "evalplus": {
            "visible": sum(theirs[tid].verdict.visible for tid in graded),
            "full": sum(theirs[tid].verdict.full for tid in graded),
        },
        "agree": len(ids("agree")),
        "time_limits": ids("time limits"),
        "unexplained": unexplained,
        "parity_holds": not unexplained,
        "evalplus_workers": workers,
    }
    return {"summary": summary, "per_problem": per_problem}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Parity with EvalPlus's evaluator (ADR-015).")
    parser.add_argument("--completions", required=True, type=Path, help="saved completions JSON")
    parser.add_argument("--out", required=True, type=Path, help="where to write the JSON report")
    parser.add_argument("--max-workers", type=int, default=DEFAULT_MAX_WORKERS)
    parser.add_argument("--evalplus-workers", type=int, help="default: EvalPlus's own")
    parser.add_argument("--overwrite", action="store_true", help="replace an existing report")
    parser.add_argument("--allow-dirty", action="store_true", help="run with uncommitted changes")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    refusal = None
    if args.out.exists() and not args.overwrite:
        refusal = f"{args.out} exists; pass --overwrite to replace it"
    git = git_state()
    if git["dirty"] and not args.allow_dirty:
        refusal = "the working tree has uncommitted changes; commit or --allow-dirty"
    if refusal is not None:
        print(f"parity check refused: {refusal}", file=sys.stderr)
        return 2
    if sys.platform == "darwin":
        # macOS cannot lower a process's address-space limit, which EvalPlus's
        # guard sets; with -1 it skips that. A preview only: Linux is of record.
        os.environ.setdefault("EVALPLUS_MAX_MEMORY_BYTES", "-1")

    completions: dict[str, str] = json.loads(args.completions.read_text())["completions"]
    records, expected = evalplus_ground_truth()
    result = compare(
        completions,
        training_problems(),
        records,
        expected,
        max_workers=args.max_workers,
        evalplus_workers=args.evalplus_workers,
    )
    meta = {
        "completions": str(args.completions),
        "ours": "docker run_tests under oracle.py (ADR-015), loader ground truth",
        "evalplus": f"evalplus {version('evalplus')} check_correctness, its own ground truth",
        "evalplus_time_limits": {
            "default": {"min_s": DEFAULT_MIN_TIME_LIMIT, "factor": DEFAULT_GT_TIME_LIMIT_FACTOR},
            "relaxed": {"min_s": RELAXED_MIN_TIME_LIMIT, "factor": RELAXED_TIME_LIMIT_FACTOR},
        },
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "git": git,
        "created_at": datetime.now(UTC).isoformat(),
    }
    report = {"meta": meta, **result}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")

    s = result["summary"]
    print(
        f"graded {s['graded']} of {s['completions']} "
        f"(set aside {s['set_aside']}; malformed {len(s['malformed'])})"
    )
    for name in ("ours", "evalplus"):
        print(f"{name:<9} visible {s[name]['visible']:>4}  full {s[name]['full']:>4}")
    print(
        f"agree {s['agree']}; time limits {len(s['time_limits'])} {s['time_limits']}; "
        f"unexplained {len(s['unexplained'])} {s['unexplained']}"
    )
    print(f"PARITY: {'HOLDS' if s['parity_holds'] else 'BROKEN'}")
    print(f"wrote {args.out}")
    return 0 if s["parity_holds"] else 1


if __name__ == "__main__":
    sys.exit(main())
