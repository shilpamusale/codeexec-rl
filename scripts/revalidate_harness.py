# scripts/revalidate_harness.py
"""Re-validate the harness after the ADR-008 per-case-timeout fix — robustly.

Runs each loaded problem's own-canonical gate in a SEPARATE subprocess with a
hard wall-clock, so one pathological problem (e.g. an oversized-output canonical)
shows up as a timeout/crash ROW instead of Terminating the whole run. Reports
every problem whose canonical no longer passes 100% of its own tests.

Expected after the fix:
  - the pre-fix passing set is unchanged (no regression);
  - 599 now PASSES (its one slow stress case is per-case-timed-out, counted as a
    single failure... wait: 599's CANONICAL is the slow one, so it still fails
    that case -> see note below);
  - oversized/slow canonicals (630, ...) show as fail/timeout rows -> honest exclusions.

Run:  python scripts/revalidate_harness.py
"""

from __future__ import annotations

import json
import subprocess
import sys

from codeexec.tasks.datasets import training_problems

PER_PROBLEM_WALLCLOCK = 20  # hard cap per problem subprocess (seconds)

WATCH = {"Mbpp/599", "Mbpp/630", "Mbpp/794", "Mbpp/737", "Mbpp/734"}

# Child program: load ONE problem, run its canonical gate, print a JSON verdict.
# Kept as source so each problem runs in full isolation (its own memory/clock).
_CHILD = r"""
import json, sys
from codeexec.tasks.datasets import training_problems
from codeexec.rewards.harness import run_tests

tid = sys.argv[1]
p = training_problems()[tid]
tests = [(tc.args, tc.expected) for tc in (*p.visible_tests, *p.heldout_tests)]
r = run_tests(p.canonical_solution, p.entry_point, tests, p.atol)
print("###V###" + json.dumps({
    "tid": tid, "passed": r.passed, "total": r.total,
    "timed_out": r.timed_out, "crashed": r.crashed, "all": r.all_passed,
}))
"""


def check_one(tid: str) -> dict:
    try:
        out = subprocess.run(
            [sys.executable, "-c", _CHILD, tid],
            capture_output=True,
            text=True,
            timeout=PER_PROBLEM_WALLCLOCK,
        )
    except subprocess.TimeoutExpired:
        return {"tid": tid, "status": "WALLCLOCK_TIMEOUT", "all": False}
    for line in out.stdout.splitlines():
        if line.startswith("###V###"):
            d = json.loads(line[len("###V###") :])
            d["status"] = "ok"
            return d
    # child died without a verdict (OOM-killed, crash): returncode tells us
    return {
        "tid": tid,
        "status": f"CHILD_DIED(rc={out.returncode})",
        "all": False,
        "stderr_tail": out.stderr[-200:],
    }


def main() -> None:
    tids = list(training_problems().keys())
    print(
        f"re-validating {len(tids)} problems, each in its own subprocess "
        f"(cap {PER_PROBLEM_WALLCLOCK}s)...\n"
    )

    failures = []
    for i, tid in enumerate(tids):
        v = check_one(tid)
        if not v.get("all", False):
            failures.append(v)
        if (i + 1) % 50 == 0:
            print(f"  ...{i + 1}/{len(tids)} done, {len(failures)} failing so far")

    print(f"\n{'=' * 60}")
    print(f"canonicals FAILING their own gate after fix: {len(failures)}/{len(tids)}")
    for v in sorted(failures, key=lambda d: d["tid"]):
        mark = "  <-- watched" if v["tid"] in WATCH else ""
        if v["status"] == "ok":
            print(
                f"  {v['tid']}: {v['passed']}/{v['total']} "
                f"timed_out={v['timed_out']} crashed={v['crashed']}{mark}"
            )
        else:
            print(f"  {v['tid']}: {v['status']}{mark}")
    print(f"{'=' * 60}")
    print("Interpretation:")
    print("  - SMALL failing list (watched + a few) = expected; exclude those honestly.")
    print("  - MANY new failures = the fix regressed something: STOP, do not exclude.")


if __name__ == "__main__":
    main()
