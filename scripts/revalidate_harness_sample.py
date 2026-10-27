# scripts/revalidate_harness_sample.py
"""Sample re-validation of the harness fix (memory-safe for a constrained box).

Runs the own-canonical gate IN-PROCESS on a random sample of normal problems
plus the watched set. 60 normal problems passing = the fix did not regress
grading (a grading regression shows up on nearly all of them). The watched set
shows each flagged problem's post-fix status. Avoids the full-374 load cliff.

Run:  python scripts/revalidate_harness_sample.py
"""

from __future__ import annotations

import random

from codeexec.rewards.harness import run_tests
from codeexec.tasks.datasets import training_problems

SAMPLE_N = 60
WATCH = ["Mbpp/599", "Mbpp/630", "Mbpp/794", "Mbpp/737", "Mbpp/734"]


def gate(p) -> tuple[int, int, bool, bool]:
    tests = [(tc.args, tc.expected) for tc in (*p.visible_tests, *p.heldout_tests)]
    r = run_tests(p.canonical_solution, p.entry_point, tests, p.atol)
    return r.passed, r.total, r.timed_out, r.crashed


def main() -> None:
    problems = training_problems()
    watch_set = set(WATCH)
    normal_ids = [t for t in problems if t not in watch_set]

    random.seed(0)
    sample = random.sample(normal_ids, min(SAMPLE_N, len(normal_ids)))

    print(f"--- {len(sample)} normal problems (expect ALL pass) ---")
    normal_fail = []
    for tid in sample:
        passed, total, to, cr = gate(problems[tid])
        if not (total > 0 and passed == total):
            normal_fail.append((tid, passed, total, to, cr))
    if normal_fail:
        print(f"  REGRESSION: {len(normal_fail)} normal problems FAILED:")
        for tid, p, t, to, cr in normal_fail:
            print(f"    {tid}: {p}/{t} timed_out={to} crashed={cr}")
    else:
        print(f"  OK: all {len(sample)} normal problems pass (no grading regression)")

    print(f"\n--- watched problems (diagnostic) ---")
    for tid in WATCH:
        if tid not in problems:
            print(f"  {tid}: not loaded")
            continue
        passed, total, to, cr = gate(problems[tid])
        ok = total > 0 and passed == total
        print(
            f"  {tid}: {passed}/{total} timed_out={to} crashed={cr}  "
            f"{'PASS' if ok else 'FAIL -> exclude'}"
        )

    print("\nVerdict: normal-sample all-pass => fix is safe to keep.")
    print("Watched FAILs => honest exclusions for the documented reasons.")


if __name__ == "__main__":
    main()
