"""Find exactly how the full 91-case harness fails for Mbpp/599.

test[0] alone passes 1/1, but the whole 91-case suite scored 0/91. This runs the
FULL generated harness as a subprocess and captures stdout/stderr/returncode, so we
see the real failure (traceback? or sentinel? wrong count?). Then it bisects to
find the single case. that breaks the suite, if that's the shape of it.

Run:  python scripts/h0_probe_599_suite.py
"""

from __future__ import annotations

import subprocess
import sys

from codeexec.rewards.harness import RESULT_SENTINEL, build_harness
from codeexec.tasks.datasets import training_problems

TID = "Mbpp/599"


def run_harness(code: str, entry: str, tests: list, atol: float):
    h = build_harness(code, entry, tests, atol)
    try:
        out = subprocess.run(
            [sys.executable, "-"], input=h, capture_output=True, text=True, timeout=15
        )
    except subprocess.TimeoutExpired:
        return None, "", "<<TIMEOUT>>", -1
    sentinel = any(ln.startswith(RESULT_SENTINEL) for ln in out.stdout.splitlines())
    return sentinel, out.stdout, out.stderr, out.returncode


def main() -> None:
    p = training_problems()[TID]
    all_tests = [(tc.args, tc.expected) for tc in (*p.visible_tests, *p.heldout_tests)]
    print(f"{TID}  entry={p.entry_point}  n_tests={len(all_tests)}")

    # 1. run the FULL suite, see what actually happens
    sentinel, stdout, stderr, rc = run_harness(
        p.canonical_solution, p.entry_point, all_tests, p.atol
    )
    print(f"\nfull suite: returncode={rc}  sentinel_printed={sentinel}")
    print("  stdout (last 300 chars):", repr(stdout[-300:]))
    print("  stderr (last 600 chars):", repr(stderr[-600:]))

    # 2. if no sentinel / crash, find the first case that breaks tit by growing prefix
    if not sentinel:
        print("\nbisecting: smallest prefix pf cases that kills the sentinel...")
        lo, hi = 1, len(all_tests)
        bad_at = None
        # linear-ish: find the first K where tests[:K] fails to print sentinel
        for k in (1, 2, 5, 10, 20, 40, 60, 80, len(all_tests)):
            if k > len(all_tests):
                break
            s, _, _, _ = run_harness(p.canonical_solution, p.entry_point, all_tests[:k], p.atol)
            print(f"    prefix[:{k:3d}] sentinel = {s}")

            if not s and bad_at is None:
                bad_at = k
                break

        # narroe to the exact offending index with the failing window
        if bad_at is not None:
            start = (
                0
                if bad_at <= 1
                else [1, 2, 5, 10, 20, 40, 60, 80][
                    max(0, [1, 2, 5, 10, 20, 40, 60, 80, len(all_tests)].index(bad_at) - 1)
                ]
            )

            for i in range(start, bad_at):
                s, _, err, _ = run_harness(
                    p.canonical_solution, p.entry_point, [all_tests[i]], p.atol
                )
                if not s:
                    a, e = all_tests[i]
                    print(f"\n >>offending case idx={i}: args={a!r} expected={e!r}")
                    print(f"    single-case stderr: {err[-300:]!r}")
    else:
        # ran but mis-graded: show the count
        print(f" suite ran; inspect the count in stdout above vs expected all-pass")


if __name__ == "__main__":
    main()
