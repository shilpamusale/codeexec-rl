# scripts/h0_confirm_599_630.py
"""Confirm cause of Mbpp/599 & Mbpp/630 gate failures — minimal output."""

from __future__ import annotations

from codeexec.rewards.harness import run_tests
from codeexec.tasks.datasets import training_problems

TARGETS = ["Mbpp/599", "Mbpp/630"]


def short(x: object, n: int = 80) -> str:
    s = repr(x)
    return s if len(s) <= n else s[:n] + f"...<{type(x).__name__},len~{len(s)}>"


def main() -> None:
    problems = training_problems()
    for tid in TARGETS:
        p = problems.get(tid)
        print("=" * 50)
        print(tid, "entry:", getattr(p, "entry_point", None))
        if p is None:
            continue
        tests = [(tc.args, tc.expected) for tc in (*p.visible_tests, *p.heldout_tests)]
        print(f"  n_tests={len(tests)} atol={p.atol}")

        whole = [
            run_tests(p.canonical_solution, p.entry_point, tests, p.atol).passed for _ in range(3)
        ]
        print(f"  whole-suite passed 3x (of {len(tests)}): {whole}")

        # find first failing idx only
        first = None
        for i, (a, e) in enumerate(tests):
            if not run_tests(p.canonical_solution, p.entry_point, [(a, e)], p.atol).all_passed:
                first = i
                break
        if first is None:
            print("  no single-test failure; whole-suite fails => interaction/order")
            continue

        a, e = tests[first]
        reruns = [
            run_tests(p.canonical_solution, p.entry_point, [(a, e)], p.atol).all_passed
            for _ in range(5)
        ]
        print(f"  first FAIL idx={first} exp_type={type(e).__name__}")
        print(f"    expected={short(e)}")
        print(f"    args={short(a)}")
        print(
            f"    rerun 5x: {reruns}  "
            f"{'MIXED->order-unstable' if len(set(reruns)) > 1 else 'STABLE'}"
        )
    print("=" * 50)


if __name__ == "__main__":
    main()
