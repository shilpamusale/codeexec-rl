# scripts/h0_diagnose_failures.py
"""Section B - diagnose the 5 H0 gate failures (CPU only).

For each failing task: show what the perturbation changed (entry_point, canonical solution
diff, prompt diff), then distinguish the two causes -
    (a) perturbation is NOT semantics-preserving  -> a real instrument bug to fix
    (b) the canonical is order-unstable (gate flaky) -> an ADR-007-style exclusion
by re-running the gate on the ORIGINAL canonical and on the PERTURBED one, and
printing the first failing test with expected vs got.

Run:    python scripts/h0_diagnose_failures.py
"""

from __future__ import annotations

import difflib
import random

from codeexec.perturb.generator import _seed_for
from codeexec.perturb.paraphrase import paraphrase_docstring
from codeexec.perturb.rename import rename_function
from codeexec.rewards.harness import run_tests
from codeexec.tasks.datasets import training_problems

BASE_SEED = 0
SAMPLE_SEED = 0
SAMPLE_N = 100

FAILING = ["Mbpp/794", "Mbpp/737", "Mbpp/599", "Mbpp/734", "Mbpp/630"]


def _select_frozen_sample() -> list:
    all_problems = list(training_problems().values())
    random.seed(SAMPLE_SEED)
    return random.sample(all_problems, SAMPLE_N)


def _tests_of(problem):
    return [(tc.args, tc.expected) for tc in (*problem.visible_tests, *problem.heldout_tests)]


def _run(problem):
    """Run a problem's canonical against its own tests; return TestResult."""
    return run_tests(
        problem.canonical_solution, problem.entry_point, _tests_of(problem), problem.atol
    )


def _first_failing_test(problem) -> str:
    """Find and describe the first test the canonical fails (expected vs got)."""
    ns: dict = {}
    try:
        exec(problem.canonical_solution, ns)  # noqa: S102  (diagnostic only, trusted canonical)
        fn = ns[problem.entry_point]
    except Exception as e:  # canonical doesn't even define/compile
        return f"canonical raised on exec/lookup: {type(e).__name__}: {e}"
    for tc in (*problem.visible_tests, *problem.heldout_tests):
        try:
            got = fn(*tc.args)
        except Exception as e:
            return f"args={tc.args!r} -> raised {type(e).__name__}: {e} (expected {tc.expected!r})"
        if got != tc.expected:
            return f"args={tc.args!r} -> got {got!r}  expected {tc.expected!r}"
    return "no single test failed on this run (→ suggests non-determinism / order-instability)"


def _diff(a: str, b: str, label: str) -> str:
    lines = list(
        difflib.unified_diff(
            a.splitlines(),
            b.splitlines(),
            fromfile=f"{label} (original)",
            tofile=f"{label} (perturbed)",
            lineterm="",
        )
    )
    return "\n".join(lines) if lines else f"  (no change to {label})"


def main() -> None:
    by_id = {p.task_id: p for p in _select_frozen_sample()}

    for tid in FAILING:
        problem = by_id.get(tid)
        print("=" * 70)
        print(tid)
        print("=" * 70)
        if problem is None:
            print("  NOT in the frozen 100 — skipping (unexpected).")
            continue

        # Rebuild the perturbed problem with the SAME rng the generator used.
        rng = random.Random(_seed_for(BASE_SEED, problem.task_id))
        renamed = rename_function(problem, rng)
        perturbed = paraphrase_docstring(renamed, rng)

        print(f"entry_point: {problem.entry_point!r} -> {perturbed.entry_point!r}")
        print(
            f"#visible={len(problem.visible_tests)}  #heldout={len(problem.heldout_tests)}  atol={problem.atol}"
        )

        print("\n--- canonical_solution diff ---")
        print(_diff(problem.canonical_solution, perturbed.canonical_solution, "canonical"))

        print("\n--- prompt diff ---")
        print(_diff(problem.prompt, perturbed.prompt, "prompt"))

        # (b) test: is the ORIGINAL canonical itself flaky? Run it 3x.
        orig_runs = [_run(problem).all_passed for _ in range(3)]
        print(f"\noriginal canonical gate (3 runs): {orig_runs}")

        # (a) test: perturbed canonical — does it pass? which test fails?
        pert_runs = [_run(perturbed).all_passed for _ in range(3)]
        print(f"perturbed canonical gate (3 runs): {pert_runs}")
        print("first failing test (perturbed):", _first_failing_test(perturbed))

        # Verdict hint
        if not all(orig_runs):
            print("→ LIKELY (b): original canonical is itself unstable → ADR-007-style exclusion.")
        elif all(orig_runs) and not any(pert_runs):
            print("→ LIKELY (a): perturbation breaks a stable canonical → instrument BUG to fix.")
        else:
            print("→ MIXED: intermittent — inspect the diffs above closely.")


if __name__ == "__main__":
    main()
