"""Reward functions for the RL training loop.
Both are thin wrappers over run_tests: they run a code string against a chosen
test set in the Phase 1 sandbox and turn the TestResult into a scalar reward.

Design(locked):
    - Arm-agnostic: the caller passes the `tests` to grade against. V1 passes the
      visible tests; V0 passes the visible tests plus held-out split A (ADR-013).
      The arm is a training-config choice, not baked into the reward. (decision a)
"""

from __future__ import annotations

from codeexec.rewards.harness import TestCases, run_tests


def r_sparse(code: str, entry_point: str, tests: TestCases, atol: float) -> float:
    """Binary terminal reward: 1.0 iff ALL tests pass, else 0.0.

    This is the H1 reward. V0 and V1 use the SAME function; they differ only in
    which test set the caller passes as `tests`. A timeout or crash yields 0.0
    because TestResult.all_passed is False in those cases.
    """
    result = run_tests(code, entry_point, tests, atol)
    return 1.0 if result.all_passed else 0.0


def r_shaped(
    code: str,
    entry_point: str,
    tests: TestCases,
    atol: float,
    run_bonus: float = 0.0,
) -> float:
    """Dense reward: fraction of tests passing, plus an optional run bonus.

    This is the H2 reward. The base signal is passed/total. `run_bonus` (default
    0.0 = off) adds a small constant when the code RAN without crashing, even if
    it passed 0 tests, so "runs but wrong" can score above "doesn't parse". It
    is a hyperparameter that affects H2's result, so it is explicit and defaults
    off. A timeout or crash yields 0.0 (no bonus — the code did not run cleanly).
    """
    result = run_tests(code, entry_point, tests, atol)

    if result.timed_out or result.crashed:
        return 0.0

    reward = result.fraction
    if run_bonus and not result.all_passed:
        # the code ran (not timed out, not crashed) -> eligible for the bonus.
        reward = max(reward, run_bonus)
    return reward
