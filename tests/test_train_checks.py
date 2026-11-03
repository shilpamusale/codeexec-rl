"""Tests for the pre-registered run checks (ADR-005, ADR-011 status updates).

Synthetic per-step logs in the shape TRL writes to trainer_state.json, so the
checks are tested without a GPU or a training run.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from codeexec.train.checks import check_run, main


def _log(
    reward: Sequence[float],
    entropy: Sequence[float] | None = None,
    grading: Sequence[float] | None = None,
    time_limited: Sequence[float] | None = None,
    step_time: float = 20.0,
) -> list[dict[str, Any]]:
    """One entry per step, as TRL logs it with logging_steps = 1, then the run summary."""
    n = len(reward)
    entries: list[dict[str, Any]] = [
        {
            "step": step + 1,
            "reward": reward[step],
            "entropy": 0.5 if entropy is None else entropy[step],
            "codeexec/grading_seconds": 4.0 if grading is None else grading[step],
            "codeexec/time_limited_rate": 0.0 if time_limited is None else time_limited[step],
            "step_time": step_time,
        }
        for step in range(n)
    ]
    entries.append({"step": n, "train_runtime": n * step_time, "train_loss": 0.0})
    return entries


def test_a_steadily_learning_run_is_stable_and_learned() -> None:
    checks = check_run(_log([0.30 + 0.10 * step / 199 for step in range(200)]))
    assert (checks.steps, checks.stable, checks.learned) == (200, True, True)
    assert checks.reward_rise == pytest.approx(0.0905, abs=1e-3)


def test_a_flat_reward_fails_the_learning_check() -> None:
    checks = check_run(_log([0.35] * 200))
    assert (checks.stable, checks.learned) == (True, False)


def test_reward_collapse_fires_at_the_first_window_past_the_drop() -> None:
    """From step 101 the reward is 0.17, not 0.4: the 20-step average is more than 0.1
    below the first 20 steps' once 9 of its steps are past the drop, at step 109.
    """
    checks = check_run(_log([0.4] * 100 + [0.17] * 100))
    assert (checks.reward_collapsed_at, checks.entropy_collapsed_at) == (109, None)
    assert checks.stable is False


def test_entropy_collapse_fires_below_a_tenth() -> None:
    """From step 61 the entropy is 0.05, not 1.0: the 20-step average falls below 0.1
    once 19 of its steps are past the drop, at step 79.
    """
    checks = check_run(_log([0.35] * 200, entropy=[1.0] * 60 + [0.05] * 140))
    assert (checks.reward_collapsed_at, checks.entropy_collapsed_at) == (None, 79)
    assert checks.stable is False


def test_a_run_under_40_stable_is_not_judged() -> None:
    checks = check_run(_log([0.4] * 20 + [0.0] * 19))
    assert (checks.stable, checks.learned, checks.reward_collapsed_at) == (None, None, None)


def test_grading_share_with_and_without_time_limits() -> None:
    """Steps where a time fired take longer to grad: 4 s at a rate of 0, 8 s at
    0.1. Over 20 s steps that is 30% in all, and 20% at a rate of zero.
    """
    rates = [0.0, 0.1] * 100
    checks = check_run(
        _log([0.35] * 200, grading=[4.0 + 40 * r for r in rates], time_limited=rates)
    )
    assert checks.grading_share == pytest.approx(0.30)
    assert checks.grading_share_without_time_limits == pytest.approx(0.20)
    assert checks.grading_verdict == "over 25% because of time limits: the budgets stay"


def test_grading_share_over_the_limit_without_time_limits_means_the_warm_pool() -> None:
    checks = check_run(_log([0.35] * 200, grading=[8.0] * 200))
    assert checks.grading_share_without_time_limits == pytest.approx(0.40)  # no time limits fired
    assert checks.grading_verdict == "over 25% even without time limits: adopt the warm pool"


def test_a_time_limited_rate_that_never_varies_cannot_be_separated() -> None:
    checks = check_run(_log([0.35] * 200, grading=[8.0] * 200, time_limited=[0.05] * 200))
    assert checks.grading_share_without_time_limits is None
    assert checks.grading_verdict.startswith("over 25%, and time limits cannot be separated")


def test_the_checks_need_every_step() -> None:
    log = _log([0.35] * 50)
    del log[10]
    with pytest.raises(ValueError, match="logging_steps must be 1"):
        check_run(log)


def test_the_command_reads_the_latest_checkpoint_and_writes_the_report(tmp_path: Path) -> None:
    for step, log in ((50, _log([0.35] * 50)), (100, _log([0.35] * 100))):
        (tmp_path / f"checkpoint-{step}").mkdir()
        (tmp_path / f"checkpoint-{step}" / "trainer_state.json").write_text(
            json.dumps({"log_history": log})
        )
    assert main([str(tmp_path)]) == 0
    written = json.loads((tmp_path / "checks.json").read_text())
    assert written["steps"] == 100  # checkpoint-100, not checkpoint-50
    assert (written["stable"], written["learned"]) == (True, False)
