"""The pre-registered checks of one training run, computed from its trainer state.

    python -m codeexec.train.checks runs/V1-seed0

Every rule here was fixed before any run (DECISIONS.md), and each is computed from the
per-step log the trainer keeps in every checkpoint (trainer_state.json; ADR-010 status
update, C6), so none is judged by eye. Only training-side signals are read, never the
held-out results the study measures:

-   Collapse (ADR-011 status update, C4). From step 40 on, the average reward over the
    last 20 steps more than 0.1 below the first 20 steps' average, or the average entropy
    below a tenth of theirs. Either moves every run to beta 0.04. A run where neither
    happens is stable.
-   Learning (ADR-011 status update, C5). The last 20 steps' average reward at least 0.05
    above the first 20 steps'. Judged on each arm's first seed.
-   Grading share (ADR-005 status update). Grading time over step time, and the same share
    with no time limit firing, estimated from a straight-line fit. Judged on the first
    stable run.

The report is printed and written to the run's folder as checks.json.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

WINDOW = 20  # steps in each average (ADR-011 status update, C4 and C5)
JUDGED_FROM_STEP = 2 * WINDOW  # from here on, the last 20 steps no longer overlap the first 20
REWARD_DROP = 0.1
ENTROPY_FRACTION = 0.1
LEARNING_RISE = 0.05
GRADING_SHARE_LIMIT = 0.25

LogHistory = Sequence[Mapping[str, Any]]


@dataclass(frozen=True)
class RunChecks:
    """What the checks read from one run, and what they found."""

    steps: int
    reward_collapsed_at: int | None  # the first step at which C4's reward signal fired
    entropy_collapsed_at: int | None  # the first step at which C4's entropy signal fired
    reward_rise: float | None  # C5: the last 20 steps' average reward minus the first 20's
    grading_share: float  # grading time over step time, over the whole run
    grading_share_without_time_limits: float | None  # None when the fit cannot separate them

    @property
    def stable(self) -> bool | None:
        """C4: neither signal fired. None for a run too short to judge."""
        if self.steps < JUDGED_FROM_STEP:
            return None
        return self.reward_collapsed_at is None and self.entropy_collapsed_at is None

    @property
    def learned(self) -> bool | None:
        """C5: the reward rose by at least 0.05. None for a run too short to judge."""
        return None if self.reward_rise is None else self.reward_rise >= LEARNING_RISE

    @property
    def grading_verdict(self) -> str:
        """ADR-005's rule, as its status update reads it."""
        if self.grading_share <= GRADING_SHARE_LIMIT:
            return "within 25%: no change"
        without = self.grading_share_without_time_limits
        if without is None:
            return "over 25%, and time limits cannot be separated: decide and record it"
        if without > GRADING_SHARE_LIMIT:
            return "over 25% even without time limits: adopt the warm pool"
        return "over 25% because of time limits: the budgets stay"


def series(log_history: LogHistory, key: str) -> list[float]:
    """One value per step, in step order. The checks need every step (logging_steps = 1)."""
    entries = [entry for entry in log_history if key in entry]
    steps = [entry["step"] for entry in entries]
    if steps != list(range(1, len(steps) + 1)):
        raise ValueError(
            f"{key!r} is not logged at every step from step 1; logging_steps must be 1"
        )
    return [float(entry[key]) for entry in entries]


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values)


def first_collapse(
    values: Sequence[float], collapsed: Callable[[float, float], bool]
) -> int | None:
    """The first step, from step 40 on, at which `collapsed(last_20, first_20)` holds."""
    first = _mean(values[:WINDOW])
    for end in range(JUDGED_FROM_STEP, len(values) + 1):
        if collapsed(_mean(values[end - WINDOW : end]), first):
            return end
    return None


def grading_share_without_time_limits(
    grading: Sequence[float], time_limited: Sequence[float], step_time: float
) -> float | None:
    """The grading share at a time-limited rate of zero: the intercept of a straight line
    fitted to each step's grading time against its time-limited rate, over step time."""
    if not any(time_limited):
        return _mean(grading) / step_time  # no time limit ever fired
    try:
        fit = statistics.linear_regression(time_limited, grading)
    except statistics.StatisticsError:
        return None  # the rate never varied, so the fit cannot separate it
    return fit.intercept / step_time


def check_run(log_history: LogHistory) -> RunChecks:
    """Every pre-registered check, from a run's per-step log."""
    reward = series(log_history, "reward")
    entropy = series(log_history, "entropy")
    grading = series(log_history, "codeexec/grading_seconds")
    time_limited = series(log_history, "codeexec/time_limited_rate")
    step_time = _mean(series(log_history, "step_time"))
    judged = len(reward) >= JUDGED_FROM_STEP
    return RunChecks(
        steps=len(reward),
        reward_collapsed_at=first_collapse(reward, lambda last, first: last < first - REWARD_DROP),
        entropy_collapsed_at=first_collapse(
            entropy, lambda last, first: last < ENTROPY_FRACTION * first
        ),
        reward_rise=_mean(reward[-WINDOW:]) - _mean(reward[:WINDOW]) if judged else None,
        grading_share=_mean(grading) / step_time,
        grading_share_without_time_limits=grading_share_without_time_limits(
            grading, time_limited, step_time
        ),
    )


def latest_trainer_state(run_dir: Path) -> Path:
    """The trainer state of the run's latest checkpoint, which logs every step up to it."""
    checkpoints = sorted(run_dir.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1]))
    if not checkpoints:
        raise FileNotFoundError(f"{run_dir} has no checkpoint")
    return checkpoints[-1] / "trainer_state.json"


def report(checks: RunChecks, source: Path) -> dict[str, Any]:
    return {
        "source": str(source),
        **asdict(checks),
        "stable": checks.stable,
        "learned": checks.learned,
        "grading_verdict": checks.grading_verdict,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="The pre-registered checks of one run.")
    parser.add_argument("run_dir", type=Path, help="the run's folder, e.g. runs/V1-seed0")
    args = parser.parse_args(argv)

    source = latest_trainer_state(args.run_dir)
    checks = check_run(json.loads(source.read_text())["log_history"])
    result = report(checks, source)
    (args.run_dir / "checks.json").write_text(json.dumps(result, indent=2) + "\n")

    print(f"steps: {checks.steps} (from {source})")
    print(
        f"stable (C4): {checks.stable}; reward collapsed at {checks.reward_collapsed_at}, "
        f"entropy collapsed at {checks.entropy_collapsed_at}"
    )
    print(f"learned (C5): {checks.learned}; reward rise {checks.reward_rise}")
    without = checks.grading_share_without_time_limits
    print(
        f"grading share (ADR-005): {checks.grading_share:.1%}, without time limits "
        f"{'cannot be separated' if without is None else f'{without:.1%}'}: "
        f"{checks.grading_verdict}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
