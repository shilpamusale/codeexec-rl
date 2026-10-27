# scripts/h0_check_sample.py

"""Section B - H0 instrument readiness check (CPU only).

Confirms, on the EXACT frozen 100 that H0 will run on, that the Section D perturbation
generator (a) gates cleanly and (b) actually changes the surface the model see. Prints
the three numbers Phase 3 Section B needs. No model, no GPU.

Run: python scripts/h0_check_sample.py
"""

from __future__ import annotations

import random

from codeexec.perturb.generator import perturb
from codeexec.tasks.datasets import training_problems

BASE_SEED = 0  #   pre-registered: H0 perturnbation base seed (ADR-002 status)
SAMPLE_SEED = 0  # MUST match the smoke-test notebook exactly
SAMPLE_N = 100
NOOP_FLAG_THRESHOLD = 0.20  # advisory only (not pre-registered): weak-leverage flag


def _select_frozen_sample() -> list:
    """Replace the smoke-test cell byte-for-byte. Nothing random may run
    before random.sample, or the 100 selected would differ."""
    all_problems = list(training_problems().values())
    random.seed(SAMPLE_SEED)
    return random.sample(all_problems, SAMPLE_N)


def main() -> None:
    sample = _select_frozen_sample()
    assert len(sample) == SAMPLE_N, f"expected {SAMPLE_N}, got {len(sample)}"

    gated_ok = 0
    gate_failures: list[str] = []
    changed = 0
    noops: list[str] = []

    for problem in sample:
        perturbed = perturb(problem, BASE_SEED)
        if perturbed is None:
            gate_failures.append(problem.task_id)
            continue
        gated_ok += 1

        # A change the model can see: prompt text (rename's assert rewrite OR
        # paraphrase's lead-in/synonym rewrite) or the entry_point name.
        if perturbed.prompt != problem.prompt or perturbed.entry_point != problem.entry_point:
            changed += 1
        else:
            noops.append(problem.task_id)

    print("=" * 56)
    print(f"B1  gated OK:       {gated_ok}/{SAMPLE_N}")
    print(f"B1  gate failures:  {len(gate_failures)}    {gate_failures}")
    print(f"B2  changed:        {changed}/{gated_ok}")
    print(f"B2  no-ops:         {len(noops)}    {noops}")

    print(
        "GATE     :",
        "OK - all 100 survived"
        if not gate_failures
        else f" {len(gate_failures)} failed - investigate before proceeding (Decision A)",
    )
    noop_rate = len(noops) / gated_ok if gated_ok else 1.0
    print(
        "CHANGE:",
        "OK - leverage healthy"
        if noop_rate <= NOOP_FLAG_THRESHOLD
        else f" {noop_rate: .0%} no-ops - weak leverage; plan to report changed-subset too.",
    )


if __name__ == "__main__":
    main()
