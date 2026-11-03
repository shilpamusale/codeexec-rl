"""ΔG for one run: each checkpoint's measurement against the reference.

    python -m codeexec.eval.delta_g runs/V1-seed0 --reference runs/adr015/g_baseline_box_adr015.json

ΔG = G_final - G_baseline (THESIS). G_final is measured at the run's last step, 200
(ADR-011 status update, C5), and G_baseline is the reference the gate re-derives on the
box at the start of the session (ADR-008 status update, C1). The checkpoints before it
give the trajectory, which is reported and never used in place of the last step.

Each checkpoint's report is read from <run>/measure/step-<N>.json, as measure_model
writes it. A subtraction means something only on one instrument, so every report must
match the reference in how it generated and graded: the same engine, dtype, decoding,
token budget, prompt and grader, the same problems, and the same commit of this
repository, with no uncommitted changes. Any difference is refused, never resolved away.
The summary is printed and written to <run>/measure/delta_g.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from codeexec.train.config import DEFAULT_CONFIG, load_config

# How the completions were generated. A regrade keeps these under meta["generation"].
GENERATION_FIELDS = ("engine", "dtype", "decoding", "max_new_tokens", "prompt")

Report = Mapping[str, Any]


class InstrumentError(ValueError):
    """A measurement and the reference were not taken on the same instrument."""


def _generation(meta: Mapping[str, Any]) -> Mapping[str, Any]:
    """The generation settings: a regrade's are nested, a fresh measurement's are not."""
    generation: Mapping[str, Any] = meta.get("generation", meta)
    return generation


def instrument_differences(report: Report, reference: Report) -> list[str]:
    """Every way `report` was measured differently from `reference`; empty if none."""
    meta, ref_meta = report["meta"], reference["meta"]
    differences = [
        f"{field}: {_generation(meta).get(field)!r}, the reference's "
        f"{_generation(ref_meta).get(field)!r}"
        for field in GENERATION_FIELDS
        if _generation(meta).get(field) != _generation(ref_meta).get(field)
    ]
    if meta["grader"] != ref_meta["grader"]:
        differences.append(f"grader: {meta['grader']!r}, the reference's {ref_meta['grader']!r}")
    if meta["git"]["dirty"] or ref_meta["git"]["dirty"]:
        differences.append("measured with uncommitted changes")
    if meta["git"]["commit"] != ref_meta["git"]["commit"]:
        differences.append(
            f"graded at commit {meta['git']['commit']}, the reference at "
            f"{ref_meta['git']['commit']}"
        )
    if set(report["per_problem"]) != set(reference["per_problem"]):
        differences.append("the problems measured differ from the reference's")
    return differences


def summarize_run(
    measurements: Mapping[int, Report], reference: Report, final_step: int
) -> dict[str, Any]:
    """The trajectory of G and ΔG, step by step, and ΔG at the final step."""
    if final_step not in measurements:
        raise InstrumentError(f"no measurement at step {final_step}, where G_final is defined")
    for step, report in measurements.items():
        differences = instrument_differences(report, reference)
        if differences:
            raise InstrumentError(f"step {step}: " + "; ".join(differences))

    baseline = reference["all_problems"]
    trajectory = {}
    for step in sorted(measurements):
        summary = measurements[step]["all_problems"]
        trajectory[step] = {
            "n": summary["n"],
            "visible": summary["visible"],
            "plus_b": summary["plus_b"],
            "gap_pp": summary["gap_pp"],
            "delta_g_pp": summary["gap_pp"] - baseline["gap_pp"],
            "delta_g_full_pp": summary["gap_full_pp"] - baseline["gap_full_pp"],
        }
    return {
        "final_step": final_step,
        "reference": {key: baseline[key] for key in ("n", "visible", "plus_b", "gap_pp")},
        "delta_g_pp": trajectory[final_step]["delta_g_pp"],
        "delta_g_full_pp": trajectory[final_step]["delta_g_full_pp"],
        "trajectory": trajectory,
    }


def load_measurements(measure_dir: Path) -> dict[int, Report]:
    """Every step-<N>.json in the run's measure folder, by step."""
    return {
        int(path.stem.removeprefix("step-")): json.loads(path.read_text())
        for path in measure_dir.glob("step-*.json")
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="ΔG for one run, checkpoint by checkpoint.")
    parser.add_argument("run_dir", type=Path, help="the run's folder, e.g. runs/V1-seed0")
    parser.add_argument("--reference", required=True, type=Path, help="the ΔG reference")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args(argv)

    measure_dir = args.run_dir / "measure"
    reference = json.loads(args.reference.read_text())
    final_step = load_config(args.config).grpo["max_steps"]
    try:
        summary = summarize_run(load_measurements(measure_dir), reference, final_step)
    except InstrumentError as error:
        print(f"ΔG refused: {error}", file=sys.stderr)
        return 2
    summary["reference_file"] = str(args.reference)
    (measure_dir / "delta_g.json").write_text(json.dumps(summary, indent=2) + "\n")

    ref = summary["reference"]
    print(
        f"reference: n={ref['n']} visible {ref['visible']} plus_b {ref['plus_b']} "
        f"G = {ref['gap_pp']:+.1f} pp ({args.reference})"
    )
    for step, row in summary["trajectory"].items():
        print(
            f"step {step:>4}: visible {row['visible']} plus_b {row['plus_b']} "
            f"G = {row['gap_pp']:+.1f} pp, ΔG = {row['delta_g_pp']:+.1f} pp"
        )
    print(
        f"ΔG at step {final_step}: {summary['delta_g_pp']:+.1f} pp "
        f"(full suite {summary['delta_g_full_pp']:+.1f} pp)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
