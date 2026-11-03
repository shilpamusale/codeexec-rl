"""Tests for ΔG over a run's checkpoints (THESIS; ADR-011 status update, C5).

Synthetic reports in the shape measure_model writes: the reference as a regrade,
with its generation settings nested under meta["generation"], and the checkpoints
as fresh measurements, with theirs at the top of meta.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from codeexec.eval.delta_g import InstrumentError, instrument_differences, main, summarize_run

PROBLEMS = ["Mbpp/2", "Mbpp/3", "Mbpp/4"]

GENERATION = {
    "engine": "vllm 0.31.0",
    "dtype": "bfloat16",
    "decoding": "greedy (temperature 0), stop at end-of-sequence only",
    "max_new_tokens": 256,
    "prompt": "zero-shot-plus-stub (build_prompt), as in training",
}

GRADER = "docker run_tests, EvalPlus's pass rule (ADR-015)"


def _summary(gap_pp: float, gap_full_pp: float | None = None) -> dict[str, Any]:
    return {
        "n": 368,
        "visible": 251,
        "plus_b": 207,
        "gap_pp": gap_pp,
        "gap_full_pp": gap_pp if gap_full_pp is None else gap_full_pp,
    }


def _measurement(gap_pp: float, **meta: Any) -> dict[str, Any]:
    """A checkpoint's report: generation settings at the top of meta."""
    return {
        "meta": {**GENERATION, "grader": GRADER, "git": {"commit": "abc", "dirty": False}, **meta},
        "all_problems": _summary(gap_pp),
        "per_problem": {tid: {} for tid in PROBLEMS},
    }


REFERENCE = {
    "meta": {"grader": GRADER, "git": {"commit": "abc", "dirty": False}, "generation": GENERATION},
    "all_problems": _summary(12.0, gap_full_pp=12.5),
    "per_problem": {tid: {} for tid in PROBLEMS},
}


def test_delta_g_is_the_last_step_minus_the_reference() -> None:
    steps = {50: 13.0, 100: 15.0, 150: 14.0, 200: 16.5}
    summary = summarize_run({s: _measurement(g) for s, g in steps.items()}, REFERENCE, 200)
    assert summary["delta_g_pp"] == pytest.approx(4.5)
    assert summary["delta_g_full_pp"] == pytest.approx(4.0)  # against the full siote's 12.5
    deltas = [row["delta_g_pp"] for row in summary["trajectory"].values()]
    assert deltas == pytest.approx([1.0, 3.0, 2.0, 4.5])  # the trajectory, in step order


def test_g_final_needs_the_last_steps() -> None:
    with pytest.raises(InstrumentError, match="no measurement at step 200"):
        summarize_run({50: _measurement(13.0), 100: _measurement(15.0)}, REFERENCE, 200)


@pytest.mark.parametrize(
    ("meta", "found"),
    [
        ({"max_new_tokens": 512}, "max_new_tokens"),
        ({"dtype": "float16"}, "dtype"),
        ({"engine": "vllm 0.32.0"}, "engine"),
        ({"grader": "docker run_tests"}, "grader"),
        ({"git": {"commit": "def", "dirty": False}}, "graded at commit def"),
        ({"git": {"commit": "abc", "dirty": True}}, "uncommitted changes"),
    ],
)
def test_a_measurement_on_another_instrument_is_refused(meta: dict[str, Any], found: str) -> None:
    (difference,) = instrument_differences(_measurement(16.5, **meta), REFERENCE)
    assert found in difference
    with pytest.raises(InstrumentError, match="step 200"):
        summarize_run({200: _measurement(16.5, **meta)}, REFERENCE, 200)


def test_a_measurement_of_other_problems_is_refused() -> None:
    other = {**_measurement(16.5), "per_problem": {tid: {} for tid in PROBLEMS[:2]}}
    assert instrument_differences(other, REFERENCE) == [
        "the problems measured differ from the reference's"
    ]


def test_the_command_writes_the_summary_or_refuses(tmp_path: Path) -> None:
    measure = tmp_path / "V1-seed0" / "measure"
    measure.mkdir(parents=True)
    reference = tmp_path / "reference.json"
    reference.write_text(json.dumps(REFERENCE))
    for step, gap in ((100, 15.0), (200, 16.5)):
        (measure / f"step-{step}.json").write_text(json.dumps(_measurement(gap)))

    assert main([str(tmp_path / "V1-seed0"), "--reference", str(reference)]) == 0
    written = json.loads((measure / "delta_g.json").read_text())
    assert written["delta_g_pp"] == pytest.approx(4.5)
    assert list(written["trajectory"]) == ["100", "200"]  # JSON keys are strings

    (measure / "delta_g.json").unlink()
    (measure / "step-200.json").write_text(json.dumps(_measurement(16.5, dtype="float16")))
    assert main([str(tmp_path / "V1-seed0"), "--reference", str(reference)]) == 2
    assert not (measure / "delta_g.json").exists()  # refused: nothing written
