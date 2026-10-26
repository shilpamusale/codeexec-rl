"""Tests for the model-measurement command (B7), without a GPU.

Generation needs vLLM on the box; everything around it, the report,
the frozen-100 subset, the engine check and the refusals, is tested here.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

from codeexec.eval.gap import ProblemResult
from codeexec.eval.measure_model import build_report, frozen_100_ids, main, parse_args
from codeexec.train.config import ConfigError, RunConfig, pinned_model_path

HASH = "df3ce67c0e24480f20468b6ef2894622d69eb73b"


def test_out_is_required() -> None:
    with pytest.raises(SystemExit):
        parse_args([])


def test_frozen_100_keeps_the_phase_3_order() -> None:
    ids = frozen_100_ids()
    assert len(ids) == len(set(ids)) == 100
    assert ids[:3] == ["Mbpp/436", "Mbpp/460", "Mbpp/63"]  # sample order, not sorted


def test_report_holds_both_sets_and_the_engine_check() -> None:
    results = {
        "Mbpp/2": ProblemResult(False, True, True, True),
        "Mbpp/3": ProblemResult(False, True, False, False),
        "Mbpp/4": ProblemResult(True, False, False, False),
    }
    record = {
        "Mbpp/3": {"visible_pass": True, "plus_pass": True},  # this engine disagrees
        "Mbpp/2": {"visible_pass": True, "plus_pass": True},
    }
    report = build_report(
        meta={"model": "m"},
        completions={tid: "      return 1\n" for tid in results},
        finish_reasons={"Mbpp/2": "stop", "Mbpp/3": "length", "Mbpp/4": "stop"},
        results=results,
        frozen_ids=["Mbpp/3", "Mbpp/2"],
        record=record,
    )

    assert report["all_problems"]["n"] == 3
    assert report["all_problems"]["truncated"] == 1
    assert report["frozen_100"]["n"] == 2
    assert report["engine_check"] == {"compared": 2, "disagreements": ["Mbpp/3"]}
    assert report["per_problem"]["Mbpp/3"]["finish_reason"] == "length"
    assert set(report["completions"]) == set(results)


def test_no_record_means_no_engine_check() -> None:
    results = {"Mbpp/2": ProblemResult(False, True, True, True)}
    report = build_report(
        meta={},
        completions={"Mbpp/2": ""},
        finish_reasons={"Mbpp/2": "stop"},
        results=results,
        frozen_ids=["Mbpp/2"],
    )

    assert report["engine_check"] is None


def test_an_existing_report_is_never_overwritten(tmp_path: Path) -> None:
    """Refused before anything loads: CI has no vLLM or torch."""
    out = tmp_path / "g_baseline_box.json"
    out.write_text("{}")
    assert main(["--out", str(out)]) == 2
    assert out.read_text() == "{}"


def _fake_hub(monkeypatch: pytest.MonkeyPatch, folder: str) -> None:
    hub = types.ModuleType("huggingface_hub")
    hub.snapshot_download = lambda repo_id, revision: f"/cache/snapshots/{folder}"  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)


def test_pinned_model_path_is_the_snapshot_named_by_its_hash(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = RunConfig(model="Qwen/Qwen2.5-Coder-1.5B", revision=HASH, grpo={})
    _fake_hub(monkeypatch, HASH)
    assert pinned_model_path(config) == Path(f"/cache/snapshots/{HASH}")
    _fake_hub(monkeypatch, "0" * 40)
    with pytest.raises(ConfigError, match="not the pinned"):
        pinned_model_path(config)
