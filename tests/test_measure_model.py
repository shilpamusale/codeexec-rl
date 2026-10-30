"""Tests for the model-measurement command (B7), without a GPU.

Generation needs vLLM on the box; everything around it, the report,
the frozen-100 subset, the engine check and the refusals, is tested here.
"""

from __future__ import annotations

import json
import sys
import types
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from codeexec.eval.gap import ProblemResult
from codeexec.eval.measure_model import build_report, frozen_100_ids, main, parse_args
from codeexec.rewards.harness import TestCases, TestResult
from codeexec.tasks.problem import Problem
from codeexec.tasks.problem import TestCase as Case

HASH = "df3ce67c0e24480f20468b6ef2894622d69eb73b"
CONFIG = Path(__file__).resolve().parents[1] / "configs" / "grpo.toml"


# --------------------------------------------------------------------------- #
# --completions: ADR-015's regrade of saved completions, with no model
# --------------------------------------------------------------------------- #
def _add_problem(task_id: str) -> Problem:
    return Problem(
        task_id=task_id,
        source="mbpp",
        prompt='"""Add two numbers."""',
        entry_point="add",
        canonical_solution="def add(a, b):\n    return a + b\n",
        visible_tests=(Case((2, 3), 5),),
        heldout_tests=tuple(Case((i, i), 2 * i) for i in range(4)),
        atol=0.0,
    )


def _passes_everything(code: str, entry_point: str, tests: TestCases, atol: float) -> TestResult:
    return TestResult(passed=len(tests), total=len(tests), timed_out=False, crashed=False)


def _saved_report(path: Path) -> dict[str, Any]:
    """An earlier report, as the B7 box run wrote it, including Mbpp/255."""
    reasons = {"Mbpp/2": "stop", "Mbpp/3": "length", "Mbpp/255": "stop"}
    report = {
        "meta": {"engine": "vllm 0.31.0", "grader": "docker run_tests"},
        "per_problem": {tid: {"finish_reason": reason} for tid, reason in reasons.items()},
        "completions": dict.fromkeys(reasons, "    return a + b\n"),
    }
    path.write_text(json.dumps(report))
    return report


def test_regrade_grades_saved_completions_without_a_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The completions are graded again in their saved order, Mbpp/255 (no longer
    in the set) is set aside, and no model is touched."""
    from codeexec.eval import gap, measure_model

    source = tmp_path / "box.json"
    saved = _saved_report(source)

    def no_model(*args: object, **kwargs: object) -> None:
        raise AssertionError("the regrade must not load a model")

    monkeypatch.setattr(measure_model, "generate", no_model)
    monkeypatch.setattr(measure_model, "pinned_model_path", no_model)
    problems = {tid: _add_problem(tid) for tid in ("Mbpp/2", "Mbpp/3")}
    monkeypatch.setattr(measure_model, "training_problems", lambda: problems)
    monkeypatch.setattr(measure_model, "frozen_100_ids", lambda: ["Mbpp/3", "Mbpp/2"])
    monkeypatch.setattr(measure_model, "git_state", lambda: {"commit": "1" * 40, "dirty": False})
    monkeypatch.setattr(
        measure_model,
        "measure",
        lambda completions, problems, max_workers: gap.measure(
            completions, problems, run=_passes_everything, max_workers=2
        ),
    )

    out = tmp_path / "regrade.json"
    assert main(["--completions", str(source), "--out", str(out)]) == 0
    report = json.loads(out.read_text())
    assert list(report["completions"]) == ["Mbpp/2", "Mbpp/3"]  # saved order, Mbpp/255 out
    assert report["meta"]["set_aside"] == ["Mbpp/255"]
    assert report["meta"]["generation"] == saved["meta"]
    assert report["meta"]["regraded_from"] == str(source)
    assert "ADR-015" in report["meta"]["grader"]
    assert (report["all_problems"]["n"], report["frozen_100"]["n"]) == (2, 2)
    assert report["all_problems"]["truncated"] == 1  # the saved finish reasons


def test_regrade_refuses_a_file_that_is_not_a_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from codeexec.eval import measure_model

    source = tmp_path / "completions_only.json"
    source.write_text(json.dumps({"completions": {"Mbpp/2": "    return a + b\n"}}))
    problems = {"Mbpp/2": _add_problem("Mbpp/2")}
    monkeypatch.setattr(measure_model, "training_problems", lambda: problems)
    monkeypatch.setattr(measure_model, "git_state", lambda: {"commit": "1" * 40, "dirty": False})
    out = tmp_path / "regrade.json"
    assert main(["--completions", str(source), "--out", str(out)]) == 2
    assert not out.exists()


def test_a_model_and_saved_completions_are_exclusive() -> None:
    with pytest.raises(SystemExit):
        parse_args(["--out", "x.json", "--model", "ckpt", "--completions", "box.json"])


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


def test_report_lists_the_problems_graded_again_alone() -> None:
    """ADR-008 status update: every regrade is listed, and the verdict it replaced is kept."""
    slow = ProblemResult(False, False, False, False, time_limited=True)
    results = {
        "Mbpp/2": ProblemResult(
            False, True, True, True, concurrent=replace(slow, visible_pass=True)
        ),
        "Mbpp/3": replace(slow, concurrent=slow),  # out of time even alone
        "Mbpp/4": ProblemResult(False, True, True, True),  # no time limit fired
    }

    report = build_report(
        meta={},
        completions=dict.fromkeys(results, ""),
        finish_reasons=dict.fromkeys(results, "stop"),
        results=results,
        frozen_ids=["Mbpp/2"],
    )

    assert report["regraded_alone"] == {
        "problems": ["Mbpp/2", "Mbpp/3"],
        "changed": ["Mbpp/2"],
        "time_limited_alone": ["Mbpp/3"],
    }
    assert report["per_problem"]["Mbpp/2"]["concurrent"]["plus_b"] is False  # what it replaced
    assert report["per_problem"]["Mbpp/4"]["concurrent"] is None


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


def test_main_runs_end_to_end_with_a_fake_engine(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The whole box path, from generation to the written report, runs here with
    stand-ins for vLLM, torch and the tokenizer. The stand-ins accept only the
    arguments the real libraries take, so a typo in this path fails CI instead
    of a paid GPU session."""
    from codeexec.eval import gap, measure_model
    from codeexec.tasks.problem import Problem
    from codeexec.tasks.problem import TestCase as Case

    class FakeLLM:
        def __init__(
            self,
            *,
            model: str,
            dtype: str,
            gpu_memory_utilization: float,
            max_model_len: int | None,
            seed: int,
        ) -> None:
            pass

        def generate(
            self, prompts: list[str], *, sampling_params: object, use_tqdm: bool
        ) -> list[types.SimpleNamespace]:
            done = types.SimpleNamespace(token_ids=[1, 2, 3], finish_reason="stop")
            return [types.SimpleNamespace(outputs=[done]) for _ in prompts]

    def fake_sampling_params(*, temperature: float, max_tokens: int, n: int) -> object:
        return object()

    class FakeTokenizer:
        def batch_decode(self, ids: list[list[int]], *, skip_special_tokens: bool) -> list[str]:
            return ["    return a + b\n" for _ in ids]

    fakes: dict[str, dict[str, Any]] = {
        "vllm": {"LLM": FakeLLM, "SamplingParams": fake_sampling_params},
        "transformers": {
            "AutoTokenizer": types.SimpleNamespace(from_pretrained=lambda path: FakeTokenizer())
        },
        "torch": {"cuda": types.SimpleNamespace(get_device_name=lambda index: "FakeGPU")},
    }
    for name, attributes in fakes.items():
        module = types.ModuleType(name)
        for attribute, value in attributes.items():
            setattr(module, attribute, value)
        monkeypatch.setitem(sys.modules, name, module)

    def add_problem(task_id: str) -> Problem:
        return Problem(
            task_id=task_id,
            source="mbpp",
            prompt='"""Add two numbers."""',
            entry_point="add",
            canonical_solution="def add(a, b):\n    return a + b\n",
            visible_tests=(Case((2, 3), 5),),
            heldout_tests=tuple(Case((i, i), 2 * i) for i in range(4)),
            atol=0.0,
        )

    def passes_everything(code: str, entry_point: str, tests: Any, atol: float) -> TestResult:
        return TestResult(passed=len(tests), total=len(tests), timed_out=False, crashed=False)

    problems = {tid: add_problem(tid) for tid in ("Mbpp/2", "Mbpp/3")}
    monkeypatch.setattr(measure_model, "training_problems", lambda: problems)
    monkeypatch.setattr(measure_model, "frozen_100_ids", lambda: ["Mbpp/3", "Mbpp/2"])
    monkeypatch.setattr(measure_model, "git_state", lambda: {"commit": "0" * 40, "dirty": False})
    monkeypatch.setattr(measure_model, "version", lambda name: "test")
    monkeypatch.setattr(
        measure_model,
        "measure",
        lambda completions, problems, max_workers: gap.measure(
            completions, problems, run=passes_everything, max_workers=2
        ),
    )

    out = tmp_path / "report.json"
    argv = ["--out", str(out), "--model", str(tmp_path / "checkpoint"), "--config", str(CONFIG)]
    assert measure_model.main(argv) == 0

    report = json.loads(out.read_text())
    assert report["all_problems"]["n"] == 2
    assert report["all_problems"]["plus_b"] == 2
    assert report["frozen_100"]["n"] == 2
    assert report["meta"]["max_new_tokens"] == 256
    assert report["meta"]["gpu"] == "FakeGPU"
    assert report["completions"]["Mbpp/2"] == "    return a + b\n"
