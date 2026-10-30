"""Measure G for a model: the base model now, each training checkpoint later (B7).

    python -m codeexec.eval.measure_model --out analysis/g_baseline_box.json \\
        --compare analysis/g_baseline_split_b.json
    python -m codeexec.eval.measure_model --model runs/V1-seed0/final \\
        --out runs/V1-seed0/gap.json
    python -m codeexec.eval.measure_model --completions analysis/g_baseline_box.json \\
        --out analysis/g_baseline_box_adr015.json --compare analysis/g_baseline_split_b.json

As pre-registered in ADR-004's status update: vLLM, greedy (temperature 0), at
most max_completion_length new tokens (256, from the shared config), stopping only
at end-of-sequence, in bf16, over every training problem with the training prompt.
Completions are decoded from their token ids exactly as TRL decodes them for the
reward (skip_special_tokens=True), and gap.py grades them.

The frozen-100 subset is summarized separately, in its Phase 3 order. With
--compare, its per-problem results are checked against an earlier record: the
engine check. Differences are reported, never resolved away.So are regrades:
every problem graded again alone because a time limit fired under load (gap.py,
ADR-008 status update) is listed, with the concurrent verdict it replaced.

Without --model, the pinned base model from configs/grpo.toml is measured. The
output keeps every completion, so any number in it can be re-derived without a
GPU. With --completions, that is done: no model is loaded, and the completions
saved in an earlier report are graded again, in their saved order, as ADR-015's
regrade requires. Completions for problems no longer in the set are set aside
and listed, and the earlier report's meta is kept under "generation". Every
refusal (an existing output, uncommitted changes, a file that is not a report)
happens before any model loads or any grading starts.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any

from codeexec.eval.gap import ProblemResult, disagreements, measure, summarize
from codeexec.tasks.datasets import training_problems
from codeexec.tasks.problem import Problem
from codeexec.train.config import DEFAULT_CONFIG, load_config, pinned_model_path
from codeexec.train.dataset import training_records
from codeexec.train.launch import git_state

FROZEN_100 = Path("analysis/completions_frozen100.json")  # its key order is the Phase 3 order
DEFAULT_GPU_MEMORY_UTILIZATION = 0.85  # measurement runs alone on the GPU
GRADER = "docker run_tests, EvalPlus's pass rule (ADR-015)"


class MeasureError(RuntimeError):
    """The measurement cannot run safely. Nothing has been generated."""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure G for a model (ADR-004).")
    parser.add_argument("--out", required=True, type=Path, help="where to write the JSON report")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--model", type=Path, help="checkpoint folder; default: the pinned base")
    source.add_argument(
        "--completions", type=Path, help="regrade an earlier report's completions; no model"
    )
    parser.add_argument("--compare", type=Path, help="earlier split-B record for the engine check")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--gpu-memory-utilization", type=float, default=DEFAULT_GPU_MEMORY_UTILIZATION
    )
    parser.add_argument("--max-workers", type=int, default=24, help="grading concurrency")
    parser.add_argument("--overwrite", action="store_true", help="replace an existing report")
    parser.add_argument("--allow-dirty", action="store_true", help="run with uncommitted changes")

    return parser.parse_args(argv)


def frozen_100_ids(path: Path = FROZEN_100) -> list[str]:
    """The frozen-100 task-ids, in the order Phase 3 bootstrapped them."""
    return list(json.loads(path.read_text())["completions"])


@dataclass(frozen=True)
class SavedRun:
    """What an earlier report saved, restricted to the problems now in the set."""

    completions: dict[str, str]  # in the report's order
    finish_reasons: dict[str, str | None]
    set_aside: list[str]  # task ids the report holds that are no longer in the set
    meta: dict[str, Any]  # how the completions were generated


def saved_completions(path: Path, problems: Mapping[str, Problem]) -> SavedRun:
    """Read the completions an earlier report of this command saved."""
    report = json.loads(path.read_text())
    if not {"meta", "per_problem", "completions"} <= report.keys():
        raise MeasureError(f"{path} is not a report written by this command")
    saved: dict[str, str] = report["completions"]
    kept = {tid: text for tid, text in saved.items() if tid in problems}
    return SavedRun(
        completions=kept,
        finish_reasons={tid: report["per_problem"][tid]["finish_reason"] for tid in kept},
        set_aside=[tid for tid in saved if tid not in problems],
        meta=report["meta"],
    )


def generate(
    model_path: Path,
    prompts: Sequence[str],
    *,
    max_tokens: int,
    max_model_len: int | None,
    gpu_memory_utilization: float,
) -> tuple[list[str], list[str | None]]:
    """Greedy completions, decoded as TRL decodes them, and each one's finish reason."""
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=str(model_path),
        dtype="bfloat16",
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        seed=0,
    )
    params = SamplingParams(temperature=0.0, max_tokens=max_tokens, n=1)
    outputs = llm.generate(list(prompts), sampling_params=params, use_tqdm=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    ids = [list(out.outputs[0].token_ids) for out in outputs]
    texts = tokenizer.batch_decode(ids, skip_special_tokens=True)
    return [str(t) for t in texts], [out.outputs[0].finish_reason for out in outputs]


def regrade_summary(results: Mapping[str, ProblemResult]) -> dict[str, list[str]]:
    """The problems measure graded again alone because a time limit fired under load,
    those whose verdict changed alone, and those that ran out of time even alone."""

    summary: dict[str, list[str]] = {"problems": [], "changed": [], "time_limited_alone": []}
    for tid, result in results.items():
        if result.concurrent is None:
            continue
        summary["problems"].append(tid)
        if _verdict(result) != _verdict(result.concurrent):
            summary["changed"].append(tid)
        if result.time_limited:
            summary["time_limited_alone"].append(tid)
    return summary


def _verdict(result: ProblemResult) -> tuple[bool, bool, bool]:
    return result.visible_pass, result.plus_b, result.plus_full


def build_report(
    *,
    meta: Mapping[str, Any],
    completions: Mapping[str, str],
    finish_reasons: Mapping[str, str | None],
    results: Mapping[str, ProblemResult],
    frozen_ids: Sequence[str],
    record: Mapping[str, Mapping[str, bool]] | None = None,
) -> dict[str, Any]:
    """The measurement: the full set, the frozen-100 subset, the engine check, and the
    problems graded again alone."""

    frozen = {tid: results[tid] for tid in frozen_ids}
    return {
        "meta": dict(meta),
        "all_problems": {
            **asdict(summarize(results)),
            "truncated": sum(reason == "length" for reason in finish_reasons.values()),
        },
        "frozen_100": asdict(summarize(frozen)),
        "engine_check": None
        if record is None
        else {"compared": len(frozen), "disagreements": disagreements(frozen, record)},
        "regraded_alone": regrade_summary(results),
        "per_problem": {
            tid: {**asdict(r), "finish_reason": finish_reasons[tid]} for tid, r in results.items()
        },
        "completions": dict(completions),
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.out.exists() and not args.overwrite:
            raise MeasureError(f"{args.out} exists; pass --overwrite to replace it")
        git = git_state()
        if git["dirty"] and not args.allow_dirty:
            raise MeasureError("the working tree has uncommitted changes; commit or --allow-dirty")
    except MeasureError as error:
        print(f"measurement refused: {error}", file=sys.stderr)
        return 2

    record = None
    if args.compare is not None:
        record = json.loads(args.compare.read_text())["split B (1/2)"]["per_problem"]
    problems = training_problems()
    if args.completions is not None:
        return regrade(args, problems, record, git)

    config = load_config(args.config)
    records = training_records(problems)  # sorted by task_id, the training prompts
    task_ids = [r["task_id"] for r in records]

    model_path = args.model if args.model is not None else pinned_model_path(config)
    texts, reasons = generate(
        model_path,
        [r["prompt"] for r in records],
        max_tokens=config.grpo["max_completion_length"],
        max_model_len=config.grpo.get("vllm_max_model_length"),
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    completions = dict(zip(task_ids, texts, strict=True))
    results = measure(completions, problems, max_workers=args.max_workers)

    import torch

    meta = {
        "model": str(args.model) if args.model is not None else config.model,
        "revision": None if args.model is not None else config.revision,
        "engine": f"vllm {version('vllm')}",
        "dtype": "bfloat16",
        "decoding": "greedy (temperature 0), stop at end-of-sequence only",
        "max_new_tokens": config.grpo["max_completion_length"],
        "prompt": "zero-shot-plus-stub (build_prompt), as in training",
        "measurement_set": f"all {len(task_ids)} training problems; frozen-100 subset alongside",
        "grader": GRADER,
        "gpu": torch.cuda.get_device_name(0),
        "git": git,
        "compared_with": None if args.compare is None else str(args.compare),
        "created_at": datetime.now(UTC).isoformat(),
    }
    report = build_report(
        meta=meta,
        completions=completions,
        finish_reasons=dict(zip(task_ids, reasons, strict=True)),
        results=results,
        frozen_ids=frozen_100_ids(),
        record=record,
    )
    return write_and_print(args.out, report)


def regrade(
    args: argparse.Namespace,
    problems: Mapping[str, Problem],
    record: Mapping[str, Mapping[str, bool]] | None,
    git: Mapping[str, Any],
) -> int:
    """Grade an earlier report's completions again; no model is loaded (ADR-015)."""
    try:
        saved = saved_completions(args.completions, problems)
    except MeasureError as error:
        print(f"measurement refused: {error}", file=sys.stderr)
        return 2
    results = measure(saved.completions, problems, max_workers=args.max_workers)
    n = len(saved.completions)

    meta = {
        "regraded_from": str(args.completions),
        "set_aside": saved.set_aside,
        "measurement_set": f"all {n} training problems; frozen-100 subset alongside",
        "grader": GRADER,
        "git": dict(git),
        "compared_with": None if args.compare is None else str(args.compare),
        "created_at": datetime.now(UTC).isoformat(),
        "generation": saved.meta,
    }

    report = build_report(
        meta=meta,
        completions=saved.completions,
        finish_reasons=saved.finish_reasons,
        results=results,
        frozen_ids=frozen_100_ids(),
        record=record,
    )
    print(f"regraded {n} completions from {args.completions}; set aside {saved.set_aside}")
    return write_and_print(args.out, report)


def write_and_print(out: Path, report: Mapping[str, Any]) -> int:
    """Write the report, then print its headline numbers."""
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")

    for name in ("all_problems", "frozen_100"):
        s = report[name]
        lo, hi = s["ci_pp"]
        print(
            f"{name:<13} n={s['n']} visible {s['visible']} plus_b {s['plus_b']} "
            f"G = {s['gap_pp']:+.1f} pp 95% CI [{lo:+.1f}, {hi:+.1f}]"
        )

    if report["engine_check"] is not None:
        found = report["engine_check"]["disagreements"]
        print(f"engine check: {len(found)} disagreements {found}")
    regraded = report["regraded_alone"]
    print(
        f"regraded alone: {len(regraded['problems'])}; verdict changed: {regraded['changed']}; "
        f"time-limited even alone: {regraded['time_limited_alone']}"
    )
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
