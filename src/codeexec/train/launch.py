"""Training launcher (B6): one GRPO run, for one arm and one seed.

    python -m codeexec.train.launch --arm V1 --seed 0
    python -m codeexec.train.launch --arm V1 --seed 0 --max-steps 1 --report-to none

The second line is the smoke step (the Section B checkpoint): one optimizer step, no tracker.
The launcher wires existing pieces together and adds no logic of its own to the experiment: the
shared config (configs/grpo.toml), the MBPP training records, CodeExecEnv for the arm, and the
CodeExecReward adapter, handed to TRL's GRPOTrainer with vLLM colocated (ADR-011).

It adds two guarantees:

-   The pinned model everywhere. The pinned revision is downloaded once, and the policy,
    the tokenizer and vLLM all load that local copy. Given a model name, TRL's vLLM engine
    would load the files on `main` whatever the revision.
-   A record of the run. Before training, run.json in the output directory holds the resolved
    settings, any overrides, the git commit, package versions and the GPU; afterwards it gains
    the peak GPU memory and the finish time.

Every refusal (a dirty tree, an existing run, missing W&B credentials) happens before TRL, torch
or the model are touched, so a bad launch fails in seconds, and everything except `train` runs,
and is tested, without the training stack.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any

from codeexec.env import ARMS, Arm, CodeExecEnv
from codeexec.tasks.datasets import training_problems
from codeexec.train.config import DEFAULT_CONFIG, grpo_kwargs, load_config, pinned_model_path
from codeexec.train.dataset import training_records
from codeexec.train.reward import CodeExecReward

DEFAULT_OUTPUT_ROOT = Path("runs")
WANDB_PROJECT = "codeexec-rl"
RECORDED_PACKAGES = ("trl", "vllm", "torch", "transformers", "accelerate", "datasets", "wandb")


class LaunchError(RuntimeError):
    """The run cannot start safely. Nothing has been trained."""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one GRPO arm for one seed.")
    parser.add_argument("--arm", required=True, choices=ARMS)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--max-steps", type=int, help="override the config (1 = smoke step)")
    parser.add_argument("--report-to", help="override the config's tracker (none = no tracker)")
    parser.add_argument("--resume", action="store_true", help="continue from the last checkpoint")
    parser.add_argument("--allow-dirty", action="store_true", help="run with uncommitted changes")
    return parser.parse_args(argv)


def git_state(repo: Path = Path(".")) -> dict[str, Any]:
    """The commit being run, and whether the working tree differs from it."""

    def git(*args: str) -> str:
        command = ["git", "-C", str(repo), *args]
        return subprocess.run(command, capture_output=True, text=True, check=True).stdout.strip()

    return {"commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain"))}


def prepare_output_dir(path: Path, resume: bool) -> None:
    """A fresh run gets an empty directory; a resumed run needs its earlier one."""

    has_run = path.is_dir() and any(path.iterdir())
    if resume and not has_run:
        raise LaunchError(f"--resume given, but {path} holds no earlier run")
    if not resume and has_run:
        raise LaunchError(f"{path} already holds a run; pass --resume or another --output-root")
    path.mkdir(parents=True, exist_ok=True)


def check_tracker(report_to: str | list[str]) -> None:
    """W&B must have credentials before training starts, not prompt for them mid-run."""
    trackers = [report_to] if isinstance(report_to, str) else report_to
    has_credentials = os.environ.get("WANDB_API_KEY") or (Path.home() / ".netrc").exists()
    if "wandb" in trackers and not has_credentials:
        raise LaunchError(
            "W&B logging is on but no credentials were found: run `wandb login`, "
            "set WANDB_API_KEY, or pass --report-to none"
        )


def check_prompt_budget(
    max_prompt_tokens: int, max_completion_length: int, max_model_length: int | None
) -> None:
    """vLLM's context must hold the longest prompt plus a full-length completion."""
    if (
        max_model_length is not None
        and max_prompt_tokens + max_completion_length > max_model_length
    ):
        raise LaunchError(
            f"the longest prompt ({max_prompt_tokens} tokens) plus a full completion "
            f"({max_completion_length}) exceeds vllm_max_model_length ({max_model_length})"
        )


def write_record(path: Path, record: dict[str, Any]) -> None:
    path.write_text(json.dumps(record, indent=2, sort_keys=True, default=str) + "\n")


def train(args: argparse.Namespace) -> None:
    """Check everything cheap, then build the trainer and run it."""
    config = load_config(args.config)
    git = git_state()
    if git["dirty"] and not args.allow_dirty:
        raise LaunchError(
            "the working tree has uncommitted changes; commit them or pass --allow-dirty"
        )
    arm: Arm = args.arm
    output_dir = args.output_root / f"{arm}-seed{args.seed}"
    kwargs = grpo_kwargs(
        config,
        arm=arm,
        seed=args.seed,
        output_dir=output_dir,
        max_steps=args.max_steps,
        report_to=args.report_to,
    )
    check_tracker(kwargs["report_to"])
    prepare_output_dir(output_dir, args.resume)

    # The training stack, only now that every cheap check has passed.
    import torch
    from datasets import Dataset
    from transformers import AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer

    model_path = pinned_model_path(config)

    problems = training_problems()
    records = training_records(problems)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    max_prompt_tokens = max(len(tokenizer(r["prompt"]).input_ids) for r in records)
    check_prompt_budget(
        max_prompt_tokens, kwargs["max_completion_length"], kwargs.get("vllm_max_model_length")
    )

    record: dict[str, Any] = {
        "arm": arm,
        "seed": args.seed,
        "config": str(args.config),
        "model": config.model,
        "revision": config.revision,
        "overrides": {
            "max_steps": args.max_steps,
            "report_to": args.report_to,
            "resume": args.resume,
            "allow_dirty": args.allow_dirty,
        },
        "grpo": kwargs,
        "git": git,
        "packages": {name: version(name) for name in RECORDED_PACKAGES},
        "gpu": torch.cuda.get_device_name(0),
        "training_problems": len(records),
        "max_prompt_tokens": max_prompt_tokens,
        "started_at": datetime.now(UTC).isoformat(),
    }
    write_record(output_dir / "run.json", record)

    os.environ.setdefault("WANDB_PROJECT", WANDB_PROJECT)
    trainer = GRPOTrainer(
        model=str(model_path),  # loads the local copy; TRL loads its tokenizer from there too
        reward_funcs=CodeExecReward(CodeExecEnv(problems, arm)),
        args=GRPOConfig(**kwargs),
        train_dataset=Dataset.from_list([dict(r) for r in records]),
    )

    trainer.train(resume_from_checkpoint=True if args.resume else None)
    trainer.save_model(str(output_dir / "final"))

    record["finished_at"] = datetime.now(UTC).isoformat()
    record["peak_gpu_memory_gib"] = round(torch.cuda.max_memory_reserved() / 2**30, 2)
    write_record(output_dir / "run.json", record)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        train(args)
    except LaunchError as error:
        print(f"launch refused: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
