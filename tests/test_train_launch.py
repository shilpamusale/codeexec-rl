"""Tests for the training launcher's checks (B6).

Everything here runs without the training stack: the launcher refuses a bad run
before it imports TRL or torch, and these tests hold it to that.

"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from codeexec.train.launch import (
    LaunchError,
    check_prompt_budget,
    check_tracker,
    git_state,
    main,
    parse_args,
    prepare_output_dir,
)


def test_arm_and_seed_are_required() -> None:
    with pytest.raises(SystemExit):
        parse_args(["--arm", "V1"])
    with pytest.raises(SystemExit):
        parse_args(["--arm", "V2", "--seed", "0"])


def test_defaults_come_from_the_shared_config() -> None:
    args = parse_args(["--arm", "V0", "--seed", "1"])
    assert (args.config, args.output_root) == (Path("configs/grpo.toml"), Path("runs"))
    assert (args.max_steps, args.report_to, args.resume) == (None, None, False)


def test_a_fresh_run_gets_its_own_directory(tmp_path: Path) -> None:
    run = tmp_path / "runs" / "V1-seed0"
    prepare_output_dir(run, resume=False)
    assert run.is_dir()


def test_an_earlier_run_is_never_overwritten(tmp_path: Path) -> None:
    (tmp_path / "run.json").write_text("{}")
    with pytest.raises(LaunchError, match="already holds a run"):
        prepare_output_dir(tmp_path, resume=False)
    prepare_output_dir(tmp_path, resume=True)  # resuming it is fine


def test_resume_needs_an_earlier_run(tmp_path: Path) -> None:
    with pytest.raises(LaunchError, match="holds no earlier run"):
        prepare_output_dir(tmp_path / "missing", resume=True)


def test_git_state_reports_the_commit_and_uncommitted_changes(tmp_path: Path) -> None:
    def git(*args: str) -> None:
        subprocess.run(["git", "-C", str(tmp_path), *args], check=True, capture_output=True)

    git("init")
    (tmp_path / "f.txt").write_text("a")
    git("add", "f.txt")
    git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-m", "init")
    clean = git_state(tmp_path)
    assert len(clean["commit"]) == 40 and clean["dirty"] is False
    (tmp_path / "f.txt").write_text("b")
    assert git_state(tmp_path)["dirty"] is True


def test_wandb_needs_credentials_before_the_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    with pytest.raises(LaunchError, match="no credentials"):
        check_tracker("wandb")
    check_tracker("none")
    monkeypatch.setenv("WANDB_API_KEY", "X")
    check_tracker(["wandb"])


def test_vllm_context_must_hold_prompt_plus_completion() -> None:
    check_prompt_budget(768, 256, 1024)
    with pytest.raises(LaunchError, match="exceeds vllm_max_model_length"):
        check_prompt_budget(769, 256, 1024)
    check_prompt_budget(5000, 256, None)  # no limit set: vLLM uses the model's own


def test_a_refusal_exits_2_before_the_training_stack_loads(tmp_path: Path) -> None:
    """Runs in CI, where TRL and torch are not installed: the refusal must come first."""
    run = tmp_path / "V1-seed0"
    run.mkdir()
    (run / "run.json").write_text("{}")
    argv = ["--arm", "V1", "--seed", "0", "--output-root", str(tmp_path), "--report-to", "none"]
    assert main([*argv, "--allow-dirty"]) == 2
