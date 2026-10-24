"""Tests for the shared training configuration."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import pytest

from codeexec.env import Arm
from codeexec.train.config import ConfigError, grpo_kwargs, load_config

REPO_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "grpo.toml"
HASH = "0123456789abcdef0123456789abcdef01234567"

VALID_GRPO: dict[str, Any] = {
    "remove_unused_columns": False,
    "max_completion_length": 256,
    "num_generations": 8,
    "per_device_train_batch_size": 4,
    "gradient_accumulation_steps": 32,
    "max_steps": 10,
}


def _write(tmp_path: Path, model: dict[str, Any] | None = None, **grpo: Any) -> Path:
    """Write a config: a valid one, with any [grpo] key replaced (None removes it)."""
    model = {"name": "Qwen/Qwen2.5-Coder-1.5B", "revision": HASH} if model is None else model
    merged = {k: v for k, v in {**VALID_GRPO, **grpo}.items() if v is not None}
    lines = ["[model]", *(f"{k} = {json.dumps(v)}" for k, v in model.items()), "[grpo]"]
    lines += [f"{k} = {json.dumps(v)}" for k, v in merged.items()]  # JSON scalars are TOML
    path = tmp_path / "grpo.toml"
    path.write_text("\n".join(lines) + "\n")
    return path


def test_repo_config_is_valid_and_holds_the_confirmed_batch() -> None:
    config = load_config(REPO_CONFIG)
    assert config.grpo["num_generations"] == 8
    assert config.prompts_per_step == 16
    assert config.completions_per_step == 128
    assert config.grpo["report_to"] == "wandb"


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"remove_unused_columns": True}, "task_id never reaches the reward"),
        ({"max_completion_length": 512}, "the baseline's generation budget"),
        ({"num_generations": 1}, "at least 2"),
        ({"num_generations": 6}, "whole groups"),  # 128 completions are not whole groups of 6
        ({"max_steps": None}, "missing"),
        ({"seed": 0}, "come from the command line only"),
        ({"output_dir": "runs/x"}, "come from the command line only"),
    ],
)
def test_broken_invariants_are_rejected(
    tmp_path: Path, change: dict[str, Any], message: str
) -> None:
    with pytest.raises(ConfigError, match=message):
        load_config(_write(tmp_path, **change))


def test_unpinned_revision_is_rejected(tmp_path: Path) -> None:
    path = _write(tmp_path, model={"name": "Qwen/Qwen2.5-Coder-1.5B", "revision": "main"})
    with pytest.raises(ConfigError, match="40-character commit hash"):
        load_config(path)


def test_unknown_section_is_rejected(tmp_path: Path) -> None:
    path = _write(tmp_path)
    path.write_text(path.read_text() + "[extra]\nx = 1\n")
    with pytest.raises(ConfigError, match="unknown sections"):
        load_config(path)


def test_every_issue_is_reported_at_once(tmp_path: Path) -> None:
    with pytest.raises(ConfigError) as excinfo:
        load_config(_write(tmp_path, remove_unused_columns=True, max_completion_length=512))
    assert "task_id" in str(excinfo.value) and "budget" in str(excinfo.value)


def test_run_identity_comes_from_the_arguments(tmp_path: Path) -> None:
    config = load_config(_write(tmp_path))
    kwargs = grpo_kwargs(config, arm="V1", seed=2, output_dir=Path("runs/V1-seed2"))
    assert (kwargs["seed"], kwargs["output_dir"], kwargs["run_name"]) == (
        2,
        "runs/V1-seed2",
        "V1-seed2",
    )
    assert kwargs["max_steps"] == 10  # from the file, unless overridden


def test_arms_share_every_setting_but_their_name(tmp_path: Path) -> None:
    """THESIS: the arms differ only in their reward, so their trainer settings match."""
    config = load_config(_write(tmp_path))
    v0 = grpo_kwargs(config, arm="V0", seed=0, output_dir=Path("runs/x"))
    v1 = grpo_kwargs(config, arm="V1", seed=0, output_dir=Path("runs/x"))
    assert {k for k in v0 if v0[k] != v1[k]} == {"run_name"}


def test_smoke_overrides_leave_the_shared_config_untouched(tmp_path: Path) -> None:
    config = load_config(_write(tmp_path))
    kwargs = grpo_kwargs(
        config, arm="V1", seed=0, output_dir=Path("runs/x"), max_steps=1, report_to="none"
    )
    assert (kwargs["max_steps"], kwargs["report_to"]) == (1, "none")
    assert config.grpo["max_steps"] == 10
    with pytest.raises(TypeError):
        config.grpo["max_steps"] = 1  # type: ignore[index]  # read-only


def test_unknown_arm_is_rejected(tmp_path: Path) -> None:
    config = load_config(_write(tmp_path))
    with pytest.raises(ValueError, match="arm must be one of"):
        grpo_kwargs(config, arm=cast(Arm, "V2"), seed=0, output_dir=Path("runs/x"))
