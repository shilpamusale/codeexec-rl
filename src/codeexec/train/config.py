"""Training configuration: one TOML file shared by every arm and seed (B6).

config/grpo.toml holds every setting the runs share. Only a run's identity, its
arm, seed, and output directory come from the command line, so the phase 5 runs
differ in exactly those and in nothing else. The file may not set them.

This module reads the file and checks the invariants the experiment depends on.
It does not import TRL, so CI tests it; the launcher turns the result into a GRPOConfig
on the training box, and TRL rejects any key it does not know.
"""

from __future__ import annotations

import re
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from codeexec.env import ARMS, Arm

DEFAULT_CONFIG = Path("configs/grpo.toml")

# G-baseline was generated with max_new_tokens = 256 (analysis/completions_frozen100.json).
BASELINE_MAX_NEW_TOKENS = 256

# A run's identity: set from the command line only, never from the file.
CLI_ONLY_KEYS = frozenset({"seed", "data_seed", "output_dir", "run_name"})
REQUIRED_GRPO_KEYS = frozenset(
    {
        "remove_unused_columns",
        "max_completion_length",
        "num_generations",
        "per_device_train_batch_size",
        "gradient_accumulation_steps",
        "max_steps",
    }
)
_COMMIT_HASH = re.compile(r"[0-9a-f]{40}")


class ConfigError(ValueError):
    """The configuration breaks an invariant the experiment depends on."""


@dataclass(frozen=True)
class RunConfig:
    """The shared configuration: the pinned model and the GRPOConfig arguments."""

    model: str
    revision: str  # a full commit hash, so the weights never float
    grpo: Mapping[str, Any]

    @property
    def completions_per_step(self) -> int:
        """Completions per optimizer step, on one GPU (ADR-010)."""
        return int(self.grpo["per_device_train_batch_size"]) * int(
            self.grpo["gradient_accumulation_steps"]
        )

    @property
    def prompts_per_step(self) -> int:
        return self.completions_per_step // int(self.grpo["num_generations"])


def load_config(path: Path = DEFAULT_CONFIG) -> RunConfig:
    """Read and validate the configuration file."""
    with path.open("rb") as f:
        data = tomllib.load(f)
    unknown = set(data) - {"model", "grpo"}
    if unknown:
        raise ConfigError(f"unknown sections: {sorted(unknown)}")
    model, grpo = data.get("model", {}), data.get("grpo", {})
    config = RunConfig(
        model=model.get("name", ""),
        revision=model.get("revision", ""),
        grpo=MappingProxyType(dict(grpo)),
    )
    _validate(config)
    return config


def _validate(config: RunConfig) -> None:
    """Collect every violated invariant, then raise once, listing them all."""
    issues: list[str] = []
    if not config.model:
        issues.append("[model] name is missing")
    if not _COMMIT_HASH.fullmatch(config.revision):
        issues.append("[model] revision must be full 40-character commit hash")
    missing = REQUIRED_GRPO_KEYS - set(config.grpo)
    if missing:
        issues.append(f"[grpo] is missing {sorted(missing)}")
    clash = CLI_ONLY_KEYS & set(config.grpo)
    if clash:
        issues.append(f"{sorted(clash)} identify a run and come from the command line only")

    if issues:
        raise ConfigError("; ".join(issues))

    if config.grpo["remove_unused_columns"] is not False:
        issues.append("remove_unused_columns must be false, or task_id never reaches the reward")

    if config.grpo["max_completion_length"] != BASELINE_MAX_NEW_TOKENS:
        issues.append(
            f"max_completion_length must be {BASELINE_MAX_NEW_TOKENS}, the baseline's "
            "generation budget, or delta-G compares different instruments"
        )
    generations = config.grpo["num_generations"]
    if not isinstance(generations, int) or generations < 2:
        issues.append("num_generations must be an integer of at least 2 (GRPO compares a group)")
    elif config.completions_per_step % generations:
        issues.append(
            f"completions per step ({config.completions_per_step}) must hold whole groups "
            f"of num_generations ({generations})"
        )
    if issues:
        raise ConfigError("; ".join(issues))


def pinned_model_path(config: RunConfig) -> Path:
    """Download the pinned revision (once; later calls hit the cache) and return its folder.
    Snapshot folders are named by their commit hash, so the folder's name proves
    which files are loaded. Every consumer loads this local copy: the policy, the tokenizer
    and vLLM, in training and in measurement alike.
    """
    from huggingface_hub import snapshot_download  # only where models are loaded

    path = Path(snapshot_download(config.model, revision=config.revision))
    if path.name != config.revision:
        raise ConfigError(f"downloaded {path.name}, not the pinned {config.revision}")
    return path


def grpo_kwargs(
    config: RunConfig,
    *,
    arm: Arm,
    seed: int,
    output_dir: Path,
    max_steps: int | None = None,
    report_to: str | None = None,
) -> dict[str, Any]:
    """GRPOConfig arguments for one run: the shared file plus the run's identity.

    `max_steps` and `report_to` may be overridden for a smoke test (one step,
    no tracker); the launcher records the overrides with the run.

    """
    if arm not in ARMS:
        raise ValueError(f"arm must be one of {ARMS}, got {arm!r}")

    kwargs = dict(config.grpo)
    kwargs.update(seed=seed, output_dir=str(output_dir), run_name=f"{arm}-seed{seed}")
    if max_steps is not None:
        kwargs["max_steps"] = max_steps
    if report_to is not None:
        kwargs["report_to"] = report_to
    return kwargs
