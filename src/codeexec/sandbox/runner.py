"""Sandbox execution runner.

Launches a container from the codeexec-sandbox image, runs code inside it via
stdin under strict isolation and a wall-clock timeout, and returns the raw
captured output. This is the plumbing layer: it proves code can be driven into
a container and results read back, with deterministic teardown. It does not yet
classify outcomes into the full status enum (that builds on this).
"""

import subprocess
import uuid
from dataclasses import dataclass

IMAGE = "codeexec-sandbox"
TIMEOUT_SECONDS = 10
# subprocess backstop: a few seconds longer than the in-container limit, so the
# container's own `timeout` fires first in the normal case and this only trips
# if the container runtime itself wedges.
SUBPROCESS_TIMEOUT_SECONDS = TIMEOUT_SECONDS + 5
# Exit code the `timeout` utility returns when it kills the process it ran.
TIMEOUT_EXIT_CODE = 124


@dataclass
class RawResult:
    """Raw output from one container execution, before status classification."""

    stdout: str
    stderr: str
    returncode: int
    timed_out: bool = False


def run_code(code: str) -> RawResult:
    """Run `code` inside a fresh, isolated container and return its raw output.

    The code is delivered over stdin, not as a command-line argument, so a
    multi-line program with quotes and special characters arrives intact. The
    container is network-isolated, runs on a read-only filesystem with a small
    writable scratch mount, and is bounded in memory, process count, and
    wall-clock time. The time limit is enforced inside the container by `timeout`,
    with a subprocess-level backstop. The container is always removed, whether it
    exits, errors, or times out.
    """
    container_name = f"codeexec-{uuid.uuid4().hex}"

    try:
        completed = subprocess.run(
            [
                "docker",
                "run",
                "--rm",
                "-i",
                "--name",
                container_name,
                "--network",
                "none",
                "--read-only",
                "--tmpfs",
                "/sandbox:size=64m,uid=1000,mode=0700",
                "--memory",
                "256m",
                "--pids-limit",
                "128",
                IMAGE,
                "timeout",
                str(TIMEOUT_SECONDS),
                "python",
            ],
            input=code,
            capture_output=True,
            text=True,
            timeout=SUBPROCESS_TIMEOUT_SECONDS,
        )
        # The in-container `timeout` returns 124 when it killed the process.
        if completed.returncode == TIMEOUT_EXIT_CODE:
            return RawResult(
                stdout=completed.stdout,
                stderr=completed.stderr,
                returncode=completed.returncode,
                timed_out=True,
            )
        return RawResult(
            stdout=completed.stdout,
            stderr=completed.stderr,
            returncode=completed.returncode,
        )
    except subprocess.TimeoutExpired:
        # Backstop: the container itself wedged past the in-container limit.
        return RawResult(
            stdout="",
            stderr=f"Execution exceeded the {TIMEOUT_SECONDS}s wall-clock limit.",
            returncode=-1,
            timed_out=True,
        )
    finally:
        # Deterministic teardown. In the normal case the container exits on its
        # own (via `timeout` + --rm) and these are quick no-ops; in the backstop
        # case the container is still running, so SIGKILL it immediately and
        # remove it. check=False so both are safe no-ops when nothing remains.
        subprocess.run(
            ["docker", "kill", container_name],
            capture_output=True,
            check=False,
        )
        subprocess.run(
            ["docker", "rm", "-f", container_name],
            capture_output=True,
            check=False,
        )
