"""Sandbox execution runner.

Launches a container from the codeexec-sandbox image, runs code inside it
via stdin under strict isolation and a wall-clock timeout, and returns the
raw captured output. This is the plumbing layer: it proves code can be driven
into a container and results read back, with deterministic teardown. It does not
classify outcomes into the full status enum (that builds on this).

It does tell one thing apart, because nothing above it can: whether the program
ran at all. When Docker or the container runtime fails, the result says nothing
about the program, so it is flagged `infra_error` and must never be scored (ADR-014).
"""

import logging
import subprocess
import uuid
from dataclasses import dataclass

logger = logging.getLogger(__name__)

IMAGE = "codeexec-sandbox"
TIMEOUT_SECONDS = 10

# subprocess backstop: a few seconds longer than the in-container limit, so the
# container's own `timeout` fires first in the normal case and this only trips
# if the container runtime itself wedges.

SUBPROCESS_TIMEOUT_SECONDS = TIMEOUT_SECONDS + 5

# Exit code the `timeout` utility returns when it kills the process it ran.
TIMEOUT_EXIT_CODE = 124

# Printed to stderr from inside `timeout`, just before the program starts. Its
# presence is positive evidence that Docker, the container, and `timeout` all
# worked, so everything after it is the program's own doing. Without it,
# something failed before the program could run (daemon down, image missing, a
# bad command line) and the result says nothing about the program. Exit codes
# alone cannot tell: Docker 29 reports an unreachable daemon as exit 1, the same
# code as an uncaught Python exception. Only the marker's absence counts, and a
# program cannot cause that, so no generated program can fake an infra failure.
# (The one failure after the marker that is not the program's, python missing
# from the image, is a fixed image bug the integration tests catch on any run.)
STARTED_MARKER = "CODEEXEC_STARTED_7c1e"

CLEANUP_TIMEOUT_SECONDS = 10


@dataclass
class RawResult:
    """Raw output from one container execution, before status classification."""

    stdout: str
    stderr: str
    returncode: int
    timed_out: bool = False
    # True when Docker or the container runtime failed, so this result says
    # nothing about the program. Callers must never score it (ADR-014).
    infra_error: bool = False


def is_infra_failure(stderr: str) -> bool:
    """True if a finished `docker run` failed before the program started."""
    return STARTED_MARKER not in stderr


def run_code(code: str) -> RawResult:
    """Run `code` inside a fresh, isolated container and return its raw output.

    The code is delivered over stdin, not as a command-line argument, so a multi-line
    program with quotes and special characters arrives intact. The container is
    network-isolated, runs on a read-only filesystem with a small writable scratch mount,
    and is bounded in memory, process count, and wall-clock time. The time limit is enforced
    inside the container by `timeout`, with a subprocess-level backdrop.Inside `timeout`,
    a shell prints STARTED_MARKER and then `exec`s the program, so exit codes are the program's own.
    `--pull never` means a missing image fails at once instead of being fetched.
    The container is always removed, whether it exits, errors, or times out.
    """
    container_name = f"codeexec-{uuid.uuid4().hex}"
    start_then_run = f"echo '{STARTED_MARKER}' >&2; exec python"

    try:
        completed = subprocess.run(
            [
                "docker",
                "run",
                "--rm",
                "-i",
                "--pull",
                "never",
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
                "sh",
                "-c",
                start_then_run,
            ],
            input=code,
            capture_output=True,
            text=True,
            timeout=SUBPROCESS_TIMEOUT_SECONDS,
        )
        return RawResult(
            stdout=completed.stdout,
            # The marker is plumbing, not program output: hand back the program's stderr.
            stderr=completed.stderr.replace(f"{STARTED_MARKER}\n", "", 1),
            returncode=completed.returncode,
            # The in-container `timeout` returns 124 when it killed the process.
            timed_out=completed.returncode == TIMEOUT_EXIT_CODE,
            infra_error=is_infra_failure(completed.stderr),
        )
    except subprocess.TimeoutExpired:
        # Backstop: the container outlived its own in-container limit, so the
        # runtime itself is wedged or badly degraded. This is an infrastructure
        # failure, not evidence about the program (ADR-014).
        return RawResult(
            stdout="",
            stderr=f"Container runtime exceeded the {SUBPROCESS_TIMEOUT_SECONDS}s backstop.",
            returncode=-1,
            infra_error=True,
        )
    finally:
        _cleanup(container_name)


def _cleanup(container_name: str) -> None:
    """Deterministic teardown that can never hang the caller.

    In the normal case the container has already exited on its own
    (`timeout` + `--rm`) and both commands are quick no-ops. In the
    backstop case it is still running, so it is killed and removed. Each command
    has its own timeout: if the Docker daemon itself hangs, the leak is logged
    and the caller carries on.
    """
    for command in (["docker", "kill", container_name], ["docker", "rm", "-f", container_name]):
        try:
            subprocess.run(
                command, capture_output=True, check=False, timeout=CLEANUP_TIMEOUT_SECONDS
            )
        except subprocess.TimeoutExpired:
            logger.warning("sandbox cleanup timed out: %s", " ".join(command))
