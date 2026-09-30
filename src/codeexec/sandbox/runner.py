"""Sandbox execution runner.

Launches a container from the codeexec-sandbox image, runs a piece of code
inside it via stdin, and returns the raw captured output. This is the plumbing layer:
it proves code can be driven into a container and results read back. It does not
yet apply isolation flags (Phase 1, later) or classify outcomes into the status enum
(that builds on this)
"""

import subprocess
from dataclasses import dataclass

IMAGE = "codeexec-sandbox"


@dataclass
class RawResult:
    """
    Raw output from one container execution, before status classification.
    """

    stdout: str
    stderr: str
    returncode: int


def run_code(code: str) -> RawResult:
    """
    Run `code` inside a fresh container and return its raw output.
    The code is delivered over stdin, not as a command-line argument,
    so a multi-line program with quotes and special characters arrives intact.

    """
    try:
        completed = subprocess.run(
            ["docker", "run", "--rm", "-i", IMAGE, "python"],
            input=code,
            capture_output=True,
            text=True,
        )
        return RawResult(
            stdout=completed.stdout,
            stderr=completed.stderr,
            returncode=completed.returncode,
        )
    finally:
        # Deterministic teardown backstop. --rm removes the container on normal
        # exit; this block is where forced cleanup will go once containers can
        # outlive their call (timeouts, later). For now it documents the intent
        # and guarantees the function has a single, explicit exit path.
        pass
