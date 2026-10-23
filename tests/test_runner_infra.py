"""Unit tests for infrastructure-failure handling in the sandbox runner (ADR-014).

No Docker needed: subprocess.run is replaced by a fake, so these run in CI. The
real-container conuterparts live in test_sandbox_runner.py (integration).
"""

import subprocess
from typing import Any

import pytest

from codeexec.sandbox.runner import (
    STARTED_MARKER,
    SUBPROCESS_TIMEOUT_SECONDS,
    is_infra_failure,
    run_code,
)

RAN = f"{STARTED_MARKER}\n"  # what stderr starts with whenever the program started


@pytest.mark.parametrize(
    ("stderr", "expected"),
    [
        # No marker: something failed before the program started.
        ("failed to connect to the docker API at unix:///var/run/docker.sock", True),
        ("docker: Error response from daemon: No such image", True),
        ("Try 'timeout --help' for more information.\n", True),  # a bad command line
        # Marker present: whatever happened next is the program's own doing.
        (RAN, False),
        (RAN + "Traceback (most recent call last): ...", False),
        # A program printing infra-looking text cannot fake an infra failure.
        (RAN + "docker: Error response from daemon: No such image", False),
    ],
)
def test_is_infra_failure(stderr: str, expected: bool) -> None:
    assert is_infra_failure(stderr) is expected


class FakeDocker:
    """Stands in for subprocess.run: a scripted `docker run`, recorded cleanup calls."""

    def __init__(
        self,
        run_outcome: "subprocess.CompletedProcess[str] | Exception",
        cleanup_hangs: bool = False,
    ) -> None:
        self.run_outcome = run_outcome
        self.cleanup_hangs = cleanup_hangs
        self.cleanup_calls: list[tuple[list[str], object]] = []

    def __call__(self, command: list[str], **kwargs: Any) -> "subprocess.CompletedProcess[str]":
        if command[1] == "run":
            if isinstance(self.run_outcome, Exception):
                raise self.run_outcome
            return self.run_outcome
        self.cleanup_calls.append((command, kwargs.get("timeout")))
        if self.cleanup_hangs:
            raise subprocess.TimeoutExpired(command, kwargs["timeout"])
        return subprocess.CompletedProcess(command, 0, "", "")


def _completed(
    returncode: int, stdout: str = "", stderr: str = RAN
) -> "subprocess.CompletedProcess[str]":
    return subprocess.CompletedProcess(["docker", "run"], returncode, stdout, stderr)


def test_unreachable_daemon_is_infra(monkeypatch: pytest.MonkeyPatch) -> None:
    """Docker 29 exits 1 when the daemon is down -- the marker's absence us what tells."""
    fake = FakeDocker(_completed(1, stderr="failed to connect to the docker API"))
    monkeypatch.setattr(subprocess, "run", fake)
    assert run_code("print(1)").infra_error is True


def test_program_exception_is_not_infra_and_marker_is_stripped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = FakeDocker(_completed(1, stderr=RAN + "ValueError: boom\n"))
    monkeypatch.setattr(subprocess, "run", fake)
    result = run_code("raise ValueError('boom')")
    assert result.infra_error is False
    assert result.stderr == "ValueError: boom\n"  # the program's stderr, without the marker


def test_program_timeout_is_not_infra(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(subprocess, "run", FakeDocker(_completed(124)))
    result = run_code("while True: pass")
    assert result.timed_out is True
    assert result.infra_error is False


def test_backstop_expiry_is_infra(monkeypatch: pytest.MonkeyPatch) -> None:
    expired = subprocess.TimeoutExpired(["docker", "run"], SUBPROCESS_TIMEOUT_SECONDS)
    monkeypatch.setattr(subprocess, "run", FakeDocker(expired))
    result = run_code("print(1)")
    assert result.infra_error is True
    assert result.returncode == -1


def test_hung_cleanup_neither_hangs_nor_raises(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    fake = FakeDocker(_completed(0, stdout="1\n"), cleanup_hangs=True)
    monkeypatch.setattr(subprocess, "run", fake)
    result = run_code("print(1)")
    assert result.returncode == 0
    assert result.infra_error is False
    assert "cleanup timed out" in caplog.text
