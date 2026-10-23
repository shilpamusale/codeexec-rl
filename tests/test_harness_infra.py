"""Unit tests for run_tests infratructure-failure handling (ADR-014).

No Docker needed: run_code is replaced by a scripted fake, so these run in
CI. The real-sandbox counterpart is in test_harness.py (integration).
"""

import json
import time

import pytest

from codeexec.rewards import harness
from codeexec.rewards.harness import (
    INFRA_RETRY_DELAY_SECONDS,
    RESULT_SENTINEL,
    SandboxInfraError,
    run_tests,
)
from codeexec.sandbox.runner import RawResult

ADD = "def add(a, b):\n     return a + b"
TESTS = [((2, 3), 5)]

PASS = RawResult(
    stdout=f"{RESULT_SENTINEL} {json.dumps({'passed': 1, 'total': 1})}\n", stderr="", returncode=0
)
INFRA = RawResult(
    stdout="",
    stderr="docker: Error response from daemon: No such image",
    returncode=125,
    infra_error=True,
)

CRASH = RawResult(stdout="", stderr="SyntaxError: invalid syntax", returncode=1)
TIMEOUT = RawResult(stdout="", stderr="", returncode=124, timed_out=True)


class FakeSandbox:
    """Stands in for run_code: returns scripted results in order, records each call."""

    def __init__(self, *results: RawResult) -> None:
        self.results = list(results)
        self.calls: list[str] = []

    def __call__(self, code: str) -> RawResult:
        self.calls.append(code)
        return self.results.pop(0)  # an unscripted extra call fails the test loudly


def _install(monkeypatch: pytest.MonkeyPatch, *results: RawResult) -> FakeSandbox:
    fake = FakeSandbox(*results)
    monkeypatch.setattr(harness, "run_code", fake)
    return fake


@pytest.fixture
def sleeps(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """Record time.sleep calls instead of sleeping."""
    recorded: list[float] = []
    monkeypatch.setattr(time, "sleep", recorded.append)
    return recorded


def test_clean_run_is_not_retried(monkeypatch: pytest.MonkeyPatch, sleeps: list[float]) -> None:
    fake = _install(monkeypatch, PASS)
    result = run_tests(ADD, "add", TESTS, 0.0)

    assert result.all_passed
    assert result.infra_retries == 0
    assert len(fake.calls) == 1
    assert sleeps == []


def test_transient_infra_failure_is_retried_once(
    monkeypatch: pytest.MonkeyPatch, sleeps: list[float], caplog: pytest.LogCaptureFixture
) -> None:
    fake = _install(monkeypatch, INFRA, PASS)
    result = run_tests(ADD, "add", TESTS, 0.0)
    assert result.all_passed  # the score comes from the retry, never from the failure
    assert result.infra_retries == 1
    assert len(fake.calls) == 2
    assert fake.calls[0] == fake.calls[1]  # the same program, rerun unchanged
    assert sleeps == [INFRA_RETRY_DELAY_SECONDS]
    assert "retrying once" in caplog.text


def test_persistent_infra_failure_instead_of_scoring(
    monkeypatch: pytest.MonkeyPatch, sleeps: list[float]
) -> None:
    fake = _install(monkeypatch, INFRA, INFRA)
    with pytest.raises(SandboxInfraError, match="No such image"):
        run_tests(ADD, "add", TESTS, 0.0)

    assert len(fake.calls) == 2  # exactly one retry, never a third attempt


@pytest.mark.parametrize("program_failure", [CRASH, TIMEOUT])
def test_program_failure_is_never_retried(
    monkeypatch: pytest.MonkeyPatch, sleeps: list[float], program_failure: RawResult
) -> None:
    fake = _install(monkeypatch, program_failure)
    result = run_tests(ADD, "add", TESTS, 0.0)
    assert result.passed == 0
    assert result.infra_retries == 0
    assert len(fake.calls) == 1
    assert sleeps == []
