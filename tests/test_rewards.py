"""Tests for the reward functions r_sparse and r_shaped.

Two kinds:
  - Pure-logic tests monkeypatch run_tests to return a fixed TestResult, so the
    reward MATH is tested without Docker (fast, deterministic).
  - Integration tests drive the real sandbox end-to-end, marked `integration`.
"""

from __future__ import annotations

import pytest

from codeexec.rewards import functions
from codeexec.rewards.functions import r_shaped, r_sparse
from codeexec.rewards.harness import TestResult

# A syntactically-valid TestCases value. Content is irrelevant in these tests
# because run_tests is stubbed; it only needs to type-check as TestCases.
_DUMMY_TESTS: list[tuple[tuple[object, ...], object]] = [((1,), 1)]


# --------------------------------------------------------------------------- #
# Pure-logic: reward math, with run_tests stubbed to a fixed TestResult
# --------------------------------------------------------------------------- #
def _stub_run_tests(monkeypatch: pytest.MonkeyPatch, result: TestResult) -> None:
    """Make functions.run_tests return `result` regardless of inputs."""
    monkeypatch.setattr(functions, "run_tests", lambda *a, **k: result)


def test_r_sparse_all_pass(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_run_tests(monkeypatch, TestResult(passed=3, total=3, timed_out=False, crashed=False))
    assert r_sparse("x", "f", _DUMMY_TESTS, 0.0) == 1.0


def test_r_sparse_partial_is_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    # sparse is all-or-nothing: 2 of 3 is still 0.0
    _stub_run_tests(monkeypatch, TestResult(passed=2, total=3, timed_out=False, crashed=False))
    assert r_sparse("x", "f", _DUMMY_TESTS, 0.0) == 0.0


def test_r_sparse_crash_is_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_run_tests(monkeypatch, TestResult(passed=0, total=3, timed_out=False, crashed=True))
    assert r_sparse("x", "f", _DUMMY_TESTS, 0.0) == 0.0


def test_r_sparse_timeout_is_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_run_tests(monkeypatch, TestResult(passed=0, total=3, timed_out=True, crashed=False))
    assert r_sparse("x", "f", _DUMMY_TESTS, 0.0) == 0.0


def test_r_shaped_fraction(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_run_tests(monkeypatch, TestResult(passed=3, total=5, timed_out=False, crashed=False))
    assert r_shaped("x", "f", _DUMMY_TESTS, 0.0) == pytest.approx(0.6)


def test_r_shaped_all_pass_is_one(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_run_tests(monkeypatch, TestResult(passed=4, total=4, timed_out=False, crashed=False))
    assert r_shaped("x", "f", _DUMMY_TESTS, 0.0) == 1.0


def test_r_shaped_crash_is_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_run_tests(monkeypatch, TestResult(passed=0, total=5, timed_out=False, crashed=True))
    assert r_shaped("x", "f", _DUMMY_TESTS, 0.0) == 0.0


def test_r_shaped_timeout_is_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_run_tests(monkeypatch, TestResult(passed=0, total=5, timed_out=True, crashed=False))
    assert r_shaped("x", "f", _DUMMY_TESTS, 0.0) == 0.0


def test_r_shaped_run_bonus_off_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    # ran, passed 0, but no bonus requested -> pure fraction = 0.0
    _stub_run_tests(monkeypatch, TestResult(passed=0, total=5, timed_out=False, crashed=False))
    assert r_shaped("x", "f", _DUMMY_TESTS, 0.0) == 0.0


def test_r_shaped_run_bonus_lifts_zero_pass(monkeypatch: pytest.MonkeyPatch) -> None:
    # ran, passed 0, bonus on -> lifted to the bonus value
    _stub_run_tests(monkeypatch, TestResult(passed=0, total=5, timed_out=False, crashed=False))
    assert r_shaped("x", "f", _DUMMY_TESTS, 0.0, run_bonus=0.1) == pytest.approx(0.1)


def test_r_shaped_run_bonus_does_not_lower_real_fraction(monkeypatch: pytest.MonkeyPatch) -> None:
    # a real 3/5 = 0.6 must not be dragged DOWN to a 0.1 bonus floor
    _stub_run_tests(monkeypatch, TestResult(passed=3, total=5, timed_out=False, crashed=False))
    assert r_shaped("x", "f", _DUMMY_TESTS, 0.0, run_bonus=0.1) == pytest.approx(0.6)


def test_r_shaped_run_bonus_not_applied_to_crash(monkeypatch: pytest.MonkeyPatch) -> None:
    # crash is hard zero even with bonus on (code did not run cleanly)
    _stub_run_tests(monkeypatch, TestResult(passed=0, total=5, timed_out=False, crashed=True))
    assert r_shaped("x", "f", _DUMMY_TESTS, 0.0, run_bonus=0.1) == 0.0


# --------------------------------------------------------------------------- #
# Integration: real sandbox, confirming the reward functions over the full path
# --------------------------------------------------------------------------- #
@pytest.mark.integration
def test_rewards_end_to_end() -> None:
    tests = [((2, 3), 5), ((10, 20), 30)]
    good = "def add(a, b):\n    return a + b"
    # passes (2,3) but not (10,20): 1 of 2
    half = "def add(a, b):\n    return a + b if a < 5 else 0"

    assert r_sparse(good, "add", tests, 0.0) == 1.0
    assert r_sparse(half, "add", tests, 0.0) == 0.0
    assert r_shaped(good, "add", tests, 0.0) == 1.0
    assert r_shaped(half, "add", tests, 0.0) == pytest.approx(0.5)
