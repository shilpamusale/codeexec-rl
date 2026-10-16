"""Tests for the perturbation driver (perturb) and dataset builder."""

from __future__ import annotations

import pytest

from codeexec.perturb import datasets, generator
from codeexec.perturb.generator import _seed_for, perturb
from codeexec.tasks.problem import Problem, TestCase


def _mk(
    task_id: str = "T/1",
    entry_point: str = "add",
    canonical: str = "def add(a, b):\n    return a + b",
) -> Problem:
    return Problem(
        task_id=task_id,
        source="mbpp",
        prompt=f'"""\nWrite a function to add two numbers.\nassert {entry_point}(2,3) == 5\n"""',
        entry_point=entry_point,
        canonical_solution=canonical,
        visible_tests=(TestCase(args=(2, 3), expected=5),),
        heldout_tests=(TestCase(args=(10, 20), expected=30),),
        atol=0.0,
    )


# --- seeding ---
def test_seed_is_deterministic_and_per_problem() -> None:
    assert _seed_for(42, "Mbpp/3") == _seed_for(42, "Mbpp/3")  # stable
    assert _seed_for(42, "Mbpp/3") != _seed_for(42, "Mbpp/4")  # per-problem
    assert _seed_for(1, "Mbpp/3") != _seed_for(2, "Mbpp/3")  # per-base-seed


# --- perturb, with the gate stubbed (no Docker) ---
def _stub_gate(monkeypatch: pytest.MonkeyPatch, passes: bool) -> None:
    from codeexec.rewards.harness import TestResult

    result = TestResult(passed=(99 if passes else 0), total=99, timed_out=False, crashed=not passes)
    monkeypatch.setattr(generator, "run_tests", lambda *a, **k: result)


def test_perturb_returns_problem_when_gate_passes(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_gate(monkeypatch, passes=True)
    out = perturb(_mk(), base_seed=42)
    assert out is not None
    assert out.entry_point != "add"


def test_perturb_returns_none_when_gate_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    """If the perturbed canonical fails its own tests, perturb drops it (None)."""
    _stub_gate(monkeypatch, passes=False)
    out = perturb(_mk(), base_seed=42)
    assert out is None


def test_perturb_is_deterministic(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_gate(monkeypatch, passes=True)
    a = perturb(_mk(), base_seed=42)
    b = perturb(_mk(), base_seed=42)
    assert a is not None and b is not None
    assert a.entry_point == b.entry_point
    assert a.prompt == b.prompt


def test_perturb_preserves_tests(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_gate(monkeypatch, passes=True)
    p = _mk()
    out = perturb(p, base_seed=42)
    assert out is not None
    assert out.visible_tests == p.visible_tests
    assert out.heldout_tests == p.heldout_tests


# --- the builder, with training_problems + gate stubbed ---
def test_perturbed_problems_keeps_survivors(monkeypatch: pytest.MonkeyPatch) -> None:
    small = {
        "Mbpp/a": _mk("Mbpp/a"),
        "Mbpp/b": _mk(
            "Mbpp/b",
            "mul",
            "def mul(a,b):\n   return a * b",
        ),
    }
    monkeypatch.setattr(datasets, "training_problems", lambda: small)
    _stub_gate(monkeypatch, passes=True)
    out = datasets.perturbed_problems(base_seed=42)
    assert set(out.keys()) == {"Mbpp/a", "Mbpp/b"}


def test_perturbed_problems_drops_gate_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    small = {"Mbpp/a": _mk("Mbpp/a")}
    monkeypatch.setattr(datasets, "training_problems", lambda: small)
    _stub_gate(monkeypatch, passes=False)
    out = datasets.perturbed_problems(base_seed=42)
    assert out == {}


# --- integration: the gate actually runs the sandbox and catches a real break ---
@pytest.mark.integration
def test_gate_catches_broken_canonical() -> None:
    """A problem whose canonical does NOT satisfy its own tests must be dropped
    by the real gate (proves the gate validates, not rubber-stamps)."""
    # canonical returns a + b + 1, so it fails its own (2,3)->5 test
    broken = _mk(canonical="def add(a, b):\n    return a + b + 1")
    assert perturb(broken, base_seed=42) is None


@pytest.mark.integration
def test_perturb_end_to_end_valid() -> None:
    """A valid problem perturbs and passes the real sandbox gate."""
    out = perturb(_mk(), base_seed=42)
    assert out is not None
    assert out.entry_point != "add"
    assert out.visible_tests == _mk().visible_tests
