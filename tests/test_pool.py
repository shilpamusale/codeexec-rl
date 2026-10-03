"""Integration tests for the async execution pool."""

import asyncio

import pytest

from codeexec.sandbox.pool import run_many

pytestmark = pytest.mark.integration


def test_run_many_returns_all_results_in_order() -> None:
    """Every snippet runs and results come back in input order, not finish order."""
    codes = [f"print({i})" for i in range(6)]
    results = asyncio.run(run_many(codes, concurrency=4))

    assert len(results) == 6
    for i, result in enumerate(results):
        assert result.returncode == 0
        assert str(i) in result.stdout


def test_run_many_handles_empty_input() -> None:
    """An empty batch returns an empty list without error."""
    results = asyncio.run(run_many([], concurrency=4))
    assert results == []
