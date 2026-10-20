"""Tests for the reward bridge: _to_source, _compare, build_harness, run_tests.

The pure functions (_to_source, _compare, build_harness, _parse_result) run with
no sandbox and are always collected. The single run_tests test drives the real
Docker sandbox and is marked `integration`, matching the Phase 1 convention.
"""

from __future__ import annotations

import ast
import math

import pytest

from codeexec.rewards.harness import (
    RESULT_SENTINEL,
    _compare,
    _parse_result,
    _to_source,
    build_harness,
    run_tests,
)


# --------------------------------------------------------------------------- #
# _to_source: must produce valid Python source that evaluates back to the value
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "value",
    [
        5,
        -3,
        "hello",
        "with 'quotes' and \"both\"",
        3.14,
        True,
        False,
        None,
        [1, 2, 3],
        (1, 2, 3),
        (5,),  # single-element tuple: the trailing-comma case
        [],
        {},
        {"a": 1, "b": [2, 3]},
        [1, [2, [3, [4]]]],  # deep nesting
        3 + 4j,  # complex
    ],
)
def test_to_source_round_trips(value: object) -> None:
    """eval(_to_source(x)) reproduces x for ordinary values."""
    assert eval(_to_source(value)) == value


@pytest.mark.parametrize(
    ("value", "expected_source"),
    [
        (float("inf"), "float('inf')"),
        (float("-inf"), "float('-inf')"),
    ],
)
def test_to_source_non_finite_floats(value: float, expected_source: str) -> None:
    """repr('inf') is a NameError; _to_source must emit the float(...) form."""
    assert _to_source(value) == expected_source
    assert eval(_to_source(value)) == value


def test_to_source_nan() -> None:
    """nan round-trips structurally (nan != nan, so compare via isnan)."""
    back = eval(_to_source(float("nan")))
    assert isinstance(back, float) and math.isnan(back)


def test_to_source_nested_non_finite() -> None:
    """A non-finite float inside a container is still emitted as valid source."""
    src = _to_source([1.0, float("inf"), (float("-inf"),)])
    back = eval(src)
    assert back[0] == 1.0
    assert math.isinf(back[1]) and back[1] > 0
    assert math.isinf(back[2][0]) and back[2][0] < 0


# --------------------------------------------------------------------------- #
# _compare: the locked decisions, each as its own assertion
# --------------------------------------------------------------------------- #
def test_compare_int_equality() -> None:
    assert _compare(4, 4, 0.0)
    assert not _compare(4, 5, 0.0)


def test_compare_int_float_match_within_tolerance() -> None:
    # decision (1): int vs float match when values agree
    assert _compare(4, 4.0, 0.0)
    assert _compare(4.0, 4, 0.0)


def test_compare_float_tolerance() -> None:
    assert _compare(0.1 + 0.2, 0.3, 1e-6)
    assert not _compare(0.1 + 0.2, 0.3, 0.0)  # strict when no tolerance given
    assert _compare(1.0, 1.5, 0.6)
    assert not _compare(1.0, 1.5, 0.4)


def test_compare_bool_distinct_from_int() -> None:
    # decision (3): True is not 1, False is not 0
    assert not _compare(True, 1, 0.0)
    assert not _compare(1, True, 0.0)
    assert not _compare(False, 0, 0.0)
    assert _compare(True, True, 0.0)


def test_compare_list_vs_tuple_distinct() -> None:
    # decision (2): list and tuple never match each other
    assert not _compare([1, 2, 3], (1, 2, 3), 0.0)
    assert not _compare((1, 2, 3), [1, 2, 3], 0.0)
    assert _compare([1, 2, 3], [1, 2, 3], 0.0)
    assert _compare((1, 2, 3), (1, 2, 3), 0.0)


def test_compare_length_guard() -> None:
    # a shorter result must not "match" a longer expected via zip truncation
    assert not _compare([1, 2], [1, 2, 3], 0.0)
    assert not _compare([1, 2, 3], [1, 2], 0.0)


def test_compare_tolerance_recurses_into_structures() -> None:
    assert _compare([1.0, 2.0], [1.0000001, 2.0], 1e-5)
    assert not _compare([1.0, 2.0], [1.1, 2.0], 1e-5)
    assert _compare({"x": [1.0]}, {"x": [1.0000001]}, 1e-5)


def test_compare_dict_keys_exact() -> None:
    assert _compare({"a": 1, "b": 2}, {"a": 1, "b": 2}, 0.0)
    assert not _compare({"a": 1}, {"a": 1, "b": 2}, 0.0)


def test_compare_inf_equal_regression() -> None:
    """Regression: inf == inf must be True. abs(inf-inf) is nan, which would
    fail the tolerance check, so _compare tries exact equality first."""
    assert _compare(float("inf"), float("inf"), 0.0001)
    assert _compare(float("-inf"), float("-inf"), 0.0001)
    assert not _compare(float("inf"), float("-inf"), 0.0001)


def test_compare_nan_never_matches() -> None:
    """nan never equals nan, mirroring Python semantics."""
    assert not _compare(float("nan"), float("nan"), 1.0)


# --------------------------------------------------------------------------- #
# build_harness: must generate valid, runnable Python
# --------------------------------------------------------------------------- #
def test_build_harness_parses() -> None:
    """The generated program must be syntactically valid Python."""
    code = "def add(a, b):\n    return a + b"
    src = build_harness(code, "add", [((2, 3), 5), ((10, 20), 30)], 0.0)
    ast.parse(src)  # raises SyntaxError if the generator produced broken code


def test_build_harness_runs_and_reports() -> None:
    """Executing the harness (no Docker) prints the sentinel line with counts."""
    import contextlib
    import io

    code = "def add(a, b):\n    return a + b"
    src = build_harness(code, "add", [((2, 3), 5), ((10, 20), 30)], 0.0)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        exec(compile(src, "<harness>", "exec"), {})
    out = buf.getvalue()
    assert RESULT_SENTINEL in out
    assert '"passed": 2' in out
    assert '"total": 2' in out


def test_embedded_compare_matches_real_compare() -> None:
    """The _compare copy embedded in the harness must agree with the real one.
    Guards against the two drifting apart (the cost of embedding it as text)."""
    cases = [
        (4, 4.0, 0.0),
        (True, 1, 0.0),
        ([1, 2, 3], (1, 2, 3), 0.0),
        ([1, 2], [1, 2, 3], 0.0),
        (0.1 + 0.2, 0.3, 1e-6),
        (float("inf"), float("inf"), 0.0001),
    ]
    # Build a harness whose "function" just echoes a stored result, so the
    # embedded _compare is exercised on each case; compare its verdict to ours.
    for result, expected, atol in cases:
        code = f"def _f():\n    return {_to_source(result)}"
        src = build_harness(code, "_f", [((), expected)], atol)
        import contextlib
        import io

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            exec(compile(src, "<h>", "exec"), {})
        passed, crashed = _parse_result(buf.getvalue(), 1)
        embedded_verdict = passed == 1 and not crashed
        assert embedded_verdict == _compare(result, expected, atol), (
            f"embedded _compare disagrees on ({result!r}, {expected!r}, {atol})"
        )


# --------------------------------------------------------------------------- #
# _parse_result: sentinel extraction, including the adversarial cases
# --------------------------------------------------------------------------- #
def test_parse_result_basic() -> None:
    out = f'{RESULT_SENTINEL} {{"passed": 3, "total": 5}}'
    assert _parse_result(out, 5) == (3, False)


def test_parse_result_missing_sentinel_is_crash() -> None:
    assert _parse_result("some output\nno sentinel here", 5) == (0, True)


def test_parse_result_takes_last_valid_line() -> None:
    """Noise before the real line (incl. a fake sentinel) must not win."""
    out = (
        f'{RESULT_SENTINEL} {{"passed": 999, "total": 999}}\n'
        "noise\n"
        f'{RESULT_SENTINEL} {{"passed": 3, "total": 5}}'
    )
    assert _parse_result(out, 5) == (3, False)


def test_parse_result_skips_trailing_garbage() -> None:
    """Regression: a malformed trailing sentinel line must not mask a valid one."""
    out = f'{RESULT_SENTINEL} {{"passed": 2, "total": 5}}\n{RESULT_SENTINEL} THIS_IS_NOT_JSON'
    assert _parse_result(out, 5) == (2, False)


def test_parse_result_clamps_overcount() -> None:
    """A parseable-but-impossible count is clamped to total."""
    out = f'{RESULT_SENTINEL} {{"passed": 99, "total": 5}}'
    assert _parse_result(out, 5) == (5, False)


# --------------------------------------------------------------------------- #
# run_tests: the one test that drives the real Docker sandbox
# --------------------------------------------------------------------------- #
@pytest.mark.integration
def test_run_tests_end_to_end() -> None:
    """Full bridge through the real sandbox: correct, wrong, crash, syntax error."""
    tests = [((2, 3), 5), ((10, 20), 30)]

    good = run_tests("def add(a, b):\n    return a + b", "add", tests, 0.0)
    assert good.passed == 2 and good.total == 2
    assert good.all_passed and not good.crashed and not good.timed_out

    wrong = run_tests("def add(a, b):\n    return 0", "add", tests, 0.0)
    assert wrong.passed == 0 and not wrong.crashed

    runtime_crash = run_tests("def add(a, b):\n    raise ValueError", "add", [((2, 3), 5)], 0.0)
    assert runtime_crash.passed == 0 and not runtime_crash.crashed  # suite completed

    syntax_error = run_tests("def add(a, b)\n    return a+b", "add", [((2, 3), 5)], 0.0)
    assert syntax_error.crashed  # no sentinel printed


@pytest.mark.integration
def test_slow_case_fails_only_itself_not_whole_suite() -> None:
    """A single case that exceeds the per-case timeout fails alone; the other
    cases in the same suite still grade and the sentinel still prints.
    Guards the ADR-008 per-case-timeout fix (one slow stress input used to
    time out the whole process -> 0/N)."""
    code = (
        "import time\n"
        "def f(n):\n"
        "    if n == 99:\n"
        "        time.sleep(10)   # exceeds PER_CASE_TIMEOUT_SECONDS\n"
        "    return n * 2\n"
    )
    tests = [((1,), 2), ((2,), 4), ((99,), 198), ((3,), 6)]
    result = run_tests(code, "f", tests, 0.0)
    assert result.total == 4
    assert result.passed == 3  # the three fast cases pass
    assert not result.crashed  # sentinel printed; suite did NOT crash
    assert not result.timed_out  # outer process did not time out
