"""Tests for the reward bridge: _to_source, _compare, build_harness, run_tests.

The pure functions (_to_source, _compare, build_harness, _parse_result) run with
no sandbox and are always collected. The single run_tests test drives the real
Docker sandbox and is marked `integration`, matching the Phase 1 convention.
"""

from __future__ import annotations

import ast
import contextlib
import io
import math
from pathlib import Path

import pytest

from codeexec.rewards import oracle
from codeexec.rewards.harness import (
    RESULT_SENTINEL,
    SandboxInfraError,
    TestCases,
    _parse_result,
    _to_source,
    build_harness,
    run_tests,
)
from codeexec.sandbox import runner


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


def _run_in_process(code: str, entry_point: str, tests: TestCases, atol: float) -> int:
    """Run a generated harness here, without Docker, and return how many tests passed."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        exec(compile(build_harness(code, entry_point, tests, atol), "<harness>", "exec"), {})
    passed, crashed = _parse_result(buf.getvalue(), len(tests))
    assert not crashed
    return passed


def test_the_embedded_rule_is_oracle_py() -> None:
    """One copy of the pass rule (ADR-015): the sandbox runs oracle.py's own source."""
    src = build_harness("def f(x):\n    return x", "f", [((1,), 1)], 0.0)
    assert Path(oracle.__file__).read_text() in src


def test_float_rounding_passes_as_in_evalplus() -> None:
    """Mbpp/234: l*l*l differs from the expected l**3 in the last digit; EvalPlus passes it."""
    code = "def volume_cube(l):\n    return l*l*l"
    tests = [((1.768116407397588,), 5.527548510985869), ((3,), 27)]
    assert _run_in_process(code, "volume_cube", tests, 0.0) == 2


def test_special_oracles_get_the_entry_point_and_the_arguments() -> None:
    """Set equality needs the entry point; a reference recomputation needs the args."""
    reversed_order = "def find_char_long(text):\n    return ['quick', 'brown'][::-1]"
    assert (
        _run_in_process(reversed_order, "find_char_long", [(("x",), ["quick", "brown"])], 0.0) == 1
    )
    by_reference = "def digit_distance_nums(n1, n2):\n    return 7"
    tests = [((123, 256), 999)]  # 7 is the reference answer for (123, 256), whatever is stored
    assert _run_in_process(by_reference, "digit_distance_nums", tests, 0.0) == 1


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


@pytest.mark.integration
def test_real_infra_failure_raises_instead_of_scoring(monkeypatch: pytest.MonkeyPatch) -> None:
    """A real sandbox failure (missing image) is retried once, then raised: never scored."""
    monkeypatch.setattr(runner, "IMAGE", "codeexec-sandbox-does-not-exist")
    with pytest.raises(SandboxInfraError, match="No such image"):
        run_tests("def add(a, b):\n    return a + b", "add", [((2, 3), 5)], 0.0)


@pytest.mark.integration
def test_the_sandbox_applies_evalplus_s_rule() -> None:
    """ADR-015 inside the real container: float rounding and a special oracle."""
    rounding = run_tests(
        "def volume_cube(l):\n    return l*l*l",
        "volume_cube",
        [((1.768116407397588,), 5.527548510985869)],
        0.0,
    )
    assert rounding.all_passed
    any_order = run_tests(
        "def find_char_long(text):\n    return ['brown', 'quick']",
        "find_char_long",
        [(("the quick brown",), ["quick", "brown"])],
        0.0,
    )
    assert any_order.all_passed
