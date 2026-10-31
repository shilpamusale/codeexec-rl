"""Tests for the reward bridge: _to_source, build_harness, run_tests.

The pure functions (_to_source, build_harness, _parse_result) run with no sandbox
and are always collected. Tests that drive the real Docker sandbox are marked
`integration`, matching the Phase 1 convention; the one that writes every
dataset problem into a harness is marked `dataset`.
"""

from __future__ import annotations

import ast
import contextlib
import io
import math
import subprocess
import sys
import warnings
from pathlib import Path

import pytest

from codeexec.rewards import harness, oracle
from codeexec.rewards.harness import (
    RESULT_SENTINEL,
    SandboxInfraError,
    TestCases,
    TestResult,
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


def test_to_source_writes_huge_ints_in_hex() -> None:
    """repr() refuses ints of more than 4,300 digits in Python 3.11 (HumanEval/83,
    HumanEval/139); hex has no limit and evaluates back to the same int."""
    for value in (18 * 10**5000, -(7**9000)):
        source = _to_source(value)
        assert source.lstrip("-").startswith("0x")
        assert eval(source) == value
    assert _to_source([10**5000]) == f"[{hex(10**5000)}]"  # nested, too


def test_to_source_keeps_ordinary_ints_decimal() -> None:
    assert _to_source(-42) == "-42"
    assert _to_source(10**3999) == repr(10**3999)  # 4,000 digits: still decimal


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
    assert '"case_timeouts": 0' in out


def _run_in_process(code: str, entry_point: str, tests: TestCases, atol: float) -> int:
    """Run a generated harness here, without Docker, and return how many tests passed."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        exec(compile(build_harness(code, entry_point, tests, atol), "<harness>", "exec"), {})
    passed, crashed, _ = _parse_result(buf.getvalue(), len(tests))
    assert not crashed
    return passed


def test_a_case_over_its_budget_is_counted_as_a_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    """A case stopped by the per-case alarm fails, and is reported as a timeout,
    apart from a wrong answer (ADR-008 status update). Budget lowered to 1s.
    """
    monkeypatch.setattr(harness, "PER_CASE_TIMEOUT_SECONDS", 1)
    code = "def f(n):\n    while n == 99:\n        pass\n    return n * 2"
    tests = [((1,), 2), ((2,), 5), ((99,), 198), ((3,), 6)]  # (2,) is a wrong answer
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        exec(compile(build_harness(code, "f", tests, 0.0), "<harness>", "exec"), {})
    assert _parse_result(buf.getvalue(), len(tests)) == (2, False, 1)


def test_time_limited_means_a_time_limit_fired() -> None:
    assert TestResult(3, 4, timed_out=False, crashed=False, case_timeouts=1).time_limited
    assert TestResult(0, 4, timed_out=True, crashed=False).time_limited
    assert not TestResult(3, 4, timed_out=False, crashed=False).time_limited


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


def test_huge_int_expected_output_is_graded() -> None:
    """HumanEval/83's shape: an expected output thousands of digits long."""
    code = "def starts_one_ends(n):\n    return 18 * 10 ** (n - 2)"
    tests = [((5000,), 18 * 10**4998), ((2,), 18)]
    assert _run_in_process(code, "starts_one_ends", tests, 0.0) == 2


def _run_as_the_sandbox_does(code: str, tests: TestCases) -> subprocess.CompletedProcess[str]:
    """The harness fed to a fresh Python over stdin, as the sandbox feeds it, minus Docker."""
    harness_src = build_harness(code, "add", tests, 0.0)
    return subprocess.run(
        [sys.executable, "-"], input=harness_src, capture_output=True, text=True, timeout=30
    )


ADDS = [((1, 2), 3), ((3, 4), 7)]


@pytest.mark.parametrize(
    "prints",
    [
        "    print(a, end=' ')\n",  # no final newline: it used to glue onto the result line
        "    print('adding', a, b)\n    print('to stderr', file=sys.stderr)\n",
        "    sys.stdout.write('partial')\n",
        "    sys.stdout = None\n",  # rebinding stdout cannot lose the result line
    ],
)
def test_the_program_s_own_output_is_swallowed(prints: str) -> None:
    """ADR-015 status update (C2): each test call writes to a sink, as in EvalPlus, so a
    correct program that prints passes, and the result line is all that comes out."""
    code = f"def add(a, b):\n    import sys\n{prints}    return a + b\n"
    out = _run_as_the_sandbox_does(code, ADDS)
    assert _parse_result(out.stdout, len(ADDS)) == (2, False, 0)
    assert len(out.stdout.splitlines()) == 1
    assert out.stderr == ""


def test_reading_stdin_fails_the_test_as_in_evalplus() -> None:
    """EvalPlus points stdin at a stream that raises when read; so does the harness."""
    code = "def add(a, b):\n    import sys\n    sys.stdin.read()\n    return a + b\n"
    out = _run_as_the_sandbox_does(code, ADDS)
    assert _parse_result(out.stdout, len(ADDS)) == (0, False, 0)


# Two HumanEval+ problems store expected outputs too large for the sandbox to run
# at all: their harnesses are about 147 MB and 124 MB of source, and the sandbox
# has 256 MB in all. They are decided when HumanEval+ is first measured (Phase 5);
# here they are built but not compiled, which would take gigabytes. Every other
# harness, MBPP's and HumanEval's, is under 10 MB.
OVERSIZED_HARNESSES = frozenset({"HumanEval/15", "HumanEval/130"})
MAX_HARNESS_BYTES = 16 * 2**20


@pytest.mark.dataset
def test_every_stored_test_can_be_written_into_a_harness() -> None:
    """Every test of both datasets becomes harness source, the ints of more than
    4,300 digits in HumanEval/83 and HumanEval/139 included, and every harness of
    a runnable size compiles: _to_source never writes a value as invalid Python."""
    from codeexec.tasks.datasets import ood_problems, training_problems

    for task_id, p in {**training_problems(), **ood_problems()}.items():
        tests = [(t.args, t.expected) for t in (*p.visible_tests, *p.heldout_tests)]
        source = build_harness(p.canonical_solution, p.entry_point, tests, p.atol)
        if task_id in OVERSIZED_HARNESSES:
            assert len(source) > MAX_HARNESS_BYTES, f"{task_id} is no longer oversized"
            continue
        assert len(source) <= MAX_HARNESS_BYTES, f"{task_id}: {len(source):,} bytes"

        # Some canonicals' regexes use invalid escapes (Mbpp/305, Mbpp/593): the
        # dataset's, not ours to fix, and silenced as the loader silences them.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            compile(source, task_id, "exec")


# --------------------------------------------------------------------------- #
# _parse_result: sentinel extraction, including the adversarial cases
# --------------------------------------------------------------------------- #
def test_parse_result_basic() -> None:
    out = f'{RESULT_SENTINEL} {{"passed": 3, "total": 5}}'
    assert _parse_result(out, 5) == (3, False, 0)


def test_parse_result_missing_sentinel_is_crash() -> None:
    assert _parse_result("some output\nno sentinel here", 5) == (0, True, 0)


def test_parse_result_takes_last_valid_line() -> None:
    """Noise before the real line (incl. a fake sentinel) must not win."""
    out = (
        f'{RESULT_SENTINEL} {{"passed": 999, "total": 999}}\n'
        "noise\n"
        f'{RESULT_SENTINEL} {{"passed": 3, "total": 5}}'
    )
    assert _parse_result(out, 5) == (3, False, 0)


def test_parse_result_skips_trailing_garbage() -> None:
    """Regression: a malformed trailing sentinel line must not mask a valid one."""
    out = f'{RESULT_SENTINEL} {{"passed": 2, "total": 5}}\n{RESULT_SENTINEL} THIS_IS_NOT_JSON'
    assert _parse_result(out, 5) == (2, False, 0)


def test_parse_result_clamps_overcount() -> None:
    """A parseable-but-impossible count is clamped to total."""
    out = f'{RESULT_SENTINEL} {{"passed": 99, "total": 5}}'
    assert _parse_result(out, 5) == (5, False, 0)


def test_parse_result_reads_case_timeouts_and_clamps_them() -> None:
    """Timed-out cases are failed cases, so there can be at most total - passed."""
    out = f'{RESULT_SENTINEL} {{"passed": 3, "total": 5, "case_timeouts": 2}}'
    assert _parse_result(out, 5) == (3, False, 2)
    out = f'{RESULT_SENTINEL} {{"passed":3, "total":5, "case_timeouts": 9}}'
    assert _parse_result(out, 5) == (3, False, 2)


# --------------------------------------------------------------------------- #
# run_tests: tests that drive the real Docker sandbox
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
    assert result.case_timeouts == 1  # the slow case is reported as a timeout
    assert result.time_limited


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


@pytest.mark.integration
def test_the_sandbox_grades_a_huge_int() -> None:
    """The container's Python reads the hex literal back: HumanEval/83's shape."""
    result = run_tests(
        "def starts_one_ends(n):\n    return 18 * 10 ** (n - 2)",
        "starts_one_ends",
        [((5000,), 18 * 10**4998)],
        0.0,
    )
    assert result.all_passed
