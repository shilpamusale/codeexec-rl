import json
import logging
import math
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from codeexec.rewards import oracle
from codeexec.sandbox.runner import RawResult, run_code

logger = logging.getLogger(__name__)

# A test case as the harness consumes it: (positional args, expected result).
# Sequence (not list) so callers can pass a list of concretely-typed tuples:
# Sequence is covariant, list is not. We only ever read `tests`, never append.
TestCases = Sequence[tuple[tuple[object, ...], object]]

# The sentinel: an ugly, versioned marker. Vanishingly unlikely to appear in the
# model output by accident, so the parser can find OUR result line in a dirty
# stdout stream. A structured JSON payload means a collision fails loudly at
# json.loads rahther than silently returning a wrong count.

RESULT_SENTINEL = "###CODEEXEC_RESULT_a3f9###"

# Per-case wall-clock budget inside the harness. A single slow test case fails
# only that case (SIGALRM -> _CaseTimeout -> counted as a failure), not the whole
# suite, and the harness reports how many cases ran out of time, so a caller can
# tell a slow failure from a wrong answer (ADR-008 status update). The outer
# sandbox timeout in run_code remains as a hard backstop for what SIGALRM can't
# interrupt (C-level calls, huge allocations). See DECISIONS.md ADR-008.
PER_CASE_TIMEOUT_SECONDS = 2

# ADR-014: an infrastructure failuire is retried once, after a short pause
# that gives a briefly overloaded or restarting Docker daemon time to recover.
# If the retry fails too, run_tests raises instead of returning a score.
INFRA_RETRY_DELAY_SECONDS = 1.0

# Python 3.11 refuses to convert an int of more than 4,300 decimal digits to or
# from a decimal string, so repr() raises on the expected outputs of HumanEval/83
# and HumanEval/139. Hexadecimal is exempt both ways and reads back exactly. The
# threshold is fixed here, not read from the host's limit, so the sandbox can
# always read what the host writes.
HEX_INT_THRESHOLD = 10**4000


class SandboxInfraError(RuntimeError):
    """The sandbox failed twice in a row, so the program never ran (ADR-014).

    Raised instead of returning a score. An infrastructure failure says nothing
    about the program, and scoring it 0.0 would push the policy away from code
    that may be correct. Training stops loudly instead.
    """


# The pass rule (ADR-015), embedded verbatim: oracle.py's own source, so the
# sandbox and the host run one copy of it. It must stay stdlib-only, and must
# never gain a `from __future__` import, which is legal only at the very top of a
# program; test_build_harness_parses would catch that.
_ORACLE_SRC = Path(oracle.__file__).read_text()

# The per-case runner, embedded after the rule. Stdlib-only and
# underscore-prefixed, like the rule, so the model's code cannot collide with it.
_HARNESS_RUNNER_SRC = """
import signal as _signal

class _CaseTimeout(Exception):
    pass

def _alarm_handler(_signum, _frame):
    raise _CaseTimeout()

_signal.signal(_signal.SIGALRM, _alarm_handler)

def _run_one_case(_fn, _args, _expected, _atol, _entry_point, _budget):
    # Arm a per-case alarm; cancel it the instant the case returns so a fast
    # case never leaves a pending alarm to fire during the next one.
    _signal.alarm(_budget)
    try:
        _result = _fn(*_args)
        return "pass" if _passes(_result, _expected, _atol, _entry_point, _args) else "fail"
    except _CaseTimeout:
        # Out of time: a failure, reported apart from a wrong answer (ADR-008).
        return "timeout"
    except BaseException:
        # Any other failure (a raise or an incomparable output) fails this case only.
        return "fail"
    finally:
        _signal.alarm(0)
"""


@dataclass(frozen=True)
class TestResult:
    """Outcome of running one code string against one test set."""

    __test__ = False  # not a pytest test class
    passed: int
    total: int
    timed_out: bool
    crashed: bool  # True when no valid result line came back (e.g. syntax error)

    # How many retries an infrastructure failure needed before this result came
    # back (0 or 1). Not evidence about the program: training logs it as a metric.
    infra_retries: int = 0

    # How many test cases ran out of their per-case budget (ADR-008). Each
    # is also counted as failed; the count tells a slow failure from a wrong one.
    case_timeouts: int = 0

    @property
    def time_limited(self) -> bool:
        """
        A time limit fired: a case ran out of its budget, or the program timed out.

        Load can push a correct but slow program over a limit, so these verdicts
        are the only ones that can depend on load (ADR-008 status update).
        """
        return self.timed_out or self.case_timeouts > 0

    @property
    def fraction(self) -> float:
        return self.passed / self.total if self.total else 0.0

    @property
    def all_passed(self) -> bool:
        return self.total > 0 and self.passed == self.total


def _to_source(value: object) -> str:
    """Return valid Python source text that evaluates back to `value`."""

    # bool is a subclass of int, but its repr ('True'/'False') is already
    # valid source, so handle it here before the float/int checks below.
    if isinstance(value, bool):
        return repr(value)

    # The one case repr() gets wrong: non-finite floats.
    # repr(float('inf)) == 'inf' is a NameError, not a literal.
    if isinstance(value, float):
        if math.isinf(value):
            return "float('-inf')" if value < 0 else "float('inf')"

        if math.isnan(value):
            return "float('nan')"
        return repr(value)

    # A huge int: written in hex, which no digit limit applies to.
    if isinstance(value, int) and abs(value) >= HEX_INT_THRESHOLD:
        return hex(value)

    # Containers: recurse, so a non-finite float nested inside still gets fixed.
    if isinstance(value, list):
        return "[" + ", ".join(_to_source(v) for v in value) + "]"

    if isinstance(value, tuple):
        inner = ", ".join(_to_source(v) for v in value)
        if len(value) == 1:
            inner += ","
        return "(" + inner + ")"
    if isinstance(value, dict):
        items = ", ".join(f"{_to_source(k)}:{_to_source(v)}" for k, v in value.items())
        return "{" + items + "}"

    # str, smaller ints, complex, None and everything else: repr is valid source.
    return repr(value)


def build_harness(code: str, entry_point: str, tests: TestCases, atol: float) -> str:
    """Return a self-contained Python program that runs `code` against `tests`
    and prints one sentinel JSON line: {"passed": int, "total": int}.

    `tests` is a list of (args_tuple, expected) pairs.
    """
    lines: list[str] = []

    # --- section 1: setup ---
    lines.append("import json as _json")
    lines.append("import warnings as _warnings")
    lines.append("_warnings.filterwarnings('ignore')")

    # --- section 2: the embedded grader ---
    lines.append(_ORACLE_SRC)
    lines.append(_HARNESS_RUNNER_SRC)

    # --- section 3: the model's code (defines the entry_point function) ---
    lines.append("# --- model-generated code ---")
    lines.append(code)

    # --- section 4: the test harness ---
    lines.append("# --- test harness ---")
    lines.append(f"_atol = {_to_source(atol)}")
    lines.append("_cases = [")
    for args, expected in tests:
        lines.append(f"    ({_to_source(tuple(args))}, {_to_source(expected)}),")
    lines.append("]")
    lines.append(f"_fn = {entry_point}")
    lines.append(f"_entry_point = {entry_point!r}")
    lines.append(f"_budget = {PER_CASE_TIMEOUT_SECONDS}")
    lines.append("_passed = 0")
    lines.append("_case_timeouts = 0")
    lines.append("for _args, _expected in _cases:")
    lines.append(
        "    _outcome = _run_one_case(_fn, _args, _expected, _atol, _entry_point, _budget)"
    )
    lines.append("    if _outcome == 'pass':")
    lines.append("        _passed += 1")
    lines.append("    elif _outcome == 'timeout':")
    lines.append("        _case_timeouts += 1")
    lines.append(
        f"print('{RESULT_SENTINEL} ' + _json.dumps("
        "{'passed': _passed, 'total': len(_cases), 'case_timeouts': _case_timeouts}))"
    )

    return "\n".join(lines)


def _parse_result(stdout: str, total: int) -> tuple[int, bool, int]:
    """Find the sentinel line. Returns (passed, crashed, case_timeouts).

    Scans from the LAST line backward, so trailing model noise can't shadow the
    real result, and skips any sentinel line whose payload won't parse — so a
    garbage line can't mask a valid result earlier in the stream. crashed=True
    means no valid result line was found at all -> treat as 0 passed.
    A payload without `case_timeouts` reads as 0.

    """
    for line in reversed(stdout.splitlines()):
        line = line.strip()
        if line.startswith(RESULT_SENTINEL):
            payload = line[len(RESULT_SENTINEL) :].strip()
            try:
                data = json.loads(payload)
                # clamp defensively: passed can never exceed total or go negative,
                # and a timed-out case is a failed one, so at most total - passed
                passed = min(max(int(data["passed"]), 0), total)
                timeouts = min(max(int(data.get("case_timeouts", 0)), 0), total - passed)
                return passed, False, timeouts
            except (json.JSONDecodeError, KeyError, ValueError, TypeError):
                continue  # malformed sentinel line: keep scanning for a valid one
    return 0, True, 0  # no valid sentinel found: the harness never reported


def _run_with_one_retry(harness: str) -> tuple[RawResult, int]:
    """Run `harness` in the sandbox, retrying once on an infra failure (ADR-014).

    Returns the raw result and how many retries it took (0 or 1). A program failure
    (wrong answer, crash, timeout) is never retried: it is evidence about the program,
    and rerunning it would only double the cost of a wrong answer.
    """

    raw = run_code(harness)
    if not raw.infra_error:
        return raw, 0
    logger.warning("sandbox infrastructure failure, retrying once: %s", raw.stderr.strip()[:200])
    time.sleep(INFRA_RETRY_DELAY_SECONDS)
    raw = run_code(harness)
    if raw.infra_error:
        raise SandboxInfraError(
            f"sandbox failed twice in a row; the program never ran: {raw.stderr.strip()[:500]}"
        )
    return raw, 1


def run_tests(
    code: str,
    entry_point: str,
    tests: TestCases,
    atol: float,
) -> TestResult:
    """Run `code` against `tests` in the Phase 1 sandbox and return a TestResult.

    Raises SandboxInfraError if the sandbox fails twice in a row: an infrastructure
    failure is never turned into a score (ADR-014).
    """
    total = len(tests)
    if total == 0:
        # No tests to run. Not a crash, not a timeout - just empty.
        return TestResult(passed=0, total=0, timed_out=False, crashed=False)

    harness = build_harness(code, entry_point, tests, atol)
    raw, retries = _run_with_one_retry(harness)

    if raw.timed_out:
        # Suite did not complete -> no evidence of correctness -> 0.
        return TestResult(
            passed=0, total=total, timed_out=True, crashed=False, infra_retries=retries
        )
    passed, crashed, case_timeouts = _parse_result(raw.stdout, total)
    return TestResult(
        passed=passed,
        total=total,
        timed_out=False,
        crashed=crashed,
        infra_retries=retries,
        case_timeouts=case_timeouts,
    )
