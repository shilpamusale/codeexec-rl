"""Integration tests for the sandbox execution runner.
These run real containers via Docker, so they are marked 'integration' and
are slower than unit test cases. They prove the plumbing: code goes into a
container and raw output comes back correctly across success, multi-line input, and
failure.
"""

import pytest

from codeexec.sandbox.runner import RawResult, run_code

pytestmark = pytest.mark.integration


def test_runs_simple_code_and_captures_stdout() -> None:
    """A trivial program runs and its stdout comes back,
    with a zero exit code.
    """
    result = run_code("print('hello from the sandbox')")
    assert isinstance(result, RawResult)
    assert result.returncode == 0
    assert "hello from the sandbox" in result.stdout


def test_runs_multiline_code() -> None:
    """A multi-line program with nested quotes arrives intact via stdin.

    This is the case that would mangle if code were passed as a `python -c`
    command-line argument. It passing proves stdin delivery preserves the program.
    """
    code = 'def greet(name):\n    return f"hi, {name}"\n\nprint(greet("world"))\n'
    result = run_code(code)

    assert result.returncode == 0
    assert "hi, world" in result.stdout


def test_captures_error_and_nonzero_exit() -> None:
    """Code that raises returns a nonzero exit code and the traceback in stderr."""
    result = run_code("raise ValueError('boom')")

    assert result.returncode != 0
    assert "ValueError" in result.stderr
