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


def test_normal_code_runs_under_isolation() -> None:
    """Normal code still runs under --read-only (bytecode writes are disabled)."""
    result = run_code("print('still works')")

    assert result.returncode == 0
    assert "still works" in result.stdout


def test_network_access_is_blocked() -> None:
    """Code cannot reach the network: --network none leaves no route out."""
    code = (
        "import urllib.request\n"
        "urllib.request.urlopen('http://example.com', timeout=5)\n"
        "print('REACHED NETWORK')\n"
    )
    result = run_code(code)

    assert result.returncode != 0
    assert "REACHED NETWORK" not in result.stdout


def test_write_outside_scratch_is_blocked() -> None:
    """The filesystem is read-only outside the scratch mount."""
    code = (
        "with open('/tmp/escape.txt', 'w') as f:\n"
        "    f.write('hi')\n"
        "print('WROTE OUTSIDE SCRATCH')\n"
    )
    result = run_code(code)

    assert result.returncode != 0
    assert "WROTE OUTSIDE SCRATCH" not in result.stdout


def test_write_inside_scratch_succeeds() -> None:
    """The scratch mount is writable by the sandbox user.

    Regression test: the tmpfs mount must be owned by the container's non-root
    user (uid=1000). Without that, a freshly-mounted tmpfs is root-owned and the
    unprivileged user cannot write to it, breaking all legitimate scratch use.
    """
    code = (
        "with open('/sandbox/ok.txt', 'w') as f:\n"
        "    f.write('hi')\n"
        "print('WROTE INSIDE SCRATCH')\n"
    )
    result = run_code(code)

    assert result.returncode == 0
    assert "WROTE INSIDE SCRATCH" in result.stdout


def test_memory_limit_stops_allocation() -> None:
    """Code cannot exhaust host memory: the container memory limit stops it.

    The allocation may fail as a Python MemoryError or be killed abruptly by the
    kernel OOM-killer, so this asserts the behavior (the allocation did not
    succeed) rather than a specific error message.
    """
    code = "x = [0] * (10 ** 10)\nprint('ALLOCATED HUGE')\n"
    result = run_code(code)

    assert result.returncode != 0
    assert "ALLOCATED HUGE" not in result.stdout


def test_process_limit_stops_fork_bomb() -> None:
    """Code cannot exhaust the process table: --pids-limit caps spawned processes."""
    code = "import os\nfor _ in range(500):\n    os.fork()\nprint('FORKED MANY')\n"
    result = run_code(code)

    assert result.returncode != 0
    assert "FORKED MANY" not in result.stdout


def test_infinite_loop_times_out() -> None:
    """Code that never terminates is killed at the wall-clock limit."""
    result = run_code("while True:\n    pass\n")

    assert result.timed_out is True
    assert result.returncode != 0
