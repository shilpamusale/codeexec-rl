"""Integration tests for the sandbox execution runner.
These run real containers via Docker, so they are marked 'integration' and
are slower than unit test cases. They prove the plumbing: code goes into a
container and raw output comes back correctly across success, multi-line input, and
failure.
"""

import pytest

from codeexec.sandbox import runner
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
    assert result.infra_error is False  # the program really ran
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

    assert result.infra_error is False  # the program really ran
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

    assert result.infra_error is False  # the program really ran
    assert result.returncode != 0
    assert "ALLOCATED HUGE" not in result.stdout


def test_process_limit_stops_fork_bomb() -> None:
    """Code cannot exhaust the process table: --pids-limit caps spawned processes."""
    code = "import os\nfor _ in range(500):\n    os.fork()\nprint('FORKED MANY')\n"
    result = run_code(code)

    assert result.infra_error is False  # the program really ran
    assert result.returncode != 0
    assert "FORKED MANY" not in result.stdout


def test_infinite_loop_times_out() -> None:
    """Code that never terminates is killed at the wall-clock limit."""
    result = run_code("while True:\n    pass\n")

    assert result.timed_out is True
    assert result.returncode != 0


def test_stdout_is_capped_and_writing_past_the_cap_fails() -> None:
    """ADR-005 status update (C2): output past the cap never reaches the host."""
    code = "import os\nos.write(1, b'x' * (3 * 2**20))\nprint('past the cap')"
    result = run_code(code)

    assert result.infra_error is False  # the program really ran
    # One write, cut short at the cap; the print after it raised instead.
    assert result.stdout.count("x") == len(result.stdout) == runner.OUTPUT_LIMIT_BYTES
    assert result.returncode != 0
    assert "File too large" in result.stderr


def test_stderr_is_capped_too() -> None:
    result = run_code("import os\nos.write(2, b'e' * (3 * 2**20))\n")

    assert result.infra_error is False
    assert result.returncode == 0
    assert result.stderr.count("e") == len(result.stderr) == runner.OUTPUT_LIMIT_BYTES


def test_the_program_cannot_lift_the_cap() -> None:
    code = (
        "import resource\n"
        "resource.setrlimit(resource.RLIMIT_FSIZE, (resource.RLIM_INFINITY,) * 2)\n"
        "print('LIFTED THE CAP')"
    )
    result = run_code(code)

    assert result.infra_error is False
    assert result.returncode != 0
    assert "LIFTED THE CAP" not in result.stdout


def test_bytes_that_are_not_utf8_cannot_raise() -> None:
    """ADR-005 status update (C2): undecodable bytes are replaced, so no output can
    stop a training step."""

    result = run_code("import os\nos.write(1, b'\\xff ok\\n')\nos.write(2, b'\\xfe')\n")

    assert result.infra_error is False
    assert result.returncode == 0
    assert result.stdout == "\ufffd ok\n"
    assert result.stderr == "\ufffd"


def test_normal_run_is_not_infra_and_stderr_is_clean() -> None:
    """The start marker reaches stderr from the real image, and is stripped before return."""
    result = run_code("print('hi')")

    assert result.infra_error is False
    assert result.stderr == ""


def test_program_exit_127_is_not_infra() -> None:
    """A program's own sys.exit(127) is a program failure, not an infrastructure one."""
    result = run_code("import sys\nsys.exit(127)")

    assert result.infra_error is False
    assert result.returncode == 127
    assert result.infra_error is False


def test_missing_image_is_infra(monkeypatch: pytest.MonkeyPatch) -> None:
    """A missing image fails at once (`--pull never`, exit 125): the program never ran."""
    monkeypatch.setattr(runner, "IMAGE", "codeexec-sandbox-does-not-exist")
    result = run_code("print('never runs')")

    assert result.returncode == 125
    assert result.infra_error is True


def test_bad_timeout_command_line_is_infra(monkeypatch: pytest.MonkeyPatch) -> None:
    """If `timeout` rejects its own command line, the program never ran.

    -1 parses as an unknown option, so `timeout` exits 125 before the marker.
    """
    monkeypatch.setattr(runner, "TIMEOUT_SECONDS", -1)
    result = run_code("print('never runs')")

    assert result.returncode == 125
    assert result.infra_error is True


def test_a_cap_that_cannot_be_set_is_infra(monkeypatch: pytest.MonkeyPatch) -> None:
    """If the shell cannot set the output cap, the program never runs: no marker,
    so an infrastructure failure, never scored (ADR-005 status update, C2)."""
    monkeypatch.setattr(runner, "OUTPUT_LIMIT_BYTES", -512)  # `ulimit -f -1` is refused
    result = run_code("print('never runs')")

    assert result.infra_error is True
    assert "never runs" not in result.stdout
