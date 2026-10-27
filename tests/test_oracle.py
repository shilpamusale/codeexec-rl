"""The per-test pass rule against EvalPlus's own evaluator (ADR-015).

Each case is decided twice: by our port (oracle._passes) and by EvalPlus 0.3.1's
evaluator itself, which runs a function returning the given output in its own
sandboxed process. The two must agree on every case. The cases cover each
branch of EvalPlus's rule, including the ones where it differs from ADR-007's
strict comparison.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pytest
from evalplus.eval import PASS, untrusted_check

from codeexec.rewards import oracle
from codeexec.rewards.harness import _to_source

INF, NAN = math.inf, math.nan

# (output, expected, atol, entry_point, args)
CASES: list[tuple[Any, Any, float, str, tuple[Any, ...]]] = [
    # exact match is Python equality
    (4, 4, 0.0, "f", ()),
    (4, 5, 0.0, "f", ()),
    (4, 4.0, 0.0, "f", ()),
    (True, 1, 0.0, "f", ()),  # ADR-007 failed this; EvalPlus passes it
    (False, 0, 0.0, "f", ()),
    ([1, 2], (1, 2), 0.0, "f", ()),
    ("ab", "ab", 0.0, "f", ()),
    ("ab", "ba", 0.0, "f", ()),
    (None, None, 0.0, "f", ()),
    ({"a": 1}, {"a": 1}, 0.0, "f", ()),
    # float outputs with atol 0: EvalPlus raises atol to 1e-6, rtol 1e-7
    (5.52754851098587, 5.527548510985869, 0.0, "f", ()),  # Mbpp/234: l*l*l vs l**3
    (0.1 + 0.2, 0.3, 0.0, "f", ()),
    (1.0, 1.0000005, 0.0, "f", ()),  # within 1e-6
    (1.0, 1.000002, 0.0, "f", ()),  # beyond it
    (4, 4.0000001, 0.0, "f", ()),  # close, but an int is not a float
    ([1.0, 2.0], [1.0, 2.0000001], 0.0, "f", ()),
    ([1.0, 2.0], (1.0, 2.0000001), 0.0, "f", ()),  # list is not tuple
    ([1.0], [1.0, 2.0], 0.0, "f", ()),
    ([[1.0]], [[1.0000001]], 0.0, "f", ()),  # nested: not "floats", so exact only
    ({"a": 1.0000001}, {"a": 1.0}, 0.0, "f", ()),
    # problems with atol above 0
    (1.0, 1.5, 0.6, "f", ()),
    (1.0, 1.5, 0.4, "f", ()),
    ([[1.0, 2.0]], [[1.0000001, 2.0]], 1e-5, "f", ()),
    (INF, INF, 1e-4, "f", ()),
    (INF, -INF, 1e-4, "f", ()),
    (NAN, NAN, 1.0, "f", ()),
    # MBPP special oracles
    (["b", "a"], ["a", "b"], 0.0, "find_char_long", ("x",)),  # set equality
    (["a"], ["a", "b"], 0.0, "find_char_long", ("x",)),
    ([[1]], [[1]], 0.0, "find_char_long", ("x",)),  # unhashable: set() raises
    ("matched", True, 0.0, "text_match_three", ("abbb",)),  # "not None"
    (None, True, 0.0, "text_match_three", ("a",)),
    (False, True, 0.0, "text_match_three", ("a",)),
    (7, 999, 0.0, "digit_distance_nums", (123, 256)),  # reference recomputation
    (6, 999, 0.0, "digit_distance_nums", (123, 256)),
    (35, 33, 0.0, "surface_Area", (3, 4)),
    (0, 5, 0.0, "sum_div", (7,)),
    (False, True, 0.0, "are_equivalent", (36, 57)),
]


@pytest.fixture(autouse=True)
def _no_evalplus_memory_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    """EvalPlus caps its child process's address space at 4 GB (RLIMIT_AS). A
    macOS process already reserves more than that, so the cap kills the child
    before it evaluates anything. -1 is EvalPlus's own switch to turn the cap
    off; the cap is sandboxing, not part of the pass rule, so no verdict changes."""
    monkeypatch.setenv("EVALPLUS_MAX_MEMORY_BYTES", "-1")


def _evalplus_verdict(
    out: Any, exp: Any, atol: float, entry_point: str, args: tuple[Any, ...]
) -> bool:
    """EvalPlus's own decision: its evaluator runs a function that returns `out`."""
    code = f"def {entry_point}(*args):\n    return {_to_source(out)}\n"
    status, _ = untrusted_check("mbpp", code, [list(args)], entry_point, [exp], atol, [0.01])
    return bool(status == PASS)


def _ours(out: Any, exp: Any, atol: float, entry_point: str, args: tuple[Any, ...]) -> bool:
    """Our port, with an exception counting as a failed test, as in the harness."""
    try:
        return oracle._passes(out, exp, atol, entry_point, args)
    except BaseException:
        return False


@pytest.mark.parametrize(("out", "exp", "atol", "entry_point", "args"), CASES)
def test_agrees_with_evalplus(
    out: Any, exp: Any, atol: float, entry_point: str, args: tuple[Any, ...]
) -> None:
    assert _ours(out, exp, atol, entry_point, args) == _evalplus_verdict(
        out, exp, atol, entry_point, args
    )


def test_the_rounding_case_that_started_adr_015_passes() -> None:
    assert oracle._passes(5.52754851098587, 5.527548510985869, 0.0, "volume_cube", (1.768,))


def test_the_module_runs_on_its_own() -> None:
    """The harness embeds this file's source into the sandbox, where neither this
    package nor numpy exists: executed alone, it must still provide _passes."""
    namespace: dict[str, Any] = {}
    exec(Path(oracle.__file__).read_text(), namespace)
    assert namespace["_passes"](0.1 + 0.2, 0.3, 0.0, "f", ())
