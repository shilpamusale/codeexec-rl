"""The per-test pass rule: EvalPlus 0.3.1's, ported (ADR-015).

A test passes exactly when EvalPlus's own evaluator would pass it: exact macth
by Python equality (so True == 1 and 1 == 1.0), then the MBPP special oracles,
then a float tolerance. When a problem's atol is 0 and the expected output is a
float, or a list or tuple of floats, atol is raised to 1e-6; a non-exact result
then passes if it has the expected type and length and every element is within
atol + 1e-7 * |expected| (numpy.allclose's rule, written out here).

This file is self-contained and stdlib-only. The harness embeds its source verbatim
into every sandboxed program, where neither numpy nor this package is installed, and
the host imports the same file, so there is one copy of the rule. Every name is
underscore-prefixed so it cannot collide with the model's code, which shares the
sandboxed program's namespace.

The special oracles apply to MBPP problems, the only ones graded so far; HumanEval's
(find_zero) is added when HumanEval+ is measured (ADR-015).

Ported from evalplus/eval/__init__.py (unsafe_execute) and
evalplus/eval/_special_oracle.py, EvalPlus 0.3.1, Apache License 2.0.
"""

import math as _math
from typing import Any as _Any

# Outputs that are not serializable: only "not None" is checked.
_MBPP_OUTPUT_NOT_NONE_TASKS = ("check_str", "text_match_three", "text_starta_endb")

# Outputs whose element order is not part of the answer: set comparison.
_MBPP_OUTPUT_SET_EQ_TASKS = (
    "similar_elements",
    "find_char_long",
    "common_in_nested_lists",
    "extract_singly",
    "larg_nnum",
    "intersection_array",
    "find_dissimilar",
    "Diff",
)
_FLOAT_ATOL = 1e-6  # EvalPlus's floor for float outputs when a problem's atol is 0
_RTOL = 1e-7  # numpy.allclose's default, which EvalPlus sets explicitly


def _surface_Area(base_edge: float, height: float) -> int:
    """EvalPlus's reference for Mbpp/581 (height is the pyramid's perpendicular height)."""
    slant_height = _math.sqrt((base_edge / 2) ** 2 + height**2)
    base_area = base_edge**2
    lateral_area = 4 * (base_edge * slant_height) / 2
    total_surface_area = base_area + lateral_area
    return round(total_surface_area)


def _digit_distance_nums(num1: int, num2: int) -> int:
    """EvalPlus's reference for Mbpp/558 (numbers zero-padded to the same length)."""
    str_num1, str_num2 = str(num1), str(num2)
    max_length = max(len(str_num1), len(str_num2))
    str_num1, str_num2 = str_num1.zfill(max_length), str_num2.zfill(max_length)
    total_difference = 0
    for digit1, digit2 in zip(str_num1, str_num2, strict=True):  # equal after zfill
        difference = abs(int(digit1) - int(digit2))
        total_difference += difference
    return total_difference


def _is_floats(x: object) -> bool:
    """A float, or a non-empty list or tuple of floats (EvalPlus's is_floats)."""
    if isinstance(x, float):
        return True
    if isinstance(x, (list, tuple)) and x:
        return all(isinstance(i, float) for i in x)
    return False


def _is_number(x: object) -> bool:
    return isinstance(x, (int, float))  # bool is an int, as in numpy


def _allclose(out: _Any, exp: _Any, atol: float) -> bool:
    """numpy.allclose(out, exp, rtol=1e-7, atol=atol) for numbers and nested sequences."""
    if isinstance(exp, (list, tuple)):
        if not isinstance(out, (list, tuple)) or len(out) != len(exp):
            return False
        return all(_allclose(o, e, atol) for o, e in zip(out, exp, strict=True))
    if not (_is_number(out) and _is_number(exp)):
        return False
    if _math.isinf(out) or _math.isinf(exp):
        return bool(out == exp)  # infinities are close only to themselves
    return bool(abs(out - exp) <= atol + _RTOL * abs(exp))  # nan is never close


def _passes(out: _Any, exp: _Any, atol: float, entry_point: str, args: tuple[_Any, ...]) -> bool:
    """Whether one test passes, as EvalPlus 0.3.1 decides it for an MBPP problem.

    Any exception raised here (an incomparable output) is a failed test; the caller treats it so,
    as EvalPlus does.
    """

    exact = out == exp
    if entry_point == "are_equivalent":  # Mbpp/164
        exact = exact or True
    elif entry_point == "sum_div":  # Mbpp/295
        exact = exact or out == 0
    elif entry_point == "surface_Area":  # Mbpp/581
        exact = exact or abs(out - _surface_Area(*args)) <= atol
    elif entry_point == "digit_distance_nums":  # Mbpp/558
        exact = exact or out == _digit_distance_nums(*args)
    elif entry_point in _MBPP_OUTPUT_SET_EQ_TASKS:
        exact = set(out) == set(exp)
    elif entry_point in _MBPP_OUTPUT_NOT_NONE_TASKS:
        exact = out == exp if isinstance(out, bool) else exp == (out is not None)

    if atol == 0 and _is_floats(exp):
        atol = _FLOAT_ATOL
    if not exact and atol != 0:
        if type(out) is not type(exp):
            return False
        if isinstance(out, (list, tuple)) and len(out) != len(exp):  # out has exp's type
            return False
        return _allclose(out, exp, atol)
    return bool(exact)
