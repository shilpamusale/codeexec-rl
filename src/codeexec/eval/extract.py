"""Prompt construction and completion extraction for the base-model smoke test.

Pure logic (no GPU, no model) so it is fully unit-testable on CPU. The prompt is
zero-shot-plus-stub: the MBPP prompt followed by the function signature, so the
base model continues inside the function body. Extraction reaseembles the stub +
completion, cuts the base model's trailing continuation, and flags completions that
do not parse into a function named `entry_point` as malformed.
"""

from __future__ import annotations

import ast

from codeexec.tasks.problem import Problem


def _signature_from_canonical(canonical: str, entry_point: str) -> str:
    """Return the 'def entry_point(args):' line taken from the canonical solution.
    The canonical defines the real signature, so parse it rather than guess."""

    tree = ast.parse(canonical)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == entry_point:
            args = ast.unparse(node.args)
            return f"def {entry_point}({args}):"
    # fallback: no-arg signature (should not happen for valid MBPP canonicals)
    return f"def {entry_point}()"


def build_prompt(problem: Problem) -> str:
    """Zero-shot-plus-stub prompt: the problem text, then the function signature,
    so the base model continues inside the function body."""

    stub = _signature_from_canonical(problem.canonical_solution, problem.entry_point)
    return f"{problem.prompt}\n{stub}\n"


def _signature_line(problem: Problem) -> str:
    return _signature_from_canonical(problem.canonical_solution, problem.entry_point)


def extract_function(completion: str, problem: Problem) -> str | None:
    """Reassemble the stub + model completion into a candidate function and cut the
    base model's trailing continutaion. Returns the function source, or None
    if the result is malformed (does not parse into a function named entry_point)."""

    stub = _signature_line(problem)
    body = completion

    # Reassemble:  the completion is the continuation after the stub line.
    full = stub + "\n" + body

    # Cut the first top-level line that starts a NEW construct after the
    # function body: a new def, a new docstring/quote, a bare assert, or any
    # other column-0 line that isn't a continuation of the function.

    lines = full.splitlines()
    kept: list[str] = [lines[0]]  # the 'def ...:' line
    for line in lines[1:]:
        stripped = line.strip()
        # a column-0, non-empty line that is not indented ends the function.
        if line and not line[0].isspace() and stripped:
            break
        kept.append(line)
    candidate = "\n".join(kept)

    # Validate: must parse and define a function named entry_point.
    try:
        tree = ast.parse(candidate)
    except SyntaxError:
        return None
    has_fn = any(
        isinstance(n, ast.FunctionDef) and n.name == problem.entry_point for n in ast.walk(tree)
    )
    if not has_fn:
        return None
    return candidate
