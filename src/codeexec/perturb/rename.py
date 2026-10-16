"""Function-rename perturnbation: rename the entry-point function consistently
across the canonical solution (via AST) and the prompt's assert call sites
(prose is left untouched - see ADR/DESIGN), leaving tests unchanged.

"""

from __future__ import annotations

import ast
import keyword
import random
import re
from dataclasses import replace

from codeexec.tasks.problem import Problem

# Neutral function names: plausible identifiers that carry no semantic hint
# about the task, so a reame disrupts a memorizer without helping a solver.
NEUTRAL_NAMES: tuple[str, ...] = (
    "solve",
    "compute_result",
    "run_task",
    "process_input",
    "handler",
    "evaluate",
    "apply_logic",
    "produce_output",
    "do_work",
    "transform_input",
    "calculate",
    "execute",
)


def _pick_name(old_name: str, canonical: str, rng: random.Random) -> str:
    """Pick a newtral replacement name that dowsn't colllide with the old name,
    a Python builtin/keyword, or an identifier already used in the canonical."""

    used = {node.id for node in ast.walk(ast.parse(canonical)) if isinstance(node, ast.Name)}
    # Deterministic order: start at a seeded offset, step through the pool
    start = rng.randrange(len(NEUTRAL_NAMES))
    for i in range(len(NEUTRAL_NAMES)):
        candidate = NEUTRAL_NAMES[(start + i) % len(NEUTRAL_NAMES)]
        if candidate == old_name:
            continue
        if keyword.iskeyword(candidate) or candidate in dir(__builtins__):
            continue
        if candidate in used:
            continue
        return candidate

    # pool exhausted (vanishingly unlikely): fall back to a mangled unique name
    return f"{old_name}_renamed"


class _RenameTransformer(ast.NodeTransformer):
    """Rename every occurrence of `old` to `new` in function defs and names."""

    def __init__(self, old: str, new: str):
        self.old = old
        self.new = new

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        if node.name == self.old:
            node.name = self.new
        self.generic_visit(node)
        return node

    def visit_Name(self, node: ast.Name) -> ast.AST:
        if node.id == self.old:
            node.id = self.new
        return node


def _rename_in_canonical(canonical: str, old: str, new: str) -> str:
    """AST-rename `old` to `new` in the canonical solution and return new source."""
    tree = ast.parse(canonical)
    _RenameTransformer(old, new).visit(tree)
    ast.fix_missing_locations(tree)
    return ast.unparse(tree)


def _rename_in_prompt(prompt: str, old: str, new: str) -> str:
    """Rename `old` to `new` only in assert (call-site) lines of the prompt.
    Prose lines are left untouched, so an entry-point name that is also an
    English word (check, sum, count, ...) does not corrupt the docstring.
    """
    pattern = re.compile(r"\b" + re.escape(old) + r"\b")
    out_lines = []

    for line in prompt.splitlines():
        if "assert" in line:
            out_lines.append(pattern.sub(new, line))
        else:
            out_lines.append(line)
    return "\n".join(out_lines)


def rename_function(problem: Problem, rng: random.Random) -> Problem:
    """Return a copy of `problem` with its entry-point function renamed consistently
    across canonical_solution and the prompt's assert lines."""

    old = problem.entry_point
    new = _pick_name(old, problem.canonical_solution, rng)
    return replace(
        problem,
        entry_point=new,
        canonical_solution=_rename_in_canonical(problem.canonical_solution, old, new),
        prompt=_rename_in_prompt(problem.prompt, old, new),
    )
