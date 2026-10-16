"""Docstring-paraphrase perturbation: rephrase the prompt's prose while
preserving the spec exactly. Lead-in rephrasing plus a tight, hand-vetted
synonym map; assert (call-site) lines are left untouched, and if no transform
applies the prompt is returned unchanged (a meaning-preserving no-op).

Paraphrase is conservative by construction because the validation gate cannot
see it (it executes tests, not prose) -- see ADR / DESIGN.
"""

from __future__ import annotations

import random
import re
from dataclasses import replace

from codeexec.tasks.problem import Problem

# Lead-in replacements, keyed by the connector the original uses ("to" vs
# "that"), so an imperative lead-in stays imperative and a declarative one stays
# declarative. Each option is equivalent to "Write a function {connector} ...".
_LEADIN_TO: tuple[str, ...] = (
    "Implement a function to",
    "Define a function to",
    "Create a function to",
    "Write a routine to",
)
_LEADIN_THAT: tuple[str, ...] = (
    "Implement a function that",
    "Define a function that",
    "Create a function that",
    "Write a routine that",
)

# The known original lead-ins, longest first so "write a python function to"
# matches before "write a function to". Each maps to the connector family.
_KNOWN_LEADINS: tuple[tuple[str, str], ...] = (
    ("write a python function to", "to"),
    ("write a python function that", "that"),
    ("write a function to", "to"),
    ("write a function that", "that"),
)

# Tight synonym map: each entry is context-independent in MBPP prose.
# Applied mid-sentence, case-insensitive, word-boundary. Prose only.
_SYNONYMS: tuple[tuple[str, str], ...] = (
    ("calculate", "compute"),
    ("check whether", "determine whether"),
    ("check if", "determine whether"),
)


def _rephrase_leadin(prose_line: str, rng: random.Random) -> tuple[str, bool]:
    """If `prose_line` starts with a known lead-in, swap it for an equivalent.
    Returns (new_line, changed)."""
    low = prose_line.lower()
    for leadin, connector in _KNOWN_LEADINS:
        if low.startswith(leadin):
            pool = _LEADIN_TO if connector == "to" else _LEADIN_THAT
            replacement = pool[rng.randrange(len(pool))]
            # keep the remainder of the line verbatim (that is the spec)
            remainder = prose_line[len(leadin) :]
            return replacement + remainder, True
    return prose_line, False


def _apply_synonyms(line: str, rng: random.Random) -> tuple[str, bool]:
    """Apply the tight synonym map to one prose line. Returns (new_line, changed)."""
    changed = False
    for old, new in _SYNONYMS:
        pattern = re.compile(r"\b" + re.escape(old) + r"\b", re.IGNORECASE)
        if pattern.search(line):
            line = pattern.sub(new, line)
            changed = True
    return line, changed


def _paraphrase_prompt(prompt: str, rng: random.Random) -> tuple[str, bool]:
    """Paraphrase prose lines of the prompt; leave assert lines untouched.
    Returns (new_prompt, changed)."""
    out_lines: list[str] = []
    changed_any = False
    leadin_done = False
    for line in prompt.splitlines():
        stripped = line.strip()
        if "assert" in line or not stripped or stripped in ('"""', "'''"):
            out_lines.append(line)
            continue
        new_line = line
        # rephrase the lead-in only once, on the first prose line
        if not leadin_done:
            new_line, led = _rephrase_leadin(new_line, rng)
            leadin_done = True
            changed_any = changed_any or led
        new_line, syn = _apply_synonyms(new_line, rng)
        changed_any = changed_any or syn
        out_lines.append(new_line)
    return "\n".join(out_lines), changed_any


def paraphrase_docstring(problem: Problem, rng: random.Random) -> Problem:
    """Return a copy of `problem` with its docstring prose paraphrased, or the
    problem unchanged if no transform applied (a meaning-preserving no-op)."""
    new_prompt, changed = _paraphrase_prompt(problem.prompt, rng)
    if not changed:
        return problem
    return replace(problem, prompt=new_prompt)
