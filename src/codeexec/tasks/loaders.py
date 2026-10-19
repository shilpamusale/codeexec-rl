"""Loaders that turn EvalPlus records into normalized Problem objects.

Each loader reads one EvalPlus dataset, absorbs its record format,
and pre-computes expected test outputs by running the canonical solution
on each input. Problems or test cases whose canonical solution can not
produce an output are dropped and logged, so a Problem only ever contains tests
it can be graded against.

The canonical solutions run here are the dataset's own trusted reference solutions,
executed once at the load time (not model output, and not in the training hot path),
so they are run directly rather than through the sandbox.
"""

import logging

from evalplus.data import get_human_eval_plus, get_mbpp_plus

from codeexec.tasks.problem import Problem, TestCase

logger = logging.getLogger(__name__)

# If a problem keeps fewer than this fraction of its visible tests after validation,
# drop the whole problem - its reference solution is too unreliable.

MIN_VISIBLE_TESTS_KEPT = 1
MIN_HELDOUT_TESTS_KEPT = 1

# MBPP problems whose canonical solution returns an unordered collection
# (e.g. tuple(set(...))), so the element order of the pre-computed `expected`
# is non-deterministic across Python processes. We compute `expected` in the
# loader process but run model code in a separate sandbox process, so a correct
# solution can be graded as failing on element order alone. Strict comparison is
# the right call (ADR-007), so we exclude these rather than loosen the grader.
# 3 of 377 MBPP+ problems (0.8%). See DECISIONS.md ADR-007.
ORDER_UNSTABLE_TASK_IDS = frozenset({"Mbpp/2", "Mbpp/579", "Mbpp/769"})

# MBPP problems whose canonical solution cannot produce a gradeable result at
# all, so they can never anchor a valid measurement (the canonical fails its own
# gate, perturbed or not). Distinct from ORDER_UNSTABLE (ADR-007, a comparison
# issue) -- here the reference solution itself is ungradeable. Surfaced by the
# Phase 3 H0 instrument-readiness check. See DECISIONS.md ADR-009.
#   Mbpp/599: canonical too slow on its own stress input (sum(range(10**8)))
#             -> exceeds the per-case + sandbox timeout (ADR-008).
#   Mbpp/630: canonical returns a ~2 MB nested list -> harness cannot handle it.
#   Mbpp/737: canonical returns a re.Match object -> no valid source representation.
#   Mbpp/794: canonical returns a re.Match object -> no valid source representation.
UNGRADEABLE_CANONICAL_TASK_IDS = frozenset({"Mbpp/599", "Mbpp/630", "Mbpp/737", "Mbpp/794"})

# MBPP problems whose canonical is correct but at least one held-out input is
# itself invalid (no correct function could satisfy it), so a correct model
# solution would be graded as failing on bad test data. We exclude rather than
# hand-edit the EvalPlus suite (keeping ADR-004's citable, unmodified benchmark).
# See DECISIONS.md ADR-009.
#   Mbpp/734: canonical passes 104/107 of its own tests; >=1 held-out input is
#             invalid (e.g. expects a nonzero result for an empty-list input).
CORRUPT_HELDOUT_TASK_IDS = frozenset({"Mbpp/734"})


def _load_function(solution_code: str, entry_point: str) -> object:
    """Execute `solution_code` and return the function named `entry_point`.

    The code is run in a fresh namespace; the named function object is pulled out
    of that namespace so it can be called to compute expected outputs. Warnings
    are suppressed: these are trusted dataset canonicals, some of which use
    regexes with unescaped backslashes (SyntaxWarning) that are not ours to fix.
    """
    import warnings

    namespace: dict[str, object] = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        exec(solution_code, namespace)
    fn = namespace[entry_point]
    return fn


def _build_tests(
    fn: object,
    inputs: list[list[object]],
    task_id: str,
    which: str,
) -> tuple[TestCase, ...]:
    """Run `fn` on each input to compute its expected output, building TestCases.
    Inputs whose execution raises are dropped and logged - a reference solution
    that errors on its own test input can not define an expected output.
    """
    cases: list[TestCase] = []
    for raw_args in inputs:
        args = tuple(raw_args)
        try:
            expected = fn(*args)  # type: ignore[operator]
        except Exception as exc:
            logger.warning(
                "dropping %s %s test: canonical raised %s", task_id, which, type(exc).__name__
            )
            continue
        cases.append(TestCase(args=args, expected=expected))
    return tuple(cases)


def _make_problem(record: dict[str, object], source: str) -> Problem | None:
    """Normalize one EvalPlus record into a Problem, or None if it must be dropped."""
    task_id = str(record["task_id"])

    # Exclude problems with order-unstable canonical outputs: their pre-computed
    # `expected` would differ from a correct solution's output on element order
    # alone, across the loader/sandbox process boundary. See ADR-007.
    if task_id in ORDER_UNSTABLE_TASK_IDS:
        logger.info("dropping %s: order-unstable canonical output (ADR-007)", task_id)
        return None
    if task_id in UNGRADEABLE_CANONICAL_TASK_IDS:
        logger.info("dropping %s: canonical not gradeable (ADR-009)", task_id)
        return None
    if task_id in CORRUPT_HELDOUT_TASK_IDS:
        logger.info("dropping %s: corrupt held-out input(s) (ADR-009)", task_id)
        return None

    entry_point = str(record["entry_point"])
    prompt = str(record["prompt"])
    atol = float(record["atol"])  # type: ignore[arg-type]

    # Build the complete, runnable canonical solution. HumanEval stores only the
    # function body, so it is the prompt (signature + docstring) plus that body;
    # MBPP stored a complete function already.

    if source == "humaneval":
        canonical = prompt + str(record["canonical_solution"])
    else:
        canonical = str(record["canonical_solution"])
    try:
        fn = _load_function(canonical, entry_point)
    except Exception as exc:  #
        logger.warning("dropping %s: canonical solution would not load (%s)", task_id, exc)
        return None

    base_inputs: list[list[object]] = record["base_input"]  # type: ignore[assignment]
    plus_inputs: list[list[object]] = record["plus_input"]  # type: ignore[assignment]

    visible = _build_tests(fn, base_inputs, task_id, "visible")
    heldout = _build_tests(fn, plus_inputs, task_id, "heldout")

    if len(visible) < MIN_VISIBLE_TESTS_KEPT:
        logger.warning("dropping %s: no usable visible tests after validation", task_id)
        return None
    if len(heldout) < MIN_HELDOUT_TESTS_KEPT:
        logger.warning("dropping %s: no held-out tests (cannot measure the gap)", task_id)
        return None

    return Problem(
        task_id=task_id,
        source=source,
        prompt=prompt,
        entry_point=entry_point,
        canonical_solution=canonical,
        visible_tests=visible,
        heldout_tests=heldout,
        atol=atol,
    )


def _load(raw: dict[str, dict[str, object]], source: str) -> dict[str, Problem]:
    """Normalize every record in a raw EvalPlus dataset into problems"""
    problems: dict[str, Problem] = {}
    for record in raw.values():
        problem = _make_problem(record, source)
        if problem is not None:
            problems[problem.task_id] = problem
    logger.info("loaded %d/%d problems for %s", len(problems), len(raw), source)
    return problems


def load_mbpp() -> dict[str, Problem]:
    """Load MBPP+ as normalized Problems, keyed by task_id."""
    return _load(get_mbpp_plus(), "mbpp")


def load_humaneval() -> dict[str, Problem]:
    """Load HumanEval+ as normalized Problems, keyed by task_id."""
    return _load(get_human_eval_plus(), "humaneval")
