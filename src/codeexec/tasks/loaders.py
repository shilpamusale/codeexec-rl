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

# If a problem keep sfewer than this fraction of its visible tests after validation,
# drop the whole problem - its referece solution is too unreliable.

MIN_VISIBLE_TESTS_KEPT = 1


def _load_function(solution_code: str, entry_point: str) -> object:
    """Execute `solution_code` and return the function named `entry_point`.

    The code is run in fresh namespace; the named function object is pulled out
    of that namespace so it can be called to compute expected outputs.
    """

    namespace: dict[str, object] = {}
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
    entry_point = str(record["entry_point"])
    prompt = str(record["prompt"])
    atol = float(record["atol"])  # type: ignore[arg-type]

    # Build the complete, runnable canonical solution. HUmanEval stores only the
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
    """Normalize every record in a raw EvalPlusdataset into problems"""
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
