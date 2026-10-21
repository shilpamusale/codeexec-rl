"""Sandbox throughput under a training-like load (Phase 4, Section A4).

Phase 1 measured throughput with a trivial program (`pass`) on a 2-core
Codespace, so its number was pure container overhead. This measures what
training will actually ask of the sandbox, on the training box: real reward
programs -- the saved frozen-100 completions inside the grading harness -- at
concurrency levels up to and past the box's vCPU count.

Workloads:
    trivial  `pass`                                 container overhead only
    visible  completion + visible tests             one V1 reward call
    full     completion + visible + held-out tests  one V0 reward call / plus eval

Every execution is checked for having run to completion (for harness programs:
the result sentinel was printed). The completed count must stay the same at
every concurrency level. If it drops as concurrency rises, load is changing
grading outcomes (cases pushed past their timeouts), which would corrupt
rewards in training -- a finding in its own right, not noise.

    python scripts/throughput_under_load.py --workload visible
    python scripts/throughput_under_load.py --workload trivial --levels 1,2 --executions 16

Writes analysis/throughput_box_<workload>.csv unless --out is given.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from pathlib import Path

from codeexec.eval.extract import extract_function
from codeexec.rewards.harness import RESULT_SENTINEL, build_harness
from codeexec.sandbox.benchmark import _clean_containers  # Phase 1's cleanup between levels
from codeexec.sandbox.pool import run_many
from codeexec.sandbox.runner import RawResult
from codeexec.tasks.datasets import training_problems

WORKLOADS = ("trivial", "visible", "full")


def build_programs(workload: str, completions_path: Path) -> list[str]:
    """The distinct programs for a workload; each level cycles through them."""
    if workload == "trivial":
        return ["pass"]
    completions: dict[str, str] = json.loads(completions_path.read_text())["completions"]
    pool = training_problems()
    programs: list[str] = []
    for task_id, completion in completions.items():
        problem = pool[task_id]
        code = extract_function(completion, problem)
        if code is None:
            continue  # malformed output never reaches the sandbox in training either
        tests = problem.visible_tests
        if workload == "full":
            tests = (*problem.visible_tests, *problem.heldout_tests)
        pairs = [(tc.args, tc.expected) for tc in tests]
        programs.append(build_harness(code, problem.entry_point, pairs, problem.atol))
    return programs


def ran_to_completion(raw: RawResult, workload: str) -> bool:
    """True if the execution finished rather than timing out or crashing."""
    if raw.timed_out:
        return False
    if workload == "trivial":
        return raw.returncode == 0
    return RESULT_SENTINEL in raw.stdout


def measure(
    programs: list[str], workload: str, concurrency: int, executions: int
) -> tuple[float, int]:
    """Run `executions` programs at `concurrency`; return (seconds, completed count)."""
    codes = [programs[i % len(programs)] for i in range(executions)]
    _clean_containers()  # no debris from the previous level
    start = time.monotonic()
    results = asyncio.run(run_many(codes, concurrency=concurrency))
    seconds = time.monotonic() - start
    completed = sum(ran_to_completion(raw, workload) for raw in results)
    return seconds, completed


def main() -> None:
    parser = argparse.ArgumentParser(description="Sandbox throughput under a training-like load.")
    parser.add_argument("--workload", choices=WORKLOADS, default="visible")
    parser.add_argument("--levels", default="1,2,4,8,16,24,32,48", help="comma-separated")
    parser.add_argument("--executions", type=int, default=192, help="executions per level")
    parser.add_argument(
        "--completions", type=Path, default=Path("analysis/completions_frozen100.json")
    )
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    levels = [int(level) for level in args.levels.split(",")]
    out: Path = args.out or Path(f"analysis/throughput_box_{args.workload}.csv")
    vcpus = os.cpu_count() or 0
    programs = build_programs(args.workload, args.completions)
    print(
        f"workload={args.workload}  programs={len(programs)}  "
        f"executions/level={args.executions}  vCPUs={vcpus}  levels={levels}"
    )

    # Warm-up: the first container start pays one-off costs (image layers into
    # the page cache). Keep that out of the first level's number.
    asyncio.run(run_many(programs[:1], concurrency=1))

    rows = ["workload,vcpus,concurrency,executions,completed,seconds,throughput_per_sec"]
    completed_counts: list[int] = []
    for concurrency in levels:
        seconds, completed = measure(programs, args.workload, concurrency, args.executions)
        throughput = args.executions / seconds
        completed_counts.append(completed)
        print(
            f"  concurrency={concurrency:>3}: {throughput:6.1f} exec/sec  "
            f"({completed}/{args.executions} completed, {seconds:.1f}s)"
        )
        rows.append(
            f"{args.workload},{vcpus},{concurrency},{args.executions},"
            f"{completed},{seconds:.2f},{throughput:.2f}"
        )
    _clean_containers()

    stable = len(set(completed_counts)) == 1
    verdict = "yes" if stable else "NO -- load changed outcomes"
    print(f"\ncompleted count stable across levels: {verdict}")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(rows) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
    