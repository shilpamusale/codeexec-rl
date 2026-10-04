"""Throughput benchmark for the sandbox execution pool.

Measure executions/second across a range of concurrency levels,
using a trivial workload so the number reflects the sandbox's own overhead (container startup,
isolation setup, teardown) rather than the cost of any particular solution.
Real solution add their own compute time on top of this.

Containers are cleaned between concurrency levels so debris from one measurement
cannot slow the next. Run as a script:
    python -m codeexec.sandbox.benchmak

Writes a CSV of (concurrency, throughput)
"""

import asyncio
import subprocess
import time
from pathlib import Path

from codeexec.sandbox.pool import run_many

# Trivial workload: ~ 0ms of actual work, so throughput reflects sandbox overhead
TRIVIAL_CODE = "pass"

# Execution per concurrency level. More = steadier average, but slower to run.
EXECUTIONS_PER_LEVEL = 32

# Concurrency levels to sweep.
CONCURRENCY_LEVELS = [1, 2, 4, 8, 16]

# Path to committed curve data.
OUTPUT_PATH = Path("analysis/throughput.csv")


def _clean_containers() -> None:
    """Remove any leftover sandbox containers so they don't skew the next lebel."""
    result = subprocess.run(
        ["docker", "ps", "-aq", "--filter", "name=codeexec"],
        capture_output=True,
        text=True,
        check=False,
    )

    ids = result.stdout.split()
    if ids:
        subprocess.run(["docker", "rm", "-f", *ids], capture_output=True, check=False)


def measure_level(concurrency: int, n: int) -> float:
    """Run `n` trivial executions at the given concurrency; return executions/sec."""
    codes = [TRIVIAL_CODE] * n
    start = time.monotonic()
    asyncio.run(run_many(codes, concurrency=concurrency))
    elapsed = time.monotonic() - start
    return n / elapsed


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    rows = ["concurrency, throughput_per_sec"]

    print(f"Benchmarking: {EXECUTIONS_PER_LEVEL} executions per level, levels={CONCURRENCY_LEVELS}")

    for concurrency in CONCURRENCY_LEVELS:
        _clean_containers()
        throughput = measure_level(concurrency, EXECUTIONS_PER_LEVEL)
        print(f" concurrency = {concurrency:>2}: {throughput:5.1f} exec/sec")
        rows.append(f"{concurrency}, {throughput: .2f}")

    _clean_containers()
    OUTPUT_PATH.write_text("\n".join(rows) + "\n")
    print(f"\n Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
