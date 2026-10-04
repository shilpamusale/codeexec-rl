"""Async execution pool

Runs many sandbox executions concurrently, with a bound on how many containers
run at once. Wraps the synchronous 'run_code' in worker threads (each execution
blocks on its container, so the work is I/O-bound from the pool's perspective
and threads give real concurrency). Bounded parallelism is enforced by a semaphore.
Cancellation and backpressure are deferred (see FUTURE.md): current usage submits known,
finite batches, which the semaphore already bounds.
"""

import asyncio
import os
from collections.abc import Sequence

from codeexec.sandbox.runner import RawResult, run_code

DEFAULT_CONCURRENCY = os.cpu_count() or 4


async def _run_code(code: str, semaphore: asyncio.Semaphore) -> RawResult:
    """Run a single execution, holding a concurrency slot for its duration."""
    async with semaphore:
        return await asyncio.to_thread(run_code, code)


async def run_many(
    codes: Sequence[str],
    concurrency: int = DEFAULT_CONCURRENCY,
) -> list[RawResult]:
    """Run many code snippets concurrently, at most `concurrency` at a time.
    Results are returned in the same order as the `codes`, regardless of the order
    in which execution finish.
    """
    semaphore = asyncio.Semaphore(concurrency)
    tasks = [_run_code(code, semaphore) for code in codes]
    return await asyncio.gather(*tasks)
