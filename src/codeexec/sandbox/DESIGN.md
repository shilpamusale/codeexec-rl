# Sandbox — Design

*The execution contract, fixed before implementation. The sandbox runs untrusted,
model-generated code against a test set and returns a structured result. This note
defines what goes in, what comes out, and why — so the container, the async pool,
and the reward function that consumes results all build against a known interface.*

## Purpose

A policy model emits a solution; that solution must be executed to know whether it
is correct, and executing untrusted code safely is the precondition for using its
pass result as a reward. The sandbox is that execution boundary: it takes code and
a test set, runs the code against the tests inside an isolated container, and
returns a single structured result describing what happened.

## Contract

### Input

- **code** — the solution to execute (Python source).
- **tests** — the test set to run the code against, passed **separately** from the
  code. (Test-format details are deferred to the task layer in Phase 2.)
- **limits** — wall-clock timeout and memory cap, fixed as sandbox **configuration**
  rather than passed per execution.

### Output

A single structured result. Every execution maps to exactly one status, plus a
fixed set of payload fields:

| Field | Meaning |
|---|---|
| `status` | The outcome — exactly one of the six statuses below. |
| `tests_passed` | Count of tests that passed. |
| `tests_total` | Count of tests run. The reward function computes the pass fraction from these two. |
| `stdout` | Captured standard output. |
| `stderr` | Captured standard error — tracebacks and assertion messages live here. |
| `duration` | Wall-clock time for the execution. Feeds the throughput curve. |

## Status enum

| Status | Meaning |
|---|---|
| `passed` | The code ran and all tests passed. |
| `failed` | The code ran and tests executed, but one or more failed. |
| `timeout` | Killed at the wall-clock limit. Includes hangs and infinite loops. |
| `out_of_memory` | Killed for exceeding the memory cap. |
| `error` | The code raised an exception or would not run **on its own** — syntax error, unhandled exception, import failure. |
| `blocked` | Terminated because it attempted a disallowed action (network, filesystem escape) or hit a hard containment limit. |

## Key design decisions

**Code and tests are separate inputs, not one combined script.** The experiment runs
the *same solution* against *different test sets* — canonical (visible) tests for one
measurement, MBPP+ (held-out) tests for another. That is the V0/V1 mechanism at the
core of H1. Fusing code and tests would force regenerating the combined artifact per
test set and risk the code drifting between runs. Separation is what the visible/
held-out partition requires, and it also makes the fixture suite clean: each fixture
is just a different (code, tests) pair fed to the same runner.

**`blocked` is distinct from `error`.** `error` is the code failing on its own terms;
`blocked` is the sandbox stopping it. One is a property of the *solution*, the other is
a property of *containment working*. Folding them would erase the single most
security-relevant signal the sandbox produces — whether any solution tried to escape —
so they stay separate.

**Counts, not per-test detail.** The result records `tests_passed` / `tests_total`, not
which specific tests passed. Sparse reward needs only the status; shaped reward needs
the pass fraction, which the two counts provide. Per-test detail has no current
consumer, so it is not carried.

**Limits are fixed config, not per-call.** Every execution in the experiment uses
identical limits — there is no reason one solution gets a different timeout or memory
cap than another. A single configured limit keeps the interface to code + tests.

## Out of scope (for now)

Deferred deliberately, to be added only when a consumer needs them:

- **Per-test detail** — which specific tests passed, beyond the summary counts. Would be
  added if a later phase's reward or analysis needs per-case outcomes.
- **Per-call limits** — different timeout/memory per execution. Would be added if a task
  source ever required non-uniform limits.