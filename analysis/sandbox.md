# Phase 1 — The execution sandbox

*What was built, what it guarantees, what it costs, and what it does not do.*

The sandbox is the execution boundary of CodeExec-RL: it runs untrusted,
model-generated code against a test set and returns a structured result. Its
output — did the code pass? — becomes the reward signal, so the sandbox is the
precondition for everything downstream. This writeup records the isolation model,
the evidence that it holds, the measured throughput, and the two optimization
decisions deferred or declined on the basis of that evidence.

The execution contract itself (inputs, the result shape, the status enum) is
documented alongside the code in `src/codeexec/sandbox/DESIGN.md`. This writeup is
about the *guarantees and measurements*, not the interface.

---

## The isolation model

The sandbox runs each execution in a container, hardened on six dimensions. Each
closes a different way untrusted code could cause harm. No single layer is trusted
alone — the design is defense in depth, each layer assuming the others might fail.

- **Non-root.** Code runs as an unprivileged user (UID 1000), not root. A
  container escape, if one occurred, would land as a powerless user rather than
  with host administrator rights. This caps the blast radius of any breach.
- **No network.** The container has no network route (`--network none`). Code
  cannot exfiltrate data, fetch a payload, or reach other machines. MBPP solutions
  never need the network, so this is pure attack-surface removal — and it also
  prevents a solution "passing" by fetching an answer.
- **Read-only filesystem, with a scratch exception.** The filesystem is frozen
  (`--read-only`) except one small, disposable, in-memory scratch mount
  (`--tmpfs /sandbox`, 64 MB, owned by the sandbox user). Code cannot persist
  itself, tamper with system files, or fill the disk; it has exactly one writable
  place, sized and owned so it cannot be abused.
- **Memory limit.** RAM is capped (`--memory 256m`). A memory bomb that tries to
  exhaust the host is killed instead. Generous for function-level solutions, far
  below host-crash territory.
- **Process limit.** The process count is capped (`--pids-limit 128`). A fork
  bomb — a distinct exhaustion vector from memory — cannot exhaust the process
  table.
- **Wall-clock timeout.** Execution is bounded in time. The limit is enforced
  *inside* the container by the `timeout` utility (`timeout 10 python`), so runaway
  code is killed at the source and the container exits cleanly on its own, with a
  subprocess-level backstop for the rare case where the container runtime itself
  wedges. Infinite loops and input-blocked hangs are both caught this way.

These six sit on top of the default container runtime's own hardening: `runc`
applies seccomp syscall filtering and AppArmor/SELinux mandatory access controls
by default, so the containers are not bare even before the explicit guarantees
above.

## Deterministic teardown

Every execution is removed when it finishes — however it finishes: success,
failure, timeout, crash, or an unexpected error in the runner itself. The
container exits on its own in the normal case (`--rm` plus the in-container
`timeout`); a `finally` block force-removes by name on every path as a backstop.
Because the removal runs unconditionally, no container leaks, which matters across
the many thousands of executions a training run will issue — a small per-execution
leak would accumulate into an eventual host failure.

## The fixture suite: isolation proven, not asserted

Each guarantee is backed by an adversarial test that attempts the forbidden thing
and confirms it is stopped. These are the difference between claiming isolation and
demonstrating it:

- normal code runs and returns correct output (the isolation does not break
  execution);
- a network connection attempt fails;
- a write outside the scratch mount fails;
- a write *inside* the scratch mount succeeds (a regression guard for the tmpfs
  ownership fix: a freshly-mounted tmpfs is root-owned, so the mount must be given
  the sandbox user's UID or legitimate writes break);
- a memory bomb is killed;
- a fork bomb is killed;
- an infinite loop is killed at the wall-clock limit.

The tests assert the *behavior* (the forbidden action did not succeed, the
execution ended) rather than a specific error message, because the same limit can
surface differently across runs — a memory limit may appear as a Python
`MemoryError` or an abrupt kernel kill. They run against real containers and are
marked `integration`, so fast unit tests can run separately during development.

## Throughput

Throughput was measured across concurrency levels with a trivial workload
(`pass`), so the number reflects the sandbox's own container-lifecycle overhead —
startup, isolation setup, teardown — rather than the cost of any particular
solution. Containers are cleaned between measurements so debris from one level
cannot slow the next. (Raw data: `analysis/throughput.csv`.)

On a two-core machine, throughput rises from ~2.7 exec/sec at concurrency 1 to
~3.4 exec/sec at concurrency 2, then plateaus flat through concurrency 16
(~3.7 exec/sec). The shape is a single step up from the second core, then a flat
line: the bottleneck is container lifecycle, and it saturates at the core count,
so added concurrency past the number of cores buys nothing.

Measurement caveats, stated plainly: this is a shared two-core development
environment, so the cores are not exclusively available and the numbers carry
run-to-run noise; the small differences along the plateau (3.6 to 3.7) are within
that noise, not a trend. The curve characterizes *this* machine. Its value is the
shape and the identified bottleneck, not the absolute figure.

### Warm container pool: deferred pending training-throughput data

ADR-005 flagged container startup latency as a cost to measure before deciding
whether to add a warm container pool (pre-started containers, recycled across
executions, to avoid paying startup per run).

The benchmark settles the first half: because the trivial-workload ceiling is
essentially lifecycle overhead, startup *is* the dominant cost — the condition
that would motivate a warm pool. But a costly bottleneck justifies the
optimization only if the throughput it caps is actually insufficient, and that
depends on how many executions per second training will demand — a Phase 4 number
not yet known. Building a warm pool now would optimize a bottleneck without
confirming it constrains anything.

Decision: defer the warm pool. The bottleneck is measured and the remedy is
identified; adoption waits until training throughput shows whether ~3.5 exec/sec
per core is a real limit. If it is, a warm pool with container recycling is the
first optimization to try.

### gVisor: evaluated, not adopted

Ordinary containers share the host kernel, so a kernel exploit is a
container-escape path. gVisor closes that by interposing a memory-safe user-space
kernel between the workload and the host, at a per-syscall performance cost.
ADR-005 called for evaluating it. The conclusion is not to adopt it, for three
reasons:

**Threat-model fit.** gVisor defends against crafted kernel exploits — the work of
a determined attacker writing exploit code. This workload is a 1.5B model
generating solutions to isolated, function-level programming problems. The
realistic failure mode is buggy code that loops forever or over-allocates, which
the existing wall-clock, memory, and process limits already contain — not a kernel
0-day. The specific threat gVisor addresses is essentially absent here.

**Cost against measured throughput.** The sandbox is already bottlenecked on
container lifecycle. gVisor's syscall overhead would worsen exactly the constraint
already identified as limiting, for no threat-model benefit.

**Existing hardening and availability.** The default runtime (`runc`) already
applies seccomp syscall filtering plus AppArmor/SELinux mandatory access controls,
so the containers are not unprotected — gVisor would be additional defense on an
already-hardened base. It is also unavailable in the development environment: only
`runc` is configured, and gVisor requires host-level runtime configuration a
managed Codespace does not permit.

This would be revisited if the threat model changed — for example, serving
untrusted user-submitted code in production rather than model-generated solutions
to fixed benchmarks.

---

## What this proves, and what it does not

What Phase 1 establishes: untrusted code can be executed safely and repeatably,
isolated on six dimensions, bounded in time, torn down deterministically, run
concurrently under a bounded pool, and the isolation is demonstrated by adversarial
tests rather than asserted. The reward signal the rest of the project depends on
has a trustworthy foundation.

What it does not establish, and does not claim:

- **Not hardened against a determined human attacker.** The threat model is
  model-generated solutions to fixed benchmarks, not adversarial exploit code. The
  gVisor decision above is explicit about this boundary.
- **Not characterized beyond this machine.** The throughput curve is a two-core
  development environment. Absolute figures will differ elsewhere; the bottleneck
  and shape are the transferable findings.
- **Not optimized for throughput.** The warm pool is deliberately deferred until
  training data shows whether the measured ceiling is a real constraint.
- **Function-level only.** The sandbox runs single, self-contained programs. Nothing
  here concerns multi-file or repository-level execution — out of scope by design.
