# Architecture Decision Records

Each record states the context, the decision, the consequences accepted, and — the part that makes it a decision rather than an assertion — **what would reverse it**.

| ADR | Decision | Status |
|---|---|---|
| [001](#adr-001) | Policy model: Qwen2.5-Coder-1.5B | Accepted |
| [002](#adr-002) | Task distribution: MBPP train, HumanEval+ held out | Accepted |
| [003](#adr-003) | GRPO as the default algorithm, PPO as a comparison arm | Accepted |
| [004](#adr-004) | EvalPlus as the held-out layer, not generated property tests | Accepted |
| [005](#adr-005) | Container isolation from day one | Accepted |

---

## ADR-001

### Policy model: Qwen2.5-Coder-1.5B

**Context.** The artifact's value is what it demonstrates about execution-reward dynamics: whether a gap opens mid-training, how it varies across seeds, whether a detector catches it early. That evidence requires many runs, not one impressive run.

**Decision.** Qwen2.5-Coder-1.5B as the policy for every arm.

**Consequences.**
- Many runs on a single A100, with seed replication affordable. Six runs for H1 rather than one.
- The model will not approach frontier coding performance. No result here transfers automatically to larger models, and the writeups state that.
- Code-specialized pretraining means the base model can actually solve some MBPP problems, which is a precondition for having optimization pressure at all.

**What would reverse it.** If 1.5B proves too weak to improve on the task distribution under GRPO — `visible_pass` flat against baseline — there is no optimization pressure and nothing to overfit with. The first remedy is a short SFT warm-start before GRPO, recorded as its own ADR rather than slipped in silently. Moving up a size class is the second, and it costs seed replication, which is the more valuable property.

---

## ADR-002

### Task distribution: MBPP for training, HumanEval+ held out entirely

**Context.** The study needs function-level Python tasks with executable tests, at enough volume for RL, from a public and standard source so results are checkable.

**Decision.** MBPP as the training distribution. HumanEval+ held out entirely as an out-of-distribution check, never trained on.

**Consequences.**
- Both are standard and public, so the setup is reproducible and comparable to published work.
- Both are also heavily contaminated in models of this generation. That is a real threat to interpretation, which is why H0 exists and is measured before training rather than discussed afterward.
- MBPP's canonical tests are sparse — often around three per problem. Sparse visible tests make overfitting *easier* to induce, which may inflate the measured gap. Named as a confound in the writeup, not buried.
- Function-level scope excludes everything about navigating a real codebase. Deliberate: it isolates the reward question.

**What would reverse it.** If the H0 contamination probe shows severe memorization, the training distribution switches to the perturbed variants, or to a less contaminated task source. This decision is explicitly contingent on H0, and H0 runs first for exactly that reason.

---

## ADR-003

### GRPO as the default algorithm, PPO as a single comparison arm

**Context.** The reward here is verifiable — code runs against tests and the outcome is deterministic. PPO's learned critic exists to reduce variance by estimating expected return, which is most valuable when the reward is noisy or dense in an unclear way.

**Decision.** GRPO as the default. PPO retained as one optional comparison arm in Phase 6.

**Consequences.**
- No value head to train, tune, or destabilize. Fewer moving parts in a project whose interesting content is the reward, not the optimizer.
- Group-relative advantage is a natural fit for sampling several completions per problem and comparing them, which is what a verifiable reward makes cheap.
- Less machinery also means fewer failure modes in the phase most likely to consume time — getting a first arm to train stably.
- The cost: PPO is the more widely documented choice, and dropping it entirely would leave the comparison unmade. Hence retaining one arm.

**What would reverse it.** If GRPO proves unstable on this task at this scale in a way PPO does not, the default flips and the reasoning is recorded here. If the group-sampling cost turns out to dominate wall-clock — several completions per problem multiplies generation, and generation is expected to dominate — that is a throughput argument for revisiting, measured in Phase 3 rather than guessed at now.

---

## ADR-004

### EvalPlus as the held-out layer, not auto-generated property tests

**Context.** H1 is measured through the gap between visible and held-out pass-rates. The held-out suite is therefore the measuring instrument. If it is weak, the experiment cannot detect the effect regardless of whether the effect is real.

The original design generated property-based tests with Hypothesis for each problem. Writing meaningful properties across arbitrary MBPP problems — automatically, without per-problem hand-authoring — is a hard open problem, and getting it wrong would silently blunt the instrument.

**Decision.** EvalPlus augmented suites (MBPP+, HumanEval+) as the held-out layer. Hypothesis-generated property tests demoted to a supplementary canary in Phase 6, if at all.

**Consequences.**
- The held-out suite is built and validated by others, constructed specifically to catch solutions that pass the sparse canonical tests. That is precisely the property this study needs.
- It is citable, so the instrument is not a thing reviewers have to take on faith.
- The single largest technical risk in the original design is removed from the critical path.
- The tradeoff: the partition is now fixed by someone else's construction rather than tunable. The visible/held-out split becomes a documented property of the benchmark rather than a knob, and its limitations are inherited.

**What would reverse it.** If the EvalPlus augmented suite turns out to be barely harder than the canonical tests for this model — a small and static `G_baseline` — then the instrument has little dynamic range and a stronger held-out layer is needed. This is measurable in Phase 3, before any training, and the result is reported either way.

---

## ADR-005

### Container isolation from day one

**Context.** The sandbox executes untrusted, model-generated code. The obvious cheap implementation is a subprocess with `resource` limits. It is genuinely easier and it would work for most cases.

**Decision.** Container-level isolation from the first commit of the sandbox: non-root execution, no network, read-only filesystem except a scratch mount, hard wall-clock and memory limits, deterministic teardown. gVisor evaluated as a hardening path, with the outcome documented either way.

**Consequences.**
- Meaningfully stronger isolation than `rlimits`, which constrain resource usage but do not contain a process that is trying to reach the filesystem or the network.
- Deterministic teardown means no orphaned processes accumulating across thousands of executions — a practical reliability property, not only a security one.
- Costs startup latency per execution, which directly affects training throughput. The Phase 1 throughput benchmark exists to measure that cost rather than assume it is acceptable.
- More upfront work in the phase before anything visibly "works."

**What would reverse it.** If container startup latency dominates the training loop — measured, not assumed — the mitigation is a warm container pool with recycling rather than a retreat to bare subprocesses. Dropping to subprocess isolation would only be justified if the measurement showed containers to be impractical *and* the threat model were narrowed explicitly, and both parts of that would be recorded here.
