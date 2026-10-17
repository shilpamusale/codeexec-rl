# Architecture Decision Records

Each record states the context, the decision, the consequences accepted, and — the part that makes it a decision rather than an assertion — **what would reverse it**.

| ADR | Decision | Status |
|---|---|---|
| [001](#adr-001) | Policy model: Qwen2.5-Coder-1.5B | Accepted |
| [002](#adr-002) | Task distribution: MBPP train, HumanEval+ held out | Accepted |
| [003](#adr-003) | GRPO as the default algorithm, PPO as a comparison arm | Accepted |
| [004](#adr-004) | EvalPlus as the held-out layer, not generated property tests | Accepted |
| [005](#adr-005) | Container isolation from day one | Accepted |
| [006](#adr-006) | Policy variant: Qwen2.5-Coder-1.5B base, measured fallback | Accepted |
| [007](#adr-007) | Strict reward comparison; drop order-unstable problems | Accepted |

---

## ADR-001

### Policy model: Qwen2.5-Coder-1.5B

**Context.** The artifact's value is what it demonstrates about execution-reward dynamics: whether a gap opens mid-training, how it varies across seeds, whether a detector catches it early. That evidence requires many runs, not one impressive run.

**Decision.** Qwen2.5-Coder-1.5B as the policy for every arm. Variant (base vs Instruct) decided separately in ADR-006.

**Consequences.**
- Many runs on a single A100, with seed replication affordable. Six runs for H1 rather than one.
- The model will not approach frontier coding performance. No result here transfers automatically to larger models, and the writeups state that.
- Code-specialized pretraining means the base model can actually solve some MBPP problems, which is a precondition for having optimization pressure at all.

**What would reverse it.** If 1.5B proves too weak to improve on the task distribution under GRPO — `visible_pass` flat against baseline — there is no optimization pressure and nothing to overfit with. The first remedy is a short SFT warm-start before GRPO, recorded as its own ADR rather than slipped in silently. Moving up a size class is the second, and it costs seed replication, which is the more valuable property.

**Status update.** Base pass@1 measured at 0.57 (57/100) on a frozen 100-problem MBPP+ sample (seed 0), Qwen2.5-Coder-1.5B base, greedy/temp-0, zero-shot-plus-stub prompt. Well above floor and below ceiling, so there is optimization headroom for H1 to move. The "too weak to improve" reversal condition is not triggered; no SFT warm-start or size-class change needed. See notebooks/smoke_test_base_model.ipynb.

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

**Status update — H0 pre-registration (recorded before the perturbed run).** The
H0 contamination probe is pre-registered here so the verdict is read off a fixed
rule rather than argued for after the number is seen.

- *Sample.* The frozen 100-problem MBPP+ sample (seed 0) behind the base-model
  smoke test. Paired: same problems, canonical prompt vs perturbed prompt, with
  everything else held constant (Qwen2.5-Coder-1.5B base, zero-shot-plus-stub,
  greedy/temp-0, same extraction + subprocess grader).
- *Instrument.* The Section D perturbation generator (semantic-preserving rename
  + docstring paraphrase, each kept only if its renamed canonical still passes the
  sandbox gate). Perturbed completions are graded against the perturbed problem's
  own tests.
- *Statistic.* drop = canonical_pass - perturbed_pass, reported with a McNemar
  test (paired) and a bootstrap CI on the drop.
- *Decision rule (fixed in advance).*
  - drop <= 5 pp -> clean. Surface form is not carrying the pass-rate; the
    contamination precondition holds; ADR-002 stands and training stays on MBPP+.
  - 5 pp < drop < 15 pp -> ambiguous. Escalate to the full gated set (~374) for a
    precise estimate before any verdict; the n=100 CI is too wide to decide.
  - drop >= 15 pp -> contaminated. ADR-002 reverses; the training distribution
    switches to the perturbed variants (the Section D generator becomes the
    fallback it was built to be), opened as its own ADR.
- *Power caveat.* At n=100 and a ~0.57 base rate, a sub-5pp result means "no drop
  large enough to detect at this n," not "proven zero." Acceptable because a drop
  that small yields the same action (train on MBPP+) either way.

The measured result and verdict will be appended once the run completes.

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

**What would reverse it.** If MBPP+ turns out to be barely harder than the canonical tests for this model — a small and static `G_baseline` — then the instrument has little dynamic range and a stronger held-out layer is needed. This is measurable in Phase 3, before any training, and the result is reported either way.

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

**Status update:** Container startup measured as the throughput bottleneck (~3.5 exec/sec/core, trivial workload). Warm pool deferred pending Phase 4 training-throughput data. gVisor evaluated and not adopted. See the Phase 1 writeup (analysis/sandbox.md).

---

## ADR-006

### Base vs Instruct variant: Qwen2.5-Coder-1.5B base, with a measured fallback

**Context.** ADR-001 fixes the model family and size but not the variant. Qwen2.5-Coder-1.5B ships in a base version and an instruction-tuned (`-Instruct`) version. The two differ in how reliably they follow a "write a function" prompt format, and in what a measured capability means. Most public RLVR-on-code efforts use `-Instruct` for format reliability and comparability; a study whose central claim is about generalization has a specific reason to prefer base.

**Decision.** The base model, not `-Instruct`, as the policy for every arm — contingent on a measured format-failure rate, with a fallback fixed in advance.

**Consequences.**
- Cleaner semantics for H0, which is the project's load-bearing precondition. H0 asks whether measured pass-rates reflect generalization rather than memorization. Instruction tuning adds a second learned layer on top of pretraining, so on an `-Instruct` model a pass-rate — or a perturbation-induced drop — could reflect either the base model's coding ability or what the tuning taught. Base isolates raw capability, keeping the one measurement everything depends on maximally clean.
- The cost is more reward lost to malformed output early in training, since base follows the function-emission format less reliably than `-Instruct`. Handled by a fixed prompt template and parsing tolerance in the reward function, not by defaulting to `-Instruct`.
- Comparability caveat: results are then not directly comparable to the `-Instruct`-based public efforts. Named in the writeup.

**What would reverse it.** The base model's malformed-output rate is measured in Phase 2/3, as part of the base-pass-rate check that runs before any training. If that rate is tolerable — unparseable completions a small enough fraction that the signal survives — base stands. If format failures are frequent enough to swamp the signal regardless of template and parsing tolerance, the first remedy is the SFT warm-start from ADR-001, which addresses format-following without full instruction-tuning; moving to `-Instruct` is the second, and the instruction-tuning confound it reintroduces into H0 is named explicitly. Which remedy is chosen is recorded here, with the measured failure rate that motivated it.

**Status update.** Malformed-output rate measured at 0.02 (2/100) on the same frozen 100-problem sample (seed 0), zero-shot-plus-stub prompt — 98% of completions parse into a function named `entry_point`. Far below the tolerance threshold, so **base stands**: no few-shot prompting, SFT warm-start, or move to -Instruct is needed, and H0's contamination semantics stay clean. The prompt is locked as zero-shot-plus-stub for the baseline and every training arm, so the prompt choice does not affect delta-G. See notebooks/smoke_test_base_model.ipynb.

---

## ADR-007

### Strict reward comparison, and dropping order-unstable problems

**Context.** The reward function decides, for each test, whether a solution's output matches the expected answer. That comparison *is* the reward's operational definition of "correct," which makes it the measuring instrument for the whole study. A comparison that is too loose scores a wrong answer as right — the exact Goodhart failure H1 is built to detect — so the instrument must not have the flaw it is meant to measure.

Expected outputs are pre-computed once at load time by running each dataset's canonical solution, then stored in the `Problem`. Model solutions are run later, in a separate sandbox process. Comparison therefore happens across a process boundary.

**Decision.** The reward comparison (`_compare`) is strict:
- `int` and `float` match within `atol` (a correct `4` vs `4.0` should not fail on type);
- `list` and `tuple` are distinct (a tuple is not a list);
- `bool` and `int` are distinct (`True` is not `1`), checked before the numeric branch since `bool` is a subclass of `int`;
- a length guard precedes element-wise recursion, so a shorter result cannot match a longer expected via `zip` truncation;
- `atol` applies only at float leaves; `inf == inf` is handled by exact-equality-before-tolerance (`abs(inf - inf)` is `nan`); `nan` never matches `nan`.

Validating the instrument against ground truth — running every canonical solution against its own stored tests, which must pass 100% by construction — surfaced three MBPP problems (`Mbpp/2`, `Mbpp/579`, `Mbpp/769`) whose canonical returns an unordered collection (`tuple(set(...))`). The element order of their stored `expected` is non-deterministic across Python processes (hash-seed dependent), so a correct model solution would be graded as failing on element order alone. These three are **excluded** rather than rescued by loosening the comparison.

**Consequences.**
- The reward is a strict instrument, consistent with the strict EvalPlus held-out suite (ADR-004) it is measured against.
- The exclusion is 3 of 377 MBPP+ problems (0.8%), measured by running all canonicals under two different hash seeds and diffing — not estimated. At that fraction the training distribution is unaffected.
- This is the third principled exclusion at the task layer: one for zero held-out tests (cannot contribute to the gap being measured) and three for order-instability, alongside non-finite float *inputs* which are handled in source generation rather than dropped. Each is logged at load time; the task-id list is a single named constant (`ORDER_UNSTABLE_TASK_IDS`) shared by the loader and its tests.
- The rejected alternative — an order-insensitive (multiset) comparison — would make `[1,2,3]` match `[3,2,1]` for *every* problem, including sorting problems where order is correctness. That trades a 0.8% false-negative for a systematic false-*positive* risk across the whole dataset, which is the loose-instrument failure the study is about. Not taken.

**What would reverse it.** If a *large* fraction of the training distribution turned out to be order-unstable, dropping would bias the distribution and the fix would shift to canonicalizing unordered outputs at load time (sorting the stored `expected` for genuinely-unordered return types, per problem) while keeping the comparison strict. At 0.8% that cost is not warranted. Loosening the comparison itself is not a reversal path — it is ruled out by the thesis.