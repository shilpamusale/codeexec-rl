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
| [007](#adr-007) | Strict reward comparison; drop order-unstable problems | Partly superseded by [015](#adr-015) |
| [008](#adr-008) | Per-case timeout in the test harness | Accepted |
| [009](#adr-009) | Exclude ungradeable / corrupt-held-out MBPP problems | Accepted |
| [010](#adr-010) | Training box: Lambda on-demand, 1x A100 40GB | Accepted |
| [011](#adr-011) | GRPO via TRL GRPOTrainer + vLLM colocate, pinned | Accepted |
| [012](#adr-012) | Environment: thin stateless plain-Python env, no Gymnasium | Accepted |
| [013](#adr-013) | Held-out split: V0 rewards on A, B is the measurement | Accepted |
| [014](#adr-014) | H1 reward is r_sparse; outcome-to-reward mapping | Accepted |
| [015](#adr-015) | Test comparison matches EvalPlus's own grader | Accepted |

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

**Status update.** Base pass@1 measured at 0.57 (57/100) on a frozen 100-problem MBPP+ sample (seed 0), Qwen2.5-Coder-1.5B base, greedy/temp-0, zero-shot-plus-stub prompt. Well above floor and below ceiling, so there is optimization headroom for H1 to move. The "too weak to improve" reversal condition is not triggered; no SFT warm-start or size-class change needed. See notebooks/smoke_test_base_model.ipynb. (This estimate predates the ADR-009 exclusions; the H0 control arm is re-measured on the 369-problem pool — see the ADR-002 status update. This pass@1 is plus-graded — visible AND held-out tests — not visible-only; the visible-only rate is 0.65. See ADR-004 status / analysis/g_baseline.md.)

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

**Status update — H0 pre-registration (recorded before the perturbed run).** The H0 contamination probe is pre-registered so the verdict is read off a fixed rule rather than argued for after the number is seen.

- *Sample.* A frozen 100-problem MBPP+ sample (seed 0) drawn from the post-exclusion pool of 369 problems (ADR-009 removed 5 from the 374 the earlier smoke test used, so the sample universe changed; the exclusions were decided on canonical-gate failures with no model-outcome data, which is what keeps this refinement honest). Paired: same problems, canonical prompt vs perturbed prompt, with everything else held constant (Qwen2.5-Coder-1.5B base, zero-shot-plus-stub, greedy/temp-0, same extraction + subprocess grader). The canonical arm is re-measured on this 369-pool sample; the earlier 0.57 stands as the pre-exclusion smoke estimate and is superseded for H0.
- *Instrument.* The Section D perturbation generator (semantic-preserving rename + docstring paraphrase, each kept only if its renamed canonical still passes the sandbox gate). Perturbed completions are graded against the perturbed problem's own tests.
- *Statistic.* drop = canonical_pass - perturbed_pass, reported with a McNemar test (paired) and a bootstrap CI on the drop.
- *Decision rule (fixed in advance).*
  - drop <= 5 pp -> clean. Surface form is not carrying the pass-rate; the contamination precondition holds; ADR-002 stands and training stays on MBPP+.
  - 5 pp < drop < 15 pp -> ambiguous. Escalate to the full gated set (~369) for a precise estimate before any verdict; the n=100 CI is too wide to decide.
  - drop >= 15 pp -> contaminated. ADR-002 reverses; the training distribution switches to the perturbed variants (the Section D generator becomes the fallback it was built to be), opened as its own ADR.
- *Power caveat.* At n=100 and a ~0.57 base rate, a sub-5pp result means "no drop large enough to detect at this n," not "proven zero." Acceptable because a drop that small yields the same action (train on MBPP+) either way.

**Result (appended after the run).** Canonical pass@1 = 0.550 (55/100), perturbed pass@1 = 0.520 (52/100), drop = +3.0 pp. McNemar exact test on 11 discordant pairs (7 canon-only, 4 pert-only): p = 0.549 — not distinguishable from zero. Bootstrap 95% CI on the drop: [-3.0, +10.0] pp (straddles zero). Perturbed malformed rate 0.010, no format-failure confound. The near-symmetric discordant split is the signature of noise, not one-directional memorization. **Verdict: CLEAN** — drop is below the 5 pp pre-registered threshold and non-significant, so ADR-002 is not reversed and training proceeds on MBPP+. No escalation to the full 369 is warranted. Power caveat: MDE at n=100 is ~20 pp, so a mild (<~20 pp) effect cannot be excluded; this does not change the action. See analysis/h0.md. (Both arms are plus-graded — visible AND held-out tests — so 0.550 is the plus rate, not visible-only; visible-only on the same sample is 0.65, see ADR-004 status. The paired comparison is unaffected because both arms used the same grading.)

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

**Status update — PPO arm availability (Phase 4).** TRL removed `PPOTrainer` in v1.13.0 (Sep 2026), and GRPO runs on TRL 1.16.0 (ADR-011), so the optional Phase 6 PPO comparison arm cannot use the same library at the pinned version. If that arm is ever run, the options are a different PPO implementation or an older TRL pinned for that arm alone. Either way the comparison would differ in implementation as well as algorithm, and that confound is stated with the result. Phase 4 is unaffected.

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

**Status update.** G_baseline measured on the frozen 100 (seed 0, 369 pool), Qwen2.5-Coder-1.5B base, greedy: visible pass@1 = 0.650, plus pass@1 (base AND held-out) = 0.550, so **G_baseline = +10.0 pp**, bootstrap 95% CI [+5.0, +16.0] (excludes zero). The reversal condition — MBPP+ barely harder than canonical, yielding a small static gap with no dynamic range — is **not** triggered: the gap is meaningfully nonzero, so the held-out suite can resolve overfitting and ΔG has ~10 pp of headroom. Instrument validated; proceed to Phase 4. See analysis/g_baseline.md.

**Status update — baseline of record for delta-G (recorded before the Phase 4 re-measurement).** The +10.0 pp above was measured on Colab (T4, fp16, HF `generate`, stdin subprocess grader). Training grades through the Docker sandbox and generates on the training box, so G_baseline is re-measured there under the exact evaluation path every training checkpoint will use: same dtype, generation engine, `max_new_tokens=256` and Docker grader. That box measurement is the reference for delta-G = G_final - G_baseline, so the subtraction stays on one instrument. Grader parity is checked separately first: the same saved completions are graded by both the subprocess grader and the Docker sandbox, and must agree on all 100 problems. The Colab +10.0 pp stays as the Phase 3 record, and if the two baselines differ, both are reported with the cause.

**Status update — grader parity (Section A, training box).** The saved frozen-100 completions (analysis/completions_frozen100.json, regenerated on Colab with the Phase 3 setup and confirmed to reproduce the Phase 3 per-problem record exactly) were graded on the training box by the Phase 3 subprocess grader (host Python 3.11.17, 30 s timeout) and by the Docker sandbox (`python:3.11-slim@sha256:cce12316475d7d56d605f76e1d811d0dc4e7227f2d4afd29951521264f504ff8`; 10 s, 256 MB, no network, read-only filesystem). **Parity holds:** both graders give visible 65/100 and plus 55/100, with 0 disagreements and 0 differences from the committed G_baseline record on all 100 problems; the sandbox's tighter limits changed no outcome on this set. The Docker grader therefore measures the same G_baseline (+10.0 pp) as the Colab grader. The one remaining difference in the evaluation path is the generation engine, which is re-measured once the Section B training library is chosen. See analysis/grader_parity.json and scripts/grader_parity.py.

**Status update — box measurement plan (pre-registered before it runs, Section B, B7).**
G_baseline is re-measured on the training box through vLLM, TRL's generation engine (ADR-011): greedy decoding (temperature 0), at most 256 new tokens, stopping only at end-of-sequence, in bf16, from the pinned model revision `df3ce67c0e24480f20468b6ef2894622d69eb73b` (`configs/grpo.toml`; `main` since 2024-11-18, so the same files every earlier baseline loaded).
Completions are extracted as in Phase 3 and graded by the Docker sandbox on the visible tests, on visible plus held-out split B (the headline), and on visible plus the full held-out suite (the citable MBPP+ number, kept for the baseline and V1 per ADR-013).

**The ΔG reference is this measurement over all 369 training problems**, not the frozen 100: generation and grading on the box take minutes, and n = 369 roughly halves the confidence interval's width, which ΔG needs to stand out from seed variance.
The CI is computed exactly as in `analysis/g_baseline.md` (paired per-problem differences, numpy `default_rng(0)`, 20,000 resamples, 2.5/97.5 percentiles).
The frozen-100 subset is reported alongside, for continuity with the Phase 3 record and as the engine check: its per-problem results are compared with `analysis/g_baseline_split_b.json`, and any differences are reported with their cause, not resolved away.
If the 369-problem CI includes zero, training does not start until the instrument is revisited under this ADR's reversal condition.
Every training checkpoint is measured by the same command, so ΔG is a subtraction on one instrument. pass@10 and the HumanEval+ baseline are deferred to Phase 5's evaluator.

**Status update — the box measurement, and the ΔG reference (Section B, B7; ADR-015).**
- *As measured, under ADR-007's comparison (2026-10-26 UTC, commit b0623e1).* vLLM 0.31.0, bf16, on the A100. Over 369 problems: visible 249, plus 203 on split B and on the full suite, G = +12.5 pp, 95% CI [+9.2, +16.0]; 2 completions malformed, 14 stopped at 256 tokens. The frozen-100 subset: visible 65, plus 56, G = +9.0 pp, [+4.0, +15.0]. The engine check found one disagreement with the Phase 3 record, Mbpp/234, which led to ADR-015 (analysis/g_baseline_box.json).
- *The ΔG reference: the same completions regraded under ADR-015, without regenerating, on the platform of record (2026-10-28 UTC, commit 1f2b8fb).* Over 368 problems, with Mbpp/255 (now excluded) set aside: visible 251, plus 207 on split B, **G_baseline = +12.0 pp, 95% CI [+8.7, +15.5]**; on the full suite, plus 206, G = +12.2 pp, [+9.0, +15.8] (analysis/g_baseline_box_adr015.json). Five results changed, all from fail to pass, each by a difference ADR-015 names: set equality (Mbpp/111, Mbpp/232), a reference oracle (Mbpp/558) and Python equality (Mbpp/167, Mbpp/294). Mbpp/255 had failed everything. Mbpp/160's full suite ran out of time in this grading (ADR-015 status update), so the full suite counts one fewer than split B.
- *The engine check, again.* Against the Phase 3 record it still reports Mbpp/234. Against the Phase 3 completions regraded under ADR-015 (tests/test_gap.py) it reports none: vLLM and HF `generate` give the same verdict on all 100 problems. The frozen-100 subset is +9.0 pp, [+4.0, +15.0], under both comparisons.
- *The stop rule.* The CI excludes zero, so training may start.

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

**Status update — training-box throughput (Phase 4, Section A).** Re-measured on the training box (ADR-010; 30 vCPUs) with real reward programs at concurrency 1-48: plateaus of 38.8 exec/s (trivial), 34.9 (visible, one V1 reward call) and 13.3 (full, one V0 reward call), all reached by concurrency ~24, below the vCPU count. Container start and teardown are 79% of a visible reward call. The ceiling is the serialized container lifecycle, not cores, so a larger instance would not raise it and a warm pool would. Correction to the status line above: Phase 1 measured ~3.7 exec/s in total on a 2-core machine (~1.8 per core), not 3.5 per core. Load does not change grading outcomes: the completed count is identical at every concurrency level in every workload. **Decision: the warm pool stays deferred, now on measured supply.** At the plateau, grading 64 visible reward calls takes ~1.8 s (128: ~3.7 s); full-suite calls cost ~2.7x more, plus up to 10 s whenever one program times out. Whether that matters depends on generation and update time per step, which Section C measures. Rule, fixed now: adopt the warm pool if grading exceeds 25% of step wall time in the first stable training run; if the timeout tail rather than the mean dominates, the timeout budgets (runner 10 s, ADR-008's 2 s per case) are the lever instead. See the training-box addendum in analysis/sandbox.md and analysis/throughput_box_*.csv.

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

**Status update.** Malformed-output rate measured at 0.02 (2/100) on the same frozen 100-problem sample (seed 0), zero-shot-plus-stub prompt — 98% of completions parse into a function named `entry_point`. Far below the tolerance threshold, so **base stands**: no few-shot prompting, SFT warm-start, or move to -Instruct is needed, and H0's contamination semantics stay clean. The prompt is locked as zero-shot-plus-stub for the baseline and every training arm, so the prompt choice does not affect delta-G. See notebooks/smoke_test_base_model.ipynb. (This pass@1 is plus-graded — visible AND held-out tests — not visible-only; the visible-only rate is 0.65. See ADR-004 status / analysis/g_baseline.md.)

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

**Status update — superseded in part by ADR-015.** The comparison semantics above are replaced by EvalPlus's own grader: float tolerance, Python equality for exact match (so `True == 1`), and EvalPlus's special oracles. The premise that the EvalPlus held-out suite is strict on floats was wrong; ADR-015 records the evidence. The three order-unstable exclusions stand.

---

## ADR-008

### Per-case timeout in the test harness

**Context.** The harness runs all of a problem's test cases in one sandbox process under a single outer wall-clock. That makes one pathological case — a stress input on which a correct-but-unoptimized solution is merely slow — abort the entire suite before it prints its result line, so every case in that problem scores as a failure (0/N). The Phase 3 H0 baseline surfaced this on `Mbpp/599`, whose own canonical times out on a large stress input and so scored 0/91 despite being correct on the other 90 cases.

This is a reward-instrument defect, not a data problem. During training it would systematically punish correct solutions that are slow on a single stress input — zeroing their whole reward rather than docking one test — which is exactly the kind of mismeasurement the project exists to avoid, and it would distort H1/H2.

**Decision.** Each test case runs under its own per-case wall-clock (`PER_CASE_TIMEOUT_SECONDS = 2`), enforced inside the generated harness with a `SIGALRM` alarm armed before each call and cancelled the instant it returns. A case that exceeds the budget raises, is counted as a single failure, and the suite continues and still reports. The outer sandbox timeout is kept as a hard backstop for what `SIGALRM` cannot interrupt (C-level calls, large allocations). A timed-out case counts as a plain failure (correct RL semantics: no answer in time = wrong); `TestResult`'s shape is unchanged, so nothing downstream moves.

**Consequences.**
- One slow case fails alone; the rest of the suite grades normally. Correct-but-slow solutions are scored per-case, not zeroed wholesale.
- The timeout logic lives inside the embedded harness source as a helper, not as line-by-line assembly, so it is written and tested once rather than reconstructed per build.
- `SIGALRM` is POSIX-only and main-thread-only — fine for the Linux sandbox, and noted as a portability assumption.
- Validated by a 60-problem sample gate (unchanged: no grading regression) and a harness unit test proving a single slow case fails in isolation while its suite-mates pass.

**What would reverse it.** If per-case `SIGALRM` proved insufficient — e.g. the dominant failure mode were C-level calls it cannot interrupt — the fallback is per-case subprocess isolation, accepting its cost in the training hot path. If 2s turned out to wrongly fail legitimately-heavy correct solutions, the budget is the single knob to revisit, measured rather than guessed.

---

## ADR-009

### Exclude MBPP problems whose canonical fails its own gate

**Context.** The ground-truth invariant behind the reward instrument (ADR-007) is that every problem's canonical solution passes 100% of its own stored tests. The Phase 3 H0 instrument-readiness check ran this gate over the H0 sample and surfaced five MBPP problems that violate it — and, critically, violate it identically with or without perturbation, so they can never anchor a valid canonical-vs-perturbed measurement. Investigation (per-problem, through the real sandbox) found two distinct causes, which this ADR keeps distinct rather than lumping under one label.

**Decision.** Two load-time exclusions, each a single named constant shared by the loader and its tests, logged on drop:

- `UNGRADEABLE_CANONICAL_TASK_IDS` — the canonical cannot produce a gradeable result at all:
  - `Mbpp/599`: canonical is too slow on its own stress input (`sum(range(10**8))`), exceeding the per-case and sandbox timeouts (ADR-008). The instrument is now correct; the reference solution genuinely cannot pass its own stress test.
  - `Mbpp/630`: canonical returns a ~2 MB nested list; the harness cannot represent or handle a value that size.
  - `Mbpp/737`, `Mbpp/794`: canonical returns a `re.Match` object, which has no valid source representation and is not comparable by value.
- `CORRUPT_HELDOUT_TASK_IDS` — the canonical is correct but at least one held-out input is itself invalid:
  - `Mbpp/734`: canonical passes 104/107 of its own tests; >=1 held-out input is nonsensical for the spec (e.g. expects a nonzero result from an empty-list input), so a correct model solution would be graded as failing on bad data.

**Consequences.**
- MBPP+ drops from 374 to 369 (5 of 374, 1.3%). Combined with the three order-unstable exclusions (ADR-007) and the one zero-held-out drop, the task layer now excludes nine problems total, each logged at load time with a documented, cause-specific reason.
- The two causes are kept in separate constants because they are different defects: one is an ungradeable reference solution, the other is corrupt test data around a valid solution. A single "gate fail" label would misdescribe `Mbpp/734`.
- The alternative — rescuing problems by changing the instrument — is rejected for the same reason as ADR-007: special-casing `re.Match` comparison, or hand-editing EvalPlus's held-out inputs for `Mbpp/734`, would modify the measuring apparatus to fit the data, surrendering ADR-004's citable unmodified benchmark and ADR-007's strict grader. These are the exact loose-instrument failures the study is about.
- H0 and all downstream measurement run on the 369 problems whose ground truth is sound.

**What would reverse it.** If a *large* fraction of the distribution turned out to be ungradeable, wholesale dropping would bias the distribution and the fix would shift to addressing the instrument limitations directly (a larger per-case budget, a size cap with a defined semantics, a value-wise comparison for non-reprable return types) rather than excluding. At five problems that cost is not warranted. Rescuing `Mbpp/734` by filtering only its bad held-out inputs would be reconsidered only if corrupt-input problems were numerous enough to matter; at one, exclusion is the honest and cheaper call.

**Status update — corrected and extended by ADR-015's canonical check (2026-10-26).** MBPP+ now loads 368 problems (374 minus 6, 1.6%); with ADR-007's three and the zero-held-out drop, ten are excluded in all.
- `Mbpp/737`, `Mbpp/794`: the cause recorded above was the loader's, not the problems'. They are two of EvalPlus's three "output is not None" tasks, for which EvalPlus stores `output is not None`, a bool, as the expected value; the loader stored the canonical's `re.Match` object instead. Since ADR-015's ground-truth fix the loader stores the bool, as EvalPlus does, and both would grade. They stay excluded so that the problem set stays within the 369 the B7 box measurement generated completions for: including them would need new generations and would change the training set after B7 was pre-registered. The rescue rejected above, special-casing `re.Match` comparison, is still not done; EvalPlus's own rule made it unnecessary.
- `Mbpp/255` joins `UNGRADEABLE_CANONICAL_TASK_IDS`: one held-out expected output, 1,663,740 tuples of 77 strings, is about 1 GB as source text, four times the sandbox's 256 MB, so no program, the canonical included, can be graded on it. That test is in split B, so every completion failed the measurement there. It is the same class as `Mbpp/630`; dropping only that test would edit EvalPlus's suite.
- The frozen 100 is a stored list and contains none of the six.

---

## ADR-010

### Training box: Lambda Cloud on-demand, 1x A100 SXM4 40GB

**Context.** Phase 4 is the first phase that trains the policy, and it needs two things on the same machine: a GPU large enough to run GRPO on Qwen2.5-Coder-1.5B, and a Docker daemon, because every completion the policy produces is graded through the container sandbox (ADR-005). Phases 1-3 ran generation on Colab's free tier (T4, no Docker, session limits), which can measure a base model but cannot train one or run the real sandbox. That constraint has been open since Phase 0.

A rough memory budget for full-parameter GRPO at 1.5B: bf16 weights ~3 GB, gradients ~3 GB, AdamW optimizer states (fp32) ~12 GB, so roughly 18 GB before activations, a ~3 GB reference model if the KL term is on, and the generation engine's memory. That fits in 40 GB, but without much spare room.

**Decision.** Lambda Cloud on-demand, a single A100 SXM4 40GB ($1.99 per GPU-hour at decision time, plus tax; 30 vCPUs, 200 GiB RAM, 0.5 TiB local SSD), on the default Lambda Stack image (Ubuntu 22.04, CUDA, Docker, NVIDIA container toolkit). A Lambda persistent filesystem holds checkpoints, results and logs; it is created once the training instance's region is known, in that region.

Backup order if the primary has no capacity when needed:
1. Lambda 1x H100 PCIe 80GB ($3.29/hr, 26 vCPUs). Same account and image, no quota process, more memory headroom.
2. GCP a2-highgpu-1g (1x A100 40GB, 12 vCPUs, ~$3.67/hr on-demand in us-central1). Requires a GPU quota grant, requested at decision time; fewer vCPUs means slower grading.

Alternatives considered:
- Colab free: ruled out. T4 GPU, no Docker, session limits.
- Lambda A100 80GB: not offered as a single-GPU instance (8-GPU only).
- GCP as primary: considered. Lambda is cheaper per GPU-hour, gives more vCPUs per GPU, and needs no quota grant.
- Lambda A6000 48GB ($1.09/hr): 14 vCPUs would roughly halve sandbox grading throughput.
- Lambda GH200 96GB ($2.29/hr): ARM64 host; every dependency and the sandbox image would need ARM builds.

**Consequences.**
- Estimated 90-180 GPU-hours across Phases 4-5 (spec estimate including a 50% buffer), roughly $180-360 on the primary. Actuals are recorded here at the end of Phase 4.
- Memory is adequate but not roomy. The known levers if it gets tight are gradient checkpointing, a capped generation memory share, and an 8-bit optimizer. Any lever used is recorded, because it can constrain group size or generation length.
- No session limit, but billing runs while the instance is up, so instances are shut down when idle.
- The instance's local disk is treated as disposable. Anything that must survive lives on the persistent filesystem (mounted at `/lambda/nfs/<name>`) or is pushed to git / experiment tracking. The filesystem must be in the same region as the instance and attached at launch.
- On-demand capacity fluctuates by the hour: at decision time the 1x A100 40GB went in and out of capacity within an hour, and every other single-GPU type except the A10 was unavailable. Steps that need only Docker and CPUs (sandbox verification, grader parity, throughput) may run on a 1x A10 (same image, 30 vCPUs), so their results carry over to the primary.
- The box becomes part of the measurement path: G_baseline is re-measured here (ADR-004 status update) so dtype, generation engine and grader match every training checkpoint.
- Sandbox throughput is bounded by vCPUs (~3.5 exec/sec/core measured in Phase 1, ADR-005): 30 on the primary, 26 on the H100 PCIe backup, 12 on GCP.

**What would reverse it.** If the memory levers cannot make a stable training configuration fit in 40 GB, move to the H100 PCIe 80GB backup and record why. If Lambda capacity repeatedly blocks work, follow the backup order above. If the Section A throughput measurement shows grading is CPU-bound enough to leave the GPU idle, the first remedy is the ADR-005 warm container pool; the second is an instance with more vCPUs.

**Status update — first launch (Section A, 2026-10-21).** Launched `gpu_1x_a100_sxm4` in us-east-1 (Virginia) on the Lambda Stack 22.04 image. The box check (analysis/box_env_lambda_a100.txt) confirmed Ubuntu 22.04.5 LTS, 30 vCPUs, 216 GiB RAM, 472 GB free local disk, Docker 29.2.1 (runc, cgroup v2), and system Python 3.10.12; the repo runs on Python 3.11.17 installed with uv. `nvidia-smi` reported an A100-SXM4-80GB on this instance although the instance type is sold as 40GB. The memory budget stays planned for 40 GB, because the SKU does not guarantee the larger card on the next launch. Throughput measured on this box (ADR-005 status update) plateaus below the vCPU count, so the vCPU figures in the consequences above do not bound grading as assumed, and an instance with more vCPUs is not a remedy for the grading ceiling.

---

## ADR-011

### GRPO implementation: TRL GRPOTrainer with vLLM colocate, pinned

**Context.** ADR-003 fixes GRPO as the algorithm but not the implementation. Phase 4 needs a trainer that runs on one GPU (ADR-010), samples several completions per prompt, and computes rewards through the Docker sandbox (ADR-005) using the validated graders (ADR-004, ADR-007, ADR-008). The implementation is the optimizer, not the subject of the study, so the criterion is the fewest moving parts that can learn.

**Decision.** Hugging Face TRL `GRPOTrainer`, with generation by vLLM in colocate mode (vLLM runs inside the trainer process and shares the single GPU). Versions are resolved together and pinned exactly at introduction: trl 1.16.0, vllm 0.31.0 (the top of TRL 1.16.0's supported range, >=0.22.0 and <=0.31.0), torch 2.13.0, transformers 5.17.0, accelerate 1.15.0, datasets 5.1.0. The full resolved set (231 packages, Python 3.11, linux x86-64) is committed as requirements/train.lock. The `train` extra is installed only on the training box; CI and the CPU-tested core do not depend on it.

Alternatives considered:
- verl: production-grade and scalable, but built around Ray and multi-GPU training; heavy for one GPU and a 1.5B model.
- A hand-written GRPO loop: full control, but every optimizer bug would be the project's own, in the phase whose main risk is stability.
- Unsloth GRPO: memory-efficient, but LoRA-centred and implemented by patching library internals; less transparent.

**Consequences.**
- The reward plugs in without an adapter: TRL passes every non-prompt dataset column to the reward function as a keyword argument, so a `task_id` column lets the reward look up the `Problem` and grade with the existing harness. Reward functions may be async, which suits the I/O-bound sandbox, and can log extra per-step metrics.
- The KL penalty is off by default (beta = 0.0): it is neither applied nor logged unless set explicitly. Whether to set it is a Section C stability decision, recorded when made.
- TRL moves fast, so the exact pins are load-bearing: an upgrade is a deliberate change with its own commit, never drift.
- TRL removed `PPOTrainer` in v1.13.0; see the ADR-003 status update.

**What would reverse it.** If a stable GRPO configuration cannot be reached because of TRL itself (a trainer bug, or an unfixable interaction between vLLM colocate and the 40 GB memory budget), as distinct from ordinary hyperparameter instability, move generation to TRL's in-process transformers path first, then to another implementation, and record which failure forced it.

**Status update — experiment tracking added (Section B, B6).**

`wandb==0.30.0` joins the `train` extra for training curves (the spec's Phase 4 gate). The lock was re-resolved with the same `uv pip compile` command, which keeps existing pins as preferences: two packages were added (`wandb 0.30.0` and its dependency `platformdirs 4.13.5`), and no existing pin changed (233 packages).The newest release, `0.31.0` (2026-10-16), was passed over because it pins a beta dependency (`cwevals 0.0.1b1`); `0.30.0` (2026-09-09) has none.

---

## ADR-012

### Environment interface: a thin, stateless, plain-Python environment; no Gymnasium

**Context.** Episodes are single-shot: the observation is a prompt, the action is one completion, and the reward ends the episode. TRL's reward-function interface (ADR-011) could serve as the environment by itself, but that would tie the environment to one trainer, and the project describes itself as an RL environment. The spec asks whether a Gymnasium wrapper earns its place.

**Decision.** A small plain-Python `CodeExecEnv`: `prompt(task_id)` returns the observation, and `step(task_id, completion)` extracts the function, grades it in the sandbox and returns the reward with its details. It is stateless: the task is an argument to `step`, not state set by a reset, so a GRPO group's completions can be graded concurrently without an instance pool. The TRL reward function is a thin adapter over `step`. Gymnasium is not used.

**Consequences.**
- The environment is independent of the trainer and CPU-testable without TRL. A different trainer, such as a Phase 6 PPO arm in another library (ADR-003 status update), reuses it unchanged, so a comparison changes the optimizer and nothing else.
- No new dependency; the cost is one small layer between the trainer and the graders.
- No Gymnasium compatibility: no trainer in use consumes Gymnasium environments, and its spaces fit token sequences poorly. TRL's own environment API, aimed at multi-turn tool use, is not used either, for the same coupling reason.

**What would reverse it.** If multi-step episodes are built (FUTURE.md, item 6), the environment gains per-episode state and this interface is revisited. If a trainer that consumes Gymnasium environments is adopted, a thin Gymnasium adapter is added over `CodeExecEnv` rather than replacing it.

---

## ADR-013

### Held-out split: V0 rewards on held-out A; held-out B is the measurement and never enters any reward

**Context.** THESIS defined V0's reward as the full MBPP+ suite while also stating that the held-out suite never enters any reward. V0 as designed therefore trains on its own measurement: G counts problems that pass the visible tests but fail held-out, and V0's reward scores exactly those cases 0, so V0's gap would shrink by construction instead of acting as a control. That biases the V1-versus-V0 comparison toward H1, and it is the defect the spec cited to drop V3 ("once the held-out set enters the reward it is no longer held out"). The spec judged a three-way partition infeasible on sparse tests, but only the visible tests are sparse (median 3 per problem); the held-out suite has a median of 105 tests per problem (minimum 13; 3 of the 369 problems have fewer than 20).

**Decision.** Each problem's held-out tests are split once, before any training, into two disjoint parts:
- **B, the measurement split.** `heldout_pass` and G are measured on the visible tests plus B. B never enters any reward, in any arm.
- **A, the reward split.** V0's reward is computed on the visible tests plus A. V1's reward is computed on the visible tests only.

Split rule: for a problem with n held-out tests, rank test indices 0..n-1 by `sha256(f"heldout-split-v1:{task_id}:{i}")` (hex digest, ascending); B is the first ceil(n/2) indices of that ranking, A the rest, and each part keeps the tests' loaded order. SHA-256 ranking is identical on every machine and Python version, unlike `random.shuffle`. This decision is committed before G_baseline is re-measured on B.

**Re-baselining, pre-registered.** G_baseline is re-derived on B from the saved frozen-100 completions (analysis/completions_frozen100.json), graded by the Docker sandbox, with a bootstrap 95% CI computed as in analysis/g_baseline.md. Dynamic-range rule, fixed now: if the CI's lower bound is above zero, the 50/50 split stands. If the CI includes zero, the split moves to one-third A and two-thirds B (same ranking; B is the first ceil(2n/3)) and is re-checked once. If that CI also includes zero, training does not start until the V0 design is revisited. The Phase 3 full-suite G_baseline (+10.0 pp) remains on the record.

**Consequences.**
- The THESIS invariant holds again: the measurement never enters any reward, in any arm. V0 becomes a genuine control, a dense reward (visible plus ~52 held-out tests) against V1's sparse one (~3 tests), instead of one constrained by construction.
- Measurement is less strict than the full suite (~52 held-out tests instead of ~105), so G_baseline on B is expected to be at most the full-suite +10.0 pp.
- The standard full-suite MBPP+ pass-rate is still reported for the baseline and V1, which never train on held-out tests, so the citable number (ADR-004) survives. For V0 the full suite includes A, which V0 trains on, so it is not used as a V0 measurement.
- `heldout_pass` keeps the EvalPlus "plus" form (ADR-004 status: visible AND held-out pass), now restricted to B.

**What would reverse it.** The dynamic-range rule above. Separately, if A and B turned out to differ systematically in difficulty, the split would be redrawn under a new seed string (`heldout-split-v2`) and the reason recorded.

**Status update — re-baselined on split B (pre-registered rule applied).** Graded through the Docker sandbox from the saved frozen-100 completions (scripts/rebaseline_split_b.py; analysis/g_baseline_split_b.json). The consistency check passed: the full held-out suite reproduces the Phase 3 record problem by problem (visible 65/100, plus 55/100) and its CI exactly ([+5.0, +16.0] pp). On split B: visible 65/100, plus 55/100, **G_baseline = +10.0 pp, bootstrap 95% CI [+5.0, +16.0] pp**. The CI's lower bound is above zero, so under the pre-registered rule **the 50/50 split stands** and the two-thirds fallback was not needed. No problem's result differs between the full suite and split B: every base-model completion that fails the full held-out suite also fails at least one split-B test, so halving the measurement cost no detection at the base model. That equivalence is a property of the untrained model, not a guarantee during training, which is why the full-suite number keeps being reported for the baseline and V1. The generation-engine re-measurement on the training box (ADR-004 status update) is taken on split B.

---

## ADR-014

### H1 reward is r_sparse; every outcome has a fixed score, and infrastructure failures are never scored

**Context.** H1 is defined on a binary terminal reward (THESIS), and `r_sparse` and `r_shaped` were built and validated in Phase 2. Before training, every way a reward call can end must map to exactly one score, fixed in advance, so no outcome is scored by accident. Laying the outcomes out exposed one gap: when the sandbox itself fails (the Docker daemon is down, the image is missing, the `docker` CLI errors), no result line comes back, so `run_tests` reports a crash and the reward is 0.0. A correct program would be punished for an infrastructure fault, silently.

**Decision.** H1 uses `r_sparse` in both arms: 1.0 if every test in the arm's reward suite passes, else 0.0. V1's reward suite is the visible tests; V0's is the visible tests plus held-out split A (ADR-013), graded in one execution. `r_shaped` is kept for H2 only. Every outcome maps as follows:

| Outcome | Score |
|---|---|
| All reward-suite tests pass | 1.0 |
| Any test fails (wrong answer) | 0.0 |
| A test case exceeds its per-case budget (ADR-008) | 0.0 (that case fails) |
| The run exceeds the sandbox wall-clock limit | 0.0 |
| The program crashes, exits early, exceeds the memory limit, or raises on a blocked action (network, filesystem) | 0.0 |
| Malformed completion (does not extract into a function named `entry_point`) | 0.0, never sent to the sandbox; the malformed rate is logged per training step |
| Sandbox infrastructure failure (`docker run` exits 125/126/127, or the runtime outlives its own wall-clock limit and the backstop fires) | **never scored**: retried once; if it fails again, the reward call raises and the training step stops |

**Consequences.**
- The reward stays a pure pass indicator in {0, 1}. Format is handled by the prompt and parser (ADR-006), not by shaping, so H1's reward is not partly an H2 reward.
- A format collapse during training is visible directly in the logged malformed rate, not only indirectly as falling reward.
- No reward is ever computed from an infrastructure fault. A failed run is loud and recoverable; silently corrupted rewards would be neither. This requires the sandbox runner to report infrastructure failure separately from a program crash, which is implemented with the reward function (Section B).
- Infrastructure retries are counted and logged, so a degrading sandbox shows up before it stops a run.

**What would reverse it.** If the malformed rate rises and stays high during training (well above the base model's 2%), format is revisited per ADR-006 (an SFT warm-start first) and recorded here, rather than by adding a format penalty to H1's reward. If infrastructure failures stop runs often enough to matter, the fix is in the sandbox (ADR-005's warm pool, daemon health), never in scoring them.


**Status update — infrastructure-failure detection and retry, as built (Section B, B5).** Correction to the table above: exit codes cannot identify an infrastructure failure. With the daemon unreachable, the Docker 29 CLI (checked on 29.9.1) exits 1, the same code an uncaught Python exception produces, and a program's own `sys.exit(125)`, `sys.exit(126)` or `sys.exit(127)` reuses the others. Detection therefore uses positive evidence. The container runs `timeout 10 sh -c "echo <start marker> >&2; exec python"`: the marker is written to stderr from inside `timeout`, immediately before the program starts, and `exec` preserves the program's exit code and `timeout`'s 124. No marker means the program never started (daemon unreachable, image missing, since `docker run --pull never` fails at once instead of fetching, or a rejected command line), so the run is an infrastructure failure whatever its exit code; expiry of the 15 s backstop remains one too. With the marker present, the outcome is the program's and is scored by the table. A program cannot remove a marker written before it started, so no completion can fake an infrastructure failure and stop a run. Two cases remain unseparated and are accepted: Python missing from the image after the marker (a fixed image fault that every integration run catches), and the daemon failing mid-run after the marker. `run_tests` retries an infrastructure failure once, after a 1 s pause, and raises `SandboxInfraError` if it recurs; program failures (wrong answer, crash, timeout) are never retried. `TestResult.infra_retries` carries the count, and the TRL reward adapter logs `codeexec/infra_retries`, `codeexec/malformed_rate` and `codeexec/grading_seconds` per batch, the last for ADR-005's warm-pool rule. The marker was first placed before `timeout`, which attributed a mistyped `timeout` command line to the program; an integration run caught this and also showed four isolation tests passing without the program having run. The marker moved inside `timeout`, and those tests now assert that the program ran. See src/codeexec/sandbox/runner.py, src/codeexec/rewards/harness.py and src/codeexec/train/reward.py.

---

## ADR-015

### Test comparison matches EvalPlus's own grader

**Context.** ADR-007 made the comparison strict and called that "consistent with the strict EvalPlus held-out suite (ADR-004)". The engine check of the box baseline (Section B, B7) showed the premise is wrong for floating-point outputs. In Mbpp/234 (`volume_cube`, atol 0, canonical `l ** 3`), a completion returning `l*l*l` fails 22 of 101 held-out tests because the two expressions differ in a float's last digit (5.527548510985869 against 5.52754851098587). EvalPlus 0.3.1's evaluator (`evalplus/eval/__init__.py`, `unsafe_execute`) passes it: when atol is 0 and the expected output is a float or a list or tuple of floats, it raises atol to 1e-6 and compares with `np.allclose(..., rtol=1e-7)`. Read in full, the evaluator differs from `_compare` in four ways:
- float outputs, as above: 50 of the 369 problems have atol 0 and float expected outputs in their held-out tests, 17 in their visible tests;
- the 13 problems with atol above 0: EvalPlus adds a relative tolerance of 1e-7;
- exact match is Python equality, so `True == 1` and `1 == 1.0` (ADR-007 made bool and int distinct);
- seven problems in this set have special oracles: set equality (Mbpp/7, Mbpp/111, Mbpp/140, Mbpp/232), "not None" (Mbpp/787), and recomputation by a reference function (Mbpp/558, Mbpp/581).

A failure caused only by these differences is not overfitting, yet it has the gap's signature: visible tests with integer outputs pass and held-out tests with float outputs fail. It inflates G with cases unrelated to H1, penalizes correct programs in the reward on 17 problems, and means the full-suite number is not the citable MBPP+ number ADR-004 and ADR-013 call it. Regrading the frozen 100 with EvalPlus's float rule changes exactly one result, Mbpp/234, which is also the only disagreement in the engine check.

**Decision.** A test passes exactly when EvalPlus 0.3.1's evaluator would pass it, in the reward and the measurement alike: exact match by Python equality, then the MBPP special oracles, then the float tolerance with its type and length checks. The comparison lives in one stdlib-only module, embedded verbatim into every sandboxed harness; the sandbox has no numpy, so `np.allclose`'s element-wise rule is written out. ADR-008's per-case and per-program time limits are unchanged: they are the sandbox's limits, not the definition of correct. ADR-007's three order-unstable exclusions (Mbpp/2, Mbpp/579, Mbpp/769, which are three of EvalPlus's set-equality tasks) stay excluded, so the problem set remains the pre-registered 369. HumanEval's special oracle is added when HumanEval+ is measured in Phase 5.

**Validation and regrade, pre-registered before either runs.**
1. Every canonical solution passes all of its own tests under the new comparison.
2. Parity with EvalPlus's own evaluator: the saved completions (the frozen 100 from Colab, analysis/completions_frozen100.json, and the 369 from the box, analysis/g_baseline_box.json) are graded both ways, and per-problem visible and full-suite results must agree. Any disagreement is reported with its cause; the two systems' different time limits are the only expected source.
3. G_baseline is then re-derived from the box's saved 369 completions under the new comparison, without regenerating, with the same CI method and the same stop rule as the ADR-004 status update. That figure is the delta-G reference.

Results graded under ADR-007's comparison (the Phase 3 G_baseline, the split-B re-baseline, H0, and the first grading of the box completions) stay on the record, labelled with the comparison they used.

**Consequences.**
- "Correct" means what the benchmark means: the full-suite MBPP+ number is citable, and no rounding artifact enters G or the reward.
- The comparison is looser than ADR-007's in exactly EvalPlus's ways and no others. The loosening ADR-007 ruled out, order-insensitive comparison for every problem, stays ruled out: set equality applies only to the four problems EvalPlus names.
- One copy of the comparison replaces two that a test kept in sync.

**What would reverse it.** If the parity check finds disagreements that time limits do not explain, the port is wrong and is fixed before any training run. The decision itself would be revisited only if EvalPlus's evaluator were found to accept wrong answers on this problem set, which would be a finding about the benchmark and reported as one.

**Status update — validation step 1, the ground truth, and the platform of record (2026-10-26).**
- *Step 1 found loader bugs, not pass-rule bugs.* Run first outside Docker, 4 of 369 canonicals failed their own tests, and failed identically under ADR-007's rule: Mbpp/635 and Mbpp/787 (ground truth), Mbpp/255 (size; excluded, ADR-009 status update) and Mbpp/583 (a timeout from CPU contention; it passes in the sandbox). The loader's ground truth differed from EvalPlus's (`trusted_exec` in `evalplus/gen/util`) in two ways. It called each canonical on the stored input itself, not on a copy, so canonicals that sort or empty their argument in place rewrote the stored visible inputs of Mbpp/57, 64, 71 and 428 (left sorted, so a program returning its input unchanged passed every visible test and failed held-out: a planted gap that inflated G and that V1's reward would have taught), of Mbpp/635 (emptied, so no program could pass) and of HumanEval/20. And for the "output is not None" tasks it stored the canonical's `re.Match` object rather than `output is not None`, so Mbpp/787 could never be graded. Step 1 cannot see the four planted gaps, because a canonical passes tests built from its own mutated inputs; a dataset test now compares every stored input with EvalPlus's. The loader builds ground truth as EvalPlus does (commit 50f3c50). Held-out inputs, and every problem's test count and order, were unaffected, so splits A and B are unchanged; none of these problems is in the frozen 100.
- *Result.* In the Docker sandbox on macOS, 367 of 368 canonicals pass all their visible and held-out tests, run in one program per problem (tests/test_canonicals.py); on Linux, outside Docker, 368 of 368.
- *Platform of record.* The exception, Mbpp/590, differs on one of its 104 tests by one unit in the last place. Its expected value needs bit-exact floats, and `cmath.polar`'s `hypot` rounds differently in macOS's math library (correctly rounded on this input) and in glibc's. Expected values are computed on the host and graded in the Linux sandbox, so such results agree only where the two share a math library. **Linux is the platform of record:** the canonical check of record, step 2's parity check, step 3's regrade and training run on the Linux training box; on macOS, Mbpp/590 is the one problem the check allows to fail. The sandbox's base image is pinned by digest, `python:3.11-slim@sha256:cce12316475d7d56d605f76e1d811d0dc4e7227f2d4afd29951521264f504ff8`, the index the Section A grader-parity check ran on (Python 3.11.17, glibc 2.41), because the tag had already moved (commit da4abf0). If the canonical check fails on the box, expected values are computed inside the sandbox image instead.
- *The frozen 100 under the new rule.* Exactly one verdict changes, Mbpp/234, from fail to pass: visible 65, plus 56, **G = +9.0 pp, 95% CI [+4.0, +15.0]**, on split B and the full suite alike (was +10.0, [+5.0, +16.0]), the same on macOS and Linux. analysis/g_baseline_split_b.json stays as the ADR-007-rule record; tests/test_gap.py pins the one change, and the engine check in `measure_model` will report Mbpp/234 against that record.
- *Steps 2 and 3, amended before either runs.* Both run on the training box, over 368 problems: the box's completions for Mbpp/255 are set aside, and the count is reported.
- *Found on the way.* HumanEval/83 and 139 cannot be graded yet: their expected values include integers of more than 4,300 digits, which Python 3.11's `repr` refuses, so the harness cannot write them. No measurement uses HumanEval+ yet; this is fixed before Phase 5.

**Status update — the checks of record, on the training box (2026-10-28 UTC).** `scripts/box_adr015_gate.sh` ran all three checks on the Lambda A100 box at commit 1f2b8fb: Ubuntu 22.04.5, with glibc 2.35 and Python 3.11.18 on the host, and the pinned sandbox image with Python 3.11.17 and glibc 2.41 (analysis/adr015_box_environment.txt).
- *Step 1.* Every canonical passes all of its own tests in the sandbox, strict on Linux and Mbpp/590 included; all 43 integration and dataset tests pass.
- *Step 2.* Parity holds on both sets, with no unexplained disagreement (analysis/evalplus_parity_frozen100.json, analysis/evalplus_parity_box.json). The visible verdicts agree on every graded problem: 65 of the 98 graded in the frozen 100, 251 of the 366 graded from the box. The full-suite verdicts differ only where EvalPlus's time limits failed a correct, slow solution that ours passed, and EvalPlus with generous limits passed it too: Mbpp/566 in both sets and Mbpp/160 in the box set, so EvalPlus counts 55 and 205 where ours counts 56 and 207. EvalPlus ran 15 workers on the 30 vCPUs, its own default; in a preview with one worker, neither problem disagreed.
- *Step 3.* The regrade: G_baseline = +12.0 pp, 95% CI [+8.7, +15.5], n = 368, the ΔG reference (ADR-004 status update).
- *Our time limits make some verdicts depend on load too.* Mbpp/160's completion is correct but quadratic, about 0.9 s on each of five held-out inputs, close to ADR-008's 2 s per test and 10 s per program when 24 programs run at once. Graded twice by the sandbox on the box, minutes apart, its full suite passed in the parity run and failed in the regrade; split B passed both times. A verdict that depends on load is measurement noise, and a policy that learns slow solutions would add more of it. Section C settles how measurement handles it, for instance by regrading alone any program that ran out of time, before the first training run is measured.