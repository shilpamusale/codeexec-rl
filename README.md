# CodeExec-RL

![CI](https://github.com/shilpamusale/codeexec-rl/actions/workflows/ci.yml/badge.svg)

**An execution-grounded RL environment for code generation, and a study of visible-test overfitting.**

A policy model reads a programming task and emits a solution. The solution runs in an isolated container against a test suite. The pass result is the reward. GRPO optimizes the policy.

The research question is whether that reward is Goodhartable: does optimizing against the tests you can see widen the gap between what the model passes and what it actually gets right?

**Start with [THESIS.md](THESIS.md).** The hypotheses were fixed at the outset — they land in the first substantive commit, before any implementation fills the skeleton in.

---

## Status

**Phase 4 — Training loop: in progress.** The sandbox, the task layer and the base-model measurements are complete. The RL environment, the TRL reward adapter, the shared training config, the launcher and the G measurement are built and tested, and a first training step has run on the training GPU. The test comparison now matches EvalPlus's own grader ([ADR-015](DECISIONS.md#adr-015)). Next, on the training box: the canonical check of record, parity with EvalPlus's evaluator, and the regrade that sets `G_baseline` over all 368 problems; then the first training run.

| Phase | State |
|---|---|
| 0 · Frame | Done |
| 1 · Sandbox | Done — Docker isolation (no network, read-only filesystem, memory, process and time limits) and a concurrent pool; throughput plateaus near 39 executions/s on the training box ([analysis/sandbox.md](analysis/sandbox.md)) |
| 2 · Task and test layer | Done — MBPP+ (368 problems after exclusions), HumanEval+ held out, a perturbation generator, sparse and shaped rewards |
| 3 · Base-model measurement (H0, `G_baseline`) | Done — H0 clean (perturbed drop +3.0 pp, McNemar p = 0.549; [analysis/h0.md](analysis/h0.md)); `G_baseline` = +10.0 pp, 95% CI [+5.0, +16.0] on the frozen 100 under the original comparison ([analysis/g_baseline.md](analysis/g_baseline.md)); +9.0 pp, [+4.0, +15.0] under EvalPlus's ([ADR-015](DECISIONS.md#adr-015)) |
| 4 · Training loop | In progress |
| 5 · H1 result | Not started |

---

## What this is

- A containerized, concurrent, reproducible sandbox for executing untrusted model-generated code, with measured throughput.
- An execution-reward harness with sparse and shaped variants, fully logged.
- A visible/held-out test partition built on MBPP+.
- A contamination probe and an untrained baseline, both reported before any training.
- A GRPO training loop on Qwen2.5-Coder-1.5B, cheap enough to run many times.
- One ablation at three seeds per arm, with a limitations-led writeup.
- An AST-based detector for degenerate solutions, with its false-positive rate characterized.

## What this is not

Stated plainly, because scope creep is the main risk to a project like this:

- **Not a production code model.** A 1.5B model will not rival frontier coders. That is not the claim, and no result here should be read as one.
- **Not multi-file or repo-level engineering.** Function-level tasks only, deliberately, to isolate the execution-reward question from the much harder problem of navigating a codebase.
- **Not multi-step or iterative editing.** Single-generation episodes only. Iterative editing is a named future extension.
- **Not an agent harness.** Single-policy by design. No tool use, no orchestration, no planning loop.
- **Not a claim about frontier-scale behavior.** Whatever shows up at 1.5B on function-level tasks may or may not hold at scale, and the writeups say so.

---

```
src/codeexec/
  sandbox/     isolated execution: Docker runner, limits, async pool
  tasks/       MBPP+ / HumanEval+ loading, the held-out split (A/B)
  perturb/     semantics-preserving perturbations for the H0 probe
  rewards/     test harness and reward functions (sparse, shaped)
  env.py       the RL environment: prompt in, completion out, reward back
  train/       shared GRPO config, TRL reward adapter, training launcher
  eval/        extraction, grading, and the G measurement
  detect/      AST-based degenerate-solution detection (Phase 5)
configs/       grpo.toml: the settings every training run shares
scripts/       one-off measurement and verification scripts
tests/         unit and integration tests, including the sandbox fixture suite
analysis/      results and writeups: sandbox, H0, G_baseline
```

Three documents carry the research signal:

- **THESIS.md** — H0, H1, H2 with falsification criteria and fixed metric definitions. Updated with results, including null ones.
- **DECISIONS.md** — architecture decision records. Each states what would reverse it.
- **FUTURE.md** — deferred work, kept out of the build deliberately.

---

```bash
pip install -e ".[dev]"
docker build -t codeexec-sandbox src/codeexec/sandbox/
make check                  # ruff, mypy, unit tests: what CI runs
pytest -q -m integration    # real containers; needs Docker running
```

Training runs on a single GPU box (ADR-010), with the training stack pinned in `requirements/train.lock`:

```bash
pip install -e ".[dev,train]" -c requirements/train.lock
python -m codeexec.train.launch --arm V1 --seed 0
```

CI runs the same checks on every commit.

---

## Definition of done

A stranger clones this repo and can:

1. Read THESIS.md and understand H0 and H1 before touching code.
2. Execute untrusted code in isolation and see the sandbox throughput curve.
3. Read the H0 contamination result and know whether the benchmark can be trusted.
4. See `G_baseline` — the gap before any training.
5. Run one arm from a config file and watch the two pass-rates diverge, or not.
6. See `ΔG` across both arms with seed variance.
7. See the degeneracy detector's output and its false-positive rate.
8. Read a writeup that leads with limitations and ends on what would falsify it.

**"Done" means the investigation is legible and reproducible — not that H1 came out true.** A cleanly documented null satisfies this. H0 is the exception: if H0 fails, the H1 result is not negative, it is uninterpretable, and the task distribution has to change first.

---

## License

[License](LICENSE)

