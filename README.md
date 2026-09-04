# CodeExec-RL

**An execution-grounded RL environment for code generation, and a study of visible-test overfitting.**

A policy model reads a programming task and emits a solution. The solution runs in an isolated container against a test suite. The pass result is the reward. GRPO optimizes the policy.

The research question is whether that reward is Goodhartable: does optimizing against the tests you can see widen the gap between what the model passes and what it actually gets right?

**Start with [THESIS.md](THESIS.md).** The hypotheses were written before the implementation, and the commit history shows it.

---

## Status

**Phase 0 — Frame.** Repository skeleton, CI, and research framing only. No implementation yet.

| Phase | State |
|---|---|
| 0 · Frame | In progress |
| 1 · Sandbox | Not started |
| 2 · Task and test layer | Not started |
| 3 · Base-model measurement (H0, `G_baseline`) | Not started |
| 4 · Training loop | Not started |
| 5 · H1 result | Not started |

---

## What this is

- A containerized, concurrent, reproducible sandbox for executing untrusted model-generated code, with measured throughput.
- An execution-reward harness with sparse and shaped variants, fully logged.
- A visible/held-out test partition built on EvalPlus.
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

## Repository layout

```
src/codeexec/
  sandbox/     isolated execution: containers, limits, async pool
  tasks/       MBPP + EvalPlus loading, visible/held-out partition, perturbations
  reward/      execution reward, sparse and shaped variants
  train/       GRPO loop and per-arm configs
  detect/      AST-based degenerate-solution detection
  eval/        pass-rate measurement, held-out and OOD
tests/         unit tests, including the sandbox fixture suite
analysis/      writeups: baseline, H1 result
```

Three documents carry the research signal:

- **THESIS.md** — H0, H1, H2 with falsification criteria and fixed metric definitions. Updated with results, including null ones.
- **DECISIONS.md** — architecture decision records. Each states what would reverse it.
- **FUTURE.md** — deferred work, kept out of the build deliberately.

---

## Development

```bash
pip install -e ".[dev]"
make check      # ruff, mypy, pytest
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

MIT
