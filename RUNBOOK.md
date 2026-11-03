# Runbook: training sessions on the box

The six H1 runs (THESIS: V0 and V1, seeds 0 to 2) are trained and measured on the
Lambda box (ADR-010) in short sessions, so the GPU is billed only while it works.
Every rule a session applies was fixed before the first run (DECISIONS.md, Section C);
this file only puts them in order. Nothing here is decided on the box.

- **Session 1** trains the first seed of each arm, applies the checks that can change
  the setup for every run (C4, C5, ADR-005), and measures both runs. About 3.5 hours.
- **Session 2** trains and measures the other four. About 6.5 hours, or two sessions of
  two runs each.

Commands marked **Mac** run on your Mac; all others run on the box, in the repo folder,
with the venv active, inside `tmux`.

---

## Before a session

1. **Mac:** everything is committed and pushed, and `git status` is clean. Note the commit
   (`git log -1 --oneline`).
2. In the Lambda console, the filesystem `codeexec-fs` exists in **us-east-1**; create it
   once if not. It can only be attached at launch (ADR-010 status update, C6).
3. Launch `gpu_1x_a100_sxm4` in **us-east-1**, with `codeexec-fs` and your SSH key
   attached. With no capacity, try ADR-010's backup types, in us-east-1 first.
4. **Mac:** set the new IP as `HostName` under `Host lambda` in `~/.ssh/config`.

## Set up the box

```bash
ssh lambda
sudo usermod -aG docker "$USER"   # then exit and `ssh lambda` again
tmux new -s s                      # reattach after a dropped connection: tmux attach -t s

git clone https://github.com/shilpamusale/codeexec-rl.git && cd codeexec-rl
git log -1 --oneline               # the commit noted on the Mac
curl -LsSf https://astral.sh/uv/install.sh | sh && source "$HOME/.local/bin/env"
uv python install 3.11 && uv venv -p 3.11 .venv && source .venv/bin/activate
uv pip install -e ".[dev,train]" -c requirements/train.lock
bash scripts/box_check.sh
wandb login                        # the key goes here or in WANDB_API_KEY, nowhere else

FS=$(ls -d /home/ubuntu/codeexec-fs /lambda/nfs/codeexec-fs 2>/dev/null | head -1)
echo "filesystem: ${FS:?not mounted, so terminate and relaunch with it attached}"
```

The box never gets a git identity or GitHub credentials: results come back to the Mac
with `scp` and are committed there.

## The gate, every session

```bash
bash scripts/box_adr015_gate.sh
REF=runs/adr015/g_baseline_box_adr015.json
```

It must end with `Gate passed`. It re-derives the ΔG reference with the regrade-alone
procedure (ADR-008 status update, C1) and applies ADR-004's stop rule. A failure stops
the session: nothing is trained. Every run of the session is measured against this
reference, at this commit (`codeexec.eval.delta_g` refuses anything else), so there is no
`git pull` on the box mid-session except in the restarts below, which run the gate again.

## Train one run

```bash
python -m codeexec.train.launch --arm V1 --seed 0 2>&1 | tee runs/V1-seed0.train.log
python -m codeexec.train.checks runs/V1-seed0
```

About 70 minutes, then the checks (C4, C5 and ADR-005, from the trainer state). A run that
fails starts over from step 0 with the same seed, never resumed (ADR-010 status update,
C6): `mv runs/V1-seed0 runs/V1-seed0.failed-1`, then launch it again. Record the failure
and its cause.

## Session 1

Train **V1 seed 0**, then **V0 seed 0**, each with its checks. Then apply the
pre-registered decisions, in this order. A change to the setup is made on the Mac (the
edit, and a status update recording the numbers that triggered it) and pushed; on the
box, `git pull`, move the gate's folder and both runs aside (`mv runs/adr015
runs/adr015.before`, and the same for each run), run the gate again and start both runs
over.

**C4 (ADR-011 status update).** If either run is not stable, every run moves to
`beta = 0.04`.

**C5 (ADR-011 status update).** If either run did not learn, the learning rate doubles to
`2e-6`, once.

**ADR-005 (status update, D1).** On the first stable run, read `grading_verdict`. A warm
pool, if called for, is built and parity-checked between sessions; this session continues
without it.

**C3 (ADR-014 status update).** On each run, the malformed completions, and twenty of
those at the limit:

```bash
python - runs/V1-seed0 <<'EOF'
import glob, sys
import pandas as pd
paths = sorted(glob.glob(f"{sys.argv[1]}/completions/completions_*.parquet"))
log = pd.concat(pd.read_parquet(path) for path in paths)
malformed = log[log["codeexec/malformed"]]
at_limit = malformed[malformed["codeexec/at_limit"]]
print(f"{len(log)} completions, {len(malformed)} malformed, {len(at_limit)} of them at the limit")
for _, row in at_limit.sample(min(20, len(at_limit)), random_state=0).iterrows():
    print(f"\n----- step {row['step']}\n{row['completion']}")
EOF
```

If most malformed completions are not at the limit, ADR-006's reversal path applies. If
most are, the twenty are read: loops keep their 0; if most are solutions cut short, the
budget is revisited before any further run.

Then measure both runs, about 15 minutes each:

```bash
bash scripts/box_measure_run.sh runs/V1-seed0 "$REF"
bash scripts/box_measure_run.sh runs/V0-seed0 "$REF"
```

The spec's Phase 4 gate needs V1 seed 0 stable, with more visible passes at step 200 than
the reference's. Then keep and export the session, and terminate (below).

## Session 2

The gate, then **V1 seed 1, V0 seed 1, V1 seed 2, V0 seed 2**, alternating arms so that
neither arm gets the box's better hours. Each run gets its checks: C4 applies to every
run, and a run that is not stable stops the session under C4's rule. Then
`box_measure_run.sh` for each, and keep and export. Split into two sessions of two runs if
a 6.5-hour session is too long; each starts with its own gate.

## Keep and export, then terminate

To the filesystem: the four checkpoints, `run.json`, the completions log, the checks and
the measurements of each run, but not `final/`, which repeats `checkpoint-200` (ADR-010
status update, C6):

```bash
mkdir -p "$FS/runs"
for run in runs/V1-seed0 runs/V0-seed0; do   # this session's runs
    rsync -a --exclude final/ "$run" "$FS/runs/"
    cp "$run.train.log" "$FS/runs/"
done
rsync -a runs/adr015/ "$FS/runs/gate-$(date -u +%F)/"
```

For git, the small files, gathered under `runs/` so the tree stays clean:

```bash
EXPORT=runs/export/$(date -u +%F)
mkdir -p "$EXPORT"
cp "$REF" "$EXPORT/delta_g_reference.json"
cp runs/adr015/environment.txt runs/adr015/stop_rule.txt "$EXPORT/"
for run in runs/V1-seed0 runs/V0-seed0; do   # this session's runs
    mkdir -p "$EXPORT/${run#runs/}"
    cp "$run/run.json" "$run/checks.json" "$run/measure/"*.json "$EXPORT/${run#runs/}/"
done
```

**Mac:**

```bash
mkdir -p analysis/sessions
scp -r lambda:codeexec-rl/runs/export/<date> analysis/sessions/
git add analysis/sessions/<date> && git commit && git push
```

Then terminate the instance in the Lambda console; `codeexec-fs` stays. Record the hours
and cost in ADR-010's spend table.

## Time and cost

At $1.99 an hour, before tax:

| Session | Work | Hours | Cost |
|---|---|---|---|
| 1 | setup, gate, 2 runs, checks, 2 measurements, export | ~3.5 | ~$7 |
| 2 | setup, gate, 4 runs, checks, 4 measurements, export | ~6.5 | ~$13 |

A run is about 70 minutes (200 steps at the first step's 19 s, plus loading and
checkpoints) and a measurement about 15 (four checkpoints). The filesystem adds about $30
a month while the checkpoints are kept (ADR-010 status update, C6). A restart under C4 or
C5 repeats the session's runs.