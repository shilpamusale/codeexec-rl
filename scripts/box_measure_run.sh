#!/usr/bin/env bash
# Measure every checkpoint of one training run, then its ΔG (THESIS; ADR-011, C5).
#
# Run on the training box, in the session whose gate produced the reference, from
# the repo root of a clean checkout with the venv active and the GPU otherwise idle:
#
#     bash scripts/box_measure_run.sh runs/V1-seed0 runs/adr015/g_baseline_box_adr015.json
#
# Each checkpoint is measured by measure_model in its own process, because vLLM
# does not hand the GPU back between models in one process. The reports go to
# <run>/measure/step-<N>.json; then delta_g refuses any not taken on the
# reference's instrument, the same grading commit included, which is why this runs
# in the gate's session, and writes <run>/measure/delta_g.json. runs/ is ignored
# by git, so every measurement still sees a clean tree. MEASURE_GRADING_WORKERS
# sets the grading concurrency (default 24, as in the gate). Exit code 0 only if
# every checkpoint was measured and ΔG was computed.

set -euo pipefail

usage="usage: bash scripts/box_measure_run.sh <run dir> <reference report>"
RUN="${1:?$usage}"
REFERENCE="${2:?$usage}"
WORKERS="${MEASURE_GRADING_WORKERS:-24}"
OUT="$RUN/measure"

die() { printf 'MEASUREMENT FAILED: %s\n' "$*" >&2; exit 1; }

# The last step, where G_final is measured, from the shared config (ADR-011, C5).
FINAL=$(python -c 'from codeexec.train.config import load_config; print(load_config().grpo["max_steps"])')

[ "$(uname -s)" = Linux ] || die "not Linux, so not the platform of record (ADR-015)"
[ -z "$(git status --porcelain)" ] || die "the working tree has uncommitted changes"
[ -f "$REFERENCE" ] || die "no reference report at $REFERENCE"
[ -d "$RUN/checkpoint-$FINAL" ] || die "$RUN has no checkpoint-$FINAL, where G_final is measured"
if [ -e "$OUT" ] && [ -n "$(ls -A "$OUT")" ]; then
    die "$OUT holds an earlier measurement; move it away first"
fi
mkdir -p "$OUT"

# In step order (version sort: checkpoint-50 before checkpoint-100).
mapfile -t checkpoints < <(printf '%s\n' "$RUN"/checkpoint-* | sort -V)
for checkpoint in "${checkpoints[@]}"; do
    step="${checkpoint##*-}"
    printf '\n== %s\n' "$checkpoint"
    python -m codeexec.eval.measure_model --model "$checkpoint" \
        --out "$OUT/step-$step.json" --max-workers "$WORKERS" \
        2>&1 | tee "$OUT/step-$step.log"
done

printf '\n== ΔG against %s\n' "$REFERENCE"
python -m codeexec.eval.delta_g "$RUN" --reference "$REFERENCE" 2>&1 | tee "$OUT/delta_g.log"