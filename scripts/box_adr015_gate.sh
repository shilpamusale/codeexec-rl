#!/usr/bin/env bash
# ADR-015's checks of record, on the training box (Linux: the platform of record).
#
# Run once per box session, after setup and before any training, from the repo
# root of a clean checkout with the venv active:
#
#       bash scripts/box_adr_015_gate.sh
#
# In order, stopping at the first failure:
#   1.  build the sandbox image from this commit's pinned Dockerfile, and record
#       the environment(host and sandbox Python and gilbc, Docker, EvalPlus)
#   2.  make check, then every integration and dataset test, strict on Linux:
#       the canonical check (ADR-015 validation step 1) and the frozen-100
#       instrument check are among them
#   3.  parity with EvalPlus's own evaluator, on the frozen 100 and on the box's 
#       completions (step 2): no unexplained disagreement.
#   4.  the regrade of the box's completions (step 3), then ADR-004's stop rule:
#       training starts only if G_baseline's 95% CI excludes zero
# 
# Everything is written to runs/adr015/, which git ignores, so each step still 
# sees the clean tree the reports require. Copy that folder back and commit the 
# reports on your own machine; the box has no git identity. Takes ~15 minutes. 
# GATE_GRADING_WORKERS sets the sandbox concurrency of steps 3 and 4 (default
# 24: ADR-005's plateau, and what the B7 measurement used); lower it on a box
# with fewer vCPUs. Exit code 0 only if every step passed.

set -euo pipefail

OUT=runs/adr015
WORKERS="${GATE_GRADING_WORKERS:-24}"

step() {printf '\n==%s\n' "$*";}
die()  {printf 'GATE FAILED: %s\n' "$*" >&2; exit 1;}

[ "$(uname -s)" = Linux] || die "not Linux, so not the platform of record (ADR-015)"
[ -z "$(git status --porcelain)"] || die "the working tree has uncommitted changes"

if [ -e "$OUT" ] && [ -n "$(ls -A "$OUT")" ]; then
    die "$OUT holds an earlier run; move it away first"
fi
mkdir -p "$OUT"

step "1/4 Sandbox image and environment"
docker build -q -t codeexec-sandbox src/codeexec/sandbox/ >/dev/null
{
    echo "date: $(date -u +%FT%TZ)"
    echo "commit: $(git rev-parse HEAD)"
    # shellcheck disable=SC1091  # os-release exists on every modern Linux
    echo "os: $(. /etc/os-release && echo "$PRETTY_NAME"), kernel $(uname -r), $(nproc) vCPUs"
    echo "host: Python $(python -c 'import sys; print(sys.version.split()[0])'), $(ldd --version | head -1)"
    echo "sandbox: $(grep '^FROM' src/codeexec/sandbox/Dockerfile)"
    echo "sandbox: Python and glibc $(docker run --rm --network none codeexec-sandbox \
        python -c 'import platform, sys; print(sys.version.split()[0], platform.libc_ver())')"
    echo "docker: $(docker --version)"
    echo "evalplus: $(python -c 'from importlib.metadata import version; print(version("evalplus"))')"
} | tee "$OUT/environment.txt"

setp "2/4 make check, then every integration and dataset test"
make check 2>&1 | tee "$OUT/make_check.log"
pytest -m "integration dataset" 2>&1 | tee "$OUT/integration_and_dataset.log"

step "3/4 Parity with EvalPlus's own evaluator"
for name in frozen100 box; do
    case "$name" in
        frozen100) completions=analysis/completions_frozen100.json ;;
        box) completions=analysis/g_baseline_box.json ;;
    esac
    python -m codeexec.eval.evalplus_parity --completions "$completions" \
        --out "$OUT/evalplus_parity_$name.json" --max-workers "$WORKERS" \
        2>&1 | tee "$OUT/evalplus_parity_$name.log"
done

step "4/4 Regrade of the box's completions, and ADR-004's stop rule"
python -m codeexec.eval.measure_model --completions analysis/g_baseline_box.json \
    --out "$OUT/g_baseline_box_adr015.json" --compare analysis/g_baseline_split_b.json \
    --max-workers "$WORKERS" 2>&1 | tee "$OUT/regrade.log"
python - "$OUT/g_baseline_box_adr015.json" <<'EOF' 2>&1 | tee "$OUT/stop_rule.txt"
import json
import sys

s = json.load(open(sys.argv[1]))["all_problems"]
lo, hi = s["ci_pp"]
print(f"G_baseline = {s['gap_pp']:+.1f} pp, 95% CI [{lo:+.1f}, {hi:+.1f}], n = {s['n']}")
if lo <= 0:
    sys.exit("ADR-004's stop rule: the CI includes zero, so training does not start")
print("ADR-004's stop rule: the CI excludes zero, so training may start")
EOF

step "Gate passed"
echo "Copy $OUT back to your machine, then commit the reports and environment.txt."