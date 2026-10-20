#!/usr/bin/env bash
# Training-box environment check (Phase 4, Section A).
#
# Reports what the box actually has -- GPU, vCPUs, RAM, disk, Python, Docker --
# and fails loudly if something the project needs is missing. Read-only: it
# installs nothing and changes nothing. Run it once right after launching:
#
#     bash scripts/box_check.sh
#
# Exit code: 0 if every required check passed, 1 otherwise. WARN lines are
# things to fix during setup (e.g. Python 3.11), not reasons to abandon the box.
# Deliberately prints no hostname or IP, so the output is safe to commit.

set -u  # unset variables are errors; NOT -e, so one failed check doesn't hide the rest

failures=0

ok()   { echo "  [OK]   $*"; }
warn() { echo "  [WARN] $*"; }
fail() { echo "  [FAIL] $*"; failures=$((failures + 1)); }

echo "== Host"
# shellcheck disable=SC1091  # os-release exists on every modern Linux
. /etc/os-release && ok "OS: ${PRETTY_NAME}"
ok "Kernel: $(uname -r)"

echo "== CPU, memory, disk"
# vCPUs bound sandbox throughput (~3.5 exec/sec/core in Phase 1), so record them.
ok "vCPUs: $(nproc)"
ok "RAM: $(awk '/MemTotal/ {printf "%.0f GiB", $2 / 1024 / 1024}' /proc/meminfo)"
ok "Disk free on /: $(df -h --output=avail / | tail -1 | tr -d ' ')"
if [ -d /lambda/nfs ]; then
    ok "Persistent filesystem(s): $(ls /lambda/nfs)"
else
    warn "No persistent filesystem attached (expected for Section A; needed for training)"
fi

echo "== GPU"
if command -v nvidia-smi >/dev/null 2>&1; then
    ok "GPU: $(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader)"
else
    fail "nvidia-smi not found: no NVIDIA driver, so no GPU training on this box"
fi

echo "== Python"
ok "System python3: $(python3 --version 2>&1)"
if command -v python3.11 >/dev/null 2>&1; then
    ok "python3.11 available: $(python3.11 --version 2>&1)"
else
    warn "python3.11 not found; the repo needs >=3.11 (install with uv during setup)"
fi

echo "== Docker"
if ! command -v docker >/dev/null 2>&1; then
    fail "docker not installed: the sandbox (ADR-005) cannot run"
elif ! docker_err=$(docker info 2>&1 >/dev/null); then
    # Show docker's own first error line rather than guessing the cause.
    fail "docker installed but 'docker info' failed: $(echo "$docker_err" | head -1)"
    echo "         If it says 'permission denied': sudo usermod -aG docker \$USER, then log out and back in"
else
    ok "$(docker --version)"
    ok "Runtime: $(docker info --format '{{.DefaultRuntime}}'), cgroup v$(docker info --format '{{.CgroupVersion}}')"
    # Prove a container can start AND run with no network -- the sandbox's
    # own isolation mode. Pulls the tiny hello-world image on first run.
    if docker run --rm --network none hello-world >/dev/null 2>&1; then
        ok "Container runs with --network none"
    else
        fail "could not run a container with --network none"
    fi
fi

echo
if [ "$failures" -eq 0 ]; then
    echo "RESULT: all required checks passed"
    exit 0
fi
echo "RESULT: ${failures} required check(s) failed"
exit 1