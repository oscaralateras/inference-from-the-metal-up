#!/usr/bin/env bash
# One command for a combined T10 + T11 session on a rented single-GPU pod.
#
#   curl -fsSL https://raw.githubusercontent.com/oscaralateras/inference-from-the-metal-up/main/scripts/t10_t11_session.sh -o s.sh
#   bash s.sh gate
#
# T10 and T11 share a pod deliberately: both need one GPU, neither needs more, and both read the
# same hardware probe. Running them in one session is what lets the cross-topic test assert they
# describe the same silicon — and it halves the rental.
#
# Subcommands, in the order they should be run:
#
#   gate     one GPU, a way to evict the page cache, and enough disk. ~10 seconds, and it
#            decides whether the pod is usable before anything is downloaded onto it.
#   setup    clone + uv sync + make probe. The slow part (~15 min), mostly downloading torch.
#   t10      cold start: cold/warm x read/mmap, then H2D pinned vs pageable, then the bands.
#   t11      fusion vs launch: the 2x2 across the batch sweep, then the crossover.
#   control  band 3's falsification run — repeats T11's sweep at shorter chain lengths.
#   all      setup -> t10 -> t11 -> control, with the gate in front.
#
# The gate's real check is page-cache eviction, because that is the one thing T10 cannot fake:
# without it every "cold" number is a warm one wearing a disk's name. It does NOT require root --
# posix_fadvise evicts a single file's pages unprivileged, and that is the mechanism T10 prefers.
set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/oscaralateras/inference-from-the-metal-up}"

# Local container disk, NOT /workspace. RunPod mounts /workspace as a network volume (MooseFS)
# which manages roughly 250 small-file creates per second; unpacking torch and vLLM writes well
# over 100,000 files, so `uv sync` grinds there while the network sits idle. That cost real rented
# time in the T9 session before it was diagnosed.
#
# For T10 this matters twice over: the weight file it reads must live on the container's NVMe, or
# stage 1 measures a network filesystem and the whole cold-start decomposition describes MooseFS
# rather than a disk.
WORKDIR="${WORKDIR:-/root/ifmu}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/root/.cache/uv}"
WEIGHT_FILE="${WEIGHT_FILE:-/root/t10_weights.bin}"
GIB="${GIB:-8}"

say() { printf '\n\033[1m== %s ==\033[0m\n' "$*"; }
die() { printf '\n\033[1;31mFAILED: %s\033[0m\n' "$*" >&2; exit 1; }

gate() {
  say "gate"
  command -v nvidia-smi >/dev/null || die "no nvidia-smi — this is not a GPU node."
  nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

  # T10 needs to evict the page cache, and it has two ways to do it. The unprivileged one is the
  # primary: posix_fadvise(POSIX_FADV_DONTNEED) drops one file's cached pages and needs no root at
  # all, which matters because rented containers usually mount /proc/sys read-only. The global
  # drop is the fallback. Only having neither is fatal, and this reports which one you have rather
  # than insisting on the one you probably do not.
  if [ -w /proc/sys/vm/drop_caches ]; then
    say "page-cache eviction: BOTH available (posix_fadvise + global drop_caches)"
  elif python3 -c 'import os; raise SystemExit(0 if hasattr(os, "posix_fadvise") else 1)'; then
    say "page-cache eviction: posix_fadvise only (unprivileged, per-file) — this is fine, and is
what T10 prefers anyway. The global drop is unavailable, which is normal in a container."
  else
    die "no way to evict the page cache — neither posix_fadvise nor a writable \
/proc/sys/vm/drop_caches. T11 would still run; T10's cold column could not be measured, and a \
cold run that silently ran warm is worse than no run."
  fi

  # Free space for the synthetic weight file, plus headroom for the venv and torch.
  local avail
  avail=$(df -BG --output=avail "$(dirname "$WEIGHT_FILE")" | tail -1 | tr -dc '0-9')
  [ "$avail" -ge $((GIB + 30)) ] || die "only ${avail}G free at $(dirname "$WEIGHT_FILE"); \
need ~$((GIB + 30))G for a ${GIB}G weight file plus torch."

  say "gate PASSED — 1 GPU, page-cache eviction available, ${avail}G free"
}

setup() {
  say "setup (~15 min, mostly torch)"
  command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"

  [ -d "$WORKDIR" ] || git clone "$REPO_URL" "$WORKDIR"
  cd "$WORKDIR"
  uv python install 3.12
  uv sync --frozen

  # One probe, shared by both topics — that shared session_id is what the cross-topic test checks.
  make probe
}

t10() {
  cd "$WORKDIR"; export PATH="$HOME/.local/bin:$PATH"
  say "T10 — cold start (~5 min, mostly writing and re-reading a ${GIB}G file)"
  uv run python -m topics.t10_os_virtual_memory.predict --write
  uv run python -m topics.t10_os_virtual_memory.measure --gib "$GIB" --path "$WEIGHT_FILE"
  uv run python -m topics.t10_os_virtual_memory.plot
}

t11() {
  cd "$WORKDIR"; export PATH="$HOME/.local/bin:$PATH"
  say "T11 — fusion vs launch (~10 min, mostly torch.compile warmups)"
  uv run python -m topics.t11_compiler_runtime.predict --write
  make t11
}

control() {
  cd "$WORKDIR"; export PATH="$HOME/.local/bin:$PATH"
  say "T11 — band 3's control at shorter chain lengths (~15 min)"
  make t11-control
}

case "${1:-all}" in
  gate)    gate ;;
  setup)   gate; setup ;;
  t10)     t10 ;;
  t11)     t11 ;;
  control) control ;;
  all)     gate; setup; t10; t11; control
           say "DONE — commit results, then DESTROY THE POD. Idle time is the only real cost." ;;
  *)       die "unknown subcommand '${1}'. Use: gate | setup | t10 | t11 | control | all" ;;
esac
