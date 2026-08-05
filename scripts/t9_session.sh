#!/usr/bin/env bash
# One command for a T9 session on a rented multi-GPU pod.
#
#   curl -fsSL https://raw.githubusercontent.com/oscaralateras/inference-from-the-metal-up/main/scripts/t9_session.sh | bash -s -- gate
#
# Subcommands, in the order they should be run:
#
#   gate     topology only. Costs ~30 seconds and decides whether to keep the pod. Run this
#            FIRST and run nothing else if it fails.
#   setup    clone + uv sync + make probe. The slow part (~20 min), mostly downloading torch.
#   run      stages 1-2: the sweep, the fit, the decode operating points, the plots.
#   tp       stage 3: the row-parallel layer, scoring band (4).
#   all      setup -> run -> tp, with the gate in front of each.
#
# `gate` deliberately needs nothing installed but `nvidia-smi`, so a bad node is abandoned before
# a single byte of torch has been downloaded onto it.
set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/oscaralateras/inference-from-the-metal-up}"
WORKDIR="${WORKDIR:-/workspace/inference-from-the-metal-up}"
WORLD="${WORLD:-4}"

say() { printf '\n\033[1m== %s ==\033[0m\n' "$*"; }
die() { printf '\n\033[1;31mFAILED: %s\033[0m\n' "$*" >&2; exit 1; }

gate() {
  say "topology gate (world=$WORLD)"
  command -v nvidia-smi >/dev/null || die "no nvidia-smi — this is not a GPU node."

  local count
  count=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
  nvidia-smi --query-gpu=index,name --format=csv,noheader
  [ "$count" -ge "$WORLD" ] || die "need $WORLD GPUs, found $count."

  echo
  nvidia-smi topo -m

  # Every pair among the first $WORLD GPUs must be NVLink. Parsed here rather than trusted by eye:
  # the whole failure mode is that a PCIe node looks fine until its numbers are published.
  local bad
  bad=$(nvidia-smi topo -m | awk -v n="$WORLD" '
    /^GPU[0-9]+/ {
      r = substr($1, 4) + 0
      if (r >= n) next
      for (c = 0; c < n; c++) {
        v = $(c + 2)
        if (c == r || v == "X") continue
        if (v !~ /^NV[0-9]+$/) printf "GPU%d-GPU%d=%s ", r, c, v
      }
    }')
  [ -z "$bad" ] || die "not NVLink between all pairs: $bad
    -> DESTROY this pod and re-rent. Do not run the benchmark on it."

  [ -z "${NCCL_P2P_DISABLE:-}" ] || die "NCCL_P2P_DISABLE=$NCCL_P2P_DISABLE is set — collectives
    would route around NVLink and report PCIe numbers under an NVLink label. Unset it."

  say "GATE PASSED — NVLink between all $WORLD GPUs, P2P enabled"
}

setup() {
  say "setup"
  command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"

  [ -d "$WORKDIR" ] || git clone "$REPO_URL" "$WORKDIR"
  cd "$WORKDIR"
  git pull --ff-only || true

  uv python install 3.12
  uv sync --frozen
  make probe
}

run()  { cd "$WORKDIR"; export PATH="$HOME/.local/bin:$PATH"; say "stages 1-2"; make t9; }
tp()   { cd "$WORKDIR"; export PATH="$HOME/.local/bin:$PATH"; say "stage 3";    make t9-tp; }

case "${1:-all}" in
  gate)  gate ;;
  setup) gate; setup ;;
  run)   run ;;
  tp)    tp ;;
  all)   gate; setup; run; tp
         say "DONE — commit results, then DESTROY THE POD. Idle time is the only real cost." ;;
  *)     die "unknown subcommand '${1}'. Use: gate | setup | run | tp | all" ;;
esac
