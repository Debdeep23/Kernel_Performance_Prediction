#!/usr/bin/env bash
set -euo pipefail

KERNEL="$1"
ARGSTR="$2"
REGS="${3:-0}"
SHMEM="${4:-0}"
TRIALS="${5:-10}"
TAG="${6:-2080ti}"

mkdir -p data

DEV_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | sed 's/^ *//;s/ *$//')"

OUT="data/trials_${KERNEL}__${TAG}.csv"
if [ ! -s "$OUT" ]; then
  echo "kernel,args,regs,shmem,device_name,block_x,block_y,block_z,grid_x,grid_y,grid_z,warmup,reps,time_ms,rows,cols,iters,block,grid_blocks" > "$OUT"
fi

parse_token () {
  local name="$1" ; local line="$2"
  echo "$line" | sed -nE "s/.*\b${name}=([^ ,)]+).*/\1/p" | head -1 || true
}

parse_tuple () {
  local name="$1"; local line="$2"
  local t="$(echo "$line" | sed -nE "s/.*\b${name}=\(([^)]*)\).*/\1/p" | head -1)"
  if [ -n "$t" ]; then
    local x="$(echo "$t" | cut -d, -f1)"
    local y="$(echo "$t" | cut -d, -f2)"
    local z="$(echo "$t" | cut -d, -f3)"
    [ -z "$y" ] && y=1
    [ -z "$z" ] && z=1
    echo "$x,$y,$z"
  else
    echo ",,"
  fi
}

ceil_div () {
  python3 - "$1" "$2" <<'PY'
import sys,math
a=int(sys.argv[1]); b=int(sys.argv[2])
print((a + b - 1)//b if b>0 else 0)
PY
}

for i in $(seq 1 "$TRIALS"); do
  LINE="$(bin/runner --kernel "$KERNEL" $ARGSTR || true)"

  time_ms="$(parse_token time_ms "$LINE")"
  warmup="$(parse_token warmup "$LINE")"
  reps="$(parse_token reps "$LINE")"
  iters="$(parse_token iters "$LINE")"
  block_flag="$(parse_token block "$LINE")"

  tuple_block="$(parse_tuple block "$LINE")"
  tuple_grid="$(parse_tuple grid "$LINE")"
  block_x="$(echo "$tuple_block" | cut -d, -f1)"
  block_y="$(echo "$tuple_block" | cut -d, -f2)"
  block_z="$(echo "$tuple_block" | cut -d, -f3)"
  grid_x="$( echo "$tuple_grid"  | cut -d, -f1)"
  grid_y="$( echo "$tuple_grid"  | cut -d, -f2)"
  grid_z="$( echo "$tuple_grid"  | cut -d, -f3)"

  [ -z "${warmup:-}" ] && warmup="$(echo "$ARGSTR" | sed -nE 's/.*--warmup +([0-9]+).*/\1/p')"
  [ -z "${reps:-}"   ] && reps="$(  echo "$ARGSTR" | sed -nE 's/.*--reps +([0-9]+).*/\1/p')"
  [ -z "${iters:-}"  ] && iters="$( echo "$ARGSTR" | sed -nE 's/.*--iters +([0-9]+).*/\1/p')"

  rows="$(echo "$ARGSTR" | sed -nE 's/.*--rows +([0-9]+).*/\1/p')"
  cols="$(echo "$ARGSTR" | sed -nE 's/.*--cols +([0-9]+).*/\1/p')"
  N="$(  echo "$ARGSTR" | sed -nE 's/.*--N +([0-9]+).*/\1/p')"
  H="$(  echo "$ARGSTR" | sed -nE 's/.*--H +([0-9]+).*/\1/p')"
  W="$(  echo "$ARGSTR" | sed -nE 's/.*--W +([0-9]+).*/\1/p')"
  matN="$(echo "$ARGSTR" | sed -nE 's/.*--matN +([0-9]+).*/\1/p')"

  if [ -n "$rows" ] || [ -n "$cols" ]; then
    :
  elif [ -n "$H" ] || [ -n "$W" ]; then
    rows="${H:-$rows}"; cols="${W:-$cols}"
  elif [ -n "$matN" ]; then
    rows="$matN"; cols="$matN"
  elif [ -n "$N" ]; then
    rows="$N"; cols="1"
  fi

  if [ -z "${block_x:-}" ] && [ -n "${block_flag:-}" ]; then
    block_x="$block_flag"; block_y="1"; block_z="1"
  fi
  if [ -z "${grid_x:-}" ] && [ -n "${rows:-}" ] && [ -n "${block_x:-}" ]; then
    grid_x="$(ceil_div "${rows}" "${block_x}")"; grid_y="${cols:-1}"; [ -z "$grid_y" ] && grid_y=1; grid_z="1"
  fi

  for v in block_x block_y block_z grid_x grid_y grid_z rows cols iters warmup reps; do
    eval "val=\${$v:-}"
    [ -z "$val" ] && eval "$v=0"
  done
  [ -z "${time_ms:-}" ] && time_ms="0.0"
  block="${block_flag:-$block_x}"
  grid_blocks=$(( ${grid_x:-0} * ${grid_y:-0} * ${grid_z:-0} ))

  CLEAN_ARGS="$(echo "$ARGSTR" | tr -s ' ' ' ' | tr -d ',')"

  printf "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%.6f,%s,%s,%s,%s,%s\n" \
    "$KERNEL" "$CLEAN_ARGS" "$REGS" "$SHMEM" "$DEV_NAME" \
    "$block_x" "$block_y" "$block_z" \
    "$grid_x"  "$grid_y"  "$grid_z" \
    "$warmup" "$reps" "$time_ms" \
    "$rows" "$cols" "$iters" "$block" "$grid_blocks" >> "$OUT"
done

echo "[OK] wrote $OUT"

