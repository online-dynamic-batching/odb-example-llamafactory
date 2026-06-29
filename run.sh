#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python}"

DATA_REPO_DIR="${ODB_MM_MIX_DATA_REPO:-$ROOT_DIR/.deps/odb-mm-mix-example}"
LLAMAFACTORY_ROOT="${LLAMAFACTORY_ROOT:-$ROOT_DIR/.deps/LLaMA-Factory-odb}"
DATA_DIR="${ODB_MM_MIX_DATA_DIR:-$ROOT_DIR/data/mm-mix-tmdb}"
RUN_DIR="${ODB_LF_RUN_DIR:-$ROOT_DIR/data/llamafactory-mm-mix}"
TRAIN_ROOT="${ODB_MM_MIX_TRAIN_ROOT:-$ROOT_DIR/outputs/llamafactory-mm-mix}"

usage() {
  cat <<'USAGE'
Usage: ./run.sh <command>

Commands:
  install          Install Python dependencies for this example
  setup-lf         Prepare the tested LLaMA-Factory checkout
  check            Check that the LLaMA-Factory checkout is ODB-ready
  data             Build the public MM-Mix TMDB data
  prepare          Create runnable LLaMA-Factory configs for local paths
  train-odb        Train with Online Dynamic Batching
  eval-odb         Run validation loss and MMMU-MC for ODB
  train-standard   Train the fixed-batch baseline
  eval-standard    Run validation loss and MMMU-MC for Standard
  all-odb          install + setup-lf + data + prepare + train-odb + eval-odb

Required environment:
  ODB_MM_MIX_MODEL=/path/to/Qwen3-VL-2B-Instruct

Optional environment:
  LLAMAFACTORY_ROOT=/custom/path/to/LLaMA-Factory
USAGE
}

need_env() {
  local name="$1"
  if [[ -z "${!name:-}" ]]; then
    echo "Missing required environment variable: $name" >&2
    exit 2
  fi
}

setup_dist_env() {
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
  if [[ -z "${NPROC_PER_NODE:-}" ]]; then
    local devices="$CUDA_VISIBLE_DEVICES"
    local count=1
    if [[ -n "$devices" ]]; then
      count="$("$PYTHON" - <<'PY'
import os
print(len([x for x in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if x.strip()]) or 1)
PY
)"
    fi
    export NPROC_PER_NODE="$count"
  fi
  export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
}

cmd_install() {
  "$PYTHON" -m pip install -r "$ROOT_DIR/requirements.txt"
}

cmd_check() {
  "$PYTHON" "$ROOT_DIR/scripts/check_llamafactory.py" \
    --llamafactory-root "$LLAMAFACTORY_ROOT"
}

cmd_setup_lf() {
  "$PYTHON" "$ROOT_DIR/scripts/setup_llamafactory.py" \
    --target "$LLAMAFACTORY_ROOT" \
    --install
}

cmd_data() {
  mkdir -p "$(dirname "$DATA_REPO_DIR")" "$(dirname "$DATA_DIR")"
  if [[ ! -d "$DATA_REPO_DIR/.git" ]]; then
    git clone https://github.com/online-dynamic-batching/odb-mm-mix-example.git "$DATA_REPO_DIR"
  fi
  "$PYTHON" -m pip install -e "$DATA_REPO_DIR"
  "$PYTHON" "$DATA_REPO_DIR/scripts/build_public_mm_mix.py" \
    --output "$DATA_DIR" \
    --overwrite
}

cmd_prepare() {
  need_env ODB_MM_MIX_MODEL
  cmd_check
  "$PYTHON" "$ROOT_DIR/scripts/prepare_lf_training.py" \
    --data "$DATA_DIR" \
    --output "$RUN_DIR" \
    --llamafactory-src "$LLAMAFACTORY_ROOT" \
    --llamafactory-root "$LLAMAFACTORY_ROOT" \
    --model "$ODB_MM_MIX_MODEL" \
    --image-max-pixels "${ODB_MM_MIX_IMAGE_MAX_PIXELS:-589824}" \
    --train-output "$TRAIN_ROOT" \
    --overwrite
}

cmd_train_odb() {
  setup_dist_env
  export MASTER_PORT="${MASTER_PORT:-29500}"
  "$PYTHON" "$RUN_DIR/wrappers/run_odb.py"
}

cmd_eval_odb() {
  "$PYTHON" "$ROOT_DIR/scripts/run_lf_eval.py" \
    --target odb \
    --lf-root "$LLAMAFACTORY_ROOT" \
    --run-dir "$RUN_DIR" \
    --train-root "$TRAIN_ROOT" \
    "$@"
}

cmd_train_standard() {
  setup_dist_env
  export MASTER_PORT="${MASTER_PORT:-29501}"
  "$PYTHON" "$RUN_DIR/wrappers/run_standard.py"
}

cmd_eval_standard() {
  "$PYTHON" "$ROOT_DIR/scripts/run_lf_eval.py" \
    --target standard \
    --lf-root "$LLAMAFACTORY_ROOT" \
    --run-dir "$RUN_DIR" \
    --train-root "$TRAIN_ROOT" \
    "$@"
}

cmd="${1:-}"
if [[ -z "$cmd" || "$cmd" == "-h" || "$cmd" == "--help" ]]; then
  usage
  exit 0
fi
shift || true

case "$cmd" in
  install) cmd_install "$@" ;;
  setup-lf | setup-llamafactory) cmd_setup_lf "$@" ;;
  check) cmd_check "$@" ;;
  data) cmd_data "$@" ;;
  prepare) cmd_prepare "$@" ;;
  train-odb) cmd_train_odb "$@" ;;
  eval-odb) cmd_eval_odb "$@" ;;
  train-standard) cmd_train_standard "$@" ;;
  eval-standard) cmd_eval_standard "$@" ;;
  all-odb)
    cmd_install
    cmd_setup_lf
    cmd_check
    cmd_data
    cmd_prepare
    cmd_train_odb
    cmd_eval_odb
    ;;
  *)
    usage
    echo "Unknown command: $cmd" >&2
    exit 2
    ;;
esac
