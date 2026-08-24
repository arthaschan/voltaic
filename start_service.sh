#!/usr/bin/env bash
# 启动 /predict 推理服务（Flask，监听 0.0.0.0:8000）。
# 前置：已运行 bash offline_install.sh（建好 .venv 并装依赖）。
# 用法：bash start_service.sh
# 可选环境变量：
#   MODEL=demmfl|allzero|slotmedian|lightgbm|gru|patchtst|timesnet   （默认 demmfl）
#   ALL_MODELS=1   本机 7 模型对照模式
#   PORT=8000
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT/code"

if [ ! -x "$ROOT/.venv/bin/python" ]; then
  echo "[!] 未找到 .venv，请先运行 bash offline_install.sh"
  exit 1
fi
# shellcheck disable=SC1091
source "$ROOT/.venv/bin/activate"

export CHARGING_DATA_DIR="${CHARGING_DATA_DIR:-$ROOT/dataset}"
export PORT="${PORT:-8000}"
export HOST="${HOST:-0.0.0.0}"
export MODEL="${MODEL:-demmfl}"

echo "==> 启动服务：模型=$MODEL，端口=$PORT，数据目录=$CHARGING_DATA_DIR"
python app.py
