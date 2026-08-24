#!/usr/bin/env bash
# 离线安装与运行脚本（方案一：在 H20 离线机上执行）。
# 前置：已在 H20 解压 excharge-offline.tar.gz，脚本位于解压后的顶层目录。
# 用法：bash offline_install.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

PY="python3"

# 1) 建 venv（若已存在则复用）
if [ ! -d ".venv" ]; then
  echo "==> 创建虚拟环境"
  "$PY" -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate

# 2) 离线安装依赖（--no-index 强制只从 wheels/ 取，绝不联网）
echo "==> 离线安装基础依赖"
python -m pip install --no-index --find-links wheels/ -r code/requirements.txt
echo "==> 离线安装 torch（CPU 版）"
python -m pip install --no-index --find-links wheels/ torch

# 3) 验证
echo "==> 验证依赖"
python -c "import torch, lightgbm, sklearn, pandas; print('  torch', torch.__version__, '| lightgbm', lightgbm.__version__)"

# 4) 运行 benchmark
echo "==> 运行 7 模型 benchmark"
cd "$ROOT/code"
python run_merged_benchmark.py

echo "==> 完成，结果见 $ROOT/predictions/merged_benchmark.json"
