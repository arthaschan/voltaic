#!/usr/bin/env bash
# 离线安装与运行脚本（方案三：包内自带 Python 3.11 运行时，不依赖目标机系统 Python）。
# 前置：已解压 excharge-offline.tar.gz，脚本位于解压后的顶层目录。
# 用法：bash offline_install.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

# 1) 选择 Python：优先用包内自带 runtime/（自包含，跨机器版本无关），否则回退系统 python3
if [ -x "$ROOT/runtime/bin/python3.11" ]; then
  PY="$ROOT/runtime/bin/python3.11"
  echo "==> 使用包内自带 Python 3.11: $PY"
else
  PY="python3"
  echo "==> 未找到自带 runtime，回退系统 python3"
fi

# 2) 建 venv（若已存在则复用）
if [ ! -d ".venv" ]; then
  echo "==> 创建虚拟环境（基于 $("$PY" --version 2>&1)）"
  "$PY" -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate

# 3) 离线安装依赖（--no-index 强制只从 wheels/ 取，绝不联网）
echo "==> 离线安装基础依赖"
python -m pip install --no-index --find-links wheels/ -r code/requirements.txt
echo "==> 离线安装 torch（CPU 版）"
python -m pip install --no-index --find-links wheels/ torch

# 4) 验证
echo "==> 验证依赖"
python -c "import torch, lightgbm, sklearn, pandas; print('  torch', torch.__version__, '| lightgbm', lightgbm.__version__, '| pandas', pandas.__version__)"

# 5) 运行 benchmark
echo "==> 运行 7 模型 benchmark"
cd "$ROOT/code"
python run_merged_benchmark.py

echo "==> 完成，结果见 $ROOT/predictions/merged_benchmark.json"
