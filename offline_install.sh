#!/usr/bin/env bash
# 离线安装（服务版）：包内自带 Python 3.11 运行时 + 全部依赖 + /predict 推理服务。
# 前置：已解压 excharge-offline.tar.gz，脚本位于解压后的顶层目录。
# 用法：bash offline_install.sh     # 一次性：建 venv → 离线装依赖 → 验证
# 之后用 bash start_service.sh 启动服务，bash test_api.sh 测试。
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
  echo "==> 创建虚拟环境（$("$PY" --version 2>&1)）"
  "$PY" -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate

# 3) 离线安装依赖（--no-index 强制只从 wheels/ 取，绝不联网）
echo "==> 离线安装全部依赖（含 flask/matplotlib/torch）"
python -m pip install --no-index --find-links wheels/ -r code/requirements.txt
python -m pip install --no-index --find-links wheels/ torch

# 4) 验证依赖
echo "==> 验证依赖"
python -c "import torch, lightgbm, sklearn, pandas, flask, matplotlib; print('  torch', torch.__version__, '| lgb', lightgbm.__version__, '| flask', flask.__version__)"

echo ""
echo "==> 安装完成。下一步："
echo "    启动服务：  bash start_service.sh"
echo "    测试接口：  bash test_api.sh http://127.0.0.1:8000"
