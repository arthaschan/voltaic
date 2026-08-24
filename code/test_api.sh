#!/usr/bin/env bash
# ============================================================
# 完整接口测试：GET /health + POST /predict
# 用法：bash test_api.sh [BASE_URL]
#   默认 BASE_URL=http://127.0.0.1:8000
#   示例：bash test_api.sh http://127.0.0.1:8000
#         bash test_api.sh http://192.168.1.10:8000
# 前置：服务已启动（离线包：python3 app.py；docker：docker run ...）
# ============================================================
set -euo pipefail

BASE_URL="${1:-http://127.0.0.1:8000}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CASE_FILE="${SCRIPT_DIR}/data_15min/data/case_001.csv"

echo "=============================================="
echo " 车网互动充电功率预测 —— 接口测试"
echo " BASE_URL = ${BASE_URL}"
echo " 输入数据 = case_001.csv 前 96 行（1 天 15min）"
echo "=============================================="
echo ""

echo "===== 1. 健康检查 GET /health ====="
HEALTH=$(curl -s -w "\nHTTP_STATUS:%{http_code}" "${BASE_URL}/health")
echo "${HEALTH}"
echo ""

echo "===== 2. 推理 POST /predict ====="
python3 - "${BASE_URL}" "${CASE_FILE}" <<'PY'
import sys, json, urllib.request, urllib.error
from collections import Counter

base_url, case_file = sys.argv[1], sys.argv[2]

# 读 case_001.csv 前 97 行（表头 + 96 行 = 1 天 15min 数据）作为输入
with open(case_file) as f:
    lines = f.readlines()[:97]
csv_text = ''.join(lines)

payload = json.dumps({"input": csv_text, "case_id": "001"}).encode()
req = urllib.request.Request(
    base_url + "/predict",
    data=payload,
    headers={"Content-Type": "application/json"},
)

try:
    resp = urllib.request.urlopen(req, timeout=180)
    result = json.loads(resp.read())
except urllib.error.HTTPError as e:
    print("HTTP 错误:", e.code)
    print(e.read().decode()[:500])
    sys.exit(1)
except Exception as e:
    print("请求失败:", type(e).__name__, str(e))
    sys.exit(1)

print("case_id     :", result.get("case_id"))
print("error_code  :", result.get("error_code"))
print("error_msg   :", result.get("error_msg"))

out = result.get("output") or ""
if out:
    rows = out.strip().split("\n")
    print("output 行数 :", len(rows), "(含表头)")
    print("output 表头 :", rows[0])
    print("---- 前 3 行数据 ----")
    for r in rows[1:4]:
        print("   ", r)
    print("---- 后 3 行数据 ----")
    for r in rows[-3:]:
        print("   ", r)
    # 统计各 forecast_horizon 的行数
    cnt = Counter(r.split(",")[0] for r in rows[1:] if r)
    print("---- 各 horizon 行数 ----")
    for h in sorted(cnt):
        print(f"     {h}: {cnt[h]} 行")
else:
    print("output 为空（可能模型未就绪，检查服务日志）")
PY
echo ""
echo "=============================================="
echo " 测试完成"
echo "=============================================="
