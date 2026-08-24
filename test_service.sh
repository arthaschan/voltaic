#!/usr/bin/env bash
# 车网互动充电功率预测 —— /predict 服务完整接口测试（含正常 + 错误用例）
# 用法：bash test_service.sh [BASE_URL]
#   默认 BASE_URL=http://127.0.0.1:8000
# 前置：服务已启动（bash start_service.sh）
set -uo pipefail

BASE_URL="${1:-http://127.0.0.1:8000}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CASE_FILE="${SCRIPT_DIR}/dataset/data/case_001.csv"

PASS=0; FAIL=0
ok(){ echo "  ✓ $1"; PASS=$((PASS+1)); }
bad(){ echo "  ✗ $1"; FAIL=$((FAIL+1)); }

echo "=============================================="
echo " /predict 服务完整接口测试"
echo " BASE_URL = $BASE_URL"
echo "=============================================="

echo ""
echo "===== 用例1：健康检查 GET /health ====="
R=$(curl -s -w "\n%{http_code}" "$BASE_URL/health")
BODY=$(echo "$R" | head -n -1); CODE=$(echo "$R" | tail -1)
echo "$BODY"
[ "$CODE" = "200" ] && echo "$BODY" | grep -q '"status":"healthy"' \
  && ok "health 返回 200 且 status=healthy" || bad "health 异常：$CODE $BODY"

echo ""
echo "===== 用例2：正常推理 POST /predict（case_001 前 96 行）====="
python3 - "$BASE_URL" "$CASE_FILE" <<'PY'
import sys, json, urllib.request
base, case_file = sys.argv[1], sys.argv[2]
with open(case_file) as f:
    csv_text = ''.join(f.readlines()[:97])   # 表头 + 96 行 = 1 天
payload = json.dumps({"input": csv_text, "case_id": "001"}).encode()
req = urllib.request.Request(base + "/predict", data=payload, headers={"Content-Type": "application/json"})
resp = json.loads(urllib.request.urlopen(req, timeout=180).read())
rows = (resp.get("output") or "").strip().split("\n")
assert resp.get("case_id") == "001", "case_id 未原样返回"
assert resp.get("error_code") is None, f"error_code={resp.get('error_code')}"
assert resp.get("error_msg") is None, f"error_msg={resp.get('error_msg')}"
assert len(rows) == 505, f"应 505 行(表头+504)，实际 {len(rows)}"
assert rows[0] == "forecast_horizon,step_index,forecast_timestamp,power_kw", f"表头错误: {rows[0]}"
from collections import Counter
cnt = Counter(r.split(",")[0] for r in rows[1:] if r)
assert cnt == {"2h":8, "4h":16, "1d":96, "4d":384}, f"各 horizon 行数错误: {dict(cnt)}"
print(f"  output {len(rows)} 行，各 horizon: {dict(cnt)}")
print("  PASS")
PY
[ $? -eq 0 ] && ok "用例2 通过（504 行输出，格式正确）" || bad "用例2 失败"

echo ""
echo "===== 用例3：缺 case_id → INPUT_MISSING_FIELD ====="
python3 - "$BASE_URL" <<'PY'
import sys, json, urllib.request
base = sys.argv[1]
payload = json.dumps({"input": "a,b\n1,2"}).encode()
req = urllib.request.Request(base+"/predict", data=payload, headers={"Content-Type":"application/json"})
r = json.loads(urllib.request.urlopen(req, timeout=30).read())
print(f"  error_code={r.get('error_code')}")
assert r.get("error_code") == "INPUT_MISSING_FIELD", f"实际 {r.get('error_code')}"
print("  PASS")
PY
[ $? -eq 0 ] && ok "用例3 通过（缺 case_id 报 INPUT_MISSING_FIELD）" || bad "用例3 失败"

echo ""
echo "===== 用例4：非法 JSON → INPUT_FORMAT_INVALID ====="
python3 - "$BASE_URL" <<'PY'
import sys, json, urllib.request
base = sys.argv[1]
req = urllib.request.Request(base+"/predict", data=b"{invalid json", headers={"Content-Type":"application/json"})
r = json.loads(urllib.request.urlopen(req, timeout=30).read())
print(f"  error_code={r.get('error_code')}")
assert r.get("error_code") == "INPUT_FORMAT_INVALID", f"实际 {r.get('error_code')}"
print("  PASS")
PY
[ $? -eq 0 ] && ok "用例4 通过（非法 JSON 报 INPUT_FORMAT_INVALID）" || bad "用例4 失败"

echo ""
echo "===== 用例5：空 input CSV → INPUT_FORMAT_INVALID ====="
python3 - "$BASE_URL" <<'PY'
import sys, json, urllib.request
base = sys.argv[1]
payload = json.dumps({"input": "   ", "case_id": "x"}).encode()
req = urllib.request.Request(base+"/predict", data=payload, headers={"Content-Type":"application/json"})
r = json.loads(urllib.request.urlopen(req, timeout=30).read())
print(f"  error_code={r.get('error_code')}")
assert r.get("error_code") == "INPUT_FORMAT_INVALID"
print("  PASS")
PY
[ $? -eq 0 ] && ok "用例5 通过（空 CSV 报 INPUT_FORMAT_INVALID）" || bad "用例5 失败"

echo ""
echo "=============================================="
echo " 结果：$PASS 通过，$FAIL 失败"
echo "=============================================="
[ "$FAIL" -eq 0 ]
