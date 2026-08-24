# 车网互动充电功率预测 — 多架构 Docker 镜像（兼容 H20 x86 + 昇腾 ARM）
#
# 构建（两个架构各出一个镜像）：
#   docker buildx build --platform linux/amd64 -t excharge-pred:amd64 .
#   docker buildx build --platform linux/arm64 -t excharge-pred:arm64 .
# 或合并成一个多架构 manifest：
#   docker buildx build --platform linux/amd64,linux/arm64 -t excharge-pred:latest --push .

FROM python:3.11-slim

WORKDIR /app

# lightgbm 运行时需要 libgomp1
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY . .

# 默认跑 v2（活动+分位数）实验；周五预测可直接覆盖 CMD 或调 predict()
CMD ["python", "scripts/run_experiment_v2.py"]
