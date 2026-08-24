# 车网互动充电功率预测 — 7 模型 benchmark 镜像（CPU-only，H20/H100/昇腾 通用）
#
# 构建上下文 = 仓库根目录（voltaic/），因为镜像需要同时包含 code/ 和 dataset/：
#   cd /home/student/arthas/voltaic
#   docker build -f code/Dockerfile -t excharge-pred:latest .
#
# 镜像自包含（代码 + 28 case 数据 + 全部依赖），目标机离线也能 docker load 后直接运行，
# 无需联网、无需 NVIDIA 驱动/CUDA 运行时（纯 CPU torch）。
#
# 导出/导入（离线迁移）：
#   docker save excharge-pred:latest | gzip > excharge-pred.tar.gz
#   # 目标机（H20，离线）：
#   gunzip -c excharge-pred.tar.gz | docker load
#   docker run --rm -v "$PWD/predictions:/app/predictions" excharge-pred:latest

FROM python:3.11-slim

WORKDIR /app

# lightgbm 运行时需要 libgomp1
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# 先装依赖（利用 Docker 层缓存：requirements 不变则不重装）
COPY code/requirements.txt ./code/requirements.txt
RUN pip install --no-cache-dir -r code/requirements.txt \
    && pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# 拷贝代码、数据、预测输出目录
COPY code/ ./code/
COPY dataset/ ./dataset/
COPY predictions/ ./predictions/

# 默认跑 7 模型统一 benchmark（输出到 /app/predictions/merged_benchmark.json）
CMD ["python", "code/run_merged_benchmark.py"]
