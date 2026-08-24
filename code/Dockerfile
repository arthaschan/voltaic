# 车网互动充电功率预测 —— /predict 推理服务镜像（离线 wheel，H20/H100 x86_64 通用）
#
# 构建上下文 = 仓库根目录（voltaic/）：
#   cd /home/student/arthas/voltaic
#   docker build -f code/Dockerfile -t excharge-pred:latest .
#
# 接口（与评测平台规范一致）：
#   GET  /health   健康检查
#   POST /predict  推理：{input: "<28字段CSV文本>", case_id} → {output: "<4字段CSV文本504行>", case_id, error_code, error_msg}
# 模型由环境变量 MODEL 决定（默认 demmfl），ALL_MODELS=1 时返回 7 模型对照。
#
# 运行：
#   docker run -d -p 8000:8000 excharge-pred:latest
FROM python:3.11-slim

# lightgbm 运行时需要 libgomp1（国内构建用阿里云源，避免 deb.debian.org 被墙卡死）
RUN sed -i "s|deb.debian.org|mirrors.aliyun.com|g" /etc/apt/sources.list.d/debian.sources 2>/dev/null || true \
    && apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 离线安装依赖（wheels/ 需包含 flask/matplotlib 等；--no-index 不联网）
COPY code/requirements.txt ./code/requirements.txt
COPY wheels/ /deps/
RUN pip install --no-index --find-links=/deps -r code/requirements.txt \
    && pip install --no-index --find-links=/deps torch

# 代码 + 数据 + 预训练模型
COPY code/ ./code/
COPY dataset/ ./dataset/

EXPOSE 8000
ENV CHARGING_DATA_DIR=/app/dataset
ENV PORT=8000
ENV HOST=0.0.0.0

CMD ["python3", "code/app.py"]
