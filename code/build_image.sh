#!/usr/bin/env bash
# 在本机（H100，有网）构建并导出 Docker 镜像为离线 tar 包。
# 用法：bash code/build_image.sh
# 说明：Docker 需要权限——当前用户若不在 docker 组，请用 sudo 或先
#       sudo usermod -aG docker $USER 并重新登录。脚本检测到无权限会自动重试 sudo。
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="excharge-pred:latest"
TARBALL="$REPO_ROOT/excharge-pred.tar.gz"

cd "$REPO_ROOT"

# 选择 docker 命令（无权限则尝试 sudo）
DOCKER="docker"
if ! docker info >/dev/null 2>&1; then
  if sudo -n true 2>/dev/null; then
    DOCKER="sudo docker"
  else
    echo "[!] 当前用户无法访问 Docker 守护进程。请任选其一："
    echo "    1) sudo usermod -aG docker \$USER 后重新登录，再运行本脚本；"
    echo "    2) 直接执行：sudo bash code/build_image.sh"
    exit 1
  fi
fi

echo "==> 构建镜像（上下文 = 仓库根目录）"
$DOCKER build -f code/Dockerfile -t "$IMAGE" .

echo "==> 导出镜像为离线包"
$DOCKER save "$IMAGE" | gzip > "$TARBALL"

echo "==> 完成"
ls -lh "$TARBALL"
echo "下一步：把 $TARBALL 拷到 H20，执行："
echo "  gunzip -c excharge-pred.tar.gz | docker load"
echo "  mkdir -p ~/excharge-predictions && docker run --rm -v ~/excharge-predictions:/app/predictions excharge-pred:latest"
