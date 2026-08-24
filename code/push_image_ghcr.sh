#!/usr/bin/env bash
# 把镜像推送到 GitHub Container Registry (ghcr.io)。
# 前置：已构建镜像 excharge-pred:latest（见 build_image.sh），且有 GitHub PAT（scope: write:packages）。
# 用法：
#   export GHCR_PAT="你的个人访问令牌"     # 或运行时输入
#   bash code/push_image_ghcr.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

IMAGE="excharge-pred:latest"
OWNER="arthaschan"          # GitHub 用户名，按需改
REPO="voltaic"              # ghcr 镜像仓库名，与 git 仓库同名
TAG="${TAG:-latest}"
TARGET="ghcr.io/${OWNER}/${REPO}:${TAG}"

# 选择 docker 命令（无权限则尝试 sudo）
DOCKER="docker"
if ! docker info >/dev/null 2>&1; then
  if sudo -n true 2>/dev/null; then DOCKER="sudo docker"; else
    echo "[!] 无法访问 Docker 守护进程。请先 sudo usermod -aG docker \$USER 并重新登录，或 sudo 运行本脚本。"
    exit 1
  fi
fi

# 登录（优先用环境变量 GHCR_PAT，否则交互输入）
if [ -n "${GHCR_PAT:-}" ]; then
  echo "$GHCR_PAT" | $DOCKER login ghcr.io -u "$OWNER" --password-stdin
else
  $DOCKER login ghcr.io -u "$OWNER"
fi

echo "==> 打 tag 并推送 $IMAGE -> $TARGET"
$DOCKER tag "$IMAGE" "$TARGET"
$DOCKER push "$TARGET"

echo "==> 完成。目标机拉取："
echo "    docker pull $TARGET"
