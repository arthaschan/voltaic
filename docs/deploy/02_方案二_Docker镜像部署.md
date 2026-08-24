# 方案二：Docker 镜像部署

> 适用于：H20 已装 Docker（29.1.3，项目文档已确认），想要"一次打包、处处一致"。
> 核心思路：在**本机（H100，有网）** `docker build` 出镜像 → `docker save` 成 tar → 拷到 H20 → `docker load` → `docker run`。目标机全程离线。

---

## 一、交付物

```
excharge-pred.tar.gz        # 单个镜像包（约 500MB，含 Python 3.11 + 全部依赖 + 代码 + 数据）
```

镜像内部结构：

```
/app/
├── code/           # 6 个脚本 + requirements.txt
├── dataset/        # 28 case 数据 + 真值
└── predictions/    # 结果输出目录
```

- 基础镜像：`python:3.11-slim`（Debian）
- 依赖：`requirements.txt` + `torch` **CPU 版**（镜像内已预装 `libgomp1` 供 lightgbm 使用）
- 入口：`python code/run_merged_benchmark.py`

---

## 二、第一步：在本机（H100，有网）构建镜像

`code/Dockerfile` 已就绪，构建上下文 = **仓库根目录**（因为要同时 COPY `code/` 和 `dataset/`）。

```bash
cd /home/student/arthas/voltaic
docker build -f code/Dockerfile -t excharge-pred:latest .
```

> 说明：本会话里 Docker 守护进程需要 sudo（当前用户不在 docker 组）。执行时用 `sudo docker build ...`，
> 或先 `sudo usermod -aG docker $USER` 重新登录后免 sudo。构建命令本身与权限无关，结果一致。

构建耗时约 2~5 分钟（主要花在拉取 `python:3.11-slim` 和 `pip install torch`）。

---

## 三、第二步：导出镜像为离线包

```bash
cd /home/student/arthas/voltaic
docker save excharge-pred:latest | gzip > excharge-pred.tar.gz
ls -lh excharge-pred.tar.gz      # 预期 ~500MB
```

---

## 四、第三步：拷贝到 H20（离线）

把 `excharge-pred.tar.gz` 用离线介质拷到 H20。

---

## 五、第四步：在 H20 上导入并运行（全程离线）

```bash
# 1) 导入镜像（离线，无需联网）
gunzip -c excharge-pred.tar.gz | docker load
docker images | grep excharge-pred

# 2) 运行（挂载宿主机目录，把结果留在宿主机上便于取走）
mkdir -p ~/excharge-predictions
docker run --rm \
    -v ~/excharge-predictions:/app/predictions \
    excharge-pred:latest

# 3) 查看结果
cat ~/excharge-predictions/merged_benchmark.json
```

> `--rm` 跑完即删容器；`-v` 把容器内 `/app/predictions` 挂到宿主机，结果 json 持久化到宿主机。
> 不加 `-v` 也行，结果留在容器内（但 `--rm` 会随容器删除），所以**务必挂载**。

---

## 六、镜像里到底跑了什么

```
python code/run_merged_benchmark.py
```

- 读 `dataset/data/case_*.csv` + `dataset/ground_truth/case_*.csv`（28 case）
- 依次跑 7 模型：All-Zero / Slot-Median / LightGBM / DEMMFL / GRU / PatchTST / TimesNet
- 输出对比表到 stdout，并写 `predictions/merged_benchmark.json`

---

## 七、进阶用法

```bash
# 进容器交互调试（不自动跑 benchmark）
docker run --rm -it excharge-pred:latest bash
python code/run_merged_benchmark.py

# 挂载宿主机数据/代码，覆盖镜像内版本（现场改代码/换数据用）
docker run --rm -it \
    -v /path/to/code:/app/code \
    -v /path/to/dataset:/app/dataset \
    -v /path/to/predictions:/app/predictions \
    excharge-pred:latest bash
```

---

## 八、常见问题（FAQ）

1. **`docker load` 后 `docker run` 报驱动/CUDA 错误？**
   不会。镜像是 CPU-only，不挂 NVIDIA 运行时、不调 CUDA，无需 `--gpus`，与 H20 的驱动/CUDA 无关。

2. **镜像很大？** ~500MB，主要来自 `python:3.11-slim`（~150MB）+ torch CPU（~200MB）。已是最小可行组合；进一步可用 `python:3.11-alpine` 但 lightgbm 等需额外系统库，不推荐。

3. **H20 的 Docker 版本 29.1.3 能 load 本机 28.3.3 构建的镜像吗？**
   能。`docker save` 输出的是标准 OCI/Docker 镜像格式，向后兼容；29.x 加载 28.x 产物没有障碍（若极保守，可在 H20 上 `docker --version` 确认）。

4. **要固定依赖版本复现？**
   当前 `requirements.txt` 用 `>=` 范围约束。若评测要求"逐字节可复现"，可把 `wheels/` 里的精确版本固化进 `requirements.txt`（见方案一的 wheel 清单），再重建镜像。

5. **想在 H20 上直接用 GPU？** 本项目不需要，见总览 §四。若坚持，需改用 CUDA 版 torch 镜像并在 `docker run` 加 `--gpus all`，同时确认 CUDA 12.9 兼容——本次不推荐、不覆盖。
