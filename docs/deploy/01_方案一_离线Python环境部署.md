# 方案一：离线 Python 环境部署

> 适用于：H20 上已有 Python 3.11 + pip，想直接在宿主机 Python 里跑（不依赖 Docker）。
> 核心思路：在**本机（有网）**把全部依赖下载成 `.whl` 离线包，连同代码+数据一起拷到 H20，在 H20 上**离线安装**。

---

## 一、交付物清单（打包成一个压缩包）

```
excharge-offline/
├── code/                  # 6 个脚本（已修复 macOS 路径）
│   ├── run_merged_benchmark.py   # 入口
│   ├── model_zoo.py              # 我方模型（LightGBM/PatchTST/TimesNet）
│   ├── professor_model_zoo.py    # 教授模型（Slot-Median/DEMMFL/GRU）
│   ├── preprocess.py             # 公共工具（运行时非必需，随包附带）
│   ├── convert_full.py           # 数据集生成（运行时非必需，随包附带）
│   └── requirements.txt
├── dataset/               # 28 case 数据 + 真值（17MB）
│   ├── data/case_xxx.csv
│   ├── ground_truth/case_xxx.csv
│   ├── manifest.json
│   └── README.md
├── predictions/           # 结果输出目录
├── wheels/                # 离线依赖（cp311, x86_64, 共 268MB / 22 个 whl）
└── offline_install.sh     # 目标机一键安装+运行脚本
```

---

## 二、第一步：在本机（H100，有网）生成离线包

本机已完成以下步骤（产物已就绪），这里给出完整命令供复现：

```bash
cd /home/student/arthas/voltaic

# 1) 建 Python 3.11 虚拟环境（与 H20 的 3.11.15 对齐）
uv venv .venv --python 3.11

# 2) 下载全部依赖到 wheels/（pip 需先装进 venv）
.venv/bin/python -m pip install pip
.venv/bin/python -m pip download -r code/requirements.txt -d wheels/
.venv/bin/python -m pip download torch \
    --index-url https://download.pytorch.org/whl/cpu -d wheels/

# 3) 打包
tar czf excharge-offline.tar.gz \
    code dataset predictions wheels offline_install.sh
```

> `wheels/` 已在 `code/.dockerignore` 与打包脚本中，约 268MB（大头是 torch CPU wheel ~200MB）。
> 所有 wheel 均为 `cp311-*-manylinux_*_x86_64`，与 H20（x86_64 + Python 3.11.15）严格匹配。

**重要**：wheel 与 Python 版本、CPU 架构强绑定。
- 若 H20 的 Python 是 3.10 或其他小版本，需在本机用对应版本重新 `pip download`。
- 若是昇腾 aarch64 机器，需下载 `manylinux_aarch64` 版本（另一套）。

---

## 三、第二步：拷贝到 H20（离线）

用 U 盘 / 内网 / 加密盘等离线介质把 `excharge-offline.tar.gz` 拷到 H20，例如：

```bash
# 在 H20 上解压
mkdir -p ~/excharge && tar xzf excharge-offline.tar.gz -C ~/excharge
cd ~/excharge
```

---

## 四、第三步：在 H20 上离线安装依赖

前提：H20 已装 Python 3.11 + pip（H20 系统自带 Python 3.11.15）。

```bash
cd ~/excharge

# 建议用 venv 隔离，避免污染系统 Python
python3 -m venv .venv
source .venv/bin/activate

# 离线安装（--no-index 强制只从 wheels/ 取，绝不联网）
pip install --no-index --find-links wheels/ -r code/requirements.txt
pip install --no-index --find-links wheels/ torch
```

验证依赖：

```bash
python -c "import torch, lightgbm, sklearn, pandas; print('torch', torch.__version__)"
# 期望输出 torch 2.x.x+cpu（+cpu 后缀说明是 CPU 版，无需 GPU）
```

---

## 五、第四步：运行

```bash
cd ~/excharge/code
source ../.venv/bin/activate
python run_merged_benchmark.py
```

运行完成后，结果在 `~/excharge/predictions/merged_benchmark.json`，控制台打印 7 模型对比表。

---

## 六、一键脚本（可选）

在目标机上可省去手动步骤，直接执行随包附带的脚本：

```bash
cd ~/excharge && bash offline_install.sh
```

---

## 七、常见问题（FAQ）

1. **报 `ModuleNotFoundError: No module named pip`**
   → 用 `python3 -m ensurepip --upgrade` 或从离线包单独装 pip wheel。

2. **报 `is not a supported wheel on this platform`**
   → 说明 wheel 架构/版本与目标机不匹配（例如 H20 是 aarch64，或 Python 版本不同）。回到本机用匹配版本重新 `pip download`。

3. **想用 GPU？** 本项目不需要，CPU 版足够（见总览 §四）。若要 GPU，需改下载 `torch` CUDA 版 wheel 并确认 H20 的 CUDA 12.9 兼容性，本文档不推荐、不覆盖。

4. **lightgbm 报 `libgomp.so.1` 缺失**
   → `sudo apt-get install libgomp1`（H20 是 Ubuntu，应有该库；Docker 方案里已在镜像内预装）。
