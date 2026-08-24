# 车网互动充电功率预测 — 7 模型合并交付包（V2）

本交付包整合**我方模型 + 教授模型**，共 7 个方法，用统一口径（预测 ground_truth 4 天，真 out-of-sample）在 28 case 数据上完成对比。

## 目录结构

```
charging_power_dataset_V2/
├── dataset/                 # 28 case 数据集（15min 颗粒度）
├── code/
│   ├── model_zoo.py         # 我方模型（LightGBM/PatchTST/TimesNet）
│   ├── professor_model_zoo.py  # 教授模型（Slot-Median/DEMMFL/GRU）
│   ├── preprocess.py        # 教授的数据预处理
│   ├── run_merged_benchmark.py  # 7 模型统一 benchmark
│   ├── convert_full.py      # 数据转换脚本
│   ├── requirements.txt / Dockerfile
├── docs/
│   └── 7模型对比报告.md      # 核心对比结果与关键发现
└── predictions/
    └── merged_benchmark.json  # 7 模型详细指标
```

## 7 个模型

| 模型 | 来源 | MAE(kW) | F1 |
|------|------|---------|-----|
| Slot-Median | 教授 | **27.68** | 0.662 |
| DEMMFL | 教授 | 29.47 | **0.715** |
| GRU | 教授 | 30.04 | 0.713 |
| All-Zero | 基线 | 31.54 | 0.000 |
| LightGBM | 我方 | 31.69 | 0.696 |
| PatchTST | 我方 | 33.74 | 0.700 |
| TimesNet | 我方 | 37.08 | 0.671 |

## 关键结论（务必先读）

1. **之前 rolling-origin 评估有数据泄漏**，深度模型表现被高估；改用真 out-of-sample 后，稀疏场景下**简单统计方法（Slot-Median）最稳**。
2. 教授方法在点预测（MAE）和非零步精度（MAPE_nz）上更优；我方"活跃枪数"特征解决了两阶段塌缩（F1 0→0.696）。
3. 详见 `docs/7模型对比报告.md`。

## 复现

```bash
cd code
python run_merged_benchmark.py
```

> 代码已改为**相对路径自适应**（不再依赖 macOS 绝对路径），可在任意 Linux 机器/容器内直接运行。

## 部署到 H20（离线）

**先看这一份总说明**：`部署总说明.md`（整合了兼容性分析 + 两个方案 + 推送分发 + 实测结果 + 文件位置）。

分篇细节：

- `docs/deploy/00_总览与H20兼容性分析.md` — 结论 + 环境对比 + 为什么可直接迁移
- `docs/deploy/01_方案一_离线Python环境部署.md` — 离线 wheel 包方式
- `docs/deploy/02_方案二_Docker镜像部署.md` — Docker 镜像方式（推荐）
- `docs/deploy/03_镜像推送与仓库分发.md` — 镜像大文件如何进 git/仓库
- `code/build_image.sh` — 一键构建+导出镜像（在 H20 上执行）
- `offline_install.sh` — H20 离线安装+运行（方案一）
