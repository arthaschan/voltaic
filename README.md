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
