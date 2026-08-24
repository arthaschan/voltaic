#!/usr/bin/env python3
"""preprocess.py — 数据预处理与公共工具函数

功能:
  - 将原始充电枪级采样记录聚合为 15min 网格级功率序列
  - 生成预测期日历信息
  - 统一评估指标计算 (MAE, RMSE, WAPE, MAPE, F1)
"""
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import f1_score

PRED_START = pd.Timestamp("2023-10-27 00:00:00")
PRED_END = pd.Timestamp("2023-10-30 23:45:00")
HORIZONS = {"2h": 8, "4h": 16, "1d": 96, "4d": 384}


def aggregate_to_grid(csv_path):
    """将充电枪级 CSV 聚合为 15min 网格级功率序列。

    Returns:
        gf: DataFrame, 列含 timestamp, power_kw, is_workday, is_major_holiday, temperature_c, rainfall_mm
        df_raw: 原始 DataFrame
    """
    df = pd.read_csv(csv_path, parse_dates=["report_time"])
    df["date"] = pd.to_datetime(df["date"])
    df["ts_15min"] = df["report_time"].dt.floor("15min")

    gun_avg = df.groupby(["ts_15min", "connector_id"])["power_w"].mean().reset_index()
    grid_power = gun_avg.groupby("ts_15min")["power_w"].sum().reset_index()
    grid_power.columns = ["timestamp", "power_w"]
    grid_power["power_kw"] = grid_power["power_w"] / 1000.0

    full_range = pd.date_range(
        start=grid_power["timestamp"].min(),
        end="2023-10-26 23:45:00", freq="15min"
    )
    gf = pd.DataFrame({"timestamp": full_range})
    gf = gf.merge(grid_power[["timestamp", "power_kw"]], on="timestamp", how="left")
    gf["power_kw"] = gf["power_kw"].fillna(0)

    date_attrs = df.groupby("date").agg({
        "is_workday": "first", "is_major_holiday": "first",
        "temperature_c": "mean", "rainfall_mm": "mean",
    }).reset_index()
    date_attrs.columns = ["date_dt", "is_workday", "is_major_holiday", "temperature_c", "rainfall_mm"]
    gf["date_dt"] = gf["timestamp"].dt.normalize()
    gf = gf.merge(date_attrs, on="date_dt", how="left")
    gf["is_workday"] = gf["is_workday"].fillna(1)
    gf["is_major_holiday"] = gf["is_major_holiday"].fillna(0)
    gf["temperature_c"] = gf["temperature_c"].fillna(gf["temperature_c"].median())
    gf["rainfall_mm"] = gf["rainfall_mm"].fillna(0)

    return gf, df


def get_prediction_calendar():
    """生成预测期 384 步的日历信息。"""
    ts = pd.date_range(PRED_START, PRED_END, freq="15min")
    cal = pd.DataFrame({"timestamp": ts})
    cal["hour"] = cal["timestamp"].dt.hour
    cal["minute_of_day"] = cal["timestamp"].dt.hour * 4 + cal["timestamp"].dt.minute // 15
    cal["dayofweek"] = cal["timestamp"].dt.dayofweek
    cal["is_workday"] = cal["dayofweek"].apply(lambda x: 1 if x < 5 else 0)
    cal["day_of_month"] = cal["timestamp"].dt.day
    return cal


def evaluate(y_true, y_pred):
    """计算统一评估指标。

    Returns:
        dict: MAE, RMSE, WAPE, MAPE_nonzero, F1
    """
    y_true = np.array(y_true, dtype=float)
    y_pred = np.maximum(np.array(y_pred, dtype=float), 0)

    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))

    total = np.sum(np.abs(y_true))
    wape = np.sum(np.abs(y_true - y_pred)) / total if total > 0 else 1.0

    nz = y_true > 0.01
    mape_nz = np.mean(np.abs(y_true[nz] - y_pred[nz]) / y_true[nz]) * 100 if nz.sum() > 0 else np.nan

    y_tb = (y_true > 0).astype(int)
    y_pb = (y_pred > 0.5).astype(int)
    f1 = f1_score(y_tb, y_pb, zero_division=0)

    return {
        "MAE": round(mae, 4), "RMSE": round(rmse, 4),
        "WAPE": round(wape, 4),
        "MAPE_nonzero": round(mape_nz, 2) if not np.isnan(mape_nz) else None,
        "F1": round(f1, 4),
    }
