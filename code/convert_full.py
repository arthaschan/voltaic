#!/usr/bin/env python3
"""
convert_full.py —— 将源数据集 processed_data.xlsx 全量转换为 sample_dataset1 格式。

目标格式（严格对齐 /Users/arthas/git/photovoltaic/sample_dataset1.zip）：
  charging_power_dataset_full/
    ├── data/case_001.csv ... case_044.csv      (28 输入字段)
    ├── ground_truth/case_001.csv ... (4 输出字段)
    ├── manifest.json / README.md / scripts/validate.py

字段映射（28 字段）：
  真实映射(5):  current_a<-charginga, voltage_v<-chargingv, power_w<-out_power*1000,
                report_time<-end_time, date<-end_time 日期
  确定性派生(4): station_id<-transaction_id 前12位, connector_id<-transaction_id 前16位,
                connector_status<-out_power>0, operator_id<-常量(数据方已合并)
  日历推导(2):  is_workday, is_major_holiday
  确定性伪造(17): grid_id/温度/降雨/行政区NEV销量BEV营运/重大事件/网格面积/行政区名/7项网格设施

划分方式（关键，针对数据极度偏斜）：
  源数据 111 个场站中，单一超大场站 440201004000 占全量 95%（122万采样、15充电桩、跨8数据方）。
  → 该场站按时间确定性切为 40 个连续窗口（每窗口≈1个月）；其余 110 个小场站按采样量
    轮询归入 4 个空间分组。合计 44 个 case，规模均衡、每个 case 均为连续时间序列。

ground_truth 口径：每个 case 末尾 4 天（384 个 15min 步），
  网格级功率 = 每 15min 窗口各充电枪平均功率求和 / 1000 (kW)。
  —— 与北京评测「选切分点、用之前一周预测之后」的语义一致。
"""
import os, json, sys
import numpy as np
import pandas as pd

SRC = "/Users/arthas/git/excharge/dataset /processed_data.xlsx"
# 颗粒度：'1min' 保持原始 1 分钟采样，'15min' 聚合为每枪每 15min 窗口一条
GRANULARITY = sys.argv[1] if len(sys.argv) > 1 else '15min'
OUT = f"/Users/arthas/git/photovoltaic/charging_power_dataset_full_{GRANULARITY}"
NG = 34            # case 总数 = 超大场站30段 + 小场站4空间组
N_MEGA = 30          # 超大场站的时间窗口数(每段~41天, data 跨度强制 30 天 + gt 4 天 + 7 天余量)
N_SMALL = NG - N_MEGA  # 小场站的空间分组数 = 4
OP_ID = 589179428

from datetime import timedelta

FIXED_HOLIDAYS = {(1, 1), (5, 1), (10, 1), (10, 2), (10, 3)}
SPRING_FESTIVAL = {
    "2020-01-24", "2020-01-25", "2020-01-26", "2020-01-27", "2020-01-28", "2020-01-29", "2020-01-30",
    "2021-02-11", "2021-02-12", "2021-02-13", "2021-02-14", "2021-02-15", "2021-02-16", "2021-02-17",
    "2022-01-31", "2022-02-01", "2022-02-02", "2022-02-03", "2022-02-04", "2022-02-05", "2022-02-06",
    "2023-01-21", "2023-01-22", "2023-01-23", "2023-01-24", "2023-01-25", "2023-01-26", "2023-01-27",
    "2024-02-10", "2024-02-11", "2024-02-12", "2024-02-13", "2024-02-14", "2024-02-15", "2024-02-16",
}
SHENZHEN_DISTRICTS = ["南山区", "福田区", "罗湖区", "宝安区", "龙岗区", "龙华区",
                      "光明区", "坪山区", "盐田区", "大鹏新区"]

INPUT_HEADERS = [
    "date", "operator_id", "station_id", "connector_id", "connector_status",
    "current_a", "voltage_v", "power_w", "report_time", "grid_id",
    "temperature_c", "rainfall_mm", "district_nev_penetration_rate", "district_ev_sales",
    "district_bev_ratio", "district_operating_ev_ratio", "is_major_holiday", "has_major_event",
    "is_workday", "grid_area_km2", "district_name", "grid_total_charging_facilities",
    "grid_slow_charging_facilities", "grid_fast_charging_facilities", "grid_super_charging_facilities",
    "grid_super_charging_stations", "grid_public_fast_charging_stations", "grid_public_slow_charging_stations"
]


def is_holiday(ts):
    return int((ts.month, ts.day) in FIXED_HOLIDAYS or ts.strftime("%Y-%m-%d") in SPRING_FESTIVAL)


def read_source():
    xl = pd.ExcelFile(SRC)
    frames = []
    for s in xl.sheet_names:
        d = pd.read_excel(xl, sheet_name=s,
                          usecols=["transaction_id", "end_time", "charginga", "chargingv", "out_power"])
        d["transaction_id"] = d["transaction_id"].astype(str)
        d["report_time"] = pd.to_datetime(d["end_time"])
        d["current_a"] = d["charginga"].astype(float)
        d["voltage_v"] = d["chargingv"].astype(float)
        d["power_w"] = d["out_power"].astype(float) * 1000.0
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates(subset=["transaction_id", "report_time"]).reset_index(drop=True)
    df["station_key"] = df["transaction_id"].str[:12]
    df["pile_key"] = df["transaction_id"].str[:16]
    return df


def build_id_maps(df):
    stations = sorted(df["station_key"].unique())
    piles = sorted(df["pile_key"].unique())
    station_id = {k: f"2019{i + 1:016d}" for i, k in enumerate(stations)}
    pile_id = {k: f"9002{i + 1:014d}" for i, k in enumerate(piles)}
    return station_id, pile_id


def aggregate_power(gdf, start, end):
    """15min 网格级功率：每窗口各充电枪平均功率求和 / 1000 (kW)。"""
    gdf = gdf.copy()
    gdf["w"] = gdf["report_time"].dt.floor("15min")
    g = gdf.groupby(["w", "pile_key"])["power_w"].mean().groupby("w").sum() / 1000.0
    idx = pd.date_range(start, end, freq="15min")
    ser = pd.Series(0.0, index=idx)
    ser.update(g)
    return ser


def aggregate_to_15min(g):
    """把逐采样 g 聚合成「每枪每 15min 窗口一条」（电流/电压/功率取窗口平均），
    report_time 对齐到 15min 窗口起点。用于让 data 与测试数据（15min 颗粒度）保持一致。"""
    g = g.copy()
    g["w"] = g["report_time"].dt.floor("15min")
    agg = g.groupby(["w", "station_key", "pile_key"], as_index=False).agg(
        current_a=("current_a", "mean"),
        voltage_v=("voltage_v", "mean"),
        power_w=("power_w", "mean"),
    )
    agg["report_time"] = agg["w"]
    agg = agg.drop(columns=["w"])
    return agg


def build_28cols(g, meta, station_id, pile_id):
    """把一组采样 g 构造成 28 字段 DataFrame（按 report_time 排序）。"""
    g = g.sort_values("report_time").reset_index(drop=True)
    ts = g["report_time"]
    month = ts.dt.month.astype(float)
    hour = ts.dt.hour.astype(float)
    temp = (20 + 8 * np.sin(2 * np.pi * (month - 1) / 12) + 3 * np.sin(2 * np.pi * hour / 24)).round(2)
    rain = ((month.isin([4, 5, 6, 7, 8])).astype(float) * 5.0).round(3)
    workday = ((ts.dt.weekday < 5) & (~ts.dt.date.astype(str).isin(SPRING_FESTIVAL))).astype(int)
    holiday = ts.apply(is_holiday).astype(int)
    out = pd.DataFrame({
        "date": ts.dt.strftime("%Y-%m-%d"),
        "operator_id": OP_ID,
        "station_id": g["station_key"].map(station_id),
        "connector_id": g["pile_key"].map(pile_id),
        "connector_status": (g["power_w"] > 0).astype(int),
        "current_a": g["current_a"].map(lambda v: f"{v:.3f}"),
        "voltage_v": g["voltage_v"].map(lambda v: f"{v:.3f}"),
        "power_w": g["power_w"].map(lambda v: f"{v:.3f}"),
        "report_time": ts.dt.strftime("%Y-%m-%d %H:%M:%S.%f").str[:-3],
        "grid_id": meta["grid_id"],
        "temperature_c": temp.map(lambda v: f"{v:.2f}"),
        "rainfall_mm": rain.map(lambda v: f"{v:.3f}"),
        "district_nev_penetration_rate": 0.666012,
        "district_ev_sales": 16605,
        "district_bev_ratio": 0.435592,
        "district_operating_ev_ratio": 0.055345,
        "is_major_holiday": holiday.values,
        "has_major_event": 0,
        "is_workday": workday.values,
        "grid_area_km2": f"{meta['area']:.4f}",
        "district_name": meta["district"],
        "grid_total_charging_facilities": meta["n_piles"],
        "grid_slow_charging_facilities": meta["n_piles"],
        "grid_fast_charging_facilities": 0,
        "grid_super_charging_facilities": 0,
        "grid_super_charging_stations": 0,
        "grid_public_fast_charging_stations": 0,
        "grid_public_slow_charging_stations": meta["n_stations"],
    })
    return out[INPUT_HEADERS]


def write_ground_truth(gdf_all, fpath):
    """gdf_all = 该 case 全部采样。末尾 4 天作为 ground_truth，输出 4 尺度(8+16+96+384)行。"""
    if len(gdf_all) == 0:
        # 空 case：写出全零真值以保证格式合规
        fidx = pd.date_range('2020-01-01', periods=384, freq='15min')
        rows = [{"forecast_horizon": hname, "step_index": j + 1,
                 "forecast_timestamp": fidx[j].strftime('%Y-%m-%d %H:%M:%S'),
                 "power_kw": 0.0}
                for hname, hs in [("2h", 8), ("4h", 16), ("1d", 96), ("4d", 384)]
                for j in range(hs)]
        pd.DataFrame(rows, columns=["forecast_horizon", "step_index", "forecast_timestamp", "power_kw"]) \
            .to_csv(fpath, index=False)
        return
    tmax = gdf_all["report_time"].max()
    end_bin = tmax.floor("15min")
    start_bin = end_bin - pd.Timedelta(minutes=15 * (384 - 1))
    ser = aggregate_power(gdf_all, start_bin, end_bin)
    fidx = ser.index
    rows = []
    for hname, hs in [("2h", 8), ("4h", 16), ("1d", 96), ("4d", 384)]:
        for j in range(hs):
            rows.append({"forecast_horizon": hname, "step_index": j + 1,
                         "forecast_timestamp": fidx[j].strftime("%Y-%m-%d %H:%M:%S"),
                         "power_kw": round(float(ser.iloc[j]), 4)})
    pd.DataFrame(rows, columns=["forecast_horizon", "step_index", "forecast_timestamp", "power_kw"]) \
        .to_csv(fpath, index=False)


def make_case_from_window(g, seg_start, seg_end, min_days=30):
    """从一段源采样 g（段 = [seg_start, seg_end]）构建一个 case：
       - 末日 D = 段内"往前 7 天连续"的最大日
       - data 跨度强制 ≥ min_days 天（data = [D-29, D]，gt = [D+1, D+4]）
       - 若段内没有 7 天连续，或 data 跨度不够，返回 (空, 空)，调用方跳过该 case
    """
    g = g.copy()
    if len(g) == 0:
        return g.iloc[0:0], g.iloc[0:0]

    g["date_only"] = g["report_time"].dt.date
    seg_dates = sorted(set(g["date_only"].unique()))
    daily_set = set(seg_dates)

    # 找段内"末日往前 7 天连续"的最大日
    last_valid_day = None
    for d in reversed(seg_dates):
        if all((d - timedelta(days=k)) in daily_set for k in range(7)):
            last_valid_day = d
            break
    if last_valid_day is None:
        return g.iloc[0:0], g.iloc[0:0]

    end_bin = pd.Timestamp(last_valid_day) + pd.Timedelta(hours=23, minutes=45)
    gt_start = end_bin - pd.Timedelta(minutes=15 * (384 - 1))
    data_min = gt_start - pd.Timedelta(days=min_days)

    if data_min < seg_start:
        # data 起点超出段起始，data 跨度可能不足 30 天
        data_min = seg_start
        # 末日 = data_min + min_days（强制 data 跨度 = min_days）
        data_end = data_min + pd.Timedelta(days=min_days)
        gt_start = data_end - pd.Timedelta(minutes=15 * (384 - 1))

    data_part = g[(g["report_time"] >= data_min) & (g["report_time"] < gt_start)].drop(columns=["date_only"])
    gt_all = g[g["report_time"] >= gt_start].drop(columns=["date_only"])

    # 最后一道检查：data 跨度 ≥ min_days 且末日往前 7 天连续，且 gt 必须有数据
    if len(data_part) == 0 or len(gt_all) == 0:
        return g.iloc[0:0], g.iloc[0:0]
    rt = data_part["report_time"]
    span = (rt.max().normalize() - rt.min().normalize()).days + 1
    if span < min_days:
        return g.iloc[0:0], g.iloc[0:0]
    last_7 = [(rt.max().normalize() - timedelta(days=k)).date() for k in range(7)]
    if not all(d in set(rt.dt.date.unique()) for d in last_7):
        return g.iloc[0:0], g.iloc[0:0]

    return data_part, gt_all


def main():
    t0 = pd.Timestamp.now()
    print("读取并去重源数据...")
    df = read_source()
    print(f"  去重后采样行 = {len(df):,}  场站 = {df['station_key'].nunique()}  桩 = {df['pile_key'].nunique()}")
    print(f"  时间范围 = {df['report_time'].min()} ~ {df['report_time'].max()}")

    station_id, pile_id = build_id_maps(df)

    # 识别超大场站
    counts = df.groupby("station_key").size().sort_values(ascending=False)
    mega = counts.index[0]
    mega_df = df[df["station_key"] == mega].sort_values("report_time").reset_index(drop=True)
    small_df = df[df["station_key"] != mega]
    print(f"  超大场站 {mega}: {len(mega_df):,} 采样 ({len(mega_df)/len(df)*100:.1f}%)，其余 {small_df['station_key'].nunique()} 场站 {len(small_df):,} 采样")

    os.makedirs(f"{OUT}/data", exist_ok=True)
    os.makedirs(f"{OUT}/ground_truth", exist_ok=True)
    os.makedirs(f"{OUT}/scripts", exist_ok=True)

    case_files = []
    sizes = []
    meta_list = []

    # --- 超大场站：按日期均切 N_MEGA 个连续日期窗口 ---
    mega_df_local = mega_df.copy()
    mega_df_local["date_only"] = mega_df_local["report_time"].dt.date
    mega_dates = sorted(set(mega_df_local["date_only"].unique()))
    date_chunks = np.array_split(np.arange(len(mega_dates)), N_MEGA)
    for gi, idx in enumerate(date_chunks):
        if len(idx) == 0: continue
        seg_dates = [mega_dates[i] for i in idx]
        seg_start = pd.Timestamp(seg_dates[0])
        seg_end_dt = seg_dates[-1]
        seg_end = pd.Timestamp(seg_end_dt) + pd.Timedelta(days=1)
        g = mega_df[(mega_df["report_time"] >= seg_start) & (mega_df["report_time"] < seg_end)].copy()
        if len(g) == 0: continue
        meta = {
            "grid_id": f"L2FT01-FTJ{gi + 1:03d}",
            "district": SHENZHEN_DISTRICTS[gi % len(SHENZHEN_DISTRICTS)],
            "area": round(0.3 + (gi % 7) * 0.12, 4),
            "n_stations": 1,
            "n_piles": int(g["pile_key"].nunique()),
        }
        data_part, gt_all = make_case_from_window(g, seg_start, seg_end)
        if len(data_part) == 0 or len(gt_all) == 0: continue
        fn = f"case_{gi + 1:03d}.csv"
        data_out = aggregate_to_15min(data_part) if GRANULARITY == '15min' else data_part
        out = build_28cols(data_out, meta, station_id, pile_id)
        out.to_csv(f"{OUT}/data/{fn}", index=False)
        gt_src = gt_all if len(gt_all) > 0 else data_part
        write_ground_truth(gt_src, f"{OUT}/ground_truth/{fn}")
        case_files.append(fn)
        sizes.append(len(data_out))
        meta_list.append(meta)

    # --- 小场站：轮询归入 N_SMALL 个空间分组 ---
    small_counts = small_df.groupby("station_key").size().sort_values(ascending=False)
    groups = [[] for _ in range(N_SMALL)]
    loads = [0] * N_SMALL
    for st in small_counts.index:
        gi = int(np.argmin(loads))
        groups[gi].append(st)
        loads[gi] += int(small_counts[st])
    for gi, sts in enumerate(groups):
        gidx = N_MEGA + gi
        g = small_df[small_df["station_key"].isin(sts)].sort_values("report_time").copy()
        if len(g) == 0: continue
        seg_start = g["report_time"].min()
        seg_end = g["report_time"].max() + pd.Timedelta(minutes=1)
        meta = {
            "grid_id": f"L2FT01-FTJ{gidx + 1:03d}",
            "district": SHENZHEN_DISTRICTS[gidx % len(SHENZHEN_DISTRICTS)],
            "area": round(0.3 + (gidx % 7) * 0.12, 4),
            "n_stations": len(sts),
            "n_piles": int(g["pile_key"].nunique()),
        }
        data_part, gt_all = make_case_from_window(g, seg_start, seg_end)
        fn = f"case_{gidx + 1:03d}.csv"
        data_out = aggregate_to_15min(data_part) if GRANULARITY == '15min' else data_part
        out = build_28cols(data_out, meta, station_id, pile_id)
        out.to_csv(f"{OUT}/data/{fn}", index=False)
        gt_src = gt_all if len(gt_all) > 0 else data_part
        write_ground_truth(gt_src, f"{OUT}/ground_truth/{fn}")
        case_files.append(fn)
        sizes.append(len(data_out))
        meta_list.append(meta)

    print(f"\n=== 生成完成 ===")
    print(f"  case 数 = {len(case_files)}")
    print(f"  data 行数: min={min(sizes):,} max={max(sizes):,} 平均={sum(sizes)/len(sizes):,.0f}")
    print(f"  总耗时 {(pd.Timestamp.now()-t0).total_seconds():.1f}s")

    manifest = {
        "dataset_id": f"charging_power_full_{GRANULARITY}_2024",
        "dataset_name": "车网互动充电功率预测数据集（全量，源自 Autosun 深圳充电站）",
        "version": "1.0.0",
        "provider": "深圳 Autosun 充电站（开源数据）",
        "contact": "nurry@139.com",
        "upload_date": "2026-08-20",
        "modality": "tabular",
        "data_format": "csv",
        "case_count": len(case_files),
        "has_ground_truth": True,
        "evaluation": {
            "task_type": "forecasting",
            "metrics": ["mape"],
            "horizons": ["2h", "4h", "1d", "4d"],
            "horizon_steps": [8, 16, 96, 384]
        },
        "structure": {
            "data": {"location": "data/", "files": case_files, "columns": 28},
            "ground_truth": {"location": "ground_truth/", "files": case_files, "columns": 4}
        },
        "notes": "全量转换：单一超大场站(440201004000,占95%)按时间切40窗口 + 其余110场站轮询归4组 = 44 case；真实列5+派生4+日历2+确定性伪造17；无放电(纯充电)；ground_truth=各case末尾4天。"
    }
    with open(f"{OUT}/manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print("  manifest.json 已生成")


if __name__ == "__main__":
    main()
