#!/usr/bin/env python3
"""数据集校验脚本：检查 data/ 和 ground_truth/ 的编码、表头、文件对应关系。"""
import os
import csv
import json
import sys

DATASET_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

INPUT_HEADERS = [
    "date", "operator_id", "station_id", "connector_id", "connector_status",
    "current_a", "voltage_v", "power_w", "report_time", "grid_id",
    "temperature_c", "rainfall_mm", "district_nev_penetration_rate", "district_ev_sales",
    "district_bev_ratio", "district_operating_ev_ratio", "is_major_holiday", "has_major_event",
    "is_workday", "grid_area_km2", "district_name", "grid_total_charging_facilities",
    "grid_slow_charging_facilities", "grid_fast_charging_facilities", "grid_super_charging_facilities",
    "grid_super_charging_stations", "grid_public_fast_charging_stations", "grid_public_slow_charging_stations"
]

OUTPUT_HEADERS = ["forecast_horizon", "step_index", "forecast_timestamp", "power_kw"]


def check_csv(filepath, expected_headers):
    """检查单个CSV文件的编码和表头。"""
    if not os.path.exists(filepath):
        return f"[MISS] {filepath}"
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            headers = next(reader)
            missing = set(expected_headers) - set(headers)
            if missing:
                return f"[WARN] {filepath}: 缺失字段 {missing}"
            return f"[ OK ] {filepath}"
    except UnicodeDecodeError:
        return f"[ERR ] {filepath}: 编码非UTF-8"


def main():
    manifest_path = os.path.join(DATASET_DIR, "manifest.json")
    if not os.path.exists(manifest_path):
        print("[ERR ] manifest.json 不存在")
        sys.exit(1)

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    print(f"数据集: {manifest['dataset_id']} v{manifest['version']}")
    print(f"Case数: {manifest['case_count']}")
    print()

    data_files = manifest["structure"]["data"]["files"]
    gt_files = manifest["structure"]["ground_truth"]["files"]

    print("--- data/ 输入文件 ---")
    for fname in data_files:
        fpath = os.path.join(DATASET_DIR, "data", fname)
        print(check_csv(fpath, INPUT_HEADERS))
    print()

    print("--- ground_truth/ 真值文件 ---")
    for fname in gt_files:
        fpath = os.path.join(DATASET_DIR, "ground_truth", fname)
        print(check_csv(fpath, OUTPUT_HEADERS))
    print()

    print("--- 文件对应关系 ---")
    missing_gt = set(data_files) - set(gt_files)
    if missing_gt:
        print(f"[WARN] ground_truth 缺少对应文件: {missing_gt}")
    else:
        print("[ OK ] data/ 与 ground_truth/ 文件名一一对应")


if __name__ == "__main__":
    main()
