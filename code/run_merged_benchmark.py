#!/usr/bin/env python3
"""统一 benchmark：7 个模型 × 28 case，统一口径"预测 ground_truth 4 天"。
7 模型 = All-Zero + Slot-Median(教授) + LightGBM(我) + DEMMFL(教授) + GRU(教授) + PatchTST(我) + TimesNet(我)
统一指标：MAE / RMSE / WAPE / MAPE_nz / F1
"""
import os, sys, time, json
os.environ['OMP_NUM_THREADS'] = '1'
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

# ---- 我的模型库 ----
MY = '/Users/arthas/git/photovoltaic/charging_power_dataset/scripts'
sys.path.insert(0, MY)
import model_zoo as myz

# ---- 教授的模型库（用 importlib 显式加载，避免与我的 model_zoo 同名冲突）----
PROF = '/Users/arthas/git/photovoltaic/charging_power_dataset_delivery_V1/code'
import importlib.util
_spec = importlib.util.spec_from_file_location('professor_model_zoo', PROF + '/model_zoo.py')
pmz = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pmz)

DATA = '/Users/arthas/git/photovoltaic/charging_power_dataset_full_15min/data'
GT = '/Users/arthas/git/photovoltaic/charging_power_dataset_full_15min/ground_truth'


def evaluate(y_true, y_pred):
    y_true = np.array(y_true, float)
    y_pred = np.maximum(np.array(y_pred, float), 0)
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    total = np.sum(np.abs(y_true))
    wape = np.sum(np.abs(y_true - y_pred)) / total if total > 0 else 1.0
    nz = y_true > 0.01
    mape_nz = np.mean(np.abs(y_true[nz] - y_pred[nz]) / y_true[nz]) * 100 if nz.sum() else np.nan
    f1 = f1_score((y_true > 0).astype(int), (y_pred > 0.5).astype(int), zero_division=0)
    return dict(MAE=mae, RMSE=rmse, WAPE=wape, MAPE_nz=mape_nz, F1=f1)


def build_gf_cal(csv_path, gt_path):
    df = pd.read_csv(csv_path, parse_dates=['report_time'])
    df['date'] = pd.to_datetime(df['date'])
    df['ts_15min'] = df['report_time'].dt.floor('15min')
    gun_avg = df.groupby(['ts_15min', 'connector_id'])['power_w'].mean().reset_index()
    grid_power = gun_avg.groupby('ts_15min')['power_w'].sum().reset_index()
    grid_power.columns = ['timestamp', 'power_w']
    grid_power['power_kw'] = grid_power['power_w'] / 1000.0
    gt = pd.read_csv(gt_path, parse_dates=['forecast_timestamp'])
    pred_start = gt['forecast_timestamp'].min()
    full_range = pd.date_range(grid_power['timestamp'].min(), pred_start - pd.Timedelta(minutes=15), freq='15min')
    gf = pd.DataFrame({'timestamp': full_range})
    gf = gf.merge(grid_power[['timestamp', 'power_kw']], on='timestamp', how='left')
    gf['power_kw'] = gf['power_kw'].fillna(0)
    date_attrs = df.groupby('date').agg({'is_workday': 'first', 'is_major_holiday': 'first',
                                         'temperature_c': 'mean', 'rainfall_mm': 'mean'}).reset_index()
    date_attrs.columns = ['date_dt', 'is_workday', 'is_major_holiday', 'temperature_c', 'rainfall_mm']
    gf['date_dt'] = gf['timestamp'].dt.normalize()
    gf = gf.merge(date_attrs, on='date_dt', how='left')
    gf['is_workday'] = gf['is_workday'].fillna(1)
    gf['is_major_holiday'] = gf['is_major_holiday'].fillna(0)
    gf['temperature_c'] = gf['temperature_c'].fillna(gf['temperature_c'].median())
    gf['rainfall_mm'] = gf['rainfall_mm'].fillna(0)
    ts = pd.date_range(pred_start, pred_start + pd.Timedelta(minutes=15 * 383), freq='15min')
    cal = pd.DataFrame({'timestamp': ts})
    cal['hour'] = cal['timestamp'].dt.hour
    cal['minute_of_day'] = cal['timestamp'].dt.hour * 4 + cal['timestamp'].dt.minute // 15
    cal['dayofweek'] = cal['timestamp'].dt.dayofweek
    cal['is_workday'] = (cal['dayofweek'] < 5).astype(int)
    cal['day_of_month'] = cal['timestamp'].dt.day
    return gf, cal, gt


# GRU 30 epoch 版（复制教授的 gru_seq，epoch 80->30 加速）
def gru_seq_fast(gf, cal, epochs=30):
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    torch.set_num_threads(1)
    device = torch.device('cpu')

    class GRUModel(nn.Module):
        def __init__(self, dim, hid=32, layers=2, drop=0.3):
            super().__init__()
            self.gru = nn.GRU(dim, hid, num_layers=layers, dropout=drop, batch_first=True)
            self.fc = nn.Linear(hid, 1)
        def forward(self, x):
            out, _ = self.gru(x)
            return self.fc(out[:, -1, :]).squeeze(-1)

    class SeqDataset(Dataset):
        def __init__(self, feats, tgts, seq=96):
            self.f, self.t, self.s = feats, tgts, seq
        def __len__(self): return len(self.f) - self.s
        def __getitem__(self, i):
            return torch.FloatTensor(self.f[i:i+self.s]), torch.FloatTensor([self.t[i+self.s]])

    gf = gf.copy()
    gf['hour_sin'] = np.sin(2*np.pi*gf['timestamp'].dt.hour/24)
    gf['hour_cos'] = np.cos(2*np.pi*gf['timestamp'].dt.hour/24)
    gf['dow_sin'] = np.sin(2*np.pi*gf['timestamp'].dt.dayofweek/7)
    gf['dow_cos'] = np.cos(2*np.pi*gf['timestamp'].dt.dayofweek/7)
    cols = ['power_kw', 'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos', 'is_workday', 'temperature_c']
    for c in cols:
        if c not in gf.columns: gf[c] = 0
    feats = gf[cols].values.astype(np.float32)
    targets = gf['power_kw'].values.astype(np.float32)
    f_mean, f_std = feats.mean(0), feats.std(0) + 1e-8
    feats_n = (feats - f_mean) / f_std
    t_max = max(targets.max(), 1e-8)
    targets_n = targets / t_max
    seq_len = 96
    ds = SeqDataset(feats_n, targets_n, seq_len)
    vs = min(96*4, len(ds)//5)
    ts_size = len(ds) - vs
    train_dl = DataLoader(torch.utils.data.Subset(ds, range(ts_size)), batch_size=64, shuffle=True)
    val_dl = DataLoader(torch.utils.data.Subset(ds, range(ts_size, len(ds))), batch_size=64)
    model = GRUModel(len(cols)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()
    best_vl, patience, cnt, best_st = float('inf'), 5, 0, None
    for ep in range(epochs):
        model.train()
        for xb, yb in train_dl:
            opt.zero_grad(); loss = loss_fn(model(xb), yb.squeeze(-1))
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        model.eval(); vl = 0
        with torch.no_grad():
            for xb, yb in val_dl:
                vl += loss_fn(model(xb), yb.squeeze(-1)).item()
        vl /= max(len(val_dl), 1)
        if vl < best_vl: best_vl, cnt, best_st = vl, 0, {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            cnt += 1
            if cnt >= patience: break
    if best_st: model.load_state_dict(best_st)
    model.eval()
    window = list(feats_n[-seq_len:])
    preds = []
    with torch.no_grad():
        for i in range(384):
            x = torch.FloatTensor([window[-seq_len:]])
            p = max(0, model(x).item() * t_max)
            preds.append(p)
            ts = pd.Timestamp(cal['timestamp'].iloc[i])
            nf = np.array([p, np.sin(2*np.pi*ts.hour/24), np.cos(2*np.pi*ts.hour/24),
                           np.sin(2*np.pi*ts.dayofweek/7), np.cos(2*np.pi*ts.dayofweek/7),
                           1.0 if ts.dayofweek < 5 else 0.0, gf['temperature_c'].iloc[-1]], dtype=np.float32)
            window.append((nf - f_mean) / f_std)
    return np.array(preds)


def main():
    ids = sorted([f.replace('case_', '').replace('.csv', '') for f in os.listdir(DATA)
                  if f.startswith('case_') and f.endswith('.csv')])
    n = len(ids)
    print(f"共 {n} 个 case")

    # 预加载：真值 + 我的序列 + 教授的 gf/cal
    y_true_all = {}
    my_series = []
    my_active = []
    prof_gf = {}
    prof_cal = {}
    for cid in ids:
        gt = pd.read_csv(f'{GT}/case_{cid}.csv', parse_dates=['forecast_timestamp'])
        y_true_all[cid] = gt[gt['forecast_horizon'] == '4d'].sort_values('step_index')['power_kw'].values
        p, a = myz.load_series(f'{DATA}/case_{cid}.csv')
        my_series.append(p)
        my_active.append(a)
        gf, cal, _ = build_gf_cal(f'{DATA}/case_{cid}.csv', f'{GT}/case_{cid}.csv')
        prof_gf[cid] = gf
        prof_cal[cid] = cal

    results = {m: [] for m in ['allzero', 'slotmedian', 'lightgbm', 'demmfl', 'gru', 'patchtst', 'timesnet']}

    # 1. All-Zero
    for cid in ids:
        results['allzero'].append((cid, evaluate(y_true_all[cid], np.zeros(384))))

    # 2. Slot-Median（教授）
    for cid in ids:
        pred = pmz.baseline_median(prof_gf[cid], prof_cal[cid])
        results['slotmedian'].append((cid, evaluate(y_true_all[cid], pred[:384])))

    # 3. 我的 LightGBM（pooled 训练）
    print("\n[LightGBM] pooled 训练...")
    lgb = myz.LightGBMModel()
    lgb.fit(my_series, my_active)
    for i, cid in enumerate(ids):
        pred = lgb.predict(my_series[i], my_active[i], fc_start=prof_cal[cid]['timestamp'].iloc[0], n_horizon=384)
        results['lightgbm'].append((cid, evaluate(y_true_all[cid], pred)))

    # 4. DEMMFL（教授，逐 case）
    print("[DEMMFL] 逐 case...")
    for cid in ids:
        pred = pmz.demmfl_lasso_ridge(prof_gf[cid], prof_cal[cid])
        results['demmfl'].append((cid, evaluate(y_true_all[cid], pred[:384])))

    # 5. GRU（教授，30 epoch，逐 case）
    print("[GRU] 逐 case（30 epoch）...")
    for cid in ids:
        pred = gru_seq_fast(prof_gf[cid], prof_cal[cid])
        results['gru'].append((cid, evaluate(y_true_all[cid], pred[:384])))

    # 6. PatchTST（我，pooled 训练）
    print("[PatchTST] pooled 训练...")
    pt = myz.PatchTSTModel()
    pt.fit(my_series)
    for i, cid in enumerate(ids):
        pred = pt.predict(my_series[i], fc_start=prof_cal[cid]['timestamp'].iloc[0])
        results['patchtst'].append((cid, evaluate(y_true_all[cid], pred)))

    # 7. TimesNet（我，pooled 训练）
    print("[TimesNet] pooled 训练...")
    tn = myz.TimesNetModel()
    tn.fit(my_series)
    for i, cid in enumerate(ids):
        pred = tn.predict(my_series[i], fc_start=prof_cal[cid]['timestamp'].iloc[0])
        results['timesnet'].append((cid, evaluate(y_true_all[cid], pred)))

    # 汇总
    print("\n" + "="*62)
    print(f"{'模型':<14} {'MAE':>7} {'RMSE':>7} {'WAPE':>7} {'MAPE_nz':>8} {'F1':>6}")
    print("-"*62)
    summary = {}
    for m in ['allzero', 'slotmedian', 'lightgbm', 'demmfl', 'gru', 'patchtst', 'timesnet']:
        rs = results[m]
        ma = np.mean([r[1]['MAE'] for r in rs])
        rm = np.mean([r[1]['RMSE'] for r in rs])
        wa = np.mean([r[1]['WAPE'] for r in rs])
        mp = np.nanmean([r[1]['MAPE_nz'] for r in rs])
        f1 = np.mean([r[1]['F1'] for r in rs])
        summary[m] = dict(MAE=round(ma,2), RMSE=round(rm,2), WAPE=round(wa,4), MAPE_nz=round(mp,1), F1=round(f1,3))
        print(f"{m:<14} {ma:>7.2f} {rm:>7.2f} {wa:>7.3f} {mp:>8.1f} {f1:>6.3f}")

    out = '/Users/arthas/git/photovoltaic/charging_power_dataset/predictions/merged_benchmark.json'
    json.dump(summary, open(out, 'w'), ensure_ascii=False, indent=2)
    print(f"\n结果已保存 {out}")


if __name__ == '__main__':
    main()