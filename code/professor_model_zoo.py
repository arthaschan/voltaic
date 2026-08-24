#!/usr/bin/env python3
"""model_zoo.py — 充电功率预测模型库

已实现的预测方法:
  1. baseline_median       同时段历史中位数
  2. lightgbm_twostage     两阶段 LightGBM (分类 + 回归)
  3. gru_seq               GRU 序列模型 (GPU 加速)
  4. demmfl_lasso_ridge     多模态动态特征 + Lasso-Ridge 两步法
  5. ensemble_demmfl_gru    DEMMFL + GRU 加权集成

每个方法接口统一: predict(gf, cal) -> np.ndarray (384,)
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.linear_model import LassoCV, RidgeCV
from sklearn.preprocessing import StandardScaler


# =====================================================================
# 1. 同时段历史中位数 (Baseline)
# =====================================================================
def baseline_median(gf, cal):
    """按 (工作日/非工作日, 15min 时段) 取历史中位数。"""
    gf = gf.copy()
    gf["minute_of_day"] = gf["timestamp"].dt.hour * 4 + gf["timestamp"].dt.minute // 15
    profiles = gf.groupby(["is_workday", "minute_of_day"])["power_kw"].median().reset_index()
    profiles.columns = ["is_workday", "minute_of_day", "pred"]
    cal = cal.copy()
    cal["mod"] = cal["minute_of_day"]
    cal = cal.merge(profiles, left_on=["is_workday", "mod"],
                    right_on=["is_workday", "minute_of_day"], how="left")
    return cal["pred"].fillna(0).values


# =====================================================================
# 2. 两阶段 LightGBM (分类 + 回归, 递归多步)
# =====================================================================
def _build_lgb_features(gf):
    gf = gf.copy()
    gf["hour"] = gf["timestamp"].dt.hour
    gf["minute_of_day"] = gf["timestamp"].dt.hour * 4 + gf["timestamp"].dt.minute // 15
    gf["dayofweek"] = gf["timestamp"].dt.dayofweek
    gf["day_of_month"] = gf["timestamp"].dt.day
    gf["hour_sin"] = np.sin(2 * np.pi * gf["hour"] / 24)
    gf["hour_cos"] = np.cos(2 * np.pi * gf["hour"] / 24)
    for lag in [1, 2, 3, 4, 8, 16, 96, 192, 288, 672]:
        gf[f"lag_{lag}"] = gf["power_kw"].shift(lag)
    for w in [4, 16, 96, 672]:
        gf[f"rolling_mean_{w}"] = gf["power_kw"].shift(1).rolling(w, min_periods=1).mean()
    gf["rolling_std_96"] = gf["power_kw"].shift(1).rolling(96, min_periods=1).std()
    gf["rolling_max_96"] = gf["power_kw"].shift(1).rolling(96, min_periods=1).max()
    gf["rolling_nonzero_96"] = gf["power_kw"].shift(1).rolling(96, min_periods=1).apply(lambda x: (x > 0).sum())
    gf["is_charging"] = (gf["power_kw"].shift(1) > 0).astype(int)
    cz_list, cnz_list = [], []
    cz, cnz = 0, 0
    for v in gf["power_kw"].values:
        if v == 0: cz += 1; cnz = 0
        else: cnz += 1; cz = 0
        cz_list.append(cz); cnz_list.append(cnz)
    gf["consecutive_zeros"] = [0] + cz_list[:-1]
    gf["consecutive_nonzeros"] = [0] + cnz_list[:-1]
    gf["target"] = gf["power_kw"]
    gf["target_binary"] = (gf["power_kw"] > 0).astype(int)
    return gf


def lightgbm_twostage(gf, cal):
    """两阶段 LightGBM: 分类门控 + 功率回归, 递归 384 步。"""
    gf_feat = _build_lgb_features(gf)
    feature_cols = [c for c in gf_feat.columns if c not in
                    ["timestamp", "power_kw", "target", "target_binary", "date_dt", "power_w"]]
    train_df = gf_feat.dropna(subset=feature_cols).copy()
    if len(train_df) < 100:
        return np.zeros(384)

    X_train = train_df[feature_cols].values
    y_bin = train_df["target_binary"].values
    y_reg = train_df["target"].values
    val_size = 96 * 4
    X_tr, X_val = X_train[:-val_size], X_train[-val_size:]

    pos_w = max(1, (y_bin[:-val_size] == 0).sum() / max((y_bin[:-val_size] == 1).sum(), 1))
    clf = lgb.LGBMClassifier(n_estimators=300, num_leaves=31, learning_rate=0.05,
                              scale_pos_weight=pos_w, verbose=-1)
    clf.fit(X_tr, y_bin[:-val_size], eval_set=[(X_val, y_bin[-val_size:])],
            callbacks=[lgb.early_stopping(30, verbose=False)])

    nz = y_reg[:-val_size] > 0
    reg = lgb.LGBMRegressor(n_estimators=300, num_leaves=31, learning_rate=0.05,
                             objective='mae', verbose=-1)
    if nz.sum() > 20:
        nz_val = y_reg[-val_size:] > 0
        if nz_val.sum() > 5:
            reg.fit(X_tr[nz], y_reg[:-val_size][nz],
                    eval_set=[(X_val[nz_val], y_reg[-val_size:][nz_val])],
                    callbacks=[lgb.early_stopping(30, verbose=False)])
        else:
            reg.fit(X_tr[nz], y_reg[:-val_size][nz])

    last_row = {c: gf_feat.iloc[-1].get(c, 0) for c in feature_cols}
    series = list(gf_feat["power_kw"].values)
    predictions = []
    for i in range(384):
        ts = pd.Timestamp(cal["timestamp"].iloc[i])
        feat = dict(last_row)
        feat["hour"] = ts.hour
        feat["minute_of_day"] = ts.hour * 4 + ts.minute // 15
        feat["dayofweek"] = ts.dayofweek
        feat["is_workday"] = 1 if ts.dayofweek < 5 else 0
        feat["is_major_holiday"] = 0
        feat["day_of_month"] = ts.day
        feat["hour_sin"] = np.sin(2 * np.pi * ts.hour / 24)
        feat["hour_cos"] = np.cos(2 * np.pi * ts.hour / 24)
        n = len(series)
        for lag in [1, 2, 3, 4, 8, 16, 96, 192, 288, 672]:
            feat[f"lag_{lag}"] = series[n - lag] if n - lag >= 0 else 0
        for w in [4, 16, 96, 672]:
            s = max(0, n - w)
            feat[f"rolling_mean_{w}"] = np.mean(series[s:n]) if s < n else 0
        w96 = series[max(0, n - 96):n]
        feat["rolling_std_96"] = np.std(w96) if len(w96) > 1 else 0
        feat["rolling_max_96"] = max(w96) if w96 else 0
        feat["rolling_nonzero_96"] = sum(1 for v in w96 if v > 0)
        feat["is_charging"] = 1 if n > 0 and series[-1] > 0 else 0
        cz = sum(1 for _ in range(len(series)) if series[-(1 + _)] == 0 and _ < len(series))
        feat["consecutive_zeros"] = min(cz, n)
        feat["consecutive_nonzeros"] = 0

        X_pred = np.array([[feat.get(c, 0) for c in feature_cols]])
        prob = clf.predict_proba(X_pred)[0][1]
        power = reg.predict(X_pred)[0] if nz.sum() > 20 else 0
        pred_val = max(0, power * prob) if prob > 0.3 else 0
        predictions.append(pred_val)
        series.append(pred_val)

    return np.array(predictions)


# =====================================================================
# 3. GRU 序列模型
# =====================================================================
def gru_seq(gf, cal):
    """2 层 GRU (hidden=32, dropout=0.3), 递归 384 步, 支持 GPU。"""
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    gf["hour_sin"] = np.sin(2 * np.pi * gf["timestamp"].dt.hour / 24)
    gf["hour_cos"] = np.cos(2 * np.pi * gf["timestamp"].dt.hour / 24)
    gf["dow_sin"] = np.sin(2 * np.pi * gf["timestamp"].dt.dayofweek / 7)
    gf["dow_cos"] = np.cos(2 * np.pi * gf["timestamp"].dt.dayofweek / 7)
    cols = ["power_kw", "hour_sin", "hour_cos", "dow_sin", "dow_cos", "is_workday", "temperature_c"]
    for c in cols:
        if c not in gf.columns: gf[c] = 0

    feats = gf[cols].values.astype(np.float32)
    targets = gf["power_kw"].values.astype(np.float32)
    f_mean, f_std = feats.mean(0), feats.std(0) + 1e-8
    feats_n = (feats - f_mean) / f_std
    t_max = max(targets.max(), 1e-8)
    targets_n = targets / t_max

    seq_len = 96
    ds = SeqDataset(feats_n, targets_n, seq_len)
    vs = min(96 * 4, len(ds) // 5)
    ts_size = len(ds) - vs
    train_dl = DataLoader(torch.utils.data.Subset(ds, range(ts_size)), batch_size=64, shuffle=True)
    val_dl = DataLoader(torch.utils.data.Subset(ds, range(ts_size, len(ds))), batch_size=64)

    model = GRUModel(len(cols)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()
    best_vl, patience, cnt, best_st = float('inf'), 10, 0, None

    for ep in range(80):
        model.train()
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(); loss = loss_fn(model(xb), yb.squeeze(-1))
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        model.eval(); vl = 0
        with torch.no_grad():
            for xb, yb in val_dl:
                xb, yb = xb.to(device), yb.to(device)
                vl += loss_fn(model(xb), yb.squeeze(-1)).item()
        vl /= max(len(val_dl), 1)
        if vl < best_vl:
            best_vl, cnt = vl, 0
            best_st = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            cnt += 1
            if cnt >= patience: break

    if best_st: model.load_state_dict(best_st); model.to(device)
    model.eval()

    window = list(feats_n[-seq_len:])
    preds = []
    with torch.no_grad():
        for i in range(384):
            x = torch.FloatTensor([window[-seq_len:]]).to(device)
            p = max(0, model(x).item() * t_max)
            preds.append(p)
            ts = pd.Timestamp(cal["timestamp"].iloc[i])
            nf = np.array([p, np.sin(2*np.pi*ts.hour/24), np.cos(2*np.pi*ts.hour/24),
                           np.sin(2*np.pi*ts.dayofweek/7), np.cos(2*np.pi*ts.dayofweek/7),
                           1.0 if ts.dayofweek<5 else 0.0, gf["temperature_c"].iloc[-1]], dtype=np.float32)
            window.append((nf - f_mean) / f_std)
    return np.array(preds)


# =====================================================================
# 4. DEMMFL: 多模态动态特征 + Lasso-Ridge
# =====================================================================
def _build_demmfl_features(gf):
    gf = gf.copy()
    h = gf["timestamp"].dt.hour.values
    gf["hour"] = h
    gf["minute_of_day"] = h * 4 + gf["timestamp"].dt.minute.values // 15
    gf["dayofweek"] = gf["timestamp"].dt.dayofweek.values
    gf["mode_night"] = ((h >= 0) & (h < 6)).astype(int)
    gf["mode_morning"] = ((h >= 6) & (h < 9)).astype(int)
    gf["mode_day"] = ((h >= 9) & (h < 18)).astype(int)
    gf["mode_evening"] = ((h >= 18) & (h < 22)).astype(int)
    gf["mode_late"] = ((h >= 22) & (h < 24)).astype(int)
    hourly_avg = gf.groupby("hour")["power_kw"].mean()
    gf["charging_like"] = gf["mode_night"].values * gf["hour"].map(hourly_avg).values
    for d in range(1, 8):
        gf[f"cl_lag_{d}d"] = gf["charging_like"].shift(d * 96)
    for lag in [1, 2, 4, 96, 192, 672]:
        gf[f"lag_{lag}"] = gf["power_kw"].shift(lag)
    gf["slow_1d"] = gf["power_kw"].shift(1).rolling(96, min_periods=1).mean()
    gf["slow_3d"] = gf["power_kw"].shift(1).rolling(288, min_periods=1).mean()
    gf["slow_7d"] = gf["power_kw"].shift(1).rolling(672, min_periods=1).mean()
    gf["fast_2h"] = gf["power_kw"].shift(1).rolling(8, min_periods=1).mean()
    gf["fast_4h"] = gf["power_kw"].shift(1).rolling(16, min_periods=1).mean()
    gf["is_charging_prev"] = (gf["power_kw"].shift(1) > 0).astype(int)
    gf["rolling_nz_96"] = gf["power_kw"].shift(1).rolling(96, min_periods=1).apply(lambda x: (x > 0).sum())
    gf["hour_sin"] = np.sin(2 * np.pi * h / 24)
    gf["hour_cos"] = np.cos(2 * np.pi * h / 24)
    gf["dow_sin"] = np.sin(2 * np.pi * gf["dayofweek"] / 7)
    gf["dow_cos"] = np.cos(2 * np.pi * gf["dayofweek"] / 7)
    gf["target"] = gf["power_kw"]
    return gf, hourly_avg


def demmfl_lasso_ridge(gf, cal):
    """运营模式划分 + charging_like 动态特征 + Lasso 变量选择 + Ridge 回归。

    参考: Liu et al., Applied Energy 2024 (DEMMFL)
    """
    gf_feat, hourly_avg = _build_demmfl_features(gf)
    exclude = {"timestamp", "power_kw", "target", "date_dt", "power_w"}
    feature_cols = [c for c in gf_feat.columns if c not in exclude
                    and gf_feat[c].dtype in [np.float64, np.int64, np.float32, np.int32, float, int]]
    train_df = gf_feat.dropna(subset=feature_cols).copy()
    if len(train_df) < 100: return np.zeros(384)

    X = train_df[feature_cols].values
    y = train_df["target"].values
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    alphas = np.exp(-np.arange(0, 10, 0.5))
    lasso = LassoCV(alphas=alphas, cv=5, max_iter=5000)
    lasso.fit(X_s, y)
    sel = np.where(np.abs(lasso.coef_) > 1e-6)[0]
    if len(sel) < 3: sel = np.argsort(np.abs(lasso.coef_))[-10:]
    ridge = RidgeCV(alphas=alphas, cv=5)
    ridge.fit(X_s[:, sel], y)

    history = list(gf["power_kw"].values)
    preds = []
    for i in range(384):
        ts = pd.Timestamp(cal["timestamp"].iloc[i])
        feat = {}
        feat["hour"] = ts.hour
        feat["minute_of_day"] = ts.hour * 4 + ts.minute // 15
        feat["dayofweek"] = ts.dayofweek
        feat["is_workday"] = 1 if ts.dayofweek < 5 else 0
        feat["is_major_holiday"] = 0
        feat["temperature_c"] = gf["temperature_c"].iloc[-1]
        feat["rainfall_mm"] = 0
        feat["mode_night"] = 1 if 0 <= ts.hour < 6 else 0
        feat["mode_morning"] = 1 if 6 <= ts.hour < 9 else 0
        feat["mode_day"] = 1 if 9 <= ts.hour < 18 else 0
        feat["mode_evening"] = 1 if 18 <= ts.hour < 22 else 0
        feat["mode_late"] = 1 if 22 <= ts.hour < 24 else 0
        feat["charging_like"] = feat["mode_night"] * hourly_avg.get(ts.hour, 0)
        n = len(history)
        for d in range(1, 8):
            idx = n - d * 96
            feat[f"cl_lag_{d}d"] = gf_feat.iloc[min(max(idx, 0), len(gf_feat)-1)].get("charging_like", 0) if idx >= 0 else 0
        for lag in [1, 2, 4, 96, 192, 672]:
            feat[f"lag_{lag}"] = history[n - lag] if n - lag >= 0 else 0
        for w, nm in [(96, "slow_1d"), (288, "slow_3d"), (672, "slow_7d")]:
            feat[nm] = np.mean(history[max(0, n-w):n]) if n > 0 else 0
        for w, nm in [(8, "fast_2h"), (16, "fast_4h")]:
            feat[nm] = np.mean(history[max(0, n-w):n]) if n > 0 else 0
        feat["is_charging_prev"] = 1 if n > 0 and history[-1] > 0 else 0
        feat["rolling_nz_96"] = sum(1 for v in history[max(0, n-96):n] if v > 0)
        feat["hour_sin"] = np.sin(2 * np.pi * ts.hour / 24)
        feat["hour_cos"] = np.cos(2 * np.pi * ts.hour / 24)
        feat["dow_sin"] = np.sin(2 * np.pi * ts.dayofweek / 7)
        feat["dow_cos"] = np.cos(2 * np.pi * ts.dayofweek / 7)

        x_vec = scaler.transform([[feat.get(c, 0) for c in feature_cols]])
        p = max(0, ridge.predict(x_vec[:, sel])[0])
        preds.append(p)
        history.append(p)
    return np.array(preds)


# =====================================================================
# 5. DEMMFL + GRU 加权集成
# =====================================================================
def ensemble_demmfl_gru(gf, cal):
    """验证集学习逐天权重 + 零值截断。"""
    pred_d = demmfl_lasso_ridge(gf, cal)
    pred_c = gru_seq(gf, cal)

    n = len(gf)
    val_len = min(384, n // 3)
    val_start = n - val_len
    if val_start < 672:
        return np.maximum(0, 0.5 * pred_d + 0.5 * pred_c)

    gf_train = gf.iloc[:val_start].copy()
    val_true = gf["power_kw"].iloc[val_start:val_start + val_len].values
    val_ts = gf["timestamp"].iloc[val_start:val_start + val_len].values
    vc = pd.DataFrame({"timestamp": val_ts})
    vc["hour"] = pd.to_datetime(vc["timestamp"]).dt.hour
    vc["minute_of_day"] = pd.to_datetime(vc["timestamp"]).dt.hour * 4 + pd.to_datetime(vc["timestamp"]).dt.minute // 15
    vc["dayofweek"] = pd.to_datetime(vc["timestamp"]).dt.dayofweek
    vc["is_workday"] = vc["dayofweek"].apply(lambda x: 1 if x < 5 else 0)

    vd = demmfl_lasso_ridge(gf_train, vc)
    vc2 = gru_seq(gf_train, vc)

    block_w = []
    for bs in range(0, 384, 96):
        be = min(bs + 96, 384, len(val_true))
        best_w, best_rmse = 0.5, float('inf')
        for w in np.arange(0, 1.01, 0.05):
            blend = w * vd[bs:be] + (1 - w) * vc2[bs:be]
            rmse = np.sqrt(np.mean((val_true[bs:be] - np.maximum(0, blend)) ** 2))
            if rmse < best_rmse: best_rmse, best_w = rmse, w
        block_w.append(best_w)

    result = np.zeros(384)
    for bi, bs in enumerate(range(0, 384, 96)):
        be = min(bs + 96, 384)
        w = block_w[bi] if bi < len(block_w) else 0.5
        result[bs:be] = np.maximum(0, w * pred_d[bs:be] + (1 - w) * pred_c[bs:be])

    return np.where(result < 1.5, 0, result)
