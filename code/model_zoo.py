#!/usr/bin/env python3
"""model_zoo.py — 可扩展的多模型充电功率预测框架。

统一接口（每个模型实现两个方法，即可被 train_eval.py 调用）：
    fit(train_series_list, **kw)      # 训练；train_series_list = list[pd.Series]（15min 网格级功率 kW）
    predict(series, fc_start, n)      # 输入一个序列，输出未来 n 步期望功率 np.ndarray

模型注册表：@register('名字') 装饰新模型类，未来加第 3/4 种方法只需新增一个类 + 注册。
已实现：lightgbm（两阶段 LightGBM）、patchtst（ICLR 2023）、timesnet（ICLR 2023）。

只用真实列 report_time + connector_id + power_w；按 case 归一化（除以自身峰值）消除量级差异。
"""
import os, time
import numpy as np
import pandas as pd
import lightgbm as lgb
import torch
import torch.nn as nn
import joblib

os.environ['OMP_NUM_THREADS'] = '1'
torch.set_num_threads(1)

STEP = pd.Timedelta(minutes=15)
HORIZONS = {'2h': 8, '4h': 16, '1d': 96, '4d': 384}
NHORIZON = 384
ACT_THR = 0.05  # kW，活跃判定阈值

# ============ 数据加载 ============
def aggregate(df):
    """28 字段 CSV/DataFrame → (15min 网格级功率序列 kW, 15min 活跃枪数序列)。"""
    df = df.copy()
    df['ts'] = pd.to_datetime(df['report_time'])
    df['w'] = df['ts'].dt.floor('15min')
    g = df.groupby(['w', 'connector_id'])['power_w'].mean().groupby('w').sum() / 1000.0
    idx = pd.date_range(df['ts'].min().floor('15min'), df['ts'].max().floor('15min'), freq='15min')
    s = pd.Series(0.0, index=idx)
    s.update(g)
    dfa = df[df['power_w'] > 0]
    ga = dfa.groupby('w')['connector_id'].nunique()
    a = pd.Series(0, index=idx)
    a.update(ga)
    return s, a


def load_series(csv_path):
    """读 case CSV，返回 (功率序列, 活跃枪数序列)。"""
    return aggregate(pd.read_csv(csv_path))


def load_ground_truth(csv_path, horizon='4d'):
    gt = pd.read_csv(csv_path)
    return gt[gt['forecast_horizon'] == horizon].sort_values('step_index')['power_kw'].values


def build_windows(series, in_len=672, out_len=384, stride=96):
    """序列 → (X, Y) 滑动窗口（已归一化到 [0,1]）。X:(N,in_len) Y:(N,out_len)"""
    s = series.values.astype(float)
    scale = max(s.max(), 1e-3)
    sn = s / scale
    X, Y = [], []
    for st in range(0, len(sn) - in_len - out_len + 1, stride):
        X.append(sn[st:st + in_len])
        Y.append(sn[st + in_len:st + in_len + out_len])
    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)


def collect_windows(series_list, in_len=672, out_len=384, stride=96, max_per_series=120):
    """多个序列 → 合并的 (X, Y) 窗口。每序列均匀采样最多 max_per_series 个窗口，避免长序列主导。"""
    Xs, Ys = [], []
    for s in series_list:
        if len(s) < in_len + out_len:
            continue
        s_arr = s.values.astype(float)
        scale = max(s_arr.max(), 1e-3)
        sn = s_arr / scale
        max_start = len(sn) - in_len - out_len
        n_w = min(max_per_series, max_start // stride + 1)
        starts = np.linspace(0, max_start, n_w).astype(int)
        for st in starts:
            Xs.append(sn[st:st + in_len])
            Ys.append(sn[st + in_len:st + in_len + out_len])
    if not Xs:
        return np.zeros((0, in_len), np.float32), np.zeros((0, out_len), np.float32)
    return np.array(Xs, dtype=np.float32), np.array(Ys, dtype=np.float32)


# ============ 模型注册表 ============
MODEL_REGISTRY = {}


def register(name):
    def deco(cls):
        MODEL_REGISTRY[name] = cls
        return cls
    return deco


def get_model(name, **kw):
    return MODEL_REGISTRY[name](**kw)


def save_model(model, name, models_dir):
    """保存模型 wrapper（joblib，含 torch nn.Module）。"""
    os.makedirs(models_dir, exist_ok=True)
    path = os.path.join(models_dir, f'{name}.joblib')
    joblib.dump(model, path)
    return path


def load_cached_model(name, models_dir):
    """若缓存存在则加载，否则返回 None。"""
    path = os.path.join(models_dir, f'{name}.joblib')
    if os.path.exists(path):
        return joblib.load(path)
    return None


def train_and_cache(name, train_series_list, models_dir, active_list=None, **kw):
    """训练模型并缓存；已缓存则直接加载。"""
    cached = load_cached_model(name, models_dir)
    if cached is not None:
        print(f"  [模型 {name}] 已缓存，直接加载")
        return cached
    print(f"  [模型 {name}] 开始训练...")
    model = get_model(name, **kw)
    t0 = time.time()
    if name == 'lightgbm':
        model.fit(train_series_list, active_list)
    else:
        model.fit(train_series_list)
    save_model(model, name, models_dir)
    print(f"  [模型 {name}] 训练完成 {time.time()-t0:.1f}s，已缓存")
    return model


# ============ 模型 1：两阶段 LightGBM（Horizon 分模型：short 2h/4h + long 1d/4d）============
@register('lightgbm')
class LightGBMModel:
    """两阶段（活动分类 + 幅度回归）× Horizon 分模型：
    - short 模型：近期特征（lag 1~16 步 + 活跃枪数 lag + rolling），预测 2h/4h
    - long  模型：周期特征（lag 96/192/336/672 + 相似 weekday×slot 剖面），预测 1d/4d
    输出期望功率 = P(active) * E(幅度)。"""
    def __init__(self, **kw):
        self.short_clf = self.short_reg = None
        self.long_clf = self.long_reg = None
        self.prof = None

    def _profiles(self, series):
        s = series.values.astype(float)
        idx = series.index
        slot = idx.hour * 4 + idx.minute // 15
        wd = idx.weekday
        ser = pd.Series(s, index=pd.MultiIndex.from_arrays([wd, slot]))
        pa, pm = {}, {}
        for (w, sl), grp in ser.groupby(level=[0, 1]):
            arr = np.asarray(grp)
            pa[(w, sl)] = float(np.mean(arr > ACT_THR))
            pm[(w, sl)] = float(np.mean(arr))
        return pa, pm

    def _feat(self, s_hist, a_hist, act_hist, ts, pa, pm, mode='short'):
        """mode='short'：近期状态特征；mode='long'：周期特征。"""
        sl = ts.hour * 4 + ts.minute // 15
        wd = ts.weekday()
        n = len(s_hist)
        base = [np.sin(2*np.pi*sl/96), np.cos(2*np.pi*sl/96),
                np.sin(2*np.pi*wd/7), np.cos(2*np.pi*wd/7),
                ts.month/12.0, ts.day/31.0, float(wd < 5)]
        if mode == 'short':
            extra = ([s_hist[-L] if n >= L else 0.0 for L in [1, 2, 3, 4, 8, 16]]
                     + [a_hist[-L] if n >= L else 0.0 for L in [1, 2, 4]]
                     + [act_hist[-L] if n >= L else 0.0 for L in [1, 2, 4]]
                     + [float(np.sum(act_hist[-R:])) if R <= n else 0.0 for R in [8, 16]]
                     + [float(np.mean(s_hist[-R:])) if R <= n else 0.0 for R in [4, 8, 16]])
        else:
            extra = ([s_hist[-L] if n >= L else 0.0 for L in [96, 192, 336, 672]]
                     + [float(np.mean(s_hist[-R:])) if R <= n else 0.0 for R in [96, 336]]
                     + [pa.get((wd, sl), 0.0), pm.get((wd, sl), 0.0)])
        return np.array(base + extra, dtype=float)

    def _fit_one(self, series_list, active_list, mode):
        X, yb, ym = [], [], []
        for series, act in zip(series_list, active_list):
            s = series.values.astype(float)
            scale = max(s.max(), 1e-3)
            sn = s / scale
            a = (sn > ACT_THR / scale).astype(float)
            actn = act.values.astype(float) / max(act.values.max(), 1.0)
            pa, pm = self._profiles(series)
            idx = series.index
            for t in range(96, len(sn)):
                X.append(self._feat(sn[:t], a[:t], actn[:t], idx[t], pa, pm, mode))
                yb.append(int(a[t])); ym.append(sn[t])
        X = np.array(X); yb = np.array(yb); ym = np.array(ym)
        clf = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31,
                                 min_child_samples=20, random_state=42, n_jobs=4, verbose=-1)
        clf.fit(X, yb)
        actmask = yb == 1
        reg = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=31,
                                min_child_samples=20, random_state=42, n_jobs=4, verbose=-1)
        reg.fit(X[actmask], ym[actmask])
        return clf, reg

    def fit(self, train_series_list, train_active_list=None, **kw):
        if train_active_list is None:
            train_active_list = [pd.Series(np.zeros(len(s)), index=s.index) for s in train_series_list]
        self.prof = {i: self._profiles(s) for i, s in enumerate(train_series_list)}
        self.short_clf, self.short_reg = self._fit_one(train_series_list, train_active_list, 'short')
        self.long_clf, self.long_reg = self._fit_one(train_series_list, train_active_list, 'long')
        return self

    def _predict_mode(self, series, act, clf, reg, mode, n_horizon, fc_start):
        pa, pm = self._profiles(series)
        s = series.values.astype(float)
        scale = max(s.max(), 1e-3)
        sn = s / scale
        a = (sn > ACT_THR / scale).astype(float)
        actn = act.values.astype(float) / max(act.values.max(), 1.0)
        se, ae, ace = list(sn), list(a), list(actn)
        if fc_start is None:
            fc_start = series.index[-1] + STEP
        fidx = pd.date_range(fc_start, fc_start + STEP * (n_horizon - 1), freq='15min')
        preds = []
        for ts in fidx:
            f = self._feat(np.array(se), np.array(ae), np.array(ace), ts, pa, pm, mode).reshape(1, -1)
            p = float(clf.predict_proba(f)[0, 1])
            m = max(0.0, float(reg.predict(f)[0]))
            v = p * m * scale
            preds.append(v); se.append(p * m); ae.append(p); ace.append(p)
        return np.array(preds)

    def predict(self, series, active=None, fc_start=None, n_horizon=NHORIZON):
        if active is None:
            active = pd.Series(np.zeros(len(series)), index=series.index)
        short_pred = self._predict_mode(series, active, self.short_clf, self.short_reg, 'short', 16, fc_start)
        long_pred = self._predict_mode(series, active, self.long_clf, self.long_reg, 'long', n_horizon, fc_start)
        if n_horizon <= 16:
            return short_pred[:n_horizon]
        return np.concatenate([short_pred, long_pred[16:]])


# ============ 深度学习公共训练函数 ============
def _train_dl(model, X, Y, epochs=30, lr=1e-3, batch=64, device='cpu', loss_fn=None):
    Xt = torch.tensor(X, dtype=torch.float32, device=device)
    Yt = torch.tensor(Y, dtype=torch.float32, device=device)
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    if loss_fn is None:
        loss_fn = nn.MSELoss()
    n = len(Xt)
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n)
        tot = 0.0
        for j in range(0, n, batch):
            idx = perm[j:j + batch]
            xb, yb = Xt[idx], Yt[idx]
            out = model(xb)
            loss = loss_fn(out, yb)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(xb)
        if (ep + 1) % 10 == 0:
            print(f"    [{model.__class__.__name__}] epoch {ep+1}/{epochs} loss={tot/n:.5f}")
    model.eval()
    return model


def _predict_dl(model, series, in_len, out_len, fc_start, device='cpu'):
    s = series.values.astype(float)
    scale = max(s.max(), 1e-3)
    sn = s / scale
    x = torch.tensor(sn[-in_len:][None, :], dtype=torch.float32, device=device)
    with torch.no_grad():
        out = model(x)[0].cpu().numpy()
    return np.maximum(out, 0.0) * scale


# ============ 模型 2：PatchTST ============
class _PatchTST(nn.Module):
    """PatchTST（ICLR 2023）：切 patch → embedding → 位置编码 → Transformer Encoder → 预测头。"""
    def __init__(self, in_len=672, out_len=384, patch_len=16, stride=8, d_model=128,
                 n_heads=8, n_layers=3, dropout=0.1):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.n_patches = (in_len - patch_len) // stride + 1
        self.patch_embed = nn.Linear(patch_len, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, self.n_patches, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(d_model, n_heads, dim_feedforward=d_model * 4,
                                           dropout=dropout, batch_first=True, activation='gelu')
        self.encoder = nn.TransformerEncoder(layer, n_layers)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(self.n_patches * d_model, out_len)

    def forward(self, x):
        # x: (B, in_len)
        patches = x.unfold(1, self.patch_len, self.stride)  # (B, n_patches, patch_len)
        z = self.patch_embed(patches) + self.pos_embed
        z = self.encoder(z)
        z = self.drop(z).reshape(x.shape[0], -1)
        return self.head(z)


@register('patchtst')
class PatchTSTModel:
    def __init__(self, in_len=672, out_len=384, patch_len=16, stride=8, d_model=64,
                 n_heads=4, n_layers=2, dropout=0.1, epochs=30, lr=1e-3, **kw):
        self.in_len, self.out_len = in_len, out_len
        self.model = _PatchTST(in_len, out_len, patch_len, stride, d_model, n_heads, n_layers, dropout)
        self.epochs, self.lr = epochs, lr

    def fit(self, train_series_list, **kw):
        X, Y = collect_windows(train_series_list, self.in_len, self.out_len)
        print(f"    PatchTST 窗口数={len(X)}")
        self.model = _train_dl(self.model, X, Y, self.epochs, self.lr)
        return self

    def predict(self, series, fc_start=None, n_horizon=None):
        return _predict_dl(self.model, series, self.in_len, self.out_len, fc_start)


# ============ 模型 3：TimesNet ============
class _TimesBlock(nn.Module):
    """TimesNet 核心块：FFT 找 top-k 周期，1D→2D 重排，2D 卷积，加权融合。"""
    def __init__(self, seq_len, d_model=32, top_k=3):
        super().__init__()
        self.seq_len = seq_len
        self.top_k = top_k
        self.conv = nn.Sequential(
            nn.Conv2d(1, d_model, kernel_size=(3, 3), padding=(1, 1)),
            nn.GELU(),
            nn.Conv2d(d_model, 1, kernel_size=(3, 3), padding=(1, 1)),
        )

    def forward(self, x):
        # x: (B, seq_len)
        B, T = x.shape
        xf = torch.fft.rfft(x, dim=1)
        freq = xf.abs().mean(0)
        freq[0] = 0
        k = min(self.top_k, T // 2 - 1)
        if k <= 0:
            return x
        _, top = torch.topk(freq, k)
        top = top.detach().cpu().numpy()
        period = np.where(top > 0, T // top, 2)
        period = np.maximum(period, 2)
        weight = freq[top]  # (k,) 每个周期的振幅权重
        res = []
        for i in range(k):
            p = int(period[i])
            if T % p != 0:
                length = ((T // p) + 1) * p
                pad = torch.zeros(B, length - T, device=x.device)
                out = torch.cat([x, pad], dim=1)
            else:
                length = T
                out = x
            out = out.reshape(B, length // p, p)          # (B, h, p)
            out = out.unsqueeze(1)                        # (B, 1, h, p) 作为单通道 2D
            out = self.conv(out)                          # 2D 卷积
            out = out.squeeze(1).reshape(B, length)       # 回到 1D
            res.append(out[:, :T])
        res = torch.stack(res, dim=-1)                    # (B, T, k)
        w = torch.softmax(weight, dim=0).unsqueeze(0).unsqueeze(0)  # (1, 1, k)
        fused = (res * w).sum(-1)                         # (B, T)
        return fused + x


class _TimesNet(nn.Module):
    def __init__(self, in_len=672, out_len=384, d_model=32, top_k=3, n_layers=2):
        super().__init__()
        self.in_len = in_len
        self.blocks = nn.ModuleList([_TimesBlock(in_len, d_model, top_k) for _ in range(n_layers)])
        self.head = nn.Linear(in_len, out_len)

    def forward(self, x):
        # x: (B, in_len)
        h = x
        for blk in self.blocks:
            h = blk(h)
        return self.head(h)


@register('timesnet')
class TimesNetModel:
    def __init__(self, in_len=672, out_len=384, d_model=32, top_k=3, n_layers=2,
                 epochs=30, lr=1e-3, **kw):
        self.in_len, self.out_len = in_len, out_len
        self.model = _TimesNet(in_len, out_len, d_model, top_k, n_layers)
        self.epochs, self.lr = epochs, lr

    def fit(self, train_series_list, **kw):
        X, Y = collect_windows(train_series_list, self.in_len, self.out_len)
        print(f"    TimesNet 窗口数={len(X)}")
        self.model = _train_dl(self.model, X, Y, self.epochs, self.lr)
        return self

    def predict(self, series, fc_start=None, n_horizon=None):
        return _predict_dl(self.model, series, self.in_len, self.out_len, fc_start)
