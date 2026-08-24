"""充电功率预测交互界面（Flask，多模型）
- GET  /                : 首页（选示例 case 或上传 CSV，选模型，设预测起点）
- POST /predict         : 表单提交，出结果页（图表+表格+下载）
- GET  /api/predict     : 程序接口，?case=001&model=lightgbm 或上传文件，返回 JSON
依赖（venv）: flask, lightgbm, pandas, numpy, matplotlib, joblib, torch
运行: python app.py   -> http://127.0.0.1:5000
"""
import os, io, base64, json, csv, sys, socket, re
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from flask import Flask, request, render_template_string, send_file, jsonify, Response

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import model_zoo as mz
import importlib.util as _ilu
_p_spec = _ilu.spec_from_file_location('professor_model_zoo',
                                       os.path.join(os.path.dirname(os.path.abspath(__file__)), 'professor_model_zoo.py'))
pmz = _ilu.module_from_spec(_p_spec)
_p_spec.loader.exec_module(pmz)

BASE = os.path.dirname(os.path.abspath(__file__))
# 数据目录：优先环境变量 CHARGING_DATA_DIR，默认项目同级目录（可移植）
DATA_DIR = os.environ.get('CHARGING_DATA_DIR',
                          os.path.join(os.path.dirname(BASE), 'dataset'))
DATA = os.path.join(DATA_DIR, 'data')
MODELS_DIR = os.path.join(BASE, 'models_zoo')

MODEL_CHOICES = {
    'allzero': 'All-Zero 基线（全零常数）',
    'slotmedian': 'Slot-Median（同时段历史中位数）',
    'lightgbm': 'LightGBM 两阶段（活动分类 + 幅度回归）',
    'demmfl': 'DEMMFL（运营模式划分 + Lasso-Ridge）',
    'gru': 'GRU（2 层门控循环单元）',
    'patchtst': 'PatchTST（ICLR 2023）',
    'timesnet': 'TimesNet（ICLR 2023）',
}

app = Flask(__name__)

# ---------- 训练序列（用于按需训练）与模型缓存 ----------
_TRAIN_SERIES = None
_CACHE = {}


def get_train_series():
    global _TRAIN_SERIES
    if _TRAIN_SERIES is None:
        power_list, active_list = [], []
        for f in sorted(os.listdir(DATA)):
            if re.fullmatch(r'case_\d+\.csv', f):
                p, a = mz.load_series(os.path.join(DATA, f))
                power_list.append(p); active_list.append(a)
        _TRAIN_SERIES = (power_list, active_list)
    return _TRAIN_SERIES


MY_MODELS = {'lightgbm', 'patchtst', 'timesnet'}

# 模型就绪状态：启动预热完成后置 True；预热完成前收到 /predict 返回 MODEL_NOT_READY
_MODELS_READY = False


def get_model(name):
    if name not in MY_MODELS:
        return None  # 教授模型/基线，predict_zoo 里现场处理
    if name not in _CACHE:
        power_list, active_list = get_train_series()
        _CACHE[name] = mz.train_and_cache(name, power_list, MODELS_DIR, active_list)
    return _CACHE[name]


def build_gf_cal_from_df(df, fc_start):
    """从 28 字段 DataFrame 构造教授的 gf（历史网格功率+日历）和 cal（预测日历，动态起点）。"""
    df = df.copy()
    df['report_time'] = pd.to_datetime(df['report_time'])
    if 'date' not in df.columns or df['date'].isna().all():
        df['date'] = df['report_time'].dt.strftime('%Y-%m-%d')
    df['date'] = pd.to_datetime(df['date'])
    df['ts_15min'] = df['report_time'].dt.floor('15min')
    gun_avg = df.groupby(['ts_15min', 'connector_id'])['power_w'].mean().reset_index()
    grid_power = gun_avg.groupby('ts_15min')['power_w'].sum().reset_index()
    grid_power.columns = ['timestamp', 'power_w']
    grid_power['power_kw'] = grid_power['power_w'] / 1000.0
    full_range = pd.date_range(grid_power['timestamp'].min(), fc_start - pd.Timedelta(minutes=15), freq='15min')
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
    ts = pd.date_range(fc_start, fc_start + pd.Timedelta(minutes=15 * 383), freq='15min')
    cal = pd.DataFrame({'timestamp': ts})
    cal['hour'] = cal['timestamp'].dt.hour
    cal['minute_of_day'] = cal['timestamp'].dt.hour * 4 + cal['timestamp'].dt.minute // 15
    cal['dayofweek'] = cal['timestamp'].dt.dayofweek
    cal['is_workday'] = (cal['dayofweek'] < 5).astype(int)
    cal['day_of_month'] = cal['timestamp'].dt.day
    return gf, cal


def build_gf_cal(csv_path, fc_start):
    """从 CSV 文件路径构造 gf/cal（UI 上传场景用）。"""
    df = pd.read_csv(csv_path, parse_dates=['report_time'])
    return build_gf_cal_from_df(df, fc_start)


def gru_fast(gf, cal, epochs=30):
    """2 层 GRU（30 epoch，界面用加速版）。"""
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    torch.set_num_threads(1)

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

    g = gf.copy()
    g['hour_sin'] = np.sin(2*np.pi*g['timestamp'].dt.hour/24)
    g['hour_cos'] = np.cos(2*np.pi*g['timestamp'].dt.hour/24)
    g['dow_sin'] = np.sin(2*np.pi*g['timestamp'].dt.dayofweek/7)
    g['dow_cos'] = np.cos(2*np.pi*g['timestamp'].dt.dayofweek/7)
    cols = ['power_kw', 'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos', 'is_workday', 'temperature_c']
    for c in cols:
        if c not in g.columns: g[c] = 0
    feats = g[cols].values.astype(np.float32)
    targets = g['power_kw'].values.astype(np.float32)
    f_mean, f_std = feats.mean(0), feats.std(0) + 1e-8
    feats_n = (feats - f_mean) / f_std
    t_max = max(targets.max(), 1e-8)
    targets_n = targets / t_max
    ds = SeqDataset(feats_n, targets_n, 96)
    vs = min(96*4, len(ds)//5)
    ts_size = len(ds) - vs
    train_dl = DataLoader(torch.utils.data.Subset(ds, range(ts_size)), batch_size=64, shuffle=True)
    model = GRUModel(len(cols))
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()
    for ep in range(epochs):
        model.train()
        for xb, yb in train_dl:
            opt.zero_grad(); loss = loss_fn(model(xb), yb.squeeze(-1))
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
    model.eval()
    window = list(feats_n[-96:])
    preds = []
    with torch.no_grad():
        for i in range(384):
            x = torch.FloatTensor([window[-96:]])
            p = max(0, model(x).item() * t_max)
            preds.append(p)
            ts = pd.Timestamp(cal['timestamp'].iloc[i])
            nf = np.array([p, np.sin(2*np.pi*ts.hour/24), np.cos(2*np.pi*ts.hour/24),
                           np.sin(2*np.pi*ts.dayofweek/7), np.cos(2*np.pi*ts.dayofweek/7),
                           1.0 if ts.dayofweek < 5 else 0.0, g['temperature_c'].iloc[-1]], dtype=np.float32)
            window.append((nf - f_mean) / f_std)
    return np.array(preds)


def predict_zoo(model_name, model, csv_path, fc_start=None):
    """预测，返回 (pred_df, meta)。支持 7 个模型（我方 3 + 教授 4）。"""
    s, act = mz.load_series(csv_path)
    if fc_start is None:
        fc_start = s.index[-1] + mz.STEP
    # 我方模型（pooled 缓存训练）
    if model_name in ('lightgbm', 'patchtst', 'timesnet'):
        if model_name == 'lightgbm':
            pred = model.predict(s, active=act, fc_start=fc_start)
        else:
            pred = model.predict(s, fc_start=fc_start)
    # 教授模型 + All-Zero（逐 case 现场训练）
    else:
        gf, cal = build_gf_cal(csv_path, fc_start)
        if model_name == 'allzero':
            pred = np.zeros(384)
        elif model_name == 'slotmedian':
            pred = pmz.baseline_median(gf, cal)
        elif model_name == 'demmfl':
            pred = pmz.demmfl_lasso_ridge(gf, cal)
        elif model_name == 'gru':
            pred = gru_fast(gf, cal)
        else:
            pred = np.zeros(384)
    pred = np.asarray(pred, dtype=float)[:384]
    rows = []
    for h, hs in mz.HORIZONS.items():
        for j in range(hs):
            rows.append({'forecast_horizon': h, 'step_index': j + 1,
                         'forecast_timestamp': (fc_start + mz.STEP * j).strftime('%Y-%m-%d %H:%M:%S'),
                         'power_kw': round(float(pred[j]), 4)})
    pred_df = pd.DataFrame(rows)
    meta = {
        'csv': os.path.basename(csv_path),
        'model': model_name,
        'model_label': MODEL_CHOICES.get(model_name, model_name),
        'input_start': str(s.index[0]), 'input_end': str(s.index[-1]),
        'fc_start': str(fc_start), 'n_input_steps': int(len(s)),
        'zero_rate_input': round(float((s.values <= mz.ACT_THR).mean()), 4),
        'pred_peak_kw': round(float(pred.max()), 3),
        'pred_mean_kw': round(float(pred.mean()), 3),
        'pred_active_rate': round(float((pred > 0.5).mean()), 4),
        'pred_energy_kwh': round(float(pred.sum()) / 4.0, 2),
    }
    return pred_df, meta


# ---------- 图表 ----------
def make_chart(pred_df, fc_start):
    df = pred_df[pred_df['forecast_horizon'] == '4d'].reset_index(drop=True)
    t = pd.to_datetime(df['forecast_timestamp'])
    v = df['power_kw'].values
    fig, ax = plt.subplots(figsize=(11, 4.2))
    ax.plot(t, v, color='#e8590c', lw=1.3, label='Predicted grid power (kW)')
    ax.fill_between(t, v, color='#e8590c', alpha=0.12)
    ax.set_xlabel(f'Forecast time ({fc_start.date()} → 4 天后)')
    ax.set_ylabel('Grid charging power (kW)')
    ax.set_title('4-day grid charging power forecast')
    ax.grid(True, alpha=0.3)
    peak = v.max()
    if peak > 0:
        ax.axhline(peak, color='#1971c2', ls='--', lw=0.8, label=f'Peak {peak:.1f} kW')
    ax.legend(loc='upper right', fontsize=8)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=110)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode('ascii')


# ---------- 首页模板 ----------
INDEX_HTML = """
<!doctype html><html lang="zh"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>电网充电功率预测</title>
<style>
 body{font-family:-apple-system,Segoe UI,Roboto,'PingFang SC','Microsoft YaHei',sans-serif;
   max-width:880px;margin:0 auto;padding:28px;color:#222;background:#fafafa}
 h1{color:#e8590c;font-size:24px;margin-bottom:4px}
 .sub{color:#666;font-size:13px;margin-bottom:20px}
 .card{background:#fff;border:1px solid #eee;border-radius:10px;padding:18px;margin-bottom:16px;box-shadow:0 1px 3px rgba(0,0,0,.04)}
 label{font-weight:600;font-size:14px;display:block;margin:10px 0 6px}
 select,input[type=date],input[type=file]{width:100%;padding:9px;border:1px solid #ccc;border-radius:7px;font-size:14px}
 button{margin-top:16px;background:#e8590c;color:#fff;border:0;padding:11px 22px;border-radius:8px;font-size:15px;cursor:pointer}
 button:hover{background:#d24d00}
 .note{font-size:12px;color:#888;margin-top:8px}
</style></head><body>
<h1>电网级充电功率预测</h1>
<div class="sub">7 模型可选（All-Zero / Slot-Median / LightGBM / DEMMFL / GRU / PatchTST / TimesNet）· 15 分钟颗粒度 · 4 尺度输出</div>
<form method="post" action="/predict_ui" enctype="multipart/form-data">
  <div class="card">
    <label>① 选择模型（共 7 个）</label>
    <select name="model">
      {% for k, v in model_choices.items() %}<option value="{{k}}">{{v}}</option>{% endfor %}
    </select>
    <label>② 选择预测对象</label>
    <select name="mode">
      <option value="sample">使用示例数据（15 分钟全量数据集的网格 case）</option>
      <option value="upload">上传我自己的 case CSV（字段同 data/case_*.csv）</option>
    </select>
    <label>③ 示例 case 编号（选"使用示例数据"时生效）</label>
    <select name="case">
      {% for c in cases %}<option value="{{c}}">case_{{c}}</option>{% endfor %}
    </select>
    <label>④ 上传 CSV（选"上传"时生效）</label>
    <input type="file" name="file" accept=".csv">
    <div class="note">CSV 需含列：report_time, connector_id, power_w（其余可缺省）</div>
    <label>⑤ 预测起点日期（留空 = 自动取数据末尾）</label>
    <input type="date" name="fc_start" value="">
    <button type="submit">运行预测</button>
  </div>
</form>
<div class="card">
  <div class="note">程序接口：<code>GET /api/predict?case=001&model=lightgbm</code>（model 可为 allzero/slotmedian/lightgbm/demmfl/gru/patchtst/timesnet）或 <code>POST /api/predict</code>（multipart 文件字段 file + 表单 model）</div>
</div>
</body></html>
"""

RESULT_HTML = """
<!doctype html><html lang="zh"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>预测结果</title>
<style>
 body{font-family:-apple-system,Segoe UI,Roboto,'PingFang SC','Microsoft YaHei',sans-serif;max-width:880px;margin:0 auto;padding:24px;color:#222;background:#fafafa}
 h1{color:#e8590c;font-size:22px}.sub{color:#666;font-size:13px;margin-bottom:14px}
 .card{background:#fff;border:1px solid #eee;border-radius:10px;padding:18px;margin-bottom:16px;box-shadow:0 1px 3px rgba(0,0,0,.04)}
 img{max-width:100%;border:1px solid #eee;border-radius:8px}
 table{border-collapse:collapse;width:100%;font-size:13px;margin-top:8px}
 th,td{border:1px solid #eee;padding:7px 9px;text-align:center}
 th{background:#fff4e6;color:#c2410c}
 .kv{display:flex;flex-wrap:wrap;gap:10px;margin-top:8px}
 .pill{background:#fff4e6;border:1px solid #ffd8a8;border-radius:8px;padding:8px 12px;font-size:13px}
 .pill b{color:#c2410c}
 a.btn{display:inline-block;margin-top:10px;background:#1971c2;color:#fff;padding:9px 16px;border-radius:7px;text-decoration:none;font-size:14px}
 a.back{color:#1971c2;text-decoration:none;font-size:13px}
</style></head><body>
<h1>预测结果 · {{ meta.csv }}（{{ meta.model_label }}）</h1>
<div class="sub">预测起点 {{ meta.fc_start }} · 输入历史 {{ meta.input_start }} → {{ meta.input_end }}（{{ meta.n_input_steps }} 步，零负荷占比 {{ (meta.zero_rate_input*100)|round(0) }}%）</div>
<div class="card"><img src="data:image/png;base64,{{ chart }}"></div>
<div class="card">
  <div class="kv">
    <div class="pill">峰值功率 <b>{{ meta.pred_peak_kw }} kW</b></div>
    <div class="pill">均值功率 <b>{{ meta.pred_mean_kw }} kW</b></div>
    <div class="pill">预测活跃率 <b>{{ (meta.pred_active_rate*100)|round(1) }}%</b></div>
    <div class="pill">预测总电量 <b>{{ meta.pred_energy_kwh }} kWh</b></div>
  </div>
  <table><tr><th>尺度</th><th>步数</th><th>时长</th><th>该尺度均值(kW)</th><th>该尺度峰值(kW)</th></tr>
  {% for h in horizons %}<tr><td>{{ h.name }}</td><td>{{ h.steps }}</td><td>{{ h.dur }}</td><td>{{ h.mean }}</td><td>{{ h.peak }}</td></tr>{% endfor %}
  </table>
  <a class="btn" href="/download?case={{ case }}">下载预测 CSV（ground_truth 格式）</a>
</div>
<div class="card"><a class="back" href="/">← 返回</a></div>
</body></html>
"""


def build_result(case_label, pred_df, meta):
    chart = make_chart(pred_df, pd.Timestamp(meta['fc_start']))
    horizons = []
    for name, steps in mz.HORIZONS.items():
        sub = pred_df[pred_df['forecast_horizon'] == name]['power_kw'].values
        horizons.append({'name': name, 'steps': steps,
                         'dur': {'2h': '2 小时', '4h': '4 小时', '1d': '1 天', '4d': '4 天'}[name],
                         'mean': round(float(sub.mean()), 3), 'peak': round(float(sub.max()), 3)})
    return render_template_string(RESULT_HTML, meta=meta, chart=chart, horizons=horizons, case=case_label)


HOST = '0.0.0.0'  # 默认 0.0.0.0，Docker 部署必需


def _err(code, msg, case_id=None):
    """生成标准错误响应。"""
    return jsonify({'output': None, 'case_id': case_id, 'error_code': code, 'error_msg': msg}), 200


def _ok(output, case_id):
    """生成标准成功响应。"""
    return jsonify({'output': output, 'case_id': case_id, 'error_code': None, 'error_msg': None}), 200


def _input_to_df(input_data):
    """把 input 字段解析成 28 字段 DataFrame。支持三种入参：
    A) CSV 文本字符串（正式规范：28 字段表头 + 数据行）→ io.StringIO + read_csv，完整解析不截断
    B) dict 含 samples（列表 of dict，旧兼容）
    C) dict 含 power_kw（15min 功率序列，无时间戳，构造占位 df）
    返回 DataFrame 或抛 ValueError。
    """
    # A) CSV 文本字符串（正式规范主格式）
    if isinstance(input_data, str):
        txt = input_data.strip()
        if not txt:
            raise ValueError('empty-input-csv')
        df = pd.read_csv(io.StringIO(txt))
        if len(df) == 0:
            raise ValueError('no-rows-in-csv')
        return df
    # B) dict: samples
    if isinstance(input_data, dict) and 'samples' in input_data and isinstance(input_data['samples'], list):
        return pd.DataFrame(input_data['samples'])
    # C) dict: power_kw（纯功率序列，无时间戳）
    if isinstance(input_data, dict) and 'power_kw' in input_data and isinstance(input_data['power_kw'], list):
        arr = np.array(input_data['power_kw'], dtype=float)
        idx = pd.date_range(end=pd.Timestamp.now().normalize(), periods=len(arr), freq='15min')
        return pd.DataFrame({'report_time': idx.strftime('%Y-%m-%d %H:%M:%S'),
                             'connector_id': ['C1'] * len(arr), 'power_w': arr * 1000.0})
    raise ValueError('input must be CSV text, or dict with samples / power_kw')


def _df_to_series(df):
    """28 字段 DataFrame → 15min 网格级功率序列（带真实 DatetimeIndex）。"""
    for col in ('report_time', 'connector_id', 'power_w'):
        if col not in df.columns:
            raise ValueError(f'missing-column-in-input: {col}')
    d = df.copy()
    d['ts'] = pd.to_datetime(d['report_time'])
    d['w'] = d['ts'].dt.floor('15min')
    g = d.groupby(['w', 'connector_id'])['power_w'].mean().groupby('w').sum() / 1000.0
    if len(g) == 0:
        raise ValueError('no-valid-samples-after-aggregation')
    idx = pd.date_range(g.index.min(), g.index.max(), freq='15min')
    ser = pd.Series(0.0, index=idx)
    ser.update(g)
    return ser


def _df_to_gf_cal(df, fc_start):
    """28 字段 DataFrame → 教授模型的 gf/cal（逐 case 模型需要）。补齐缺失的日历/天气列。"""
    d = df.copy()
    d['report_time'] = pd.to_datetime(d['report_time'])
    if 'date' not in d.columns:
        d['date'] = d['report_time'].dt.strftime('%Y-%m-%d')
    if 'is_workday' not in d.columns:
        d['is_workday'] = (d['report_time'].dt.dayofweek < 5).astype(int)
    if 'is_major_holiday' not in d.columns:
        d['is_major_holiday'] = 0
    if 'temperature_c' not in d.columns:
        d['temperature_c'] = 20.0
    if 'rainfall_mm' not in d.columns:
        d['rainfall_mm'] = 0.0
    return build_gf_cal_from_df(d, fc_start)


@app.route('/health', methods=['GET'])
def health():
    """标准健康检查端点（正式规范 1.1.2.1）。规范要求返回 status=healthy + timestamp。"""
    return jsonify({'status': 'healthy',
                    'timestamp': datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')}), 200


def _predict_single(model_name, df, s, fc_start):
    """单模型预测，返回 384 步预测数组（kW）。"""
    if model_name in ('lightgbm', 'patchtst', 'timesnet'):
        model = get_model(model_name)
        if model_name == 'lightgbm':
            act = (s > mz.ACT_THR).astype(float)
            pred = model.predict(s, active=act, fc_start=fc_start)
        else:
            pred = model.predict(s, fc_start=fc_start)
    elif model_name == 'allzero':
        pred = np.zeros(384)
    else:
        gf, cal = _df_to_gf_cal(df, fc_start)
        if model_name == 'slotmedian':
            pred = pmz.baseline_median(gf, cal)
        elif model_name == 'demmfl':
            pred = pmz.demmfl_lasso_ridge(gf, cal)
        elif model_name == 'gru':
            pred = gru_fast(gf, cal)
        else:
            pred = np.zeros(384)
    return np.asarray(pred, dtype=float)[:384]


def _pred_to_csv(pred, fc_start):
    """预测数组 → output CSV 文本（4 字段表头 + 504 行，power_kw 保留 3 位小数）。"""
    lines = ['forecast_horizon,step_index,forecast_timestamp,power_kw']
    for h, hs in mz.HORIZONS.items():
        for j in range(hs):
            ts = (fc_start + mz.STEP * j).strftime('%Y-%m-%d %H:%M:%S')
            lines.append(f'{h},{j + 1},{ts},{float(pred[j]):.3f}')
    return '\n'.join(lines)


@app.route('/predict', methods=['POST'])
def predict_official():
    """标准推理端点（正式规范 1.1.2.2）。模型由启动环境变量 MODEL 决定（默认 demmfl）。
    请求：{input: "<CSV文本 28字段含表头>", case_id: "001"}（input 也可能是 dict 兼容）
    响应：{output: "<CSV文本 4字段含表头 504行>", case_id, error_code, error_msg}
    入参长度不做约束，完整解析不截断。

    本机对比模式：若启动时设置 ALL_MODELS=1，则 output 变为 {模型名: CSV文本} 的 7 模型对照。
    Docker 环境不设 ALL_MODELS，仍按规范单模型 + CSV。
    """
    model_name = os.environ.get('MODEL', 'demmfl')
    if model_name not in MODEL_CHOICES:
        model_name = 'demmfl'
    all_models = os.environ.get('ALL_MODELS', '0') == '1'
    if not _MODELS_READY:
        return _err('MODEL_NOT_READY', 'model weights still loading')
    try:
        body = request.get_json(force=True, silent=False)
    except Exception as e:
        return _err('INPUT_FORMAT_INVALID', f'invalid-json: {e}')
    if not isinstance(body, dict):
        return _err('INPUT_FORMAT_INVALID', 'body must be a JSON object')
    if 'case_id' not in body:
        return _err('INPUT_MISSING_FIELD', 'missing required field: case_id')
    case_id = str(body['case_id'])
    if 'input' not in body:
        return _err('INPUT_MISSING_FIELD', 'missing required field: input', case_id)
    inp = body['input']
    # 解析 input → 28 字段 DataFrame（CSV 文本 / samples / power_kw）
    try:
        df = _input_to_df(inp)
    except ValueError as e:
        return _err('INPUT_FORMAT_INVALID', str(e), case_id)
    try:
        s = _df_to_series(df)
        fc_start = s.index[-1] + mz.STEP
        if all_models:
            # 本机对比模式：同一输入跑 7 个模型，返回 {模型名: CSV 文本}
            results = {}
            for mn in MODEL_CHOICES.keys():
                try:
                    pred = _predict_single(mn, df, s, fc_start)
                    results[mn] = _pred_to_csv(pred, fc_start)
                except Exception as e:
                    results[mn] = f'ERROR: {e}'
            return _ok(results, case_id)
        # 规范单模型模式
        pred = _predict_single(model_name, df, s, fc_start)
        output_csv = _pred_to_csv(pred, fc_start)
        return _ok(output_csv, case_id)
    except Exception as e:
        return _err('MODEL_INFERENCE_ERROR', f'forward-failed: {e}', case_id)


@app.errorhandler(Exception)
def handle_unexpected(e):
    """规范错误码兜底：未捕获异常返回 SYSTEM_INTERNAL_ERROR，而非 Flask 默认 500。"""
    return _err('SYSTEM_INTERNAL_ERROR', f'unexpected error in predict: {e}')


@app.route('/')
def index():
    cases = sorted(f.replace('case_', '').replace('.csv', '') for f in os.listdir(DATA)
                   if re.fullmatch(r'case_\d+\.csv', f))
    return render_template_string(INDEX_HTML, cases=cases, model_choices=MODEL_CHOICES)


@app.route('/predict_ui', methods=['POST'])
def predict_ui():
    mode = request.form.get('mode', 'sample')
    model_name = request.form.get('model', 'lightgbm')
    if model_name not in MODEL_CHOICES:
        model_name = 'lightgbm'
    model = get_model(model_name)
    if mode == 'upload' and 'file' in request.files and request.files['file'].filename:
        f = request.files['file']
        path = os.path.join(BASE, 'uploads', f.filename)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        f.save(path)
        case_label = f.filename
    else:
        cid = request.form.get('case', '001')
        path = os.path.join(DATA, f'case_{cid}.csv')
        case_label = f'case_{cid}'
    fc_start = request.form.get('fc_start')
    fc_start = pd.Timestamp(fc_start) if fc_start else None
    pred_df, meta = predict_zoo(model_name, model, path, fc_start=fc_start)
    os.makedirs(os.path.join(BASE, 'predictions_ui'), exist_ok=True)
    pred_df.to_csv(os.path.join(BASE, 'predictions_ui', f'{case_label}_{model_name}.csv'), index=False)
    return build_result(case_label, pred_df, meta)


@app.route('/api/predict')
def api_predict():
    model_name = request.args.get('model', 'lightgbm')
    if model_name not in MODEL_CHOICES:
        model_name = 'lightgbm'
    model = get_model(model_name)
    cid = request.args.get('case')
    fc_start = request.args.get('fc_start')
    fc_start = pd.Timestamp(fc_start) if fc_start else None
    if cid:
        cid_key = f'{int(cid):03d}' if cid.isdigit() else str(cid)
        path = os.path.join(DATA, f'case_{cid_key}.csv')
    else:
        return jsonify({'error': 'provide ?case=001&model=lightgbm'}), 400
    pred_df, meta = predict_zoo(model_name, model, path, fc_start=fc_start)
    return jsonify({'meta': meta, 'predictions': pred_df.to_dict(orient='records')})


@app.route('/api/predict', methods=['POST'])
def api_predict_post():
    model_name = request.form.get('model', 'lightgbm')
    if model_name not in MODEL_CHOICES:
        model_name = 'lightgbm'
    model = get_model(model_name)
    if 'file' in request.files and request.files['file'].filename:
        f = request.files['file']
        path = os.path.join(BASE, 'uploads', f.filename)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        f.save(path)
    else:
        return jsonify({'error': 'no file'}), 400
    fc_start = request.form.get('fc_start')
    fc_start = pd.Timestamp(fc_start) if fc_start else None
    pred_df, meta = predict_zoo(model_name, model, path, fc_start=fc_start)
    return jsonify({'meta': meta, 'predictions': pred_df.to_dict(orient='records')})


@app.route('/download')
def download():
    case = request.args.get('case', 'case_001')
    model_name = request.args.get('model', 'lightgbm')
    path = os.path.join(BASE, 'predictions_ui', f'{case}_{model_name}.csv')
    if not os.path.exists(path):
        return 'not found', 404
    return send_file(path, as_attachment=True, download_name=f'{case}_{model_name}_forecast.csv')


if __name__ == '__main__':
    def free_port(preferred=8000, max_try=10):
        for p in range(preferred, preferred + max_try):
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            try:
                s.bind(('0.0.0.0', p)); s.close(); return p
            except OSError:
                continue
        return preferred
    get_train_series()  # 预热训练序列
    # 预热模型：ALL_MODELS=1 时加载全部 3 个 pooled 模型（逐 case 模型现场训练无需预热）；否则只加载默认模型
    try:
        all_models = os.environ.get('ALL_MODELS', '0') == '1'
        default_model = os.environ.get('MODEL', 'demmfl')
        if all_models:
            for mn in MY_MODELS:
                get_model(mn)
            print(f'[本机对比模式] 已加载全部 {len(MODEL_CHOICES)} 个模型（pooled 3 个预加载 + 逐 case 4 个现场）')
        elif default_model in MY_MODELS:
            get_model(default_model)
            print(f'模型就绪：默认模型={default_model}')
        else:
            print(f'模型就绪：默认模型={default_model}（逐 case 现场训练）')
        _MODELS_READY = True
    except Exception as e:
        print(f'[WARN] 模型预热失败: {e}，/predict 将返回 MODEL_NOT_READY')
    port = int(os.environ.get('PORT', free_port(8000)))  # 正式规范端口 8000
    host = os.environ.get('HOST', '0.0.0.0')  # Docker 部署必须 0.0.0.0
    print(f'充电功率预测界面已启动: http://{host}:{port}  (健康检查 /health  推理 /predict POST JSON)')
    app.run(host=host, port=port, debug=False)
