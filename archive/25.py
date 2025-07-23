# Synthetic vs. Real Data Evaluation (Enhanced)
# ------------------------------------------------
#   * Multi‑seasonal sinusoid + phase‑average smoother
#   * Stationary AR(p) chosen by AIC and shrunk to |φ|max < 0.95
#   * GARCH(1,1) conditional variance
#   * Student‑t heavy‑tailed innovations
#   * Bootstrap metrics + forecast TSTR/TRTS

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import periodogram, find_peaks
from scipy.stats import ks_2samp, entropy
from statsmodels.tsa.ar_model import AutoReg, ar_select_order
from statsmodels.tsa.seasonal import STL
from arch import arch_model
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, mean_squared_error
from statsmodels.tsa.stattools import acf
from IPython.display import display

# -------------------------------
# 1. Load & split data
csv_path = './data/weather/temp.csv'
df = pd.read_csv(csv_path, parse_dates=['date']).set_index('date')
y = df['T (degC)'].values.astype(float)
time = np.arange(len(y))
T = len(y)
train_end = int(0.8 * T)
y_train, y_hold = y[:train_end], y[train_end:]
print(f"Training on {train_end} pts, holding {T-train_end} pts")

# -------------------------------
# 2. Seasonality
# 2.1 Detect peaks
freqs, psd = periodogram(y_train)
peaks, _ = find_peaks(psd[freqs > 0], distance=10)
f_peak = freqs[freqs > 0][peaks]
power = psd[freqs > 0][peaks]
order = np.argsort(power)[::-1]
cum = np.cumsum(power[order])
keep = order[cum <= 0.95 * power.sum()]
if len(keep) < len(order):
    keep = np.append(keep, order[len(keep)])
periods_base = np.unique(np.round(1 / f_peak[keep]).astype(int))
# add harmonics
harm = []
for P in periods_base:
    for k in (2, 3):
        if P % k == 0:
            harm.append(P // k)
periods = np.sort(np.unique(np.concatenate([periods_base, harm])))
print("Periods used:", periods)

# 2.2 Fourier basis (keep first 10 harmonics of 24h)
if 24 not in periods:
    periods = np.insert(periods, 0, 24)
K = 10
extra = np.arange(2, K + 1) * 24
periods = np.sort(np.unique(np.concatenate([periods, extra])))

# 2.3 Fit sinusoids
X_full = np.column_stack(
    [np.sin(2 * np.pi * time / P) for P in periods] +
    [np.cos(2 * np.pi * time / P) for P in periods]
)
X_train = X_full[:train_end]
lr_sin = LinearRegression().fit(X_train, y_train)
season_sin_train = lr_sin.predict(X_train)

# 2.4 STL on residuals
primary = 24
stl = STL(y_train - season_sin_train, period=primary, robust=True).fit()
trend_stl = stl.trend
season_stl = stl.seasonal
phase = (time[:train_end] % primary).astype(int)
phase_avg = np.array([season_stl[phase == i].mean() for i in range(primary)])
season_phase_full = phase_avg[time % primary]
season_full = lr_sin.predict(X_full) + season_phase_full

# -------------------------------
# 3. Trend (linear)
trend_lr = LinearRegression().fit(time[:train_end].reshape(-1, 1), trend_stl)
trend_full = trend_lr.predict(time.reshape(-1, 1))

# -------------------------------
# 4. Residuals & AR(p)
resid_train = y_train - season_full[:train_end] - trend_full[:train_end]

# 4.1 Order selection with AIC
sel = ar_select_order(resid_train, maxlag=25, ic='aic')
lags = sel.ar_lags
print("Selected lags by AIC:", lags)

# Fit AR using those lags
ar_mod = AutoReg(resid_train, lags=lags, old_names=False).fit()
phi_raw = ar_mod.params[1:]
phi = 0.95 * phi_raw / np.max(np.abs(phi_raw))  # shrink for stability
p = len(phi)
print(f"AR({p}) with shrunk coeffs:", phi)

# 4.2 Fit GARCH(1,1) on AR residuals, disable automatic rescaling
am_res = arch_model(
    ar_mod.resid,
    vol='Garch', p=1, q=1,
    dist='normal',
    rescale=False
).fit(disp='off')

# -------------------------------
# 5. Package params
sdg_params = dict(
    periods=periods,
    lr_sin=lr_sin,
    phase_avg=phase_avg,
    primary=primary,
    trend_lr=trend_lr,
    phi=phi,
    p=p,
    garch_params=am_res.params,
    garch_model=am_res.model,  # store the ARCHModel itself
    df_t=5
)

# -------------------------------
# 6. Synthetic generator
def generate_synthetic(N, params, seed=None):
    rng = np.random.default_rng(seed)
    t = np.arange(N)
    X = np.column_stack(
        [np.sin(2 * np.pi * t / P) for P in params['periods']] +
        [np.cos(2 * np.pi * t / P) for P in params['periods']]
    )
    season = params['lr_sin'].predict(X) + params['phase_avg'][t % params['primary']]
    trend = params['trend_lr'].predict(t.reshape(-1, 1))

    # simulate via the ARCHModel
    sim_data = params['garch_model'].simulate(params['garch_params'], N)
    # correctly extract conditional volatility
    sigma = sim_data['volatility'].values

    eps = rng.standard_t(df=params['df_t'], size=N) * sigma
    x = np.zeros(N)
    for i in range(N):
        ar_part = (np.dot(params['phi'], x[i-params['p']:i][::-1])
                   if i >= params['p'] else 0)
        x[i] = trend[i] + season[i] + ar_part + eps[i]
    return x

# -------------------------------
# 7. Compare
syn = generate_synthetic(T, sdg_params, seed=42)
plt.figure(figsize=(12, 4))
plt.plot(df.index, y, label='Real')
plt.plot(df.index, syn, label='Synth', alpha=.7)
plt.legend()
plt.show()

# -------------------------------
# 8. Metrics & Forecasts
win = primary
nlags = 30
acf_l2 = lambda a, b: np.sum((acf(a, nlags=nlags) - acf(b, nlags=nlags))**2)

def spectral_kl(a, b):
    p = periodogram(a)[1]; q = periodogram(b)[1]
    return entropy(p/p.sum(), q/q.sum())

ks_stat = lambda a, b: ks_2samp(a, b).statistic

def clf_auc(a, b, w):
    X, Y = [], []
    for i in range(0, len(a)-w, w):
        X.append(a[i:i+w]); Y.append(0)
    for i in range(0, len(b)-w, w):
        X.append(b[i:i+w]); Y.append(1)
    X = np.array(X).reshape(-1, w); Y = np.array(Y)
    Xtr, Xte, Ytr, Yte = train_test_split(X, Y, test_size=.3, random_state=42)
    return roc_auc_score(Yte,
                         RandomForestClassifier(random_state=42)
                           .fit(Xtr, Ytr)
                           .predict_proba(Xte)[:,1]
                        )

B = 200
def boot(fn):
    vals = []
    for s in range(B):
        ys = generate_synthetic(T, sdg_params, seed=s)
        vals.append(fn(y_hold, ys[train_end:]))
    a = np.array(vals)
    return a.mean(), np.percentile(a, [2.5, 97.5])

metrics = {
    'ACF L2': acf_l2,
    'Spectral KL': spectral_kl,
    'KS': ks_stat,
    'AUC': lambda r, s: clf_auc(r, s, win)
}
rows = []
for k, f in metrics.items():
    m, ci = boot(f)
    rows.append((k, m, ci[0], ci[1]))
summary = pd.DataFrame(rows, columns=['Metric', 'Mean', 'CI_L', 'CI_U'])
print(summary)

# Forecast tests
def tstr(real_hold, synth):
    Xs, ys = [], []
    for i in range(train_end-win):
        Xs.append(synth[i:i+win]); ys.append(synth[i+win])
    model = Ridge().fit(np.array(Xs), np.array(ys))
    preds = [model.predict(real_hold[i-win:i].reshape(1, -1))[0]
             for i in range(win, len(real_hold))]
    return mean_squared_error(real_hold[win:], preds)

def trts(real_train, synth_hold):
    Xr, yr = [], []
    for i in range(train_end-win):
        Xr.append(real_train[i:i+win]); yr.append(real_train[i+win])
    model = Ridge().fit(np.array(Xr), np.array(yr))
    preds = [model.predict(synth_hold[i-win:i].reshape(1, -1))[0]
             for i in range(win, len(synth_hold))]
    return mean_squared_error(synth_hold[win:], preds)

print("TSTR", tstr(y_hold, syn), "TRTS", trts(y_train, syn[train_end:]))
