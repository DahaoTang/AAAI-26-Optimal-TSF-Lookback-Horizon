# Synthetic vs. Real Data Evaluation
# ----------------------------------
# This script fits an enhanced synthetic data generator (SDG) to real-world
# temperature data, automatically extracting multiple seasonalities (with harmonics),
# smoothing seasonal residuals via cycle-phase averaging, fitting AR on residuals,
# and evaluating representation with bootstrapped metrics.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import periodogram, find_peaks
from scipy.stats import ks_2samp, entropy
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.seasonal import STL
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, mean_squared_error
from statsmodels.tsa.stattools import acf
from IPython.display import display

# 1. Load & Preprocess Real Data
csv_path = './data/weather/temp.csv'
df = pd.read_csv(csv_path, parse_dates=['date']).set_index('date')
y = df['T (degC)'].values
time = np.arange(len(y))

# Split for estimation and hold-out
T = len(y)
train_end = int(0.8 * T)
y_train, y_hold = y[:train_end], y[train_end:]
print(f"Training on first {train_end} points, holding out {T-train_end} points.")

# 2. Seasonal & Trend Estimation
# 2.1 Detect multiple seasonal periods via periodogram
freqs, psd = periodogram(y_train)
positive = freqs > 0
peaks, _ = find_peaks(psd[positive], distance=10)
f_peaks = freqs[positive][peaks]
psd_peaks = psd[positive][peaks]
order = np.argsort(psd_peaks)[::-1]
cum = np.cumsum(psd_peaks[order])
keep = order[cum <= 0.95 * psd_peaks.sum()]
if len(keep) < len(order):
    keep = np.append(keep, order[len(keep)])
f_sel = f_peaks[keep]
base_periods = np.unique(np.round(1 / f_sel).astype(int))
# add divisors for k=2,3 harmonics
harmonics = []
for P in base_periods:
    for k in (2,3):
        if P % k == 0:
            harmonics.append(P//k)
harmonics = np.unique(harmonics)
periods = np.sort(np.concatenate([base_periods, harmonics]))
J = len(periods)
print(f"Selected seasonal periods (with harmonics): {periods}")

# 2.2 Multi-sinusoidal seasonal fit
X_sin_full = np.column_stack(
    [np.sin(2*np.pi*time/P) for P in periods] + [np.cos(2*np.pi*time/P) for P in periods]
)
X_sin_train = X_sin_full[:train_end]
lr_sin = LinearRegression().fit(X_sin_train, y_train)
season_sin_train = lr_sin.predict(X_sin_train)

# 2.3 STL on residuals of sinusoidal fit
primary = periods[0]
stl = STL(y_train - season_sin_train, period=primary, robust=True).fit()
trend_stl = stl.trend
season_stl = stl.seasonal

# 2.4 Smooth seasonal residuals by cycle-phase averaging
day_phase = (time[:train_end] % primary).astype(int)
seasonal_means = np.array([
    season_stl[day_phase==i].mean() for i in range(primary)
])
season_smooth_full = seasonal_means[time % primary]

# Combined seasonal estimate
y_season_full = lr_sin.predict(X_sin_full) + season_smooth_full

# 2.5 Trend via linear regression on STL trend
trend_lr = LinearRegression().fit(time[:train_end].reshape(-1,1), trend_stl)
trend_full = trend_lr.predict(time.reshape(-1,1))

# 2.6 Residual & AR(p)
y_detr = y_train - y_season_full[:train_end] - trend_full[:train_end]
max_lag = 10
ar_res = AutoReg(y_detr, lags=max_lag, old_names=False).fit()
phi = ar_res.params[1:]
noise_std = ar_res.resid.std()
print(f"Fitted AR({max_lag}) coefficients: {phi}")

# Package SDG parameters
sdg_params = {
    'periods': periods,
    'lr_sin': lr_sin,
    'seasonal_means': seasonal_means,
    'primary': primary,
    'trend_lr': trend_lr,
    'phi': phi,
    'noise_std': noise_std
}

# 3. Synthetic Data Generator
def generate_synthetic(N, params, seed=None):
    rng = np.random.default_rng(seed)
    t_idx = np.arange(N)
    # sinusoidal
    X_full = np.column_stack(
        [np.sin(2*np.pi*t_idx/P) for P in params['periods']] +
        [np.cos(2*np.pi*t_idx/P) for P in params['periods']]
    )
    season1 = params['lr_sin'].predict(X_full)
    # cycle-phase
    season2 = params['seasonal_means'][t_idx % params['primary']]
    season = season1 + season2
    # trend
    trend = params['trend_lr'].predict(t_idx.reshape(-1,1))
    # AR noise\    
    p = len(params['phi'])
    x = np.zeros(N)
    eps = rng.standard_normal(N) * params['noise_std']
    for i in range(N):
        ar_term = np.dot(params['phi'], x[i-p:i][::-1]) if i>=p else 0
        x[i] = trend[i] + season[i] + ar_term + eps[i]
    return x

# 4. Visual comparison
y_synth = generate_synthetic(T, sdg_params, seed=42)
plt.figure(figsize=(12,4))
plt.plot(df.index, y, label='Real')
plt.plot(df.index, y_synth, label='Synthetic', alpha=0.7)
plt.legend()
plt.title('Real vs Synthetic Temperature Series')
plt.show()

# 5. Evaluation Metrics
window = periods[0]
nlags = 30

def acf_l2(a,b): return np.sum((acf(a, nlags=nlags)-acf(b, nlags=nlags))**2)

def spectral_kl(x,y):
    p = periodogram(x)[1]; q = periodogram(y)[1]
    return entropy(p/p.sum(), q/q.sum())

def ks_stat(x,y): return ks_2samp(x,y).statistic

def classifier_auc(x,y,win):
    X, Y = [], []
    for i in range(0,len(x)-win,win): X.append(x[i:i+win]); Y.append(0)
    for i in range(0,len(y)-win,win): X.append(y[i:i+win]); Y.append(1)
    X = np.array(X).reshape(-1,win); Y = np.array(Y)
    Xtr,Xte,Ytr,Yte = train_test_split(X,Y,test_size=0.3,random_state=42)
    return roc_auc_score(Yte, RandomForestClassifier(random_state=42).fit(Xtr,Ytr).predict_proba(Xte)[:,1])

# Bootstrap metrics
B = 200

def bootstrap_metric(fn):
    vals=[]
    for seed in range(B):
        yb = generate_synthetic(T, sdg_params, seed)
        vals.append(fn(y_hold, yb[train_end:]))
    arr = np.array(vals)
    return arr.mean(), np.percentile(arr,[2.5,97.5])

metrics = {'ACF L2': acf_l2,'Spectral KL': spectral_kl,'KS stat': ks_stat,
           'Classifier AUC': lambda r,s: classifier_auc(r,s,window)}
results=[]
for name, fn in metrics.items():
    m,ci = bootstrap_metric(fn)
    results.append((name,m,ci[0],ci[1]))

# Forecast tests

def tstr(real_hold, synth_all):
    Xs, ys = [], []
    for i in range(train_end-window): Xs.append(synth_all[i:i+window]); ys.append(synth_all[i+window])
    mdl = Ridge().fit(np.array(Xs),np.array(ys))
    preds = [mdl.predict(real_hold[i-window:i].reshape(1,-1))[0] for i in range(window,len(real_hold))]
    return mean_squared_error(real_hold[window:], preds)

def trts(real_train, synth_hold):
    Xr, yr = [], []
    for i in range(train_end-window): Xr.append(real_train[i:i+window]); yr.append(real_train[i+window])
    mdl = Ridge().fit(np.array(Xr),np.array(yr))
    preds = [mdl.predict(synth_hold[i-window:i].reshape(1,-1))[0] for i in range(window,len(synth_hold))]
    return mean_squared_error(synth_hold[window:], preds)

tstr_mse = tstr(y_hold, y_synth)
trts_mse = trts(y_train, y_synth[train_end:])

# 6. Results
summary = pd.DataFrame(results, columns=['Metric','Mean','CI Lower','CI Upper'])
print(summary)
print(f"TSTR MSE: {tstr_mse:.4f}, TRTS MSE: {trts_mse:.4f}")
display(summary)
