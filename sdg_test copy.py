# Synthetic vs. Real Data Evaluation

```python
# Jupyter Notebook: synthetic_vs_real_analysis.ipynb
# ------------------------------------------------
# This notebook loads a real-world temperature time series,
# fits a synthetic data generator (trend + adaptive multi-cycle seasonality + AR residuals),
# auto-selects all hyperparameters from data analysis, generates synthetic series,
# computes a suite of metrics with bootstrapped confidence intervals,
# and summarizes the results.

# 0. Mathematical Formulation
# ----------------------------
# y_t = β t + c                        (Trend)
#     + sum_{k=1}^K [A_k sin(2π t/T_k + φ_k)]  (Seasonal cycles)
#     + e_t,  e_t = AR(p) residual process
#     + ε_t, ε_t ~ N(0,σ²)
# Hyperparameter selection:
# - Detect K cycles via periodogram peak prominence
# - Choose AR order p by minimizing AIC
# - Bootstrap metrics for uncertainty quantification

# 1. Imports & Reproducibility
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.stattools import acf
from scipy.signal import periodogram, find_peaks
from scipy.stats import ks_2samp
from sklearn.metrics import pairwise_distances, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from IPython.display import display
np.random.seed(42)

# 2. Load & Preprocess
csv_path = '/mnt/data/4bf67efd-abaa-448b-9607-47f1950d1b7f.csv'
df = pd.read_csv(csv_path)
df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace=True)
y = df['T (degC)'].values
n = len(y)
t = np.arange(n)
print("Real data (first 10 rows):")
display(df.head(10))

# 3. Trend Estimation
beta, c = np.polyfit(t, y, 1)
trend = beta * t + c

# 4. Adaptive Seasonal Detection by Prominence
f, Pxx = periodogram(y)
# Use peak prominence threshold (90th percentile)
prom = np.percentile(Pxx, 90)
peaks, props = find_peaks(Pxx, prominence=prom)
# Fallback to top-2 if <1 peak detected
def top_k_peaks(Pxx, k=2):
    idx = np.argsort(Pxx[1:])[::-1] + 1
    return idx[:k]
if len(peaks) < 1:
    peaks = top_k_peaks(Pxx, k=2)
K = len(peaks)
print(f"Detected {K} cycle(s) at frequencies: {f[peaks]}")
T = (1.0/f[peaks]).astype(int)
print(f"Periods (samples): {T}")

# Build seasonal regressors
X_season = np.column_stack([
    func(2*np.pi*t/Tk) for Tk in T for func in (np.sin, np.cos)
])
resid_trend = y - trend
coef_season, *_ = np.linalg.lstsq(X_season, resid_trend, rcond=None)
seasonal = X_season @ coef_season

# 5. Residual AR(p) Model Selection & Simulation
residuals = y - trend - seasonal
Pmax = min(30, n//50)
aic_list = []
for p_try in range(1, Pmax+1):
    model = AutoReg(residuals, lags=p_try, old_names=False).fit()
    aic_list.append((model.aic, p_try, model))
aic_list.sort(key=lambda x: x[0])
best_aic, p_best, ar_model = aic_list[0]
print(f"Selected AR(p) order: p={p_best} with AIC={best_aic:.1f}")
# Simulate residuals via built-in simulate to avoid warm-start bias
syn_resid = ar_model.simulate(n)
synthetic = trend + seasonal + syn_resid

# 6. Synthesis Function & Bootstrap Setup

def generate_synthetic():
    # re-simulate residuals
    sr = ar_model.simulate(n)
    return trend + seasonal + sr

# Bootstrapping utility
def bootstrap_metric(metric_fn, real, gen_fn, B=200):
    vals = []
    for _ in range(B):
        sim = gen_fn()
        vals.append(metric_fn(real, sim))
    arr = np.array(vals)
    return arr.mean(), np.percentile(arr, [2.5, 97.5])

# 7. Metric Functions
# -------------------
def mean_diff(r, s): return r.mean() - s.mean()
def var_diff(r, s): return r.var() - s.var()
def acf_l2(r, s, nlags=30): return np.sum((acf(r, nlags=nlags, fft=True) - acf(s, nlags=nlags, fft=True))**2)
def psd_l2(r, s): return np.sum((periodogram(r)[1] - periodogram(s)[1])**2)
def ks_stat(r, s): return ks_2samp(r, s).statistic

def compute_mmd(r, s, m=None):
    m = m or min(500, int(np.sqrt(len(r))))
    ix = np.random.choice(len(r), m, replace=False)
    iy = np.random.choice(len(s), m, replace=False)
    Xs, Ys = r[ix].reshape(-1,1), s[iy].reshape(-1,1)
    Z = np.vstack([Xs, Ys])
    sig = np.median(pairwise_distances(Z))
    Kxx = np.exp(-pairwise_distances(Xs, Xs, squared=True)/(2*sig**2))
    Kyy = np.exp(-pairwise_distances(Ys, Ys, squared=True)/(2*sig**2))
    Kxy = np.exp(-pairwise_distances(Xs, Ys, squared=True)/(2*sig**2))
    m_ = len(Xs)
    return ((Kxx.sum()-np.trace(Kxx))/(m_*(m_-1)) + (Kyy.sum()-np.trace(Kyy))/(m_*(m_-1)) - 2*Kxy.sum()/(m_*m_))

def dtw_dist(a, b):
    N, M = len(a), len(b)
    D = np.full((N+1, M+1), np.inf); D[0,0]=0
    for i in range(1, N+1):
        for j in range(1, M+1):
            cost = abs(a[i-1]-b[j-1])
            D[i,j] = cost + min(D[i-1,j], D[i,j-1], D[i-1,j-1])
    return D[N,M]

# Classifier accuracy on non-overlapping windows
def classifier_acc(r, s, w):
    # non-overlapping windows
    X, Y = [], []
    for i in range(0, len(r)-w, w):
        X.append(r[i:i+w]); Y.append(0)
    for i in range(0, len(s)-w, w):
        X.append(s[i:i+w]); Y.append(1)
    X = np.array(X).reshape(-1, w); Y = np.array(Y)
    Xtr, Xte, Ytr, Yte = train_test_split(X, Y, test_size=0.3, random_state=42)
    return RandomForestClassifier(n_estimators=100, random_state=42).fit(Xtr, Ytr).score(Xte, Yte)

# 8. Compute & Bootstrap Metrics
metrics = {}
# Marginals with bootstrap CIs
metrics['Mean Diff'], md_ci = bootstrap_metric(mean_diff, y, generate_synthetic)
metrics['Var Diff'],  vd_ci = bootstrap_metric(var_diff, y, generate_synthetic)
print(f"Mean Diff: {metrics['Mean Diff']:.4f}  CI {md_ci}")
print(f"Var Diff:  {metrics['Var Diff']:.4f}  CI {vd_ci}")
# ACF L2, PSD L2, KS without bootstrap (single-synth)
metrics['ACF L2'] = acf_l2(y, synthetic)
metrics['PSD L2'] = psd_l2(y, synthetic)
metrics['KS D']   = ks_stat(y, synthetic)
# MMD bootstrap
metrics['MMD'], mmd_ci = bootstrap_metric(compute_mmd, y, generate_synthetic)
print(f"MMD:      {metrics['MMD']:.4e}  CI {mmd_ci}")
# DTW avg over segments bootstrap
L = T[0]
segments = min(10, n//(2*L))
def dtw_seg(r, s):
    vals=[]
    for _ in range(segments):
        idx = np.random.randint(0, len(r)-L)
        vals.append(dtw_dist(r[idx:idx+L], s[idx:idx+L]))
    return np.mean(vals)
metrics['DTW avg'], dtw_ci = bootstrap_metric(dtw_seg, y, generate_synthetic)
print(f"DTW avg:  {metrics['DTW avg']:.4f}  CI {dtw_ci}")
# Classifier bootstrap
def clf_fn(r, s): return classifier_acc(r, s, w=T[0])
metrics['Clf Acc'], clf_ci = bootstrap_metric(clf_fn, y, generate_synthetic)
print(f"Clf Acc: {metrics['Clf Acc']:.4f}  CI {clf_ci}")

# 9. Summary Table
print("\nOverall Metrics:")
display(pd.Series(metrics).to_frame('Value'))
```
