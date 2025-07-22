# Synthetic vs. Real Data Evaluation

```python
# Jupyter Notebook: synthetic_vs_real_analysis.ipynb
# ------------------------------------------------
# This notebook loads a real-world temperature time series,
# fits a synthetic data generator (trend + seasonality + AR residuals),
# generates synthetic series, computes a suite of metrics,
# and summarizes the results.

# 1. Imports & Configuration
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.stattools import acf
from scipy.signal import periodogram
from scipy.stats import ks_2samp
from sklearn.metrics import pairwise_distances, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from scipy.spatial.distance import euclidean
from IPython.display import display

# Reproducible seed
np.random.seed(42)

# 2. Load & Preprocess Data
# -------------------------
csv_path = '/mnt/data/4bf67efd-abaa-448b-9607-47f1950d1b7f.csv'

df = pd.read_csv(csv_path)
df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace=True)

y = df['T (degC)'].values
n = len(y)
t = np.arange(n)

print("Real data (first 10 rows):")
print(df.head(10))

# 3. Parameter Estimation: Trend & Seasonality
# -------------------------------------------
# 3.1 Linear trend: y_t = beta * t + c
beta, c = np.polyfit(t, y, 1)
trend = beta * t + c

# 3.2 Seasonal components: daily + annual sinusoids
T_daily = 144  # 24h * 6 samples/hour
T_annual = 365 * T_daily  # one full year
# Construct four regressors: sin & cos for each cycle
X_season = np.column_stack([
    np.sin(2 * np.pi * t / T_daily),
    np.cos(2 * np.pi * t / T_daily),
    np.sin(2 * np.pi * t / T_annual),
    np.cos(2 * np.pi * t / T_annual)
])
resid_trend = y - trend
# Least squares fit for both cycles
alpha, *_ = np.linalg.lstsq(X_season, resid_trend, rcond=None)
# Extract amplitudes and phases
A_daily = np.hypot(alpha[0], alpha[1])
phi_daily = np.arctan2(alpha[1], alpha[0])
A_annual = np.hypot(alpha[2], alpha[3])
phi_annual = np.arctan2(alpha[3], alpha[2])
# Reconstruct seasonal components
season_daily = A_daily * np.sin(2 * np.pi * t / T_daily + phi_daily)
season_annual = A_annual * np.sin(2 * np.pi * t / T_annual + phi_annual)
seasonal = season_daily + season_annual

# 4. AR Residual Modeling & Synthetic Generation
# --------------------------------------------- AR Residual Modeling & Synthetic Generation
# ---------------------------------------------
residuals = y - trend - seasonal
p = 5
ar = AutoReg(residuals, lags=p, old_names=False).fit()
phi = ar.params[1:]
sigma = np.sqrt(ar.sigma2)

epsilon = np.random.normal(0, sigma, size=n)
syn_resid = np.zeros(n)
syn_resid[:p] = residuals[:p]
for i in range(p, n):
    syn_resid[i] = (phi @ syn_resid[i-p:i][::-1]) + epsilon[i]

synthetic = trend + seasonal + syn_resid

series_real = pd.Series(y, index=df.index, name='Real')
series_syn = pd.Series(synthetic, index=df.index, name='Synthetic')

print("\nReal vs. Synthetic (first 10 rows):")
compare = pd.concat([series_real, series_syn], axis=1)
display(compare.head(10))

plt.figure(figsize=(12, 4))
series_real.plot(label='Real')
series_syn.plot(label='Synthetic', alpha=0.7)
plt.legend(); plt.title('Real vs. Synthetic Time Series'); plt.show()

# 5. Metric Computations with Immediate Display
# --------------------------------------------
metrics = {}

# 5.1 Mean & Variance Differences
print("\n5.1 Mean & Variance Differences:")
df_marg = pd.DataFrame({
    'Real': [series_real.mean(), series_real.var()],
    'Synthetic': [series_syn.mean(), series_syn.var()]
}, index=['Mean', 'Variance'])
display(df_marg)
metrics['Mean Diff'] = series_real.mean() - series_syn.mean()
metrics['Var Diff'] = series_real.var() - series_syn.var()
print(f"Mean Difference: {metrics['Mean Diff']:.4f}")
print(f"Variance Difference: {metrics['Var Diff']:.4f}")

# 5.2 ACF distance
print("\n5.2 ACF L2 Distance:")
def acf_l2(x, y, nlags=30):
    ax = acf(x, nlags=nlags, fft=True)
    ay = acf(y, nlags=nlags, fft=True)
    return np.sum((ax - ay)**2)
metrics['ACF L2 Dist'] = acf_l2(series_real, series_syn)
print(f"ACF L2 Distance (lags 0–30): {metrics['ACF L2 Dist']:.4f}")

# 5.3 PSD distance
print("\n5.3 PSD L2 Distance:")
def psd_l2(x, y):
    fx, Px = periodogram(x)
    fy, Py = periodogram(y)
    return np.sum((Px - Py)**2)
metrics['PSD L2 Dist'] = psd_l2(series_real, series_syn)
print(f"PSD L2 Distance: {metrics['PSD L2 Dist']:.4f}")

# 5.4 KS test
print("\n5.4 Kolmogorov–Smirnov Test:")
ksd, ksp = ks_2samp(series_real, series_syn)
metrics['KS Statistic'] = ksd
metrics['KS p-value'] = ksp
print(f"KS Statistic D: {ksd:.4f}, p-value: {ksp:.4f}")

# 5.5 MMD (subsample)
print("\n5.5 Maximum Mean Discrepancy (MMD):")
def mmd_sub(x, y, m=500):
    ix = np.random.choice(len(x), m, replace=False)
    iy = np.random.choice(len(y), m, replace=False)
    Xs, Ys = x[ix].reshape(-1,1), y[iy].reshape(-1,1)
    Z = np.vstack([Xs, Ys])
    sig = np.median(pairwise_distances(Z, metric='euclidean'))
    Kxx = np.exp(-pairwise_distances(Xs, Xs, squared=True)/(2*sig**2))
    Kyy = np.exp(-pairwise_distances(Ys, Ys, squared=True)/(2*sig**2))
    Kxy = np.exp(-pairwise_distances(Xs, Ys, squared=True)/(2*sig**2))
    m, n_ = len(Xs), len(Ys)
    return ((Kxx.sum()-np.trace(Kxx))/(m*(m-1))
          +(Kyy.sum()-np.trace(Kyy))/(n_*(n_-1))
          -2*Kxy.sum()/(m*n_))
metrics['MMD'] = mmd_sub(series_real.values, series_syn.values)
print(f"MMD: {metrics['MMD']:.6f}")

# 5.6 DTW (segment avg)
print("\n5.6 Dynamic Time Warping (DTW) Average:")
def dtw(x, y):
    nx, ny = len(x), len(y)
    D = np.full((nx+1, ny+1), np.inf)
    D[0,0] = 0
    for i in range(1, nx+1):
        for j in range(1, ny+1):
            cost = abs(x[i-1] - y[j-1])
            D[i,j] = cost + min(D[i-1,j], D[i,j-1], D[i-1,j-1])
    return D[nx, ny]
segs, L = 5, 1000
dvals = []
for _ in range(segs):
    s = np.random.randint(0, n-L)
    dvals.append(dtw(y[s:s+L], synthetic[s:s+L]))
metrics['DTW Dist (avg)'] = np.mean(dvals)
print(f"Avg DTW over {segs} segments: {metrics['DTW Dist (avg')]:.4f}")

# 5.7 Classifier accuracy
print("\n5.7 Classifier Discriminability:")
w = 50
X, yl = [], []
for i in range(n-w):
    X.append(y[i:i+w]);    yl.append(0)
    X.append(synthetic[i:i+w]); yl.append(1)
X = np.array(X).reshape(-1, w)
yl = np.array(yl)
Xtr, Xte, ytr, yte = train_test_split(X, yl, test_size=0.3, random_state=42)
clf = RandomForestClassifier(n_estimators=100, random_state=42).fit(Xtr, ytr)
metrics['Classifier Acc'] = accuracy_score(yte, clf.predict(Xte))
print(f"Classifier Accuracy: {metrics['Classifier Acc']:.4f}")

# 6. Summary of All Metrics
print("\n6. Overall Metrics Summary:")
results = pd.Series(metrics).to_frame('Value')
display(results)
```
