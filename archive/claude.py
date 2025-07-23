# Comprehensive Synthetic Data Generator Evaluation Framework
# =============================================================
# This notebook provides a robust framework for testing whether synthetic data generators
# can accurately describe real-world time series data, addressing common methodological flaws.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.signal import periodogram, welch, coherence
from scipy.stats import ks_2samp, anderson_ksamp, entropy, jarque_bera
from statsmodels.tsa.stattools import acf, pacf, adfuller, kpss
from statsmodels.tsa.seasonal import STL, seasonal_decompose
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model
from arch.unitroot import ADF
from pygam import LinearGAM, s
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, mean_squared_error
from sklearn.linear_model import Ridge
from sklearn.feature_extraction import image
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class ComprehensiveSDGEvaluator:
    """
    Comprehensive evaluation framework for Synthetic Data Generators on time series data.
    
    This class provides multiple evaluation perspectives:
    1. Direct comparison metrics (real vs synthetic)
    2. Distributional tests
    3. Temporal dependency analysis
    4. Forecasting performance (TSTR/TRTS)
    5. Statistical property preservation
    6. Robustness across multiple realizations
    """
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.rng = np.random.default_rng(random_state)
        self.results = {}
        
    def load_data(self, data, date_column=None, value_column=None):
        """Load and preprocess time series data."""
        if isinstance(data, pd.DataFrame):
            if date_column and value_column:
                self.data = data.set_index(date_column)[value_column].values
            else:
                self.data = data.iloc[:, -1].values  # Assume last column is target
        else:
            self.data = np.array(data)
            
        self.n = len(self.data)
        print(f"Loaded time series with {self.n} observations")
        return self
    
    def setup_splits(self, n_splits=5, test_size=0.2):
        """Setup temporal cross-validation splits."""
        self.n_splits = n_splits
        self.test_size = test_size
        
        # Create temporal splits
        self.splits = []
        total_size = int(self.n * (1 - test_size))
        split_size = total_size // n_splits
        
        for i in range(n_splits):
            train_end = (i + 1) * split_size
            test_start = train_end
            test_end = min(train_end + int(split_size * test_size), self.n)
            
            if test_end <= test_start:
                break
                
            self.splits.append({
                'train': (0, train_end),
                'test': (test_start, test_end)
            })
        
        print(f"Created {len(self.splits)} temporal splits")
        return self
    
    # =====================================
    # 1. DISTRIBUTIONAL TESTS
    # =====================================
    
    def marginal_distribution_tests(self, real, synthetic):
        """Test marginal distribution similarity."""
        tests = {}
        
        # Kolmogorov-Smirnov test
        ks_stat, ks_p = ks_2samp(real, synthetic)
        tests['ks_statistic'] = ks_stat
        tests['ks_pvalue'] = ks_p
        
        # Anderson-Darling test
        try:
            ad_stat, ad_crit, ad_p = anderson_ksamp([real, synthetic])
            tests['anderson_statistic'] = ad_stat
            tests['anderson_pvalue'] = ad_p
        except:
            tests['anderson_statistic'] = np.nan
            tests['anderson_pvalue'] = np.nan
        
        # Quantile comparison
        quantiles = np.linspace(0.05, 0.95, 19)
        real_q = np.quantile(real, quantiles)
        synth_q = np.quantile(synthetic, quantiles)
        tests['quantile_mae'] = np.mean(np.abs(real_q - synth_q))
        tests['quantile_mse'] = np.mean((real_q - synth_q)**2)
        
        # Moments comparison
        tests['mean_diff'] = abs(np.mean(real) - np.mean(synthetic))
        tests['std_diff'] = abs(np.std(real) - np.std(synthetic))
        tests['skew_diff'] = abs(stats.skew(real) - stats.skew(synthetic))
        tests['kurt_diff'] = abs(stats.kurtosis(real) - stats.kurtosis(synthetic))
        
        # Extreme value comparison
        real_extremes = np.quantile(real, [0.01, 0.99])
        synth_extremes = np.quantile(synthetic, [0.01, 0.99])
        tests['extreme_01_diff'] = abs(real_extremes[0] - synth_extremes[0])
        tests['extreme_99_diff'] = abs(real_extremes[1] - synth_extremes[1])
        
        return tests
    
    def normality_tests(self, real, synthetic):
        """Test normality assumptions."""
        tests = {}
        
        # Jarque-Bera test for normality
        real_jb, real_jb_p = jarque_bera(real)
        synth_jb, synth_jb_p = jarque_bera(synthetic)
        
        tests['real_jb_stat'] = real_jb
        tests['real_jb_pvalue'] = real_jb_p
        tests['synth_jb_stat'] = synth_jb  
        tests['synth_jb_pvalue'] = synth_jb_p
        tests['jb_stat_diff'] = abs(real_jb - synth_jb)
        
        return tests
    
    # =====================================
    # 2. TEMPORAL DEPENDENCY TESTS
    # =====================================
    
    def temporal_dependency_tests(self, real, synthetic, max_lags=50):
        """Comprehensive temporal dependency analysis."""
        tests = {}
        
        # Autocorrelation Function (ACF) comparison
        real_acf = acf(real, nlags=max_lags, fft=True)
        synth_acf = acf(synthetic, nlags=max_lags, fft=True)
        
        # Weighted ACF distance (higher weight on shorter lags)
        weights = 1 / (np.arange(max_lags + 1) + 1)
        tests['acf_weighted_l2'] = np.sum(weights * (real_acf - synth_acf)**2)
        tests['acf_mae'] = np.mean(np.abs(real_acf - synth_acf))
        tests['acf_max_diff'] = np.max(np.abs(real_acf - synth_acf))
        
        # Partial Autocorrelation Function (PACF) comparison
        real_pacf = pacf(real, nlags=min(max_lags, len(real)//4), method='ywm')
        synth_pacf = pacf(synthetic, nlags=min(max_lags, len(synthetic)//4), method='ywm')
        min_len = min(len(real_pacf), len(synth_pacf))
        tests['pacf_mae'] = np.mean(np.abs(real_pacf[:min_len] - synth_pacf[:min_len]))
        
        # Ljung-Box test for autocorrelation
        real_lb = acorr_ljungbox(real, lags=min(20, len(real)//5), return_df=True)
        synth_lb = acorr_ljungbox(synthetic, lags=min(20, len(synthetic)//5), return_df=True)
        
        tests['real_ljung_box_stat'] = real_lb['lb_stat'].iloc[-1]
        tests['synth_ljung_box_stat'] = synth_lb['lb_stat'].iloc[-1]
        tests['real_ljung_box_pvalue'] = real_lb['lb_pvalue'].iloc[-1]
        tests['synth_ljung_box_pvalue'] = synth_lb['lb_pvalue'].iloc[-1]
        
        return tests
    
    def spectral_analysis_tests(self, real, synthetic):
        """Spectral domain analysis."""
        tests = {}
        
        # Power spectral density comparison
        freqs_real, psd_real = periodogram(real)
        freqs_synth, psd_synth = periodogram(synthetic)
        
        # Normalize PSDs
        psd_real_norm = psd_real / np.sum(psd_real)
        psd_synth_norm = psd_synth / np.sum(psd_synth)
        
        # Spectral KL divergence
        # Add small epsilon to avoid log(0)
        epsilon = 1e-10
        psd_real_norm += epsilon
        psd_synth_norm += epsilon
        
        tests['spectral_kl_div'] = entropy(psd_real_norm, psd_synth_norm)
        tests['spectral_l2'] = np.mean((psd_real_norm - psd_synth_norm)**2)
        
        # Dominant frequency comparison
        peak_real = freqs_real[np.argmax(psd_real)]
        peak_synth = freqs_synth[np.argmax(psd_synth)]
        tests['dominant_freq_diff'] = abs(peak_real - peak_synth)
        
        return tests
    
    def stationarity_tests(self, real, synthetic):
        """Test stationarity properties."""
        tests = {}
        
        # Augmented Dickey-Fuller test
        try:
            real_adf = adfuller(real)
            synth_adf = adfuller(synthetic)
            
            tests['real_adf_stat'] = real_adf[0]
            tests['real_adf_pvalue'] = real_adf[1]
            tests['synth_adf_stat'] = synth_adf[0]
            tests['synth_adf_pvalue'] = synth_adf[1]
            
            # KPSS test
            real_kpss = kpss(real, regression='c')
            synth_kpss = kpss(synthetic, regression='c')
            
            tests['real_kpss_stat'] = real_kpss[0]
            tests['real_kpss_pvalue'] = real_kpss[1]
            tests['synth_kpss_stat'] = synth_kpss[0]
            tests['synth_kpss_pvalue'] = synth_kpss[1]
            
        except Exception as e:
            print(f"Stationarity tests failed: {e}")
            for key in ['real_adf_stat', 'real_adf_pvalue', 'synth_adf_stat', 'synth_adf_pvalue',
                       'real_kpss_stat', 'real_kpss_pvalue', 'synth_kpss_stat', 'synth_kpss_pvalue']:
                tests[key] = np.nan
        
        return tests
    
    # =====================================
    # 3. VOLATILITY AND HETEROSKEDASTICITY
    # =====================================
    
    def volatility_tests(self, real, synthetic):
        """Test volatility clustering and heteroskedasticity."""
        tests = {}
        
        # Compute squared returns for volatility proxy
        real_vol = np.diff(real)**2
        synth_vol = np.diff(synthetic)**2
        
        # ACF of squared returns (volatility clustering)
        real_vol_acf = acf(real_vol, nlags=20, fft=True)
        synth_vol_acf = acf(synth_vol, nlags=20, fft=True)
        
        tests['volatility_acf_l2'] = np.mean((real_vol_acf - synth_vol_acf)**2)
        
        # ARCH LM test for heteroskedasticity
        try:
            from statsmodels.stats.diagnostic import het_arch
            real_arch_lm, real_arch_p, _, _ = het_arch(real[1:])
            synth_arch_lm, synth_arch_p, _, _ = het_arch(synthetic[1:])
            
            tests['real_arch_lm'] = real_arch_lm
            tests['real_arch_pvalue'] = real_arch_p
            tests['synth_arch_lm'] = synth_arch_lm
            tests['synth_arch_pvalue'] = synth_arch_p
            
        except:
            tests['real_arch_lm'] = np.nan
            tests['real_arch_pvalue'] = np.nan
            tests['synth_arch_lm'] = np.nan
            tests['synth_arch_pvalue'] = np.nan
        
        return tests
    
    # =====================================
    # 4. CLASSIFICATION-BASED TESTS
    # =====================================
    
    def discriminability_test(self, real, synthetic, window_size=50, n_windows=200):
        """Test if classifier can distinguish real from synthetic data."""
        
        # Create overlapping windows
        def create_windows(data, window_size, n_windows):
            if len(data) < window_size:
                return np.array([])
            
            max_starts = len(data) - window_size + 1
            if max_starts <= n_windows:
                starts = np.arange(max_starts)
            else:
                starts = self.rng.choice(max_starts, n_windows, replace=False)
            
            windows = np.array([data[start:start+window_size] for start in starts])
            return windows
        
        real_windows = create_windows(real, window_size, n_windows)
        synth_windows = create_windows(synthetic, window_size, n_windows)
        
        if len(real_windows) == 0 or len(synth_windows) == 0:
            return {'discriminator_auc': np.nan, 'discriminator_accuracy': np.nan}
        
        # Combine and create labels
        X = np.vstack([real_windows, synth_windows])
        y = np.hstack([np.zeros(len(real_windows)), np.ones(len(synth_windows))])
        
        # Add statistical features
        features = []
        for window in X:
            feat = [
                np.mean(window), np.std(window), stats.skew(window), stats.kurtosis(window),
                np.min(window), np.max(window), np.median(window),
                np.percentile(window, 25), np.percentile(window, 75)
            ]
            # Add autocorrelations
            if len(window) > 10:
                acf_vals = acf(window, nlags=min(5, len(window)//4), fft=True)
                feat.extend(acf_vals[1:])  # Exclude lag 0
            features.append(feat)
        
        X_features = np.array(features)
        
        # Train classifier with time series split
        tscv = TimeSeriesSplit(n_splits=3)
        auc_scores = []
        acc_scores = []
        
        for train_idx, test_idx in tscv.split(X_features):
            X_train, X_test = X_features[train_idx], X_features[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train classifier
            clf = RandomForestClassifier(n_estimators=100, random_state=self.random_state)
            clf.fit(X_train_scaled, y_train)
            
            # Predictions
            y_pred_proba = clf.predict_proba(X_test_scaled)[:, 1]
            y_pred = clf.predict(X_test_scaled)
            
            auc_scores.append(roc_auc_score(y_test, y_pred_proba))
            acc_scores.append(np.mean(y_pred == y_test))
        
        return {
            'discriminator_auc': np.mean(auc_scores),
            'discriminator_accuracy': np.mean(acc_scores)
        }
    
    # =====================================
    # 5. FORECASTING PERFORMANCE TESTS
    # =====================================
    
    def forecasting_tests(self, real_train, real_test, synth_train, synth_test, 
                         window_size=20, horizon=1):
        """TSTR and TRTS forecasting evaluation."""
        
        def create_forecasting_data(data, window_size, horizon):
            X, y = [], []
            for i in range(len(data) - window_size - horizon + 1):
                X.append(data[i:i+window_size])
                y.append(data[i+window_size:i+window_size+horizon])
            return np.array(X), np.array(y).flatten()
        
        tests = {}
        
        try:
            # TSTR: Train on Synthetic, Test on Real
            X_synth, y_synth = create_forecasting_data(synth_train, window_size, horizon)
            X_real_test, y_real_test = create_forecasting_data(real_test, window_size, horizon)
            
            if len(X_synth) > 0 and len(X_real_test) > 0:
                model_tstr = Ridge(alpha=1.0, random_state=self.random_state)
                model_tstr.fit(X_synth, y_synth)
                y_pred_tstr = model_tstr.predict(X_real_test)
                tests['tstr_mse'] = mean_squared_error(y_real_test, y_pred_tstr)
                tests['tstr_mae'] = np.mean(np.abs(y_real_test - y_pred_tstr))
            else:
                tests['tstr_mse'] = np.nan
                tests['tstr_mae'] = np.nan
            
            # TRTS: Train on Real, Test on Synthetic  
            X_real, y_real = create_forecasting_data(real_train, window_size, horizon)
            X_synth_test, y_synth_test = create_forecasting_data(synth_test, window_size, horizon)
            
            if len(X_real) > 0 and len(X_synth_test) > 0:
                model_trts = Ridge(alpha=1.0, random_state=self.random_state)
                model_trts.fit(X_real, y_real)
                y_pred_trts = model_trts.predict(X_synth_test)
                tests['trts_mse'] = mean_squared_error(y_synth_test, y_pred_trts)
                tests['trts_mae'] = np.mean(np.abs(y_synth_test - y_pred_trts))
            else:
                tests['trts_mse'] = np.nan
                tests['trts_mae'] = np.nan
            
            # Baseline: Train on Real, Test on Real
            if len(X_real) > 0 and len(X_real_test) > 0:
                model_baseline = Ridge(alpha=1.0, random_state=self.random_state)
                model_baseline.fit(X_real, y_real)
                y_pred_baseline = model_baseline.predict(X_real_test)
                tests['baseline_mse'] = mean_squared_error(y_real_test, y_pred_baseline)
                tests['baseline_mae'] = np.mean(np.abs(y_real_test - y_pred_baseline))
                
                # Relative performance
                if not np.isnan(tests.get('tstr_mse', np.nan)):
                    tests['tstr_relative_mse'] = tests['tstr_mse'] / tests['baseline_mse']
                if not np.isnan(tests.get('trts_mse', np.nan)):
                    tests['trts_relative_mse'] = tests['trts_mse'] / tests['baseline_mse']
            
        except Exception as e:
            print(f"Forecasting tests failed: {e}")
            for key in ['tstr_mse', 'tstr_mae', 'trts_mse', 'trts_mae', 
                       'baseline_mse', 'baseline_mae', 'tstr_relative_mse', 'trts_relative_mse']:
                tests[key] = np.nan
        
        return tests
    
    # =====================================
    # 6. MAIN EVALUATION PIPELINE
    # =====================================
    
    def evaluate_single_pair(self, real, synthetic):
        """Evaluate a single real-synthetic pair comprehensively."""
        
        results = {}
        
        # 1. Distributional tests
        results.update(self.marginal_distribution_tests(real, synthetic))
        results.update(self.normality_tests(real, synthetic))
        
        # 2. Temporal dependency tests
        results.update(self.temporal_dependency_tests(real, synthetic))
        results.update(self.spectral_analysis_tests(real, synthetic))
        results.update(self.stationarity_tests(real, synthetic))
        
        # 3. Volatility tests
        results.update(self.volatility_tests(real, synthetic))
        
        # 4. Classification test
        results.update(self.discriminability_test(real, synthetic))
        
        return results
    
    def evaluate_generator(self, generator, n_realizations=10):
        """
        Comprehensive evaluation of a synthetic data generator.
        
        Parameters:
        -----------
        generator : object
            Must have fit(data) and generate(length) methods
        n_realizations : int
            Number of synthetic realizations to generate and test
        """
        
        if not hasattr(self, 'splits'):
            self.setup_splits()
        
        all_results = []
        
        print("Starting comprehensive SDG evaluation...")
        print("=" * 50)
        
        for split_idx, split in enumerate(self.splits):
            print(f"\nEvaluating split {split_idx + 1}/{len(self.splits)}")
            
            # Get train/test data
            train_start, train_end = split['train']
            test_start, test_end = split['test']
            
            real_train = self.data[train_start:train_end]
            real_test = self.data[test_start:test_end]
            
            # Fit generator on training data only
            print(f"Fitting generator on {len(real_train)} training points...")
            generator.fit(real_train)
            
            split_results = []
            
            # Generate multiple realizations
            for real_idx in range(n_realizations):
                print(f"  Realization {real_idx + 1}/{n_realizations}", end='\r')
                
                # Generate synthetic test data
                synthetic_test = generator.generate(len(real_test))
                
                # Direct comparison: real test vs synthetic test
                direct_results = self.evaluate_single_pair(real_test, synthetic_test)
                direct_results['split'] = split_idx
                direct_results['realization'] = real_idx
                direct_results['comparison_type'] = 'direct'
                
                split_results.append(direct_results)
                
                # Forecasting tests (TSTR/TRTS)
                synthetic_train = generator.generate(len(real_train))
                forecast_results = self.forecasting_tests(
                    real_train, real_test, synthetic_train, synthetic_test
                )
                forecast_results['split'] = split_idx
                forecast_results['realization'] = real_idx
                forecast_results['comparison_type'] = 'forecasting'
                
                split_results.append(forecast_results)
            
            all_results.extend(split_results)
            print(f"  Completed split {split_idx + 1}")
        
        print("\n" + "=" * 50)
        print("Evaluation completed!")
        
        # Convert to DataFrame for analysis
        self.results_df = pd.DataFrame(all_results)
        return self.results_df
    
    def summarize_results(self):
        """Generate comprehensive summary of evaluation results."""
        
        if not hasattr(self, 'results_df'):
            print("No results to summarize. Run evaluate_generator first.")
            return
        
        # Separate direct and forecasting results
        direct_results = self.results_df[self.results_df['comparison_type'] == 'direct']
        forecast_results = self.results_df[self.results_df['comparison_type'] == 'forecasting']
        
        print("SYNTHETIC DATA GENERATOR EVALUATION SUMMARY")
        print("=" * 60)
        
        # 1. Distributional Quality
        print("\n1. DISTRIBUTIONAL QUALITY")
        print("-" * 30)
        dist_metrics = ['ks_statistic', 'quantile_mae', 'mean_diff', 'std_diff', 
                       'skew_diff', 'kurt_diff']
        for metric in dist_metrics:
            if metric in direct_results.columns:
                mean_val = direct_results[metric].mean()
                std_val = direct_results[metric].std()
                print(f"{metric:20s}: {mean_val:.4f} ± {std_val:.4f}")
        
        # 2. Temporal Dependencies
        print("\n2. TEMPORAL DEPENDENCIES")
        print("-" * 30)
        temp_metrics = ['acf_weighted_l2', 'pacf_mae', 'spectral_kl_div', 
                       'dominant_freq_diff']
        for metric in temp_metrics:
            if metric in direct_results.columns:
                mean_val = direct_results[metric].mean()
                std_val = direct_results[metric].std()
                print(f"{metric:20s}: {mean_val:.4f} ± {std_val:.4f}")
        
        # 3. Discriminability
        print("\n3. DISCRIMINABILITY")
        print("-" * 30)
        if 'discriminator_auc' in direct_results.columns:
            auc_mean = direct_results['discriminator_auc'].mean()
            auc_std = direct_results['discriminator_auc'].std()
            print(f"Discriminator AUC       : {auc_mean:.4f} ± {auc_std:.4f}")
            print(f"Quality Assessment      : {'EXCELLENT' if auc_mean < 0.6 else 'GOOD' if auc_mean < 0.7 else 'FAIR' if auc_mean < 0.8 else 'POOR'}")
        
        # 4. Forecasting Performance
        print("\n4. FORECASTING PERFORMANCE")
        print("-" * 30)
        forecast_metrics = ['tstr_relative_mse', 'trts_relative_mse']
        for metric in forecast_metrics:
            if metric in forecast_results.columns:
                valid_values = forecast_results[metric].dropna()
                if len(valid_values) > 0:
                    mean_val = valid_values.mean()
                    std_val = valid_values.std()
                    print(f"{metric:20s}: {mean_val:.4f} ± {std_val:.4f}")
        
        # 5. Overall Assessment
        print("\n5. OVERALL ASSESSMENT")
        print("-" * 30)
        
        # Calculate composite score
        scores = []
        
        # Distributional score (lower is better)
        if 'ks_statistic' in direct_results.columns:
            ks_score = 1 - direct_results['ks_statistic'].mean()
            scores.append(('Distributional', ks_score))
        
        # Temporal score (lower is better for most metrics)
        if 'acf_weighted_l2' in direct_results.columns:
            acf_score = 1 / (1 + direct_results['acf_weighted_l2'].mean())
            scores.append(('Temporal', acf_score))
        
        # Discriminability score (lower AUC is better)
        if 'discriminator_auc' in direct_results.columns:
            disc_score = 1 - direct_results['discriminator_auc'].mean()
            scores.append(('Discriminability', disc_score))
        
        # Print component scores
        for name, score in scores:
            print(f"{name:20s}: {score:.3f}")
        
        if scores:
            overall_score = np.mean([score for _, score in scores])
            print(f"\nOverall Quality Score   : {overall_score:.3f}")
            
            if overall_score > 0.8:
                assessment = "EXCELLENT - Synthetic data closely matches real data"
            elif overall_score > 0.6:
                assessment = "GOOD - Synthetic data captures most important patterns"
            elif overall_score > 0.4:
                assessment = "FAIR - Synthetic data has moderate quality"
            else:
                assessment = "POOR - Synthetic data does not adequately represent real data"
            
            print(f"Assessment              : {assessment}")
    
    def plot_comparison(self, generator, n_samples=1000, figsize=(15, 12)):
        """Create comprehensive visualization comparing real and synthetic data."""
        
        # Generate synthetic data for plotting
        generator.fit(self.data)
        synthetic = generator.generate(n_samples)
        real_sample = self.data[:n_samples] if len(self.data) >= n_samples else self.data
        
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        fig.suptitle('Real vs Synthetic Data Comparison', fontsize=16, fontweight='bold')
        
        # 1. Time series plot
        axes[0, 0].plot(real_sample[:500], label='Real', alpha=0.7, linewidth=1)
        axes[0, 0].plot(synthetic[:500], label='Synthetic', alpha=0.7, linewidth=1)
        axes[0, 0].set_title('Time Series Comparison')
        axes[0, 0].set_xlabel('Time')
        axes[0, 0].set_ylabel('Value')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Distribution comparison
        axes[0, 1].hist(real_sample, bins=50, alpha=0.6, label='Real', density=True)
        axes[0, 1].hist(synthetic, bins=50, alpha=0.6, label='Synthetic', density=True)
        axes[0, 1].set_title('Distribution Comparison')
        axes[0, 1].set_xlabel('Value')
        axes[0, 1].set_ylabel('Density')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Q-Q plot
        from scipy.stats import probplot
        real_quantiles = np.quantile(real_sample, np.linspace(0.01, 0.99, 99))
        synth_quantiles = np.quantile(synthetic, np.linspace(0.01, 0.99, 99))
        axes[0, 2].scatter(real_quantiles, synth_quantiles, alpha=0.6, s=20)
        min_val = min(real_quantiles.min(), synth_quantiles.min())
        max_val = max(real_quantiles.max(), synth_quantiles.max())
        axes[0, 2].plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
        axes[0, 2].set_title('Q-Q Plot')
        axes[0, 2].set_xlabel('Real Quantiles')
        axes[0, 2].set_ylabel('Synthetic Quantiles')
        axes[0, 2].grid(True, alpha=0.3)
        
        # 4. Autocorrelation comparison
        max_lags = min(50, len(real_sample)//4)
        real_acf = acf(real_sample, nlags=max_lags, fft=True)
        synth_acf = acf(synthetic, nlags=max_lags, fft=True)
        
        lags = np.arange(max_lags + 1)
        axes[1, 0].plot(lags, real_acf, 'o-', label='Real', alpha=0.7, markersize=3)
        axes[1, 0].plot(lags, synth_acf, 's-', label='Synthetic', alpha=0.7, markersize=3)
        axes[1, 0].axhline(y=0, color='k', linestyle='-', alpha=0.3)
        axes[1, 0].set_title('Autocorrelation Function')
        axes[1, 0].set_xlabel('Lag')
        axes[1, 0].set_ylabel('ACF')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 5. Power spectral density
        from scipy.signal import welch
        freqs_real, psd_real = welch(real_sample, nperseg=min(256, len(real_sample)//4))
        freqs_synth, psd_synth = welch(synthetic, nperseg=min(256, len(synthetic)//4))
        
        axes[1, 1].semilogy(freqs_real, psd_real, label='Real', alpha=0.7)
        axes[1, 1].semilogy(freqs_synth, psd_synth, label='Synthetic', alpha=0.7)
        axes[1, 1].set_title('Power Spectral Density')
        axes[1, 1].set_xlabel('Frequency')
        axes[1, 1].set_ylabel('PSD')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        # 6. Rolling statistics comparison
        window = min(50, len(real_sample)//10)
        real_rolling_mean = pd.Series(real_sample).rolling(window).mean()
        real_rolling_std = pd.Series(real_sample).rolling(window).std()
        synth_rolling_mean = pd.Series(synthetic).rolling(window).mean()
        synth_rolling_std = pd.Series(synthetic).rolling(window).std()
        
        axes[1, 2].plot(real_rolling_std, label='Real', alpha=0.7)
        axes[1, 2].plot(synth_rolling_std, label='Synthetic', alpha=0.7)
        axes[1, 2].set_title('Rolling Standard Deviation')
        axes[1, 2].set_xlabel('Time')
        axes[1, 2].set_ylabel('Rolling Std')
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()


# =========================================================================
# EXAMPLE SYNTHETIC DATA GENERATORS
# =========================================================================

class ImprovedTimeSeriesGenerator:
    """
    Improved time series generator using STL decomposition, GAM for trend,
    periodic splines for seasonality, and GARCH for residuals.
    """
    
    def __init__(self, period=None, random_state=42):
        self.period = period
        self.random_state = random_state
        self.rng = np.random.default_rng(random_state)
        
    def fit(self, data):
        """Fit the generator to training data."""
        self.data = np.array(data)
        self.n_train = len(data)
        
        # Auto-detect period if not provided
        if self.period is None:
            self.period = self._detect_period(data)
        
        # STL decomposition
        try:
            stl = STL(data, period=self.period, robust=True)
            decomposition = stl.fit()
            
            self.trend = decomposition.trend
            self.seasonal = decomposition.seasonal
            self.residual = decomposition.resid
        except:
            # Fallback to simple decomposition
            self.trend = pd.Series(data).rolling(window=self.period, center=True).mean().fillna(method='bfill').fillna(method='ffill').values
            self.seasonal = np.zeros_like(data)
            self.residual = data - self.trend
        
        # Fit GAM to trend
        time_points = np.arange(len(data)).reshape(-1, 1)
        try:
            self.trend_model = LinearGAM(s(0)).fit(time_points, self.trend)
        except:
            # Fallback to linear trend
            self.trend_coef = np.polyfit(np.arange(len(data)), self.trend, 1)
            self.trend_model = None
        
        # Fit periodic spline to seasonality
        cycle_indices = (np.arange(len(data)) % self.period).reshape(-1, 1)
        try:
            self.seasonal_model = LinearGAM(s(0, basis='cc')).fit(cycle_indices, self.seasonal)
        except:
            # Fallback to mean seasonality
            seasonal_pattern = np.zeros(self.period)
            for i in range(self.period):
                mask = (np.arange(len(data)) % self.period) == i
                if np.any(mask):
                    seasonal_pattern[i] = np.mean(self.seasonal[mask])
            self.seasonal_pattern = seasonal_pattern
            self.seasonal_model = None
        
        # Fit GARCH to residuals
        try:
            self.garch_model = arch_model(self.residual, vol='Garch', p=1, q=1, dist='normal')
            self.garch_fit = self.garch_model.fit(disp='off')
        except:
            # Fallback to normal residuals
            self.residual_std = np.std(self.residual)
            self.garch_fit = None
            
        return self
    
    def _detect_period(self, data):
        """Auto-detect dominant period in the data."""
        try:
            # Use periodogram to find dominant frequency
            freqs, power = periodogram(data)
            # Find peaks in power spectrum
            from scipy.signal import find_peaks
            peaks, _ = find_peaks(power, height=np.percentile(power, 75))
            
            if len(peaks) > 0:
                # Get the frequency with highest power
                dominant_freq_idx = peaks[np.argmax(power[peaks])]
                dominant_freq = freqs[dominant_freq_idx]
                
                if dominant_freq > 0:
                    period = int(1 / dominant_freq)
                    # Ensure reasonable period bounds
                    period = max(2, min(period, len(data) // 4))
                    return period
        except:
            pass
        
        # Fallback periods based on data length
        if len(data) > 365:
            return 365  # Annual
        elif len(data) > 52:
            return 52   # Weekly
        elif len(data) > 24:
            return 24   # Daily
        else:
            return max(2, len(data) // 4)
    
    def generate(self, length):
        """Generate synthetic time series of specified length."""
        
        # Generate trend
        future_time = np.arange(length).reshape(-1, 1)
        if self.trend_model is not None:
            try:
                trend_component = self.trend_model.predict(future_time)
            except:
                # Extend trend linearly
                last_trend = self.trend[-1]
                if len(self.trend) > 1:
                    trend_slope = self.trend[-1] - self.trend[-2]
                else:
                    trend_slope = 0
                trend_component = last_trend + trend_slope * np.arange(length)
        else:
            # Use linear trend coefficients
            trend_component = np.polyval(self.trend_coef, np.arange(length))
        
        # Generate seasonality
        cycle_indices = (np.arange(length) % self.period).reshape(-1, 1)
        if self.seasonal_model is not None:
            try:
                seasonal_component = self.seasonal_model.predict(cycle_indices)
            except:
                seasonal_component = self.seasonal_pattern[np.arange(length) % self.period]
        else:
            seasonal_component = self.seasonal_pattern[np.arange(length) % self.period]
        
        # Generate residuals
        if self.garch_fit is not None:
            try:
                # Set numpy random seed for GARCH reproducibility
                np.random.seed(self.rng.integers(0, 2**32))
                garch_sim = self.garch_fit.simulate(self.garch_fit.params, length)
                residual_component = garch_sim['data'].values
            except:
                residual_component = self.rng.normal(0, self.residual_std, length)
        else:
            residual_component = self.rng.normal(0, self.residual_std, length)
        
        # Combine components
        synthetic_series = trend_component + seasonal_component + residual_component
        
        return synthetic_series


class SimpleARIMAGenerator:
    """Simple ARIMA-based generator for comparison."""
    
    def __init__(self, order=(1, 1, 1), random_state=42):
        self.order = order
        self.random_state = random_state
        self.rng = np.random.default_rng(random_state)
        
    def fit(self, data):
        """Fit ARIMA model to data."""
        try:
            self.model = ARIMA(data, order=self.order)
            self.fitted_model = self.model.fit()
        except:
            # Fallback to AR(1) model
            self.order = (1, 0, 0)
            self.model = ARIMA(data, order=self.order)
            self.fitted_model = self.model.fit()
        
        return self
    
    def generate(self, length):
        """Generate synthetic series using fitted ARIMA model."""
        try:
            # Set random seed for reproducibility
            np.random.seed(self.rng.integers(0, 2**32))
            forecast = self.fitted_model.simulate(length)
            return forecast.values
        except:
            # Fallback to random walk
            return np.cumsum(self.rng.normal(0, 1, length))


# =========================================================================
# USAGE EXAMPLE
# =========================================================================

def run_comprehensive_evaluation_example():
    """
    Complete example of how to use the comprehensive evaluation framework.
    """
    
    print("COMPREHENSIVE SDG EVALUATION EXAMPLE")
    print("=" * 50)
    
    # 1. Generate or load real data
    print("\n1. Generating example real-world time series...")
    np.random.seed(42)
    n_points = 1000
    time = np.arange(n_points)
    
    # Create realistic time series with trend + seasonality + noise
    trend = 0.002 * time + 10
    seasonality = 3 * np.sin(2 * np.pi * time / 50) + 1.5 * np.sin(2 * np.pi * time / 200)
    noise = np.random.normal(0, 1, n_points)
    
    # Add some volatility clustering
    volatility = np.ones(n_points)
    for i in range(1, n_points):
        volatility[i] = 0.7 * volatility[i-1] + 0.3 * abs(noise[i-1]) + 0.1
    
    noise = noise * volatility
    real_data = trend + seasonality + noise
    
    print(f"Generated real time series with {n_points} points")
    
    # 2. Initialize evaluator
    print("\n2. Initializing comprehensive evaluator...")
    evaluator = ComprehensiveSDGEvaluator(random_state=42)
    evaluator.load_data(real_data)
    evaluator.setup_splits(n_splits=3, test_size=0.2)
    
    # 3. Test multiple generators
    print("\n3. Testing multiple synthetic data generators...")
    
    generators = {
        'Improved Generator': ImprovedTimeSeriesGenerator(random_state=42),
        'Simple ARIMA': SimpleARIMAGenerator(order=(2, 1, 1), random_state=42)
    }
    
    results = {}
    
    for name, generator in generators.items():
        print(f"\n   Testing {name}...")
        print("   " + "-" * 30)
        
        try:
            # Run evaluation
            result_df = evaluator.evaluate_generator(generator, n_realizations=5)
            results[name] = result_df
            
            # Print summary
            print(f"\n   Summary for {name}:")
            evaluator.summarize_results()
            
        except Exception as e:
            print(f"   Error evaluating {name}: {e}")
            continue
    
    # 4. Compare generators
    if len(results) > 1:
        print("\n4. COMPARATIVE ANALYSIS")
        print("=" * 50)
        
        # Compare key metrics
        comparison_metrics = ['ks_statistic', 'acf_weighted_l2', 'discriminator_auc', 'tstr_relative_mse']
        
        for metric in comparison_metrics:
            print(f"\n{metric.upper()}:")
            print("-" * 30)
            
            for name, result_df in results.items():
                direct_results = result_df[result_df['comparison_type'] == 'direct']
                
                if metric in direct_results.columns:
                    values = direct_results[metric].dropna()
                    if len(values) > 0:
                        print(f"{name:20s}: {values.mean():.4f} ± {values.std():.4f}")
                else:
                    forecast_results = result_df[result_df['comparison_type'] == 'forecasting']
                    if metric in forecast_results.columns:
                        values = forecast_results[metric].dropna()
                        if len(values) > 0:
                            print(f"{name:20s}: {values.mean():.4f} ± {values.std():.4f}")
    
    # 5. Create visualizations
    print("\n5. Generating comparison plots...")
    for name, generator in generators.items():
        try:
            print(f"   Creating plots for {name}...")
            evaluator.plot_comparison(generator, n_samples=min(1000, len(real_data)))
        except Exception as e:
            print(f"   Error creating plots for {name}: {e}")
    
    print("\n" + "=" * 50)
    print("EVALUATION COMPLETED!")
    print("\nKey takeaways:")
    print("- Lower KS statistic = better distributional match")
    print("- Lower ACF weighted L2 = better temporal correlation match") 
    print("- Lower discriminator AUC = harder to distinguish real from synthetic")
    print("- TSTR relative MSE close to 1.0 = good forecasting utility")
    print("\nFor production use:")
    print("- Increase n_realizations for more robust results")
    print("- Add more sophisticated generators")
    print("- Customize evaluation metrics for your specific domain")

# Run the example
if __name__ == "__main__":
    run_comprehensive_evaluation_example()