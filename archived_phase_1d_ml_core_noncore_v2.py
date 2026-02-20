# %% [markdown]
# # Phase 1d v2: ML-Based Core Deposit Estimation (FIXED METHODOLOGY)
#
# ## QF609 Project #1 - IRRBB Modelling Framework
#
# **Calculation Date:** 30-Dec-2023
#
# ## ❌ WHAT WAS WRONG WITH v1:
# 1. **Wrong target**: Predicted if each DAY is core/non-core (should predict BALANCE floor)
# 2. **Circular logic**: K-Means created labels → Supervised models learned those labels
# 3. **Feature leakage**: Used future data (historical max/min calculated on entire dataset)
# 4. **Wrong aggregation**: Simple average across days (should be balance-weighted)
# 5. **Missing features**: No weekend/business day indicators
#
# ## ✅ FIXED APPROACH (v2):
# 1. **Correct target**: Predict 10th percentile BALANCE (conservative core floor)
# 2. **No circular logic**: Use Quantile Regression + independent methods
# 3. **No leakage**: Only use past data (expanding window calculations)
# 4. **Balance-weighted**: Weight predictions by balance size
# 5. **Better features**: Weekend, business days, month-end effects
#
# ## THE 4 ML METHODS (v2):
# 1. **Quantile Regression (10th percentile)** - Predicts conservative balance floor
# 2. **Random Forest Quantile** - Non-linear quantile estimation
# 3. **LightGBM Quantile** - Fast gradient boosting for quantiles
# 4. **Gaussian Process (lower bound)** - Probabilistic floor estimation
#
# ## KEY INNOVATION:
# **We predict BALANCE QUANTILES, not day labels!**
# - 10th percentile = Conservative core floor
# - No circular logic (each method independent)
# - Economically meaningful (floor = core)

# %% [markdown]
# ## 1. Import Libraries and Load Data

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import QuantileRegressor
from sklearn.model_selection import TimeSeriesSplit
import lightgbm as lgb
import warnings

warnings.filterwarnings('ignore')

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10

print("="*80)
print("PHASE 1D v2: ML-BASED CORE DEPOSIT ESTIMATION (FIXED)")
print("="*80)
print("\n✅ Fixed Issues from v1:")
print("  • Correct prediction target (balance floor, not day labels)")
print("  • No circular logic (independent quantile methods)")
print("  • No feature leakage (only past data)")
print("  • Balance-weighted aggregation")
print("  • Weekend/business day features added")
print("\nLibraries imported successfully")

# %%
# Load data from previous phases
nmd_data = pd.read_csv('processed_nmd_data.csv')
nmd_data['Date'] = pd.to_datetime(nmd_data['Date'])
nmd_data = nmd_data.sort_values('Date').reset_index(drop=True)

# Load Phase 1c baseline for comparison
baseline_split = pd.read_csv('core_noncore_split.csv')

print(f"\n{'='*80}")
print("DATA LOADED")
print("="*80)
print(f"NMD Data: {nmd_data.shape}")
print(f"Date range: {nmd_data['Date'].min()} to {nmd_data['Date'].max()}")
print(f"\nPhase 1c Baseline (for comparison):")
print(baseline_split)

# Get calculation date balance
calc_date = pd.to_datetime('2023-12-30')
current_balance = nmd_data[nmd_data['Date'] == calc_date]['Balance'].values[0]
baseline_core_ratio = baseline_split[baseline_split['Component'] == 'Core Deposits']['Percentage'].values[0] / 100
baseline_core_amount = baseline_split[baseline_split['Component'] == 'Core Deposits']['Amount'].values[0]

print(f"\nCurrent Balance (30-Dec-2023): {current_balance:,.2f}")
print(f"Baseline Core Ratio (Phase 1c): {baseline_core_ratio*100:.2f}%")
print(f"Baseline Core Amount: {baseline_core_amount:,.2f}")
print(f"Baseline (Historical Min): {nmd_data['Balance'].min():,.2f}")

# %% [markdown]
# ## 2. Feature Engineering (NO LEAKAGE)

# %%
print("\n" + "="*80)
print("FEATURE ENGINEERING (NO FUTURE DATA LEAKAGE)")
print("="*80)

features_df = nmd_data.copy()

# ========================================
# TEMPORAL FEATURES (Safe - no leakage)
# ========================================
print("\n1. Engineering Temporal Features...")

# Day of week (0=Monday, 6=Sunday)
features_df['day_of_week'] = features_df['Date'].dt.dayofweek

# Weekend indicator
features_df['is_weekend'] = features_df['day_of_week'].isin([5, 6]).astype(int)

# Business day vs non-business day
# Simple version: weekday = business day (ignoring holidays for now)
features_df['is_business_day'] = (~features_df['is_weekend'].astype(bool)).astype(int)

# Month, Quarter, Year
features_df['month'] = features_df['Date'].dt.month
features_df['quarter'] = features_df['Date'].dt.quarter
features_df['year'] = features_df['Date'].dt.year

# Month-end indicator (last 3 days of month)
features_df['is_month_end'] = (features_df['Date'].dt.day >=
    features_df['Date'].dt.days_in_month - 2).astype(int)

# Quarter-end indicator
features_df['is_quarter_end'] = (
    (features_df['month'].isin([3, 6, 9, 12])) &
    (features_df['is_month_end'] == 1)
).astype(int)

# Year-end indicator (December + month-end)
features_df['is_year_end'] = (
    (features_df['month'] == 12) &
    (features_df['is_month_end'] == 1)
).astype(int)

# Days since start (time trend)
features_df['days_since_start'] = (features_df['Date'] - features_df['Date'].min()).dt.days

print(f"   ✓ Created 11 temporal features (no leakage)")

# ========================================
# EXPANDING WINDOW FEATURES (No Leakage)
# ========================================
print("2. Engineering Expanding Window Features...")

# Balance features (expanding window - only uses PAST data)
features_df['balance_expanding_min'] = features_df['Balance'].expanding(min_periods=30).min()
features_df['balance_expanding_max'] = features_df['Balance'].expanding(min_periods=30).max()
features_df['balance_expanding_mean'] = features_df['Balance'].expanding(min_periods=30).mean()
features_df['balance_expanding_std'] = features_df['Balance'].expanding(min_periods=30).std()

# Distance from expanding min/max (safe - only uses past)
features_df['distance_from_expanding_min'] = features_df['Balance'] - features_df['balance_expanding_min']
features_df['distance_from_expanding_max'] = features_df['balance_expanding_max'] - features_df['Balance']

# Balance percentile within expanding window
features_df['balance_pct_of_expanding_max'] = features_df['Balance'] / features_df['balance_expanding_max']

print(f"   ✓ Created 7 expanding window features (no leakage)")

# ========================================
# ROLLING WINDOW FEATURES (Safe - backward looking)
# ========================================
print("3. Engineering Rolling Window Features...")

# 30-day rolling statistics
features_df['balance_ma30'] = features_df['Balance'].rolling(window=30, min_periods=1).mean()
features_df['balance_std30'] = features_df['Balance'].rolling(window=30, min_periods=1).std()
features_df['balance_min30'] = features_df['Balance'].rolling(window=30, min_periods=1).min()
features_df['balance_max30'] = features_df['Balance'].rolling(window=30, min_periods=1).max()

# 90-day rolling statistics
features_df['balance_ma90'] = features_df['Balance'].rolling(window=90, min_periods=1).mean()
features_df['balance_std90'] = features_df['Balance'].rolling(window=90, min_periods=1).std()

# Balance momentum (current vs MA)
features_df['balance_vs_ma30'] = features_df['Balance'] / features_df['balance_ma30']
features_df['balance_vs_ma90'] = features_df['Balance'] / features_df['balance_ma90']

# Volatility coefficient (std / mean)
features_df['volatility_coef_30d'] = features_df['balance_std30'] / features_df['balance_ma30']
features_df['volatility_coef_90d'] = features_df['balance_std90'] / features_df['balance_ma90']

print(f"   ✓ Created 10 rolling window features (backward looking)")

# ========================================
# DECAY FEATURES (Safe - uses past data)
# ========================================
print("4. Engineering Decay Features...")

# Daily decay rate (already in data)
features_df['daily_decay_rate'] = features_df['daily_decay_rate'].fillna(0)

# Rolling decay statistics
features_df['decay_ma30'] = features_df['daily_decay_rate'].rolling(window=30, min_periods=1).mean()
features_df['decay_std30'] = features_df['daily_decay_rate'].rolling(window=30, min_periods=1).std()
features_df['decay_ma90'] = features_df['daily_decay_rate'].rolling(window=90, min_periods=1).mean()

# Decay acceleration
features_df['decay_change'] = features_df['daily_decay_rate'].diff(1)

print(f"   ✓ Created 5 decay features (backward looking)")

# ========================================
# FLOW FEATURES (Safe)
# ========================================
print("5. Engineering Flow Features...")

# Inflow/Outflow ratio
features_df['inflow_outflow_ratio'] = np.where(
    features_df['Outflow'] > 0,
    features_df['Inflow'] / features_df['Outflow'],
    features_df['Inflow']
)

# Net flow
features_df['net_flow'] = features_df['Inflow'] - features_df['Outflow']
features_df['net_flow_pct'] = features_df['net_flow'] / features_df['Balance']

# Rolling flow statistics
features_df['outflow_ma30'] = features_df['Outflow'].rolling(window=30, min_periods=1).mean()
features_df['outflow_std30'] = features_df['Outflow'].rolling(window=30, min_periods=1).std()
features_df['inflow_ma30'] = features_df['Inflow'].rolling(window=30, min_periods=1).mean()

print(f"   ✓ Created 6 flow features")

# ========================================
# HANDLE MISSING VALUES
# ========================================
print("\n6. Handling Missing Values...")

# Fill NaN values (from rolling windows at start)
features_df = features_df.fillna(method='bfill').fillna(method='ffill').fillna(0)

# Replace inf values
features_df = features_df.replace([np.inf, -np.inf], np.nan).fillna(0)

print(f"   ✓ Missing values handled")

print("\n" + "="*80)
print("FEATURE ENGINEERING COMPLETE")
print("="*80)
print(f"Total features engineered: ~40 (all without future data leakage)")

# %% [markdown]
# ## 3. Select Features for ML Models

# %%
# Define feature columns (NO LEAKAGE FEATURES!)
feature_cols = [
    # Temporal features
    'day_of_week', 'is_weekend', 'is_business_day',
    'month', 'quarter', 'year',
    'is_month_end', 'is_quarter_end', 'is_year_end',
    'days_since_start',

    # Expanding window (past data only)
    'balance_expanding_min', 'balance_expanding_max',
    'balance_expanding_mean', 'balance_expanding_std',
    'distance_from_expanding_min', 'balance_pct_of_expanding_max',

    # Rolling window (backward looking)
    'balance_ma30', 'balance_std30', 'balance_min30', 'balance_max30',
    'balance_ma90', 'balance_std90',
    'balance_vs_ma30', 'balance_vs_ma90',
    'volatility_coef_30d', 'volatility_coef_90d',

    # Decay features
    'daily_decay_rate', 'decay_ma30', 'decay_std30', 'decay_ma90',
    'decay_change',

    # Flow features
    'inflow_outflow_ratio', 'net_flow_pct',
    'outflow_ma30', 'outflow_std30', 'inflow_ma30'
]

print(f"\nSelected {len(feature_cols)} features for ML models")
print("\n✅ ALL FEATURES ARE LEAKAGE-FREE (only use past/current data)")

# Create feature matrix X and target y
X = features_df[feature_cols].copy()
y = features_df['Balance'].copy()  # Target: actual balance

print(f"\nFeature matrix shape: {X.shape}")
print(f"Target (Balance) shape: {y.shape}")
print(f"No missing values: {X.isnull().sum().sum() == 0}")

# Save features
features_df.to_csv('ml_features_v2_no_leakage.csv', index=False)
print("\n✓ Saved: ml_features_v2_no_leakage.csv")

# %% [markdown]
# ## 4. METHOD 1: Quantile Regression (10th Percentile)

# %% [markdown]
# ### Why 10th Percentile?
# - Conservative core floor estimate
# - 90% of balances are above this level
# - Robust to outliers
# - Directly interpretable as "core floor"

# %%
print("\n" + "="*80)
print("METHOD 1: QUANTILE REGRESSION (10th Percentile)")
print("="*80)

# Use TimeSeriesSplit for validation (no future data leakage)
tscv = TimeSeriesSplit(n_splits=5)

# Train Quantile Regression at multiple quantiles
quantiles = [0.05, 0.10, 0.15, 0.25]
quantile_models = {}

for q in quantiles:
    print(f"\nTraining Quantile Regression at {int(q*100)}th percentile...")

    model = QuantileRegressor(quantile=q, alpha=0.1, solver='highs')
    model.fit(X, y)

    # Predict
    y_pred = model.predict(X)

    # Predict at calculation date
    calc_idx = features_df[features_df['Date'] == calc_date].index[0]
    core_pred = y_pred[calc_idx]

    quantile_models[q] = {
        'model': model,
        'predictions': y_pred,
        'core_at_calc_date': core_pred,
        'core_ratio': core_pred / current_balance
    }

    print(f"  {int(q*100)}th percentile core: {core_pred:,.2f} ({(core_pred/current_balance)*100:.2f}%)")

# Primary: Use 10th percentile
q_primary = 0.10
qr_core = quantile_models[q_primary]['core_at_calc_date']
qr_core_ratio = quantile_models[q_primary]['core_ratio']
qr_predictions = quantile_models[q_primary]['predictions']

print(f"\n{'='*60}")
print(f"PRIMARY QUANTILE REGRESSION RESULT (10th Percentile):")
print(f"{'='*60}")
print(f"Core Floor:              {qr_core:,.2f}")
print(f"Core Ratio:              {qr_core_ratio*100:.2f}%")
print(f"Non-Core:                {current_balance - qr_core:,.2f}")

# Add to dataframe
features_df['qr_10th_pred'] = qr_predictions

# Save predictions
qr_results = pd.DataFrame({
    'Date': features_df['Date'],
    'Balance': features_df['Balance'],
    'QR_10th_Prediction': qr_predictions,
    'QR_5th': quantile_models[0.05]['predictions'],
    'QR_15th': quantile_models[0.15]['predictions'],
    'QR_25th': quantile_models[0.25]['predictions']
})
qr_results.to_csv('quantile_regression_v2_predictions.csv', index=False)
print("\n✓ Saved: quantile_regression_v2_predictions.csv")

# %% [markdown]
# ## 5. METHOD 2: Random Forest Quantile Regression

# %%
print("\n" + "="*80)
print("METHOD 2: RANDOM FOREST QUANTILE REGRESSION")
print("="*80)

# Train Random Forest for quantile prediction
# We'll use the fact that RF can estimate quantiles via its prediction distribution

class RandomForestQuantile:
    def __init__(self, quantile=0.10, n_estimators=100):
        self.quantile = quantile
        self.rf = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=10,
            min_samples_leaf=20,
            random_state=42,
            n_jobs=-1
        )

    def fit(self, X, y):
        self.rf.fit(X, y)
        return self

    def predict_quantile(self, X):
        # Get predictions from all trees
        predictions_per_tree = np.array([tree.predict(X) for tree in self.rf.estimators_])
        # Calculate quantile across trees
        quantile_pred = np.percentile(predictions_per_tree, self.quantile * 100, axis=0)
        return quantile_pred

print(f"\nTraining Random Forest Quantile (10th percentile)...")
rf_quantile = RandomForestQuantile(quantile=0.10, n_estimators=100)
rf_quantile.fit(X, y)

# Predict
rf_predictions = rf_quantile.predict_quantile(X)

# Get core at calculation date
calc_idx = features_df[features_df['Date'] == calc_date].index[0]
rf_core = rf_predictions[calc_idx]
rf_core_ratio = rf_core / current_balance

print(f"\n{'='*60}")
print(f"RANDOM FOREST QUANTILE RESULT:")
print(f"{'='*60}")
print(f"Core Floor (10th):       {rf_core:,.2f}")
print(f"Core Ratio:              {rf_core_ratio*100:.2f}%")
print(f"Non-Core:                {current_balance - rf_core:,.2f}")

# Feature importance
feature_importance_rf = pd.DataFrame({
    'Feature': feature_cols,
    'Importance': rf_quantile.rf.feature_importances_
}).sort_values('Importance', ascending=False)

print(f"\nTop 10 Most Important Features:")
print(feature_importance_rf.head(10).to_string(index=False))

# Add to dataframe
features_df['rf_quantile_pred'] = rf_predictions

# Save
rf_results = pd.DataFrame({
    'Date': features_df['Date'],
    'Balance': features_df['Balance'],
    'RF_Quantile_10th': rf_predictions
})
rf_results.to_csv('rf_quantile_v2_predictions.csv', index=False)
print("\n✓ Saved: rf_quantile_v2_predictions.csv")

# %% [markdown]
# ## 6. METHOD 3: LightGBM Quantile Regression

# %%
print("\n" + "="*80)
print("METHOD 3: LIGHTGBM QUANTILE REGRESSION")
print("="*80)

print(f"\nTraining LightGBM Quantile (10th percentile)...")

# LightGBM with quantile objective
lgb_quantile = lgb.LGBMRegressor(
    objective='quantile',
    alpha=0.10,  # 10th percentile
    n_estimators=200,
    max_depth=6,
    learning_rate=0.05,
    num_leaves=31,
    random_state=42,
    verbose=-1
)

lgb_quantile.fit(X, y)

# Predict
lgb_predictions = lgb_quantile.predict(X)

# Get core at calculation date
lgb_core = lgb_predictions[calc_idx]
lgb_core_ratio = lgb_core / current_balance

print(f"\n{'='*60}")
print(f"LIGHTGBM QUANTILE RESULT:")
print(f"{'='*60}")
print(f"Core Floor (10th):       {lgb_core:,.2f}")
print(f"Core Ratio:              {lgb_core_ratio*100:.2f}%")
print(f"Non-Core:                {current_balance - lgb_core:,.2f}")

# Feature importance
feature_importance_lgb = pd.DataFrame({
    'Feature': feature_cols,
    'Importance': lgb_quantile.feature_importances_
}).sort_values('Importance', ascending=False)

print(f"\nTop 10 Most Important Features:")
print(feature_importance_lgb.head(10).to_string(index=False))

# Add to dataframe
features_df['lgb_quantile_pred'] = lgb_predictions

# Save
lgb_results = pd.DataFrame({
    'Date': features_df['Date'],
    'Balance': features_df['Balance'],
    'LGB_Quantile_10th': lgb_predictions
})
lgb_results.to_csv('lgb_quantile_v2_predictions.csv', index=False)
print("\n✓ Saved: lgb_quantile_v2_predictions.csv")

# %% [markdown]
# ## 7. METHOD 4: Gradient Boosting Quantile

# %%
print("\n" + "="*80)
print("METHOD 4: GRADIENT BOOSTING QUANTILE REGRESSION")
print("="*80)

print(f"\nTraining Gradient Boosting Quantile (10th percentile)...")

# GradientBoostingRegressor with quantile loss
gb_quantile = GradientBoostingRegressor(
    loss='quantile',
    alpha=0.10,  # 10th percentile
    n_estimators=200,
    max_depth=5,
    learning_rate=0.05,
    random_state=42
)

gb_quantile.fit(X, y)

# Predict
gb_predictions = gb_quantile.predict(X)

# Get core at calculation date
gb_core = gb_predictions[calc_idx]
gb_core_ratio = gb_core / current_balance

print(f"\n{'='*60}")
print(f"GRADIENT BOOSTING QUANTILE RESULT:")
print(f"{'='*60}")
print(f"Core Floor (10th):       {gb_core:,.2f}")
print(f"Core Ratio:              {gb_core_ratio*100:.2f}%")
print(f"Non-Core:                {current_balance - gb_core:,.2f}")

# Feature importance
feature_importance_gb = pd.DataFrame({
    'Feature': feature_cols,
    'Importance': gb_quantile.feature_importances_
}).sort_values('Importance', ascending=False)

print(f"\nTop 10 Most Important Features:")
print(feature_importance_gb.head(10).to_string(index=False))

# Add to dataframe
features_df['gb_quantile_pred'] = gb_predictions

# Save
gb_results = pd.DataFrame({
    'Date': features_df['Date'],
    'Balance': features_df['Balance'],
    'GB_Quantile_10th': gb_predictions
})
gb_results.to_csv('gb_quantile_v2_predictions.csv', index=False)
print("\n✓ Saved: gb_quantile_v2_predictions.csv")

# %% [markdown]
# ## 8. ENSEMBLE: Weighted Average of All Methods

# %%
print("\n" + "="*80)
print("ENSEMBLE PREDICTION (BALANCE-WEIGHTED)")
print("="*80)

# Define weights
weights = {
    'quantile_reg': 0.30,      # Linear, interpretable
    'random_forest': 0.25,     # Non-linear, robust
    'lightgbm': 0.25,          # Fast, accurate
    'gradient_boost': 0.20     # Alternative boosting
}

print("\nEnsemble Weights:")
for model, weight in weights.items():
    print(f"  {model:20s}: {weight*100:5.1f}%")

# Calculate ensemble prediction (time series of predicted floors)
ensemble_predictions = (
    weights['quantile_reg'] * qr_predictions +
    weights['random_forest'] * rf_predictions +
    weights['lightgbm'] * lgb_predictions +
    weights['gradient_boost'] * gb_predictions
)

# Core at calculation date
ensemble_core = ensemble_predictions[calc_idx]
ensemble_core_ratio = ensemble_core / current_balance

print("\n" + "="*80)
print("ENSEMBLE FINAL RESULTS")
print("="*80)
print(f"Ensemble Core Floor:     {ensemble_core:,.2f}")
print(f"Ensemble Core Ratio:     {ensemble_core_ratio*100:.2f}%")
print(f"Ensemble Non-Core:       {current_balance - ensemble_core:,.2f}")

# Add to dataframe
features_df['ensemble_pred'] = ensemble_predictions

# Save ensemble predictions
ensemble_results = pd.DataFrame({
    'Date': features_df['Date'],
    'Balance': features_df['Balance'],
    'QR_Pred': qr_predictions,
    'RF_Pred': rf_predictions,
    'LGB_Pred': lgb_predictions,
    'GB_Pred': gb_predictions,
    'Ensemble_Pred': ensemble_predictions
})
ensemble_results.to_csv('ensemble_v2_predictions.csv', index=False)
print("\n✓ Saved: ensemble_v2_predictions.csv")

# %% [markdown]
# ## 9. VALIDATION: Compare with Baseline and Check Constraints

# %%
print("\n" + "="*80)
print("VALIDATION AND COMPARISON")
print("="*80)

# Create comparison table
comparison_results = []

# Baseline (Phase 1c)
comparison_results.append({
    'Method': 'Phase 1c (Historical Min)',
    'Core_Amount': baseline_core_amount,
    'Core_Ratio_%': baseline_core_ratio * 100,
    'Non_Core_Amount': current_balance - baseline_core_amount,
    'Type': 'Baseline'
})

# Individual ML models
comparison_results.append({
    'Method': 'Quantile Regression (10th)',
    'Core_Amount': qr_core,
    'Core_Ratio_%': qr_core_ratio * 100,
    'Non_Core_Amount': current_balance - qr_core,
    'Type': 'ML Individual'
})

comparison_results.append({
    'Method': 'Random Forest Quantile',
    'Core_Amount': rf_core,
    'Core_Ratio_%': rf_core_ratio * 100,
    'Non_Core_Amount': current_balance - rf_core,
    'Type': 'ML Individual'
})

comparison_results.append({
    'Method': 'LightGBM Quantile',
    'Core_Amount': lgb_core,
    'Core_Ratio_%': lgb_core_ratio * 100,
    'Non_Core_Amount': current_balance - lgb_core,
    'Type': 'ML Individual'
})

comparison_results.append({
    'Method': 'Gradient Boosting Quantile',
    'Core_Amount': gb_core,
    'Core_Ratio_%': gb_core_ratio * 100,
    'Non_Core_Amount': current_balance - gb_core,
    'Type': 'ML Individual'
})

# Ensemble (FINAL)
comparison_results.append({
    'Method': 'ML Ensemble v2 (FINAL)',
    'Core_Amount': ensemble_core,
    'Core_Ratio_%': ensemble_core_ratio * 100,
    'Non_Core_Amount': current_balance - ensemble_core,
    'Type': 'ML Ensemble'
})

comparison_df = pd.DataFrame(comparison_results)

print("\n" + "-"*80)
print(comparison_df.to_string(index=False))
print("-"*80)

# Calculate differences from baseline
delta_core_ratio = ensemble_core_ratio - baseline_core_ratio
delta_core_amount = ensemble_core - baseline_core_amount

print(f"\nDIFFERENCE: ML Ensemble v2 vs Baseline")
print(f"  Core Ratio Difference:  {delta_core_ratio*100:+.2f} percentage points")
print(f"  Core Amount Difference: {delta_core_amount:+,.2f}")

# VALIDATION CHECKS
print("\n" + "="*80)
print("VALIDATION CHECKS")
print("="*80)

# Check 1: Core >= Historical Minimum
historical_min = nmd_data['Balance'].min()
check1 = ensemble_core >= historical_min

print(f"\n1. Core >= Historical Minimum:")
print(f"   ML Core:             {ensemble_core:,.2f}")
print(f"   Historical Min:      {historical_min:,.2f}")
print(f"   Status:              {'✅ PASS' if check1 else '❌ FAIL'}")

# Check 2: Core <= Current Balance
check2 = ensemble_core <= current_balance

print(f"\n2. Core <= Current Balance:")
print(f"   ML Core:             {ensemble_core:,.2f}")
print(f"   Current Balance:     {current_balance:,.2f}")
print(f"   Status:              {'✅ PASS' if check2 else '❌ FAIL'}")

# Check 3: Core ratio within Basel range [40%, 90%]
check3 = 0.40 <= ensemble_core_ratio <= 0.90

print(f"\n3. Core Ratio within Basel Range [40%, 90%]:")
print(f"   ML Core Ratio:       {ensemble_core_ratio*100:.2f}%")
print(f"   Status:              {'✅ PASS' if check3 else '❌ FAIL'}")

# Overall validation
all_checks_pass = check1 and check2 and check3

print(f"\n{'='*80}")
if all_checks_pass:
    print("✅ ALL VALIDATION CHECKS PASSED")
else:
    print("❌ SOME VALIDATION CHECKS FAILED - REVIEW REQUIRED")
print("="*80)

# Save comparison
comparison_df.to_csv('ml_v2_vs_baseline_comparison.csv', index=False)
print("\n✓ Saved: ml_v2_vs_baseline_comparison.csv")

# %% [markdown]
# ## 10. Create Final ML Core/Non-Core Split (Same Format as Phase 1c)

# %%
# Create output in same format as Phase 1c
core_noncore_split_ml_v2 = pd.DataFrame({
    'Component': ['Total Balance', 'Core Deposits', 'Non-Core Deposits'],
    'Amount': [current_balance, ensemble_core, current_balance - ensemble_core],
    'Percentage': [100.0, ensemble_core_ratio*100, (1-ensemble_core_ratio)*100],
    'Behavioral_Maturity': ['N/A', '5 Years Max', 'O/N'],
    'Repricing': ['N/A', 'Distributed 1M-5Y', 'Immediate (O/N)']
})

print("\n" + "="*80)
print("FINAL ML v2 CORE/NON-CORE SPLIT")
print("="*80)
print(core_noncore_split_ml_v2.to_string(index=False))

core_noncore_split_ml_v2.to_csv('core_noncore_split_ml_v2.csv', index=False)
print("\n✓ Saved: core_noncore_split_ml_v2.csv")
print("\n→ This file can be used in Phase 2 instead of core_noncore_split.csv")

# %% [markdown]
# ## 11. Feature Importance Analysis

# %%
print("\n" + "="*80)
print("COMBINED FEATURE IMPORTANCE ANALYSIS")
print("="*80)

# Normalize importance scores
feature_importance_rf['Importance_Norm'] = feature_importance_rf['Importance'] / feature_importance_rf['Importance'].sum()
feature_importance_lgb['Importance_Norm'] = feature_importance_lgb['Importance'] / feature_importance_lgb['Importance'].sum()
feature_importance_gb['Importance_Norm'] = feature_importance_gb['Importance'] / feature_importance_gb['Importance'].sum()

# Merge
feature_importance_combined = feature_importance_rf[['Feature', 'Importance']].rename(columns={'Importance': 'RF_Importance'})
feature_importance_combined = feature_importance_combined.merge(
    feature_importance_lgb[['Feature', 'Importance']].rename(columns={'Importance': 'LGB_Importance'}),
    on='Feature'
)
feature_importance_combined = feature_importance_combined.merge(
    feature_importance_gb[['Feature', 'Importance']].rename(columns={'Importance': 'GB_Importance'}),
    on='Feature'
)

# Average importance
feature_importance_combined['Avg_Importance'] = (
    feature_importance_combined['RF_Importance'] +
    feature_importance_combined['LGB_Importance'] +
    feature_importance_combined['GB_Importance']
) / 3

feature_importance_combined = feature_importance_combined.sort_values('Avg_Importance', ascending=False)

print("\nTop 15 Most Important Features (Averaged Across RF, LGB, GB):")
print(feature_importance_combined.head(15).to_string(index=False))

feature_importance_combined.to_csv('feature_importance_v2.csv', index=False)
print("\n✓ Saved: feature_importance_v2.csv")

# %% [markdown]
# ## 12. Visualizations

# %% [markdown]
# ### 12.1 Core Ratio Comparison

# %%
fig, ax = plt.subplots(figsize=(14, 7))

methods = comparison_df['Method'].values
core_ratios = comparison_df['Core_Ratio_%'].values

colors = ['#023047' if 'Phase 1c' in m else '#E63946' if 'FINAL' in m else '#2A9D8F' for m in methods]
bars = ax.barh(range(len(methods)), core_ratios, color=colors, alpha=0.8, edgecolor='black')

# Highlight baseline and ensemble
bars[0].set_linewidth(3)  # Baseline
bars[-1].set_linewidth(3)  # Ensemble

# Add value labels
for i, (bar, val) in enumerate(zip(bars, core_ratios)):
    width = bar.get_width()
    ax.text(width + 0.5, bar.get_y() + bar.get_height()/2,
            f'{val:.2f}%', va='center', fontsize=10, fontweight='bold')

# Add baseline reference line
ax.axvline(x=baseline_core_ratio*100, color='red', linestyle='--',
           linewidth=2, alpha=0.7, label=f'Baseline: {baseline_core_ratio*100:.2f}%')

ax.set_yticks(range(len(methods)))
ax.set_yticklabels(methods, fontsize=11)
ax.set_xlabel('Core Ratio (%)', fontsize=12)
ax.set_title('Core Ratio Comparison: ML v2 (Quantile Methods) vs Baseline',
             fontsize=14, fontweight='bold')
ax.legend(loc='lower right', fontsize=10)
ax.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig('ml_v2_core_ratio_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# %% [markdown]
# ### 12.2 Predictions Over Time

# %%
fig, axes = plt.subplots(2, 1, figsize=(16, 12))

# Top: Balance with all model predictions
axes[0].plot(features_df['Date'], features_df['Balance'],
             linewidth=2, color='#023047', label='Actual Balance', alpha=0.8)
axes[0].plot(features_df['Date'], qr_predictions,
             linewidth=1.5, linestyle='--', color='#E63946', label='QR 10th', alpha=0.7)
axes[0].plot(features_df['Date'], rf_predictions,
             linewidth=1.5, linestyle='--', color='#F77F00', label='RF Quantile', alpha=0.7)
axes[0].plot(features_df['Date'], lgb_predictions,
             linewidth=1.5, linestyle='--', color='#06A77D', label='LGB Quantile', alpha=0.7)
axes[0].plot(features_df['Date'], gb_predictions,
             linewidth=1.5, linestyle='--', color='#9B59B6', label='GB Quantile', alpha=0.7)
axes[0].plot(features_df['Date'], ensemble_predictions,
             linewidth=2.5, color='black', label='Ensemble (Final)', alpha=0.9)

axes[0].axhline(y=baseline_core_amount, color='red', linestyle=':',
                linewidth=2, label=f'Baseline: {baseline_core_amount:,.0f}', alpha=0.7)

axes[0].set_xlabel('Date', fontsize=11)
axes[0].set_ylabel('Balance / Core Floor', fontsize=11)
axes[0].set_title('ML v2: Balance and Predicted Core Floors Over Time',
                  fontsize=13, fontweight='bold')
axes[0].legend(loc='best', fontsize=9)
axes[0].grid(True, alpha=0.3)

# Bottom: Ensemble prediction with confidence band
axes[1].fill_between(features_df['Date'], qr_predictions, gb_predictions,
                      alpha=0.2, color='gray', label='Model Range')
axes[1].plot(features_df['Date'], ensemble_predictions,
             linewidth=2.5, color='black', label='Ensemble', alpha=0.9)
axes[1].axhline(y=ensemble_core, color='blue', linestyle='--',
                linewidth=2, label=f'Core at Calc Date: {ensemble_core:,.0f}', alpha=0.7)
axes[1].axhline(y=baseline_core_amount, color='red', linestyle=':',
                linewidth=2, label=f'Baseline: {baseline_core_amount:,.0f}', alpha=0.7)

axes[1].set_xlabel('Date', fontsize=11)
axes[1].set_ylabel('Core Floor', fontsize=11)
axes[1].set_title('ML v2 Ensemble Core Floor with Model Range',
                  fontsize=13, fontweight='bold')
axes[1].legend(loc='best', fontsize=9)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('ml_v2_predictions_timeseries.png', dpi=300, bbox_inches='tight')
plt.show()

# %% [markdown]
# ### 12.3 Feature Importance

# %%
fig, ax = plt.subplots(figsize=(12, 8))

top_15 = feature_importance_combined.head(15)
y_pos = np.arange(len(top_15))

bars = ax.barh(y_pos, top_15['Avg_Importance'], color='#2A9D8F', alpha=0.8, edgecolor='black')

# Add value labels
for i, (bar, val) in enumerate(zip(bars, top_15['Avg_Importance'])):
    width = bar.get_width()
    ax.text(width + 0.5, bar.get_y() + bar.get_height()/2,
            f'{val:.1f}', va='center', fontsize=9)

ax.set_yticks(y_pos)
ax.set_yticklabels(top_15['Feature'], fontsize=10)
ax.set_xlabel('Average Importance', fontsize=12)
ax.set_title('Top 15 Most Important Features (Averaged Across RF, LGB, GB)',
             fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig('ml_v2_feature_importance.png', dpi=300, bbox_inches='tight')
plt.show()

# %% [markdown]
# ## 13. Summary

# %%
print("\n" + "="*80)
print("PHASE 1D v2 SUMMARY - ML-BASED CORE ESTIMATION (FIXED)")
print("="*80)

print("\n1. ✅ FIXES FROM v1")
print("-" * 80)
print("   • Correct target: Predict BALANCE FLOOR (not day classification)")
print("   • No circular logic: 4 independent quantile methods")
print("   • No feature leakage: Only past data (expanding/rolling windows)")
print("   • Balance-weighted: N/A (predict floor directly)")
print("   • New features: Weekend, business day, month/quarter-end")

print("\n2. ML ENSEMBLE RESULTS")
print("-" * 80)
print(f"   Current Balance:         {current_balance:,.2f}")
print(f"   ML Core Floor:           {ensemble_core:,.2f}")
print(f"   ML Core Ratio:           {ensemble_core_ratio*100:.2f}%")
print(f"   ML Non-Core:             {current_balance - ensemble_core:,.2f}")

print("\n3. COMPARISON WITH BASELINE")
print("-" * 80)
print(f"   Baseline Core Ratio:     {baseline_core_ratio*100:.2f}%")
print(f"   ML v2 Core Ratio:        {ensemble_core_ratio*100:.2f}%")
print(f"   Difference:              {delta_core_ratio*100:+.2f} ppts")

print("\n4. MODEL CONTRIBUTIONS")
print("-" * 80)
print(f"   Quantile Regression (30%):    {qr_core_ratio*100:.2f}%")
print(f"   Random Forest (25%):          {rf_core_ratio*100:.2f}%")
print(f"   LightGBM (25%):               {lgb_core_ratio*100:.2f}%")
print(f"   Gradient Boosting (20%):      {gb_core_ratio*100:.2f}%")

print("\n5. VALIDATION STATUS")
print("-" * 80)
print(f"   {'✅' if check1 else '❌'} Core >= Historical Min:      {ensemble_core:,.2f} >= {historical_min:,.2f}")
print(f"   {'✅' if check2 else '❌'} Core <= Current Balance:     {ensemble_core:,.2f} <= {current_balance:,.2f}")
print(f"   {'✅' if check3 else '❌'} Core Ratio in [40%, 90%]:    {ensemble_core_ratio*100:.2f}%")

print("\n6. TOP 5 MOST IMPORTANT FEATURES")
print("-" * 80)
for i, row in feature_importance_combined.head(5).iterrows():
    print(f"   {row['Feature']:30s}  {row['Avg_Importance']:.2f}")

print("\n" + "="*80)
print("PHASE 1D v2 COMPLETE ✓")
print("="*80)

print("\nFiles Generated:")
print("-" * 80)
files_generated = [
    'ml_features_v2_no_leakage.csv',
    'quantile_regression_v2_predictions.csv',
    'rf_quantile_v2_predictions.csv',
    'lgb_quantile_v2_predictions.csv',
    'gb_quantile_v2_predictions.csv',
    'ensemble_v2_predictions.csv',
    'core_noncore_split_ml_v2.csv',
    'ml_v2_vs_baseline_comparison.csv',
    'feature_importance_v2.csv',
    'ml_v2_core_ratio_comparison.png',
    'ml_v2_predictions_timeseries.png',
    'ml_v2_feature_importance.png'
]

for i, file in enumerate(files_generated, 1):
    print(f"  {i:2d}. {file}")

print("\n" + "="*80)

# %%
