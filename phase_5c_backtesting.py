# %% [markdown]
# # Phase 5c: Backtesting the Decay Model
#
# ## QF609 Project #1 - IRRBB Modelling Framework
#
# **Calculation Date:** 30-Dec-2023
#
# Trains decay model on 2017-2022 data and backtests on 2023.

# %%
from phase_5_helpers import *

print("="*80)
print("SECTION 5c: BACKTESTING THE DECAY MODEL")
print("="*80)

# %% [markdown]
# ## Train/Test Split and Pure Decay Prediction

# %%
# Train: 2017-2022, Test: 2023
train_data = nmd_data[(nmd_data['year'] >= 2017) & (nmd_data['year'] <= 2022)].copy()
test_data = nmd_data[nmd_data['year'] == 2023].copy().reset_index(drop=True)

lambda_daily_train = train_data['daily_decay_rate'].mean()
lambda_std_train = train_data['daily_decay_rate'].std()

print(f"\nTraining Period: 2017-2022 ({len(train_data)} observations)")
print(f"Testing Period:  2023 ({len(test_data)} observations)")
print(f"Lambda (train):  {lambda_daily_train:.6f}")
print(f"Std Dev (train): {lambda_std_train:.6f}")

# --- Pure Decay Prediction ---
B_start = test_data['Balance'].iloc[0]
n_days = len(test_data)
predicted_balance = np.array([B_start * (1 - lambda_daily_train)**t for t in range(n_days)])
actual_balance = test_data['Balance'].values
dates_test = test_data['Date'].values

# %% [markdown]
# ## Monthly Re-Anchored Prediction

# %%
test_data['month_start'] = test_data['Date'].dt.to_period('M')
monthly_reanchored = np.full(n_days, np.nan)

for month_period in test_data['month_start'].unique():
    mask = test_data['month_start'] == month_period
    idx = test_data[mask].index
    b_month_start = test_data.loc[idx[0], 'Balance']
    for j, ix in enumerate(idx):
        monthly_reanchored[ix] = b_month_start * (1 - lambda_daily_train)**j

# %% [markdown]
# ## Backtest Metrics

# %%
errors = actual_balance - predicted_balance
abs_errors = np.abs(errors)
pct_errors = abs_errors / actual_balance * 100

rmse = np.sqrt(np.mean(errors**2))
mape = np.mean(pct_errors)
max_abs_error = np.max(abs_errors)
bias = np.mean(errors)

valid_mask = ~np.isnan(monthly_reanchored)
errors_monthly = actual_balance[valid_mask] - monthly_reanchored[valid_mask]
rmse_monthly = np.sqrt(np.mean(errors_monthly**2))
mape_monthly = np.mean(np.abs(errors_monthly) / actual_balance[valid_mask] * 100)

print(f"\n--- Pure Decay Prediction (from Jan 1) ---")
print(f"RMSE:              {rmse:,.2f}")
print(f"MAPE:              {mape:.2f}%")
print(f"Max Abs Error:     {max_abs_error:,.2f}")
print(f"Bias (mean error): {bias:,.2f}")
print(f"\nNote: Pure decay diverges because it ignores inflows.")
print(f"The model predicts monotonic decline, but actual balance")
print(f"fluctuates due to new deposits. This is a known limitation.")

print(f"\n--- Monthly Re-Anchored Prediction ---")
print(f"RMSE:              {rmse_monthly:,.2f}")
print(f"MAPE:              {mape_monthly:.2f}%")
print(f"Re-anchoring significantly improves short-horizon accuracy.")

# Save metrics
backtest_metrics = pd.DataFrame({
    'Metric': ['RMSE_Pure', 'MAPE_Pure', 'Max_Abs_Error_Pure', 'Bias_Pure',
               'RMSE_Monthly', 'MAPE_Monthly', 'Lambda_Train', 'Lambda_Std_Train'],
    'Value': [rmse, mape, max_abs_error, bias,
              rmse_monthly, mape_monthly, lambda_daily_train, lambda_std_train]
})
backtest_metrics.to_csv('backtest_results.csv', index=False)

backtest_path = pd.DataFrame({
    'Date': dates_test,
    'Actual_Balance': actual_balance,
    'Predicted_Pure_Decay': predicted_balance,
    'Predicted_Monthly_Reanchored': monthly_reanchored,
    'Error_Pure': errors,
    'Pct_Error_Pure': pct_errors
})
backtest_path.to_csv('backtest_predicted_vs_actual.csv', index=False)

print("\nSaved: backtest_results.csv, backtest_predicted_vs_actual.csv")

# %% [markdown]
# ## Charts

# %%
# Chart 1: Actual vs Predicted Balance Path (2023)
fig, ax = plt.subplots(figsize=(14, 7))
ax.plot(dates_test, actual_balance, linewidth=2, color='#023047', label='Actual Balance')
ax.plot(dates_test, predicted_balance, linewidth=2, linestyle='--', color='#E63946',
        label='Pure Decay Prediction')
ax.fill_between(dates_test, actual_balance, predicted_balance, alpha=0.1, color='red')
ax.set_xlabel('Date', fontsize=12)
ax.set_ylabel('Balance', fontsize=12)
ax.set_title('Backtest: Actual vs Pure Decay Prediction (2023)',
            fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# %%
# Chart 2: Prediction Error Over Time
fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(dates_test, errors, linewidth=1.5, color='#E63946')
ax.fill_between(dates_test, 0, errors, alpha=0.2, color='red')
ax.axhline(y=0, color='black', linewidth=1)
ax.axhline(y=bias, color='blue', linestyle='--', linewidth=1.5,
           label=f'Mean Bias: {bias:,.0f}')
ax.set_xlabel('Date', fontsize=12)
ax.set_ylabel('Prediction Error (Actual - Predicted)', fontsize=12)
ax.set_title('Pure Decay Prediction Error Over Time (2023)',
            fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# %%
# Chart 3: Monthly Re-Anchored Predictions
fig, ax = plt.subplots(figsize=(14, 7))
ax.plot(dates_test, actual_balance, linewidth=2, color='#023047', label='Actual Balance')
ax.plot(dates_test, monthly_reanchored, linewidth=1.5, linestyle='--', color='#06A77D',
        label='Monthly Re-Anchored', alpha=0.8)
ax.plot(dates_test, predicted_balance, linewidth=1.5, linestyle=':', color='#E63946',
        label='Pure Decay (from Jan 1)', alpha=0.6)
ax.set_xlabel('Date', fontsize=12)
ax.set_ylabel('Balance', fontsize=12)
ax.set_title('Backtest: Monthly Re-Anchored vs Pure Decay (2023)',
            fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## ML-Enhanced Backtesting
#
# We train regression and ML models on 2017-2022 features to predict
# daily decay rates, then simulate 2023 balance paths.

# %%
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm

# %% [markdown]
# ### Feature Engineering

# %%
def build_features(df):
    """Build feature matrix for ML models."""
    feat = pd.DataFrame(index=df.index)
    feat['log_balance'] = np.log(df['Balance'])
    feat['trend'] = (df['Date'] - df['Date'].min()).dt.days
    feat['rolling_30d_decay'] = df['decay_rate_ma30']
    # Day-of-week dummies
    dow = pd.get_dummies(df['day_of_week'], prefix='dow', drop_first=True, dtype=float)
    dow.index = feat.index
    feat = pd.concat([feat, dow], axis=1)
    # Month dummies
    mon = pd.get_dummies(df['month'], prefix='month', drop_first=True, dtype=float)
    mon.index = feat.index
    feat = pd.concat([feat, mon], axis=1)
    return feat

# Build features for train and test
# Use nmd_data loaded from helpers -- drop NaN from rolling MA
nmd_clean = nmd_data.dropna(subset=['daily_decay_rate', 'decay_rate_ma30']).copy()
train_ml = nmd_clean[(nmd_clean['year'] >= 2017) & (nmd_clean['year'] <= 2022)].copy()
test_ml = nmd_clean[nmd_clean['year'] == 2023].copy().reset_index(drop=True)

X_train = build_features(train_ml)
y_train = train_ml['daily_decay_rate'].values

X_test = build_features(test_ml)
y_test = test_ml['daily_decay_rate'].values

# Align columns (ensure test has same dummy columns as train)
missing_cols = set(X_train.columns) - set(X_test.columns)
for c in missing_cols:
    X_test[c] = 0.0
X_test = X_test[X_train.columns]

print(f"ML Training set: {len(X_train)} obs, {X_train.shape[1]} features")
print(f"ML Test set:     {len(X_test)} obs")

# %% [markdown]
# ### Train OLS, Ridge, and Random Forest Models

# %%
# 1. OLS Regression (statsmodels)
X_train_ols = sm.add_constant(X_train)
X_test_ols = sm.add_constant(X_test)
ols_bt = sm.OLS(y_train, X_train_ols).fit()
pred_ols = ols_bt.predict(X_test_ols)

# 2. Ridge Regression (sklearn)
ridge_model = Ridge(alpha=1.0)
ridge_model.fit(X_train, y_train)
pred_ridge = ridge_model.predict(X_test)

# 3. Random Forest (sklearn)
rf_model = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)
pred_rf = rf_model.predict(X_test)

print("Models trained successfully:")
print(f"  OLS R-squared (train):         {ols_bt.rsquared:.4f}")
print(f"  Ridge R-squared (train):       {ridge_model.score(X_train, y_train):.4f}")
print(f"  Random Forest R-squared (train): {rf_model.score(X_train, y_train):.4f}")

# %% [markdown]
# ### ML-Predicted Balance Paths (2023)

# %%
# Simulate 2023 balance path using each model's predicted daily decay rates
# B(t+1) = B(t) * (1 - predicted_lambda_t)
B_start_ml = test_ml['Balance'].iloc[0]
actual_balance_ml = test_ml['Balance'].values
dates_test_ml = test_ml['Date'].values
n_test = len(test_ml)

def simulate_balance_path(start_balance, predicted_lambdas):
    """Simulate balance path from predicted daily decay rates."""
    path = np.zeros(len(predicted_lambdas))
    path[0] = start_balance
    for t in range(1, len(predicted_lambdas)):
        path[t] = path[t-1] * (1 - predicted_lambdas[t-1])
    return path

# Simple average path (baseline)
path_simple = simulate_balance_path(B_start_ml, np.full(n_test, lambda_daily_train))

# OLS-predicted path
pred_ols_clipped = np.clip(pred_ols, 0, 1)
path_ols = simulate_balance_path(B_start_ml, pred_ols_clipped)

# Ridge-predicted path
pred_ridge_clipped = np.clip(pred_ridge, 0, 1)
path_ridge = simulate_balance_path(B_start_ml, pred_ridge_clipped)

# Random Forest-predicted path
pred_rf_clipped = np.clip(pred_rf, 0, 1)
path_rf = simulate_balance_path(B_start_ml, pred_rf_clipped)

print("Balance paths simulated for 2023")

# %% [markdown]
# ### Model Comparison Metrics

# %%
def calc_metrics(actual, predicted, name):
    """Calculate RMSE and MAPE for a predicted balance path."""
    errors = actual - predicted
    rmse = np.sqrt(np.mean(errors**2))
    mape = np.mean(np.abs(errors) / actual * 100)
    return {'Model': name, 'RMSE': rmse, 'MAPE_%': mape}

metrics_list = [
    calc_metrics(actual_balance_ml, path_simple, 'Simple Average'),
    calc_metrics(actual_balance_ml, path_ols, 'OLS Regression'),
    calc_metrics(actual_balance_ml, path_ridge, 'Ridge Regression'),
    calc_metrics(actual_balance_ml, path_rf, 'Random Forest'),
]

# Also compute decay rate prediction accuracy (on test set)
def calc_decay_metrics(actual_decay, predicted_decay, name):
    """Calculate RMSE for decay rate predictions."""
    rmse = np.sqrt(mean_squared_error(actual_decay, predicted_decay))
    return {'Model': name, 'Decay_RMSE': rmse}

decay_metrics_list = [
    calc_decay_metrics(y_test, np.full(len(y_test), lambda_daily_train), 'Simple Average'),
    calc_decay_metrics(y_test, pred_ols, 'OLS Regression'),
    calc_decay_metrics(y_test, pred_ridge, 'Ridge Regression'),
    calc_decay_metrics(y_test, pred_rf, 'Random Forest'),
]

comparison_df = pd.DataFrame(metrics_list)
decay_comparison_df = pd.DataFrame(decay_metrics_list)
comparison_full = comparison_df.merge(decay_comparison_df, on='Model')

print("\n" + "="*80)
print("MODEL COMPARISON: BALANCE PATH PREDICTION (2023)")
print("="*80)
print(comparison_full.to_string(index=False))

comparison_full.to_csv('ml_model_comparison.csv', index=False)
print("\nSaved: ml_model_comparison.csv")

# %% [markdown]
# ### Chart: Actual vs All Model Predictions (2023)

# %%
fig, ax = plt.subplots(figsize=(14, 7))
ax.plot(dates_test_ml, actual_balance_ml, linewidth=2.5, color='#023047',
        label='Actual Balance')
ax.plot(dates_test_ml, path_simple, linewidth=1.5, linestyle='--', color='#E63946',
        label='Simple Average', alpha=0.8)
ax.plot(dates_test_ml, path_ols, linewidth=1.5, linestyle='-.', color='#2A9D8F',
        label='OLS Regression', alpha=0.8)
ax.plot(dates_test_ml, path_ridge, linewidth=1.5, linestyle=':', color='#F77F00',
        label='Ridge Regression', alpha=0.8)
ax.plot(dates_test_ml, path_rf, linewidth=1.5, linestyle='-', color='#6A0572',
        label='Random Forest', alpha=0.7)

ax.set_xlabel('Date', fontsize=12)
ax.set_ylabel('Balance', fontsize=12)
ax.set_title('ML Backtest: Actual vs Model-Predicted Balance Paths (2023)',
             fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# %% [markdown]
# ### Chart: RMSE Comparison

# %%
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Balance path RMSE
models = comparison_full['Model']
rmse_vals = comparison_full['RMSE']
colors_bar = ['#E63946', '#2A9D8F', '#F77F00', '#6A0572']

axes[0].bar(models, rmse_vals, color=colors_bar, edgecolor='black', alpha=0.8)
for i, (m, v) in enumerate(zip(models, rmse_vals)):
    axes[0].text(i, v + rmse_vals.max()*0.02, f'{v:,.0f}', ha='center', fontsize=9)
axes[0].set_ylabel('RMSE (Balance)', fontsize=11)
axes[0].set_title('Balance Path RMSE by Model', fontsize=12, fontweight='bold')
axes[0].grid(True, alpha=0.3, axis='y')
axes[0].tick_params(axis='x', rotation=20)

# MAPE
mape_vals = comparison_full['MAPE_%']
axes[1].bar(models, mape_vals, color=colors_bar, edgecolor='black', alpha=0.8)
for i, (m, v) in enumerate(zip(models, mape_vals)):
    axes[1].text(i, v + mape_vals.max()*0.02, f'{v:.2f}%', ha='center', fontsize=9)
axes[1].set_ylabel('MAPE (%)', fontsize=11)
axes[1].set_title('Balance Path MAPE by Model', fontsize=12, fontweight='bold')
axes[1].grid(True, alpha=0.3, axis='y')
axes[1].tick_params(axis='x', rotation=20)

plt.tight_layout()
plt.show()

# %% [markdown]
# ### Random Forest Feature Importance

# %%
rf_importances = pd.Series(rf_model.feature_importances_, index=X_train.columns)
rf_importances_sorted = rf_importances.sort_values(ascending=True)

fig, ax = plt.subplots(figsize=(10, max(6, len(rf_importances_sorted)*0.35)))
ax.barh(range(len(rf_importances_sorted)), rf_importances_sorted.values,
        color='#2A9D8F', edgecolor='black', alpha=0.8)
ax.set_yticks(range(len(rf_importances_sorted)))
ax.set_yticklabels(rf_importances_sorted.index, fontsize=9)
ax.set_xlabel('Feature Importance (MDI)', fontsize=11)
ax.set_title('Random Forest Feature Importance',
             fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.show()

print("Top 5 features by importance:")
for feat, imp in rf_importances.sort_values(ascending=False).head(5).items():
    print(f"  {feat}: {imp:.4f}")

# %%
print("\n" + "="*80)
print("SECTION 5c SUMMARY")
print("="*80)
print("\n1. BASELINE BACKTEST")
print(f"   Pure Decay RMSE: {rmse:,.2f}  |  MAPE: {mape:.2f}%")
print(f"   Monthly Re-Anchored RMSE: {rmse_monthly:,.2f}  |  MAPE: {mape_monthly:.2f}%")
print("\n2. ML MODEL COMPARISON (Balance Path)")
for _, row in comparison_full.iterrows():
    print(f"   {row['Model']:20s}  RMSE={row['RMSE']:>10,.2f}  MAPE={row['MAPE_%']:>6.2f}%")
print("\n3. KEY INSIGHTS")
print("   - All models show drift from actual due to ignoring inflows")
print("   - ML models capture time-varying decay patterns better")
print("   - Random Forest captures non-linear effects (balance level, seasonality)")
print("   - For IRRBB purposes, the simple average remains the primary model")
print("   - ML provides validation and enhanced understanding of decay dynamics")

print("\nSection 5c Complete: 6 charts generated + ML model comparison")

# %%
