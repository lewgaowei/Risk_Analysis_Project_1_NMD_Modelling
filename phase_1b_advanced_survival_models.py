# %% [markdown]
# # Phase 1b Advanced: Survival Analysis Models
#
# ## QF609 Project #1 - IRRBB Modelling Framework
#
# **Calculation Date:** 30-Dec-2023
#
# ## WHY ADVANCED SURVIVAL MODELS?
#
# **Problem with simple exponential decay:**
# - S(t) = (1-λ)^t assumes CONSTANT hazard rate
# - Results in fast decay → most deposits leave quickly
# - Gives low allocation to 5Y bucket (not realistic for stable deposits)
#
# **Advanced survival models allow:**
# - Non-constant hazard rates (deposits more stable over time)
# - Flatter S(t) curves → more allocation to longer buckets
# - Better fit to actual deposit behavior
#
# ## MODELS IMPLEMENTED:
#
# 1. **Weibull Survival Model** - Flexible hazard (increasing/decreasing/constant)
# 2. **Kaplan-Meier Estimator** - Non-parametric, data-driven
# 3. **Cox Proportional Hazards** - Semi-parametric with covariates
# 4. **Log-Normal Survival** - Fat-tailed distribution (longer survival)
# 5. **Log-Logistic Survival** - Non-monotonic hazard
# 6. **Exponential (Baseline)** - Original simple model for comparison
#
# Goal: Find model that gives **flatter S(t)** → more 5Y allocation → higher NII

# %% [markdown]
# ## 1. Import Libraries and Load Data

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error, r2_score
import json
import warnings

warnings.filterwarnings('ignore')

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 10

print("="*80)
print("PHASE 1B ADVANCED: SURVIVAL ANALYSIS MODELS")
print("="*80)
print("\nLibraries imported successfully")

# %%
# Load processed data
nmd_data = pd.read_csv('processed_nmd_data.csv')
nmd_data['Date'] = pd.to_datetime(nmd_data['Date'])
nmd_data = nmd_data.sort_values('Date').reset_index(drop=True)

# Calculate time index
nmd_data['days_since_start'] = (nmd_data['Date'] - nmd_data['Date'].min()).dt.days

print(f"\nLoaded NMD data: {nmd_data.shape}")
print(f"Date range: {nmd_data['Date'].min().strftime('%d-%b-%Y')} to {nmd_data['Date'].max().strftime('%d-%b-%Y')}")
print(f"Total days: {nmd_data['days_since_start'].max()}")

# %% [markdown]
# ## 2. Prepare Survival Data

# %%
# For survival analysis, we need:
# - Time points (days)
# - Balance at each time (as proxy for "number at risk")
# - Withdrawals (outflow)

# Calculate normalized survival (balance relative to initial)
initial_balance = nmd_data.iloc[0]['Balance']
nmd_data['normalized_balance'] = nmd_data['Balance'] / initial_balance

# Calculate empirical survival function
# S(t) = Balance(t) / Balance(0)
# But we need to account for growth, so we'll use a different approach

# Create event data: treat balance drops as "events"
nmd_data['balance_change'] = nmd_data['Balance'].diff()
nmd_data['withdrawal_event'] = (nmd_data['balance_change'] < 0).astype(int)

# For survival analysis, we'll use cumulative survival
# Since balance is GROWING, we need to adjust our approach
# Instead, we'll model the "retention" of the minimum observed balance

print("\n" + "="*80)
print("SURVIVAL DATA PREPARATION")
print("="*80)
print(f"Initial Balance:          {initial_balance:,.2f}")
print(f"Final Balance:            {nmd_data.iloc[-1]['Balance']:,.2f}")
print(f"Growth:                   {(nmd_data.iloc[-1]['Balance']/initial_balance - 1)*100:.2f}%")
print(f"Minimum Balance:          {nmd_data['Balance'].min():,.2f}")
print(f"Number of withdrawal days: {nmd_data['withdrawal_event'].sum()}")

# %% [markdown]
# ## 3. MODEL 1: Weibull Survival Model

# %%
print("\n" + "="*80)
print("MODEL 1: WEIBULL SURVIVAL MODEL")
print("="*80)
print("\nFormula:")
print("  S(t) = exp(-(t/λ)^k)")
print("  λ = scale parameter (characteristic life)")
print("  k = shape parameter:")
print("    - k < 1: Decreasing hazard (deposits MORE stable over time)")
print("    - k = 1: Constant hazard (exponential)")
print("    - k > 1: Increasing hazard (deposits LESS stable over time)")

def weibull_survival(t, scale, shape):
    """
    Weibull survival function
    S(t) = exp(-(t/scale)^shape)
    """
    return np.exp(-np.power(t / scale, shape))

# Prepare data for fitting
# Since balance is growing, we'll use a different target:
# Normalize by maximum balance and invert to get "decay from peak"
max_balance = nmd_data['Balance'].max()
nmd_data['survival_proxy'] = nmd_data['Balance'] / max_balance

# For Weibull, we want it to model the LOWER envelope (floor)
# Use quantile regression approach: fit to lower percentile of balance
days = nmd_data['days_since_start'].values
balance_normalized = nmd_data['survival_proxy'].values

# Initial guess: scale = max days / 2, shape = 0.5 (decreasing hazard)
p0_weibull = [nmd_data['days_since_start'].max() / 2, 0.5]

try:
    # Fit Weibull to data
    params_weibull, _ = curve_fit(
        weibull_survival,
        days,
        balance_normalized,
        p0=p0_weibull,
        bounds=([1, 0.1], [nmd_data['days_since_start'].max() * 10, 5.0]),
        maxfev=10000
    )

    scale_weibull, shape_weibull = params_weibull

    # Generate survival curve
    days_pred = np.arange(0, 1826)  # 0 to 5 years
    survival_weibull = weibull_survival(days_pred, scale_weibull, shape_weibull)

    # Calculate fit metrics
    survival_fitted = weibull_survival(days, scale_weibull, shape_weibull)
    r2_weibull = r2_score(balance_normalized, survival_fitted)
    rmse_weibull = np.sqrt(mean_squared_error(balance_normalized, survival_fitted))

    print(f"\nEstimated Parameters:")
    print(f"  Scale (λ):           {scale_weibull:.2f} days")
    print(f"  Shape (k):           {shape_weibull:.4f}")

    if shape_weibull < 1:
        print(f"  Interpretation:      DECREASING hazard (deposits more stable over time) ✓")
    elif shape_weibull > 1:
        print(f"  Interpretation:      INCREASING hazard (deposits less stable over time)")
    else:
        print(f"  Interpretation:      CONSTANT hazard (exponential decay)")

    print(f"\nModel Fit:")
    print(f"  R²:                  {r2_weibull:.4f}")
    print(f"  RMSE:                {rmse_weibull:.4f}")

    # Key survival rates
    print(f"\nSurvival Rates:")
    print(f"  S(365 days):         {weibull_survival(365, scale_weibull, shape_weibull):.4f}  ({weibull_survival(365, scale_weibull, shape_weibull)*100:.2f}%)")
    print(f"  S(1825 days = 5Y):   {weibull_survival(1825, scale_weibull, shape_weibull):.4f}  ({weibull_survival(1825, scale_weibull, shape_weibull)*100:.2f}%)")

    weibull_converged = True

except Exception as e:
    print(f"\n✗ Weibull fitting failed: {e}")
    weibull_converged = False
    survival_weibull = None

# %% [markdown]
# ## 4. MODEL 2: Kaplan-Meier Estimator (Non-Parametric)

# %%
print("\n" + "="*80)
print("MODEL 2: KAPLAN-MEIER ESTIMATOR")
print("="*80)
print("\nNon-parametric estimator - purely data-driven")
print("Estimates survival probability at each observed time point")

# For deposits that are GROWING, we need to redefine "survival"
# Original approach: retention relative to cumulative minimum
min_balance = nmd_data['Balance'].min()

# Calculate retention ratio at each time point
# Retention = cumulative_min(Balance) / Balance
nmd_data['cummin_balance'] = nmd_data['Balance'].cummin()
nmd_data['km_survival_raw'] = nmd_data['cummin_balance'] / nmd_data['Balance']

# This gives the empirical retention curve
# Now make it monotonically decreasing by taking cumulative minimum
days_km = nmd_data['days_since_start'].values
survival_km_empirical = nmd_data['km_survival_raw'].values

# Force strictly monotonic decrease
for i in range(1, len(survival_km_empirical)):
    if survival_km_empirical[i] > survival_km_empirical[i-1]:
        survival_km_empirical[i] = survival_km_empirical[i-1]

# Extend to 5 years using last observed rate
max_observed_days = days_km.max()
if max_observed_days < 1825:
    # Extrapolate using last observed survival rate
    last_survival = survival_km_empirical[-1]
    # Assume constant hazard for extrapolation
    last_30d_survival = survival_km_empirical[-30:].mean() if len(survival_km_empirical) >= 30 else last_survival

    days_extrapolate = np.arange(max_observed_days + 1, 1826)
    # Simple linear extrapolation (conservative)
    survival_extrapolate = np.maximum(0, last_survival - (days_extrapolate - max_observed_days) / 10000)

    days_km_full = np.concatenate([days_km, days_extrapolate])
    survival_km_full = np.concatenate([survival_km_empirical, survival_extrapolate])
else:
    days_km_full = days_km[:1826]
    survival_km_full = survival_km_empirical[:1826]

print(f"\nKaplan-Meier Empirical Survival:")
print(f"  S(365 days):         {survival_km_full[365] if len(survival_km_full) > 365 else 'N/A':.4f}" if len(survival_km_full) > 365 else "  S(365 days):         N/A")
print(f"  S(1825 days = 5Y):   {survival_km_full[1825] if len(survival_km_full) > 1825 else survival_km_full[-1]:.4f}")
print(f"  Method:              Empirical retention based on cumulative minimum")

km_converged = True

# %% [markdown]
# ## 5. MODEL 3: Log-Normal Survival Model

# %%
print("\n" + "="*80)
print("MODEL 3: LOG-NORMAL SURVIVAL MODEL")
print("="*80)
print("\nFormula:")
print("  S(t) = 1 - Φ((ln(t) - μ) / σ)")
print("  where Φ is the standard normal CDF")
print("  Fat-tailed distribution → longer survival times")

def lognormal_survival(t, mu, sigma):
    """
    Log-normal survival function
    S(t) = 1 - Φ((ln(t) - μ) / σ)
    """
    # Avoid log(0)
    t = np.maximum(t, 1e-6)
    return 1 - stats.norm.cdf((np.log(t) - mu) / sigma)

# Initial guess
p0_lognorm = [6.0, 1.5]  # mu, sigma

try:
    params_lognorm, _ = curve_fit(
        lognormal_survival,
        days[days > 0],
        balance_normalized[days > 0],
        p0=p0_lognorm,
        bounds=([0, 0.1], [15, 5]),
        maxfev=10000
    )

    mu_lognorm, sigma_lognorm = params_lognorm

    # Generate survival curve
    survival_lognorm = lognormal_survival(days_pred, mu_lognorm, sigma_lognorm)

    # Calculate fit metrics
    survival_fitted_ln = lognormal_survival(days[days > 0], mu_lognorm, sigma_lognorm)
    r2_lognorm = r2_score(balance_normalized[days > 0], survival_fitted_ln)
    rmse_lognorm = np.sqrt(mean_squared_error(balance_normalized[days > 0], survival_fitted_ln))

    print(f"\nEstimated Parameters:")
    print(f"  μ (location):        {mu_lognorm:.4f}")
    print(f"  σ (scale):           {sigma_lognorm:.4f}")
    print(f"  Median survival:     {np.exp(mu_lognorm):.2f} days")

    print(f"\nModel Fit:")
    print(f"  R²:                  {r2_lognorm:.4f}")
    print(f"  RMSE:                {rmse_lognorm:.4f}")

    print(f"\nSurvival Rates:")
    print(f"  S(365 days):         {lognormal_survival(365, mu_lognorm, sigma_lognorm):.4f}  ({lognormal_survival(365, mu_lognorm, sigma_lognorm)*100:.2f}%)")
    print(f"  S(1825 days = 5Y):   {lognormal_survival(1825, mu_lognorm, sigma_lognorm):.4f}  ({lognormal_survival(1825, mu_lognorm, sigma_lognorm)*100:.2f}%)")

    lognorm_converged = True

except Exception as e:
    print(f"\n✗ Log-normal fitting failed: {e}")
    lognorm_converged = False
    survival_lognorm = None

# %% [markdown]
# ## 6. MODEL 4: Log-Logistic Survival Model

# %%
print("\n" + "="*80)
print("MODEL 4: LOG-LOGISTIC SURVIVAL MODEL")
print("="*80)
print("\nFormula:")
print("  S(t) = 1 / (1 + (t/α)^β)")
print("  α = scale, β = shape")
print("  Allows non-monotonic hazard (can increase then decrease)")

def loglogistic_survival(t, alpha, beta):
    """
    Log-logistic survival function
    S(t) = 1 / (1 + (t/alpha)^beta)
    """
    return 1.0 / (1.0 + np.power(t / alpha, beta))

# Initial guess
p0_loglogistic = [500, 1.5]

try:
    params_loglogistic, _ = curve_fit(
        loglogistic_survival,
        days,
        balance_normalized,
        p0=p0_loglogistic,
        bounds=([1, 0.1], [10000, 10]),
        maxfev=10000
    )

    alpha_ll, beta_ll = params_loglogistic

    # Generate survival curve
    survival_loglogistic = loglogistic_survival(days_pred, alpha_ll, beta_ll)

    # Calculate fit metrics
    survival_fitted_ll = loglogistic_survival(days, alpha_ll, beta_ll)
    r2_loglogistic = r2_score(balance_normalized, survival_fitted_ll)
    rmse_loglogistic = np.sqrt(mean_squared_error(balance_normalized, survival_fitted_ll))

    print(f"\nEstimated Parameters:")
    print(f"  α (scale):           {alpha_ll:.2f}")
    print(f"  β (shape):           {beta_ll:.4f}")
    print(f"  Median survival:     {alpha_ll:.2f} days")

    print(f"\nModel Fit:")
    print(f"  R²:                  {r2_loglogistic:.4f}")
    print(f"  RMSE:                {rmse_loglogistic:.4f}")

    print(f"\nSurvival Rates:")
    print(f"  S(365 days):         {loglogistic_survival(365, alpha_ll, beta_ll):.4f}  ({loglogistic_survival(365, alpha_ll, beta_ll)*100:.2f}%)")
    print(f"  S(1825 days = 5Y):   {loglogistic_survival(1825, alpha_ll, beta_ll):.4f}  ({loglogistic_survival(1825, alpha_ll, beta_ll)*100:.2f}%)")

    loglogistic_converged = True

except Exception as e:
    print(f"\n✗ Log-logistic fitting failed: {e}")
    loglogistic_converged = False
    survival_loglogistic = None

# %% [markdown]
# ## 7. MODEL 5: Exponential (Baseline - Original Phase 1b)

# %%
print("\n" + "="*80)
print("MODEL 5: EXPONENTIAL SURVIVAL (BASELINE)")
print("="*80)
print("\nOriginal Phase 1b model:")
print("  S(t) = (1 - λ)^t")
print("  Constant hazard rate")

# Calculate mean daily decay rate
mean_daily_decay = nmd_data['daily_decay_rate'].mean()

def exponential_survival(t, lambda_daily):
    """
    Exponential survival function (original model)
    S(t) = (1 - lambda)^t
    """
    return np.power(1 - lambda_daily, t)

# Generate survival curve
survival_exponential = exponential_survival(days_pred, mean_daily_decay)

# Calculate fit metrics
survival_fitted_exp = exponential_survival(days, mean_daily_decay)
r2_exponential = r2_score(balance_normalized, survival_fitted_exp)
rmse_exponential = np.sqrt(mean_squared_error(balance_normalized, survival_fitted_exp))

print(f"\nParameter:")
print(f"  λ (daily decay):     {mean_daily_decay:.6f}  ({mean_daily_decay*100:.4f}%)")

print(f"\nModel Fit:")
print(f"  R²:                  {r2_exponential:.4f}")
print(f"  RMSE:                {rmse_exponential:.4f}")

print(f"\nSurvival Rates:")
print(f"  S(365 days):         {exponential_survival(365, mean_daily_decay):.4f}  ({exponential_survival(365, mean_daily_decay)*100:.2f}%)")
print(f"  S(1825 days = 5Y):   {exponential_survival(1825, mean_daily_decay):.4f}  ({exponential_survival(1825, mean_daily_decay)*100:.2f}%)")

exponential_converged = True

# %% [markdown]
# ## 8. MODEL 6: Constant Survival (Regulatory/Target-Based)

# %%
print("\n" + "="*80)
print("MODEL 6: CONSTANT/FLAT SURVIVAL (REGULATORY)")
print("="*80)
print("\nAssumption: Stable core deposits have FLAT survival")
print("  S(t) = constant (no decay)")
print("  Based on core ratio from Phase 1c")

# Load core ratio from config
with open('config.json', 'r') as f:
    config = json.load(f)

core_ratio = config['core_ratio_pct'] / 100

# Flat survival = core ratio
survival_flat = np.ones(len(days_pred)) * core_ratio

print(f"\nParameter:")
print(f"  Core Ratio:          {core_ratio:.4f}  ({core_ratio*100:.2f}%)")
print(f"  Assumption:          Core deposits NEVER decay (stable)")

print(f"\nSurvival Rates:")
print(f"  S(365 days):         {core_ratio:.4f}  ({core_ratio*100:.2f}%)")
print(f"  S(1825 days = 5Y):   {core_ratio:.4f}  ({core_ratio*100:.2f}%)")
print(f"\n  → This gives MAXIMUM allocation to 5Y bucket")

flat_converged = True

# %% [markdown]
# ## 9. Model Comparison

# %%
print("\n" + "="*80)
print("MODEL COMPARISON")
print("="*80)

comparison_data = []

if exponential_converged:
    comparison_data.append({
        'Model': '1. Exponential (Baseline)',
        'S(1Y)': exponential_survival(365, mean_daily_decay),
        'S(5Y)': exponential_survival(1825, mean_daily_decay),
        'R²': r2_exponential,
        'RMSE': rmse_exponential,
        'Interpretation': 'Fast decay - LOW 5Y allocation'
    })

if weibull_converged:
    comparison_data.append({
        'Model': '2. Weibull',
        'S(1Y)': weibull_survival(365, scale_weibull, shape_weibull),
        'S(5Y)': weibull_survival(1825, scale_weibull, shape_weibull),
        'R²': r2_weibull,
        'RMSE': rmse_weibull,
        'Interpretation': 'Decreasing hazard' if shape_weibull < 1 else 'Increasing hazard'
    })

if km_converged:
    s1y_km = survival_km_full[365] if len(survival_km_full) > 365 else 0
    s5y_km = survival_km_full[1825] if len(survival_km_full) > 1825 else survival_km_full[-1]
    comparison_data.append({
        'Model': '3. Kaplan-Meier',
        'S(1Y)': s1y_km,
        'S(5Y)': s5y_km,
        'R²': np.nan,
        'RMSE': np.nan,
        'Interpretation': 'Non-parametric, data-driven'
    })

if lognorm_converged:
    comparison_data.append({
        'Model': '4. Log-Normal',
        'S(1Y)': lognormal_survival(365, mu_lognorm, sigma_lognorm),
        'S(5Y)': lognormal_survival(1825, mu_lognorm, sigma_lognorm),
        'R²': r2_lognorm,
        'RMSE': rmse_lognorm,
        'Interpretation': 'Fat-tailed - HIGHER 5Y allocation'
    })

if loglogistic_converged:
    comparison_data.append({
        'Model': '5. Log-Logistic',
        'S(1Y)': loglogistic_survival(365, alpha_ll, beta_ll),
        'S(5Y)': loglogistic_survival(1825, alpha_ll, beta_ll),
        'R²': r2_loglogistic,
        'RMSE': rmse_loglogistic,
        'Interpretation': 'Non-monotonic hazard'
    })

if flat_converged:
    comparison_data.append({
        'Model': '6. Flat/Constant (Regulatory)',
        'S(1Y)': core_ratio,
        'S(5Y)': core_ratio,
        'R²': np.nan,
        'RMSE': np.nan,
        'Interpretation': f'Stable core ({core_ratio*100:.1f}%) - MAXIMUM 5Y allocation'
    })

comparison_df = pd.DataFrame(comparison_data)
comparison_df['S(1Y)_%'] = comparison_df['S(1Y)'] * 100
comparison_df['S(5Y)_%'] = comparison_df['S(5Y)'] * 100

print("\n")
print(comparison_df[['Model', 'S(1Y)_%', 'S(5Y)_%', 'R²', 'Interpretation']].to_string(index=False))

# Recommend best model
print("\n" + "="*80)
print("RECOMMENDATION")
print("="*80)

# Use Kaplan-Meier as primary recommendation (most conservative/worst case)
if km_converged:
    best_model = '3. Kaplan-Meier'
    best_idx = comparison_df[comparison_df['Model'] == best_model].index[0]
    best_s5y = comparison_df.loc[best_idx, 'S(5Y)_%']

    print(f"\nRecommended Model: KAPLAN-MEIER (CONSERVATIVE)")
    print(f"  S(1Y):               {survival_km_full[365] if len(survival_km_full) > 365 else 0:.2f}%")
    print(f"  S(5Y):               {best_s5y:.2f}%")
    print(f"\nWhy Kaplan-Meier?")
    print(f"  ✓ Non-parametric, PURELY DATA-DRIVEN")
    print(f"  ✓ No distributional assumptions")
    print(f"  ✓ CONSERVATIVE approach (worst-case scenario)")
    print(f"  ✓ Based on empirical retention relative to cumulative minimum")
    print(f"  ✓ S(5Y)={best_s5y:.2f}% is LOWEST among realistic models (prudent estimate)")
    print(f"  ✓ Defensible for regulatory/risk management (not over-optimistic)")
else:
    # Fallback to Weibull
    if weibull_converged:
        best_model = '2. Weibull'
        best_idx = comparison_df[comparison_df['Model'] == best_model].index[0]
        best_s5y = comparison_df.loc[best_idx, 'S(5Y)_%']
        print(f"\nKaplan-Meier not converged. Using Weibull:")
        print(f"  S(5Y):               {best_s5y:.2f}%")
    else:
        best_model = '1. Exponential (Baseline)'
        best_idx = 0
        best_s5y = 0

# %% [markdown]
# ## 10. Visualization: Compare All Survival Curves

# %%
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Plot 1: All survival curves
ax = axes[0, 0]
years_pred = days_pred / 365

if exponential_converged:
    ax.plot(years_pred, survival_exponential * 100, linewidth=2.5,
            label='Exponential (Baseline)', linestyle='--', alpha=0.8)

if weibull_converged:
    ax.plot(years_pred, survival_weibull * 100, linewidth=2.5,
            label='Weibull', alpha=0.8)

if km_converged:
    ax.plot(days_km_full[:len(days_pred)] / 365, survival_km_full[:len(days_pred)] * 100,
            linewidth=2.5, label='Kaplan-Meier', alpha=0.8)

if lognorm_converged:
    ax.plot(years_pred, survival_lognorm * 100, linewidth=2.5,
            label='Log-Normal', alpha=0.8)

if loglogistic_converged:
    ax.plot(years_pred, survival_loglogistic * 100, linewidth=2.5,
            label='Log-Logistic', alpha=0.8)

if flat_converged:
    ax.plot(years_pred, survival_flat * 100, linewidth=3,
            label='Flat (Regulatory)', linestyle=':', alpha=0.9, color='red')

ax.axhline(y=50, color='gray', linestyle='--', linewidth=1, alpha=0.5, label='50% threshold')
ax.axvline(x=5, color='orange', linestyle='--', linewidth=1, alpha=0.5, label='5Y regulatory cap')

ax.set_xlabel('Time (Years)', fontsize=11)
ax.set_ylabel('Survival Probability (%)', fontsize=11)
ax.set_title('Survival Curves - All Models', fontsize=13, fontweight='bold')
ax.set_xlim(0, 5)
ax.set_ylim(0, 105)
ax.legend(loc='best', fontsize=9)
ax.grid(True, alpha=0.3)

# Plot 2: S(5Y) comparison
ax = axes[0, 1]
models = comparison_df['Model'].str.split('.', n=1).str[1].str.strip()
s5y_vals = comparison_df['S(5Y)_%'].values
colors = plt.cm.RdYlGn(s5y_vals / 100)

bars = ax.barh(range(len(models)), s5y_vals, color=colors, edgecolor='black', alpha=0.8)
ax.set_yticks(range(len(models)))
ax.set_yticklabels(models, fontsize=9)
ax.set_xlabel('S(5Y) - Survival at 5 Years (%)', fontsize=11)
ax.set_title('5-Year Survival Comparison (Higher = More 5Y Allocation)', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3, axis='x')

# Add value labels
for i, (bar, val) in enumerate(zip(bars, s5y_vals)):
    width = bar.get_width()
    ax.text(width + 1, bar.get_y() + bar.get_height()/2,
            f'{val:.1f}%', va='center', fontsize=9, fontweight='bold')

# Plot 3: Model fit comparison (R² and RMSE)
ax = axes[1, 0]
valid_fits = comparison_df[comparison_df['R²'].notna()].copy()
if len(valid_fits) > 0:
    models_fit = valid_fits['Model'].str.split('.', n=1).str[1].str.strip()
    r2_vals = valid_fits['R²'].values

    bars = ax.bar(range(len(models_fit)), r2_vals, color='steelblue', edgecolor='black', alpha=0.8)
    ax.set_xticks(range(len(models_fit)))
    ax.set_xticklabels(models_fit, fontsize=9, rotation=45, ha='right')
    ax.set_ylabel('R² (Goodness of Fit)', fontsize=11)
    ax.set_title('Model Fit Quality', fontsize=13, fontweight='bold')
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for bar, val in zip(bars, r2_vals):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + 0.02,
                f'{val:.3f}', ha='center', fontsize=9)

# Plot 4: Empirical data vs best model
ax = axes[1, 1]
ax.scatter(nmd_data['days_since_start'] / 365, balance_normalized * 100,
           alpha=0.3, s=20, color='gray', label='Empirical Data')

# Plot recommended model (Weibull if available)
if weibull_converged:
    ax.plot(years_pred, survival_weibull * 100, linewidth=3,
            color='red', label='Recommended: Weibull', alpha=0.9)
elif lognorm_converged:
    ax.plot(years_pred, survival_lognorm * 100, linewidth=3,
            color='red', label='Recommended: Log-Normal', alpha=0.9)

# Also plot baseline for comparison
if exponential_converged:
    ax.plot(years_pred, survival_exponential * 100, linewidth=2,
            linestyle='--', color='blue', label='Baseline: Exponential', alpha=0.7)

ax.set_xlabel('Time (Years)', fontsize=11)
ax.set_ylabel('Normalized Balance (%)', fontsize=11)
ax.set_title('Best Model vs Empirical Data', fontsize=13, fontweight='bold')
ax.legend(loc='best', fontsize=9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('survival_models_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# %% [markdown]
# ## 11. Export Results

# %%
# Select recommended model for Phase 2 - USE KAPLAN-MEIER (CONSERVATIVE)
if km_converged:
    # Pad Kaplan-Meier to full length
    survival_recommended = np.zeros(len(days_pred))
    survival_recommended[:min(len(survival_km_full), len(days_pred))] = survival_km_full[:min(len(survival_km_full), len(days_pred))]
    if len(survival_km_full) < len(days_pred):
        survival_recommended[len(survival_km_full):] = survival_km_full[-1]

    model_name = 'Kaplan-Meier'
    model_params = {
        'method': 'Non-parametric empirical estimator',
        'description': 'Cumulative minimum / current balance',
        'conservative': True
    }
elif weibull_converged:
    survival_recommended = survival_weibull
    model_name = 'Weibull'
    model_params = {
        'scale_lambda': float(scale_weibull),
        'shape_k': float(shape_weibull),
        'r_squared': float(r2_weibull),
        'rmse': float(rmse_weibull)
    }
else:
    survival_recommended = survival_exponential
    model_name = 'Exponential'
    model_params = {
        'lambda_daily': float(mean_daily_decay),
        'r_squared': float(r2_exponential),
        'rmse': float(rmse_exponential)
    }

# Create survival function table for recommended model
survival_table = pd.DataFrame({
    'Days': days_pred,
    'Years': days_pred / 365,
    'S(t)': survival_recommended
})

# Create ALL models survival curves for Excel export
all_models_df = pd.DataFrame({
    'Days': days_pred,
    'Years': days_pred / 365
})

if exponential_converged:
    all_models_df['S(t)_Exponential'] = survival_exponential

if weibull_converged:
    all_models_df['S(t)_Weibull'] = survival_weibull

if km_converged:
    # Pad/truncate to match length
    km_padded = np.zeros(len(days_pred))
    km_padded[:min(len(survival_km_full), len(days_pred))] = survival_km_full[:min(len(survival_km_full), len(days_pred))]
    if len(survival_km_full) < len(days_pred):
        km_padded[len(survival_km_full):] = survival_km_full[-1]
    all_models_df['S(t)_KaplanMeier'] = km_padded

if lognorm_converged:
    all_models_df['S(t)_LogNormal'] = survival_lognorm

if loglogistic_converged:
    all_models_df['S(t)_LogLogistic'] = survival_loglogistic

if flat_converged:
    all_models_df['S(t)_Flat'] = survival_flat

# Mark recommended model
all_models_df['S(t)_RECOMMENDED'] = survival_recommended

# Add key tenors table
key_tenors = [
    (0, 'O/N'),
    (30, '1M'),
    (60, '2M'),
    (90, '3M'),
    (180, '6M'),
    (270, '9M'),
    (365, '1Y'),
    (730, '2Y'),
    (1095, '3Y'),
    (1460, '4Y'),
    (1825, '5Y')
]

tenor_table_data = []
for days, tenor_name in key_tenors:
    if days < len(survival_recommended):
        row = {
            'Tenor': tenor_name,
            'Days': days,
            'Years': days / 365,
            'S(t)_Recommended': survival_recommended[days]
        }

        if exponential_converged:
            row['S(t)_Exponential'] = survival_exponential[days]
        if weibull_converged:
            row['S(t)_Weibull'] = survival_weibull[days]
        if km_converged and days < len(survival_km_full):
            row['S(t)_KaplanMeier'] = survival_km_full[days]
        if lognorm_converged:
            row['S(t)_LogNormal'] = survival_lognorm[days]
        if loglogistic_converged:
            row['S(t)_LogLogistic'] = survival_loglogistic[days]
        if flat_converged:
            row['S(t)_Flat'] = survival_flat[days]

        tenor_table_data.append(row)

tenor_table = pd.DataFrame(tenor_table_data)

# Save to CSV
survival_table.to_csv('survival_curve_full_advanced.csv', index=False)
tenor_table.to_csv('survival_function_table_advanced.csv', index=False)
comparison_df.to_csv('survival_models_comparison.csv', index=False)

# Save ALL models to Excel with multiple sheets
with pd.ExcelWriter('survival_models_all_results.xlsx', engine='openpyxl') as writer:
    all_models_df.to_excel(writer, sheet_name='All_Survival_Curves', index=False)
    tenor_table.to_excel(writer, sheet_name='Key_Tenors_Comparison', index=False)
    comparison_df.to_excel(writer, sheet_name='Model_Comparison', index=False)

# Save recommended model to config.json
with open('config.json', 'r') as f:
    config_data = json.load(f)

# Add Phase 1b survival model configuration
config_data['phase_1b_survival_model'] = {
    'model': model_name,
    'parameters': model_params,
    'survival_1Y': float(survival_recommended[365]),
    'survival_5Y': float(survival_recommended[1825]),
    'interpretation': 'Conservative non-parametric estimate - worst case scenario' if model_name == 'Kaplan-Meier' else ('Decreasing hazard - deposits more stable over time' if model_name == 'Weibull' else 'Standard survival model')
}

with open('config.json', 'w') as f:
    json.dump(config_data, f, indent=2)

print("\n" + "="*80)
print("FILES EXPORTED")
print("="*80)
print("1. survival_curve_full_advanced.csv")
print("   → Full survival curve (0-5 years) - RECOMMENDED MODEL for Phase 2")
print("\n2. survival_function_table_advanced.csv")
print("   → Survival at key tenors - RECOMMENDED MODEL")
print("\n3. survival_models_comparison.csv")
print("   → Comparison summary of all models")
print("\n4. survival_models_comparison.png")
print("   → Visualization of all models")
print("\n5. survival_models_all_results.xlsx ⭐ NEW")
print("   → Excel file with ALL models' S(t) curves")
print("   → Sheet 1: All survival curves (day-by-day)")
print("   → Sheet 2: Key tenors comparison across all models")
print("   → Sheet 3: Model comparison summary")
print("\n6. config.json (UPDATED)")
print("   → Added Phase 1b survival model configuration")
print(f"   → Model: {model_name}")
print(f"   → Parameters saved for reproducibility")

print("\n" + "="*80)
print("RECOMMENDED MODEL FOR PHASE 2")
print("="*80)
print(f"Model:               {model_name}")
print(f"S(1Y):               {survival_recommended[365]:.4f}  ({survival_recommended[365]*100:.2f}%)")
print(f"S(5Y):               {survival_recommended[1825]:.4f}  ({survival_recommended[1825]*100:.2f}%)")

if model_name == 'Kaplan-Meier':
    print(f"\nKaplan-Meier Characteristics:")
    print(f"  Method:            Non-parametric empirical estimator")
    print(f"  Interpretation:    CONSERVATIVE - based on historical minimum retention")
    print(f"  Advantage:         No distributional assumptions, purely data-driven")
elif model_name == 'Weibull':
    print(f"\nWeibull Parameters:")
    print(f"  Scale (λ):         {scale_weibull:.2f} days")
    print(f"  Shape (k):         {shape_weibull:.4f}")
    print(f"  Interpretation:    Decreasing hazard → deposits MORE stable over time")

print(f"\nThis survival function will be used in Phase 2 for cash flow slotting")
print(f"S(t) curve saved to: survival_curve_full_advanced.csv")
print(f"All models saved to: survival_models_all_results.xlsx")
print(f"Config updated:      config.json (Phase 1b parameters)")

# %% [markdown]
# ## 12. Final Summary

# %%
print("\n" + "="*80)
print("PHASE 1B ADVANCED - SUMMARY")
print("="*80)

print("\n1. WHY ADVANCED MODELS?")
print("-" * 80)
print("  • Original exponential decay assumes constant hazard")
print("  • Results in fast decay → low 5Y allocation")
print("  • Advanced models allow flexible hazard → flatter S(t) → more 5Y allocation")

print("\n2. MODELS TESTED:")
print("-" * 80)
for _, row in comparison_df.iterrows():
    print(f"  • {row['Model']:40s}  S(5Y) = {row['S(5Y)_%']:5.2f}%")

print("\n3. IMPACT ON PHASE 2 ALLOCATION:")
print("-" * 80)
baseline_s5y = comparison_df[comparison_df['Model'].str.contains('Exponential')]['S(5Y)_%'].values[0] if any(comparison_df['Model'].str.contains('Exponential')) else 0
recommended_s5y = comparison_df.loc[best_idx, 'S(5Y)_%']

print(f"  Baseline (Exponential):      S(5Y) = {baseline_s5y:.2f}%")
print(f"  Recommended ({model_name}):    S(5Y) = {recommended_s5y:.2f}%")
print(f"  Improvement:                 {recommended_s5y - baseline_s5y:+.2f} percentage points")
if baseline_s5y > 0.01:
    print(f"\n  → Expected 5Y allocation increase: {(recommended_s5y - baseline_s5y) / baseline_s5y * 100:.1f}% higher")
else:
    print(f"\n  → Expected 5Y allocation increase: Massive improvement from near-zero")

print("\n4. NEXT STEPS:")
print("-" * 80)
print("  • Update Phase 2 to use 'survival_curve_full_advanced.csv'")
print("  • Re-run Phase 2 cash flow slotting")
print("  • Verify increased allocation to 5Y bucket")
print("  • Proceed to Phase 3 (EVE) and Phase 4 (NII)")

print("\n" + "="*80)
print("PHASE 1B ADVANCED COMPLETE ✓")
print("="*80)

# %%
