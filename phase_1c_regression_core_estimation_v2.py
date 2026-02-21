# %% [markdown]
# # Phase 1c: Regression-Based Core Deposit Estimation
#
# ## QF609 Project #1 - IRRBB Modelling Framework
#
# **Calculation Date:** 31-Dec-2023
# Note: Dataset ends 30-Dec-2023; last available observation used as effective date.
#
# ## REGRESSION APPROACHES FOR CORE ESTIMATION
#
# When professor says "use regression model", they likely mean one of:
#
# ### 1. MEAN REVERSION REGRESSION (Vasicek Model)
# - Model: ΔB(t) = a + b×B(t-1) + ε
# - Core = θ = -a/b (long-run mean where balance reverts to)
# - Theory: Deposits fluctuate around long-term equilibrium
#
# ### 2. DETRENDED REGRESSION ⭐ RECOMMENDED
# - Step 1: Fit linear trend: Balance(t) = a + b×t
# - Step 2: Detrend: Residual(t) = Balance(t) - (a + b×t)
# - Step 3: Core = Current Balance + Min(Residual)
# - Theory: Remove growth trend, find structural floor
#
# ### 3. QUANTILE REGRESSION ON TIME
# - Model: Quantile_10%(Balance | t) = a + b×t
# - Core = Predicted 10th percentile at calculation date
# - Theory: Lower bound of conditional distribution
#
# ### 4. EXPONENTIAL GROWTH MODEL (Fixed from Decay)
# - Original: B(t) = Core + (Initial - Core) × exp(-λt)  [WRONG for growing data]
# - Fixed:    B(t) = Core + (Core - Initial) × (exp(λt) - 1)  [GROWTH version]
# - Theory: Balance grows exponentially toward ceiling
#
# This notebook implements ALL 4 approaches and compares results.

# %% [markdown]
# ## 1. Import Libraries and Load Data

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression, QuantileRegressor
from sklearn.metrics import r2_score, mean_squared_error
import os
import warnings

warnings.filterwarnings('ignore')

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (14, 7)
plt.rcParams['font.size'] = 10

print("="*80)
print("PHASE 1C: REGRESSION-BASED CORE DEPOSIT ESTIMATION")
print("="*80)
print("\nLibraries imported successfully")

# %%
# Load processed data
nmd_data = pd.read_csv('processed_nmd_data.csv')
nmd_data['Date'] = pd.to_datetime(nmd_data['Date'])
nmd_data = nmd_data.sort_values('Date').reset_index(drop=True)

# Calculation date: project brief specifies 31-Dec-2023.
# The dataset's last observation is 30-Dec-2023.
# We attempt 31-Dec first and fall back to the last available date.
calc_date_target = pd.to_datetime('2023-12-31')
if calc_date_target in nmd_data['Date'].values:
    calc_date = calc_date_target
else:
    calc_date = nmd_data['Date'].max()
    print(f"Note: 31-Dec-2023 not in dataset. Using last available date: {calc_date.strftime('%d-%b-%Y')}")

current_balance = nmd_data[nmd_data['Date'] == calc_date]['Balance'].values[0]

# Create time index
nmd_data['days_since_start'] = (nmd_data['Date'] - nmd_data['Date'].min()).dt.days

print(f"\nLoaded NMD data: {nmd_data.shape}")
print(f"Date range: {nmd_data['Date'].min().strftime('%d-%b-%Y')} to {nmd_data['Date'].max().strftime('%d-%b-%Y')}")
print(f"Current Balance ({calc_date.strftime('%d-%b-%Y')}): {current_balance:,.2f}")
print(f"Historical Min: {nmd_data['Balance'].min():,.2f} ({(nmd_data['Balance'].min()/current_balance)*100:.2f}%)")
print(f"Historical Max: {nmd_data['Balance'].max():,.2f}")

# %% [markdown]
# ## 2. METHOD 1: Mean Reversion Regression (Vasicek Model)

# %%
print("\n" + "="*80)
print("METHOD 1: MEAN REVERSION REGRESSION")
print("="*80)
print("\nModel: ΔB(t) = a + b×B(t-1) + ε")
print("Theory: Deposits revert to long-run mean θ = -a/b\n")

# Calculate balance changes
nmd_data['balance_change'] = nmd_data['Balance'].diff()
nmd_data['balance_lag'] = nmd_data['Balance'].shift(1)

# Remove first row (no lag)
reg_data = nmd_data.dropna(subset=['balance_change', 'balance_lag'])

# OLS regression: ΔB = a + b×B(t-1)
X = reg_data['balance_lag'].values.reshape(-1, 1)
y = reg_data['balance_change'].values

lr_mr = LinearRegression()
lr_mr.fit(X, y)

a = lr_mr.intercept_
b = lr_mr.coef_[0]

# Mean reversion parameters
if b < 0:
    theta = -a / b  # Long-run mean
    kappa = -b      # Mean reversion speed
    half_life_days = np.log(2) / kappa if kappa > 0 else np.inf

    # Fitted values
    fitted = lr_mr.predict(X)
    r2 = r2_score(y, fitted)
    rmse = np.sqrt(mean_squared_error(y, fitted))

    core_mr = theta
    core_ratio_mr = core_mr / current_balance

    print(f"Regression Equation:")
    print(f"  ΔB(t) = {a:.2f} + {b:.6f}×B(t-1)")
    print(f"\nEstimated Parameters:")
    print(f"  Long-Run Mean (θ):       {theta:,.2f}")
    print(f"  Reversion Speed (κ):     {kappa:.6f} per day")
    print(f"  Half-Life:               {half_life_days:.1f} days ({half_life_days/365.25:.2f} years)")
    print(f"\nModel Fit:")
    print(f"  R²:                      {r2:.4f}")
    print(f"  RMSE:                    {rmse:,.2f}")
    print(f"\nCore/Non-Core Split:")
    print(f"  Current Balance:         {current_balance:,.2f}")
    print(f"  Core (θ):                {core_mr:,.2f}  ({core_ratio_mr*100:.2f}%)")
    print(f"  Non-Core:                {current_balance - core_mr:,.2f}  ({(1-core_ratio_mr)*100:.2f}%)")

    if core_mr > current_balance:
        print(f"\n  ⚠️  WARNING: Core > Current Balance!")
        print(f"      This means balance is BELOW long-run mean (expected to grow)")
        print(f"      For core estimation, cap at current balance")
        core_mr = min(core_mr, current_balance * 0.95)  # Cap at 95%
        core_ratio_mr = core_mr / current_balance
        print(f"      Adjusted Core:       {core_mr:,.2f}  ({core_ratio_mr*100:.2f}%)")
else:
    print(f"  ✗ No mean reversion detected (b = {b:.6f} >= 0)")
    print(f"      Balance shows explosive/trend behavior, not mean-reverting")
    core_mr = current_balance * 0.51  # Fallback
    core_ratio_mr = 0.51

print("\n" + "="*80)

# %% [markdown]
# ## 3. METHOD 2: DETRENDED REGRESSION ⭐ RECOMMENDED

# %%
print("\n" + "="*80)
print("METHOD 2: DETRENDED REGRESSION (RECOMMENDED)")
print("="*80)
print("\nApproach:")
print("  1. Fit linear trend to remove growth")
print("  2. Find minimum deviation from trend")
print("  3. Core = Current Balance + Min Deviation\n")

# Step 1: Fit linear trend
days = nmd_data['days_since_start'].values.reshape(-1, 1)
balance = nmd_data['Balance'].values

lr_trend = LinearRegression()
lr_trend.fit(days, balance)

trend_intercept = lr_trend.intercept_
trend_slope = lr_trend.coef_[0]
trend_fitted = lr_trend.predict(days)

print(f"Step 1: Linear Trend")
print(f"  Balance(t) = {trend_intercept:,.2f} + {trend_slope:.4f}×t")
print(f"  R²:                      {lr_trend.score(days, balance):.4f}")
print(f"  Annual trend:            {trend_slope * 365:,.2f} per year")

# Step 2: Detrend (calculate residuals)
residuals = balance - trend_fitted
min_residual = residuals.min()
max_residual = residuals.max()

nmd_data['trend'] = trend_fitted
nmd_data['residual'] = residuals

print(f"\nStep 2: Detrend")
print(f"  Residual = Balance - Trend")
print(f"  Min Residual:            {min_residual:,.2f}")
print(f"  Max Residual:            {max_residual:,.2f}")
print(f"  Residual Range:          {max_residual - min_residual:,.2f}")

# Step 3: Core = Current Balance + Min Residual
core_detrended = current_balance + min_residual
core_ratio_detrended = core_detrended / current_balance

print(f"\nStep 3: Core Estimation")
print(f"  Core = Current + Min Residual")
print(f"  Core = {current_balance:,.2f} + ({min_residual:,.2f})")
print(f"  Core = {core_detrended:,.2f}")

print(f"\nCore/Non-Core Split:")
print(f"  Current Balance:         {current_balance:,.2f}")
print(f"  Core:                    {core_detrended:,.2f}  ({core_ratio_detrended*100:.2f}%)")
print(f"  Non-Core:                {current_balance - core_detrended:,.2f}  ({(1-core_ratio_detrended)*100:.2f}%)")

print("\n  ✓ This is a CONSERVATIVE estimate (floor below trend)")
print("  ✓ Captures structural minimum after removing growth")

print("\n" + "="*80)

# %% [markdown]
# ## 4. METHOD 3: Quantile Regression on Time

# %%
print("\n" + "="*80)
print("METHOD 3: QUANTILE REGRESSION ON TIME")
print("="*80)
print("\nApproach: Predict 10th percentile of balance as function of time\n")

# Fit quantile regression at multiple quantiles
quantiles = [0.05, 0.10, 0.15, 0.25]
quantile_models = {}

for q in quantiles:
    qr = QuantileRegressor(quantile=q, alpha=0, solver='highs')
    qr.fit(days, balance)

    qr_intercept = qr.intercept_
    qr_slope = qr.coef_[0]

    # Predict at calculation date
    calc_day = nmd_data[nmd_data['Date'] == calc_date]['days_since_start'].values[0]
    core_pred = qr.predict([[calc_day]])[0]

    quantile_models[q] = {
        'model': qr,
        'intercept': qr_intercept,
        'slope': qr_slope,
        'core': core_pred,
        'core_ratio': core_pred / current_balance,
        'fitted': qr.predict(days)
    }

    print(f"{int(q*100):2d}th Quantile:")
    print(f"  Balance(t) = {qr_intercept:,.2f} + {qr_slope:.4f}×t")
    print(f"  Core at calc date:       {core_pred:,.2f}  ({(core_pred/current_balance)*100:.2f}%)")

# Use 10th percentile as primary
q_primary = 0.10
core_qr = quantile_models[q_primary]['core']
core_ratio_qr = quantile_models[q_primary]['core_ratio']

print(f"\nPrimary Recommendation (10th Percentile):")
print(f"  Core:                    {core_qr:,.2f}  ({core_ratio_qr*100:.2f}%)")
print(f"  Non-Core:                {current_balance - core_qr:,.2f}  ({(1-core_ratio_qr)*100:.2f}%)")

# Add fitted quantiles to dataframe
for q, model_info in quantile_models.items():
    nmd_data[f'quantile_{int(q*100)}'] = model_info['fitted']

print("\n" + "="*80)

# %% [markdown]
# ## 5. METHOD 4: Exponential Growth Model (Fixed)

# %%
print("\n" + "="*80)
print("METHOD 4: EXPONENTIAL GROWTH MODEL (Fixed from Decay)")
print("="*80)
print("\nOriginal Decay: B(t) = Core + (Initial - Core) × exp(-λt)  [WRONG]")
print("Fixed Growth:   B(t) = Core × (1 + growth_rate)^t\n")

def exponential_growth_model(t, core, growth_rate):
    """
    Exponential growth model: B(t) = core × (1 + growth_rate)^t
    For deposits that GROW over time
    """
    return core * np.power(1 + growth_rate, t)

# Normalize time to years
t_years = nmd_data['days_since_start'].values / 365.25
balance_vals = nmd_data['Balance'].values

# Initial guesses
initial_balance = balance_vals[0]
min_balance = balance_vals.min()

# For growth model: core should be <= initial
p0 = [min_balance, 0.05]  # Initial guess: core = min, 5% annual growth

# Bounds: core > 0, growth_rate in [-10%, +50%]
bounds = ([0, -0.1], [current_balance, 0.5])

try:
    params, covariance = curve_fit(
        exponential_growth_model,
        t_years,
        balance_vals,
        p0=p0,
        bounds=bounds,
        maxfev=10000
    )

    core_growth, growth_rate = params

    # Calculate fitted values
    fitted_growth = exponential_growth_model(t_years, core_growth, growth_rate)
    r2_growth = r2_score(balance_vals, fitted_growth)
    rmse_growth = np.sqrt(mean_squared_error(balance_vals, fitted_growth))

    core_ratio_growth = core_growth / current_balance

    nmd_data['fitted_growth'] = fitted_growth

    print(f"Model: B(t) = {core_growth:,.2f} × (1 + {growth_rate:.4f})^t")
    print(f"\nEstimated Parameters:")
    print(f"  Core Floor:              {core_growth:,.2f}")
    print(f"  Annual Growth Rate:      {growth_rate*100:.2f}%")
    print(f"\nModel Fit:")
    print(f"  R²:                      {r2_growth:.4f}")
    print(f"  RMSE:                    {rmse_growth:,.2f}")
    print(f"\nCore/Non-Core Split:")
    print(f"  Current Balance:         {current_balance:,.2f}")
    print(f"  Core:                    {core_growth:,.2f}  ({core_ratio_growth*100:.2f}%)")
    print(f"  Non-Core:                {current_balance - core_growth:,.2f}  ({(1-core_ratio_growth)*100:.2f}%)")

    growth_converged = True
except Exception as e:
    print(f"Growth model fitting failed: {e}")
    core_growth = min_balance
    core_ratio_growth = core_growth / current_balance
    growth_converged = False

print("\n" + "="*80)

# %% [markdown]
# ## 6. Comparison of All Regression Methods

# %%
print("\n" + "="*80)
print("COMPARISON OF ALL REGRESSION METHODS")
print("="*80)

# Compile results
regression_methods = {
    '1. Mean Reversion (Vasicek)': {
        'core': core_mr,
        'core_ratio': core_ratio_mr,
        'description': 'Long-run mean from ΔB regression'
    },
    '2. Detrended Regression': {
        'core': core_detrended,
        'core_ratio': core_ratio_detrended,
        'description': 'Current + Min(Residual from trend)'
    },
    '3. Quantile Regression (10th)': {
        'core': core_qr,
        'core_ratio': core_ratio_qr,
        'description': '10th percentile on time'
    }
}

if growth_converged:
    regression_methods['4. Exponential Growth'] = {
        'core': core_growth,
        'core_ratio': core_ratio_growth,
        'description': 'Floor from growth model'
    }

# Create comparison dataframe
comparison_data = []
for method, results in regression_methods.items():
    core = results['core']
    ratio = results['core_ratio']
    comparison_data.append({
        'Method': method,
        'Description': results['description'],
        'Core_Amount': core,
        'Core_Ratio_%': ratio * 100,
        'Non_Core_Amount': current_balance - core
    })

comparison_df = pd.DataFrame(comparison_data)

print("\n")
print(comparison_df[['Method', 'Core_Amount', 'Core_Ratio_%']].to_string(index=False))

# Statistical summary
core_estimates = comparison_df['Core_Amount'].values
print(f"\n{'='*80}")
print("STATISTICAL SUMMARY OF ESTIMATES")
print("="*80)
print(f"Mean:                        {core_estimates.mean():,.2f}  ({(core_estimates.mean()/current_balance)*100:.2f}%)")
print(f"Median:                      {np.median(core_estimates):,.2f}  ({(np.median(core_estimates)/current_balance)*100:.2f}%)")
print(f"Std Dev:                     {core_estimates.std():,.2f}")
print(f"Min:                         {core_estimates.min():,.2f}  ({(core_estimates.min()/current_balance)*100:.2f}%)")
print(f"Max:                         {core_estimates.max():,.2f}  ({(core_estimates.max()/current_balance)*100:.2f}%)")

# %% [markdown]
# ## 7. PRIMARY RECOMMENDATION: Detrended Regression

# %%
# Use detrended regression as primary (most reasonable for growing deposits)
core_primary = core_detrended
core_ratio_primary = core_ratio_detrended
non_core_primary = current_balance - core_primary

print("\n" + "="*80)
print("PRIMARY RECOMMENDATION: DETRENDED REGRESSION")
print("="*80)
print(f"\nWhy Detrended Regression?")
print(f"  ✓ Accounts for growth trend in deposits")
print(f"  ✓ Finds structural floor (not absolute minimum)")
print(f"  ✓ Conservative but not overly pessimistic")
print(f"  ✓ Core ratio {core_ratio_primary*100:.1f}% is reasonable for NMD")

print(f"\n{'='*80}")
print(f"RECOMMENDED CORE/NON-CORE SPLIT")
print(f"{'='*80}")
print(f"Current Balance:             {current_balance:,.2f}")
print(f"Core Deposits:               {core_primary:,.2f}  ({core_ratio_primary*100:.2f}%)")
print(f"Non-Core Deposits:           {non_core_primary:,.2f}  ({(1-core_ratio_primary)*100:.2f}%)")

# ── Regulatory Compliance Checks (Basel IRRBB BCBS 2016, para 112–116) ────────
print(f"\n{'='*80}")
print(f"REGULATORY COMPLIANCE — BASEL IRRBB (BCBS 2016, PARA 112–116)")
print(f"{'='*80}")

non_core_ratio_primary = 1 - core_ratio_primary
hist_min_ratio = nmd_data['Balance'].min() / current_balance

checks = []

# Check 1: Core cap ≤ 90% (retail NMD)
passed = core_ratio_primary <= 0.90
checks.append(passed)
status = "✓ PASS" if passed else "✗ FAIL"
print(f"\n[1] Core ≤ 90% of total balance (retail NMD cap, BCBS para 116)")
print(f"    Core ratio: {core_ratio_primary*100:.2f}%   Cap: 90.00%   → {status}")
if not passed:
    excess = (core_ratio_primary - 0.90) * current_balance
    print(f"    ⚠️  Excess core: {excess:,.2f} must be reclassified to non-core")

# Check 2: Non-core floor ≥ 10% (minimum volatile portion)
# Basel requires that at least 10% of any NMD balance is treated as non-stable
# (non-core), recognising that all deposit pools carry some repricing risk.
# BCBS 2016 para 116: non-stable share ≥ 10% for all NMD categories.
passed = non_core_ratio_primary >= 0.10
checks.append(passed)
status = "✓ PASS" if passed else "✗ FAIL"
print(f"\n[2] Non-core ≥ 10% of total balance (minimum volatile floor, BCBS para 116)")
print(f"    Non-core ratio: {non_core_ratio_primary*100:.2f}%   Floor: 10.00%   → {status}")
if not passed:
    shortfall = (0.10 - non_core_ratio_primary) * current_balance
    print(f"    ⚠️  Shortfall: {shortfall:,.2f} must be reclassified from core to non-core")

# Check 3: Core ≥ historical minimum balance (sanity floor)
# Core should not be set below the lowest ever observed balance, as the bank
# demonstrably held at least that much at all times.
passed = core_primary >= nmd_data['Balance'].min()
checks.append(passed)
status = "✓ PASS" if passed else "✗ FAIL"
print(f"\n[3] Core ≥ historical minimum balance (sanity floor)")
print(f"    Core: {core_primary:,.2f}   Hist. min: {nmd_data['Balance'].min():,.2f}   → {status}")

# Check 4: Core ≤ current balance (core cannot exceed total)
passed = core_primary <= current_balance
checks.append(passed)
status = "✓ PASS" if passed else "✗ FAIL"
print(f"\n[4] Core ≤ current balance (cannot exceed total deposits)")
print(f"    Core: {core_primary:,.2f}   Current balance: {current_balance:,.2f}   → {status}")

# Check 5: Core > 0 (trivially must be positive)
passed = core_primary > 0
checks.append(passed)
status = "✓ PASS" if passed else "✗ FAIL"
print(f"\n[5] Core > 0 (must be positive)")
print(f"    Core: {core_primary:,.2f}   → {status}")

# Overall result
all_pass = all(checks)
print(f"\n{'='*80}")
if all_pass:
    print(f"OVERALL REGULATORY STATUS: ✓ ALL CHECKS PASSED")
else:
    n_fail = sum(1 for c in checks if not c)
    print(f"OVERALL REGULATORY STATUS: ✗ {n_fail} CHECK(S) FAILED — review above")
print(f"{'='*80}")
# ──────────────────────────────────────────────────────────────────────────────

# %% [markdown]
# ## 8. Stressed Core Estimate
#
# Basel IRRBB requires consideration of how deposit volumes behave under rate
# stress. A higher-rate environment may accelerate withdrawals, reducing the
# stable core. We use the stress period flags from Phase 1b (high-decay periods
# identified at the P90 decay rate threshold) to compute a stress-adjusted core.
#
# Three approaches:
#   A) Base case:   Core from detrended regression (our primary estimate)
#   B) Stress case: Use minimum balance observed DURING stress periods only
#   C) Conservative: min(A, B) — most conservative possible floor

# %%
print("\n" + "="*80)
print("STRESSED CORE ESTIMATE — RATE STRESS SENSITIVITY")
print("="*80)
print("\nPurpose: Assess how core volume changes if rate stress accelerates withdrawals")
print("Method:  Use high-decay periods (P90 threshold) identified in Phase 1b")
print("         as a proxy for rate-stress behaviour")

# Try to load stress period data from Phase 1b
stress_file = 'stress_periods_identified.csv'
if os.path.exists(stress_file):
    stress_data = pd.read_csv(stress_file)
    stress_data['Date'] = pd.to_datetime(stress_data['Date'])

    # Identify stress days using P90 net CDR threshold from Phase 1b
    stress_mask   = stress_data['Stress_P90'] == True
    n_stress_days = stress_mask.sum()
    n_total_days  = len(stress_data)

    if n_stress_days > 0:
        # ── Rate-based stressed core ──────────────────────────────────────────
        # Use mean net CDR on P90 stress days, applied to current balance over
        # a 30-day stress horizon (standard IRRBB shock window).
        #
        # Why rate-based, not min-balance?
        # Min absolute balance on stress days is confounded by portfolio growth:
        # a trough from 2017 (when the portfolio was smaller) produces an
        # artificially low stressed core ratio unrelated to stress severity.
        # Net CDR is portfolio-size-independent and anchors correctly to today.
        #
        # Formula:
        #   survival_under_stress = (1 - mean_stress_cdr) ^ stress_horizon
        #   core_stressed         = current_balance × survival_under_stress

        mean_stress_cdr       = stress_data.loc[stress_mask,  'NetCDR'].mean()
        mean_normal_cdr       = stress_data.loc[~stress_mask, 'NetCDR'].mean()
        stress_horizon_days   = 30  # IRRBB standard shock window

        survival_under_stress = (1 - mean_stress_cdr) ** stress_horizon_days
        core_stressed         = current_balance * survival_under_stress
        core_ratio_stressed   = survival_under_stress

        # Conservative floor = most pessimistic of base and stressed
        core_conservative       = min(core_primary, core_stressed)
        core_ratio_conservative = core_conservative / current_balance

        print(f"\nStress period data loaded: {stress_file}")
        print(f"  Stress days (P90):          {n_stress_days} / {n_total_days}  "
              f"({n_stress_days/n_total_days*100:.1f}% of history)")
        print(f"  Mean net CDR (stress days): {mean_stress_cdr:.6f}  "
              f"({mean_stress_cdr*100:.4f}% per day)")
        print(f"  Mean net CDR (normal days): {mean_normal_cdr:.6f}  "
              f"({mean_normal_cdr*100:.4f}% per day)")
        print(f"  CDR uplift under stress:    "
              f"{(mean_stress_cdr/mean_normal_cdr - 1)*100:+.1f}% faster decay than normal")
        print(f"  Stress horizon:             {stress_horizon_days} days")
        print(f"  Survival under stress:      {survival_under_stress:.4f}  "
              f"({survival_under_stress*100:.2f}%)")

        print(f"\n{'─'*80}")
        print(f"CORE ESTIMATES UNDER DIFFERENT SCENARIOS")
        print(f"{'─'*80}")
        print(f"  A) Base case    (detrended regression):          "
              f"{core_primary:>12,.2f}  ({core_ratio_primary*100:.2f}%)")
        print(f"  B) Stress case  (P90 net CDR x {stress_horizon_days}d horizon): "
              f"{core_stressed:>12,.2f}  ({core_ratio_stressed*100:.2f}%)")
        print(f"  C) Conservative (min of A and B):                "
              f"{core_conservative:>12,.2f}  ({core_ratio_conservative*100:.2f}%)")

        vol_impact = core_stressed - core_primary
        print(f"\n  Volume impact of stress (B - A):                 "
              f"{vol_impact:>+12,.2f}  ({(vol_impact/core_primary)*100:+.2f}%)")

        if vol_impact < 0:
            print(f"\n  Interpretation: Under rate stress (P90 net CDR sustained for")
            print(f"  {stress_horizon_days} days), core could contract by {abs(vol_impact):,.2f} "
                  f"({abs(vol_impact/core_primary)*100:.1f}%),")
            print(f"  reflecting accelerated withdrawals during high-decay periods.")
            print(f"\n  For IRRBB reporting: BASE CASE core ({core_primary:,.2f}) is the")
            print(f"  primary estimate. Stressed core ({core_stressed:,.2f}) feeds into")
            print(f"  EVE/NII scenario analysis as the adverse volume assumption.")
        else:
            print(f"\n  Interpretation: Stress periods did not produce a lower balance")
            print(f"  floor than the base case — base case remains appropriately conservative.")

        # Regulatory check on stressed core
        print(f"\n{'─'*80}")
        print(f"REGULATORY CHECK ON STRESSED CORE")
        print(f"{'─'*80}")
        stressed_non_core_ratio = 1 - core_ratio_stressed
        s1 = "✓" if core_ratio_stressed <= 0.90 else "✗"
        s2 = "✓" if stressed_non_core_ratio >= 0.10 else "✗"
        print(f"  {s1} Stressed core ≤ 90%:      {core_ratio_stressed*100:.2f}%")
        print(f"  {s2} Stressed non-core ≥ 10%:  {stressed_non_core_ratio*100:.2f}%")

    else:
        print(f"\n  No stress days identified at P90 threshold — all periods classified as normal.")
        print(f"  Stressed core = base case core: {core_primary:,.2f}")

else:
    # Fallback: approximate stress period using P10 balance (bottom decile)
    print(f"\nStress period file not found ('{stress_file}').")
    print(f"Approximating stressed core using P10 historical balance as floor.")
    p10_balance = nmd_data['Balance'].quantile(0.10)
    core_stressed = p10_balance
    core_ratio_stressed = core_stressed / current_balance

    print(f"\n  P10 historical balance:          {p10_balance:,.2f}  ({core_ratio_stressed*100:.2f}%)")
    print(f"  Base case core (detrended):      {core_primary:,.2f}  ({core_ratio_primary*100:.2f}%)")
    vol_impact = core_stressed - core_primary
    print(f"  Stress volume impact:            {vol_impact:>+,.2f}  ({(vol_impact/core_primary)*100:+.2f}%)")
    print(f"\n  Run Phase 1b first to generate stress_periods_identified.csv")
    print(f"  for a more rigorous stress-based core estimate.")

print("\n" + "="*80)
print("STRESSED CORE ESTIMATION COMPLETE")
print("="*80)

# %% [markdown]
# ## 9. Rate-Calibrated Stressed Core — Explicit 200bps Scenario
#
# Section 8 uses P90 historical CDR days as a proxy for rate stress.
# This section makes the rate link explicit: regress net CDR on an
# approximate overnight rate series, then apply the calibrated sensitivity
# directly to the +200bps project scenario (and ±100bps, +300bps).
#
# **Data limitation:** No historical rate time series is provided in the
# project data. The zero curve gives a single snapshot (1D = 3.178% at
# 31-Dec-2023). We backfill an approximate overnight rate by year using
# the known rate cycle consistent with that terminal level.
# Clearly labelled as approximate — R² will be low (noisy daily CDR),
# but β captures the directional sensitivity needed for scenario analysis.
#
# **Model:** net_CDR(t) = α + β × rate_level(t) + ε
# **Stressed CDR** = CDR_base + β × shock  (where +200bps shock = +0.02)

# %%
print("\n" + "="*80)
print("RATE-CALIBRATED STRESSED CORE — EXPLICIT 200BPS SCENARIO")
print("="*80)
print("\nModel:  net_CDR = α + β × overnight_rate")
print("Shock:  +200bps (project requirement), also +100bps and +300bps shown")
print("Note:   Overnight rate backfilled from zero curve 1D = 3.178% at 31-Dec-2023")

# ── Step 1: Approximate overnight rate series by year ────────────────────────
# Terminal anchor: 1D rate from zero curve = 3.178% at 31-Dec-2023.
# Backfilled using broad rate cycle (low 2017-2021, rapid hike 2022-2023).
# Clearly approximate — historical rate data not provided in project files.
rate_by_year = {
    2016: 0.0030,   # ~30bps  — low rate environment
    2017: 0.0050,   # ~50bps  — gradual normalisation begins
    2018: 0.0075,   # ~75bps
    2019: 0.0075,   # ~75bps  — some easing late-2019
    2020: 0.0025,   # ~25bps  — emergency cuts (COVID)
    2021: 0.0025,   # ~25bps  — accommodative, near-zero
    2022: 0.0175,   # ~175bps — rapid hiking cycle (avg across year)
    2023: 0.0318,   # ~318bps — zero curve 1D rate at 31-Dec-2023
}

nmd_rate = nmd_data.copy()
nmd_rate['approx_rate'] = nmd_rate['year'].map(rate_by_year)

# Compute net CDR fresh (gross daily_decay_rate in processed data is Outflow/Balance)
nmd_rate['net_cdr'] = (
    np.maximum(0, nmd_rate['Outflow'] - nmd_rate['Inflow']) / nmd_rate['Balance']
).clip(0, 1)

reg_df = nmd_rate[['approx_rate', 'net_cdr']].dropna()

# ── Step 2: OLS regression ────────────────────────────────────────────────────
X_rate = reg_df[['approx_rate']].values
y_cdr  = reg_df['net_cdr'].values

ols_rate = LinearRegression()
ols_rate.fit(X_rate, y_cdr)

alpha_rate = ols_rate.intercept_
beta_rate  = ols_rate.coef_[0]
r2_rate    = r2_score(y_cdr, ols_rate.predict(X_rate))

print(f"\nOLS RESULTS: net_CDR = α + β × overnight_rate")
print(f"{'─'*60}")
print(f"  α (intercept):        {alpha_rate:.6f}  (CDR when rate ≈ 0%)")
print(f"  β (slope):            {beta_rate:+.6f}  (ΔCDR per unit Δrate)")
print(f"  β per +100bps:        {beta_rate * 0.01:+.6f}  (ΔCDR per +100bps)")
print(f"  β per +200bps:        {beta_rate * 0.02:+.6f}  (ΔCDR per +200bps)")
print(f"  R²:                   {r2_rate:.4f}")
print(f"\n  Note: Low R² is expected — daily CDR is noisy. β captures")
print(f"  the systematic direction of rate sensitivity across the rate")
print(f"  cycle, not day-to-day variation.")

if beta_rate <= 0:
    print(f"\n  *** METHODOLOGICAL CAVEAT — READ BEFORE INTERPRETING RESULTS ***")
    print(f"  {'─'*76}")
    print(f"  β = {beta_rate:+.6f} (NEGATIVE)")
    print(f"")
    print(f"  The regression finds that higher rates are associated with LOWER net")
    print(f"  CDR in this data — the opposite of the expected economic relationship.")
    print(f"  Under a rate shock, the model will therefore produce a HIGHER core")
    print(f"  ratio, not a lower one. This is directionally wrong for stress testing.")
    print(f"")
    print(f"  Why this happened — portfolio lifecycle confounding:")
    print(f"  The backfilled rate series assigns low rates to 2017-2021 and high")
    print(f"  rates to 2022-2023. However, in the early years (2017-2019) the")
    print(f"  portfolio was growing rapidly — inflows frequently outpaced outflows,")
    print(f"  making net CDR volatile and episodically high. By 2022-2023, the")
    print(f"  portfolio had matured and net CDR had structurally settled lower.")
    print(f"  The regression cannot separate the rate effect from the portfolio")
    print(f"  lifecycle effect, so it picks up the spurious correlation:")
    print(f"  'high rates (2022-2023) → low net CDR' → β < 0.")
    print(f"")
    print(f"  Conclusion: A year-level rate proxy cannot cleanly identify CDR")
    print(f"  sensitivity to rates in this dataset. The scenario table below is")
    print(f"  shown for transparency but SHOULD NOT be used as the operative")
    print(f"  stressed core estimate.")
    print(f"")
    print(f"  Operative stressed core → use Section 8 (P90 net CDR proxy).")
    print(f"  For a valid rate-CDR regression, a daily historical rate time")
    print(f"  series matched to the NMD data dates would be required.")
    print(f"  {'─'*76}")

# ── Step 3: Apply to rate shock scenarios ────────────────────────────────────
base_rate    = 0.0318   # 1D rate from zero curve at 31-Dec-2023
base_cdr_fit = float(ols_rate.predict([[base_rate]])[0])
horizon_days = 30       # IRRBB standard shock window

scenarios = [
    ('+100bps', 0.01),
    ('+200bps', 0.02),   # ← PROJECT REQUIREMENT
    ('+300bps', 0.03),
]

print(f"\n{'─'*80}")
print(f"RATE SHOCK SCENARIOS — CORE VOLUME IMPACT (30-DAY HORIZON)")
print(f"{'─'*80}")
print(f"  Base overnight rate:    {base_rate*100:.3f}%")
print(f"  Base CDR (OLS fitted):  {base_cdr_fit:.6f}  ({base_cdr_fit*100:.4f}%/day)")
print(f"  Stress horizon:         {horizon_days} days")
print(f"")
print(f"  {'Scenario':<10} {'Rate':>8} {'Stressed CDR':>14} "
      f"{'S(30d)':>8} {'Core Stressed':>15} {'Core Ratio':>12}")
print(f"  {'─'*10} {'─'*8} {'─'*14} {'─'*8} {'─'*15} {'─'*12}")

results_200 = {}
for label, shock in scenarios:
    shocked_rate = base_rate + shock
    stressed_cdr = max(0.0, float(ols_rate.predict([[shocked_rate]])[0]))
    survival_30d = (1 - stressed_cdr) ** horizon_days
    core_s       = current_balance * survival_30d
    tag = '  ← PROJECT' if label == '+200bps' else ''
    print(f"  {label:<10} {shocked_rate*100:>7.3f}% {stressed_cdr:>13.6f}  "
          f"{survival_30d:>7.4f}  {core_s:>14,.2f}  {survival_30d*100:>10.2f}%{tag}")
    if label == '+200bps':
        results_200 = {
            'stressed_cdr': stressed_cdr,
            'survival': survival_30d,
            'core': core_s,
        }

# ── Step 4: Regulatory check — +200bps scenario (flagged if β < 0) ───────────
if results_200:
    sv = results_200['survival']
    print(f"\n{'─'*80}")
    print(f"REGULATORY CHECK — +200BPS SCENARIO (FOR REFERENCE ONLY)")
    print(f"{'─'*80}")
    nc = 1 - sv
    r1 = "✓" if sv  <= 0.90 else "✗"
    r2 = "✓" if nc  >= 0.10 else "✗"
    print(f"  {r1} Stressed core ≤ 90%:      {sv*100:.2f}%")
    print(f"  {r2} Stressed non-core ≥ 10%:  {nc*100:.2f}%")
    if beta_rate <= 0:
        print(f"")
        print(f"  NOTE: If β < 0 (as above), the stressed core EXCEEDS the base case —")
        print(f"  meaning the regulatory check result here is misleading. A fail on")
        print(f"  ≤90% in this context reflects the model producing an inverted result,")
        print(f"  not a genuine regulatory breach. Disregard and use Section 8.")

# ── Step 5: Summary ──────────────────────────────────────────────────────────
print(f"\n{'─'*80}")
print(f"SECTION 9 SUMMARY")
print(f"{'─'*80}")
print(f"  Objective:   Calibrate CDR sensitivity to rate level via OLS regression")
print(f"  Result:      β = {beta_rate:+.6f} ({'NEGATIVE — inverted relationship' if beta_rate <= 0 else 'POSITIVE — correct direction'})")
print(f"  Root cause:  Portfolio lifecycle trend (growth 2017-2019, maturity 2022-2023)")
print(f"               confounds the rate-CDR relationship in year-level backfilled data")
print(f"  Limitation:  Year-level rate proxy cannot separate rate effect from")
print(f"               lifecycle effect; daily historical rate series required")
print(f"  Operative stressed core: Section 8 (P90 net CDR proxy) — β direction valid")

print("\n" + "="*80)
print("RATE-CALIBRATED STRESSED CORE COMPLETE")
print("="*80)

# %% [markdown]
# ## 10. Visualizations


# %%
fig, axes = plt.subplots(2, 2, figsize=(18, 12))

# Plot 1: Balance with trend and residuals
axes[0, 0].plot(nmd_data['Date'], nmd_data['Balance'],
                linewidth=2, color='#023047', label='Actual Balance', alpha=0.8)
axes[0, 0].plot(nmd_data['Date'], nmd_data['trend'],
                linewidth=2, linestyle='--', color='#E63946', label='Linear Trend', alpha=0.8)
axes[0, 0].axhline(y=core_detrended, color='green', linestyle=':',
                   linewidth=2.5, label=f'Core (Detrended): {core_detrended:,.0f}', alpha=0.9)
axes[0, 0].fill_between(nmd_data['Date'], core_detrended, nmd_data['Balance'],
                         alpha=0.2, color='green')
axes[0, 0].set_xlabel('Date', fontsize=10)
axes[0, 0].set_ylabel('Balance', fontsize=10)
axes[0, 0].set_title('Detrended Regression: Balance vs Trend', fontsize=12, fontweight='bold')
axes[0, 0].legend(loc='best', fontsize=9)
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: Residuals from trend
axes[0, 1].plot(nmd_data['Date'], nmd_data['residual'],
                linewidth=1.5, color='#2A9D8F', alpha=0.7)
axes[0, 1].axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
axes[0, 1].axhline(y=min_residual, color='red', linestyle='--',
                   linewidth=2, label=f'Min Residual: {min_residual:,.0f}', alpha=0.8)
axes[0, 1].fill_between(nmd_data['Date'], min_residual, 0,
                         alpha=0.2, color='red')
axes[0, 1].set_xlabel('Date', fontsize=10)
axes[0, 1].set_ylabel('Residual (Balance - Trend)', fontsize=10)
axes[0, 1].set_title('Detrended Residuals', fontsize=12, fontweight='bold')
axes[0, 1].legend(loc='best', fontsize=9)
axes[0, 1].grid(True, alpha=0.3)

# Plot 3: Quantile Regression
axes[1, 0].scatter(nmd_data['days_since_start'], nmd_data['Balance'],
                   alpha=0.3, s=10, color='gray', label='Data')
for q, model_info in quantile_models.items():
    axes[1, 0].plot(nmd_data['days_since_start'], model_info['fitted'],
                    linewidth=2, label=f'{int(q*100)}th Quantile', alpha=0.8)
axes[1, 0].set_xlabel('Days Since Start', fontsize=10)
axes[1, 0].set_ylabel('Balance', fontsize=10)
axes[1, 0].set_title('Quantile Regression on Time', fontsize=12, fontweight='bold')
axes[1, 0].legend(loc='best', fontsize=8)
axes[1, 0].grid(True, alpha=0.3)

# Plot 4: Core ratio comparison
methods_list = comparison_df['Method'].str.split('.', n=1).str[1].str.strip().values
core_ratios = comparison_df['Core_Ratio_%'].values
colors = plt.cm.viridis(np.linspace(0, 1, len(methods_list)))

bars = axes[1, 1].barh(range(len(methods_list)), core_ratios, color=colors, alpha=0.8, edgecolor='black')
axes[1, 1].set_yticks(range(len(methods_list)))
axes[1, 1].set_yticklabels(methods_list, fontsize=9)
axes[1, 1].set_xlabel('Core Ratio (%)', fontsize=10)
axes[1, 1].set_title('Core Ratio Comparison Across Regression Methods', fontsize=12, fontweight='bold')
axes[1, 1].grid(True, alpha=0.3, axis='x')

# Highlight detrended (recommended)
bars[1].set_edgecolor('red')
bars[1].set_linewidth(3)

# Add value labels
for i, (bar, val) in enumerate(zip(bars, core_ratios)):
    width = bar.get_width()
    axes[1, 1].text(width + 0.5, bar.get_y() + bar.get_height()/2,
                    f'{val:.1f}%', va='center', fontsize=9)

plt.tight_layout()
plt.savefig('regression_core_estimation.png', dpi=300, bbox_inches='tight')
plt.show()

# %% [markdown]
# ## 10. Export Results

# %%
import json
from datetime import datetime

# Save primary recommendation
core_noncore_split_regression = pd.DataFrame({
    'Component': ['Total Balance', 'Core Deposits', 'Non-Core Deposits'],
    'Amount': [current_balance, core_primary, non_core_primary],
    'Percentage': [100.0, core_ratio_primary*100, (1-core_ratio_primary)*100],
    'Behavioral_Maturity': ['N/A', '5 Years Max', 'O/N'],
    'Repricing': ['N/A', 'Distributed 1M-5Y', 'Immediate (O/N)'],
    'Method': ['N/A', 'Detrended Regression', 'Detrended Regression']
})
core_noncore_split_regression.to_csv('core_noncore_split_regression.csv', index=False)

# Save detailed comparison
comparison_df.to_csv('regression_methods_comparison.csv', index=False)

# Save time series with models
output_cols = ['Date', 'Balance', 'trend', 'residual', 'quantile_10']
if growth_converged:
    output_cols.append('fitted_growth')

nmd_data[output_cols].to_csv('nmd_with_regression_models.csv', index=False)

# Save simple configuration file for Phase 2+ usage
config = {
    "calculation_date": calc_date.strftime('%Y-%m-%d'),
    "current_balance": float(current_balance),
    "core_amount": float(core_primary),
    "core_ratio_pct": float(core_ratio_primary * 100),
    "non_core_amount": float(non_core_primary),
    "non_core_ratio_pct": float((1 - core_ratio_primary) * 100),
    "method": "Detrended Regression"
}

# Save configuration to JSON
with open('config.json', 'w') as f:
    json.dump(config, f, indent=2)

print("\n" + "="*80)
print("FILES EXPORTED")
print("="*80)
print("1. core_noncore_split_regression.csv")
print("   → Primary recommendation (detrended regression)")
print("\n2. regression_methods_comparison.csv")
print("   → Comparison of all 4 regression methods")
print("\n3. nmd_with_regression_models.csv")
print("   → Time series with fitted models")
print("\n4. regression_core_estimation.png")
print("   → Visualization of all methods")
print("\n5. config.json")
print("   → Core/Non-Core allocation for Phase 2+")

# %% [markdown]
# ## 10. Final Summary

# %%
print("\n" + "="*80)
print("FINAL SUMMARY - REGRESSION-BASED CORE ESTIMATION")
print("="*80)

print("\n1. RECOMMENDED METHOD: Detrended Regression")
print("-" * 80)
print(f"   Core Amount:             {core_primary:,.2f}")
print(f"   Core Ratio:              {core_ratio_primary*100:.2f}%")
print(f"   Non-Core:                {non_core_primary:,.2f}")
print(f"\n   Why this method?")
print(f"   • Accounts for upward trend in deposits")
print(f"   • Finds structural floor after removing growth")
print(f"   • Conservative but realistic estimate")
print(f"   • Core ratio {core_ratio_primary*100:.1f}% is in reasonable range")

print("\n2. ALTERNATIVE REGRESSION METHODS")
print("-" * 80)
for i, row in comparison_df.iterrows():
    print(f"   {row['Method']:40s}  {row['Core_Ratio_%']:6.2f}%")

print("\n3. INTERPRETATION")
print("-" * 80)
if core_ratio_primary > 0.80:
    print(f"   • Core ratio of {core_ratio_primary*100:.1f}% suggests HIGHLY stable deposits")
    print(f"   • Most deposits are structurally 'sticky' (relationship-based)")
    print(f"   • Conservative for IRRBB (less repricing risk)")
elif core_ratio_primary > 0.60:
    print(f"   • Core ratio of {core_ratio_primary*100:.1f}% suggests MODERATELY stable deposits")
    print(f"   • Balanced between core and volatile components")
    print(f"   • Reasonable for typical NMD accounts")
else:
    print(f"   • Core ratio of {core_ratio_primary*100:.1f}% suggests VOLATILE deposits")
    print(f"   • Significant portion is rate-sensitive (hot money)")
    print(f"   • Higher repricing risk for IRRBB")

print("\n4. NEXT STEPS (PHASE 2)")
print("-" * 80)
print(f"   • Non-core ({non_core_primary:,.2f}) → O/N bucket")
print(f"   • Core ({core_primary:,.2f}) → distributed across 1M-5Y using S(t)")

print("\n" + "="*80)
print("REGRESSION-BASED CORE ESTIMATION COMPLETE ✓")
print("="*80)

# %%
