# %% [markdown]
# # Phase 5d (Enhanced): Monte Carlo Simulation with Stochastic Rates
#
# ## QF609 Project #1 - IRRBB Modelling Framework
#
# **Calculation Date:** 30-Dec-2023
#
# **Enhancements over original phase_5d:**
# 1. Stochastic interest rate paths (Vasicek model) alongside stochastic balance paths
# 2. Dynamic core/non-core separation using reference methodology (max of 3 methods x 0.9 haircut)
# 3. 9 Basel buckets matching reference (O/N, 1M, 3M, 6M, 1Y, 2Y, 3Y, 4Y, 5Y)
# 4. Comparison of fixed vs dynamic core ratio approaches

# %%
from phase_5_helpers import *
from numpy.polynomial import polynomial as P

print("=" * 80)
print("SECTION 5d (ENHANCED): MONTE CARLO WITH STOCHASTIC RATES")
print("=" * 80)

# %% [markdown]
# ## 1. Simulation Parameters

# %%
# Balance path parameters (stochastic decay)
sigma_decay = nmd_data['daily_decay_rate'].std()
n_paths = 1000
n_days_sim = 365  # 1-year horizon for EVE/NII evaluation
seed = 42

# Vasicek interest rate model parameters
# dr = kappa * (theta - r) * dt + sigma_r * dW
r0 = base_rates[0]            # initial short rate (O/N rate from curve data)
kappa = 0.5                   # mean-reversion speed
theta = base_rates[0]         # long-run mean (set to current O/N rate)
sigma_r = 0.01                # rate volatility (100bps annual vol)
dt = 1 / 365                  # daily time step

np.random.seed(seed)

print(f"\nSimulation Parameters:")
print(f"  Number of paths:              {n_paths}")
print(f"  Horizon:                      {n_days_sim} days (1 year)")
print(f"  Starting balance:             {current_balance:,.2f}")
print(f"  Random seed:                  {seed}")
print(f"\nBalance Path (Stochastic Decay):")
print(f"  Lambda (mean daily decay):    {lambda_daily:.6f}")
print(f"  Sigma (std of decay rate):    {sigma_decay:.6f}")
print(f"\nInterest Rate Path (Vasicek):")
print(f"  r0 (initial short rate):      {r0:.6f}")
print(f"  Kappa (mean-reversion speed): {kappa:.2f}")
print(f"  Theta (long-run mean rate):   {theta:.6f}")
print(f"  Sigma_r (rate volatility):    {sigma_r:.4f}")

# %% [markdown]
# ## 2. Reference Core/Non-Core Methodology
#
# Uses max(5th percentile, trend-based core, historical minimum) x 0.9 haircut.
# Applied dynamically to each simulated balance path.

# %%
def compute_core_noncore_reference(balance_history, current_bal):
    """
    Compute core/non-core split using reference methodology.

    Core = max(5th_percentile, trend_core, historical_min) * 0.9

    Parameters
    ----------
    balance_history : array-like
        Array of historical daily balances (simulated path up to evaluation date).
    current_bal : float
        Balance at evaluation date.

    Returns
    -------
    core_amount, non_core_amount, core_ratio
    """
    bal = np.array(balance_history)

    # Method 1: Historical minimum
    hist_min = bal.min()

    # Method 2: 5th percentile
    q05 = np.percentile(bal, 5)

    # Method 3: Trend-based core (linear polynomial fit)
    t_days = np.arange(len(bal))
    if len(bal) > 2:
        coeffs = P.polyfit(t_days, bal, 1)
        trend_vals = P.polyval(t_days, coeffs)
        trend_at_end = P.polyval(t_days[-1], coeffs)
        min_residual = (bal - trend_vals).min()
        core_trend = trend_at_end + min_residual
    else:
        core_trend = hist_min

    # Final core: max of all methods with 10% conservative haircut
    core_amount = max(q05, core_trend, hist_min) * 0.9
    core_amount = max(core_amount, 0)                # floor at 0
    core_amount = min(core_amount, current_bal)       # cap at current balance

    non_core_amount = current_bal - core_amount
    core_ratio = core_amount / current_bal if current_bal > 0 else 0.0

    return core_amount, non_core_amount, core_ratio


# Verify with actual historical data
hist_balance = nmd_data['Balance'].values
core_ref, noncore_ref, ratio_ref = compute_core_noncore_reference(
    hist_balance, current_balance
)

print(f"\nReference Core/Non-Core Method (on actual data):")
print(f"  Core Deposits:     {core_ref:,.2f}  ({ratio_ref*100:.2f}%)")
print(f"  Non-Core Deposits: {noncore_ref:,.2f}  ({(1-ratio_ref)*100:.2f}%)")
print(f"  (vs. current fixed core ratio: {core_ratio_primary*100:.2f}%)")

# %% [markdown]
# ## 3. Basel 9-Bucket Slotting (Reference Style)

# %%
def slot_cashflows_9bucket(balance, core_ratio_val, lambda_d):
    """
    Slot NMD balance into 9 Basel time buckets (reference style).

    Non-core -> O/N bucket.
    Core -> distributed using survival function S(t) = (1 - lambda_d)^t_days.
    Balance surviving beyond 5Y (1825 days) is capped at 5Y bucket.

    Parameters
    ----------
    balance : float
        Total NMD balance.
    core_ratio_val : float
        Core ratio (0 to 1).
    lambda_d : float
        Daily decay rate for survival function.

    Returns
    -------
    DataFrame with columns: Bucket, Midpoint_Years, Core_CF, Non_Core_CF, Total_CF
    """
    core_amt = balance * core_ratio_val
    non_core_amt = balance * (1 - core_ratio_val)

    # 9 Basel buckets with boundaries in DAYS (using daily survival model)
    buckets = [
        {'Bucket': 'O/N',  'start_d': 0,    'end_d': 1,    'Midpoint_Years': 1/365},
        {'Bucket': '1M',   'start_d': 1,    'end_d': 30,   'Midpoint_Years': 1/24},
        {'Bucket': '3M',   'start_d': 30,   'end_d': 90,   'Midpoint_Years': 2/12},
        {'Bucket': '6M',   'start_d': 90,   'end_d': 180,  'Midpoint_Years': 4.5/12},
        {'Bucket': '1Y',   'start_d': 180,  'end_d': 365,  'Midpoint_Years': 9/12},
        {'Bucket': '2Y',   'start_d': 365,  'end_d': 730,  'Midpoint_Years': 1.5},
        {'Bucket': '3Y',   'start_d': 730,  'end_d': 1095, 'Midpoint_Years': 2.5},
        {'Bucket': '4Y',   'start_d': 1095, 'end_d': 1460, 'Midpoint_Years': 3.5},
        {'Bucket': '5Y',   'start_d': 1460, 'end_d': 1825, 'Midpoint_Years': 4.5},
    ]

    rows = []

    for b in buckets:
        s_start = (1 - lambda_d) ** b['start_d']
        s_end = (1 - lambda_d) ** b['end_d']
        core_cf = core_amt * (s_start - s_end)

        if b['Bucket'] == 'O/N':
            rows.append({
                'Bucket': 'O/N',
                'Midpoint_Years': b['Midpoint_Years'],
                'Core_CF': core_cf,
                'Non_Core_CF': non_core_amt,
                'Total_CF': non_core_amt + core_cf
            })
        else:
            rows.append({
                'Bucket': b['Bucket'],
                'Midpoint_Years': b['Midpoint_Years'],
                'Core_CF': core_cf,
                'Non_Core_CF': 0.0,
                'Total_CF': core_cf
            })

    # Regulatory cap: balance surviving beyond 5Y goes to 5Y bucket
    survival_5y = (1 - lambda_d) ** 1825
    extra_5y = core_amt * survival_5y
    rows[-1]['Core_CF'] += extra_5y
    rows[-1]['Total_CF'] += extra_5y

    return pd.DataFrame(rows)


# Survival check with daily model
print(f"\nDecay Profile (daily model):")
print(f"  Lambda daily:       {lambda_daily:.6f}")
print(f"  S(1Y=365d):         {(1-lambda_daily)**365:.4f}")
print(f"  S(5Y=1825d):        {(1-lambda_daily)**1825:.6f}")

# %% [markdown]
# ## 4. EVE/NII Computation Functions (Reference Style)

# %%
def discount_factor_simple(rate, years):
    """DF = 1 / (1 + r)^t"""
    return 1.0 / (1.0 + rate) ** years


def compute_eve_9bucket(slotting_df, rate_curve, tenor_yrs):
    """
    EVE = sum(CF_i * DF_i) using interpolated rates.

    Parameters
    ----------
    slotting_df : DataFrame
        Must have 'Midpoint_Years' and 'Total_CF' columns.
    rate_curve : array
        Rate values at tenor points.
    tenor_yrs : array
        Tenor points in years.
    """
    # Interpolate rates to bucket midpoints
    interp_func = interp1d(tenor_yrs, rate_curve, kind='linear',
                           bounds_error=False, fill_value='extrapolate')
    midpoints = slotting_df['Midpoint_Years'].values
    rates_at_midpoints = interp_func(midpoints)
    rates_at_midpoints = np.maximum(rates_at_midpoints, 0)

    pv = 0.0
    for cf, r, t in zip(slotting_df['Total_CF'].values, rates_at_midpoints, midpoints):
        pv += cf * discount_factor_simple(r, t)
    return pv


def compute_nii_9bucket(slotting_df, rate_curve, tenor_yrs):
    """
    NII = sum(CF_i * r_i) for all buckets (simplified: full year interest).

    Parameters
    ----------
    slotting_df : DataFrame
        Must have 'Midpoint_Years' and 'Total_CF' columns.
    rate_curve : array
        Rate values at tenor points.
    tenor_yrs : array
        Tenor points in years.
    """
    interp_func = interp1d(tenor_yrs, rate_curve, kind='linear',
                           bounds_error=False, fill_value='extrapolate')
    midpoints = slotting_df['Midpoint_Years'].values
    rates_at_midpoints = interp_func(midpoints)
    rates_at_midpoints = np.maximum(rates_at_midpoints, 0)

    nii = 0.0
    for cf, r in zip(slotting_df['Total_CF'].values, rates_at_midpoints):
        nii += cf * r * 1.0  # full year interest at bucket rate
    return nii

# %% [markdown]
# ## 5. Run Monte Carlo Simulation

# %%
print("\nRunning Monte Carlo simulation (1,000 paths)...")
print("  Simulating stochastic balance paths + Vasicek interest rate paths...")

# Storage arrays
balance_paths = np.zeros((n_paths, n_days_sim + 1))
balance_paths[:, 0] = current_balance

rate_paths = np.zeros((n_paths, n_days_sim + 1))
rate_paths[:, 0] = r0

# Simulate all paths
for path in range(n_paths):
    # Generate correlated noise (balance and rate shocks are independent)
    eps_decay = np.random.normal(0, sigma_decay, n_days_sim)
    eps_rate = np.random.normal(0, 1, n_days_sim)

    for t in range(n_days_sim):
        # Balance path: stochastic decay
        decay_t = max(lambda_daily + eps_decay[t], 0)
        outflow_t = balance_paths[path, t] * decay_t
        balance_paths[path, t + 1] = balance_paths[path, t] - outflow_t

        # Rate path: Vasicek
        r_t = rate_paths[path, t]
        dr = kappa * (theta - r_t) * dt + sigma_r * np.sqrt(dt) * eps_rate[t]
        rate_paths[path, t + 1] = max(r_t + dr, 0)  # floor at 0

# Compute balance percentiles
days_sim = np.arange(n_days_sim + 1)
bal_pct_5 = np.percentile(balance_paths, 5, axis=0)
bal_pct_25 = np.percentile(balance_paths, 25, axis=0)
bal_pct_50 = np.percentile(balance_paths, 50, axis=0)
bal_pct_75 = np.percentile(balance_paths, 75, axis=0)
bal_pct_95 = np.percentile(balance_paths, 95, axis=0)

print(f"\nBalance Distribution at Year 1 (day {n_days_sim}):")
print(f"  5th percentile:   {bal_pct_5[-1]:,.2f}")
print(f"  25th percentile:  {bal_pct_25[-1]:,.2f}")
print(f"  Median:           {bal_pct_50[-1]:,.2f}")
print(f"  75th percentile:  {bal_pct_75[-1]:,.2f}")
print(f"  95th percentile:  {bal_pct_95[-1]:,.2f}")

# Rate distribution at Year 1
rate_y1 = rate_paths[:, -1]
print(f"\nShort Rate Distribution at Year 1:")
print(f"  5th percentile:   {np.percentile(rate_y1, 5)*100:.3f}%")
print(f"  Median:           {np.median(rate_y1)*100:.3f}%")
print(f"  95th percentile:  {np.percentile(rate_y1, 95)*100:.3f}%")
print(f"  Mean:             {np.mean(rate_y1)*100:.3f}%")
print(f"  Std:              {np.std(rate_y1)*100:.3f}%")

# %% [markdown]
# ## 6. Compute EVE/NII for Each Path

# %%
# Generate stochastic perturbation for core ratio (estimation uncertainty)
# sigma_core ~ 5% reflects uncertainty across different core estimation methods
sigma_core = 0.05
core_noise = np.random.normal(0, sigma_core, n_paths)

print("\nComputing EVE/NII for each simulated path at Year 1...")

# Use existing curve tenors
tenor_yrs_arr = tenors_years  # from phase_5_helpers

# Base case EVE/NII (actual current balance, reference core/non-core, base rates)
base_slot = slot_cashflows_9bucket(current_balance, ratio_ref, lambda_daily)
eve_base = compute_eve_9bucket(base_slot, base_rates, tenor_yrs_arr)
nii_base = compute_nii_9bucket(base_slot, base_rates, tenor_yrs_arr)

print(f"\n  Base EVE (current balance):  {eve_base:,.2f}")
print(f"  Base NII (current balance):  {nii_base:,.2f}")

# Storage for MC results
sim_eve_base = np.zeros(n_paths)     # EVE at base rates (per path's balance)
sim_eve_shocked = np.zeros(n_paths)  # EVE at Vasicek-shifted rates
sim_nii_base = np.zeros(n_paths)     # NII at base rates
sim_nii_shocked = np.zeros(n_paths)  # NII at Vasicek-shifted rates
sim_delta_eve = np.zeros(n_paths)    # pure rate shock effect on EVE
sim_delta_nii = np.zeros(n_paths)    # pure rate shock effect on NII
sim_core_ratios = np.zeros(n_paths)
sim_core_amounts = np.zeros(n_paths)

for i in range(n_paths):
    # Year 1 balance for this path
    bal_y1 = balance_paths[i, -1]

    # Dynamic core/non-core: base ratio from reference method on real data,
    # with stochastic perturbation to model estimation uncertainty.
    # Noise ~ N(0, sigma_core^2), clipped to [0.3, 0.95] for Basel plausibility.
    core_ratio_i = ratio_ref + core_noise[i]
    core_ratio_i = np.clip(core_ratio_i, 0.30, 0.95)
    core_amt_i = bal_y1 * core_ratio_i
    noncore_amt_i = bal_y1 - core_amt_i
    sim_core_ratios[i] = core_ratio_i
    sim_core_amounts[i] = core_amt_i

    # Slot into 9 buckets
    slot_i = slot_cashflows_9bucket(bal_y1, core_ratio_i, lambda_daily)

    # Build shocked curve: base curve + parallel shift from Vasicek rate change
    rate_change = rate_paths[i, -1] - r0  # change in short rate over 1Y
    shocked_curve = np.maximum(base_rates + rate_change, 0)  # parallel shift

    # EVE and NII at base rates (same balance, no rate shock)
    eve_b_i = compute_eve_9bucket(slot_i, base_rates, tenor_yrs_arr)
    nii_b_i = compute_nii_9bucket(slot_i, base_rates, tenor_yrs_arr)

    # EVE and NII at Vasicek-shifted rates
    eve_s_i = compute_eve_9bucket(slot_i, shocked_curve, tenor_yrs_arr)
    nii_s_i = compute_nii_9bucket(slot_i, shocked_curve, tenor_yrs_arr)

    sim_eve_base[i] = eve_b_i
    sim_eve_shocked[i] = eve_s_i
    sim_nii_base[i] = nii_b_i
    sim_nii_shocked[i] = nii_s_i

    # dEVE/dNII = pure interest rate shock effect (holding balance fixed per path)
    sim_delta_eve[i] = eve_s_i - eve_b_i
    sim_delta_nii[i] = nii_s_i - nii_b_i

print(f"\n  Paths computed: {n_paths}")
print(f"  Mean dEVE:     {np.mean(sim_delta_eve):,.2f}")
print(f"  Mean dNII:     {np.mean(sim_delta_nii):,.2f}")

# %% [markdown]
# ## 7. Risk Metrics (VaR & ES at 95% Confidence)

# %%
# VaR: 5th percentile of loss distribution
var_eve_95 = np.percentile(sim_delta_eve, 5)
var_nii_95 = np.percentile(sim_delta_nii, 5)

# ES: mean of losses worse than VaR
es_eve_95 = np.mean(sim_delta_eve[sim_delta_eve <= var_eve_95])
es_nii_95 = np.mean(sim_delta_nii[sim_delta_nii <= var_nii_95])

print("\n" + "=" * 80)
print("RISK METRICS (95% Confidence Level)")
print("=" * 80)

print(f"\n  {'Metric':<25} {'dEVE':>14} {'dNII':>14}")
print(f"  {'-'*25} {'-'*14} {'-'*14}")
print(f"  {'VaR (95%)':<25} {var_eve_95:>14,.2f} {var_nii_95:>14,.2f}")
print(f"  {'ES  (95%)':<25} {es_eve_95:>14,.2f} {es_nii_95:>14,.2f}")
print(f"  {'Mean':<25} {np.mean(sim_delta_eve):>14,.2f} {np.mean(sim_delta_nii):>14,.2f}")
print(f"  {'Std Dev':<25} {np.std(sim_delta_eve):>14,.2f} {np.std(sim_delta_nii):>14,.2f}")
print(f"  {'Min':<25} {np.min(sim_delta_eve):>14,.2f} {np.min(sim_delta_nii):>14,.2f}")
print(f"  {'Max':<25} {np.max(sim_delta_eve):>14,.2f} {np.max(sim_delta_nii):>14,.2f}")

print(f"\nDynamic Core Ratio Distribution:")
print(f"  Mean:    {np.mean(sim_core_ratios)*100:.2f}%")
print(f"  Std:     {np.std(sim_core_ratios)*100:.2f}%")
print(f"  Min:     {np.min(sim_core_ratios)*100:.2f}%")
print(f"  Max:     {np.max(sim_core_ratios)*100:.2f}%")
print(f"  Median:  {np.median(sim_core_ratios)*100:.2f}%")

# %% [markdown]
# ## 8. Comparison: Fixed vs Dynamic Core Ratio

# %%
print("\n" + "=" * 80)
print("COMPARISON: FIXED vs DYNAMIC CORE RATIO")
print("=" * 80)

# Re-run with fixed core ratio for comparison (same rate shock isolation)
sim_delta_eve_fixed = np.zeros(n_paths)
sim_delta_nii_fixed = np.zeros(n_paths)

for i in range(n_paths):
    bal_y1 = balance_paths[i, -1]

    # Fixed core ratio (from phase_1c)
    slot_fixed = slot_cashflows_9bucket(bal_y1, core_ratio_primary, lambda_daily)

    rate_change = rate_paths[i, -1] - r0
    shocked_curve = np.maximum(base_rates + rate_change, 0)

    # EVE/NII at base rates and shocked rates (same balance)
    eve_b_fixed = compute_eve_9bucket(slot_fixed, base_rates, tenor_yrs_arr)
    eve_s_fixed = compute_eve_9bucket(slot_fixed, shocked_curve, tenor_yrs_arr)
    nii_b_fixed = compute_nii_9bucket(slot_fixed, base_rates, tenor_yrs_arr)
    nii_s_fixed = compute_nii_9bucket(slot_fixed, shocked_curve, tenor_yrs_arr)

    sim_delta_eve_fixed[i] = eve_s_fixed - eve_b_fixed
    sim_delta_nii_fixed[i] = nii_s_fixed - nii_b_fixed

var_eve_fixed = np.percentile(sim_delta_eve_fixed, 5)
es_eve_fixed = np.mean(sim_delta_eve_fixed[sim_delta_eve_fixed <= var_eve_fixed])
var_nii_fixed = np.percentile(sim_delta_nii_fixed, 5)
es_nii_fixed = np.mean(sim_delta_nii_fixed[sim_delta_nii_fixed <= var_nii_fixed])

print(f"\n  {'Metric':<25} {'Dynamic Core':>14} {'Fixed Core':>14}")
print(f"  {'-'*25} {'-'*14} {'-'*14}")
print(f"  {'VaR dEVE (95%)':<25} {var_eve_95:>14,.2f} {var_eve_fixed:>14,.2f}")
print(f"  {'ES  dEVE (95%)':<25} {es_eve_95:>14,.2f} {es_eve_fixed:>14,.2f}")
print(f"  {'VaR dNII (95%)':<25} {var_nii_95:>14,.2f} {var_nii_fixed:>14,.2f}")
print(f"  {'ES  dNII (95%)':<25} {es_nii_95:>14,.2f} {es_nii_fixed:>14,.2f}")
print(f"  {'Mean dEVE':<25} {np.mean(sim_delta_eve):>14,.2f} {np.mean(sim_delta_eve_fixed):>14,.2f}")
print(f"  {'Mean dNII':<25} {np.mean(sim_delta_nii):>14,.2f} {np.mean(sim_delta_nii_fixed):>14,.2f}")

print(f"\n  Fixed core ratio:   {core_ratio_primary*100:.2f}%")
print(f"  Dynamic core ratio: {np.mean(sim_core_ratios)*100:.2f}% (mean)")

# %% [markdown]
# ## 9. Charts

# %%
# Chart 1: Fan chart of balance paths
fig, ax = plt.subplots(figsize=(14, 8))
days_axis = days_sim / 365

ax.fill_between(days_axis, bal_pct_5, bal_pct_95, alpha=0.15,
                color='#E63946', label='5th-95th pct')
ax.fill_between(days_axis, bal_pct_25, bal_pct_75, alpha=0.3,
                color='#F77F00', label='25th-75th pct')
ax.plot(days_axis, bal_pct_50, linewidth=2.5, color='#023047', label='Median')
ax.plot(days_axis, bal_pct_5, linewidth=1, color='#E63946', linestyle='--', alpha=0.7)
ax.plot(days_axis, bal_pct_95, linewidth=1, color='#E63946', linestyle='--', alpha=0.7)

ax.set_xlabel('Time (Years)', fontsize=12)
ax.set_ylabel('Balance', fontsize=12)
ax.set_title(f'Monte Carlo Balance Paths ({n_paths} simulations, 1-Year Horizon)',
             fontsize=14, fontweight='bold')
ax.legend(fontsize=10, loc='upper right')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# %%
# Chart 2: Distribution of simulated core ratios
fig, ax = plt.subplots(figsize=(12, 7))
ax.hist(sim_core_ratios * 100, bins=50, color='#2A9D8F', alpha=0.7, edgecolor='black')
ax.axvline(x=core_ratio_primary * 100, color='#E63946', linewidth=2.5, linestyle='--',
           label=f'Fixed Core Ratio: {core_ratio_primary*100:.1f}%')
ax.axvline(x=np.mean(sim_core_ratios) * 100, color='#023047', linewidth=2,
           label=f'Mean Dynamic: {np.mean(sim_core_ratios)*100:.1f}%')

ax.set_xlabel('Core Ratio (%)', fontsize=12)
ax.set_ylabel('Frequency', fontsize=12)
ax.set_title('Distribution of Dynamic Core Ratios Across MC Paths',
             fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.show()

# %%
# Chart 3: Histogram of dEVE (dynamic core)
fig, ax = plt.subplots(figsize=(12, 7))
ax.hist(sim_delta_eve, bins=50, color='#2A9D8F', alpha=0.7, edgecolor='black')
ax.axvline(x=var_eve_95, color='#E63946', linewidth=2.5, linestyle='--',
           label=f'VaR 95%: {var_eve_95:,.1f}')
ax.axvline(x=es_eve_95, color='#D62828', linewidth=2.5, linestyle='-.',
           label=f'ES 95%: {es_eve_95:,.1f}')
ax.axvline(x=np.mean(sim_delta_eve), color='#023047', linewidth=2,
           label=f'Mean: {np.mean(sim_delta_eve):,.1f}')

ax.set_xlabel('dEVE (Stochastic Rates + Dynamic Core)', fontsize=12)
ax.set_ylabel('Frequency', fontsize=12)
ax.set_title('Distribution of Simulated dEVE at Year 1', fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.show()

# %%
# Chart 4: Histogram of dNII (dynamic core)
fig, ax = plt.subplots(figsize=(12, 7))
ax.hist(sim_delta_nii, bins=50, color='#264653', alpha=0.7, edgecolor='black')
ax.axvline(x=var_nii_95, color='#E63946', linewidth=2.5, linestyle='--',
           label=f'VaR 95%: {var_nii_95:,.1f}')
ax.axvline(x=es_nii_95, color='#D62828', linewidth=2.5, linestyle='-.',
           label=f'ES 95%: {es_nii_95:,.1f}')
ax.axvline(x=np.mean(sim_delta_nii), color='#023047', linewidth=2,
           label=f'Mean: {np.mean(sim_delta_nii):,.1f}')

ax.set_xlabel('dNII (Stochastic Rates + Dynamic Core)', fontsize=12)
ax.set_ylabel('Frequency', fontsize=12)
ax.set_title('Distribution of Simulated dNII at Year 1', fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 10. Save Outputs

# %%
# Save summary
mc_summary = pd.DataFrame({
    'Metric': [
        'n_paths', 'horizon_days', 'lambda_daily', 'sigma_decay',
        'r0', 'kappa', 'theta', 'sigma_r',
        'Core_Method', 'Base_EVE', 'Base_NII',
        'Balance_Y1_5pct', 'Balance_Y1_Median', 'Balance_Y1_95pct',
        'Rate_Y1_5pct', 'Rate_Y1_Median', 'Rate_Y1_95pct',
        'Dynamic_Core_Ratio_Mean', 'Dynamic_Core_Ratio_Std',
        'VaR_dEVE_95_Dynamic', 'ES_dEVE_95_Dynamic',
        'VaR_dNII_95_Dynamic', 'ES_dNII_95_Dynamic',
        'VaR_dEVE_95_Fixed', 'ES_dEVE_95_Fixed',
        'VaR_dNII_95_Fixed', 'ES_dNII_95_Fixed',
        'Mean_dEVE_Dynamic', 'Std_dEVE_Dynamic',
        'Mean_dNII_Dynamic', 'Std_dNII_Dynamic'
    ],
    'Value': [
        n_paths, n_days_sim, lambda_daily, sigma_decay,
        r0, kappa, theta, sigma_r,
        'max(q05,trend,histmin)*0.9', eve_base, nii_base,
        bal_pct_5[-1], bal_pct_50[-1], bal_pct_95[-1],
        np.percentile(rate_y1, 5), np.median(rate_y1), np.percentile(rate_y1, 95),
        np.mean(sim_core_ratios), np.std(sim_core_ratios),
        var_eve_95, es_eve_95,
        var_nii_95, es_nii_95,
        var_eve_fixed, es_eve_fixed,
        var_nii_fixed, es_nii_fixed,
        np.mean(sim_delta_eve), np.std(sim_delta_eve),
        np.mean(sim_delta_nii), np.std(sim_delta_nii)
    ]
})
mc_summary.to_csv('monte_carlo_enhanced_summary.csv', index=False)

# Save sample paths
sample_idx = list(range(0, n_paths, 100))
sample_paths_df = pd.DataFrame({'Day': days_sim})
for idx in sample_idx:
    sample_paths_df[f'Balance_Path_{idx}'] = balance_paths[idx, :]
    sample_paths_df[f'Rate_Path_{idx}'] = rate_paths[idx, :]
sample_paths_df['Balance_Pct_5'] = bal_pct_5
sample_paths_df['Balance_Pct_25'] = bal_pct_25
sample_paths_df['Balance_Pct_50'] = bal_pct_50
sample_paths_df['Balance_Pct_75'] = bal_pct_75
sample_paths_df['Balance_Pct_95'] = bal_pct_95
sample_paths_df.to_csv('monte_carlo_enhanced_paths.csv', index=False)

print("\nSaved: monte_carlo_enhanced_summary.csv, monte_carlo_enhanced_paths.csv")

print("\n" + "=" * 80)
print("Section 5d (Enhanced) Complete: 4 charts generated")
print("=" * 80)

# %%
