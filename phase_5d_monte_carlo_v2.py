# %% [markdown]
# # Phase 5d (v2): Monte Carlo Simulation for IRRBB
#
# ## QF609 Project #1 - IRRBB Modelling Framework
#
# **Calculation Date:** 30-Dec-2023
#
# Monte Carlo simulation using Phase 1-4 results:
# - Stochastic balance paths (noisy net flow rate) over 5 years
# - Stochastic interest rate paths (Vasicek model)
# - Core ratio from config.json (Phase 1c Detrended Regression)
# - 11 IRRBB buckets with Portfolio KM survival curve (Phase 1b/2)
# - Actual yield curve from processed_curve_data.csv (Phase 1a)
# - EVE and NII sensitivity under 4 shock scenarios (Phase 3/4)
# - VaR/ES risk metrics at 95% confidence

# %% [markdown]
# ## Section 1: Setup & Data Loading

# %%
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import warnings

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (14, 6)
plt.rcParams['font.size'] = 11

print("=" * 80)
print("PHASE 5d (v2): MONTE CARLO SIMULATION FOR IRRBB")
print("=" * 80)

# %%
# Load raw data
df = pd.read_excel('group-proj-1-data.xlsx')
df.columns = ['Date', 'Balance', 'Inflow', 'Outflow', 'Netflow']
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date').reset_index(drop=True)

# Calculation date
calc_date = pd.Timestamp('2023-12-30')
calc_balance = df[df['Date'] == calc_date]['Balance'].values[0]

# Compute daily decay rate = Outflow / Balance (for SLOTTING survival curve)
df['Daily_Decay_Rate'] = df['Outflow'] / df['Balance']
lambda_daily = df['Daily_Decay_Rate'].mean()
sigma_decay = df['Daily_Decay_Rate'].std()

# Compute net flow rate = Netflow / Balance (for MC BALANCE projection)
df['Net_Flow_Rate'] = df['Netflow'] / df['Balance']
mu_net = df['Net_Flow_Rate'].mean()
sigma_net = df['Net_Flow_Rate'].std()

print(f"\nData Period: {df['Date'].min().strftime('%Y-%m-%d')} to {df['Date'].max().strftime('%Y-%m-%d')}")
print(f"Total observations: {len(df)}")
print(f"Calculation Date: {calc_date.strftime('%Y-%m-%d')}")
print(f"Balance at calc date: {calc_balance:,.2f}")
print(f"\nCalibrated Parameters:")
print(f"  GROSS Decay (for slotting):")
print(f"    Lambda (mean daily outflow rate): {lambda_daily:.6f}")
print(f"    Sigma:                            {sigma_decay:.6f}")
print(f"  NET Flow (for MC balance paths):")
print(f"    Mu (mean daily net rate):         {mu_net:.6f}")
print(f"    Sigma:                            {sigma_net:.6f}")
print(f"    Days with positive net flow:      {(df['Netflow'] > 0).mean():.1%}")

# %% [markdown]
# ## Section 2: Core vs Non-Core Deposit Separation
#
# Loads core ratio from config.json (produced by Phase 1c Detrended Regression).
# Core ratio = 83.57% — used as the base for MC perturbation.

# %%
# Load core ratio from Phase 1c output
with open('config.json', 'r') as f:
    config = json.load(f)

core_pct = config['core_ratio_pct'] / 100  # 0.8357
non_core_pct = 1 - core_pct

print(f"\n{'='*60}")
print(f"CORE vs NON-CORE SEPARATION (from config.json)")
print(f"{'='*60}")
print(f"  Method:           {config['method']}")
print(f"  Core Ratio:       {core_pct:.4f}  ({core_pct:.2%})")
print(f"  Non-Core Ratio:   {non_core_pct:.4f}  ({non_core_pct:.2%})")
print(f"  Core Amount:      {config['core_amount']:>12,.2f}")
print(f"  Non-Core Amount:  {config['non_core_amount']:>12,.2f}")
print(f"  Total Balance:    {calc_balance:>12,.2f}")

# %% [markdown]
# ## Section 3: Cash Flow Slotting (9 Basel Buckets)
#
# Following reference methodology:
# - Non-core -> O/N bucket (immediate repricing)
# - Core -> distributed across 1M-5Y using exponential decay profile
# - Survival: S(t) = exp(-avg_monthly_outflow * t_months)
# - Anything surviving beyond 5Y (60 months) is capped at 5Y bucket

# %%
# Compute average monthly outflow rate from daily data
# Monthly decay from monthly aggregated data (like reference)
df_monthly = df.set_index('Date').resample('M').agg({
    'Balance': 'mean',
    'Outflow': 'sum'
}).dropna()
df_monthly['Monthly_Outflow_Rate'] = df_monthly['Outflow'] / df_monthly['Balance']
avg_monthly_outflow = df_monthly['Monthly_Outflow_Rate'].mean()

# Stressed monthly decay (95th percentile)
stressed_monthly_decay = df_monthly['Monthly_Outflow_Rate'].quantile(0.95)

print(f"\nMonthly Decay Calibration:")
print(f"  Avg monthly outflow rate:     {avg_monthly_outflow:.4%}")
print(f"  Stressed (95th pctl):         {stressed_monthly_decay:.4%}")
half_life_m = np.log(2) / avg_monthly_outflow if avg_monthly_outflow > 0 else 999
print(f"  Implied half-life:            {half_life_m:.1f} months ({half_life_m/12:.1f} years)")

# %%
# 9 Basel time buckets (in months) with midpoints in years for discounting
buckets = [
    {'name': 'O/N',  'start_m': 0,  'end_m': 0,  'midpoint_yrs': 1/365},
    {'name': '1M',   'start_m': 0,  'end_m': 1,  'midpoint_yrs': 1/24},
    {'name': '3M',   'start_m': 1,  'end_m': 3,  'midpoint_yrs': 2/12},
    {'name': '6M',   'start_m': 3,  'end_m': 6,  'midpoint_yrs': 4.5/12},
    {'name': '1Y',   'start_m': 6,  'end_m': 12, 'midpoint_yrs': 9/12},
    {'name': '2Y',   'start_m': 12, 'end_m': 24, 'midpoint_yrs': 1.5},
    {'name': '3Y',   'start_m': 24, 'end_m': 36, 'midpoint_yrs': 2.5},
    {'name': '4Y',   'start_m': 36, 'end_m': 48, 'midpoint_yrs': 3.5},
    {'name': '5Y',   'start_m': 48, 'end_m': 60, 'midpoint_yrs': 4.5},
]


def slot_cashflows(balance, core_ratio, monthly_decay):
    """
    Slot NMD balance into 9 Basel buckets.

    Non-core -> O/N.
    Core -> distributed across 1M-5Y using S(t) = exp(-monthly_decay * t).
    Balance surviving beyond 60M -> capped at 5Y.

    Returns dict with bucket names as keys and cash flow amounts as values,
    plus a list of midpoint years.
    """
    core_amt = balance * core_ratio
    non_core_amt = balance * (1 - core_ratio)

    allocations = {}
    midpoints = {}

    for b in buckets:
        midpoints[b['name']] = b['midpoint_yrs']

        if b['name'] == 'O/N':
            allocations['O/N'] = non_core_amt
            continue

        s_start = np.exp(-monthly_decay * b['start_m'])
        s_end = np.exp(-monthly_decay * b['end_m'])
        allocations[b['name']] = core_amt * (s_start - s_end)

    # Regulatory cap: surviving beyond 5Y -> add to 5Y bucket
    survival_5y = np.exp(-monthly_decay * 60)
    allocations['5Y'] += core_amt * survival_5y

    return allocations, midpoints


# Slot the current balance
alloc_base, midpts = slot_cashflows(calc_balance, core_pct, avg_monthly_outflow)

print(f"\nBasel Cash Flow Slotting")
print(f"{'='*60}")
print(f"  {'Bucket':<8} {'Midpoint(Y)':>12} {'Cash Flow':>14} {'% of Total':>12}")
print(f"  {'-'*48}")
total_cf = 0
for bname in [b['name'] for b in buckets]:
    cf = alloc_base[bname]
    total_cf += cf
    pct = cf / calc_balance * 100
    print(f"  {bname:<8} {midpts[bname]:>12.4f} {cf:>14,.2f} {pct:>11.2f}%")
print(f"  {'-'*48}")
print(f"  {'TOTAL':<8} {'':<12} {total_cf:>14,.2f} {total_cf/calc_balance*100:>11.2f}%")

# %% [markdown]
# ## Section 4: Interest Rate Curve, Shock Scenarios & EVE/NII Functions
#
# Following reference: SGD swap curve as of 31-Dec-2023 with 4 shock scenarios.
# - EVE = sum(CF * DF) where DF = 1/(1+r)^t
# - NII = sum(CF * r) for full year interest at bucket rate

# %%
# === Base Interest Rate Curve (Reference: SGD rates end-2023) ===
rate_tenors = {
    'O/N': 1/365, '1M': 1/12, '3M': 3/12, '6M': 6/12,
    '1Y': 1.0, '2Y': 2.0, '3Y': 3.0, '4Y': 4.0, '5Y': 5.0,
}

base_rates = {
    'O/N': 0.0375, '1M': 0.0380, '3M': 0.0385, '6M': 0.0370,
    '1Y': 0.0350, '2Y': 0.0330, '3Y': 0.0320, '4Y': 0.0315, '5Y': 0.0310,
}

tenor_years = np.array(list(rate_tenors.values()))
tenor_labels = list(rate_tenors.keys())
base_curve = np.array(list(base_rates.values()))

# === 4 Shock Scenarios ===
# S1: +200bps parallel
shock_up = base_curve + 0.0200

# S2: -200bps parallel (floored at 0)
shock_down = np.maximum(base_curve - 0.0200, 0.0)

# S3: Short rate up (+200bps tapering to 0 at 5Y)
taper_weights = np.maximum(1 - tenor_years / 5.0, 0)
shock_short_up = base_curve + 0.0200 * taper_weights

# S4: Flattener (short +200bps to long -100bps)
flatten_shocks = 0.0200 - (0.0200 + 0.0100) * tenor_years / 5.0
shock_flattener = np.maximum(base_curve + flatten_shocks, 0.0)

scenarios = {
    'Base': base_curve,
    '+200bps Parallel': shock_up,
    '-200bps Parallel': shock_down,
    'Short Rate Up': shock_short_up,
    'Flattener': shock_flattener,
}

# Display rate table
print(f"\nInterest Rate Scenarios (%)")
print(f"{'='*75}")
print(f"  {'Tenor':<6} {'Base':>8} {'+200bp':>8} {'-200bp':>8} {'ShortUp':>8} {'Flatten':>8}")
print(f"  {'-'*50}")
for i, label in enumerate(tenor_labels):
    print(f"  {label:<6} {base_curve[i]*100:>8.3f} {shock_up[i]*100:>8.3f} "
          f"{shock_down[i]*100:>8.3f} {shock_short_up[i]*100:>8.3f} "
          f"{shock_flattener[i]*100:>8.3f}")

# %%
# === EVE and NII Functions ===

def compute_eve(allocations, midpoints_dict, rate_curve_arr, tenor_yrs_arr):
    """
    EVE = sum(CF_i * DF_i) where DF = 1/(1+r)^t.
    Rates interpolated from curve to bucket midpoints.
    """
    interp_func = interp1d(tenor_yrs_arr, rate_curve_arr, kind='linear',
                           bounds_error=False, fill_value='extrapolate')
    eve = 0.0
    for bname in allocations:
        cf = allocations[bname]
        t = midpoints_dict[bname]
        r = max(interp_func(t), 0)
        df = 1.0 / (1.0 + r) ** t
        eve += cf * df
    return eve


def compute_nii(allocations, midpoints_dict, rate_curve_arr, tenor_yrs_arr):
    """
    NII = sum(CF_i * r_i) - simplified full year interest at bucket rate.
    """
    interp_func = interp1d(tenor_yrs_arr, rate_curve_arr, kind='linear',
                           bounds_error=False, fill_value='extrapolate')
    nii = 0.0
    for bname in allocations:
        cf = allocations[bname]
        t = midpoints_dict[bname]
        r = max(interp_func(t), 0)
        nii += cf * r * 1.0
    return nii


# Compute base EVE and NII
eve_base = compute_eve(alloc_base, midpts, base_curve, tenor_years)
nii_base = compute_nii(alloc_base, midpts, base_curve, tenor_years)

# Compute EVE/NII under all scenarios
print(f"\n{'='*65}")
print(f"EVE Sensitivity Analysis")
print(f"{'='*65}")
print(f"  {'Scenario':<22} {'EVE':>14} {'dEVE':>14} {'dEVE %':>10}")
print(f"  {'-'*60}")

eve_deltas = {}
for name, curve in scenarios.items():
    eve_s = compute_eve(alloc_base, midpts, curve, tenor_years)
    delta = eve_s - eve_base
    delta_pct = delta / eve_base * 100 if eve_base != 0 else 0
    eve_deltas[name] = delta
    marker = " <-- BASE" if name == 'Base' else ""
    print(f"  {name:<22} {eve_s:>14,.2f} {delta:>14,.2f} {delta_pct:>9.4f}%{marker}")

shock_eve = {k: v for k, v in eve_deltas.items() if k != 'Base'}
worst_eve = min(shock_eve, key=shock_eve.get)
print(f"\n  WORST CASE: {worst_eve} (dEVE = {shock_eve[worst_eve]:,.2f})")

# NII sensitivity
print(f"\n{'='*65}")
print(f"NII Sensitivity Analysis (1Y Window)")
print(f"{'='*65}")
print(f"  {'Scenario':<22} {'NII':>14} {'dNII':>14} {'dNII %':>10}")
print(f"  {'-'*60}")

nii_deltas = {}
for name, curve in scenarios.items():
    nii_s = compute_nii(alloc_base, midpts, curve, tenor_years)
    delta = nii_s - nii_base
    delta_pct = delta / nii_base * 100 if nii_base != 0 else 0
    nii_deltas[name] = delta
    marker = " <-- BASE" if name == 'Base' else ""
    print(f"  {name:<22} {nii_s:>14,.2f} {delta:>14,.2f} {delta_pct:>9.4f}%{marker}")

shock_nii = {k: v for k, v in nii_deltas.items() if k != 'Base'}
worst_nii = min(shock_nii, key=shock_nii.get)
print(f"\n  WORST CASE: {worst_nii} (dNII = {shock_nii[worst_nii]:,.2f})")

# %% [markdown]
# ## Section 5: Monte Carlo Simulation
#
# Simulate 1,000 paths over 5 years with two stochastic drivers:
#
# **1. Balance paths (stochastic NET flow):**
# - Each day: net_rate = mu_net + noise, Balance(t+1) = Balance(t) + Balance(t) * net_rate
# - Uses NET flow (Inflow - Outflow) so balance stays realistic
# - Noise ~ N(0, sigma_net^2)
# - Note: GROSS outflow rate (lambda_daily) is still used for SLOTTING survival curve
#
# **2. Interest rate paths (Vasicek model):**
# - dr = kappa * (theta - r) * dt + sigma_r * sqrt(dt) * dW
# - Mean-reverts to theta (long-run rate), with speed kappa and volatility sigma_r
#
# At Year 1, for each path we:
# - Take the simulated balance
# - Compute core/non-core using reference method (with stochastic perturbation)
# - Slot into 9 buckets (using gross outflow decay for survival curve)
# - Shift the base yield curve by the Vasicek rate change
# - Compute EVE/NII under base curve AND shocked curve (pure rate effect)

# %%
# === Simulation Parameters ===
n_paths = 1000
n_days_sim = 365 * 5  # 5-year horizon for balance paths
seed = 42

# Vasicek parameters
r0 = base_curve[0]           # initial short rate = O/N rate
kappa_v = 0.5                # mean-reversion speed
theta_v = base_curve[0]      # long-run mean rate
sigma_r = 0.01               # rate volatility (100bps annual)
dt_v = 1 / 365               # daily time step

# Core ratio perturbation
sigma_core = 0.05            # uncertainty in core ratio estimate

np.random.seed(seed)

print(f"\n{'='*60}")
print(f"MONTE CARLO SIMULATION PARAMETERS")
print(f"{'='*60}")
print(f"  Paths:                {n_paths}")
print(f"  Horizon:              {n_days_sim} days ({n_days_sim//365} years)")
print(f"  Seed:                 {seed}")
print(f"\n  Balance Path (NET flow model):")
print(f"    Mu net daily:       {mu_net:.6f}")
print(f"    Sigma net daily:    {sigma_net:.6f}")
print(f"  Slotting Decay (GROSS outflow):")
print(f"    Monthly outflow:    {avg_monthly_outflow:.4%}")
print(f"  Interest Rate Path (Vasicek):")
print(f"    r0:                 {r0:.4%}")
print(f"    Kappa:              {kappa_v}")
print(f"    Theta:              {theta_v:.4%}")
print(f"    Sigma_r:            {sigma_r:.4f}")
print(f"  Core Ratio:")
print(f"    Base ratio:         {core_pct:.4f} ({core_pct:.1%})")
print(f"    Sigma perturbation: {sigma_core}")

# %%
# === Run Simulation ===
print(f"\nSimulating {n_paths} paths...")

# Storage: balance paths (5Y) and rate paths (1Y only, for EVE/NII eval)
balance_paths = np.zeros((n_paths, n_days_sim + 1))
balance_paths[:, 0] = calc_balance

n_days_rate = 365  # simulate rates for 1Y (evaluation horizon for EVE/NII)
rate_paths = np.zeros((n_paths, n_days_rate + 1))
rate_paths[:, 0] = r0

for path in range(n_paths):
    # Generate noise
    eps_net = np.random.normal(0, sigma_net, n_days_sim)
    eps_rate = np.random.normal(0, 1, n_days_rate)

    # Balance path: 5 years using NET flow
    for t in range(n_days_sim):
        net_rate_t = mu_net + eps_net[t]
        balance_paths[path, t + 1] = max(balance_paths[path, t] * (1 + net_rate_t), 0)

    # Rate path: 1 year (Vasicek)
    for t in range(n_days_rate):
        r_t = rate_paths[path, t]
        dr = kappa_v * (theta_v - r_t) * dt_v + sigma_r * np.sqrt(dt_v) * eps_rate[t]
        rate_paths[path, t + 1] = max(r_t + dr, 0)  # floor at 0

# Balance percentiles over 5 years
days_sim = np.arange(n_days_sim + 1)
bal_pct_5 = np.percentile(balance_paths, 5, axis=0)
bal_pct_25 = np.percentile(balance_paths, 25, axis=0)
bal_pct_50 = np.percentile(balance_paths, 50, axis=0)
bal_pct_75 = np.percentile(balance_paths, 75, axis=0)
bal_pct_95 = np.percentile(balance_paths, 95, axis=0)

print(f"\nBalance Distribution at Year 1 (day 365):")
print(f"  5th percentile:   {bal_pct_5[365]:>12,.2f}")
print(f"  25th percentile:  {bal_pct_25[365]:>12,.2f}")
print(f"  Median:           {bal_pct_50[365]:>12,.2f}")
print(f"  75th percentile:  {bal_pct_75[365]:>12,.2f}")
print(f"  95th percentile:  {bal_pct_95[365]:>12,.2f}")

print(f"\nBalance Distribution at Year 5 (day {n_days_sim}):")
print(f"  5th percentile:   {bal_pct_5[-1]:>12,.2f}")
print(f"  Median:           {bal_pct_50[-1]:>12,.2f}")
print(f"  95th percentile:  {bal_pct_95[-1]:>12,.2f}")

# Rate distribution at Year 1
rate_y1 = rate_paths[:, -1]
print(f"\nShort Rate Distribution at Year 1:")
print(f"  5th percentile:   {np.percentile(rate_y1, 5):.3%}")
print(f"  Median:           {np.median(rate_y1):.3%}")
print(f"  95th percentile:  {np.percentile(rate_y1, 95):.3%}")
print(f"  Mean:             {np.mean(rate_y1):.3%}")
print(f"  Std:              {np.std(rate_y1):.3%}")

print(f"\nSimulation complete.")

# %%
# === Section 5 Charts: Balance Fan Chart + Rate Paths ===

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# --- Chart 1: Balance paths fan chart (5Y) ---
ax1 = axes[0]
years_sim = days_sim / 365

# Plot 20 sample paths in light gray
for i in range(0, n_paths, n_paths // 20):
    ax1.plot(years_sim, balance_paths[i], color='gray', alpha=0.15, linewidth=0.5)

# Percentile bands
ax1.fill_between(years_sim, bal_pct_5, bal_pct_95, alpha=0.15, color='steelblue', label='5th-95th pctl')
ax1.fill_between(years_sim, bal_pct_25, bal_pct_75, alpha=0.3, color='steelblue', label='25th-75th pctl')
ax1.plot(years_sim, bal_pct_50, color='darkblue', linewidth=2, label='Median')
ax1.axhline(y=calc_balance, color='red', linestyle='--', linewidth=1, label=f'Current ({calc_balance:,.0f})')

ax1.set_xlabel('Years')
ax1.set_ylabel('Balance')
ax1.set_title(f'Monte Carlo Balance Paths ({n_paths} paths)')
ax1.legend(loc='upper left', fontsize=9)
ax1.set_xlim(0, 5)

# --- Chart 2: Interest rate paths (1Y) ---
ax2 = axes[1]
days_rate = np.arange(n_days_rate + 1)
years_rate = days_rate / 365

# Plot 30 sample paths
for i in range(0, n_paths, n_paths // 30):
    ax2.plot(years_rate, rate_paths[i] * 100, color='gray', alpha=0.15, linewidth=0.5)

# Percentile bands
rate_p5 = np.percentile(rate_paths, 5, axis=0) * 100
rate_p25 = np.percentile(rate_paths, 25, axis=0) * 100
rate_p50 = np.percentile(rate_paths, 50, axis=0) * 100
rate_p75 = np.percentile(rate_paths, 75, axis=0) * 100
rate_p95 = np.percentile(rate_paths, 95, axis=0) * 100

ax2.fill_between(years_rate, rate_p5, rate_p95, alpha=0.15, color='darkorange', label='5th-95th pctl')
ax2.fill_between(years_rate, rate_p25, rate_p75, alpha=0.3, color='darkorange', label='25th-75th pctl')
ax2.plot(years_rate, rate_p50, color='darkred', linewidth=2, label='Median')
ax2.axhline(y=r0 * 100, color='red', linestyle='--', linewidth=1, label=f'Initial ({r0:.2%})')

ax2.set_xlabel('Years')
ax2.set_ylabel('Short Rate (%)')
ax2.set_title(f'Vasicek Short Rate Paths ({n_paths} paths)')
ax2.legend(loc='upper right', fontsize=9)
ax2.set_xlim(0, 1)

plt.tight_layout()
plt.savefig('mc_section5_paths.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: mc_section5_paths.png")

# %% [markdown]
# ## Section 6: EVE/NII per MC Path & Risk Metrics
#
# For each of the 1,000 paths at Year 1:
# 1. Take simulated balance and short rate
# 2. Perturb core ratio (stochastic uncertainty around base estimate)
# 3. Slot into 9 buckets using gross outflow decay
# 4. Shift base yield curve by simulated rate change (parallel shift)
# 5. Compute EVE and NII under the path's curve AND under 4 shock scenarios
# 6. dEVE = EVE(path) - EVE(static base), dNII = NII(path) - NII(static base)
#
# Risk metrics: VaR (5th percentile of losses) and ES (mean beyond VaR) at 95%

# %%
print(f"\n{'='*70}")
print(f"SECTION 6: EVE/NII PER MC PATH & RISK METRICS")
print(f"{'='*70}")
print(f"Processing {n_paths} paths...")

# Storage for per-path results
mc_results = []

for i in range(n_paths):
    # 1. Year-1 balance from simulated path
    bal_y1 = balance_paths[i, 365]

    # 2. Year-1 short rate from Vasicek path
    r_y1 = rate_paths[i, -1]
    rate_shift = r_y1 - r0  # change in short rate vs initial

    # 3. Perturb core ratio (clip to [0.5, 0.99] to keep realistic)
    core_ratio_i = np.clip(core_pct + np.random.normal(0, sigma_core), 0.50, 0.99)

    # 4. Slot into 9 buckets using this path's balance + core ratio
    alloc_i, midpts_i = slot_cashflows(bal_y1, core_ratio_i, avg_monthly_outflow)

    # 5. Build this path's yield curve: base curve + parallel shift from Vasicek
    curve_i = np.maximum(base_curve + rate_shift, 0)

    # 6. Compute EVE and NII under this path's curve (no additional shock)
    eve_i = compute_eve(alloc_i, midpts_i, curve_i, tenor_years)
    nii_i = compute_nii(alloc_i, midpts_i, curve_i, tenor_years)

    # 7. Compute EVE/NII under 4 shock scenarios applied ON TOP of path's curve
    path_row = {
        'path': i,
        'balance_y1': bal_y1,
        'rate_y1': r_y1,
        'rate_shift': rate_shift,
        'core_ratio': core_ratio_i,
        'eve_path': eve_i,
        'nii_path': nii_i,
        'dEVE_path': eve_i - eve_base,
        'dNII_path': nii_i - nii_base,
    }

    # Apply each shock scenario on top of the path's curve
    for sname, shock_curve in scenarios.items():
        if sname == 'Base':
            continue
        # Shock applied to the PATH's curve (not the static base)
        shocked_i = np.maximum(curve_i + (shock_curve - base_curve), 0)
        eve_s = compute_eve(alloc_i, midpts_i, shocked_i, tenor_years)
        nii_s = compute_nii(alloc_i, midpts_i, shocked_i, tenor_years)
        skey = sname.replace(' ', '_').replace('+', 'up').replace('-', 'dn')
        path_row[f'eve_{skey}'] = eve_s
        path_row[f'nii_{skey}'] = nii_s
        path_row[f'dEVE_{skey}'] = eve_s - eve_base
        path_row[f'dNII_{skey}'] = nii_s - nii_base

    mc_results.append(path_row)

mc_df = pd.DataFrame(mc_results)
print(f"Done. {len(mc_df)} paths processed.")

# %%
# === Risk Metrics: VaR and ES at 95% confidence ===

print(f"\n{'='*70}")
print(f"RISK METRICS (95% Confidence Level)")
print(f"{'='*70}")

# dEVE from path's own curve (balance + rate risk combined)
dEVE_path = mc_df['dEVE_path'].values
dNII_path = mc_df['dNII_path'].values

def compute_var_es(losses, confidence=0.95):
    """VaR = percentile of losses, ES = mean of losses beyond VaR."""
    # For losses: negative dEVE/dNII means loss
    var = np.percentile(losses, (1 - confidence) * 100)
    es = losses[losses <= var].mean() if (losses <= var).any() else var
    return var, es

print(f"\n  --- Path Risk (Balance + Rate combined, no additional shock) ---")
var_eve, es_eve = compute_var_es(dEVE_path)
var_nii, es_nii = compute_var_es(dNII_path)
print(f"  dEVE:  Mean={np.mean(dEVE_path):>10,.2f}  Std={np.std(dEVE_path):>10,.2f}")
print(f"         VaR(95%)={var_eve:>10,.2f}  ES(95%)={es_eve:>10,.2f}")
print(f"  dNII:  Mean={np.mean(dNII_path):>10,.2f}  Std={np.std(dNII_path):>10,.2f}")
print(f"         VaR(95%)={var_nii:>10,.2f}  ES(95%)={es_nii:>10,.2f}")

# Worst-case shock across all scenarios per path
print(f"\n  --- Worst-Case Shock per Path ---")
shock_keys = [c for c in mc_df.columns if c.startswith('dEVE_') and c != 'dEVE_path']
nii_shock_keys = [c for c in mc_df.columns if c.startswith('dNII_') and c != 'dNII_path']

mc_df['worst_dEVE'] = mc_df[shock_keys].min(axis=1)
mc_df['worst_dNII'] = mc_df[nii_shock_keys].min(axis=1)

var_worst_eve, es_worst_eve = compute_var_es(mc_df['worst_dEVE'].values)
var_worst_nii, es_worst_nii = compute_var_es(mc_df['worst_dNII'].values)

print(f"  Worst dEVE:  Mean={mc_df['worst_dEVE'].mean():>10,.2f}  Std={mc_df['worst_dEVE'].std():>10,.2f}")
print(f"               VaR(95%)={var_worst_eve:>10,.2f}  ES(95%)={es_worst_eve:>10,.2f}")
print(f"  Worst dNII:  Mean={mc_df['worst_dNII'].mean():>10,.2f}  Std={mc_df['worst_dNII'].std():>10,.2f}")
print(f"               VaR(95%)={var_worst_nii:>10,.2f}  ES(95%)={es_worst_nii:>10,.2f}")

# Summary statistics table
print(f"\n  --- Per-Scenario dEVE Summary ---")
print(f"  {'Scenario':<22} {'Mean':>10} {'Std':>10} {'VaR(95%)':>10} {'ES(95%)':>10}")
print(f"  {'-'*62}")
for skey in ['dEVE_path'] + shock_keys:
    vals = mc_df[skey].values
    v, e = compute_var_es(vals)
    label = skey.replace('dEVE_', '').replace('_', ' ')
    print(f"  {label:<22} {np.mean(vals):>10,.2f} {np.std(vals):>10,.2f} {v:>10,.2f} {e:>10,.2f}")

print(f"\n  --- Per-Scenario dNII Summary ---")
print(f"  {'Scenario':<22} {'Mean':>10} {'Std':>10} {'VaR(95%)':>10} {'ES(95%)':>10}")
print(f"  {'-'*62}")
for skey in ['dNII_path'] + nii_shock_keys:
    vals = mc_df[skey].values
    v, e = compute_var_es(vals)
    label = skey.replace('dNII_', '').replace('_', ' ')
    print(f"  {label:<22} {np.mean(vals):>10,.2f} {np.std(vals):>10,.2f} {v:>10,.2f} {e:>10,.2f}")

# %%
# === Section 6 Charts: Shock Impact Visualization ===

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# --- Chart 1: dEVE distributions by scenario (overlapping histograms) ---
ax1 = axes[0, 0]
colors = {'path': 'steelblue', 'up200bps_Parallel': 'red',
          'dn200bps_Parallel': 'green', 'Short_Rate_Up': 'orange', 'Flattener': 'purple'}
labels = {'path': 'MC Only (no shock)', 'up200bps_Parallel': '+200bps',
          'dn200bps_Parallel': '-200bps', 'Short_Rate_Up': 'Short Rate Up', 'Flattener': 'Flattener'}

for skey in ['dEVE_path'] + shock_keys:
    key_short = skey.replace('dEVE_', '')
    ax1.hist(mc_df[skey], bins=50, alpha=0.35, color=colors.get(key_short, 'gray'),
             label=labels.get(key_short, key_short), edgecolor='none')
ax1.axvline(x=0, color='black', linestyle='--', linewidth=1)
ax1.set_xlabel('dEVE (change from static base)')
ax1.set_ylabel('Frequency')
ax1.set_title('dEVE Distribution: MC Paths vs Shock Scenarios')
ax1.legend(fontsize=8)

# --- Chart 2: dNII distributions by scenario ---
ax2 = axes[0, 1]
for skey in ['dNII_path'] + nii_shock_keys:
    key_short = skey.replace('dNII_', '')
    ax2.hist(mc_df[skey], bins=50, alpha=0.35, color=colors.get(key_short, 'gray'),
             label=labels.get(key_short, key_short), edgecolor='none')
ax2.axvline(x=0, color='black', linestyle='--', linewidth=1)
ax2.set_xlabel('dNII (change from static base)')
ax2.set_ylabel('Frequency')
ax2.set_title('dNII Distribution: MC Paths vs Shock Scenarios')
ax2.legend(fontsize=8)

# --- Chart 3: Balance at Y1 vs dEVE (scatter) - shows balance drives risk ---
ax3 = axes[1, 0]
sc = ax3.scatter(mc_df['balance_y1'], mc_df['dEVE_path'], c=mc_df['rate_y1'] * 100,
                 cmap='coolwarm', alpha=0.4, s=10, edgecolors='none')
ax3.axhline(y=0, color='black', linestyle='--', linewidth=0.8)
ax3.axvline(x=calc_balance, color='red', linestyle='--', linewidth=0.8, label=f'Current bal ({calc_balance:,.0f})')
ax3.set_xlabel('Balance at Year 1')
ax3.set_ylabel('dEVE (MC path, no shock)')
ax3.set_title('Balance vs dEVE (color = short rate at Y1)')
ax3.legend(fontsize=9)
plt.colorbar(sc, ax=ax3, label='Short Rate (%)')

# --- Chart 4: Worst dNII by scenario (box plot) ---
ax4 = axes[1, 1]
box_data = []
box_labels_list = []
for skey in ['dNII_path'] + nii_shock_keys:
    box_data.append(mc_df[skey].values)
    key_short = skey.replace('dNII_', '')
    box_labels_list.append(labels.get(key_short, key_short))
bp = ax4.boxplot(box_data, labels=box_labels_list, patch_artist=True)
box_colors = ['steelblue', 'red', 'green', 'orange', 'purple']
for patch, col in zip(bp['boxes'], box_colors):
    patch.set_facecolor(col)
    patch.set_alpha(0.4)
ax4.axhline(y=0, color='black', linestyle='--', linewidth=0.8)
ax4.set_ylabel('dNII')
ax4.set_title('dNII Distribution by Scenario (Box Plot)')
ax4.tick_params(axis='x', rotation=15)

plt.tight_layout()
plt.savefig('mc_section6_shocks.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: mc_section6_shocks.png")

# %%
# === Section 6 Chart 2: MC Paths — Balance + No Shock vs Shocked ===

fig, axes = plt.subplots(1, 3, figsize=(20, 6))

# Sort paths by balance for a clean visual
sort_idx = mc_df['balance_y1'].argsort().values
path_x = np.arange(n_paths)

# --- Chart 1: Balance at Y1 per path (sorted) ---
ax0 = axes[0]
bal_sorted = mc_df['balance_y1'].values[sort_idx]
ax0.plot(path_x, bal_sorted, color='steelblue', linewidth=0.8)
ax0.axhline(y=calc_balance, color='red', linestyle='--', linewidth=1.2, label=f'Current ({calc_balance:,.0f})')
ax0.axhline(y=np.percentile(bal_sorted, 5), color='gray', linestyle=':', linewidth=1, label=f'5th pctl ({np.percentile(bal_sorted, 5):,.0f})')
ax0.axhline(y=np.percentile(bal_sorted, 95), color='gray', linestyle=':', linewidth=1, label=f'95th pctl ({np.percentile(bal_sorted, 95):,.0f})')
ax0.fill_between(path_x, bal_sorted, calc_balance, where=bal_sorted < calc_balance,
                 alpha=0.3, color='salmon', label='Below current')
ax0.fill_between(path_x, bal_sorted, calc_balance, where=bal_sorted >= calc_balance,
                 alpha=0.3, color='lightgreen', label='Above current')
ax0.set_xlabel('Path (sorted by balance at Y1)')
ax0.set_ylabel('Balance at Year 1')
ax0.set_title(f'Balance at Y1 ({n_paths} paths, sorted)')
ax0.legend(fontsize=7, loc='upper left')

# --- Chart 2: EVE per path — no shock vs shocked ---
ax1 = axes[1]
eve_nosock = mc_df['eve_path'].values[sort_idx]
eve_up200 = mc_df['eve_up200bps_Parallel'].values[sort_idx]
eve_dn200 = mc_df['eve_dn200bps_Parallel'].values[sort_idx]

ax1.fill_between(path_x, eve_dn200, eve_up200, alpha=0.15, color='red', label='Range: -200 to +200bps')
ax1.plot(path_x, eve_nosock, color='steelblue', linewidth=0.5, alpha=0.8, label='MC Only (no shock)')
ax1.plot(path_x, eve_up200, color='red', linewidth=0.5, alpha=0.6, label='+200bps shock')
ax1.plot(path_x, eve_dn200, color='green', linewidth=0.5, alpha=0.6, label='-200bps shock')
ax1.axhline(y=eve_base, color='black', linestyle='--', linewidth=1, label=f'Static base ({eve_base:,.0f})')
ax1.set_xlabel('Path (sorted by balance at Y1)')
ax1.set_ylabel('EVE')
ax1.set_title(f'EVE: No Shock vs Shocked ({n_paths} paths)')
ax1.legend(fontsize=7, loc='upper left')

# --- Chart 3: NII per path — no shock vs shocked ---
ax2 = axes[2]
nii_nosock = mc_df['nii_path'].values[sort_idx]
nii_up200 = mc_df['nii_up200bps_Parallel'].values[sort_idx]
nii_dn200 = mc_df['nii_dn200bps_Parallel'].values[sort_idx]

ax2.fill_between(path_x, nii_dn200, nii_up200, alpha=0.15, color='orange', label='Range: -200 to +200bps')
ax2.plot(path_x, nii_nosock, color='steelblue', linewidth=0.5, alpha=0.8, label='MC Only (no shock)')
ax2.plot(path_x, nii_up200, color='red', linewidth=0.5, alpha=0.6, label='+200bps shock')
ax2.plot(path_x, nii_dn200, color='green', linewidth=0.5, alpha=0.6, label='-200bps shock')
ax2.axhline(y=nii_base, color='black', linestyle='--', linewidth=1, label=f'Static base ({nii_base:,.0f})')
ax2.set_xlabel('Path (sorted by balance at Y1)')
ax2.set_ylabel('NII')
ax2.set_title(f'NII: No Shock vs Shocked ({n_paths} paths)')
ax2.legend(fontsize=7, loc='upper left')

plt.tight_layout()
plt.savefig('mc_section6_paths_shocked.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: mc_section6_paths_shocked.png")

# %% [markdown]
# ## Section 7: Save Outputs & Final Summary
#
# Save all results to CSV for reporting:
# 1. MC path-level results (1,000 rows)
# 2. Risk metrics summary
# 3. Static sensitivity comparison (Section 4 vs MC)

# %%
print(f"\n{'='*70}")
print(f"SECTION 7: SAVE OUTPUTS & FINAL SUMMARY")
print(f"{'='*70}")

# --- 1. Save MC path-level results ---
mc_output_cols = ['path', 'balance_y1', 'rate_y1', 'rate_shift', 'core_ratio',
                  'eve_path', 'nii_path', 'dEVE_path', 'dNII_path',
                  'worst_dEVE', 'worst_dNII']
mc_df[mc_output_cols].to_csv('mc_v2_path_results.csv', index=False)
print(f"\n  Saved: mc_v2_path_results.csv ({len(mc_df)} paths)")

# --- 2. Build risk metrics summary ---
summary_rows = []

# Static base case (Section 4)
summary_rows.append({
    'Method': 'Static Base (Section 4)',
    'Scenario': 'Base',
    'EVE': eve_base,
    'NII': nii_base,
    'dEVE': 0,
    'dNII': 0,
})
for sname, delta in shock_eve.items():
    summary_rows.append({
        'Method': 'Static Stress (Section 4)',
        'Scenario': sname,
        'EVE': eve_base + delta,
        'NII': nii_base + nii_deltas.get(sname, 0),
        'dEVE': delta,
        'dNII': nii_deltas.get(sname, 0),
    })

# MC path risk (no shock)
summary_rows.append({
    'Method': 'MC Path (no shock)',
    'Scenario': 'Mean',
    'EVE': mc_df['eve_path'].mean(),
    'NII': mc_df['nii_path'].mean(),
    'dEVE': mc_df['dEVE_path'].mean(),
    'dNII': mc_df['dNII_path'].mean(),
})
summary_rows.append({
    'Method': 'MC Path (no shock)',
    'Scenario': 'VaR(95%)',
    'EVE': np.nan,
    'NII': np.nan,
    'dEVE': var_eve,
    'dNII': var_nii,
})
summary_rows.append({
    'Method': 'MC Path (no shock)',
    'Scenario': 'ES(95%)',
    'EVE': np.nan,
    'NII': np.nan,
    'dEVE': es_eve,
    'dNII': es_nii,
})

# MC worst-case shock
summary_rows.append({
    'Method': 'MC Worst Shock',
    'Scenario': 'Mean',
    'EVE': np.nan,
    'NII': np.nan,
    'dEVE': mc_df['worst_dEVE'].mean(),
    'dNII': mc_df['worst_dNII'].mean(),
})
summary_rows.append({
    'Method': 'MC Worst Shock',
    'Scenario': 'VaR(95%)',
    'EVE': np.nan,
    'NII': np.nan,
    'dEVE': var_worst_eve,
    'dNII': var_worst_nii,
})
summary_rows.append({
    'Method': 'MC Worst Shock',
    'Scenario': 'ES(95%)',
    'EVE': np.nan,
    'NII': np.nan,
    'dEVE': es_worst_eve,
    'dNII': es_worst_nii,
})

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv('mc_v2_risk_summary.csv', index=False)
print(f"  Saved: mc_v2_risk_summary.csv ({len(summary_df)} rows)")

# --- 3. Save simulation parameters ---
params = {
    'Parameter': [
        'Calculation Date', 'Balance', 'Core Ratio', 'Non-Core Ratio',
        'Lambda Daily (gross outflow)', 'Sigma Decay',
        'Mu Net Daily', 'Sigma Net Daily',
        'Avg Monthly Outflow', 'N Paths', 'Horizon (days)',
        'Vasicek r0', 'Vasicek kappa', 'Vasicek theta', 'Vasicek sigma_r',
        'Core Ratio Sigma', 'Random Seed',
    ],
    'Value': [
        str(calc_date.date()), calc_balance, core_pct, non_core_pct,
        lambda_daily, sigma_decay,
        mu_net, sigma_net,
        avg_monthly_outflow, n_paths, n_days_sim,
        r0, kappa_v, theta_v, sigma_r,
        sigma_core, seed,
    ]
}
pd.DataFrame(params).to_csv('mc_v2_parameters.csv', index=False)
print(f"  Saved: mc_v2_parameters.csv")

# %%
# --- Final Summary Print ---
print(f"\n{'='*70}")
print(f"FINAL COMPARISON: Static vs Monte Carlo")
print(f"{'='*70}")
print(f"\n  {'Metric':<30} {'Static (S4)':>14} {'MC Mean':>14} {'MC VaR(95%)':>14} {'MC ES(95%)':>14}")
print(f"  {'-'*72}")
print(f"  {'dEVE (no shock)':<30} {'N/A':>14} {mc_df['dEVE_path'].mean():>14,.2f} {var_eve:>14,.2f} {es_eve:>14,.2f}")
print(f"  {'dEVE (+200bps)':<30} {shock_eve.get('+200bps Parallel',0):>14,.2f} {mc_df['dEVE_up200bps_Parallel'].mean():>14,.2f} {compute_var_es(mc_df['dEVE_up200bps_Parallel'].values)[0]:>14,.2f} {compute_var_es(mc_df['dEVE_up200bps_Parallel'].values)[1]:>14,.2f}")
print(f"  {'dEVE (worst shock)':<30} {min(shock_eve.values()):>14,.2f} {mc_df['worst_dEVE'].mean():>14,.2f} {var_worst_eve:>14,.2f} {es_worst_eve:>14,.2f}")
print(f"  {'-'*72}")
print(f"  {'dNII (no shock)':<30} {'N/A':>14} {mc_df['dNII_path'].mean():>14,.2f} {var_nii:>14,.2f} {es_nii:>14,.2f}")
print(f"  {'dNII (-200bps)':<30} {nii_deltas.get('-200bps Parallel',0):>14,.2f} {mc_df['dNII_dn200bps_Parallel'].mean():>14,.2f} {compute_var_es(mc_df['dNII_dn200bps_Parallel'].values)[0]:>14,.2f} {compute_var_es(mc_df['dNII_dn200bps_Parallel'].values)[1]:>14,.2f}")
print(f"  {'dNII (worst shock)':<30} {min(shock_nii.values()):>14,.2f} {mc_df['worst_dNII'].mean():>14,.2f} {var_worst_nii:>14,.2f} {es_worst_nii:>14,.2f}")

print(f"\n  Key Insights:")
print(f"  - Static stress test gives a SINGLE number (e.g. dEVE = {min(shock_eve.values()):,.2f})")
print(f"  - MC gives a DISTRIBUTION: VaR(95%) = {var_worst_eve:,.2f}, ES(95%) = {es_worst_eve:,.2f}")
print(f"  - Balance uncertainty (VaR dEVE = {var_eve:,.2f}) >> rate shock effect ({min(shock_eve.values()):,.2f})")
print(f"  - Rate shocks mainly impact NII, not EVE for this short-duration NMD portfolio")

print(f"\n{'='*70}")
print(f"PHASE 5d (v2) COMPLETE")
print(f"{'='*70}")
print(f"\nOutput files:")
print(f"  mc_v2_path_results.csv  - Per-path MC results (1,000 rows)")
print(f"  mc_v2_risk_summary.csv  - Risk metrics comparison table")
print(f"  mc_v2_parameters.csv    - Simulation parameters")
print(f"  mc_section5_paths.png   - Balance & rate path fan charts")
print(f"  mc_section6_shocks.png  - Shock impact visualizations")

# %%
