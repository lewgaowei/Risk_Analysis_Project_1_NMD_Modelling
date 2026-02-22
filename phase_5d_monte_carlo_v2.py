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
# ## Section 3: Cash Flow Slotting (11 IRRBB Buckets)
#
# Uses Phase 1b Portfolio KM survival curve and Phase 2 bucket definitions:
# - Non-core -> O/N bucket (immediate repricing)
# - Core -> distributed across 1M-5Y using Portfolio KM S(t)
# - Residual beyond 5Y (Basel cap) added to 5Y bucket

# %%
# Load Portfolio KM survival curve from Phase 1b
surv_df = pd.read_csv('survival_curve_full_advanced.csv')
survival_interp = interp1d(surv_df['Days'].values, surv_df['S(t)'].values,
                           kind='linear', bounds_error=False,
                           fill_value=(1.0, surv_df['S(t)'].values[-1]))

print(f"\nSurvival Curve (Portfolio KM) loaded:")
print(f"  Days range: 0 to {surv_df['Days'].max()}")
print(f"  S(0)   = {float(survival_interp(0)):.4f}")
print(f"  S(365) = {float(survival_interp(365)):.4f}")
print(f"  S(1825)= {float(survival_interp(1825)):.4f}")

# %%
# 11 IRRBB time buckets (in days) matching Phase 2 definitions
buckets = [
    {'name': 'O/N',  'start_days': 0,    'end_days': 1,    'midpoint_yrs': 1/365},
    {'name': '1M',   'start_days': 1,    'end_days': 30,   'midpoint_yrs': 0.0417},
    {'name': '2M',   'start_days': 30,   'end_days': 60,   'midpoint_yrs': 0.125},
    {'name': '3M',   'start_days': 60,   'end_days': 90,   'midpoint_yrs': 0.2083},
    {'name': '6M',   'start_days': 90,   'end_days': 180,  'midpoint_yrs': 0.375},
    {'name': '9M',   'start_days': 180,  'end_days': 270,  'midpoint_yrs': 0.625},
    {'name': '1Y',   'start_days': 270,  'end_days': 365,  'midpoint_yrs': 0.875},
    {'name': '2Y',   'start_days': 365,  'end_days': 730,  'midpoint_yrs': 1.5},
    {'name': '3Y',   'start_days': 730,  'end_days': 1095, 'midpoint_yrs': 2.5},
    {'name': '4Y',   'start_days': 1095, 'end_days': 1460, 'midpoint_yrs': 3.5},
    {'name': '5Y',   'start_days': 1460, 'end_days': 1825, 'midpoint_yrs': 4.5},
]


def slot_cashflows(balance, core_ratio, surv_interp):
    """
    Slot NMD balance into 11 IRRBB buckets using Portfolio KM survival curve.

    Non-core -> O/N.
    Core -> distributed across 1M-5Y using S(t) from survival curve.
    Residual beyond 5Y (1825 days) -> capped at 5Y bucket.

    Returns dict of allocations and dict of midpoint years.
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

        s_start = float(surv_interp(b['start_days']))
        s_end = float(surv_interp(b['end_days']))
        allocations[b['name']] = core_amt * (s_start - s_end)

    # Basel 5Y cap: residual surviving beyond 1825 days -> add to 5Y bucket
    s_5y = float(surv_interp(1825))
    allocations['5Y'] += core_amt * s_5y

    return allocations, midpoints


# Slot the current balance
alloc_base, midpts = slot_cashflows(calc_balance, core_pct, survival_interp)

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
# Uses actual yield curve from processed_curve_data.csv (Phase 1a).
# 4 shock scenarios matching Phase 3/4 definitions:
# - EVE = sum(CF * DF) where DF = 1/(1+r)^t
# - NII = sum(CF_i * r_i * (1 - t_i)) for buckets with t_i <= 1Y (Phase 4)

# %%
# === Base Interest Rate Curve (from Phase 1a processed_curve_data.csv) ===
curve_df = pd.read_csv('processed_curve_data.csv')
tenor_years = curve_df['Tenor_Years'].values
tenor_labels = curve_df['Tenor'].values.tolist()
base_curve = curve_df['ZeroRate'].values

# === 4 Shock Scenarios (matching Phase 3/4) ===
# S1: +200bps parallel
shock_up = base_curve + 0.0200

# S2: -200bps parallel (floored at 0)
shock_down = np.maximum(base_curve - 0.0200, 0.0)

# S3: Steepener — short rate up (+200bps tapering to 0 at 5Y)
taper_weights = np.maximum(1 - tenor_years / 5.0, 0)
shock_short_up = base_curve + 0.0200 * taper_weights

# S4: Flattener — piecewise: +200bps at t=0, 0 at 2Y pivot, -100bps at 5Y
flatten_shocks = np.where(
    tenor_years <= 2,
    0.0200 * (1 - tenor_years / 2),
    -0.0100 * (tenor_years - 2) / (5 - 2)
)
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
    NII = sum(CF_i * r_i * (1 - t_i)) for buckets with t_i <= 1Y.
    Matches Phase 4 IRRBB methodology: only buckets repricing within 1Y,
    time-weighted by remaining fraction of the year.
    """
    interp_func = interp1d(tenor_yrs_arr, rate_curve_arr, kind='linear',
                           bounds_error=False, fill_value='extrapolate')
    nii = 0.0
    for bname in allocations:
        cf = allocations[bname]
        t = midpoints_dict[bname]
        if t > 1.0:  # Only buckets repricing within 1Y contribute to NII
            continue
        r = max(float(interp_func(t)), 0)
        nii += cf * r * (1 - t)  # time-weighted
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
#
# **2. Interest rate paths (Vasicek model):**
# - dr = kappa * (theta - r) * dt + sigma_r * sqrt(dt) * dW
# - Mean-reverts to theta (long-run rate), with speed kappa and volatility sigma_r
#
# At Year 1, for each path we:
# - Take the simulated balance
# - Perturb core ratio (stochastic uncertainty around Phase 1c estimate)
# - Slot into 11 buckets using Portfolio KM survival curve (Phase 1b/2)
# - Shift the actual yield curve by the Vasicek rate change
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
print(f"  Slotting: Portfolio KM survival curve (Phase 1b)")
print(f"    S(1Y) = {float(survival_interp(365)):.4f},  S(5Y) = {float(survival_interp(1825)):.4f}")
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

n_days_rate = 365 * 5  # simulate rates for 5Y (multi-year EVE/NII evaluation)
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

    # Rate path: 5 years (Vasicek)
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

# Rate distribution per year
print(f"\nShort Rate Distribution:")
print(f"  {'Year':>4}  {'5th pctl':>10}  {'Median':>10}  {'95th pctl':>10}  {'Mean':>10}  {'Std':>10}")
for yr in range(1, 6):
    r_yr = rate_paths[:, yr * 365]
    print(f"  {yr:>4}  {np.percentile(r_yr, 5):>10.3%}  {np.median(r_yr):>10.3%}  "
          f"{np.percentile(r_yr, 95):>10.3%}  {np.mean(r_yr):>10.3%}  {np.std(r_yr):>10.3%}")

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

# --- Chart 2: Interest rate paths (5Y) ---
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
ax2.set_title(f'Vasicek Short Rate Paths ({n_paths} paths, 5Y)')
ax2.legend(loc='upper right', fontsize=9)
ax2.set_xlim(0, 5)

plt.tight_layout()
plt.savefig('mc_section5_paths.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: mc_section5_paths.png")

# %% [markdown]
# ## Section 6: Multi-Year EVE/NII per MC Path & Risk Metrics
#
# For each of the 1,000 paths at Years 1, 2, 3, 4, 5:
# 1. Take simulated balance and short rate at that year
# 2. Perturb core ratio (one draw per path, persistent across years)
# 3. Slot into 11 buckets using Portfolio KM survival curve
# 4. Shift actual yield curve by simulated rate change (parallel shift)
# 5. Compute EVE and NII under the path's curve AND under 4 shock scenarios
# 6. dEVE = EVE(path) - EVE(static base), dNII = NII(path) - NII(static base)
#
# Risk metrics: VaR and ES at 95% confidence, computed per evaluation year

# %%
eval_years = [1, 2, 3, 4, 5]

print(f"\n{'='*70}")
print(f"SECTION 6: MULTI-YEAR EVE/NII PER MC PATH & RISK METRICS")
print(f"{'='*70}")
print(f"Processing {n_paths} paths x {len(eval_years)} years...")

# Storage for per-path, per-year results
mc_results = []

for i in range(n_paths):
    # Core ratio perturbation: ONE per path (structural estimate uncertainty)
    core_ratio_i = np.clip(core_pct + np.random.normal(0, sigma_core), 0.50, 0.99)

    for yr in eval_years:
        day = yr * 365

        # 1. Balance and short rate at this year
        bal = balance_paths[i, day]
        r = rate_paths[i, day]
        rate_shift = r - r0

        # 2. Slot into 11 buckets
        alloc_i, midpts_i = slot_cashflows(bal, core_ratio_i, survival_interp)

        # 3. Build this path's yield curve: base + parallel shift from Vasicek
        curve_i = np.maximum(base_curve + rate_shift, 0)

        # 4. Compute EVE and NII under this path's curve
        eve_i = compute_eve(alloc_i, midpts_i, curve_i, tenor_years)
        nii_i = compute_nii(alloc_i, midpts_i, curve_i, tenor_years)

        path_row = {
            'path': i,
            'year': yr,
            'balance': bal,
            'rate': r,
            'rate_shift': rate_shift,
            'core_ratio': core_ratio_i,
            'eve_path': eve_i,
            'nii_path': nii_i,
            'dEVE_path': eve_i - eve_base,
            'dNII_path': nii_i - nii_base,
        }

        # 5. Apply each shock scenario on top of the path's curve
        for sname, shock_curve in scenarios.items():
            if sname == 'Base':
                continue
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
print(f"Done. {len(mc_df)} records ({n_paths} paths x {len(eval_years)} years).")

# %%
# === Risk Metrics: VaR and ES at 95% confidence, per year ===

def compute_var_es(losses, confidence=0.95):
    """VaR = percentile of losses, ES = mean of losses beyond VaR."""
    var = np.percentile(losses, (1 - confidence) * 100)
    es = losses[losses <= var].mean() if (losses <= var).any() else var
    return var, es

# Add worst-case columns
shock_keys = [c for c in mc_df.columns if c.startswith('dEVE_') and c != 'dEVE_path']
nii_shock_keys = [c for c in mc_df.columns if c.startswith('dNII_') and c != 'dNII_path']
mc_df['worst_dEVE'] = mc_df[shock_keys].min(axis=1)
mc_df['worst_dNII'] = mc_df[nii_shock_keys].min(axis=1)

# Collect risk evolution data for charts
risk_evolution = []

print(f"\n{'='*70}")
print(f"RISK METRICS BY YEAR (95% Confidence Level)")
print(f"{'='*70}")

for yr in eval_years:
    yr_df = mc_df[mc_df['year'] == yr]

    var_eve, es_eve = compute_var_es(yr_df['dEVE_path'].values)
    var_nii, es_nii = compute_var_es(yr_df['dNII_path'].values)
    var_weve, es_weve = compute_var_es(yr_df['worst_dEVE'].values)
    var_wnii, es_wnii = compute_var_es(yr_df['worst_dNII'].values)

    bal_5pctl = yr_df['balance'].quantile(0.05)
    bal_median = yr_df['balance'].median()

    print(f"\n  --- Year {yr} (balance median: {bal_median:,.0f}, "
          f"balance 5th pctl: {bal_5pctl:,.0f}, rate median: {yr_df['rate'].median():.3%}) ---")
    print(f"  dEVE path:   Mean={yr_df['dEVE_path'].mean():>10,.2f}  "
          f"VaR={var_eve:>10,.2f}  ES={es_eve:>10,.2f}")
    print(f"  dNII path:   Mean={yr_df['dNII_path'].mean():>10,.2f}  "
          f"VaR={var_nii:>10,.2f}  ES={es_nii:>10,.2f}")
    print(f"  Worst dEVE:  Mean={yr_df['worst_dEVE'].mean():>10,.2f}  "
          f"VaR={var_weve:>10,.2f}  ES={es_weve:>10,.2f}")
    print(f"  Worst dNII:  Mean={yr_df['worst_dNII'].mean():>10,.2f}  "
          f"VaR={var_wnii:>10,.2f}  ES={es_wnii:>10,.2f}")

    risk_evolution.append({
        'year': yr,
        'bal_median': bal_median,
        'bal_5pctl': bal_5pctl,
        'rate_median': yr_df['rate'].median(),
        'dEVE_mean': yr_df['dEVE_path'].mean(),
        'dEVE_var': var_eve, 'dEVE_es': es_eve,
        'dNII_mean': yr_df['dNII_path'].mean(),
        'dNII_var': var_nii, 'dNII_es': es_nii,
        'worst_dEVE_mean': yr_df['worst_dEVE'].mean(),
        'worst_dEVE_var': var_weve, 'worst_dEVE_es': es_weve,
        'worst_dNII_mean': yr_df['worst_dNII'].mean(),
        'worst_dNII_var': var_wnii, 'worst_dNII_es': es_wnii,
    })

risk_evo_df = pd.DataFrame(risk_evolution)

# Per-scenario detail for Year 1
print(f"\n{'='*70}")
print(f"YEAR 1 DETAILED SCENARIO BREAKDOWN")
print(f"{'='*70}")
mc_df_y1 = mc_df[mc_df['year'] == 1].copy().reset_index(drop=True)

print(f"\n  --- Per-Scenario dEVE (Year 1) ---")
print(f"  {'Scenario':<22} {'Mean':>10} {'Std':>10} {'VaR(95%)':>10} {'ES(95%)':>10}")
print(f"  {'-'*62}")
for skey in ['dEVE_path'] + shock_keys:
    vals = mc_df_y1[skey].values
    v, e = compute_var_es(vals)
    label = skey.replace('dEVE_', '').replace('_', ' ')
    print(f"  {label:<22} {np.mean(vals):>10,.2f} {np.std(vals):>10,.2f} {v:>10,.2f} {e:>10,.2f}")

print(f"\n  --- Per-Scenario dNII (Year 1) ---")
print(f"  {'Scenario':<22} {'Mean':>10} {'Std':>10} {'VaR(95%)':>10} {'ES(95%)':>10}")
print(f"  {'-'*62}")
for skey in ['dNII_path'] + nii_shock_keys:
    vals = mc_df_y1[skey].values
    v, e = compute_var_es(vals)
    label = skey.replace('dNII_', '').replace('_', ' ')
    print(f"  {label:<22} {np.mean(vals):>10,.2f} {np.std(vals):>10,.2f} {v:>10,.2f} {e:>10,.2f}")

# %%
# === Section 6 Chart 1: Multi-Year Risk Evolution ===

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
yrs = risk_evo_df['year'].values

# --- EVE Risk Evolution ---
ax1 = axes[0]
ax1.fill_between(yrs, risk_evo_df['dEVE_es'], risk_evo_df['dEVE_mean'],
                 alpha=0.15, color='steelblue')
ax1.plot(yrs, risk_evo_df['dEVE_mean'], 'o-', color='steelblue', linewidth=2, label='Mean dEVE')
ax1.plot(yrs, risk_evo_df['dEVE_var'], 's-', color='darkorange', linewidth=2, label='VaR(95%)')
ax1.plot(yrs, risk_evo_df['dEVE_es'], '^-', color='red', linewidth=2, label='ES(95%)')
ax1.plot(yrs, risk_evo_df['worst_dEVE_var'], 's--', color='darkred', linewidth=1.5, label='Worst Shock VaR(95%)')
ax1.plot(yrs, risk_evo_df['worst_dEVE_es'], '^--', color='maroon', linewidth=1.5, label='Worst Shock ES(95%)')
ax1.axhline(y=0, color='black', linestyle=':', linewidth=0.8)
ax1.axhline(y=min(shock_eve.values()), color='gray', linestyle='--', linewidth=1,
            label=f'Static worst ({min(shock_eve.values()):,.0f})')
ax1.set_xlabel('Evaluation Year')
ax1.set_ylabel('dEVE')
ax1.set_title('EVE Risk Evolution Over 5 Years')
ax1.legend(fontsize=7, loc='lower left')
ax1.set_xticks([1, 2, 3, 4, 5])

# --- NII Risk Evolution ---
ax2 = axes[1]
ax2.fill_between(yrs, risk_evo_df['dNII_es'], risk_evo_df['dNII_mean'],
                 alpha=0.15, color='steelblue')
ax2.plot(yrs, risk_evo_df['dNII_mean'], 'o-', color='steelblue', linewidth=2, label='Mean dNII')
ax2.plot(yrs, risk_evo_df['dNII_var'], 's-', color='darkorange', linewidth=2, label='VaR(95%)')
ax2.plot(yrs, risk_evo_df['dNII_es'], '^-', color='red', linewidth=2, label='ES(95%)')
ax2.plot(yrs, risk_evo_df['worst_dNII_var'], 's--', color='darkred', linewidth=1.5, label='Worst Shock VaR(95%)')
ax2.plot(yrs, risk_evo_df['worst_dNII_es'], '^--', color='maroon', linewidth=1.5, label='Worst Shock ES(95%)')
ax2.axhline(y=0, color='black', linestyle=':', linewidth=0.8)
ax2.axhline(y=min(shock_nii.values()), color='gray', linestyle='--', linewidth=1,
            label=f'Static worst ({min(shock_nii.values()):,.0f})')
ax2.set_xlabel('Evaluation Year')
ax2.set_ylabel('dNII')
ax2.set_title('NII Risk Evolution Over 5 Years')
ax2.legend(fontsize=7, loc='lower left')
ax2.set_xticks([1, 2, 3, 4, 5])

plt.tight_layout()
plt.savefig('mc_section6_risk_evolution.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: mc_section6_risk_evolution.png")

# %%
# === Section 6 Chart 2: Year 1 Shock Impact (histograms + scatter + box) ===

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

colors = {'path': 'steelblue', 'up200bps_Parallel': 'red',
          'dn200bps_Parallel': 'green', 'Short_Rate_Up': 'orange', 'Flattener': 'purple'}
labels_map = {'path': 'MC Only (no shock)', 'up200bps_Parallel': '+200bps',
              'dn200bps_Parallel': '-200bps', 'Short_Rate_Up': 'Short Rate Up', 'Flattener': 'Flattener'}

# --- Chart 1: dEVE distributions (Year 1) ---
ax1 = axes[0, 0]
for skey in ['dEVE_path'] + shock_keys:
    key_short = skey.replace('dEVE_', '')
    ax1.hist(mc_df_y1[skey], bins=50, alpha=0.35, color=colors.get(key_short, 'gray'),
             label=labels_map.get(key_short, key_short), edgecolor='none')
ax1.axvline(x=0, color='black', linestyle='--', linewidth=1)
ax1.set_xlabel('dEVE (change from static base)')
ax1.set_ylabel('Frequency')
ax1.set_title('dEVE Distribution at Year 1')
ax1.legend(fontsize=8)

# --- Chart 2: dNII distributions (Year 1) ---
ax2 = axes[0, 1]
for skey in ['dNII_path'] + nii_shock_keys:
    key_short = skey.replace('dNII_', '')
    ax2.hist(mc_df_y1[skey], bins=50, alpha=0.35, color=colors.get(key_short, 'gray'),
             label=labels_map.get(key_short, key_short), edgecolor='none')
ax2.axvline(x=0, color='black', linestyle='--', linewidth=1)
ax2.set_xlabel('dNII (change from static base)')
ax2.set_ylabel('Frequency')
ax2.set_title('dNII Distribution at Year 1')
ax2.legend(fontsize=8)

# --- Chart 3: Balance vs dEVE scatter (Year 1) ---
ax3 = axes[1, 0]
sc = ax3.scatter(mc_df_y1['balance'], mc_df_y1['dEVE_path'], c=mc_df_y1['rate'] * 100,
                 cmap='coolwarm', alpha=0.4, s=10, edgecolors='none')
ax3.axhline(y=0, color='black', linestyle='--', linewidth=0.8)
ax3.axvline(x=calc_balance, color='red', linestyle='--', linewidth=0.8, label=f'Current bal ({calc_balance:,.0f})')
ax3.set_xlabel('Balance at Year 1')
ax3.set_ylabel('dEVE (MC path, no shock)')
ax3.set_title('Balance vs dEVE at Year 1 (color = short rate)')
ax3.legend(fontsize=9)
plt.colorbar(sc, ax=ax3, label='Short Rate (%)')

# --- Chart 4: dNII box plot by scenario (Year 1) ---
ax4 = axes[1, 1]
box_data = []
box_labels_list = []
for skey in ['dNII_path'] + nii_shock_keys:
    box_data.append(mc_df_y1[skey].values)
    key_short = skey.replace('dNII_', '')
    box_labels_list.append(labels_map.get(key_short, key_short))
bp = ax4.boxplot(box_data, labels=box_labels_list, patch_artist=True)
box_colors = ['steelblue', 'red', 'green', 'orange', 'purple']
for patch, col in zip(bp['boxes'], box_colors):
    patch.set_facecolor(col)
    patch.set_alpha(0.4)
ax4.axhline(y=0, color='black', linestyle='--', linewidth=0.8)
ax4.set_ylabel('dNII')
ax4.set_title('dNII by Scenario at Year 1 (Box Plot)')
ax4.tick_params(axis='x', rotation=15)

plt.tight_layout()
plt.savefig('mc_section6_shocks_y1.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: mc_section6_shocks_y1.png")

# %%
# === Section 6 Chart 3: dEVE Box Plots Across Years (multi-year comparison) ===

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# --- dEVE worst shock box plots per year ---
ax1 = axes[0]
box_data_eve = [mc_df[mc_df['year'] == yr]['worst_dEVE'].values for yr in eval_years]
bp1 = ax1.boxplot(box_data_eve, labels=[f'Y{yr}' for yr in eval_years], patch_artist=True)
yr_colors = ['#2196F3', '#4CAF50', '#FF9800', '#F44336', '#9C27B0']
for patch, col in zip(bp1['boxes'], yr_colors):
    patch.set_facecolor(col)
    patch.set_alpha(0.5)
ax1.axhline(y=0, color='black', linestyle=':', linewidth=0.8)
ax1.axhline(y=min(shock_eve.values()), color='gray', linestyle='--', linewidth=1,
            label=f'Static worst ({min(shock_eve.values()):,.0f})')
ax1.set_xlabel('Evaluation Year')
ax1.set_ylabel('Worst dEVE (across 4 shocks)')
ax1.set_title('Worst-Case dEVE Distribution by Year')
ax1.legend(fontsize=9)

# --- dNII worst shock box plots per year ---
ax2 = axes[1]
box_data_nii = [mc_df[mc_df['year'] == yr]['worst_dNII'].values for yr in eval_years]
bp2 = ax2.boxplot(box_data_nii, labels=[f'Y{yr}' for yr in eval_years], patch_artist=True)
for patch, col in zip(bp2['boxes'], yr_colors):
    patch.set_facecolor(col)
    patch.set_alpha(0.5)
ax2.axhline(y=0, color='black', linestyle=':', linewidth=0.8)
ax2.axhline(y=min(shock_nii.values()), color='gray', linestyle='--', linewidth=1,
            label=f'Static worst ({min(shock_nii.values()):,.0f})')
ax2.set_xlabel('Evaluation Year')
ax2.set_ylabel('Worst dNII (across 4 shocks)')
ax2.set_title('Worst-Case dNII Distribution by Year')
ax2.legend(fontsize=9)

plt.tight_layout()
plt.savefig('mc_section6_multiyear_box.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: mc_section6_multiyear_box.png")

# %% [markdown]
# ## Section 7: Save Outputs & Final Summary
#
# Save all results to CSV for reporting:
# 1. MC path-level results (1,000 paths x 5 years)
# 2. Risk evolution summary (per year)
# 3. Static sensitivity comparison (Section 4 vs MC Year 1)

# %%
print(f"\n{'='*70}")
print(f"SECTION 7: SAVE OUTPUTS & FINAL SUMMARY")
print(f"{'='*70}")

# --- 1. Save MC path-level results ---
mc_output_cols = ['path', 'year', 'balance', 'rate', 'rate_shift', 'core_ratio',
                  'eve_path', 'nii_path', 'dEVE_path', 'dNII_path',
                  'worst_dEVE', 'worst_dNII']
mc_df[mc_output_cols].to_csv('mc_v2_path_results.csv', index=False)
print(f"\n  Saved: mc_v2_path_results.csv ({len(mc_df)} records)")

# --- 2. Save risk evolution per year ---
risk_evo_df.to_csv('mc_v2_risk_evolution.csv', index=False)
print(f"  Saved: mc_v2_risk_evolution.csv ({len(risk_evo_df)} years)")

# --- 3. Build risk metrics summary ---
summary_rows = []

# Static base case (Section 4)
summary_rows.append({
    'Method': 'Static Base (Section 4)',
    'Scenario': 'Base',
    'Year': '-',
    'EVE': eve_base,
    'NII': nii_base,
    'dEVE': 0,
    'dNII': 0,
})
for sname, delta in shock_eve.items():
    summary_rows.append({
        'Method': 'Static Stress (Section 4)',
        'Scenario': sname,
        'Year': '-',
        'EVE': eve_base + delta,
        'NII': nii_base + nii_deltas.get(sname, 0),
        'dEVE': delta,
        'dNII': nii_deltas.get(sname, 0),
    })

# MC per-year summary
for yr in eval_years:
    yr_df = mc_df[mc_df['year'] == yr]
    v_eve, e_eve = compute_var_es(yr_df['dEVE_path'].values)
    v_nii, e_nii = compute_var_es(yr_df['dNII_path'].values)
    v_weve, e_weve = compute_var_es(yr_df['worst_dEVE'].values)
    v_wnii, e_wnii = compute_var_es(yr_df['worst_dNII'].values)

    for method, scenario, deve, dnii in [
        ('MC Path (no shock)', 'Mean', yr_df['dEVE_path'].mean(), yr_df['dNII_path'].mean()),
        ('MC Path (no shock)', 'VaR(95%)', v_eve, v_nii),
        ('MC Path (no shock)', 'ES(95%)', e_eve, e_nii),
        ('MC Worst Shock', 'Mean', yr_df['worst_dEVE'].mean(), yr_df['worst_dNII'].mean()),
        ('MC Worst Shock', 'VaR(95%)', v_weve, v_wnii),
        ('MC Worst Shock', 'ES(95%)', e_weve, e_wnii),
    ]:
        summary_rows.append({
            'Method': method,
            'Scenario': scenario,
            'Year': yr,
            'EVE': np.nan,
            'NII': np.nan,
            'dEVE': deve,
            'dNII': dnii,
        })

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv('mc_v2_risk_summary.csv', index=False)
print(f"  Saved: mc_v2_risk_summary.csv ({len(summary_df)} rows)")

# --- 4. Save simulation parameters ---
params = {
    'Parameter': [
        'Calculation Date', 'Balance', 'Core Ratio', 'Non-Core Ratio',
        'Core Method', 'Survival Model',
        'Mu Net Daily', 'Sigma Net Daily',
        'N Paths', 'Horizon (days)', 'Evaluation Years',
        'Vasicek r0', 'Vasicek kappa', 'Vasicek theta', 'Vasicek sigma_r',
        'Core Ratio Sigma', 'Random Seed',
    ],
    'Value': [
        str(calc_date.date()), calc_balance, core_pct, non_core_pct,
        config['method'], 'Portfolio KM (Phase 1b)',
        mu_net, sigma_net,
        n_paths, n_days_sim, '1,2,3,4,5',
        r0, kappa_v, theta_v, sigma_r,
        sigma_core, seed,
    ]
}
pd.DataFrame(params).to_csv('mc_v2_parameters.csv', index=False)
print(f"  Saved: mc_v2_parameters.csv")

# %%
# --- Section 7: Final Summary & Presentation Analysis ---
print(f"\n{'='*80}")
print(f"SECTION 7: FINAL COMPARISON — Static vs Monte Carlo (All Years)")
print(f"{'='*80}")

# Static baseline
print(f"\n  STATIC STRESS TEST (Phase 3/4 — point-in-time):")
print(f"    Worst dEVE: {min(shock_eve.values()):>10,.2f}  (scenario: {min(shock_eve, key=shock_eve.get)})")
print(f"    Worst dNII: {min(shock_nii.values()):>10,.2f}  (scenario: {min(shock_nii, key=shock_nii.get)})")

# Multi-year MC comparison with balance context
print(f"\n  MONTE CARLO RISK METRICS ({n_paths:,} paths, 95% confidence):")
print(f"  {'':>4}  {'-- Balance --':^20}  {'--- dEVE (Economic Value) ---':^42}  {'--- dNII (Net Interest Income) ---':^42}")
print(f"  {'Year':>4}  {'Median':>10}  {'5th Pctl':>10}  {'Mean':>10}  {'VaR(95%)':>10}  {'ES(95%)':>10}  {'Worst VaR':>10}  {'Mean':>10}  {'VaR(95%)':>10}  {'ES(95%)':>10}  {'Worst VaR':>10}")
print(f"  {'-'*116}")
for _, row in risk_evo_df.iterrows():
    yr = int(row['year'])
    print(f"  {yr:>4}  {row['bal_median']:>10,.0f}  {row['bal_5pctl']:>10,.0f}  "
          f"{row['dEVE_mean']:>10,.0f}  {row['dEVE_var']:>10,.0f}  {row['dEVE_es']:>10,.0f}  "
          f"{row['worst_dEVE_var']:>10,.0f}  "
          f"{row['dNII_mean']:>10,.1f}  {row['dNII_var']:>10,.1f}  {row['dNII_es']:>10,.1f}  "
          f"{row['worst_dNII_var']:>10,.1f}")
print(f"  {'-'*116}")
print(f"  Static: {calc_balance:>10,.0f}  {'':>10}  "
      f"{'':>10}  {min(shock_eve.values()):>10,.0f}  {'':>10}  {'':>10}  "
      f"{'':>10}  {min(shock_nii.values()):>10,.1f}")

# ---- KEY FINDINGS ----
static_worst_eve = min(shock_eve.values())
y1_worst_var = risk_evo_df.iloc[0]['worst_dEVE_var']
y1_worst_es = risk_evo_df.iloc[0]['worst_dEVE_es']
y1_path_var = risk_evo_df.iloc[0]['dEVE_var']
# Find the year with worst ES
worst_es_idx = risk_evo_df['worst_dEVE_es'].idxmin()
worst_es_yr = int(risk_evo_df.iloc[worst_es_idx]['year'])
worst_es_val = risk_evo_df.iloc[worst_es_idx]['worst_dEVE_es']
pct_from_paths = abs(y1_path_var) / abs(y1_worst_var) * 100

print(f"\n{'='*80}")
print(f"KEY FINDINGS — PRESENTATION ANALYSIS")
print(f"{'='*80}")

print(f"""
  NOTE ON BALANCE MEDIAN vs NEGATIVE dEVE:
  ─────────────────────────────────────────
  The balance median shows the TYPICAL (50th percentile) path — most paths
  see balance growth due to positive net flow drift.

  The negative dEVE VaR/ES values represent the WORST 5% of paths (tail risk),
  where balances dropped significantly AND rates moved adversely at the same time.

  Example at Year 1:
    Balance median  = {risk_evo_df.iloc[0]['bal_median']:>10,.0f}  (typical path — balance grew)
    Balance 5th pctl= {risk_evo_df.iloc[0]['bal_5pctl']:>10,.0f}  (tail path — balance dropped)
    dEVE mean       = {risk_evo_df.iloc[0]['dEVE_mean']:>+10,.0f}  (average outcome is POSITIVE)
    dEVE VaR(95%)   = {risk_evo_df.iloc[0]['dEVE_var']:>+10,.0f}  (worst 5% — large NEGATIVE)

  This is exactly what Monte Carlo adds: it reveals that while the median
  outlook is favourable, the tail scenarios carry significant downside risk
  that static stress tests cannot capture.
""")

print(f"  ┌────────────────────────────────────────────────────────────┐")
print(f"  │  SLIDE 1: Static vs Monte Carlo — Why MC Matters          │")
print(f"  └────────────────────────────────────────────────────────────┘")
print(f"""
  1. STATIC UNDERESTIMATES RISK
     Static worst dEVE    = {static_worst_eve:>10,.0f}  (+200bps parallel shock)
     MC Worst VaR(95%) Y1 = {y1_worst_var:>10,.0f}  (stochastic balance + rate + shock)
     MC is {abs(y1_worst_var / static_worst_eve):.1f}x LARGER than static
     -> Static assumes deterministic balance & fixed curve — MC adds real-world
        uncertainty in deposit flows and interest rate dynamics.

  2. TAIL RISK GROWS WITH HORIZON
     Worst ES(95%): Y1 = {y1_worst_es:>10,.0f} → Y{worst_es_yr} = {worst_es_val:>10,.0f} ({abs(worst_es_val/y1_worst_es - 1)*100:.0f}% increase)
     -> Longer horizons compound balance volatility and rate drift,
        amplifying worst-case outcomes over time.

  3. BALANCE/RATE PATHS DOMINATE THE RISK
     Path-only VaR(95%) = {y1_path_var:>10,.0f}
     -> {pct_from_paths:.0f}% of Year 1 tail risk comes from stochastic balance/rate
        paths ALONE, before any shock scenario is applied.
     -> The "which shock scenario?" question matters less than "what if
        deposit balances decline and rates drift?"
""")

print(f"  ┌────────────────────────────────────────────────────────────┐")
print(f"  │  SLIDE 2: NII Risk & Multi-Year Dynamics                  │")
print(f"  └────────────────────────────────────────────────────────────┘")
print(f"""
  4. NII ZERO-FLOOR SATURATION
     Worst NII VaR = {risk_evo_df.iloc[0]['worst_dNII_var']:>10,.1f} at ALL years
     Base NII      = {nii_base:>10,.1f}
     -> Under -200bps parallel shock, rates hit the zero floor, eliminating
        ALL net interest income (dNII = -base NII exactly).
     -> This means NII tail risk is bounded: you cannot lose more than
        your entire interest income. The zero floor acts as a natural cap.

  5. BALANCE GROWTH vs TAIL DRAWDOWN
     Median balance: {risk_evo_df.iloc[0]['bal_median']:>8,.0f} (Y1) → {risk_evo_df.iloc[4]['bal_median']:>8,.0f} (Y5)  (+{(risk_evo_df.iloc[4]['bal_median']/risk_evo_df.iloc[0]['bal_median']-1)*100:.0f}%)
     5th pctl bal:   {risk_evo_df.iloc[0]['bal_5pctl']:>8,.0f} (Y1) → {risk_evo_df.iloc[4]['bal_5pctl']:>8,.0f} (Y5)
     -> Most paths show healthy growth, but the worst 5% of paths see
        balance decline to {risk_evo_df.iloc[4]['bal_5pctl']:,.0f} — a {abs(risk_evo_df.iloc[4]['bal_5pctl']/calc_balance-1)*100:.0f}% {"drop" if risk_evo_df.iloc[4]['bal_5pctl'] < calc_balance else "gain"} from the starting {calc_balance:,.0f}.
     -> These tail paths drive the large negative dEVE VaR/ES values.

  6. RISK MANAGEMENT IMPLICATIONS
     -> Capital buffers should be sized to MC ES, not static dEVE:
        MC ES at Y{worst_es_yr} = {abs(worst_es_val):,.0f} vs Static = {abs(static_worst_eve):,.0f}
        ({abs(worst_es_val / static_worst_eve):.1f}x ratio)
     -> NII hedging is less critical at short tenors (zero-floor caps losses),
        but EVE sensitivity requires active duration management.
     -> Multi-year horizon reveals compounding risk that 1Y-only analysis misses.
""")

print(f"{'='*80}")
print(f"PHASE 5d (v2) COMPLETE — Monte Carlo Simulation for IRRBB")
print(f"{'='*80}")
print(f"\nOutput files:")
print(f"  mc_v2_path_results.csv        - Per-path MC results ({n_paths} paths x {len(eval_years)} years)")
print(f"  mc_v2_risk_evolution.csv       - Risk metrics per evaluation year")
print(f"  mc_v2_risk_summary.csv         - Full risk metrics comparison table")
print(f"  mc_v2_parameters.csv           - Simulation parameters")
print(f"  mc_section5_paths.png          - Balance & rate path fan charts (5Y)")
print(f"  mc_section6_risk_evolution.png  - Risk evolution over 5 years")
print(f"  mc_section6_shocks_y1.png      - Year 1 shock impact visualizations")
print(f"  mc_section6_multiyear_box.png  - Multi-year worst-case box plots")

# %%
