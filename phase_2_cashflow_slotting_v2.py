# %% [markdown]
# # Phase 2: Cash Flow Slotting (Basel Repricing Profile)
#
# ## QF609 Project #1 - IRRBB Modelling Framework
#
# **Calculation Date:** 31-Dec-2023
# Note: Dataset ends 30-Dec-2023 (last available observation), used as effective date.
#
# **WHAT ARE WE DOING HERE?**
# Distributing our $18,652 deposit balance into 11 time buckets based on
# WHEN we expect the deposits to reprice (i.e., when customers might withdraw).
#
# **WHY?**
# We need to know the TIMING of cash flows to:
# 1. Calculate present value (earlier CF = higher PV)
# 2. Measure repricing risk (what's exposed to rate changes?)
# 3. Meet Basel reporting requirements
#
# **THE 11 BUCKETS:**
# O/N, 1M, 2M, 3M, 6M, 9M, 1Y, 2Y, 3Y, 4Y, 5Y
#
# **THE SLOTTING RULE:**
# - Non-Core → ALL goes to O/N (reprices immediately)
# - Core → Distributed using survival function S(t)
#
# **FORMULA:**
# Cash Flow in bucket i = Core × [S(t_start) - S(t_end)]
#
# **EXAMPLE for 1Y bucket (9M to 1Y):**
# S(270 days) = 0.65  (65% survive to 9 months)
# S(365 days) = 0.60  (60% survive to 1 year)
# CF = 9,511 × (0.65 - 0.60) = 475.55
# → $475.55 will reprice between 9 months and 1 year
#
# This notebook performs:
# - Define standard IRRBB time buckets
# - Slot non-core deposits into O/N bucket
# - Distribute core deposits across buckets using survival function S(t)
# - Generate repricing profile table and visualizations
# - Calculate cumulative repricing schedule

# %% [markdown]
# ## 1. Import Libraries and Load Data

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10

print("Libraries imported successfully")

# %%
import json
import os

# Load data from previous phases
# Load core/non-core allocation from config
with open('config.json', 'r') as f:
    config = json.load(f)

current_balance = config['current_balance']
core_amount = config['core_amount']
non_core_amount = config['non_core_amount']

# ── Survival model selection ───────────────────────────────────────────────────
# The survival model drives ALL cash flow slotting and therefore all EVE/NII
# results. We make this choice explicit and documented here rather than relying
# on which files happen to exist on disk.
#
# CURRENT SELECTION: Advanced model (Empirical Floor Retention Estimator)
# Set USE_ADVANCED_MODEL = False to fall back to the basic exponential model.
# This should match the SELECTED_MODEL in phase_1b_advanced_survival_models.py.
USE_ADVANCED_MODEL = True  # ← change here if needed

if USE_ADVANCED_MODEL and os.path.exists('survival_curve_full_advanced.csv'):
    survival_full = pd.read_csv('survival_curve_full_advanced.csv')
    survival_df   = pd.read_csv('survival_function_table_advanced.csv')
    survival_model = config.get('phase_1b_survival_model', {}).get('model', 'Advanced (Empirical Floor)')
    print(f"✓ Using ADVANCED survival model: {survival_model}")
    if 'phase_1b_survival_model' not in config:
        print(f"  ⚠️  Model name not found in config.json — run Phase 1b Advanced first")
elif not USE_ADVANCED_MODEL and os.path.exists('survival_curve_full.csv'):
    survival_full  = pd.read_csv('survival_curve_full.csv')
    survival_df    = pd.read_csv('survival_function_table.csv')
    survival_model = 'Basic (Exponential)'
    print(f"✓ Using BASIC survival model: {survival_model}")
    print(f"  (USE_ADVANCED_MODEL=False — set to True to use Empirical Floor model)")
else:
    # Hard stop — missing file means all downstream results are invalid
    missing = 'survival_curve_full_advanced.csv' if USE_ADVANCED_MODEL else 'survival_curve_full.csv'
    raise FileNotFoundError(
        f"Required survival curve file not found: '{missing}'\n"
        f"Run Phase 1b {'Advanced' if USE_ADVANCED_MODEL else ''} first to generate it."
    )
# ──────────────────────────────────────────────────────────────────────────────

print(f"\nCurrent Balance:     {current_balance:,.2f}")
print(f"Core Deposits:       {core_amount:,.2f}  ({core_amount/current_balance*100:.2f}%)")
print(f"Non-Core Deposits:   {non_core_amount:,.2f}  ({non_core_amount/current_balance*100:.2f}%)")
print(f"Core/Non-Core Method: {config['method']}")

if 'phase_1b_survival_model' in config:
    print(f"\nSurvival Model Info:")
    print(f"  Model:             {config['phase_1b_survival_model']['model']}")
    print(f"  S(1Y):             {config['phase_1b_survival_model']['survival_1Y']:.4f}  ({config['phase_1b_survival_model']['survival_1Y']*100:.2f}%)")
    print(f"  S(5Y):             {config['phase_1b_survival_model']['survival_5Y']:.4f}  ({config['phase_1b_survival_model']['survival_5Y']*100:.2f}%)")
    print(f"  Interpretation:    {config['phase_1b_survival_model']['interpretation']}")

print(f"\nSurvival function loaded with {len(survival_df)} tenors")

# %% [markdown]
# ## 2. Base Curve Loading and Interpolation Utility
#
# The provided zero curve has tenors: 1D, 1M, 2M, 3M, 6M, 9M, 1Y, 2Y, 3Y, 4Y, 5Y, ...
# For EVE/NII calculations in later phases, we need discount rates at arbitrary
# tenors (e.g., bucket midpoints like 0.0417Y, 0.125Y, 1.5Y, 2.5Y).
# We build a reusable interpolation function here once so Phase 3/4 can import it.

# %%
import os

# Load base curve if available
curve_path = 'group-proj-1-curve.xlsx'
if os.path.exists(curve_path):
    curve_raw = pd.read_excel(curve_path)
    curve_raw.columns = ['Tenor', 'ZeroRate']
    curve_raw = curve_raw.dropna()

    # Convert tenor strings to years
    def tenor_to_years(t):
        t = str(t).strip()
        if t == '1D':   return 1/365
        if t.endswith('M'): return int(t[:-1]) / 12
        if t.endswith('Y'): return int(t[:-1])
        return float(t)

    curve_raw['Tenor_Years'] = curve_raw['Tenor'].apply(tenor_to_years)
    curve_raw = curve_raw.sort_values('Tenor_Years').reset_index(drop=True)

    print("\n" + "="*80)
    print("BASE ZERO RATE CURVE (31-Dec-2023)")
    print("="*80)
    print(curve_raw[['Tenor', 'Tenor_Years', 'ZeroRate']].to_string(index=False))
else:
    print(f"\n⚠️  Curve file '{curve_path}' not found. Interpolation function will need it.")
    curve_raw = None

def interpolate_zero_rate(tenor_years, curve_df=None):
    """
    Linearly interpolate zero rate at any tenor from the base curve.

    Parameters
    ----------
    tenor_years : float or array-like
        Tenor(s) in years at which to interpolate the zero rate.
    curve_df : pd.DataFrame, optional
        DataFrame with columns ['Tenor_Years', 'ZeroRate'].
        Defaults to the module-level curve_raw loaded above.

    Returns
    -------
    float or np.ndarray : Interpolated zero rate(s).

    Notes
    -----
    Uses linear interpolation between knot points.
    For tenors beyond the longest knot, the last rate is held constant (flat extrapolation).
    For tenors before the shortest knot (< 1D), the first rate is used.
    """
    if curve_df is None:
        curve_df = curve_raw
    if curve_df is None:
        raise ValueError("No curve data available. Load curve file first.")

    tenor_arr = np.array(tenor_years, dtype=float)
    rates = np.interp(
        tenor_arr,
        curve_df['Tenor_Years'].values,
        curve_df['ZeroRate'].values
    )
    # Return scalar if input was scalar
    return float(rates) if np.ndim(tenor_years) == 0 else rates

def get_discount_factor(tenor_years, curve_df=None):
    """
    Compute the discount factor at a given tenor using the zero rate curve.
    DF(t) = exp(-r(t) * t)  [continuous compounding]

    Parameters
    ----------
    tenor_years : float or array-like
    curve_df : pd.DataFrame, optional

    Returns
    -------
    float or np.ndarray : Discount factor(s).
    """
    r = interpolate_zero_rate(tenor_years, curve_df)
    t = np.array(tenor_years, dtype=float)
    df = np.exp(-r * t)
    return float(df) if np.ndim(tenor_years) == 0 else df

# Verify interpolation at bucket midpoints
if curve_raw is not None:
    print("\n" + "="*80)
    print("INTERPOLATED RATES AT BUCKET MIDPOINTS (Preview for Phase 3 EVE)")
    print("="*80)
    test_tenors = {
        'O/N':  1/365,
        '1M':   0.0417,
        '2M':   0.125,
        '3M':   0.2083,
        '6M':   0.375,
        '9M':   0.625,
        '1Y':   0.875,
        '2Y':   1.5,
        '3Y':   2.5,
        '4Y':   3.5,
        '5Y':   4.5,
    }
    print(f"{'Bucket':<6}  {'Midpoint (Y)':<13}  {'Zero Rate':>10}  {'Discount Factor':>16}")
    print("-" * 52)
    for bucket_name, t in test_tenors.items():
        r = interpolate_zero_rate(t)
        df = get_discount_factor(t)
        print(f"{bucket_name:<6}  {t:<13.4f}  {r*100:>9.4f}%  {df:>16.6f}")
    print("\n✓ Interpolation function ready. Import interpolate_zero_rate() and")
    print("  get_discount_factor() in Phase 3 (EVE) and Phase 4 (NII).")

# %% [markdown]
# ## 3. Define Time Buckets

# %%
# Define standard IRRBB time buckets
# Structure: Bucket name, Start (days), End (days), Midpoint (years)

buckets = [
    {'Bucket': 'O/N',  'Start_Days': 0,    'End_Days': 1,    'Midpoint_Years': 1/365},
    {'Bucket': '1M',   'Start_Days': 1,    'End_Days': 30,   'Midpoint_Years': 0.0417},
    {'Bucket': '2M',   'Start_Days': 30,   'End_Days': 60,   'Midpoint_Years': 0.125},
    {'Bucket': '3M',   'Start_Days': 60,   'End_Days': 90,   'Midpoint_Years': 0.2083},
    {'Bucket': '6M',   'Start_Days': 90,   'End_Days': 180,  'Midpoint_Years': 0.375},
    {'Bucket': '9M',   'Start_Days': 180,  'End_Days': 270,  'Midpoint_Years': 0.625},
    {'Bucket': '1Y',   'Start_Days': 270,  'End_Days': 365,  'Midpoint_Years': 0.875},
    {'Bucket': '2Y',   'Start_Days': 365,  'End_Days': 730,  'Midpoint_Years': 1.5},
    {'Bucket': '3Y',   'Start_Days': 730,  'End_Days': 1095, 'Midpoint_Years': 2.5},
    {'Bucket': '4Y',   'Start_Days': 1095, 'End_Days': 1460, 'Midpoint_Years': 3.5},
    {'Bucket': '5Y',   'Start_Days': 1460, 'End_Days': 1825, 'Midpoint_Years': 4.5}
]

buckets_df = pd.DataFrame(buckets)

# Calculate bucket width
buckets_df['Width_Days'] = buckets_df['End_Days'] - buckets_df['Start_Days']
buckets_df['Width_Years'] = buckets_df['Width_Days'] / 365

print("="*80)
print("TIME BUCKET DEFINITIONS")
print("="*80)
print(buckets_df.to_string(index=False))

# %% [markdown]
# ## 4. Slot Non-Core Deposits

# %%
# Non-core deposits reprice immediately at O/N
# Entire non-core amount goes into O/N bucket

print("\n" + "="*80)
print("NON-CORE DEPOSIT SLOTTING")
print("="*80)
print(f"Non-Core Amount:     {non_core_amount:,.2f}")
print(f"Bucket Assignment:   O/N (reprices immediately)")
print(f"Behavioral Maturity: Overnight")

# %% [markdown]
# ## 5. Slot Core Deposits Using Survival Function

# %% [markdown]
# ### 4.1 Calculate Cash Flow in Each Bucket
#
# For core deposits distributed across buckets:
# - Cash flow in bucket i = Core × [S(t_{i-1}) - S(t_i)]
# - Where t_i is the end time of bucket i in days

# %%
# Helper function to get survival probability at a given day
def get_survival(days):
    """Get survival probability at specified day from full survival curve.
    Checks for S(t)_RECOMMENDED column first (advanced model output),
    falls back to S(t) for basic model output."""
    col_name = 'S(t)_RECOMMENDED' if 'S(t)_RECOMMENDED' in survival_full.columns else 'S(t)'
    if days >= len(survival_full):
        return survival_full.iloc[-1][col_name]
    return survival_full.iloc[days][col_name]

# Calculate cash flows for each bucket
cashflow_slots = []

for idx, row in buckets_df.iterrows():
    bucket_name = row['Bucket']
    start_days = row['Start_Days']
    end_days = row['End_Days']
    midpoint_years = row['Midpoint_Years']

    # Get survival probabilities at bucket boundaries
    s_start = get_survival(start_days)
    s_end = get_survival(end_days)

    # Calculate cash flow in this bucket
    if bucket_name == 'O/N':
        # O/N bucket gets all non-core + marginal core decay from day 0 to 1
        core_cf = core_amount * (s_start - s_end)
        total_cf = non_core_amount + core_cf
    elif bucket_name == '5Y':
        # 5Y bucket gets marginal decay (4Y-5Y) + residual beyond 5Y (Basel cap)
        marginal_decay = core_amount * (s_start - s_end)
        residual_beyond_5y = core_amount * s_end  # S(1825) remainder - everything surviving beyond 5Y
        core_cf = marginal_decay + residual_beyond_5y
        total_cf = core_cf
    else:
        # Other buckets get marginal core decay only
        core_cf = core_amount * (s_start - s_end)
        total_cf = core_cf

    cashflow_slots.append({
        'Bucket': bucket_name,
        'Start_Days': start_days,
        'End_Days': end_days,
        'Midpoint_Years': midpoint_years,
        'S(t_start)': s_start,
        'S(t_end)': s_end,
        'Marginal_Decay': s_start - s_end,
        'Core_CF': core_cf,
        'Non_Core_CF': non_core_amount if bucket_name == 'O/N' else 0,
        'Total_CF': total_cf,
        'CF_Percent': 0  # Will calculate after
    })

repricing_profile = pd.DataFrame(cashflow_slots)

# Calculate percentages
repricing_profile['CF_Percent'] = (repricing_profile['Total_CF'] / current_balance) * 100

# Verify total matches current balance
total_cf = repricing_profile['Total_CF'].sum()
total_core_cf = repricing_profile['Core_CF'].sum()

print("\n" + "="*80)
print("CASH FLOW SLOTTING RESULTS")
print("="*80)
print(f"Total Cash Flows:        {total_cf:,.2f}")
print(f"Current Balance:         {current_balance:,.2f}")
print(f"Difference:              {abs(total_cf - current_balance):,.6f}")
print(f"Match: {'✓ PASS' if abs(total_cf - current_balance) < 0.01 else '✗ FAIL'}")

print(f"\nCore Allocation Check:")
print(f"Total Core CF:           {total_core_cf:,.2f}")
print(f"Expected Core:           {core_amount:,.2f}")
print(f"Difference:              {abs(total_core_cf - core_amount):,.6f}")
print(f"Match: {'✓ PASS' if abs(total_core_cf - core_amount) < 0.01 else '✗ FAIL'}")

# Show 5Y bucket breakdown
five_y_row = repricing_profile[repricing_profile['Bucket'] == '5Y'].iloc[0]
s_5y = five_y_row['S(t_end)']
marginal_5y = core_amount * five_y_row['Marginal_Decay']
residual_5y = core_amount * s_5y

print(f"\n5Y Bucket Breakdown (Basel 5Y Cap):")
print(f"  Marginal decay (4Y-5Y):   {marginal_5y:,.2f}  ({marginal_5y/core_amount*100:.2f}% of core)")
print(f"  Residual (beyond 5Y):     {residual_5y:,.2f}  ({residual_5y/core_amount*100:.2f}% of core)")
print(f"  Total 5Y allocation:      {five_y_row['Core_CF']:,.2f}  ({five_y_row['CF_Percent']:.2f}% of balance)")
print(f"\n  → S(5Y) = {s_5y*100:.2f}% of deposits survive beyond 5 years")
print(f"  → Per Basel, these are capped at 5Y maturity")

# %% [markdown]
# ### 4.2 Repricing Profile Table

# %%
print("\n" + "="*80)
print("REPRICING PROFILE - CASH FLOW BY BUCKET")
print("="*80)
print(repricing_profile[['Bucket', 'Midpoint_Years', 'Core_CF', 'Non_Core_CF',
                          'Total_CF', 'CF_Percent']].to_string(index=False))

# Summary statistics
print("\n" + "-"*80)
print("SUMMARY STATISTICS")
print("-"*80)
print(f"O/N Bucket (immediate repricing):    {repricing_profile[repricing_profile['Bucket']=='O/N']['Total_CF'].values[0]:,.2f}  ({repricing_profile[repricing_profile['Bucket']=='O/N']['CF_Percent'].values[0]:.2f}%)")
print(f"1M-5Y Buckets (behavioral maturity): {repricing_profile[repricing_profile['Bucket']!='O/N']['Total_CF'].sum():,.2f}  ({repricing_profile[repricing_profile['Bucket']!='O/N']['CF_Percent'].sum():.2f}%)")

# Weighted average maturity of total balance
weighted_maturity = (repricing_profile['Total_CF'] * repricing_profile['Midpoint_Years']).sum() / repricing_profile['Total_CF'].sum()
print(f"\nWeighted Average Maturity (total):    {weighted_maturity:.3f} years")

# ── WAM of core only (Basel-relevant measure) ─────────────────────────────────
# Basel constrains the average maturity of CORE deposits (not including O/N non-core)
# to ≤ 5 years for retail NMDs. We verify this explicitly.
core_only = repricing_profile[repricing_profile['Bucket'] != 'O/N'].copy()
# Core-only WAM excludes non-core from both numerator and denominator
wam_core = (core_only['Core_CF'] * core_only['Midpoint_Years']).sum() / core_only['Core_CF'].sum()
print(f"Weighted Average Maturity (core):     {wam_core:.3f} years")
print(f"Basel 5Y Core WAM Cap:                5.000 years")
if wam_core <= 5.0:
    print(f"Core WAM Check:                       ✓ PASS  ({wam_core:.3f}Y ≤ 5Y)")
else:
    print(f"Core WAM Check:                       ✗ FAIL  ({wam_core:.3f}Y > 5Y — exceeds Basel cap!)")
# ──────────────────────────────────────────────────────────────────────────────

# %% [markdown]
# ## 6. Visualizations

# %%
# Create output directories for charts and data exports if they don't exist
os.makedirs('visualizations', exist_ok=True)
os.makedirs('viz_data', exist_ok=True)

# %% [markdown]
# ### 5.1 Bar Chart: Cash Flow Distribution Across Buckets

# %%
fig, ax = plt.subplots(figsize=(14, 7))

buckets_list = repricing_profile['Bucket'].values
core_cfs = repricing_profile['Core_CF'].values
non_core_cfs = repricing_profile['Non_Core_CF'].values

# Create stacked bar chart
x_pos = np.arange(len(buckets_list))
bars1 = ax.bar(x_pos, non_core_cfs, color='#D62828', alpha=0.8,
               edgecolor='black', label='Non-Core')
bars2 = ax.bar(x_pos, core_cfs, bottom=non_core_cfs, color='#06A77D',
               alpha=0.8, edgecolor='black', label='Core')

# Add value labels on bars
for i, (bucket, total_cf, pct) in enumerate(zip(buckets_list, repricing_profile['Total_CF'], repricing_profile['CF_Percent'])):
    if total_cf > 100:  # Only label bars with significant amounts
        ax.text(i, total_cf + 100, f'{total_cf:,.0f}\n({pct:.1f}%)',
                ha='center', va='bottom', fontsize=9, fontweight='bold')

ax.set_xlabel('Time Bucket', fontsize=12)
ax.set_ylabel('Cash Flow Amount', fontsize=12)
ax.set_title('NMD Repricing Profile - Cash Flow Distribution Across Buckets',
             fontsize=14, fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels(buckets_list, fontsize=10)
ax.legend(loc='upper right', fontsize=11)
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('visualizations/cashflow_distribution_bar.png', dpi=150, bbox_inches='tight')
df_bar = repricing_profile[['Bucket', 'Core_CF', 'Non_Core_CF']].copy()
df_bar.to_csv('viz_data/cashflow_distribution_bar.csv', index=False)
plt.close()

# %% [markdown]
# ### 5.2 Waterfall Chart: Cumulative Runoff

# %%
fig, ax = plt.subplots(figsize=(14, 7))

# Calculate cumulative cash flows
cumulative_cf = repricing_profile['Total_CF'].cumsum()
cumulative_pct = (cumulative_cf / current_balance) * 100

# Plot cumulative line
ax.plot(buckets_list, cumulative_cf, marker='o', linewidth=3, markersize=10,
        color='#023047', label='Cumulative Cash Flow')

# Add percentage labels
for i, (bucket, cum_cf, cum_pct) in enumerate(zip(buckets_list, cumulative_cf, cumulative_pct)):
    ax.text(i, cum_cf + 300, f'{cum_pct:.1f}%',
            ha='center', va='bottom', fontsize=9, fontweight='bold')

# Mark key milestones
ax.axhline(y=current_balance, color='red', linestyle='--', linewidth=2,
           alpha=0.7, label=f'Total Balance: {current_balance:,.0f}')
ax.axhline(y=current_balance/2, color='orange', linestyle=':', linewidth=1.5,
           alpha=0.6, label='50% Repriced')

ax.set_xlabel('Time Bucket', fontsize=12)
ax.set_ylabel('Cumulative Cash Flow', fontsize=12)
ax.set_title('Cumulative Repricing Schedule (Waterfall)', fontsize=14, fontweight='bold')
ax.legend(loc='lower right', fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('visualizations/cumulative_runoff_waterfall.png', dpi=150, bbox_inches='tight')
df_waterfall = pd.DataFrame({
    'Bucket': buckets_list,
    'Cumulative_CF': cumulative_cf.values,
    'Cumulative_Pct': cumulative_pct.values,
})
df_waterfall.to_csv('viz_data/cumulative_runoff_waterfall.csv', index=False)
plt.close()

# Find bucket where 50% is repriced
half_balance = current_balance / 2
bucket_50pct = repricing_profile[cumulative_cf >= half_balance].iloc[0]['Bucket']
print(f"\n50% of balance repriced by bucket: {bucket_50pct}")

# %% [markdown]
# ### 5.3 Percentage Distribution Pie Chart

# %%
# Group small buckets for cleaner visualization
pie_data = repricing_profile.copy()
pie_data_grouped = []

for _, row in pie_data.iterrows():
    if row['CF_Percent'] >= 3:  # Only show buckets >= 3%
        pie_data_grouped.append({
            'Bucket': row['Bucket'],
            'CF_Percent': row['CF_Percent']
        })
    else:
        # Add to "Other" category
        if len(pie_data_grouped) > 0 and pie_data_grouped[-1]['Bucket'] == 'Other':
            pie_data_grouped[-1]['CF_Percent'] += row['CF_Percent']
        else:
            pie_data_grouped.append({
                'Bucket': 'Other',
                'CF_Percent': row['CF_Percent']
            })

pie_df = pd.DataFrame(pie_data_grouped)

fig, ax = plt.subplots(figsize=(10, 8))

colors_pie = ['#D62828', '#06A77D', '#2A9D8F', '#264653', '#E76F51',
              '#F4A261', '#E9C46A', '#8ECAE6', '#219EBC', '#023047', '#CCCCCC']

wedges, texts, autotexts = ax.pie(pie_df['CF_Percent'], labels=pie_df['Bucket'],
                                    autopct='%1.1f%%', startangle=90,
                                    colors=colors_pie[:len(pie_df)],
                                    textprops={'fontsize': 11, 'fontweight': 'bold'})

ax.set_title('Repricing Profile - Percentage Distribution', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('visualizations/percentage_distribution_pie.png', dpi=150, bbox_inches='tight')
pie_df.to_csv('viz_data/percentage_distribution_pie.csv', index=False)
plt.close()

# %% [markdown]
# ### 5.4 Survival Function vs Cumulative Repricing

# %%
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Left: Survival function with bucket boundaries
# Use get_survival() to handle both S(t) and S(t)_RECOMMENDED column names
days_full  = np.arange(0, 5 * 365 + 1)
years_full = days_full / 365.0
s_full     = np.array([get_survival(d) for d in days_full])

axes[0].plot(years_full, s_full * 100, linewidth=3, color='#023047',
             label=f'Survival Function S(t) — {survival_model}')

# Mark bucket boundaries
for _, row in repricing_profile.iterrows():
    if row['Bucket'] != 'O/N':
        axes[0].axvline(x=row['End_Days']/365, color='red', linestyle='--',
                       linewidth=1, alpha=0.5)
        axes[0].text(row['End_Days']/365, 5, row['Bucket'],
                    ha='center', fontsize=8, rotation=90)

axes[0].set_xlabel('Time (Years)', fontsize=11)
axes[0].set_ylabel('Survival Probability (%)', fontsize=11)
axes[0].set_title('Survival Function with Bucket Boundaries', fontsize=12, fontweight='bold')
axes[0].set_xlim(0, 5)
axes[0].set_ylim(0, 105)
axes[0].grid(True, alpha=0.3)
axes[0].legend()

# Right: Cumulative repricing (inverse of survival for core portion)
axes[1].plot(buckets_list, cumulative_pct, marker='o', linewidth=3,
             markersize=10, color='#E63946', label='Cumulative Repricing %')

axes[1].axhline(y=50, color='orange', linestyle='--', linewidth=2,
                alpha=0.7, label='50% Threshold')
axes[1].axhline(y=100, color='green', linestyle='--', linewidth=2,
                alpha=0.7, label='100% (Full Balance)')

axes[1].set_xlabel('Time Bucket', fontsize=11)
axes[1].set_ylabel('Cumulative Repricing (%)', fontsize=11)
axes[1].set_title('Cumulative Repricing Schedule', fontsize=12, fontweight='bold')
axes[1].set_ylim(0, 110)
axes[1].grid(True, alpha=0.3)
axes[1].legend()

plt.tight_layout()
plt.savefig('visualizations/survival_vs_repricing.png', dpi=150, bbox_inches='tight')
df_survival_repricing = pd.DataFrame({
    'Years': years_full,
    'S(t)_pct': s_full * 100,
})
df_survival_repricing.to_csv('viz_data/survival_vs_repricing.csv', index=False)
plt.close()

# %% [markdown]
# ## 7. Detailed Analysis by Bucket

# %%
print("\n" + "="*80)
print("DETAILED BUCKET ANALYSIS")
print("="*80)

for idx, row in repricing_profile.iterrows():
    print(f"\n{row['Bucket']} Bucket:")
    print(f"  Time Range:         {row['Start_Days']}-{row['End_Days']} days ({row['Start_Days']/365:.3f}-{row['End_Days']/365:.3f} years)")
    print(f"  Midpoint:           {row['Midpoint_Years']:.4f} years")
    print(f"  Survival Start:     {row['S(t_start)']:.4f}")
    print(f"  Survival End:       {row['S(t_end)']:.4f}")
    print(f"  Marginal Decay:     {row['Marginal_Decay']:.4f}  ({row['Marginal_Decay']*100:.2f}%)")
    print(f"  Core Cash Flow:     {row['Core_CF']:,.2f}")
    print(f"  Non-Core Cash Flow: {row['Non_Core_CF']:,.2f}")
    print(f"  Total Cash Flow:    {row['Total_CF']:,.2f}  ({row['CF_Percent']:.2f}% of balance)")

# %% [markdown]
# ## 8. Summary and Export

# %%
print("\n" + "="*80)
print("PHASE 2 SUMMARY - CASH FLOW SLOTTING")
print("="*80)

print("\n1. REPRICING PROFILE OVERVIEW")
print("-" * 80)
print(f"Total Balance:                    {current_balance:,.2f}")
print(f"Number of Time Buckets:           {len(buckets_df)}")
print(f"Time Horizon:                     O/N to 5Y")
print(f"Weighted Average Maturity:        {weighted_maturity:.3f} years")
print(f"Survival Model Used:              {survival_model}")
if 'phase_1b_survival_model' in config:
    print(f"  S(5Y) of model:                 {get_survival(1825)*100:.2f}%")

print("\n2. IMMEDIATE REPRICING (O/N)")
print("-" * 80)
on_cf = repricing_profile[repricing_profile['Bucket']=='O/N']['Total_CF'].values[0]
on_pct = repricing_profile[repricing_profile['Bucket']=='O/N']['CF_Percent'].values[0]
print(f"O/N Bucket Cash Flow:             {on_cf:,.2f}  ({on_pct:.2f}%)")
print(f"  - Non-Core (volatile):          {non_core_amount:,.2f}")
print(f"  - Core (day 0-1 decay):         {on_cf - non_core_amount:,.2f}")

print("\n3. BEHAVIORAL MATURITY (1M-5Y)")
print("-" * 80)
longer_cf = repricing_profile[repricing_profile['Bucket']!='O/N']['Total_CF'].sum()
longer_pct = repricing_profile[repricing_profile['Bucket']!='O/N']['CF_Percent'].sum()
print(f"1M-5Y Buckets Cash Flow:          {longer_cf:,.2f}  ({longer_pct:.2f}%)")
print(f"Distributed across 10 buckets using survival function S(t)")

print("\n4. KEY TENORS")
print("-" * 80)
key_tenors_list = ['1M', '3M', '6M', '1Y', '2Y', '3Y', '4Y', '5Y']
for tenor in key_tenors_list:
    if tenor in repricing_profile['Bucket'].values:
        cf = repricing_profile[repricing_profile['Bucket']==tenor]['Total_CF'].values[0]
        pct = repricing_profile[repricing_profile['Bucket']==tenor]['CF_Percent'].values[0]
        print(f"  {tenor:4s} Bucket:  {cf:10,.2f}  ({pct:5.2f}%)")

print("\n5. CUMULATIVE REPRICING MILESTONES")
print("-" * 80)
cumulative_cf = repricing_profile['Total_CF'].cumsum()
cumulative_pct = (cumulative_cf / current_balance) * 100

milestones = [25, 50, 75, 90]
for milestone in milestones:
    bucket_idx = (cumulative_pct >= milestone).idxmax()
    bucket_name = repricing_profile.loc[bucket_idx, 'Bucket']
    print(f"  {milestone}% repriced by:        {bucket_name} bucket")

print("\n6. VALIDATION")
print("-" * 80)
print(f"Sum of Core CF:                   {repricing_profile['Core_CF'].sum():,.2f}")
print(f"Expected Core:                    {core_amount:,.2f}")
print(f"Core Match: {'✓ PASS' if abs(repricing_profile['Core_CF'].sum() - core_amount) < 0.01 else '✗ FAIL'}")
sum_total_cf = repricing_profile['Total_CF'].sum()
print(f"\nSum of Total CF:                  {sum_total_cf:,.2f}")
print(f"Current Balance:                  {current_balance:,.2f}")
print(f"Total Match: {'✓ PASS' if abs(sum_total_cf - current_balance) < 0.01 else '✗ FAIL'}")

print("\n7. NEXT STEPS (PHASE 3 & 4)")
print("-" * 80)
print("  - Phase 3: Calculate EVE (Economic Value of Equity) sensitivity")
print("    → Apply 4 rate shock scenarios to repricing profile")
print("    → Discount cash flows using shocked curves")
print("    → Calculate ΔEVE for each scenario")
print("\n  - Phase 4: Calculate NII (Net Interest Income) sensitivity")
print("    → Focus on buckets within 12-month horizon")
print("    → Calculate interest income impact from rate shocks")
print("    → Identify worst-case scenarios")

print("\n" + "="*80)
print("PHASE 2 COMPLETE")
print("="*80)

# %%
# Save repricing profile for use in Phase 3 and 4
repricing_profile.to_csv('repricing_profile.csv', index=False)

# Create simplified version for presentation
repricing_summary = repricing_profile[['Bucket', 'Midpoint_Years', 'Total_CF', 'CF_Percent']].copy()
repricing_summary.columns = ['Bucket', 'Midpoint (Years)', 'Cash Flow', 'Percentage (%)']
repricing_summary.to_csv('repricing_profile_summary.csv', index=False)

print("\nData saved:")
print("- repricing_profile.csv (full repricing profile with all details)")
print("- repricing_profile_summary.csv (simplified version for presentation)")

# %%
