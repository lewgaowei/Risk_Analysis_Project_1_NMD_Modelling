# %% [markdown]
# # Phase 4: NII (Net Interest Income) Sensitivity
#
# ## QF609 Project #1 - IRRBB Modelling Framework
#
# **Calculation Date:** 30-Dec-2023
#
# **WHAT IS NII?**
# NII = Net Interest Income = Interest earned - Interest paid
# It measures the bank's SHORT-TERM earnings (next 12 months)
#
# **EVE vs NII - What's the Difference?**
#
# | Metric | Time Horizon | What It Measures | Why It Matters |
# |--------|--------------|------------------|----------------|
# | **EVE** | Long-term (ALL cash flows) | Economic value | Will bank survive? (solvency) |
# | **NII** | Short-term (12 months) | Earnings | Will bank be profitable this year? |
#
# **ANALOGY:**
# - EVE = Your total retirement portfolio value
# - NII = Your salary this year
# Both matter! You need income today AND wealth for the future.
#
# **WHY ONLY 12 MONTHS?**
# NII looks at earnings impact over the next year, so we only include
# buckets that REPRICE within 1 year:
# - O/N, 1M, 2M, 3M, 6M, 9M, 1Y ✓ (included)
# - 2Y, 3Y, 4Y, 5Y ✗ (excluded)
#
# **THE FORMULA:**
# ΔNII = Σ [CF(i) × shock(t_i) × (1 - t_i)]
#
# The (1 - t_i) factor accounts for "how much of the year is left"
#
# **EXAMPLE for 3M bucket:**
# - CF = $5,000
# - Shock = +2% (200bps rate increase)
# - t = 0.25 years (3 months)
# - Time remaining = 1 - 0.25 = 0.75 years
# - ΔNII = $5,000 × 0.02 × 0.75 = $75
# → You earn an extra $75 over the next 12 months
#
# This notebook performs:
# - Calculate NII impact for buckets repricing within 12 months
# - Apply same 4 rate shock scenarios as Phase 3
# - Calculate ΔNII (change in net interest income)
# - Identify worst-case NII scenario
# - Compare EVE vs NII binding constraints (might be different!)
# - Generate combined IRRBB summary report

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
# Load data from previous phases
repricing_profile = pd.read_csv('repricing_profile_with_eve.csv')
eve_summary = pd.read_csv('eve_sensitivity_summary.csv')
shocked_curves = pd.read_csv('shocked_yield_curves.csv')

print(f"Repricing Profile: {len(repricing_profile)} buckets")
print(f"EVE Summary loaded: {len(eve_summary)} scenarios")
print(f"\nTotal Cash Flow: {repricing_profile['Total_CF'].sum():,.2f}")

# %% [markdown]
# ## 2. NII Framework Overview

# %%
print("="*80)
print("NII SENSITIVITY FRAMEWORK")
print("="*80)
print("\nNII (Net Interest Income) measures earnings impact over NEXT 12 MONTHS")
print("\nKey Differences from EVE:")
print("  - EVE: Long-term economic value (ALL cash flows)")
print("  - NII: Short-term earnings (12-month horizon ONLY)")
print("\nNII Calculation:")
print("  - Only includes buckets that REPRICE within 1 year")
print("  - Formula: ΔNII(i) = CF(i) × shock(t_i) × (1 - t_i)")
print("  - (1 - t_i) factor: accounts for remaining time in year after repricing")
print("\nExample:")
print("  - 3M bucket: reprices at t=0.25Y")
print("  - Earns shocked rate for remaining (1 - 0.25) = 0.75 years")

# %% [markdown]
# ## 3. Identify Buckets Within 12-Month Horizon

# %%
# Filter buckets that reprice within 1 year
nii_buckets = repricing_profile[repricing_profile['Midpoint_Years'] <= 1.0].copy()

print("\n" + "="*80)
print("BUCKETS INCLUDED IN NII CALCULATION")
print("="*80)
print(nii_buckets[['Bucket', 'Midpoint_Years', 'Total_CF', 'CF_Percent']].to_string(index=False))

total_nii_cf = nii_buckets['Total_CF'].sum()
total_cf = repricing_profile['Total_CF'].sum()
nii_cf_pct = (total_nii_cf / total_cf) * 100

print(f"\nTotal CF in NII buckets:     {total_nii_cf:,.2f}  ({nii_cf_pct:.2f}% of total)")
print(f"Total CF beyond 1Y:          {total_cf - total_nii_cf:,.2f}  ({100-nii_cf_pct:.2f}% of total)")

# %% [markdown]
# ## 4. Define Rate Shocks at Each Bucket Midpoint

# %%
# Helper function to calculate shock at a given tenor
def get_shock_s1(t_years):
    """S1: +200bps parallel"""
    return 0.02

def get_shock_s2(t_years):
    """S2: -200bps parallel (with zero floor on resulting rate)"""
    return -0.02

def get_shock_s3(t_years):
    """S3: Steepener - +200bps tapering to 0 at 10Y"""
    return 0.02 * max(1 - t_years / 10, 0)

def get_shock_s4(t_years):
    """S4: Flattener - +200bps to 0 at 5Y, then -100bps at 10Y"""
    if t_years <= 5:
        return 0.02 * (1 - t_years / 5)
    else:
        return -0.01 * (t_years - 5) / 5

# Calculate shocks at each NII bucket midpoint
nii_buckets['Shock_S1_bps'] = nii_buckets['Midpoint_Years'].apply(lambda t: get_shock_s1(t) * 10000)
nii_buckets['Shock_S2_bps'] = nii_buckets['Midpoint_Years'].apply(lambda t: get_shock_s2(t) * 10000)
nii_buckets['Shock_S3_bps'] = nii_buckets['Midpoint_Years'].apply(lambda t: get_shock_s3(t) * 10000)
nii_buckets['Shock_S4_bps'] = nii_buckets['Midpoint_Years'].apply(lambda t: get_shock_s4(t) * 10000)

print("\n" + "="*80)
print("RATE SHOCKS AT NII BUCKET MIDPOINTS (basis points)")
print("="*80)
print(nii_buckets[['Bucket', 'Midpoint_Years', 'Shock_S1_bps', 'Shock_S2_bps',
                    'Shock_S3_bps', 'Shock_S4_bps']].to_string(index=False))

# %% [markdown]
# ## 5. Calculate ΔNII for Each Scenario

# %% [markdown]
# ### 5.1 NII Calculation Formula
#
# For each bucket i that reprices within 1 year:
# - **ΔNII(i) = CF(i) × shock(t_i) × (1 - t_i)**
# - Where:
#   - CF(i) = cash flow in bucket i
#   - shock(t_i) = rate shock at bucket midpoint (decimal)
#   - (1 - t_i) = remaining time in year after repricing
#
# Total ΔNII = sum across all buckets within 1Y horizon

# %%
# Calculate time factor (remaining time in year)
nii_buckets['Time_Factor'] = 1 - nii_buckets['Midpoint_Years']

# Calculate ΔNII for each scenario
nii_buckets['ΔNII_S1'] = (nii_buckets['Total_CF'] *
                          nii_buckets['Shock_S1_bps'] / 10000 *
                          nii_buckets['Time_Factor'])

nii_buckets['ΔNII_S2'] = (nii_buckets['Total_CF'] *
                          nii_buckets['Shock_S2_bps'] / 10000 *
                          nii_buckets['Time_Factor'])

nii_buckets['ΔNII_S3'] = (nii_buckets['Total_CF'] *
                          nii_buckets['Shock_S3_bps'] / 10000 *
                          nii_buckets['Time_Factor'])

nii_buckets['ΔNII_S4'] = (nii_buckets['Total_CF'] *
                          nii_buckets['Shock_S4_bps'] / 10000 *
                          nii_buckets['Time_Factor'])

print("\n" + "="*80)
print("ΔNII BY BUCKET AND SCENARIO")
print("="*80)
print(nii_buckets[['Bucket', 'Total_CF', 'Time_Factor', 'ΔNII_S1',
                    'ΔNII_S2', 'ΔNII_S3', 'ΔNII_S4']].to_string(index=False))

# %% [markdown]
# ### 5.2 Total ΔNII for Each Scenario

# %%
# Sum ΔNII across all buckets
delta_nii_s1 = nii_buckets['ΔNII_S1'].sum()
delta_nii_s2 = nii_buckets['ΔNII_S2'].sum()
delta_nii_s3 = nii_buckets['ΔNII_S3'].sum()
delta_nii_s4 = nii_buckets['ΔNII_S4'].sum()

# Create NII summary table
nii_summary = pd.DataFrame({
    'Scenario': [
        'S1: +200bps Parallel',
        'S2: -200bps Parallel',
        'S3: Short Rate Up (Steepener)',
        'S4: Flattener'
    ],
    'ΔNII': [delta_nii_s1, delta_nii_s2, delta_nii_s3, delta_nii_s4]
})

# Calculate percentage of base earnings (using total CF as proxy for base earnings)
# Note: In practice, would use actual interest margin × balance
base_earnings_proxy = total_cf * 0.02  # Assume 2% margin as proxy
nii_summary['ΔNII_%_of_Base'] = (nii_summary['ΔNII'] / base_earnings_proxy) * 100

print("\n" + "="*80)
print("NII SENSITIVITY SUMMARY")
print("="*80)
print(nii_summary.to_string(index=False))

# Identify worst case
worst_case_nii_idx = nii_summary['ΔNII'].idxmin()
worst_case_nii_scenario = nii_summary.loc[worst_case_nii_idx, 'Scenario']
worst_case_nii_delta = nii_summary.loc[worst_case_nii_idx, 'ΔNII']

print("\n" + "="*80)
print("WORST CASE NII SCENARIO")
print("="*80)
print(f"Scenario:        {worst_case_nii_scenario}")
print(f"ΔNII:            {worst_case_nii_delta:,.2f}")
print(f"NII Impact:      {abs(worst_case_nii_delta):,.2f} earnings reduction over 12 months")

# %% [markdown]
# ## 6. Visualizations - NII Sensitivity

# %% [markdown]
# ### 6.1 Bar Chart: ΔNII by Scenario

# %%
fig, ax = plt.subplots(figsize=(12, 7))

scenarios_list = nii_summary['Scenario'].values
delta_niis = nii_summary['ΔNII'].values
colors = ['#E63946' if x < 0 else '#06A77D' for x in delta_niis]

bars = ax.bar(range(len(scenarios_list)), delta_niis, color=colors,
              alpha=0.8, edgecolor='black', linewidth=1.5)

# Highlight worst case
worst_idx = np.argmin(delta_niis)
bars[worst_idx].set_linewidth(3)
bars[worst_idx].set_edgecolor('darkred')

# Add value labels
for i, (bar, val) in enumerate(zip(bars, delta_niis)):
    height = bar.get_height()
    label_y = height + (5 if height > 0 else -15)
    ax.text(bar.get_x() + bar.get_width()/2., label_y,
            f'{val:,.0f}',
            ha='center', va='bottom' if height > 0 else 'top',
            fontsize=11, fontweight='bold')

ax.axhline(y=0, color='black', linewidth=1.5, alpha=0.7)
ax.set_ylabel('ΔNII (Change in Net Interest Income)', fontsize=12)
ax.set_title('NII Sensitivity Across Rate Shock Scenarios (12-Month Horizon)',
             fontsize=14, fontweight='bold')
ax.set_xticks(range(len(scenarios_list)))
ax.set_xticklabels(['S1: +200\nParallel', 'S2: -200\nParallel',
                    'S3: Short Up\n(Steepener)', 'S4: Flattener'],
                   fontsize=10)
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.show()

# %% [markdown]
# ### 6.2 Stacked Bar: ΔNII Contribution by Bucket

# %%
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

scenarios_nii = [
    ('ΔNII_S1', 'S1: +200bps Parallel', '#E63946'),
    ('ΔNII_S2', 'S2: -200bps Parallel', '#06A77D'),
    ('ΔNII_S3', 'S3: Steepener', '#F77F00'),
    ('ΔNII_S4', 'S4: Flattener', '#9B59B6')
]

for idx, (col, title, color) in enumerate(scenarios_nii):
    ax = axes[idx // 2, idx % 2]

    buckets_list = nii_buckets['Bucket'].values
    delta_nii_values = nii_buckets[col].values

    colors_bars = [color if x >= 0 else '#D62828' for x in delta_nii_values]

    bars = ax.bar(range(len(buckets_list)), delta_nii_values,
                  color=colors_bars, alpha=0.8, edgecolor='black')

    # Add value labels
    for i, (bucket, val) in enumerate(zip(buckets_list, delta_nii_values)):
        if abs(val) > 1:  # Only label significant values
            label_y = val + (0.5 if val > 0 else -1.5)
            ax.text(i, label_y, f'{val:.0f}',
                   ha='center', va='bottom' if val > 0 else 'top',
                   fontsize=8)

    ax.axhline(y=0, color='black', linewidth=1, alpha=0.5)
    ax.set_xlabel('Time Bucket', fontsize=11)
    ax.set_ylabel('ΔNII Contribution', fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xticks(range(len(buckets_list)))
    ax.set_xticklabels(buckets_list, fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.show()

# %% [markdown]
# ## 7. Combined EVE and NII Analysis

# %% [markdown]
# ### 7.1 Combined Summary Table

# %%
# Merge EVE and NII results
# Extract EVE deltas from EVE summary (excluding base case)
eve_data = eve_summary[eve_summary['Scenario'] != 'Base Case'].copy()

combined_summary = pd.DataFrame({
    'Scenario': ['S1: +200bps Parallel', 'S2: -200bps Parallel',
                 'S3: Short Rate Up (Steepener)', 'S4: Flattener'],
    'ΔEVE': [
        eve_data[eve_data['Scenario'] == 'S1: +200bps Parallel']['ΔEVE'].values[0],
        eve_data[eve_data['Scenario'] == 'S2: -200bps Parallel']['ΔEVE'].values[0],
        eve_data[eve_data['Scenario'] == 'S3: Short Rate Up (Steepener)']['ΔEVE'].values[0],
        eve_data[eve_data['Scenario'] == 'S4: Flattener']['ΔEVE'].values[0]
    ],
    'ΔEVE_%': [
        eve_data[eve_data['Scenario'] == 'S1: +200bps Parallel']['ΔEVE_%'].values[0],
        eve_data[eve_data['Scenario'] == 'S2: -200bps Parallel']['ΔEVE_%'].values[0],
        eve_data[eve_data['Scenario'] == 'S3: Short Rate Up (Steepener)']['ΔEVE_%'].values[0],
        eve_data[eve_data['Scenario'] == 'S4: Flattener']['ΔEVE_%'].values[0]
    ],
    'ΔNII': [delta_nii_s1, delta_nii_s2, delta_nii_s3, delta_nii_s4]
})

print("\n" + "="*80)
print("COMBINED EVE AND NII SENSITIVITY")
print("="*80)
print(combined_summary.to_string(index=False))

# Identify binding constraints
worst_eve_idx = combined_summary['ΔEVE'].idxmin()
worst_nii_idx = combined_summary['ΔNII'].idxmin()

print("\n" + "="*80)
print("WORST CASE SCENARIOS")
print("="*80)
print(f"\nEVE Worst Case:")
print(f"  Scenario:  {combined_summary.loc[worst_eve_idx, 'Scenario']}")
print(f"  ΔEVE:      {combined_summary.loc[worst_eve_idx, 'ΔEVE']:,.2f}  ({combined_summary.loc[worst_eve_idx, 'ΔEVE_%']:+.2f}%)")

print(f"\nNII Worst Case:")
print(f"  Scenario:  {combined_summary.loc[worst_nii_idx, 'Scenario']}")
print(f"  ΔNII:      {combined_summary.loc[worst_nii_idx, 'ΔNII']:,.2f}")

if worst_eve_idx == worst_nii_idx:
    print(f"\n✓ SAME SCENARIO binds for both EVE and NII")
else:
    print(f"\n⚠ DIFFERENT SCENARIOS bind for EVE vs NII")
    print(f"  Bank must manage BOTH constraints")

# %% [markdown]
# ### 7.2 Combined Visualization: EVE vs NII

# %%
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

scenarios_short = ['S1: +200\nParallel', 'S2: -200\nParallel',
                   'S3: Steepener', 'S4: Flattener']

# Left: ΔEVE
delta_eves = combined_summary['ΔEVE'].values
colors_eve = ['#E63946' if x < 0 else '#06A77D' for x in delta_eves]

bars_eve = axes[0].bar(range(len(scenarios_short)), delta_eves,
                       color=colors_eve, alpha=0.8, edgecolor='black', linewidth=1.5)

# Highlight worst case
worst_idx_eve = np.argmin(delta_eves)
bars_eve[worst_idx_eve].set_linewidth(3)
bars_eve[worst_idx_eve].set_edgecolor('darkred')

for i, (bar, val, pct) in enumerate(zip(bars_eve, delta_eves, combined_summary['ΔEVE_%'])):
    height = bar.get_height()
    label_y = height + (50 if height > 0 else -150)
    axes[0].text(bar.get_x() + bar.get_width()/2., label_y,
                f'{val:,.0f}\n({pct:+.2f}%)',
                ha='center', va='bottom' if height > 0 else 'top',
                fontsize=10, fontweight='bold')

axes[0].axhline(y=0, color='black', linewidth=1.5, alpha=0.7)
axes[0].set_ylabel('ΔEVE', fontsize=12)
axes[0].set_title('EVE Sensitivity (Long-Term Economic Value)',
                  fontsize=13, fontweight='bold')
axes[0].set_xticks(range(len(scenarios_short)))
axes[0].set_xticklabels(scenarios_short, fontsize=10)
axes[0].grid(True, alpha=0.3, axis='y')

# Right: ΔNII
delta_niis_combined = combined_summary['ΔNII'].values
colors_nii = ['#E63946' if x < 0 else '#06A77D' for x in delta_niis_combined]

bars_nii = axes[1].bar(range(len(scenarios_short)), delta_niis_combined,
                       color=colors_nii, alpha=0.8, edgecolor='black', linewidth=1.5)

# Highlight worst case
worst_idx_nii = np.argmin(delta_niis_combined)
bars_nii[worst_idx_nii].set_linewidth(3)
bars_nii[worst_idx_nii].set_edgecolor('darkred')

for i, (bar, val) in enumerate(zip(bars_nii, delta_niis_combined)):
    height = bar.get_height()
    label_y = height + (5 if height > 0 else -15)
    axes[1].text(bar.get_x() + bar.get_width()/2., label_y,
                f'{val:,.0f}',
                ha='center', va='bottom' if height > 0 else 'top',
                fontsize=10, fontweight='bold')

axes[1].axhline(y=0, color='black', linewidth=1.5, alpha=0.7)
axes[1].set_ylabel('ΔNII', fontsize=12)
axes[1].set_title('NII Sensitivity (12-Month Earnings Impact)',
                  fontsize=13, fontweight='bold')
axes[1].set_xticks(range(len(scenarios_short)))
axes[1].set_xticklabels(scenarios_short, fontsize=10)
axes[1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.show()

# %% [markdown]
# ### 7.3 Scatter Plot: EVE vs NII by Scenario

# %%
fig, ax = plt.subplots(figsize=(10, 8))

colors_scatter = ['#E63946', '#06A77D', '#F77F00', '#9B59B6']
markers = ['o', 's', 'D', '^']

for i, (scenario, color, marker) in enumerate(zip(combined_summary['Scenario'],
                                                   colors_scatter, markers)):
    eve_val = combined_summary.loc[i, 'ΔEVE']
    nii_val = combined_summary.loc[i, 'ΔNII']

    ax.scatter(eve_val, nii_val, s=300, color=color, marker=marker,
              edgecolor='black', linewidth=2, alpha=0.8,
              label=scenario.replace(': ', ':\n'), zorder=5)

    # Add scenario label
    ax.text(eve_val, nii_val + 5, f'S{i+1}',
           ha='center', va='bottom', fontsize=11, fontweight='bold')

# Add reference lines
ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
ax.axvline(x=0, color='black', linestyle='--', linewidth=1, alpha=0.5)

# Add quadrant labels
ax.text(0.02, 0.98, 'EVE Loss\nNII Gain',
       transform=ax.transAxes, fontsize=10, va='top', ha='left',
       bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
ax.text(0.98, 0.98, 'EVE Gain\nNII Gain',
       transform=ax.transAxes, fontsize=10, va='top', ha='right',
       bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))
ax.text(0.02, 0.02, 'EVE Loss\nNII Loss',
       transform=ax.transAxes, fontsize=10, va='bottom', ha='left',
       bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.3))
ax.text(0.98, 0.02, 'EVE Gain\nNII Loss',
       transform=ax.transAxes, fontsize=10, va='bottom', ha='right',
       bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.3))

ax.set_xlabel('ΔEVE (Economic Value Change)', fontsize=12)
ax.set_ylabel('ΔNII (Earnings Change)', fontsize=12)
ax.set_title('EVE vs NII Sensitivity - Scenario Comparison',
            fontsize=14, fontweight='bold')
ax.legend(loc='upper left', fontsize=9, framealpha=0.9)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 8. IRRBB Summary Report

# %%
print("\n" + "="*80)
print("PHASE 4 SUMMARY - NII SENSITIVITY AND COMBINED IRRBB ANALYSIS")
print("="*80)

print("\n1. NII CALCULATION SCOPE")
print("-" * 80)
print(f"   Buckets included:              {len(nii_buckets)} (repricing within 1 year)")
print(f"   Total CF in NII horizon:       {total_nii_cf:,.2f}  ({nii_cf_pct:.2f}% of total)")
print(f"   CF beyond 1Y (not in NII):     {total_cf - total_nii_cf:,.2f}  ({100-nii_cf_pct:.2f}% of total)")

print("\n2. NII SENSITIVITY RESULTS")
print("-" * 80)
for _, row in nii_summary.iterrows():
    print(f"   {row['Scenario']:35s}  ΔNII: {row['ΔNII']:+10,.2f}")

print("\n3. WORST CASE SCENARIOS")
print("-" * 80)
print(f"   EVE Worst Case:    {combined_summary.loc[worst_eve_idx, 'Scenario']}")
print(f"      ΔEVE:           {combined_summary.loc[worst_eve_idx, 'ΔEVE']:,.2f}  ({combined_summary.loc[worst_eve_idx, 'ΔEVE_%']:+.2f}%)")
print(f"\n   NII Worst Case:    {combined_summary.loc[worst_nii_idx, 'Scenario']}")
print(f"      ΔNII:           {combined_summary.loc[worst_nii_idx, 'ΔNII']:,.2f}")

print("\n4. BINDING CONSTRAINT ANALYSIS")
print("-" * 80)
if worst_eve_idx == worst_nii_idx:
    print(f"   ✓ Same scenario ({combined_summary.loc[worst_eve_idx, 'Scenario']}) binds for BOTH EVE and NII")
    print(f"   → Consistent risk profile across time horizons")
else:
    print(f"   ⚠ Different scenarios bind:")
    print(f"      EVE: {combined_summary.loc[worst_eve_idx, 'Scenario']}")
    print(f"      NII: {combined_summary.loc[worst_nii_idx, 'Scenario']}")
    print(f"   → Bank faces DIFFERENT risks in short-term vs long-term")
    print(f"   → Must manage BOTH constraints simultaneously")

print("\n5. KEY INSIGHTS")
print("-" * 80)
# Analyze position
if combined_summary['ΔEVE'].mean() < 0 and combined_summary['ΔNII'].mean() < 0:
    print("   - Portfolio is VULNERABLE to rate increases on average")
    print("   - Asset-sensitive / short duration position")
elif combined_summary['ΔEVE'].mean() > 0 and combined_summary['ΔNII'].mean() > 0:
    print("   - Portfolio is VULNERABLE to rate decreases on average")
    print("   - Liability-sensitive / long duration position")
else:
    print("   - Mixed sensitivity depending on scenario")
    print("   - Complex risk profile requiring detailed scenario analysis")

# Check O/N bucket concentration
on_bucket_pct = repricing_profile[repricing_profile['Bucket']=='O/N']['CF_Percent'].values[0]
if on_bucket_pct > 50:
    print(f"   - HIGH O/N concentration ({on_bucket_pct:.1f}%) → significant repricing risk")
else:
    print(f"   - Moderate O/N concentration ({on_bucket_pct:.1f}%)")

print("\n6. BASEL IRRBB REPORTING")
print("-" * 80)
print(f"   Worst ΔEVE:              {combined_summary.loc[worst_eve_idx, 'ΔEVE']:,.2f}")
print(f"   Worst ΔEVE %:            {abs(combined_summary.loc[worst_eve_idx, 'ΔEVE_%']):.2f}%")
print(f"   Basel outlier threshold: 15% of Tier 1 Capital")
print(f"   Status: Compare ΔEVE to 15% of Tier 1 Capital to determine outlier status")
print(f"\n   Worst ΔNII:              {combined_summary.loc[worst_nii_idx, 'ΔNII']:,.2f}")
print(f"   Impact: 12-month earnings reduction")

print("\n7. NEXT STEPS (PHASE 5 - BONUS ANALYSIS)")
print("-" * 80)
print("   - Sensitivity to core deposit ratio assumptions")
print("   - Pass-through rate analysis")
print("   - Backtesting decay model predictions")
print("   - Monte Carlo simulation of deposit paths")
print("   - Stress-adjusted decay scenarios")

print("\n" + "="*80)
print("PHASE 4 COMPLETE - CORE IRRBB ANALYSIS FINISHED")
print("="*80)

# %%
# Save results
nii_summary.to_csv('nii_sensitivity_summary.csv', index=False)
combined_summary.to_csv('combined_eve_nii_summary.csv', index=False)
nii_buckets.to_csv('nii_buckets_detailed.csv', index=False)

# Create final IRRBB report
irrbb_report = pd.DataFrame({
    'Metric': [
        'Total Balance',
        'Core Deposits',
        'Non-Core Deposits',
        'Weighted Avg Maturity',
        'EVE Base',
        'Worst EVE Scenario',
        'Worst ΔEVE',
        'Worst ΔEVE %',
        'Worst NII Scenario',
        'Worst ΔNII',
        'Binding Constraint'
    ],
    'Value': [
        f"{repricing_profile['Total_CF'].sum():,.2f}",
        f"{repricing_profile['Core_CF'].sum():,.2f}",
        f"{repricing_profile['Non_Core_CF'].sum():,.2f}",
        f"{(repricing_profile['Total_CF'] * repricing_profile['Midpoint_Years']).sum() / repricing_profile['Total_CF'].sum():.3f} years",
        f"{eve_summary[eve_summary['Scenario']=='Base Case']['EVE'].values[0]:,.2f}",
        combined_summary.loc[worst_eve_idx, 'Scenario'],
        f"{combined_summary.loc[worst_eve_idx, 'ΔEVE']:,.2f}",
        f"{combined_summary.loc[worst_eve_idx, 'ΔEVE_%']:+.2f}%",
        combined_summary.loc[worst_nii_idx, 'Scenario'],
        f"{combined_summary.loc[worst_nii_idx, 'ΔNII']:,.2f}",
        'Same' if worst_eve_idx == worst_nii_idx else 'Different'
    ]
})
irrbb_report.to_csv('irrbb_final_report.csv', index=False)

print("\nData saved:")
print("- nii_sensitivity_summary.csv (NII results)")
print("- combined_eve_nii_summary.csv (combined EVE + NII)")
print("- nii_buckets_detailed.csv (bucket-level NII details)")
print("- irrbb_final_report.csv (executive summary)")

# %%
