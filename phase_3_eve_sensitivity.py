# %% [markdown]
# # Phase 3: EVE (Economic Value of Equity) Sensitivity
#
# ## QF609 Project #1 - IRRBB Modelling Framework
#
# **Calculation Date:** 30-Dec-2023
#
# This notebook performs:
# - Build discount factors from zero rate curve
# - Calculate base case EVE (present value of cash flows)
# - Apply 4 rate shock scenarios
# - Calculate ΔEVE for each scenario
# - Identify worst-case EVE impact
# - Visualize yield curves and sensitivity results

# %% [markdown]
# ## 1. Import Libraries and Load Data

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.interpolate import interp1d
import warnings

warnings.filterwarnings('ignore')

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10

print("Libraries imported successfully")

# %%
# Load data from previous phases
repricing_profile = pd.read_csv('repricing_profile.csv')
curve_data = pd.read_csv('processed_curve_data.csv')

print(f"Repricing Profile: {len(repricing_profile)} buckets")
print(f"Zero Rate Curve:   {len(curve_data)} tenors")
print(f"\nTotal Cash Flow:   {repricing_profile['Total_CF'].sum():,.2f}")

# Display curve data
print("\nBase Zero Rate Curve:")
print(curve_data[['Tenor', 'Tenor_Years', 'ZeroRate']])

# %% [markdown]
# ## 2. Discount Factor Construction

# %% [markdown]
# ### 2.1 Build Discount Factor Function
#
# Convert zero rates to discount factors:
# - Assuming annual compounding: DF(t) = 1 / (1 + r)^t
# - Use interpolation for intermediate tenors

# %%
# Create interpolation function for zero rates
# Using log-linear interpolation on discount factors (industry standard)

def build_discount_function(tenors_years, zero_rates):
    """
    Build interpolated discount factor function

    Args:
        tenors_years: array of tenors in years
        zero_rates: array of zero rates (decimal form)

    Returns:
        function that takes time in years and returns discount factor
    """
    # Calculate discount factors at known tenors
    discount_factors = 1 / (1 + zero_rates) ** tenors_years

    # Create log-linear interpolation on discount factors
    log_df = np.log(discount_factors)
    interp_func = interp1d(tenors_years, log_df, kind='linear',
                          bounds_error=False, fill_value='extrapolate')

    def get_discount_factor(t):
        """Get discount factor at time t (years)"""
        if isinstance(t, (list, np.ndarray)):
            return np.exp(interp_func(t))
        else:
            return np.exp(interp_func(t))

    return get_discount_factor

# Build base case discount function
df_base_func = build_discount_function(curve_data['Tenor_Years'].values,
                                       curve_data['ZeroRate'].values)

# Test discount function at bucket midpoints
print("="*80)
print("DISCOUNT FACTORS AT BUCKET MIDPOINTS (BASE CURVE)")
print("="*80)
for _, row in repricing_profile.iterrows():
    t = row['Midpoint_Years']
    df = df_base_func(t)
    print(f"{row['Bucket']:4s}  t={t:.4f}y  →  DF={df:.6f}")

# %% [markdown]
# ### 2.2 Create Discount Factor Table for Key Tenors

# %%
# Create comprehensive DF table
tenor_points = np.array([1/365, 1/12, 2/12, 3/12, 6/12, 9/12, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
df_table = pd.DataFrame({
    'Tenor_Years': tenor_points,
    'Zero_Rate': [df_base_func(t) for t in tenor_points]  # This is wrong, should get rate not DF
})

# Actually, let's interpolate the zero rates first, then calculate DF
rate_interp_func = interp1d(curve_data['Tenor_Years'].values,
                            curve_data['ZeroRate'].values,
                            kind='linear', bounds_error=False,
                            fill_value='extrapolate')

df_table = pd.DataFrame({
    'Tenor_Years': tenor_points,
    'Zero_Rate': rate_interp_func(tenor_points),
    'Discount_Factor': df_base_func(tenor_points)
})

print("\n" + "="*80)
print("BASE CASE DISCOUNT FACTORS")
print("="*80)
print(df_table.to_string(index=False))

# %% [markdown]
# ## 3. Base Case EVE Calculation

# %%
# Calculate base case EVE
# EVE = Σ CF(i) × DF(i)

repricing_profile['DF_Base'] = repricing_profile['Midpoint_Years'].apply(df_base_func)
repricing_profile['PV_Base'] = repricing_profile['Total_CF'] * repricing_profile['DF_Base']

eve_base = repricing_profile['PV_Base'].sum()

print("="*80)
print("BASE CASE EVE CALCULATION")
print("="*80)
print(f"\nTotal Cash Flow (undiscounted): {repricing_profile['Total_CF'].sum():,.2f}")
print(f"EVE Base (present value):       {eve_base:,.2f}")
print(f"Discount Effect:                {repricing_profile['Total_CF'].sum() - eve_base:,.2f}")
print(f"Average Discount:               {(eve_base / repricing_profile['Total_CF'].sum()):.4f}")

print("\nPresent Value by Bucket:")
print(repricing_profile[['Bucket', 'Total_CF', 'DF_Base', 'PV_Base']].to_string(index=False))

# %% [markdown]
# ## 4. Rate Shock Scenarios

# %% [markdown]
# ### 4.1 Define Shock Functions

# %%
def apply_parallel_shock(zero_rates, shock_bps):
    """
    Apply parallel shift to zero rates

    Args:
        zero_rates: original zero rates (decimal)
        shock_bps: shock in basis points (e.g., 200 for +200bps)

    Returns:
        shocked zero rates (with zero floor applied)
    """
    shock_decimal = shock_bps / 10000
    shocked_rates = zero_rates + shock_decimal
    return np.maximum(shocked_rates, 0)  # Zero floor

def apply_steepener_shock(tenors_years, zero_rates):
    """
    Apply short rate up shock (steepener)
    +200bps at shortest tenor, tapering linearly to 0bps at 10Y

    Formula: shock(t) = 200bps × max(1 - t/10, 0)
    """
    shocks = 200 * np.maximum(1 - tenors_years / 10, 0)
    shocked_rates = zero_rates + shocks / 10000
    return np.maximum(shocked_rates, 0)

def apply_flattener_shock(tenors_years, zero_rates):
    """
    Apply flattener shock
    Short end: +200bps at shortest, tapering to 0bps at 5Y
    Long end: 0bps at 5Y, tapering to -100bps at 10Y

    Formula:
    - For t <= 5Y: shock(t) = 200bps × (1 - t/5)
    - For t > 5Y: shock(t) = -100bps × (t-5)/5
    """
    shocks = np.zeros_like(tenors_years)

    for i, t in enumerate(tenors_years):
        if t <= 5:
            shocks[i] = 200 * (1 - t / 5)
        else:
            shocks[i] = -100 * (t - 5) / 5

    shocked_rates = zero_rates + shocks / 10000
    return np.maximum(shocked_rates, 0)

print("="*80)
print("RATE SHOCK SCENARIO DEFINITIONS")
print("="*80)
print("\nScenario 1: +200bps Parallel Shift")
print("  - All rates shift up by 200 basis points")
print("\nScenario 2: -200bps Parallel Shift")
print("  - All rates shift down by 200 basis points (with zero floor)")
print("\nScenario 3: Short Rate Up (Steepener)")
print("  - +200bps at shortest tenor")
print("  - Tapering linearly to 0bps at 10Y")
print("  - Formula: shock(t) = 200bps × max(1 - t/10, 0)")
print("\nScenario 4: Flattener")
print("  - +200bps at short end → 0bps at 5Y")
print("  - 0bps at 5Y → -100bps at 10Y")

# %% [markdown]
# ### 4.2 Generate Shocked Curves

# %%
# Apply shocks to base curve
tenors_years = curve_data['Tenor_Years'].values
base_rates = curve_data['ZeroRate'].values

# Scenario 1: +200bps parallel
rates_s1 = apply_parallel_shock(base_rates, 200)

# Scenario 2: -200bps parallel
rates_s2 = apply_parallel_shock(base_rates, -200)

# Scenario 3: Steepener
rates_s3 = apply_steepener_shock(tenors_years, base_rates)

# Scenario 4: Flattener
rates_s4 = apply_flattener_shock(tenors_years, base_rates)

# Create shocked curves table
shocked_curves = pd.DataFrame({
    'Tenor': curve_data['Tenor'],
    'Tenor_Years': tenors_years,
    'Base': base_rates * 10000,  # Convert to bps for display
    'S1_+200_Parallel': rates_s1 * 10000,
    'S2_-200_Parallel': rates_s2 * 10000,
    'S3_Steepener': rates_s3 * 10000,
    'S4_Flattener': rates_s4 * 10000
})

print("\n" + "="*80)
print("SHOCKED ZERO RATE CURVES (in basis points)")
print("="*80)
print(shocked_curves.to_string(index=False))

# %% [markdown]
# ### 4.3 Visualize Shocked Yield Curves

# %%
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

scenarios = [
    ('S1_+200_Parallel', '+200bps Parallel Shift', '#E63946', rates_s1),
    ('S2_-200_Parallel', '-200bps Parallel Shift', '#06A77D', rates_s2),
    ('S3_Steepener', 'Short Rate Up (Steepener)', '#F77F00', rates_s3),
    ('S4_Flattener', 'Flattener', '#9B59B6', rates_s4)
]

for idx, (name, title, color, shocked_rates) in enumerate(scenarios):
    ax = axes[idx // 2, idx % 2]

    # Plot base curve
    ax.plot(tenors_years, base_rates * 100, marker='o', linewidth=2.5,
            markersize=8, color='#023047', label='Base Curve', alpha=0.8)

    # Plot shocked curve
    ax.plot(tenors_years, shocked_rates * 100, marker='s', linewidth=2.5,
            markersize=8, color=color, label='Shocked Curve', alpha=0.8)

    # Calculate and show shock amounts
    shocks = (shocked_rates - base_rates) * 10000  # in bps

    ax.set_xlabel('Tenor (Years)', fontsize=11)
    ax.set_ylabel('Zero Rate (%)', fontsize=11)
    ax.set_title(f'Scenario {idx+1}: {title}', fontsize=12, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    # Add shock annotations for first and last tenor
    if len(shocks) > 0:
        ax.text(0.02, 0.98, f'Short shock: {shocks[0]:+.0f}bps',
                transform=ax.transAxes, fontsize=9,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax.text(0.02, 0.88, f'Long shock: {shocks[-1]:+.0f}bps',
                transform=ax.transAxes, fontsize=9,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.show()

# %% [markdown]
# ### 4.4 Combined Yield Curve Comparison

# %%
fig, ax = plt.subplots(figsize=(14, 8))

ax.plot(tenors_years, base_rates * 100, marker='o', linewidth=3,
        markersize=10, color='#023047', label='Base', alpha=0.9, zorder=5)
ax.plot(tenors_years, rates_s1 * 100, marker='s', linewidth=2,
        markersize=7, color='#E63946', label='S1: +200 Parallel', alpha=0.8)
ax.plot(tenors_years, rates_s2 * 100, marker='^', linewidth=2,
        markersize=7, color='#06A77D', label='S2: -200 Parallel', alpha=0.8)
ax.plot(tenors_years, rates_s3 * 100, marker='D', linewidth=2,
        markersize=7, color='#F77F00', label='S3: Steepener', alpha=0.8)
ax.plot(tenors_years, rates_s4 * 100, marker='v', linewidth=2,
        markersize=7, color='#9B59B6', label='S4: Flattener', alpha=0.8)

ax.set_xlabel('Tenor (Years)', fontsize=12)
ax.set_ylabel('Zero Rate (%)', fontsize=12)
ax.set_title('All Rate Shock Scenarios vs Base Curve', fontsize=14, fontweight='bold')
ax.legend(loc='best', fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 5. Calculate Shocked EVE for All Scenarios

# %%
# Build discount functions for each shocked scenario
df_s1_func = build_discount_function(tenors_years, rates_s1)
df_s2_func = build_discount_function(tenors_years, rates_s2)
df_s3_func = build_discount_function(tenors_years, rates_s3)
df_s4_func = build_discount_function(tenors_years, rates_s4)

# Calculate discount factors at bucket midpoints for each scenario
repricing_profile['DF_S1'] = repricing_profile['Midpoint_Years'].apply(df_s1_func)
repricing_profile['DF_S2'] = repricing_profile['Midpoint_Years'].apply(df_s2_func)
repricing_profile['DF_S3'] = repricing_profile['Midpoint_Years'].apply(df_s3_func)
repricing_profile['DF_S4'] = repricing_profile['Midpoint_Years'].apply(df_s4_func)

# Calculate present values
repricing_profile['PV_S1'] = repricing_profile['Total_CF'] * repricing_profile['DF_S1']
repricing_profile['PV_S2'] = repricing_profile['Total_CF'] * repricing_profile['DF_S2']
repricing_profile['PV_S3'] = repricing_profile['Total_CF'] * repricing_profile['DF_S3']
repricing_profile['PV_S4'] = repricing_profile['Total_CF'] * repricing_profile['DF_S4']

# Calculate EVE for each scenario
eve_s1 = repricing_profile['PV_S1'].sum()
eve_s2 = repricing_profile['PV_S2'].sum()
eve_s3 = repricing_profile['PV_S3'].sum()
eve_s4 = repricing_profile['PV_S4'].sum()

print("="*80)
print("EVE CALCULATIONS - ALL SCENARIOS")
print("="*80)
print(f"\nBase Case EVE:           {eve_base:,.2f}")
print(f"Scenario 1 EVE (+200):   {eve_s1:,.2f}")
print(f"Scenario 2 EVE (-200):   {eve_s2:,.2f}")
print(f"Scenario 3 EVE (Steep):  {eve_s3:,.2f}")
print(f"Scenario 4 EVE (Flat):   {eve_s4:,.2f}")

# %% [markdown]
# ## 6. Calculate ΔEVE (EVE Sensitivity)

# %%
# Calculate changes in EVE
delta_eve_s1 = eve_s1 - eve_base
delta_eve_s2 = eve_s2 - eve_base
delta_eve_s3 = eve_s3 - eve_base
delta_eve_s4 = eve_s4 - eve_base

# Calculate percentage changes
pct_delta_s1 = (delta_eve_s1 / eve_base) * 100
pct_delta_s2 = (delta_eve_s2 / eve_base) * 100
pct_delta_s3 = (delta_eve_s3 / eve_base) * 100
pct_delta_s4 = (delta_eve_s4 / eve_base) * 100

# Create EVE sensitivity summary table
eve_summary = pd.DataFrame({
    'Scenario': [
        'Base Case',
        'S1: +200bps Parallel',
        'S2: -200bps Parallel',
        'S3: Short Rate Up (Steepener)',
        'S4: Flattener'
    ],
    'EVE': [eve_base, eve_s1, eve_s2, eve_s3, eve_s4],
    'ΔEVE': [0, delta_eve_s1, delta_eve_s2, delta_eve_s3, delta_eve_s4],
    'ΔEVE_%': [0, pct_delta_s1, pct_delta_s2, pct_delta_s3, pct_delta_s4]
})

print("\n" + "="*80)
print("EVE SENSITIVITY SUMMARY")
print("="*80)
print(eve_summary.to_string(index=False))

# Identify worst case
worst_case_idx = eve_summary.iloc[1:]['ΔEVE'].idxmin()  # Exclude base case
worst_case_scenario = eve_summary.loc[worst_case_idx, 'Scenario']
worst_case_delta = eve_summary.loc[worst_case_idx, 'ΔEVE']
worst_case_pct = eve_summary.loc[worst_case_idx, 'ΔEVE_%']

print("\n" + "="*80)
print("WORST CASE EVE SCENARIO")
print("="*80)
print(f"Scenario:        {worst_case_scenario}")
print(f"ΔEVE:            {worst_case_delta:,.2f}  ({worst_case_pct:+.2f}%)")
print(f"EVE Loss:        {abs(worst_case_delta):,.2f}")

# %% [markdown]
# ## 7. Visualizations - EVE Sensitivity

# %% [markdown]
# ### 7.1 Bar Chart: ΔEVE by Scenario

# %%
fig, ax = plt.subplots(figsize=(12, 7))

scenarios_list = eve_summary['Scenario'].iloc[1:].values  # Exclude base case
delta_eves = eve_summary['ΔEVE'].iloc[1:].values
colors = ['#E63946' if x < 0 else '#06A77D' for x in delta_eves]

bars = ax.bar(range(len(scenarios_list)), delta_eves, color=colors,
              alpha=0.8, edgecolor='black', linewidth=1.5)

# Highlight worst case
worst_idx = np.argmin(delta_eves)
bars[worst_idx].set_linewidth(3)
bars[worst_idx].set_edgecolor('darkred')

# Add value labels
for i, (bar, val, pct) in enumerate(zip(bars, delta_eves,
                                        eve_summary['ΔEVE_%'].iloc[1:].values)):
    height = bar.get_height()
    label_y = height + (50 if height > 0 else -150)
    ax.text(bar.get_x() + bar.get_width()/2., label_y,
            f'{val:,.0f}\n({pct:+.2f}%)',
            ha='center', va='bottom' if height > 0 else 'top',
            fontsize=10, fontweight='bold')

ax.axhline(y=0, color='black', linewidth=1.5, alpha=0.7)
ax.set_ylabel('ΔEVE (Change in Economic Value)', fontsize=12)
ax.set_title('EVE Sensitivity Across Rate Shock Scenarios', fontsize=14, fontweight='bold')
ax.set_xticks(range(len(scenarios_list)))
ax.set_xticklabels(['S1: +200\nParallel', 'S2: -200\nParallel',
                    'S3: Short Up\n(Steepener)', 'S4: Flattener'],
                   fontsize=10)
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.show()

# %% [markdown]
# ### 7.2 Waterfall Chart: EVE Components by Bucket

# %%
# Show how each bucket contributes to base EVE
fig, ax = plt.subplots(figsize=(14, 7))

buckets_list = repricing_profile['Bucket'].values
pv_base_values = repricing_profile['PV_Base'].values

bars = ax.bar(range(len(buckets_list)), pv_base_values, color='#2A9D8F',
              alpha=0.8, edgecolor='black')

# Add value labels
for i, (bucket, pv) in enumerate(zip(buckets_list, pv_base_values)):
    if pv > 100:  # Only label significant buckets
        ax.text(i, pv + 50, f'{pv:,.0f}',
                ha='center', va='bottom', fontsize=9)

ax.set_xlabel('Time Bucket', fontsize=12)
ax.set_ylabel('Present Value (Base Case)', fontsize=12)
ax.set_title('EVE Composition by Time Bucket (Base Case)', fontsize=14, fontweight='bold')
ax.set_xticks(range(len(buckets_list)))
ax.set_xticklabels(buckets_list, fontsize=10)
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.show()

# %% [markdown]
# ### 7.3 Heatmap: ΔEVE by Bucket and Scenario

# %%
# Calculate change in PV for each bucket in each scenario
delta_pv_s1 = repricing_profile['PV_S1'] - repricing_profile['PV_Base']
delta_pv_s2 = repricing_profile['PV_S2'] - repricing_profile['PV_Base']
delta_pv_s3 = repricing_profile['PV_S3'] - repricing_profile['PV_Base']
delta_pv_s4 = repricing_profile['PV_S4'] - repricing_profile['PV_Base']

# Create heatmap data
heatmap_data = pd.DataFrame({
    'S1: +200 Parallel': delta_pv_s1.values,
    'S2: -200 Parallel': delta_pv_s2.values,
    'S3: Steepener': delta_pv_s3.values,
    'S4: Flattener': delta_pv_s4.values
}, index=buckets_list)

fig, ax = plt.subplots(figsize=(12, 8))

sns.heatmap(heatmap_data.T, annot=True, fmt='.0f', cmap='RdYlGn',
            center=0, linewidths=1, linecolor='black',
            cbar_kws={'label': 'ΔPV'},
            ax=ax)

ax.set_xlabel('Time Bucket', fontsize=12)
ax.set_ylabel('Scenario', fontsize=12)
ax.set_title('Change in Present Value by Bucket and Scenario (ΔPVΔPV)',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 8. Duration and Convexity Analysis (Bonus)

# %%
# Calculate effective duration and convexity
# Duration ≈ (EVE_down - EVE_up) / (2 × EVE_base × Δy)
# Convexity ≈ (EVE_down + EVE_up - 2×EVE_base) / (EVE_base × Δy²)

delta_y = 0.02  # 200bps = 0.02

effective_duration = (eve_s2 - eve_s1) / (2 * eve_base * delta_y)
effective_convexity = (eve_s2 + eve_s1 - 2*eve_base) / (eve_base * delta_y**2)

print("\n" + "="*80)
print("DURATION AND CONVEXITY ANALYSIS")
print("="*80)
print(f"Effective Duration:      {effective_duration:.4f} years")
print(f"Effective Convexity:     {effective_convexity:.4f}")
print(f"\nInterpretation:")
print(f"  - Duration of {effective_duration:.2f} years means:")
print(f"    EVE changes by ~{abs(effective_duration):.2f}% for 1% rate change")
print(f"  - {'Positive' if effective_convexity > 0 else 'Negative'} convexity: "
      f"beneficial price behavior under large rate changes")

# %% [markdown]
# ## 9. Summary and Export

# %%
print("\n" + "="*80)
print("PHASE 3 SUMMARY - EVE SENSITIVITY ANALYSIS")
print("="*80)

print("\n1. BASE CASE EVE")
print("-" * 80)
print(f"   Total Cash Flow (undiscounted):  {repricing_profile['Total_CF'].sum():,.2f}")
print(f"   Base Case EVE (present value):   {eve_base:,.2f}")
print(f"   Weighted Average Discount:       {(eve_base/repricing_profile['Total_CF'].sum()):.4f}")

print("\n2. EVE UNDER RATE SHOCKS")
print("-" * 80)
for _, row in eve_summary.iterrows():
    print(f"   {row['Scenario']:35s}  EVE: {row['EVE']:10,.2f}  ΔEVE: {row['ΔEVE']:+10,.2f} ({row['ΔEVE_%']:+6.2f}%)")

print("\n3. WORST CASE SCENARIO")
print("-" * 80)
print(f"   Scenario:      {worst_case_scenario}")
print(f"   EVE Loss:      {abs(worst_case_delta):,.2f}")
print(f"   % Loss:        {abs(worst_case_pct):.2f}%")
print(f"   This scenario represents the BINDING EVE risk measure")

print("\n4. RISK METRICS")
print("-" * 80)
print(f"   Effective Duration:    {effective_duration:.4f} years")
print(f"   Effective Convexity:   {effective_convexity:.4f}")

print("\n5. KEY INSIGHTS")
print("-" * 80)
if worst_case_scenario == 'S1: +200bps Parallel':
    print("   - Most vulnerable to RISING interest rates")
    print("   - Asset-sensitive position (short duration)")
elif worst_case_scenario == 'S2: -200bps Parallel':
    print("   - Most vulnerable to FALLING interest rates")
    print("   - Liability-sensitive position (long duration)")
elif 'Steepener' in worst_case_scenario:
    print("   - Most vulnerable to curve STEEPENING")
    print("   - Short-end rate increases hurt most")
else:
    print("   - Most vulnerable to curve FLATTENING")
    print("   - Long-end rate decreases hurt most")

print("\n6. BASEL IRRBB REPORTING")
print("-" * 80)
print(f"   ΔEVE / Base EVE:               {abs(worst_case_pct):.2f}%")
print(f"   Basel outlier threshold:       15% of Tier 1 Capital")
print(f"   Note: Compare to bank's Tier 1 Capital for compliance assessment")

print("\n7. NEXT STEPS (PHASE 4)")
print("-" * 80)
print("   - Calculate NII (Net Interest Income) sensitivity")
print("   - Focus on 12-month repricing horizon")
print("   - Identify NII worst-case scenario")
print("   - Compare EVE vs NII binding constraints")

print("\n" + "="*80)
print("PHASE 3 COMPLETE")
print("="*80)

# %%
# Save results
eve_summary.to_csv('eve_sensitivity_summary.csv', index=False)

# Save detailed bucket-level results
repricing_profile.to_csv('repricing_profile_with_eve.csv', index=False)

# Save shocked curves
shocked_curves.to_csv('shocked_yield_curves.csv', index=False)

# Save key metrics
key_metrics = pd.DataFrame({
    'Metric': ['Base_EVE', 'Worst_Case_Scenario', 'Worst_Case_ΔEVE',
               'Worst_Case_ΔEVE_%', 'Effective_Duration', 'Effective_Convexity'],
    'Value': [eve_base, worst_case_scenario, worst_case_delta,
              worst_case_pct, effective_duration, effective_convexity]
})
key_metrics.to_csv('eve_key_metrics.csv', index=False)

print("\nData saved:")
print("- eve_sensitivity_summary.csv (EVE results for all scenarios)")
print("- repricing_profile_with_eve.csv (detailed bucket-level EVE)")
print("- shocked_yield_curves.csv (all shocked curves)")
print("- eve_key_metrics.csv (duration, convexity, worst case)")

# %%
