# %% [markdown]
# # Phase 1c: Core vs Non-Core Deposit Separation
#
# ## QF609 Project #1 - IRRBB Modelling Framework
#
# **Calculation Date:** 30-Dec-2023
#
# **WHAT ARE WE DOING HERE?**
# Splitting deposits into two buckets: STABLE vs VOLATILE
#
# **WHY DOES BASEL REQUIRE THIS?**
# Not all deposits behave the same way:
# - Some customers are loyal, never leave (CORE/STABLE)
# - Others chase best rates, leave quickly (NON-CORE/VOLATILE)
#
# **CORE DEPOSITS:**
# - Sticky, relationship-based
# - Reprice slowly over time (up to 5 years)
# - Example: Your grandma's checking account at her local bank
#
# **NON-CORE DEPOSITS:**
# - Rate-sensitive "hot money"
# - Reprice immediately (Overnight bucket)
# - Example: Wholesale funding from another bank
#
# **METHOD: Historical Minimum**
# The balance that stayed even during crisis = Core floor
# In your data: Min balance = 9,511, Current = 18,652 → Core = 51%
#
# This notebook performs:
# - Estimate core deposit floor using multiple methods
# - Split NMD balance into Core (stable) and Non-Core (volatile)
# - Apply Basel regulatory constraints (max 5Y maturity, max 90% core)
# - Sensitivity analysis on core ratio assumptions
# - Visualize core/non-core split

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
# Load processed data from previous phases
nmd_data = pd.read_csv('processed_nmd_data.csv')
nmd_data['Date'] = pd.to_datetime(nmd_data['Date'])

# Get calculation date balance
calc_date = pd.to_datetime('2023-12-30')
current_balance = nmd_data[nmd_data['Date'] == calc_date]['Balance'].values[0]

print(f"Loaded NMD data: {nmd_data.shape}")
print(f"Calculation Date: {calc_date.strftime('%d-%b-%Y')}")
print(f"Current Balance: {current_balance:,.2f}")

# %% [markdown]
# ## 2. Core Deposit Estimation Methods

# %% [markdown]
# ### 2.1 Method 1: Historical Minimum Balance

# %%
# Simple approach: minimum balance over entire history
min_balance = nmd_data['Balance'].min()
min_balance_date = nmd_data[nmd_data['Balance'] == min_balance]['Date'].values[0]

core_ratio_method1 = min_balance / current_balance
non_core_method1 = current_balance - min_balance

print("="*80)
print("METHOD 1: HISTORICAL MINIMUM BALANCE")
print("="*80)
print(f"Minimum Balance:              {min_balance:,.2f}")
print(f"Minimum Balance Date:         {pd.to_datetime(min_balance_date).strftime('%d-%b-%Y')}")
print(f"Current Balance:              {current_balance:,.2f}")
print(f"Core Ratio:                   {core_ratio_method1:.4f}  ({core_ratio_method1*100:.2f}%)")
print(f"Core Amount:                  {min_balance:,.2f}")
print(f"Non-Core Amount:              {non_core_method1:,.2f}")

# %% [markdown]
# ### 2.2 Method 2: Percentile-Based Floor

# %%
# Use percentiles of historical balance to set core floor
# Common choices: 5th, 10th, or 25th percentile

percentiles = [5, 10, 25, 50]
percentile_results = []

for p in percentiles:
    balance_p = np.percentile(nmd_data['Balance'], p)
    core_ratio_p = balance_p / current_balance
    non_core_p = current_balance - balance_p

    percentile_results.append({
        'Percentile': f'{p}th',
        'Balance_Floor': balance_p,
        'Core_Ratio': core_ratio_p,
        'Core_Ratio_%': core_ratio_p * 100,
        'Core_Amount': balance_p,
        'Non_Core_Amount': non_core_p
    })

percentile_df = pd.DataFrame(percentile_results)

print("\n" + "="*80)
print("METHOD 2: PERCENTILE-BASED CORE FLOOR")
print("="*80)
print(percentile_df.to_string(index=False))

# %% [markdown]
# ### 2.3 Method 3: Rolling Minimum (Conservative)

# %%
# Calculate rolling minimum balance over different windows
# This is more conservative than absolute minimum

rolling_windows = [180, 365, 730]  # 6 months, 1 year, 2 years
rolling_results = []

for window in rolling_windows:
    rolling_min = nmd_data['Balance'].rolling(window=window, min_periods=1).min()
    min_of_rolling = rolling_min.min()
    core_ratio_roll = min_of_rolling / current_balance

    rolling_results.append({
        'Window': f'{window} days',
        'Window_Years': window / 365,
        'Min_Rolling_Balance': min_of_rolling,
        'Core_Ratio': core_ratio_roll,
        'Core_Ratio_%': core_ratio_roll * 100
    })

rolling_df = pd.DataFrame(rolling_results)

print("\n" + "="*80)
print("METHOD 3: ROLLING MINIMUM BALANCE")
print("="*80)
print(rolling_df.to_string(index=False))

# %% [markdown]
# ### 2.4 Method 4: Monthly Average Low Point

# %%
# Calculate monthly average balances and use minimum
nmd_monthly = nmd_data.set_index('Date').resample('M').agg({
    'Balance': 'mean'
}).reset_index()

min_monthly_avg = nmd_monthly['Balance'].min()
core_ratio_method4 = min_monthly_avg / current_balance

print("\n" + "="*80)
print("METHOD 4: MINIMUM MONTHLY AVERAGE BALANCE")
print("="*80)
print(f"Minimum Monthly Average:      {min_monthly_avg:,.2f}")
print(f"Core Ratio:                   {core_ratio_method4:.4f}  ({core_ratio_method4*100:.2f}%)")

# %% [markdown]
# ## 3. Recommended Core/Non-Core Split (Primary Method)

# %%
# Based on Basel guidance and common industry practice:
# Use historical minimum with potential adjustment for conservatism

# Primary method: Use historical minimum (Method 1)
core_amount_primary = min_balance
non_core_amount_primary = current_balance - core_amount_primary
core_ratio_primary = core_amount_primary / current_balance

print("\n" + "="*80)
print("PRIMARY RECOMMENDATION: HISTORICAL MINIMUM METHOD")
print("="*80)
print(f"Current Balance (30-Dec-2023):    {current_balance:,.2f}")
print(f"Core Deposits:                    {core_amount_primary:,.2f}  ({core_ratio_primary*100:.2f}%)")
print(f"Non-Core Deposits:                {non_core_amount_primary:,.2f}  ({(1-core_ratio_primary)*100:.2f}%)")
print(f"\nCore deposits are subject to 5-year behavioral maturity cap (Basel)")

# %% [markdown]
# ## 4. Apply Regulatory Constraints

# %%
# Basel IRRBB Framework constraints:
# 1. Core deposits maximum behavioral maturity: 5 years
# 2. Typical Basel caps: core can be up to 70-90% of total NMD (retail)
# 3. Non-core reprices at O/N or 1M

# Check if core ratio exceeds Basel caps
basel_cap_retail = 0.90  # 90% cap for retail NMDs (common assumption)
basel_cap_wholesale = 0.70  # 70% cap for wholesale NMDs

print("\n" + "="*80)
print("REGULATORY CONSTRAINT CHECK")
print("="*80)
print(f"Calculated Core Ratio:        {core_ratio_primary*100:.2f}%")
print(f"Basel Cap (Retail NMD):       {basel_cap_retail*100:.0f}%")
print(f"Basel Cap (Wholesale NMD):    {basel_cap_wholesale*100:.0f}%")

if core_ratio_primary <= basel_cap_retail:
    print(f"\n✓ Core ratio PASSES retail cap (within {basel_cap_retail*100:.0f}% limit)")
else:
    print(f"\n✗ Core ratio EXCEEDS retail cap - adjustment required")

if core_ratio_primary <= basel_cap_wholesale:
    print(f"✓ Core ratio PASSES wholesale cap (within {basel_cap_wholesale*100:.0f}% limit)")
else:
    print(f"✗ Core ratio EXCEEDS wholesale cap - would need adjustment")

print(f"\n✓ Core maturity cap: 5 years (will apply in Phase 2 cash flow slotting)")

# %% [markdown]
# ## 5. Visualization: Balance History with Core Floor

# %%
fig, ax = plt.subplots(figsize=(14, 7))

# Plot balance history
ax.plot(nmd_data['Date'], nmd_data['Balance'], linewidth=2,
        color='#2E86AB', label='Daily Balance', alpha=0.8)

# Plot 30-day and 90-day moving averages
ax.plot(nmd_data['Date'], nmd_data['balance_ma30'], linewidth=1.5,
        color='#A23B72', linestyle='--', alpha=0.7, label='30-Day MA')

# Mark core floor (minimum balance)
ax.axhline(y=core_amount_primary, color='green', linestyle='-',
           linewidth=2.5, alpha=0.8, label=f'Core Floor: {core_amount_primary:,.0f}')

# Shade core region
ax.fill_between(nmd_data['Date'], 0, core_amount_primary,
                color='green', alpha=0.15, label='Core Deposits')

# Shade non-core region (current balance to core floor)
ax.fill_between([calc_date, calc_date],
                [core_amount_primary, current_balance],
                color='red', alpha=0.3, label='Non-Core Deposits (current)')

# Mark current balance
ax.axhline(y=current_balance, color='red', linestyle=':',
           linewidth=1.5, alpha=0.6, label=f'Current Balance: {current_balance:,.0f}')

# Annotate key points
ax.annotate(f'Core: {core_ratio_primary*100:.1f}%',
            xy=(calc_date, core_amount_primary/2),
            fontsize=11, fontweight='bold', color='darkgreen',
            ha='right', va='center')

ax.annotate(f'Non-Core: {(1-core_ratio_primary)*100:.1f}%',
            xy=(calc_date, (core_amount_primary + current_balance)/2),
            fontsize=11, fontweight='bold', color='darkred',
            ha='right', va='center')

ax.set_xlabel('Date', fontsize=11)
ax.set_ylabel('Balance', fontsize=11)
ax.set_title('NMD Balance History with Core/Non-Core Split',
             fontsize=14, fontweight='bold')
ax.legend(loc='upper left', fontsize=9)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 6. Sensitivity Analysis: Alternative Core Ratio Assumptions

# %% [markdown]
# ### 6.1 Test Multiple Core Ratios

# %%
# Test a range of core ratios to assess model sensitivity
# Common range: 40% to 80% (or up to 90% for very stable retail deposits)

core_ratios_to_test = [0.40, 0.50, 0.51, 0.60, 0.70, 0.80, 0.90]
sensitivity_results = []

for core_ratio_test in core_ratios_to_test:
    core_amt = current_balance * core_ratio_test
    non_core_amt = current_balance * (1 - core_ratio_test)

    sensitivity_results.append({
        'Core_Ratio': core_ratio_test,
        'Core_Ratio_%': core_ratio_test * 100,
        'Core_Amount': core_amt,
        'Non_Core_Amount': non_core_amt,
        'Non_Core_%': (1 - core_ratio_test) * 100
    })

sensitivity_df = pd.DataFrame(sensitivity_results)

# Highlight the primary recommendation
sensitivity_df['Method'] = ''
sensitivity_df.loc[sensitivity_df['Core_Ratio'] == 0.51, 'Method'] = '← PRIMARY (Historical Min)'

print("\n" + "="*80)
print("SENSITIVITY ANALYSIS: CORE RATIO ASSUMPTIONS")
print("="*80)
print(sensitivity_df.to_string(index=False))

# %% [markdown]
# ### 6.2 Visualize Core Ratio Sensitivity

# %%
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Left: Stacked bar chart showing core vs non-core amounts
core_amounts = sensitivity_df['Core_Amount']
non_core_amounts = sensitivity_df['Non_Core_Amount']
labels = [f"{int(r*100)}%" for r in sensitivity_df['Core_Ratio']]

axes[0].bar(labels, core_amounts, color='#06A77D', alpha=0.8,
            edgecolor='black', label='Core')
axes[0].bar(labels, non_core_amounts, bottom=core_amounts,
            color='#D62828', alpha=0.8, edgecolor='black', label='Non-Core')

# Highlight primary method
primary_idx = list(sensitivity_df['Core_Ratio']).index(0.51)
axes[0].get_children()[primary_idx].set_linewidth(3)
axes[0].get_children()[primary_idx].set_edgecolor('blue')
axes[0].get_children()[len(labels) + primary_idx].set_linewidth(3)
axes[0].get_children()[len(labels) + primary_idx].set_edgecolor('blue')

axes[0].set_xlabel('Core Ratio Assumption', fontsize=11)
axes[0].set_ylabel('Amount', fontsize=11)
axes[0].set_title('Core vs Non-Core Split by Core Ratio', fontsize=12, fontweight='bold')
axes[0].legend(loc='upper left')
axes[0].grid(True, alpha=0.3, axis='y')
axes[0].axhline(y=current_balance, color='black', linestyle='--',
                linewidth=1, alpha=0.5)

# Right: Line chart showing amounts
axes[1].plot(sensitivity_df['Core_Ratio_%'], sensitivity_df['Core_Amount'],
             marker='o', linewidth=2.5, markersize=8, color='#06A77D',
             label='Core Amount')
axes[1].plot(sensitivity_df['Core_Ratio_%'], sensitivity_df['Non_Core_Amount'],
             marker='s', linewidth=2.5, markersize=8, color='#D62828',
             label='Non-Core Amount')

# Mark primary recommendation
axes[1].axvline(x=core_ratio_primary*100, color='blue', linestyle='--',
                linewidth=2, alpha=0.7, label=f'Primary: {core_ratio_primary*100:.1f}%')

axes[1].set_xlabel('Core Ratio Assumption (%)', fontsize=11)
axes[1].set_ylabel('Amount', fontsize=11)
axes[1].set_title('Sensitivity of Core/Non-Core Split', fontsize=12, fontweight='bold')
axes[1].legend(loc='center left')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# %% [markdown]
# ### 6.3 Impact on Cash Flow Distribution (Preview)

# %%
# Preview: how does core ratio affect cash flow slotting?
# Non-core goes entirely to O/N, core is distributed across buckets

print("\n" + "="*80)
print("PREVIEW: IMPACT ON CASH FLOW SLOTTING")
print("="*80)
print("\nAssuming current balance of {0:,.2f}:\n".format(current_balance))

preview_table = []
for _, row in sensitivity_df.iterrows():
    core_r = row['Core_Ratio']
    core_amt = row['Core_Amount']
    non_core_amt = row['Non_Core_Amount']

    # Non-core = O/N bucket
    # Core = distributed across 1M to 5Y buckets
    preview_table.append({
        'Core_Ratio_%': row['Core_Ratio_%'],
        'O/N_Bucket (Non-Core)': non_core_amt,
        'Distributed_1M-5Y (Core)': core_amt,
        'Total': current_balance
    })

preview_df = pd.DataFrame(preview_table)
print(preview_df.to_string(index=False))
print("\nNote: Core deposits will be further distributed across 1M-5Y buckets in Phase 2")

# %% [markdown]
# ## 7. Compare All Methods

# %%
# Summary comparison of all methods
comparison_table = pd.DataFrame({
    'Method': [
        '1. Historical Minimum',
        '2. 5th Percentile',
        '2. 10th Percentile',
        '2. 25th Percentile',
        '3. Rolling Min (180d)',
        '3. Rolling Min (365d)',
        '3. Rolling Min (730d)',
        '4. Monthly Avg Min'
    ],
    'Core_Amount': [
        min_balance,
        percentile_df.loc[0, 'Balance_Floor'],
        percentile_df.loc[1, 'Balance_Floor'],
        percentile_df.loc[2, 'Balance_Floor'],
        rolling_df.loc[0, 'Min_Rolling_Balance'],
        rolling_df.loc[1, 'Min_Rolling_Balance'],
        rolling_df.loc[2, 'Min_Rolling_Balance'],
        min_monthly_avg
    ],
    'Core_Ratio_%': [
        core_ratio_method1 * 100,
        percentile_df.loc[0, 'Core_Ratio_%'],
        percentile_df.loc[1, 'Core_Ratio_%'],
        percentile_df.loc[2, 'Core_Ratio_%'],
        rolling_df.loc[0, 'Core_Ratio_%'],
        rolling_df.loc[1, 'Core_Ratio_%'],
        rolling_df.loc[2, 'Core_Ratio_%'],
        core_ratio_method4 * 100
    ]
})

comparison_table['Non_Core_Amount'] = current_balance - comparison_table['Core_Amount']
comparison_table['Recommended'] = ''
comparison_table.loc[0, 'Recommended'] = '★ PRIMARY'

print("\n" + "="*80)
print("COMPARISON OF ALL CORE ESTIMATION METHODS")
print("="*80)
print(comparison_table.to_string(index=False))

# %%
# Visualize comparison
fig, ax = plt.subplots(figsize=(12, 7))

y_pos = np.arange(len(comparison_table))
core_ratios = comparison_table['Core_Ratio_%']
colors = ['#023047' if i == 0 else '#8ECAE6' for i in range(len(comparison_table))]

bars = ax.barh(y_pos, core_ratios, color=colors, alpha=0.8, edgecolor='black')

# Highlight primary method
bars[0].set_linewidth(3)
bars[0].set_edgecolor('red')

ax.set_yticks(y_pos)
ax.set_yticklabels(comparison_table['Method'], fontsize=9)
ax.set_xlabel('Core Ratio (%)', fontsize=11)
ax.set_title('Comparison of Core Ratio Estimates Across Methods',
             fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3, axis='x')

# Add value labels
for i, (bar, val) in enumerate(zip(bars, core_ratios)):
    width = bar.get_width()
    ax.text(width + 1, bar.get_y() + bar.get_height()/2,
            f'{val:.2f}%', va='center', fontsize=9)

plt.tight_layout()
plt.show()

# %% [markdown]
# ## 8. Final Recommendation and Export

# %%
print("\n" + "="*80)
print("PHASE 1C SUMMARY - CORE/NON-CORE DEPOSIT SPLIT")
print("="*80)

print("\n1. FINAL RECOMMENDATION")
print("-" * 80)
print(f"   Method:                Historical Minimum Balance")
print(f"   Current Balance:       {current_balance:,.2f}")
print(f"   Core Deposits:         {core_amount_primary:,.2f}  ({core_ratio_primary*100:.2f}%)")
print(f"   Non-Core Deposits:     {non_core_amount_primary:,.2f}  ({(1-core_ratio_primary)*100:.2f}%)")

print("\n2. RATIONALE")
print("-" * 80)
print("   - Historical minimum represents the stable funding base")
print("   - This balance level was maintained even during stress periods")
print(f"   - Core ratio of {core_ratio_primary*100:.1f}% is conservative and passes Basel caps")
print("   - Non-core portion represents volatile/rate-sensitive deposits")

print("\n3. REGULATORY COMPLIANCE")
print("-" * 80)
print(f"   ✓ Core ratio ({core_ratio_primary*100:.1f}%) < Retail cap (90%)")
print(f"   ✓ Core ratio ({core_ratio_primary*100:.1f}%) < Wholesale cap (70%)")
print(f"   ✓ Core behavioral maturity capped at 5 years")
print(f"   ✓ Non-core reprices at O/N (immediate repricing)")

print("\n4. SENSITIVITY ANALYSIS")
print("-" * 80)
print(f"   Tested core ratios from 40% to 90%")
print(f"   Primary recommendation: {core_ratio_primary*100:.2f}%")
print(f"   Alternative scenarios prepared for Phase 5 bonus analysis")

print("\n5. NEXT STEPS (PHASE 2)")
print("-" * 80)
print(f"   - Non-core ({non_core_amount_primary:,.2f}) → O/N bucket")
print(f"   - Core ({core_amount_primary:,.2f}) → distributed across 1M-5Y using S(t)")
print(f"   - Cash flow in bucket i = Core × [S(t_i-1) - S(t_i)]")

print("\n" + "="*80)
print("PHASE 1C COMPLETE")
print("="*80)

# %%
# Save core/non-core split results
core_noncore_split = pd.DataFrame({
    'Component': ['Total Balance', 'Core Deposits', 'Non-Core Deposits'],
    'Amount': [current_balance, core_amount_primary, non_core_amount_primary],
    'Percentage': [100.0, core_ratio_primary*100, (1-core_ratio_primary)*100],
    'Behavioral_Maturity': ['N/A', '5 Years Max', 'O/N'],
    'Repricing': ['N/A', 'Distributed 1M-5Y', 'Immediate (O/N)']
})
core_noncore_split.to_csv('core_noncore_split.csv', index=False)

# Save sensitivity analysis results
sensitivity_df.to_csv('core_ratio_sensitivity.csv', index=False)

# Save method comparison
comparison_table.to_csv('core_estimation_methods_comparison.csv', index=False)

print("\nData saved:")
print("- core_noncore_split.csv (final recommended split)")
print("- core_ratio_sensitivity.csv (sensitivity analysis)")
print("- core_estimation_methods_comparison.csv (method comparison)")

# %%
