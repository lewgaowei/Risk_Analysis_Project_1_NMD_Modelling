# %% [markdown]
# # Phase 2: Cash Flow Slotting (Basel Repricing Profile)
#
# ## QF609 Project #1 - IRRBB Modelling Framework
#
# **Calculation Date:** 30-Dec-2023
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
# Load data from previous phases
core_noncore = pd.read_csv('core_noncore_split.csv')
survival_df = pd.read_csv('survival_function_table.csv')
survival_full = pd.read_csv('survival_curve_full.csv')

# Extract key values
current_balance = core_noncore[core_noncore['Component'] == 'Total Balance']['Amount'].values[0]
core_amount = core_noncore[core_noncore['Component'] == 'Core Deposits']['Amount'].values[0]
non_core_amount = core_noncore[core_noncore['Component'] == 'Non-Core Deposits']['Amount'].values[0]

print(f"Current Balance:     {current_balance:,.2f}")
print(f"Core Deposits:       {core_amount:,.2f}  ({core_amount/current_balance*100:.2f}%)")
print(f"Non-Core Deposits:   {non_core_amount:,.2f}  ({non_core_amount/current_balance*100:.2f}%)")
print(f"\nSurvival function loaded with {len(survival_df)} tenors")

# %% [markdown]
# ## 2. Define Time Buckets

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
# ## 3. Slot Non-Core Deposits

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
# ## 4. Slot Core Deposits Using Survival Function

# %% [markdown]
# ### 4.1 Calculate Cash Flow in Each Bucket
#
# For core deposits distributed across buckets:
# - Cash flow in bucket i = Core × [S(t_{i-1}) - S(t_i)]
# - Where t_i is the end time of bucket i in days

# %%
# Helper function to get survival probability at a given day
def get_survival(days):
    """Get survival probability at specified day from full survival curve"""
    if days >= len(survival_full):
        return survival_full.iloc[-1]['S(t)']
    return survival_full.iloc[days]['S(t)']

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
    else:
        # Other buckets get marginal core decay
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
print("\n" + "="*80)
print("CASH FLOW SLOTTING RESULTS")
print("="*80)
print(f"Total Cash Flows:        {total_cf:,.2f}")
print(f"Current Balance:         {current_balance:,.2f}")
print(f"Difference:              {abs(total_cf - current_balance):,.6f}")
print(f"Match: {'✓ PASS' if abs(total_cf - current_balance) < 0.01 else '✗ FAIL'}")

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

# Weighted average maturity of core deposits
weighted_maturity = (repricing_profile['Total_CF'] * repricing_profile['Midpoint_Years']).sum() / repricing_profile['Total_CF'].sum()
print(f"\nWeighted Average Maturity:            {weighted_maturity:.3f} years")

# %% [markdown]
# ## 5. Visualizations

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
plt.show()

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
plt.show()

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
plt.show()

# %% [markdown]
# ### 5.4 Survival Function vs Cumulative Repricing

# %%
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Left: Survival function with bucket boundaries
years_full = survival_full['Years'].values
s_full = survival_full['S(t)'].values

axes[0].plot(years_full, s_full * 100, linewidth=3, color='#023047',
             label='Survival Function S(t)')

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
plt.show()

# %% [markdown]
# ## 6. Detailed Analysis by Bucket

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
# ## 7. Summary and Export

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
print(f"\nSum of Total CF:                  {total_cf:,.2f}")
print(f"Current Balance:                  {current_balance:,.2f}")
print(f"Total Match: {'✓ PASS' if abs(total_cf - current_balance) < 0.01 else '✗ FAIL'}")

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
