# %% [markdown]
# # Phase 2: Cash Flow Slotting - Compare All Survival Models
#
# ## QF609 Project #1 - IRRBB Modelling Framework
#
# **Purpose:** Run Phase 2 allocation using ALL Phase 1b survival models and compare results

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import warnings

warnings.filterwarnings('ignore')

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (16, 10)
plt.rcParams['font.size'] = 10

print("="*80)
print("PHASE 2: CASH FLOW SLOTTING - ALL MODELS COMPARISON")
print("="*80)

# %%
# Load config
with open('config.json', 'r') as f:
    config = json.load(f)

current_balance = config['current_balance']
core_amount = config['core_amount']
non_core_amount = config['non_core_amount']

print(f"\nCore/Non-Core Split:")
print(f"  Current Balance:   {current_balance:,.2f}")
print(f"  Core:              {core_amount:,.2f}  ({core_amount/current_balance*100:.2f}%)")
print(f"  Non-Core:          {non_core_amount:,.2f}  ({non_core_amount/current_balance*100:.2f}%)")

# %%
# Load ALL models' survival curves from Excel
all_models_df = pd.read_excel('survival_models_all_results.xlsx', sheet_name='All_Survival_Curves')

print(f"\nLoaded survival curves for {len([c for c in all_models_df.columns if c.startswith('S(t)')])} models")
print(f"Columns: {[c for c in all_models_df.columns if c.startswith('S(t)')]}")

# %%
# Define time buckets (same as Phase 2)
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

# %%
# Function to calculate cash flow allocation for a given S(t) curve
def calculate_cashflow_allocation(survival_curve, core_amt, non_core_amt, buckets_df, model_name):
    """Calculate cash flow allocation for a survival model"""

    cashflow_slots = []

    for idx, row in buckets_df.iterrows():
        bucket_name = row['Bucket']
        start_days = row['Start_Days']
        end_days = row['End_Days']
        midpoint_years = row['Midpoint_Years']

        # Get survival probabilities
        s_start = survival_curve[start_days] if start_days < len(survival_curve) else survival_curve[-1]
        s_end = survival_curve[end_days] if end_days < len(survival_curve) else survival_curve[-1]

        # Calculate cash flow
        if bucket_name == 'O/N':
            core_cf = core_amt * (s_start - s_end)
            total_cf = non_core_amt + core_cf
        elif bucket_name == '5Y':
            # 5Y bucket gets marginal decay + residual beyond 5Y (Basel cap)
            marginal_decay = core_amt * (s_start - s_end)
            residual_beyond_5y = core_amt * s_end  # S(1825) remainder
            core_cf = marginal_decay + residual_beyond_5y
            total_cf = core_cf
        else:
            core_cf = core_amt * (s_start - s_end)
            total_cf = core_cf

        cashflow_slots.append({
            'Bucket': bucket_name,
            'Midpoint_Years': midpoint_years,
            'S(t_start)': s_start,
            'S(t_end)': s_end,
            'Marginal_Decay': s_start - s_end,
            'Core_CF': core_cf,
            'Non_Core_CF': non_core_amt if bucket_name == 'O/N' else 0,
            'Total_CF': total_cf,
            'Model': model_name
        })

    repricing_df = pd.DataFrame(cashflow_slots)
    repricing_df['CF_Percent'] = (repricing_df['Total_CF'] / (core_amt + non_core_amt)) * 100

    return repricing_df

# %%
# Run Phase 2 for ALL models
print("\n" + "="*80)
print("RUNNING PHASE 2 FOR ALL MODELS")
print("="*80)

all_results = []

# Get list of all S(t) columns (excluding RECOMMENDED which is duplicate)
survival_columns = [c for c in all_models_df.columns if c.startswith('S(t)_') and c != 'S(t)_RECOMMENDED']

for col in survival_columns:
    model_name = col.replace('S(t)_', '')
    print(f"\nProcessing model: {model_name}")

    survival_curve = all_models_df[col].values

    # Calculate allocation
    result_df = calculate_cashflow_allocation(
        survival_curve,
        core_amount,
        non_core_amount,
        buckets_df,
        model_name
    )

    all_results.append(result_df)

# Combine all results
all_allocations = pd.concat(all_results, ignore_index=True)

print(f"\nProcessed {len(survival_columns)} models")
print(f"Total allocation records: {len(all_allocations)}")

# %%
# Create comparison table for KEY TENORS
print("\n" + "="*80)
print("KEY TENORS COMPARISON ACROSS ALL MODELS")
print("="*80)

key_tenors_list = ['O/N', '1M', '3M', '6M', '1Y', '2Y', '3Y', '4Y', '5Y']

comparison_data = []

for model in survival_columns:
    model_name = model.replace('S(t)_', '')
    model_data = all_allocations[all_allocations['Model'] == model_name]

    row = {'Model': model_name}

    for tenor in key_tenors_list:
        if tenor in model_data['Bucket'].values:
            cf_pct = model_data[model_data['Bucket'] == tenor]['CF_Percent'].values[0]
            row[f'{tenor}_pct'] = cf_pct
        else:
            row[f'{tenor}_pct'] = 0.0

    # Calculate weighted average maturity
    wam = (model_data['Total_CF'] * model_data['Midpoint_Years']).sum() / model_data['Total_CF'].sum()
    row['WAM_Years'] = wam

    comparison_data.append(row)

comparison_table = pd.DataFrame(comparison_data)

print("\n")
print(comparison_table.to_string(index=False))

# %%
# Highlight 5Y allocation specifically
print("\n" + "="*80)
print("5Y BUCKET ALLOCATION COMPARISON")
print("="*80)

five_y_comparison = all_allocations[all_allocations['Bucket'] == '5Y'][['Model', 'Total_CF', 'CF_Percent']].copy()
five_y_comparison = five_y_comparison.sort_values('CF_Percent', ascending=False)

print("\n")
print(five_y_comparison.to_string(index=False))

# Find best model for 5Y allocation
best_5y_model = five_y_comparison.iloc[0]
print(f"\n✓ BEST for 5Y allocation: {best_5y_model['Model']} with {best_5y_model['CF_Percent']:.2f}%")

# %%
# Visualizations
fig, axes = plt.subplots(2, 2, figsize=(18, 12))

# Plot 1: 5Y allocation comparison
ax = axes[0, 0]
models_list = five_y_comparison['Model'].values
five_y_pcts = five_y_comparison['CF_Percent'].values

colors = plt.cm.RdYlGn(five_y_pcts / five_y_pcts.max())
bars = ax.barh(range(len(models_list)), five_y_pcts, color=colors, edgecolor='black', alpha=0.8)

ax.set_yticks(range(len(models_list)))
ax.set_yticklabels(models_list, fontsize=9)
ax.set_xlabel('5Y Bucket Allocation (%)', fontsize=11)
ax.set_title('5Y Bucket Allocation Across All Models', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3, axis='x')

for i, (bar, val) in enumerate(zip(bars, five_y_pcts)):
    width = bar.get_width()
    ax.text(width + 0.05, bar.get_y() + bar.get_height()/2,
            f'{val:.2f}%', va='center', fontsize=9, fontweight='bold')

# Plot 2: Full tenor profile for each model
ax = axes[0, 1]
key_tenors_short = ['1M', '3M', '6M', '1Y', '2Y', '3Y', '4Y', '5Y']

for model in survival_columns[:5]:  # Plot first 5 models to avoid clutter
    model_name = model.replace('S(t)_', '')
    model_data = all_allocations[all_allocations['Model'] == model_name]

    tenors_plot = []
    cf_pcts_plot = []

    for tenor in key_tenors_short:
        if tenor in model_data['Bucket'].values:
            tenors_plot.append(tenor)
            cf_pcts_plot.append(model_data[model_data['Bucket'] == tenor]['CF_Percent'].values[0])

    ax.plot(tenors_plot, cf_pcts_plot, marker='o', linewidth=2, label=model_name, alpha=0.8)

ax.set_xlabel('Tenor', fontsize=11)
ax.set_ylabel('Allocation (%)', fontsize=11)
ax.set_title('Allocation Profile Across Tenors (Top 5 Models)', fontsize=13, fontweight='bold')
ax.legend(loc='best', fontsize=8)
ax.grid(True, alpha=0.3)

# Plot 3: Weighted Average Maturity comparison
ax = axes[1, 0]
wam_data = comparison_table[['Model', 'WAM_Years']].sort_values('WAM_Years', ascending=False)

bars = ax.barh(range(len(wam_data)), wam_data['WAM_Years'].values,
               color='steelblue', edgecolor='black', alpha=0.8)

ax.set_yticks(range(len(wam_data)))
ax.set_yticklabels(wam_data['Model'].values, fontsize=9)
ax.set_xlabel('Weighted Average Maturity (Years)', fontsize=11)
ax.set_title('WAM Comparison Across Models', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3, axis='x')

for bar, val in zip(bars, wam_data['WAM_Years'].values):
    width = bar.get_width()
    ax.text(width + 0.01, bar.get_y() + bar.get_height()/2,
            f'{val:.3f}', va='center', fontsize=9)

# Plot 4: Stacked allocation for RECOMMENDED model
ax = axes[1, 1]
recommended_data = all_allocations[all_allocations['Model'] == 'RECOMMENDED']

buckets_plot = recommended_data['Bucket'].values
core_cfs = recommended_data['Core_CF'].values
non_core_cfs = recommended_data['Non_Core_CF'].values

x_pos = np.arange(len(buckets_plot))
bars1 = ax.bar(x_pos, non_core_cfs, color='#D62828', alpha=0.8,
               edgecolor='black', label='Non-Core')
bars2 = ax.bar(x_pos, core_cfs, bottom=non_core_cfs, color='#06A77D',
               alpha=0.8, edgecolor='black', label='Core')

ax.set_xlabel('Bucket', fontsize=11)
ax.set_ylabel('Cash Flow Amount', fontsize=11)
ax.set_title('RECOMMENDED Model: Cash Flow Distribution', fontsize=13, fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels(buckets_plot, fontsize=9)
ax.legend(loc='upper right', fontsize=10)
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('phase_2_all_models_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# %%
# Export results
comparison_table.to_csv('phase_2_all_models_key_tenors.csv', index=False)
all_allocations.to_csv('phase_2_all_models_full_allocation.csv', index=False)

with pd.ExcelWriter('phase_2_all_models_results.xlsx', engine='openpyxl') as writer:
    comparison_table.to_excel(writer, sheet_name='Key_Tenors_Comparison', index=False)
    all_allocations.to_excel(writer, sheet_name='Full_Allocation', index=False)
    five_y_comparison.to_excel(writer, sheet_name='5Y_Ranking', index=False)

print("\n" + "="*80)
print("FILES EXPORTED")
print("="*80)
print("1. phase_2_all_models_key_tenors.csv")
print("   → Key tenor allocations for all models")
print("\n2. phase_2_all_models_full_allocation.csv")
print("   → Complete allocation data for all models")
print("\n3. phase_2_all_models_results.xlsx")
print("   → Excel workbook with all results")
print("\n4. phase_2_all_models_comparison.png")
print("   → Visualizations comparing all models")

print("\n" + "="*80)
print("PHASE 2 ALL MODELS COMPARISON COMPLETE ✓")
print("="*80)

# %%
