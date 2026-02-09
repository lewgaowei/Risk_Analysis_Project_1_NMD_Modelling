# %% [markdown]
# # Phase 5a: Sensitivity to Core Deposit Ratio
#
# ## QF609 Project #1 - IRRBB Modelling Framework
#
# **Calculation Date:** 30-Dec-2023
#
# Tests how varying the core deposit ratio assumption affects EVE and NII results.

# %%
from phase_5_helpers import *

print("="*80)
print("SECTION 5a: SENSITIVITY TO CORE DEPOSIT RATIO")
print("="*80)

# %% [markdown]
# ## Verification Gate

# %%
rp_check = slot_cashflows(current_balance, core_ratio_primary, lambda_daily)
total_cf_check = rp_check['Total_CF'].sum()
print(f"\nVerification - Cash Flow Slotting:")
print(f"   Total CF:      {total_cf_check:,.2f}  (expected: {current_balance:,.2f})")
print(f"   Match: {'PASS' if abs(total_cf_check - current_balance) < 0.01 else 'FAIL'}")

eve_base_check = compute_eve(rp_check, tenors_years, base_rates)
eve_base_expected = eve_summary_orig[eve_summary_orig['Scenario'] == 'Base Case']['EVE'].values[0]
print(f"\nVerification - Base EVE:")
print(f"   Computed:  {eve_base_check:,.2f}  (expected: {eve_base_expected:,.2f})")
print(f"   Match: {'PASS' if abs(eve_base_check - eve_base_expected) < 0.1 else 'FAIL'}")

# %% [markdown]
# ## Core Ratio Sensitivity Analysis

# %%
core_ratios_test = [0.40, 0.50, 0.60, 0.70, 0.80]
sensitivity_results = []

for cr in core_ratios_test:
    rp = slot_cashflows(current_balance, cr, lambda_daily)
    eve_b = compute_eve(rp, tenors_years, base_rates)

    row = {
        'Core_Ratio': cr,
        'Core_Ratio_Pct': cr * 100,
        'EVE_Base': eve_b,
    }

    for sname, sfunc in shock_funcs.items():
        shocked_rates = apply_shock_to_curve(tenors_years, base_rates, sfunc)
        eve_shocked = compute_eve(rp, tenors_years, shocked_rates)
        delta_eve = eve_shocked - eve_b
        row[f'dEVE_{sname[:2]}'] = delta_eve

    for sname, sfunc in shock_funcs.items():
        delta_nii = compute_nii(rp, sfunc)
        row[f'dNII_{sname[:2]}'] = delta_nii

    eve_deltas = [row[f'dEVE_{sname[:2]}'] for sname in shock_funcs]
    nii_deltas = [row[f'dNII_{sname[:2]}'] for sname in shock_funcs]
    row['Worst_dEVE'] = min(eve_deltas)
    row['Worst_dNII'] = min(nii_deltas)

    sensitivity_results.append(row)

sensitivity_df = pd.DataFrame(sensitivity_results)

print("\nCore Ratio Sensitivity Results:")
print(sensitivity_df[['Core_Ratio_Pct', 'EVE_Base', 'Worst_dEVE', 'Worst_dNII']].to_string(index=False))

sensitivity_df.to_csv('sensitivity_core_ratio_results.csv', index=False)
print("\nSaved: sensitivity_core_ratio_results.csv")

# %% [markdown]
# ## Charts

# %%
# Chart 1: Worst dEVE vs Core Ratio
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(sensitivity_df['Core_Ratio_Pct'], sensitivity_df['Worst_dEVE'],
        marker='o', linewidth=2.5, markersize=10, color='#E63946')
for _, r in sensitivity_df.iterrows():
    ax.text(r['Core_Ratio_Pct'], r['Worst_dEVE'] - 1.5, f"{r['Worst_dEVE']:.1f}",
            ha='center', va='top', fontsize=9, fontweight='bold')
ax.set_xlabel('Core Deposit Ratio (%)', fontsize=12)
ax.set_ylabel('Worst dEVE', fontsize=12)
ax.set_title('Worst-Case dEVE vs Core Deposit Ratio', fontsize=14, fontweight='bold')
ax.axvline(x=core_ratio_primary*100, color='blue', linestyle='--', linewidth=1.5,
           alpha=0.7, label=f'Primary: {core_ratio_primary*100:.1f}%')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# %%
# Chart 2: Worst dNII vs Core Ratio
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(sensitivity_df['Core_Ratio_Pct'], sensitivity_df['Worst_dNII'],
        marker='s', linewidth=2.5, markersize=10, color='#023047')
for _, r in sensitivity_df.iterrows():
    ax.text(r['Core_Ratio_Pct'], r['Worst_dNII'] - 15, f"{r['Worst_dNII']:.1f}",
            ha='center', va='top', fontsize=9, fontweight='bold')
ax.set_xlabel('Core Deposit Ratio (%)', fontsize=12)
ax.set_ylabel('Worst dNII', fontsize=12)
ax.set_title('Worst-Case dNII vs Core Deposit Ratio', fontsize=14, fontweight='bold')
ax.axvline(x=core_ratio_primary*100, color='blue', linestyle='--', linewidth=1.5,
           alpha=0.7, label=f'Primary: {core_ratio_primary*100:.1f}%')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# %%
# Chart 3: Grouped bar - All dEVE scenarios for each core ratio
fig, ax = plt.subplots(figsize=(14, 7))
x = np.arange(len(core_ratios_test))
width = 0.18
scenario_labels = ['S1', 'S2', 'S3', 'S4']
colors_bar = ['#E63946', '#06A77D', '#F77F00', '#9B59B6']

for i, (slabel, color) in enumerate(zip(scenario_labels, colors_bar)):
    vals = sensitivity_df[f'dEVE_{slabel}'].values
    ax.bar(x + i * width, vals, width, label=slabel, color=color, alpha=0.8, edgecolor='black')

ax.set_xlabel('Core Deposit Ratio (%)', fontsize=12)
ax.set_ylabel('dEVE', fontsize=12)
ax.set_title('dEVE Across All Scenarios by Core Deposit Ratio', fontsize=14, fontweight='bold')
ax.set_xticks(x + width * 1.5)
ax.set_xticklabels([f'{cr:.0f}%' for cr in sensitivity_df['Core_Ratio_Pct']], fontsize=10)
ax.axhline(y=0, color='black', linewidth=1)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.show()

print("\nSection 5a Complete: 3 charts generated")

# %%
