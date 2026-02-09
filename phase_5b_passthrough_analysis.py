# %% [markdown]
# # Phase 5b: Pass-Through Rate Analysis
#
# ## QF609 Project #1 - IRRBB Modelling Framework
#
# **Calculation Date:** 30-Dec-2023
#
# Analyzes how deposit pass-through rate (beta) affects NII sensitivity.

# %%
from phase_5_helpers import *

print("="*80)
print("SECTION 5b: PASS-THROUGH RATE ANALYSIS")
print("="*80)
print("\nPass-through (beta) measures how much of a rate shock is passed to depositors.")
print("Effective shock = shock(t) * (1 - beta)")
print("Higher beta -> less NII impact (bank passes rate changes to customers)")

# %% [markdown]
# ## Pass-Through Sensitivity

# %%
beta_values = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

rp_primary = slot_cashflows(current_balance, core_ratio_primary, lambda_daily)

passthrough_results = []

for beta in beta_values:
    row = {'Beta': beta, 'Beta_Pct': beta * 100}

    for sname, sfunc in shock_funcs.items():
        def make_adjusted_shock(original_shock, b):
            def adjusted(t):
                return original_shock(t) * (1 - b)
            return adjusted

        adj_shock = make_adjusted_shock(sfunc, beta)
        delta_nii = compute_nii(rp_primary, adj_shock)
        row[f'dNII_{sname[:2]}'] = delta_nii

    nii_vals = [row[f'dNII_{sname[:2]}'] for sname in shock_funcs]
    row['Worst_dNII'] = min(nii_vals)

    passthrough_results.append(row)

passthrough_df = pd.DataFrame(passthrough_results)

print("\nPass-Through NII Sensitivity:")
print(passthrough_df.to_string(index=False))

# Validate beta=0 matches Phase 4
beta0_s2 = passthrough_df[passthrough_df['Beta'] == 0.0]['dNII_S2'].values[0]
phase4_s2 = nii_summary_orig[nii_summary_orig['Scenario'] == 'S2: -200bps Parallel']['\u0394NII'].values[0]
print(f"\nValidation (beta=0 vs Phase 4):")
print(f"  Beta=0 dNII_S2: {beta0_s2:,.2f}")
print(f"  Phase 4 dNII_S2: {phase4_s2:,.2f}")
print(f"  Match: {'PASS' if abs(beta0_s2 - phase4_s2) < 0.1 else 'FAIL'}")

passthrough_df.to_csv('passthrough_nii_sensitivity.csv', index=False)
print("\nSaved: passthrough_nii_sensitivity.csv")

# %% [markdown]
# ## Charts

# %%
# Chart 1: Multi-line dNII vs beta for each scenario
fig, ax = plt.subplots(figsize=(12, 7))
scenario_labels = ['S1', 'S2', 'S3', 'S4']
colors_line = ['#E63946', '#06A77D', '#F77F00', '#9B59B6']
markers = ['o', 's', 'D', '^']

for slabel, color, marker in zip(scenario_labels, colors_line, markers):
    ax.plot(passthrough_df['Beta_Pct'], passthrough_df[f'dNII_{slabel}'],
            marker=marker, linewidth=2, markersize=8, color=color, label=slabel, alpha=0.8)

ax.axhline(y=0, color='black', linewidth=1, alpha=0.5)
ax.set_xlabel('Pass-Through Rate Beta (%)', fontsize=12)
ax.set_ylabel('dNII', fontsize=12)
ax.set_title('dNII Sensitivity to Pass-Through Rate (Beta)', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# %%
# Chart 2: Worst dNII vs beta
fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(passthrough_df['Beta_Pct'], passthrough_df['Worst_dNII'],
       color=['#D62828' if v < 0 else '#06A77D' for v in passthrough_df['Worst_dNII']],
       alpha=0.8, edgecolor='black', width=12)
for _, r in passthrough_df.iterrows():
    val = r['Worst_dNII']
    label_y = val - 15 if val < 0 else val + 5
    ax.text(r['Beta_Pct'], label_y, f'{val:.0f}',
            ha='center', va='top' if val < 0 else 'bottom',
            fontsize=10, fontweight='bold')
ax.axhline(y=0, color='black', linewidth=1.5)
ax.set_xlabel('Pass-Through Rate Beta (%)', fontsize=12)
ax.set_ylabel('Worst dNII', fontsize=12)
ax.set_title('Worst-Case dNII vs Pass-Through Rate', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.show()

print("\nSection 5b Complete: 2 charts generated")

# %%
