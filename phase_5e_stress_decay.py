# %% [markdown]
# # Phase 5e: Stress-Adjusted Decay
#
# ## QF609 Project #1 - IRRBB Modelling Framework
#
# **Calculation Date:** 30-Dec-2023
#
# Tests impact of stressed (1.5x, 2.0x) decay rates on survival, cash flow slotting, EVE, and NII.

# %%
from phase_5_helpers import *

print("="*80)
print("SECTION 5e: STRESS-ADJUSTED DECAY")
print("="*80)

# %% [markdown]
# ## Stress Scenarios

# %%
stress_multipliers = [1.0, 1.5, 2.0]
stress_labels = ['1.0x (Base)', '1.5x Stress', '2.0x Stress']

stress_results = []
stress_survival_data = {'Day': np.arange(0, 1826)}

for mult, label in zip(stress_multipliers, stress_labels):
    lambda_stressed = lambda_daily * mult

    # Survival curve
    days_arr = np.arange(0, 1826)
    surv = np.array([get_survival(d, lambda_stressed) for d in days_arr])
    stress_survival_data[f'S(t)_{label}'] = surv

    # Re-slot cash flows
    rp_stress = slot_cashflows(current_balance, core_ratio_primary, lambda_stressed)

    # Compute EVE base and under shocks
    eve_b_stress = compute_eve(rp_stress, tenors_years, base_rates)

    row = {
        'Stress_Multiplier': mult,
        'Label': label,
        'Lambda_Stressed': lambda_stressed,
        'EVE_Base': eve_b_stress
    }

    for sname, sfunc in shock_funcs.items():
        shocked_rates = apply_shock_to_curve(tenors_years, base_rates, sfunc)
        eve_s = compute_eve(rp_stress, tenors_years, shocked_rates)
        delta_nii = compute_nii(rp_stress, sfunc)
        row[f'dEVE_{sname[:2]}'] = eve_s - eve_b_stress
        row[f'dNII_{sname[:2]}'] = delta_nii

    eve_deltas = [row[f'dEVE_{sname[:2]}'] for sname in shock_funcs]
    nii_deltas = [row[f'dNII_{sname[:2]}'] for sname in shock_funcs]
    row['Worst_dEVE'] = min(eve_deltas)
    row['Worst_dNII'] = min(nii_deltas)

    stress_results.append(row)

    print(f"\n{label}:")
    print(f"  Lambda:      {lambda_stressed:.6f}")
    print(f"  EVE Base:    {eve_b_stress:,.2f}")
    print(f"  Worst dEVE:  {row['Worst_dEVE']:,.2f}")
    print(f"  Worst dNII:  {row['Worst_dNII']:,.2f}")

stress_df = pd.DataFrame(stress_results)
stress_surv_df = pd.DataFrame(stress_survival_data)

stress_df.to_csv('stress_decay_results.csv', index=False)
stress_surv_df.to_csv('stress_survival_curves.csv', index=False)
print("\nSaved: stress_decay_results.csv, stress_survival_curves.csv")

# %% [markdown]
# ## Charts

# %%
# Chart 1: Survival curves under 3 stress levels
fig, ax = plt.subplots(figsize=(12, 7))
colors_stress = ['#023047', '#F77F00', '#E63946']

for label, color in zip(stress_labels, colors_stress):
    col = f'S(t)_{label}'
    years_surv = stress_surv_df['Day'].values / 365
    ax.plot(years_surv, stress_surv_df[col].values * 100,
            linewidth=2.5, color=color, label=label)

ax.set_xlabel('Time (Years)', fontsize=12)
ax.set_ylabel('Survival Probability (%)', fontsize=12)
ax.set_title('Survival Curves Under Stress Scenarios', fontsize=14, fontweight='bold')
ax.set_xlim(0, 5)
ax.set_ylim(0, 105)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# %%
# Chart 2: Repricing profile comparison (grouped bars)
fig, ax = plt.subplots(figsize=(14, 7))

bucket_names = ['O/N', '1M', '2M', '3M', '6M', '9M', '1Y', '2Y', '3Y', '4Y', '5Y']
x = np.arange(len(bucket_names))
width = 0.25

for i, (mult, label, color) in enumerate(zip(stress_multipliers, stress_labels, colors_stress)):
    lambda_s = lambda_daily * mult
    rp_s = slot_cashflows(current_balance, core_ratio_primary, lambda_s)
    cf_pct = (rp_s['Total_CF'] / current_balance) * 100
    ax.bar(x + i * width, cf_pct, width, label=label, color=color, alpha=0.8, edgecolor='black')

ax.set_xlabel('Time Bucket', fontsize=12)
ax.set_ylabel('Cash Flow (% of Balance)', fontsize=12)
ax.set_title('Repricing Profile Under Stress Scenarios', fontsize=14, fontweight='bold')
ax.set_xticks(x + width)
ax.set_xticklabels(bucket_names, fontsize=10)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.show()

# %%
# Chart 3: Worst dEVE and dNII comparison
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Left: Worst dEVE
bars_eve = axes[0].bar(stress_labels, stress_df['Worst_dEVE'],
                       color=colors_stress, alpha=0.8, edgecolor='black')
for bar, val in zip(bars_eve, stress_df['Worst_dEVE']):
    axes[0].text(bar.get_x() + bar.get_width()/2., val - 2,
                f'{val:.1f}', ha='center', va='top', fontsize=10, fontweight='bold')
axes[0].axhline(y=0, color='black', linewidth=1)
axes[0].set_ylabel('Worst dEVE', fontsize=12)
axes[0].set_title('Worst dEVE by Stress Level', fontsize=13, fontweight='bold')
axes[0].grid(True, alpha=0.3, axis='y')

# Right: Worst dNII
bars_nii = axes[1].bar(stress_labels, stress_df['Worst_dNII'],
                       color=colors_stress, alpha=0.8, edgecolor='black')
for bar, val in zip(bars_nii, stress_df['Worst_dNII']):
    label_y = val - 15 if val < 0 else val + 5
    axes[1].text(bar.get_x() + bar.get_width()/2., label_y,
                f'{val:.1f}', ha='center',
                va='top' if val < 0 else 'bottom',
                fontsize=10, fontweight='bold')
axes[1].axhline(y=0, color='black', linewidth=1)
axes[1].set_ylabel('Worst dNII', fontsize=12)
axes[1].set_title('Worst dNII by Stress Level', fontsize=13, fontweight='bold')
axes[1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.show()

print("\nSection 5e Complete: 3 charts generated")

# %%
