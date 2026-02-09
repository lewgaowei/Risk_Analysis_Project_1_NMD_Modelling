# %% [markdown]
# # Phase 5d: Monte Carlo Simulation
#
# ## QF609 Project #1 - IRRBB Modelling Framework
#
# **Calculation Date:** 30-Dec-2023
#
# Simulates 1,000 deposit balance paths with stochastic decay and computes VaR/ES.

# %%
from phase_5_helpers import *

print("="*80)
print("SECTION 5d: MONTE CARLO SIMULATION")
print("="*80)

# %% [markdown]
# ## Simulation Setup

# %%
# Model: Outflow(t) = Balance(t) * max(lambda + epsilon, 0), epsilon ~ N(0, sigma^2)
sigma_decay = nmd_data['daily_decay_rate'].std()
n_paths = 1000
n_days_sim = 365 * 5  # 5 years forward
seed = 42

np.random.seed(seed)

print(f"\nSimulation Parameters:")
print(f"  Lambda (mean daily decay):  {lambda_daily:.6f}")
print(f"  Sigma (std of decay rate):  {sigma_decay:.6f}")
print(f"  Number of paths:            {n_paths}")
print(f"  Horizon:                    {n_days_sim} days (5 years)")
print(f"  Starting balance:           {current_balance:,.2f}")
print(f"  Random seed:                {seed}")

# %% [markdown]
# ## Run Simulation

# %%
balance_paths = np.zeros((n_paths, n_days_sim + 1))
balance_paths[:, 0] = current_balance

for path in range(n_paths):
    epsilons = np.random.normal(0, sigma_decay, n_days_sim)
    for t in range(n_days_sim):
        decay_t = max(lambda_daily + epsilons[t], 0)
        outflow_t = balance_paths[path, t] * decay_t
        balance_paths[path, t + 1] = balance_paths[path, t] - outflow_t

# Compute percentiles
days_sim = np.arange(n_days_sim + 1)
pct_5 = np.percentile(balance_paths, 5, axis=0)
pct_25 = np.percentile(balance_paths, 25, axis=0)
pct_50 = np.percentile(balance_paths, 50, axis=0)
pct_75 = np.percentile(balance_paths, 75, axis=0)
pct_95 = np.percentile(balance_paths, 95, axis=0)

print(f"\nBalance Distribution at Year 1 (day 365):")
print(f"  5th percentile:   {pct_5[365]:,.2f}")
print(f"  25th percentile:  {pct_25[365]:,.2f}")
print(f"  Median:           {pct_50[365]:,.2f}")
print(f"  75th percentile:  {pct_75[365]:,.2f}")
print(f"  95th percentile:  {pct_95[365]:,.2f}")

# %% [markdown]
# ## EVE/NII Risk Metrics

# %%
print("\nComputing EVE/NII for each simulated path at Year 1...")
year1_balances = balance_paths[:, 365]

sim_delta_eves = np.zeros(n_paths)
sim_delta_niis = np.zeros(n_paths)

rates_s1_arr = apply_shock_to_curve(tenors_years, base_rates, shock_s1)

for i in range(n_paths):
    bal = year1_balances[i]
    rp_sim = slot_cashflows(bal, core_ratio_primary, lambda_daily)
    eve_b_sim = compute_eve(rp_sim, tenors_years, base_rates)
    eve_s1_sim = compute_eve(rp_sim, tenors_years, rates_s1_arr)
    sim_delta_eves[i] = eve_s1_sim - eve_b_sim
    sim_delta_niis[i] = compute_nii(rp_sim, shock_s2)

# VaR and ES (at 95% confidence level)
var_eve_95 = np.percentile(sim_delta_eves, 5)
es_eve_95 = np.mean(sim_delta_eves[sim_delta_eves <= var_eve_95])

var_nii_95 = np.percentile(sim_delta_niis, 5)
es_nii_95 = np.mean(sim_delta_niis[sim_delta_niis <= var_nii_95])

print(f"\nRisk Metrics (95% confidence):")
print(f"  VaR dEVE:   {var_eve_95:,.2f}")
print(f"  ES  dEVE:   {es_eve_95:,.2f}")
print(f"  VaR dNII:   {var_nii_95:,.2f}")
print(f"  ES  dNII:   {es_nii_95:,.2f}")

# Save summary
mc_summary = pd.DataFrame({
    'Metric': ['n_paths', 'horizon_years', 'lambda_daily', 'sigma_decay',
               'Balance_Y1_5pct', 'Balance_Y1_Median', 'Balance_Y1_95pct',
               'VaR_dEVE_95', 'ES_dEVE_95', 'VaR_dNII_95', 'ES_dNII_95',
               'Mean_dEVE', 'Std_dEVE', 'Mean_dNII', 'Std_dNII'],
    'Value': [n_paths, 5, lambda_daily, sigma_decay,
              pct_5[365], pct_50[365], pct_95[365],
              var_eve_95, es_eve_95, var_nii_95, es_nii_95,
              np.mean(sim_delta_eves), np.std(sim_delta_eves),
              np.mean(sim_delta_niis), np.std(sim_delta_niis)]
})
mc_summary.to_csv('monte_carlo_summary.csv', index=False)

# Save sample paths
sample_idx = list(range(0, n_paths, 100))
sample_paths_df = pd.DataFrame({'Day': days_sim})
for idx in sample_idx:
    sample_paths_df[f'Path_{idx}'] = balance_paths[idx, :]
sample_paths_df['Pct_5'] = pct_5
sample_paths_df['Pct_25'] = pct_25
sample_paths_df['Pct_50'] = pct_50
sample_paths_df['Pct_75'] = pct_75
sample_paths_df['Pct_95'] = pct_95
sample_paths_df.to_csv('monte_carlo_paths_sample.csv', index=False)

print("\nSaved: monte_carlo_summary.csv, monte_carlo_paths_sample.csv")

# %% [markdown]
# ## Charts

# %%
# Chart 1: Fan chart of balance paths
fig, ax = plt.subplots(figsize=(14, 8))
years_axis = days_sim / 365

ax.fill_between(years_axis, pct_5, pct_95, alpha=0.15, color='#E63946', label='5th-95th pct')
ax.fill_between(years_axis, pct_25, pct_75, alpha=0.3, color='#F77F00', label='25th-75th pct')
ax.plot(years_axis, pct_50, linewidth=2.5, color='#023047', label='Median')
ax.plot(years_axis, pct_5, linewidth=1, color='#E63946', linestyle='--', alpha=0.7)
ax.plot(years_axis, pct_95, linewidth=1, color='#E63946', linestyle='--', alpha=0.7)

ax.axvline(x=1, color='grey', linestyle=':', linewidth=1.5, alpha=0.7, label='Year 1')

ax.set_xlabel('Time (Years)', fontsize=12)
ax.set_ylabel('Balance', fontsize=12)
ax.set_title(f'Monte Carlo Balance Paths ({n_paths} simulations, 5-Year Horizon)',
            fontsize=14, fontweight='bold')
ax.legend(fontsize=10, loc='upper right')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# %%
# Chart 2: Histogram of simulated dEVE
fig, ax = plt.subplots(figsize=(12, 7))
ax.hist(sim_delta_eves, bins=50, color='#2A9D8F', alpha=0.7, edgecolor='black')
ax.axvline(x=var_eve_95, color='#E63946', linewidth=2.5, linestyle='--',
           label=f'VaR 95%: {var_eve_95:.1f}')
ax.axvline(x=es_eve_95, color='#D62828', linewidth=2.5, linestyle='-.',
           label=f'ES 95%: {es_eve_95:.1f}')
ax.axvline(x=np.mean(sim_delta_eves), color='#023047', linewidth=2,
           label=f'Mean: {np.mean(sim_delta_eves):.1f}')
ax.set_xlabel('dEVE (S1: +200bps)', fontsize=12)
ax.set_ylabel('Frequency', fontsize=12)
ax.set_title('Distribution of Simulated dEVE at Year 1', fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.show()

# %%
# Chart 3: Histogram of simulated dNII
fig, ax = plt.subplots(figsize=(12, 7))
ax.hist(sim_delta_niis, bins=50, color='#264653', alpha=0.7, edgecolor='black')
ax.axvline(x=var_nii_95, color='#E63946', linewidth=2.5, linestyle='--',
           label=f'VaR 95%: {var_nii_95:.1f}')
ax.axvline(x=es_nii_95, color='#D62828', linewidth=2.5, linestyle='-.',
           label=f'ES 95%: {es_nii_95:.1f}')
ax.axvline(x=np.mean(sim_delta_niis), color='#023047', linewidth=2,
           label=f'Mean: {np.mean(sim_delta_niis):.1f}')
ax.set_xlabel('dNII (S2: -200bps)', fontsize=12)
ax.set_ylabel('Frequency', fontsize=12)
ax.set_title('Distribution of Simulated dNII at Year 1', fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.show()

print("\nSection 5d Complete: 3 charts generated")

# %%
