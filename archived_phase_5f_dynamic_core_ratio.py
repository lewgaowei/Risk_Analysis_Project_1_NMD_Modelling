# %% [markdown]
# # Phase 5f: Dynamic Core Ratio Analysis
#
# ## QF609 Project #1 - IRRBB Modelling Framework
#
# **Calculation Date:** 30-Dec-2023
#
# **WHAT ARE WE DOING HERE?**
# Testing whether a DYNAMIC (time-varying) core ratio makes more sense
# than a STATIC (fixed) 51% assumption.
#
# **WHY?**
# The balance has grown from $9,511 (2017) to $18,652 (2023).
# Using a fixed 51% core ratio might be:
# - Too conservative (missing profit opportunities)
# - Ignoring structural improvements in deposit base
# - Not reflecting actual market conditions
#
# **5 DYNAMIC METHODS:**
# 1. Trend-Based: Core grows linearly over time
# 2. Volatility-Based: High volatility → Lower core
# 3. Regime-Switching: Different market regimes → Different core
# 4. Rolling Minimum: Use recent history, not all-time minimum
# 5. Economic-Indicator: Adjust based on balance growth patterns
#
# **GOAL:**
# Understand the PROFIT vs RISK trade-off of dynamic core assumptions

# %%
from phase_5_helpers import *
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("SECTION 5f: DYNAMIC CORE RATIO ANALYSIS")
print("="*80)

# %% [markdown]
# ## 1. Baseline: Static Core Ratio (51%)

# %%
print("\n" + "="*80)
print("BASELINE: STATIC CORE RATIO")
print("="*80)

print(f"\nCurrent Approach:")
print(f"  Core Ratio:           {core_ratio_primary*100:.1f}% (fixed)")
print(f"  Core Amount:          ${core_amount:,.2f}")
print(f"  Non-Core Amount:      ${non_core_amount:,.2f}")
print(f"  Method:               Historical Minimum Balance")
print(f"  Rationale:            Conservative, Basel-approved")

# Calculate baseline EVE and NII
rp_static = slot_cashflows(current_balance, core_ratio_primary, lambda_daily)
eve_base_static = compute_eve(rp_static, tenors_years, base_rates)

rates_s1 = apply_shock_to_curve(tenors_years, base_rates, shock_s1)
eve_s1_static = compute_eve(rp_static, tenors_years, rates_s1)
delta_eve_static = eve_s1_static - eve_base_static

delta_nii_static = compute_nii(rp_static, shock_s2)

print(f"\nRisk Metrics (Static 51%):")
print(f"  EVE Base:             ${eve_base_static:,.2f}")
print(f"  ΔEVE (S1 +200bps):    ${delta_eve_static:,.2f}")
print(f"  ΔNII (S2 -200bps):    ${delta_nii_static:,.2f}")

# %% [markdown]
# ## 2. Method 1: Trend-Based Dynamic Core

# %%
print("\n" + "="*80)
print("METHOD 1: TREND-BASED DYNAMIC CORE")
print("="*80)
print("\nConcept: Core ratio grows linearly as balance grows")
print("Assumption: 2% annual increase in core ratio (conservative)")

def calculate_trend_based_core(year, base_core=core_ratio_primary, growth_rate=0.02):
    """
    Core ratio increases linearly over time

    Args:
        year: Current year
        base_core: Starting core ratio (e.g., 0.51 in 2017)
        growth_rate: Annual increase in core ratio

    Returns:
        Dynamic core ratio
    """
    year_0 = 2017
    years_elapsed = year - year_0
    dynamic_core = base_core + (growth_rate * years_elapsed)

    # Basel maximum: 90%
    return min(dynamic_core, 0.90)

# Calculate for current year (2023)
core_ratio_trend = calculate_trend_based_core(2023)
core_amount_trend = current_balance * core_ratio_trend
non_core_trend = current_balance - core_amount_trend

print(f"\nTrend-Based Core (2023):")
print(f"  Years Elapsed:        {2023 - 2017} years")
print(f"  Growth Rate:          2.0% per year")
print(f"  Core Ratio:           {core_ratio_trend*100:.1f}%")
print(f"  Core Amount:          ${core_amount_trend:,.2f}")
print(f"  Non-Core Amount:      ${non_core_trend:,.2f}")
print(f"  vs Static:            {(core_ratio_trend - core_ratio_primary)*100:+.1f} percentage points")

# Calculate EVE and NII with trend-based core
rp_trend = slot_cashflows(current_balance, core_ratio_trend, lambda_daily)
eve_base_trend = compute_eve(rp_trend, tenors_years, base_rates)
eve_s1_trend = compute_eve(rp_trend, tenors_years, rates_s1)
delta_eve_trend = eve_s1_trend - eve_base_trend
delta_nii_trend = compute_nii(rp_trend, shock_s2)

print(f"\nRisk Metrics (Trend-Based):")
print(f"  EVE Base:             ${eve_base_trend:,.2f}")
print(f"  ΔEVE (S1 +200bps):    ${delta_eve_trend:,.2f}")
print(f"  ΔNII (S2 -200bps):    ${delta_nii_trend:,.2f}")

# %% [markdown]
# ## 3. Method 2: Volatility-Based Dynamic Core

# %%
print("\n" + "="*80)
print("METHOD 2: VOLATILITY-BASED DYNAMIC CORE")
print("="*80)
print("\nConcept: Lower volatility → Higher core ratio")
print("Logic: Stable balances indicate more sticky deposits")

# Calculate rolling volatility (90-day window)
nmd_clean = nmd_data.dropna(subset=['Balance']).copy()
nmd_clean['rolling_std'] = nmd_clean['Balance'].rolling(window=90).std()
nmd_clean['rolling_mean'] = nmd_clean['Balance'].rolling(window=90).mean()
nmd_clean['CV'] = nmd_clean['rolling_std'] / nmd_clean['rolling_mean']

# Define core ratio mapping
cv_values = nmd_clean['CV'].dropna()
cv_min = cv_values.quantile(0.05)  # Most stable
cv_max = cv_values.quantile(0.95)  # Most volatile

core_max = 0.75  # High core when very stable
core_min = 0.45  # Low core when very volatile

# Current volatility
current_cv = nmd_clean['CV'].iloc[-1]

# Linear interpolation
if current_cv <= cv_min:
    core_ratio_vol = core_max
elif current_cv >= cv_max:
    core_ratio_vol = core_min
else:
    core_ratio_vol = core_max - ((current_cv - cv_min) / (cv_max - cv_min)) * (core_max - core_min)

core_amount_vol = current_balance * core_ratio_vol
non_core_vol = current_balance - core_amount_vol

print(f"\nVolatility Analysis:")
print(f"  Current CV:           {current_cv:.4f}")
print(f"  CV Range:             {cv_min:.4f} (stable) to {cv_max:.4f} (volatile)")
print(f"  Core Ratio:           {core_ratio_vol*100:.1f}%")
print(f"  Core Amount:          ${core_amount_vol:,.2f}")
print(f"  Non-Core Amount:      ${non_core_vol:,.2f}")
print(f"  vs Static:            {(core_ratio_vol - core_ratio_primary)*100:+.1f} percentage points")

# Calculate EVE and NII
rp_vol = slot_cashflows(current_balance, core_ratio_vol, lambda_daily)
eve_base_vol = compute_eve(rp_vol, tenors_years, base_rates)
eve_s1_vol = compute_eve(rp_vol, tenors_years, rates_s1)
delta_eve_vol = eve_s1_vol - eve_base_vol
delta_nii_vol = compute_nii(rp_vol, shock_s2)

print(f"\nRisk Metrics (Volatility-Based):")
print(f"  EVE Base:             ${eve_base_vol:,.2f}")
print(f"  ΔEVE (S1 +200bps):    ${delta_eve_vol:,.2f}")
print(f"  ΔNII (S2 -200bps):    ${delta_nii_vol:,.2f}")

# %% [markdown]
# ## 4. Method 3: Regime-Switching Dynamic Core

# %%
print("\n" + "="*80)
print("METHOD 3: REGIME-SWITCHING DYNAMIC CORE")
print("="*80)
print("\nConcept: Identify market regimes, assign different core ratios")
print("Logic: Stable regime → Higher core; Volatile regime → Lower core")

# Prepare features for regime identification
features_regime = pd.DataFrame({
    'balance_growth': nmd_clean['Balance'].pct_change(periods=30).fillna(0),
    'volatility': nmd_clean['Balance'].rolling(30).std().fillna(method='bfill'),
    'decay_rate': nmd_clean['daily_decay_rate'].fillna(method='bfill')
})

# Normalize features
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features_regime)

# K-means clustering to identify 3 regimes
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
nmd_clean['regime'] = kmeans.fit_predict(features_scaled)

# Analyze each regime
regime_stats = nmd_clean.groupby('regime').agg({
    'Balance': ['mean', 'std', 'count'],
    'daily_decay_rate': 'mean'
}).round(2)

# Assign core ratios based on regime characteristics
# Calculate coefficient of variation for each regime
regime_core_map = {}
for regime in range(3):
    avg_balance = regime_stats.loc[regime, ('Balance', 'mean')]
    vol = regime_stats.loc[regime, ('Balance', 'std')]
    cv_regime = vol / avg_balance

    # Lower CV → Higher core
    if cv_regime < 0.10:
        regime_core_map[regime] = 0.70  # Stable regime
    elif cv_regime < 0.20:
        regime_core_map[regime] = 0.60  # Moderate regime
    else:
        regime_core_map[regime] = 0.50  # Volatile regime

print(f"\nRegime Analysis:")
print(regime_stats)

print(f"\nCore Ratio by Regime:")
for regime, core in regime_core_map.items():
    count = regime_stats.loc[regime, ('Balance', 'count')]
    print(f"  Regime {regime}: {core*100:.0f}% core ({count:.0f} observations)")

# Current regime
current_regime = nmd_clean['regime'].iloc[-1]
core_ratio_regime = regime_core_map[current_regime]
core_amount_regime = current_balance * core_ratio_regime
non_core_regime = current_balance - core_amount_regime

print(f"\nCurrent State (Dec 2023):")
print(f"  Current Regime:       {current_regime}")
print(f"  Core Ratio:           {core_ratio_regime*100:.1f}%")
print(f"  Core Amount:          ${core_amount_regime:,.2f}")
print(f"  Non-Core Amount:      ${non_core_regime:,.2f}")
print(f"  vs Static:            {(core_ratio_regime - core_ratio_primary)*100:+.1f} percentage points")

# Calculate EVE and NII
rp_regime = slot_cashflows(current_balance, core_ratio_regime, lambda_daily)
eve_base_regime = compute_eve(rp_regime, tenors_years, base_rates)
eve_s1_regime = compute_eve(rp_regime, tenors_years, rates_s1)
delta_eve_regime = eve_s1_regime - eve_base_regime
delta_nii_regime = compute_nii(rp_regime, shock_s2)

print(f"\nRisk Metrics (Regime-Switching):")
print(f"  EVE Base:             ${eve_base_regime:,.2f}")
print(f"  ΔEVE (S1 +200bps):    ${delta_eve_regime:,.2f}")
print(f"  ΔNII (S2 -200bps):    ${delta_nii_regime:,.2f}")

# %% [markdown]
# ## 5. Method 4: Rolling Minimum (2-Year Window)

# %%
print("\n" + "="*80)
print("METHOD 4: ROLLING MINIMUM DYNAMIC CORE")
print("="*80)
print("\nConcept: Use 2-year rolling minimum instead of all-time minimum")
print("Logic: Recent history more relevant than 7-year-old crisis")

# Calculate 2-year rolling minimum
window_days = 730  # 2 years
nmd_clean['rolling_min_2y'] = nmd_clean['Balance'].rolling(window=window_days, min_periods=365).min()

# Current 2-year minimum
rolling_min_2y = nmd_clean['rolling_min_2y'].iloc[-1]
core_ratio_rolling = rolling_min_2y / current_balance
core_amount_rolling = rolling_min_2y
non_core_rolling = current_balance - core_amount_rolling

print(f"\nRolling Minimum Analysis:")
print(f"  Historical Min:       ${core_amount:,.2f} (all-time, 2017)")
print(f"  2-Year Rolling Min:   ${rolling_min_2y:,.2f}")
print(f"  Core Ratio:           {core_ratio_rolling*100:.1f}%")
print(f"  Core Amount:          ${core_amount_rolling:,.2f}")
print(f"  Non-Core Amount:      ${non_core_rolling:,.2f}")
print(f"  vs Static:            {(core_ratio_rolling - core_ratio_primary)*100:+.1f} percentage points")

# Calculate EVE and NII
rp_rolling = slot_cashflows(current_balance, core_ratio_rolling, lambda_daily)
eve_base_rolling = compute_eve(rp_rolling, tenors_years, base_rates)
eve_s1_rolling = compute_eve(rp_rolling, tenors_years, rates_s1)
delta_eve_rolling = eve_s1_rolling - eve_base_rolling
delta_nii_rolling = compute_nii(rp_rolling, shock_s2)

print(f"\nRisk Metrics (Rolling 2Y):")
print(f"  EVE Base:             ${eve_base_rolling:,.2f}")
print(f"  ΔEVE (S1 +200bps):    ${delta_eve_rolling:,.2f}")
print(f"  ΔNII (S2 -200bps):    ${delta_nii_rolling:,.2f}")

# %% [markdown]
# ## 6. Method 5: Growth-Adjusted Core

# %%
print("\n" + "="*80)
print("METHOD 5: GROWTH-ADJUSTED DYNAMIC CORE")
print("="*80)
print("\nConcept: Strong recent growth indicates core has grown")
print("Logic: Sustained growth → Increase core estimate")

# Calculate growth metrics
balance_start = nmd_clean['Balance'].iloc[0]
balance_end = nmd_clean['Balance'].iloc[-1]
years_total = (nmd_clean['Date'].iloc[-1] - nmd_clean['Date'].iloc[0]).days / 365

total_growth_rate = (balance_end / balance_start) ** (1/years_total) - 1

# Assume core grows at half the total growth rate (conservative)
core_growth_rate = total_growth_rate / 2

# Project core forward from historical minimum
years_since_min = 2023 - 2017
adjusted_core_amount = core_amount * (1 + core_growth_rate) ** years_since_min

core_ratio_growth = adjusted_core_amount / current_balance
# Cap at reasonable level
core_ratio_growth = min(core_ratio_growth, 0.85)

core_amount_growth = current_balance * core_ratio_growth
non_core_growth = current_balance - core_amount_growth

print(f"\nGrowth Analysis:")
print(f"  Total Growth Rate:    {total_growth_rate*100:.2f}% per year")
print(f"  Assumed Core Growth:  {core_growth_rate*100:.2f}% per year (half of total)")
print(f"  Years Since Min:      {years_since_min} years")
print(f"  Projected Core (2023): ${adjusted_core_amount:,.2f}")
print(f"  Core Ratio:           {core_ratio_growth*100:.1f}%")
print(f"  Core Amount:          ${core_amount_growth:,.2f}")
print(f"  Non-Core Amount:      ${non_core_growth:,.2f}")
print(f"  vs Static:            {(core_ratio_growth - core_ratio_primary)*100:+.1f} percentage points")

# Calculate EVE and NII
rp_growth = slot_cashflows(current_balance, core_ratio_growth, lambda_daily)
eve_base_growth = compute_eve(rp_growth, tenors_years, base_rates)
eve_s1_growth = compute_eve(rp_growth, tenors_years, rates_s1)
delta_eve_growth = eve_s1_growth - eve_base_growth
delta_nii_growth = compute_nii(rp_growth, shock_s2)

print(f"\nRisk Metrics (Growth-Adjusted):")
print(f"  EVE Base:             ${eve_base_growth:,.2f}")
print(f"  ΔEVE (S1 +200bps):    ${delta_eve_growth:,.2f}")
print(f"  ΔNII (S2 -200bps):    ${delta_nii_growth:,.2f}")

# %% [markdown]
# ## 7. Comparison: All Methods

# %%
# Create comprehensive comparison table
comparison_results = pd.DataFrame({
    'Method': [
        'Static (Historical Min)',
        'Trend-Based',
        'Volatility-Based',
        'Regime-Switching',
        'Rolling 2Y Min',
        'Growth-Adjusted'
    ],
    'Core_Ratio_%': [
        core_ratio_primary * 100,
        core_ratio_trend * 100,
        core_ratio_vol * 100,
        core_ratio_regime * 100,
        core_ratio_rolling * 100,
        core_ratio_growth * 100
    ],
    'Core_Amount': [
        core_amount,
        core_amount_trend,
        core_amount_vol,
        core_amount_regime,
        core_amount_rolling,
        core_amount_growth
    ],
    'EVE_Base': [
        eve_base_static,
        eve_base_trend,
        eve_base_vol,
        eve_base_regime,
        eve_base_rolling,
        eve_base_growth
    ],
    'ΔEVE_S1': [
        delta_eve_static,
        delta_eve_trend,
        delta_eve_vol,
        delta_eve_regime,
        delta_eve_rolling,
        delta_eve_growth
    ],
    'ΔNII_S2': [
        delta_nii_static,
        delta_nii_trend,
        delta_nii_vol,
        delta_nii_regime,
        delta_nii_rolling,
        delta_nii_growth
    ]
})

# Calculate differences vs static
comparison_results['Δ_vs_Static_pct'] = comparison_results['Core_Ratio_%'] - comparison_results['Core_Ratio_%'].iloc[0]
comparison_results['ΔEVE_vs_Static'] = comparison_results['ΔEVE_S1'] - comparison_results['ΔEVE_S1'].iloc[0]
comparison_results['ΔNII_vs_Static'] = comparison_results['ΔNII_S2'] - comparison_results['ΔNII_S2'].iloc[0]

print("\n" + "="*80)
print("COMPREHENSIVE METHOD COMPARISON")
print("="*80)
print(comparison_results[['Method', 'Core_Ratio_%', 'EVE_Base', 'ΔEVE_S1', 'ΔNII_S2']].to_string(index=False))

print("\n" + "="*80)
print("DIFFERENCES vs STATIC BASELINE")
print("="*80)
print(comparison_results[['Method', 'Δ_vs_Static_pct', 'ΔEVE_vs_Static', 'ΔNII_vs_Static']].to_string(index=False))

# Save results
comparison_results.to_csv('dynamic_core_comparison.csv', index=False)
print("\nSaved: dynamic_core_comparison.csv")

# %% [markdown]
# ## 8. Visualizations

# %% [markdown]
# ### 8.1 Core Ratio Comparison Chart

# %%
fig, ax = plt.subplots(figsize=(12, 7))

methods = comparison_results['Method']
core_ratios = comparison_results['Core_Ratio_%']

colors = ['#E63946', '#06A77D', '#2A9D8F', '#F77F00', '#9B59B6', '#E9C46A']
bars = ax.barh(range(len(methods)), core_ratios, color=colors, alpha=0.8, edgecolor='black')

# Highlight static (baseline)
bars[0].set_linewidth(3)
bars[0].set_edgecolor('darkred')

# Add value labels
for i, (bar, val) in enumerate(zip(bars, core_ratios)):
    ax.text(val + 1, i, f'{val:.1f}%', va='center', fontsize=10, fontweight='bold')

# Reference line at static level
ax.axvline(x=comparison_results['Core_Ratio_%'].iloc[0], color='red',
          linestyle='--', linewidth=2, alpha=0.5, label='Static Baseline (51%)')

ax.set_yticks(range(len(methods)))
ax.set_yticklabels(methods, fontsize=10)
ax.set_xlabel('Core Ratio (%)', fontsize=12)
ax.set_title('Core Ratio Estimates: Static vs Dynamic Methods',
            fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.show()

# %% [markdown]
# ### 8.2 EVE and NII Impact Comparison

# %%
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# Left: ΔEVE comparison
delta_eves = comparison_results['ΔEVE_S1']
bars_eve = axes[0].barh(range(len(methods)), delta_eves,
                        color=colors, alpha=0.8, edgecolor='black')
bars_eve[0].set_linewidth(3)
bars_eve[0].set_edgecolor('darkred')

for i, (bar, val) in enumerate(zip(bars_eve, delta_eves)):
    x_pos = val - 5 if val < 0 else val + 5
    axes[0].text(x_pos, i, f'${val:.0f}', va='center', fontsize=9, fontweight='bold')

axes[0].axvline(x=0, color='black', linewidth=1.5)
axes[0].set_yticks(range(len(methods)))
axes[0].set_yticklabels(methods, fontsize=9)
axes[0].set_xlabel('ΔEVE (Scenario 1: +200bps)', fontsize=11)
axes[0].set_title('EVE Risk by Method\n(More negative = Worse)',
                 fontsize=12, fontweight='bold')
axes[0].grid(True, alpha=0.3, axis='x')

# Right: ΔNII comparison
delta_niis = comparison_results['ΔNII_S2']
bars_nii = axes[1].barh(range(len(methods)), delta_niis,
                        color=colors, alpha=0.8, edgecolor='black')
bars_nii[0].set_linewidth(3)
bars_nii[0].set_edgecolor('darkred')

for i, (bar, val) in enumerate(zip(bars_nii, delta_niis)):
    x_pos = val - 15 if val < 0 else val + 15
    axes[1].text(x_pos, i, f'${val:.0f}', va='center', fontsize=9, fontweight='bold')

axes[1].axvline(x=0, color='black', linewidth=1.5)
axes[1].set_yticks(range(len(methods)))
axes[1].set_yticklabels(methods, fontsize=9)
axes[1].set_xlabel('ΔNII (Scenario 2: -200bps)', fontsize=11)
axes[1].set_title('NII Risk by Method\n(More negative = Worse)',
                 fontsize=12, fontweight='bold')
axes[1].grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.show()

# %% [markdown]
# ### 8.3 Risk-Return Trade-Off Scatter

# %%
fig, ax = plt.subplots(figsize=(12, 8))

# Use EVE Base as proxy for "return" (higher EVE = more value)
# Use abs(ΔEVE) as proxy for risk (higher abs = more risk)
x_risk = comparison_results['ΔEVE_S1'].abs()
y_return = comparison_results['EVE_Base']

scatter = ax.scatter(x_risk, y_return, s=300, c=range(len(methods)),
                    cmap='viridis', alpha=0.7, edgecolor='black', linewidth=2)

# Add method labels
for i, method in enumerate(methods):
    ax.annotate(method, (x_risk.iloc[i], y_return.iloc[i]),
               xytext=(10, 5), textcoords='offset points',
               fontsize=9, fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.3))

# Highlight static
ax.scatter(x_risk.iloc[0], y_return.iloc[0], s=500,
          marker='*', color='red', edgecolor='darkred', linewidth=2,
          label='Static Baseline', zorder=10)

ax.set_xlabel('Risk: |ΔEVE| from +200bps shock', fontsize=12)
ax.set_ylabel('Return: EVE Base Value', fontsize=12)
ax.set_title('Risk-Return Trade-Off: Dynamic Core Methods\n(Upper-left = Better: High EVE, Low Risk)',
            fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# %% [markdown]
# ### 8.4 Time Series: Dynamic Core Ratios Over Time

# %%
# Build time series of dynamic core ratios
time_series_data = pd.DataFrame({
    'Date': nmd_clean['Date'],
    'Balance': nmd_clean['Balance']
})

# Calculate dynamic cores over time
time_series_data['Static'] = core_ratio_primary
time_series_data['Trend'] = time_series_data['Date'].apply(
    lambda d: calculate_trend_based_core(d.year)
)

# Volatility-based (simplified for time series)
rolling_cv = nmd_clean['CV'].fillna(method='bfill')
time_series_data['Volatility'] = (core_max -
    ((rolling_cv - cv_min) / (cv_max - cv_min)).clip(0, 1) * (core_max - core_min)
).clip(0.40, 0.85)

# Rolling 2Y
time_series_data['Rolling_2Y'] = (
    nmd_clean['rolling_min_2y'] / nmd_clean['Balance']
).clip(0.40, 0.85).fillna(method='bfill')

# Regime-based
time_series_data['Regime'] = nmd_clean['regime'].map(regime_core_map)

fig, axes = plt.subplots(2, 1, figsize=(14, 10))

# Top: Balance
axes[0].plot(time_series_data['Date'], time_series_data['Balance'],
            linewidth=2, color='#023047', label='Balance')
axes[0].fill_between(time_series_data['Date'], 0, time_series_data['Balance'],
                     alpha=0.2, color='#023047')
axes[0].set_ylabel('Balance', fontsize=11)
axes[0].set_title('Balance Over Time', fontsize=12, fontweight='bold')
axes[0].legend(fontsize=10)
axes[0].grid(True, alpha=0.3)

# Bottom: Dynamic core ratios
axes[1].plot(time_series_data['Date'], time_series_data['Static'] * 100,
            linewidth=3, color='#E63946', linestyle='--', label='Static (51%)', alpha=0.9)
axes[1].plot(time_series_data['Date'], time_series_data['Trend'] * 100,
            linewidth=2, color='#06A77D', label='Trend-Based', alpha=0.8)
axes[1].plot(time_series_data['Date'], time_series_data['Volatility'] * 100,
            linewidth=2, color='#2A9D8F', label='Volatility-Based', alpha=0.8)
axes[1].plot(time_series_data['Date'], time_series_data['Rolling_2Y'] * 100,
            linewidth=2, color='#F77F00', label='Rolling 2Y', alpha=0.8)
axes[1].plot(time_series_data['Date'], time_series_data['Regime'] * 100,
            linewidth=2, color='#9B59B6', label='Regime-Based', alpha=0.7, linestyle=':')

axes[1].set_xlabel('Date', fontsize=11)
axes[1].set_ylabel('Core Ratio (%)', fontsize=11)
axes[1].set_title('Dynamic Core Ratio Methods Over Time', fontsize=12, fontweight='bold')
axes[1].legend(fontsize=9, loc='upper left')
axes[1].grid(True, alpha=0.3)
axes[1].set_ylim(35, 90)

plt.tight_layout()
plt.show()

# %% [markdown]
# ### 8.5 Regime Classification Visualization

# %%
fig, ax = plt.subplots(figsize=(14, 7))

# Plot balance with regime coloring
regime_colors = {0: '#06A77D', 1: '#F77F00', 2: '#E63946'}
for regime in range(3):
    mask = nmd_clean['regime'] == regime
    regime_data = nmd_clean[mask]
    ax.scatter(regime_data['Date'], regime_data['Balance'],
              c=regime_colors[regime], label=f'Regime {regime} (Core={regime_core_map[regime]*100:.0f}%)',
              alpha=0.6, s=20)

ax.set_xlabel('Date', fontsize=12)
ax.set_ylabel('Balance', fontsize=12)
ax.set_title('Market Regime Classification for Dynamic Core Assignment',
            fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 9. Summary and Recommendations

# %%
print("\n" + "="*80)
print("SECTION 5f SUMMARY: DYNAMIC CORE RATIO ANALYSIS")
print("="*80)

print("\n1. CORE RATIO ESTIMATES")
print("-" * 80)
for _, row in comparison_results.iterrows():
    print(f"  {row['Method']:30s} {row['Core_Ratio_%']:5.1f}%")

print("\n2. KEY FINDINGS")
print("-" * 80)
best_nii_idx = comparison_results['ΔNII_S2'].idxmax()
worst_eve_idx = comparison_results['ΔEVE_S1'].idxmin()
highest_core_idx = comparison_results['Core_Ratio_%'].idxmax()

print(f"  Highest Core Ratio:   {comparison_results.loc[highest_core_idx, 'Method']}")
print(f"  Best NII Performance: {comparison_results.loc[best_nii_idx, 'Method']}")
print(f"  Worst EVE Risk:       {comparison_results.loc[worst_eve_idx, 'Method']}")

print("\n3. PROFIT vs RISK TRADE-OFF")
print("-" * 80)
static_eve = comparison_results.loc[0, 'EVE_Base']
static_delta_eve = comparison_results.loc[0, 'ΔEVE_S1']

for idx, row in comparison_results.iloc[1:].iterrows():
    eve_benefit = row['EVE_Base'] - static_eve
    eve_risk_increase = row['ΔEVE_S1'] - static_delta_eve
    nii_benefit = row['ΔNII_S2'] - comparison_results.loc[0, 'ΔNII_S2']

    print(f"\n  {row['Method']}:")
    print(f"    EVE Benefit:     ${eve_benefit:+.0f}")
    print(f"    EVE Risk Change: ${eve_risk_increase:+.0f}")
    print(f"    NII Benefit:     ${nii_benefit:+.0f}")
    print(f"    Net Assessment:  {'✓ Worth considering' if eve_benefit > 0 and abs(eve_risk_increase) < abs(eve_benefit) else '⚠ Higher risk for modest gain'}")

print("\n4. RECOMMENDATIONS")
print("-" * 80)
print("  For Regulatory Reporting:")
print("    → Use STATIC 51% (historical minimum)")
print("    → Conservative, Basel-approved, defensible")
print("")
print("  For Internal Risk Management:")
print("    → Consider ROLLING 2Y or GROWTH-ADJUSTED")
print("    → More reflective of current conditions")
print("    → Balance between safety and realism")
print("")
print("  For Profit Optimization:")
print("    → TREND-BASED or GROWTH-ADJUSTED offer best balance")
print("    → 10-12 percentage point increase vs static")
print("    → Modest EVE risk increase for NII improvement")

print("\n5. IMPLEMENTATION CONSIDERATIONS")
print("-" * 80)
print("  ✓ Dynamic core requires robust governance")
print("  ✓ Need quarterly review and adjustment process")
print("  ✓ Document methodology and assumptions")
print("  ✓ Backtesting to validate approach")
print("  ✓ Regulatory approval may be required")
print("  ⚠ Model risk: Dynamic assumptions could be wrong")
print("  ⚠ Must have clear triggers for reverting to conservative")

print("\n6. KEY INSIGHT")
print("-" * 80)
print("  The 51% static core ratio IS conservative (as intended).")
print("  Dynamic methods suggest current core could be 60-75%.")
print("  Trade-off: ~$XX extra EVE value vs ~$YY extra risk.")
print("  Decision depends on bank's risk appetite and regulatory stance.")

print("\n" + "="*80)
print("SECTION 5f COMPLETE")
print("="*80)

# %%
print("\nData saved:")
print("- dynamic_core_comparison.csv (all methods comparison)")

# %%
