# %% [markdown]
# # Phase 3: EVE (Economic Value of Equity) Sensitivity
#
# ## QF609 Project #1 - IRRBB Modelling Framework
#
# **Calculation Date:** 31-Dec-2023
#
# **Method:**
# EVE = Σ CF_i × DF(t_i)      where DF(t) = 1 / (1 + r(t))^t
# ΔEVE = EVE_shocked - EVE_base
#
# **4 Rate Shock Scenarios (per project brief):**
# (a) +200 bps parallel shift — all tenors up by 200 bps
# (b) −200 bps parallel shift — all tenors down by 200 bps (zero floor)
# (c) Short-rate up / Steepener — +200 bps at shortest tenor, tapering to 0 bps at 5Y (Basel cap)
# (d) Flattener — short end: +200 bps at t→0, tapering to 0 at pivot 2Y; long end: 0 at 2Y, −100 bps at 5Y

# %% [markdown]
# ## 1. Imports and Data Load

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.interpolate import interp1d
import warnings

warnings.filterwarnings('ignore')
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (14, 7)
plt.rcParams['font.size'] = 10

# ── Load inputs ───────────────────────────────────────────────────────────────
repricing = pd.read_csv('repricing_profile.csv')
curve     = pd.read_csv('processed_curve_data.csv')

print("=" * 80)
print("PHASE 3: EVE SENSITIVITY ANALYSIS")
print("=" * 80)
print(f"\nRepricing buckets:   {len(repricing)}")
print(f"Curve tenors:        {len(curve)}")
print(f"Total cash flow:     {repricing['Total_CF'].sum():,.2f}")
print(f"Curve tenors (1D–10Y):  {len(curve)}")

# %%
def build_df_func(tenors_yrs, zero_rates):
    """
    Returns a callable DF(t) using log-linear interpolation.

    Construction:
      1. Compute DF nodes:   DF_i = 1 / (1 + r_i)^t_i
      2. Take log:           log_DF_i = ln(DF_i)
      3. Build piecewise-linear interp on (t_i, log_DF_i)
      4. Recover DF(t) = exp( interp(t) )

    Annual compounding convention.
    Extrapolation: flat forward rate beyond last tenor.
    """
    # Step 1: discount factor at each known tenor
    df_nodes = 1.0 / (1.0 + zero_rates) ** tenors_yrs

    # Step 2: log-transform (makes interpolation linear in log-space)
    log_df = np.log(df_nodes)

    # Step 3: piecewise-linear interpolator on log(DF)
    interp = interp1d(tenors_yrs, log_df, kind='linear',
                      bounds_error=False, fill_value='extrapolate')

    # Step 4: return a function that exponentiates back to DF space
    return lambda t: float(np.exp(interp(t))) if np.ndim(t) == 0 \
                     else np.exp(interp(np.asarray(t, dtype=float)))


def build_rate_interp(tenors_yrs, zero_rates):
    """Linear interpolation of zero rates (for shocked curve construction)."""
    return interp1d(tenors_yrs, zero_rates, kind='linear',
                    bounds_error=False, fill_value='extrapolate')

tenors_yrs = curve['Tenor_Years'].values
base_rates  = curve['ZeroRate'].values

df_base = build_df_func(tenors_yrs, base_rates)

rate_interp_base = build_rate_interp(tenors_yrs, base_rates)

# %% [markdown]
# ## 3. Base Case EVE
#
# ### How EVE_base is calculated — step by step
#
# EVE (Economic Value of Equity) is the present value of ALL expected future
# cash flows from NMD deposits, discounted at current market zero rates.
#
# Under IRRBB, deposits are modelled as a portfolio of zero-coupon cash flows
# slotted into repricing time buckets (Phase 2 output).  Each bucket i has:
#   CF_i      — the cash flow (core deposit principal) maturing in that bucket
#   t_i       — the bucket midpoint (representative repricing date in years)
#   DF(t_i)   — discount factor at that midpoint using the base zero curve
#
# FORMULA:
#
#   EVE_base = Σ_i  CF_i  ×  DF_base(t_i)
#
#            = Σ_i  CF_i  ×  1 / (1 + r_base(t_i))^t_i
#
# Each term CF_i × DF(t_i) = PV_i is the present value of bucket i's cash flow.
# Summing all PV_i gives the total EVE — the economic value of the deposit book.
#
# Midpoint approximation:
#   Each bucket's cash flow is treated as a single zero-coupon amount occurring
#   at the bucket midpoint t_i. This is a standard IRRBB simplification adopted
#   across all phases of this project (consistent with Phase 2 slotting).
#   It implicitly assumes uniform distribution of cash flows within each bucket.
#
# Interpretation:
#   EVE_base  > Total_CF  → impossible (discount factors < 1 for t > 0)
#   EVE_base  < Total_CF  → the gap is the "cost of time" (time value of money)
#   A HIGHER rate environment → lower DF → lower EVE
#   A LOWER  rate environment → higher DF → higher EVE

# %%
# Step 1: compute DF at each bucket midpoint using the base curve
repricing['DF_Base'] = repricing['Midpoint_Years'].apply(df_base)

# Step 2: compute PV of each bucket  →  PV_i = CF_i × DF_base(t_i)
repricing['PV_Base'] = repricing['Total_CF'] * repricing['DF_Base']

# Step 3: sum all bucket PVs  →  EVE_base = Σ PV_i
eve_base = repricing['PV_Base'].sum()

print("\n" + "=" * 80)
print("BASE CASE EVE — STEP-BY-STEP CALCULATION")
print("=" * 80)
print()
print("  Formula:  EVE_base = Σ  CF_i × DF_base(t_i)")
print("                     = Σ  CF_i × 1 / (1 + r_base(t_i))^t_i")
print()
print(f"  {'Bucket':>6}  {'t_mid (Y)':>10}  {'CF_i':>12}  "
      f"{'DF_base(t)':>12}  {'PV_i = CF_i × DF':>18}  {'% of EVE':>9}")
print("  " + "-" * 77)
for _, row in repricing.iterrows():
    pv_pct = row['PV_Base'] / eve_base * 100
    print(f"  {row['Bucket']:>6}  {row['Midpoint_Years']:>10.4f}  "
          f"{row['Total_CF']:>12,.2f}  {row['DF_Base']:>12.6f}  "
          f"{row['PV_Base']:>18,.2f}  {pv_pct:>8.2f}%")
print("  " + "-" * 77)
print(f"  {'TOTAL':>6}  {'':>10}  {repricing['Total_CF'].sum():>12,.2f}  "
      f"{'':>12}  {eve_base:>18,.2f}  {'100.00%':>9}")
print()
print(f"  Total CF (undiscounted):       {repricing['Total_CF'].sum():>12,.2f}")
print(f"  EVE Base (present value):      {eve_base:>12,.2f}")
print(f"  Discount effect (CF − EVE):    {repricing['Total_CF'].sum() - eve_base:>12,.2f}")
print(f"  Average DF (EVE / Total_CF):   "
      f"{eve_base / repricing['Total_CF'].sum():>12.6f}")

# %% [markdown]
# ## 4. Shock Functions (4 Scenarios)

# %%
# ── Modelling assumption: Zero Lower Bound ───────────────────────────────────
# All shocked zero rates are floored at 0% via np.maximum(..., 0.0).
# This prevents negative rates in all scenarios, including the −200 bps
# parallel and the long-end of the flattener. This is a modelling assumption
# for simplicity; actual regulatory practice (e.g. BCBS d368) may permit
# negative rates in certain jurisdictions. Results should be interpreted with
# this constraint in mind.
# ─────────────────────────────────────────────────────────────────────────────

# ── Scenario (a): +200 bps Parallel ──────────────────────────────────────────
def shock_parallel_up(base_r, bps=200):
    return np.maximum(base_r + bps / 10_000, 0.0)

# ── Scenario (b): −200 bps Parallel (zero floor) ────────────────────────────
def shock_parallel_down(base_r, bps=200):
    return np.maximum(base_r - bps / 10_000, 0.0)

# ── Scenario (c): Short-rate up / Steepener ──────────────────────────────────
# +200 bps at shortest tenor, tapering linearly to 0 bps at 5Y (Basel cap).
# Taper ends at 5Y: deposits beyond 5Y are capped at 5Y per Basel, so no
# repricing bucket midpoint exceeds 4.5Y — using 5Y as taper end is sufficient
# and avoids carrying non-zero shock beyond the repricing horizon.
# shock(t) = 200 × max(1 − t/5, 0)  bps
def shock_steepener(t_arr, base_r, max_bps=200, taper_end_yr=5):
    add = max_bps * np.maximum(1 - t_arr / taper_end_yr, 0.0) / 10_000
    return np.maximum(base_r + add, 0.0)

# ── Scenario (d): Flattener ───────────────────────────────────────────────────
# Pivot at 2Y so that long-end buckets (3Y, 4Y, 5Y) actually receive NEGATIVE
# shocks within the repricing horizon. Without an early pivot, all bucket
# midpoints (max 4.5Y) fall in the positive region and the flattener would
# behave identically to the steepener — defeating its purpose.
#
# Short end: +200 bps at t→0, tapers linearly to 0 bps at pivot (2Y)
# Long  end:   0 bps at pivot (2Y), tapers to −100 bps at 5Y (min: −100 bps)
#
# Shocks at bucket midpoints:
#   O/N (0.003Y): +200 bps   1Y (0.875Y): +112 bps
#   1M  (0.042Y): +196 bps   2Y (1.5Y):    +50 bps
#   3M  (0.208Y): +179 bps   3Y (2.5Y):    −17 bps
#   6M  (0.375Y): +163 bps   4Y (3.5Y):    −50 bps
#   9M  (0.625Y): +138 bps   5Y (4.5Y):    −83 bps
#
# shock(t) = +200×(1 − t/2)        for t ≤ 2Y   (max: +200 bps)
#           = −100×(t − 2)/(5−2)   for t > 2Y   (min: −100 bps at 5Y)
def shock_flattener(t_arr, base_r, short_max_bps=200, long_min_bps=-100,
                    pivot_yr=2, long_end_yr=5):
    add = np.where(
        t_arr <= pivot_yr,
        short_max_bps * (1 - t_arr / pivot_yr),
        long_min_bps  * (t_arr - pivot_yr) / (long_end_yr - pivot_yr)
    ) / 10_000
    return np.maximum(base_r + add, 0.0)

# ── Build shocked curves ──────────────────────────────────────────────────────
rates_a = shock_parallel_up(base_rates)
rates_b = shock_parallel_down(base_rates)
rates_c = shock_steepener(tenors_yrs, base_rates)
rates_d = shock_flattener(tenors_yrs, base_rates)

# Display shocked curves table
shocked_curves = pd.DataFrame({
    'Tenor'         : curve['Tenor'],
    'Tenor_Years'   : tenors_yrs,
    'Base (bps)'    : (base_rates  * 10_000).round(1),
    '(a)+200 Par'   : (rates_a     * 10_000).round(1),
    '(b)-200 Par'   : (rates_b     * 10_000).round(1),
    '(c) Steepener' : (rates_c     * 10_000).round(1),
    '(d) Flattener' : (rates_d     * 10_000).round(1),
})

print("\n" + "=" * 80)
print("SHOCKED ZERO CURVES (basis points) — tenors up to 5Y")
print("=" * 80)
print(shocked_curves[shocked_curves['Tenor_Years'] <= 5].to_string(index=False))

# %% [markdown]
# ## 5. EVE Under Each Shocked Scenario

# %%
df_a = build_df_func(tenors_yrs, rates_a)
df_b = build_df_func(tenors_yrs, rates_b)
df_c = build_df_func(tenors_yrs, rates_c)
df_d = build_df_func(tenors_yrs, rates_d)

for scen, func, col_df, col_pv in [
    ('a', df_a, 'DF_ScenA', 'PV_ScenA'),
    ('b', df_b, 'DF_ScenB', 'PV_ScenB'),
    ('c', df_c, 'DF_ScenC', 'PV_ScenC'),
    ('d', df_d, 'DF_ScenD', 'PV_ScenD'),
]:
    repricing[col_df] = repricing['Midpoint_Years'].apply(func)
    repricing[col_pv] = repricing['Total_CF'] * repricing[col_df]

eve_a = repricing['PV_ScenA'].sum()
eve_b = repricing['PV_ScenB'].sum()
eve_c = repricing['PV_ScenC'].sum()
eve_d = repricing['PV_ScenD'].sum()

delta_a = eve_a - eve_base
delta_b = eve_b - eve_base
delta_c = eve_c - eve_base
delta_d = eve_d - eve_base

pct_a = delta_a / eve_base * 100
pct_b = delta_b / eve_base * 100
pct_c = delta_c / eve_base * 100
pct_d = delta_d / eve_base * 100

# ── EVE summary table ─────────────────────────────────────────────────────────
eve_summary = pd.DataFrame({
    'Scenario': [
        'Base Case',
        '(a) +200 bps Parallel',
        '(b) −200 bps Parallel',
        '(c) Steepener (Short Up)',
        '(d) Flattener',
    ],
    'EVE': [eve_base, eve_a, eve_b, eve_c, eve_d],
    'ΔEVE': [0.0, delta_a, delta_b, delta_c, delta_d],
    'ΔEVE_%': [0.0, pct_a, pct_b, pct_c, pct_d],
})

print("\n" + "=" * 80)
print("EVE SENSITIVITY SUMMARY")
print("=" * 80)
print(f"\n{'Scenario':<30}  {'EVE':>12}  {'ΔEVE':>10}  {'ΔEVE %':>8}")
print("-" * 66)
for _, row in eve_summary.iterrows():
    marker = " ← WORST" if row['ΔEVE'] == eve_summary['ΔEVE'].min() and row['ΔEVE'] < 0 else ""
    print(f"{row['Scenario']:<30}  {row['EVE']:>12,.2f}  {row['ΔEVE']:>+10,.2f}  {row['ΔEVE_%']:>+7.2f}%{marker}")

# ── Worst-case identification ─────────────────────────────────────────────────
worst_idx      = eve_summary.iloc[1:]['ΔEVE'].idxmin()
worst_scenario = eve_summary.loc[worst_idx, 'Scenario']
worst_delta    = eve_summary.loc[worst_idx, 'ΔEVE']
worst_pct      = eve_summary.loc[worst_idx, 'ΔEVE_%']

print("\n" + "=" * 80)
print("WORST-CASE EVE (Binding IRRBB Measure)")
print("=" * 80)
print(f"  Scenario:       {worst_scenario}")
print(f"  EVE:            {eve_summary.loc[worst_idx, 'EVE']:,.2f}")
print(f"  ΔEVE:           {worst_delta:+,.2f}")
print(f"  ΔEVE %:         {worst_pct:+.4f}%")

# ── Bucket-level ΔPV breakdown ────────────────────────────────────────────────
for col_pv, label in [('PV_ScenA','(a)'), ('PV_ScenB','(b)'),
                       ('PV_ScenC','(c)'), ('PV_ScenD','(d)')]:
    repricing[f'ΔPV_{label}'] = repricing[col_pv] - repricing['PV_Base']

# ── Duration & Convexity (from parallel shocks) ───────────────────────────────
delta_y             = 0.02   # 200 bps
effective_duration  = (eve_b - eve_a) / (2 * eve_base * delta_y)
effective_convexity = (eve_b + eve_a - 2 * eve_base) / (eve_base * delta_y ** 2)

print("\n" + "=" * 80)
print("DURATION & CONVEXITY (Parallel Shocks ±200 bps)")
print("=" * 80)
print(f"  Effective Duration:   {effective_duration:.4f} years")
print(f"  Effective Convexity:  {effective_convexity:.4f}")
print(f"\n  Sign convention (liability book):")
print(f"  Deposits are liabilities. A NEGATIVE duration means EVE falls when")
print(f"  rates rise — consistent with the deposit book being short duration.")
print(f"  Duration = {effective_duration:.4f}Y → 1% rate ↑ → EVE changes by ≈ {effective_duration:.2f}%")
print(f"  (negative = EVE loss; positive = EVE gain for a 1% rate increase)")
print(f"\n  Sanity checks:")
print(f"  EVE_base < Total_CF:        {eve_base:.2f} < {repricing['Total_CF'].sum():.2f}  →  "
      f"{'PASS' if eve_base < repricing['Total_CF'].sum() else 'FAIL'}")
symmetry_ratio = abs(delta_b + delta_a) / (0.5 * (abs(delta_a) + abs(delta_b))) * 100
print(f"  Shock asymmetry |b+a|/avg:  {symmetry_ratio:.2f}%  →  "
      f"{'PASS (convexity expected)' if symmetry_ratio < 20 else 'CHECK — high asymmetry'}")
print(f"  Convexity > 0:              {effective_convexity:.4f}  →  "
      f"{'PASS' if effective_convexity > 0 else 'FAIL — negative convexity unexpected'}")

# %% [markdown]
# ## 6. Save Outputs (before visualisations)

# %%
# Save EVE sensitivity summary
eve_summary.to_csv('eve_sensitivity_summary.csv', index=False)

# Save full bucket-level detail with all DFs and PVs
repricing.to_csv('repricing_profile_with_eve.csv', index=False)

# Save shocked curves
shocked_curves.to_csv('shocked_yield_curves.csv', index=False)

# Save key metrics
key_metrics = pd.DataFrame({
    'Metric': ['Base_EVE', 'EVE_ScenA', 'EVE_ScenB', 'EVE_ScenC', 'EVE_ScenD',
               'ΔEVE_a', 'ΔEVE_b', 'ΔEVE_c', 'ΔEVE_d',
               'ΔEVE_%_a', 'ΔEVE_%_b', 'ΔEVE_%_c', 'ΔEVE_%_d',
               'Worst_Case_Scenario', 'Worst_ΔEVE', 'Worst_ΔEVE_%',
               'Effective_Duration', 'Effective_Convexity'],
    'Value': [eve_base, eve_a, eve_b, eve_c, eve_d,
              delta_a, delta_b, delta_c, delta_d,
              pct_a, pct_b, pct_c, pct_d,
              worst_scenario, worst_delta, worst_pct,
              effective_duration, effective_convexity]
})
key_metrics.to_csv('eve_key_metrics.csv', index=False)

print("Files saved:")
print("  eve_sensitivity_summary.csv")
print("  repricing_profile_with_eve.csv")
print("  shocked_yield_curves.csv")
print("  eve_key_metrics.csv")

# %% [markdown]
# ## 7. Visualisations

# %%
# ── Plot 1: All shocked curves vs base ───────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
ax = axes[0]
t_plot = np.linspace(0, 5, 200)
r_interp = {
    'Base'          : build_rate_interp(tenors_yrs, base_rates)(t_plot),
    '(a) +200 bps'  : build_rate_interp(tenors_yrs, rates_a)(t_plot),
    '(b) −200 bps'  : build_rate_interp(tenors_yrs, rates_b)(t_plot),
    '(c) Steepener' : build_rate_interp(tenors_yrs, rates_c)(t_plot),
    '(d) Flattener' : build_rate_interp(tenors_yrs, rates_d)(t_plot),
}
colors = ['#1E3358', '#EF4444', '#0D9488', '#F59E0B', '#8B5CF6']
styles = ['-', '--', '--', '-.', ':']
for (label, rates), color, ls in zip(r_interp.items(), colors, styles):
    ax.plot(t_plot, rates * 100, label=label, color=color,
            linewidth=2.5, linestyle=ls)
ax.scatter(tenors_yrs, base_rates * 100, color='#1E3358', zorder=5, s=40)
ax.set_facecolor('#F0F4F8')
ax.set_xlabel('Tenor (Years)', fontsize=11, color='#44546A')
ax.set_ylabel('Zero Rate (%)', fontsize=11, color='#44546A')
ax.set_title('Interest Rate Curves under 4 rate shock scenarios', fontweight='bold', fontsize=12, color='#1A2744')
ax.legend(fontsize=9, framealpha=0.9)
ax.grid(True, alpha=0.3, color='#94A3B8')
ax.tick_params(colors='#44546A')
ax.spines[['top', 'right']].set_visible(False)
ax.set_xlim(0, 5)

# ── Plot 2: ΔEVE bar chart ────────────────────────────────────────────────────
ax = axes[1]
labels   = ['(a) +200 bps\nParallel', '(b) −200 bps\nParallel',
            '(c) Steepener\n(Short Up)', '(d) Flattener']
deltas   = [delta_a, delta_b, delta_c, delta_d]
bar_cols = ['#EF4444' if d < 0 else '#0D9488' for d in deltas]
bars = ax.bar(labels, deltas, color=bar_cols, edgecolor='white',
              linewidth=0.8, alpha=0.92, width=0.55)
ax.set_facecolor('#F0F4F8')
# Highlight worst case with amber border
worst_bar_idx = int(np.argmin(deltas))
bars[worst_bar_idx].set_edgecolor('#92400E')
bars[worst_bar_idx].set_linewidth(2.5)
for bar, d, p in zip(bars, deltas, [pct_a, pct_b, pct_c, pct_d]):
    offset = max(abs(d) * 0.05, 3)
    ypos   = d + offset if d >= 0 else d - offset
    ax.text(bar.get_x() + bar.get_width() / 2, ypos,
            f'{d:+,.1f}\n({p:+.2f}%)',
            ha='center', va='bottom' if d >= 0 else 'top',
            fontsize=9, fontweight='bold', color='#1A2744')
ax.axhline(0, color='#64748B', linewidth=1.2)
ax.set_ylabel('ΔEVE', fontsize=11, color='#44546A')
ax.set_title('ΔEVE by Shock Scenario', fontweight='bold', fontsize=12, color='#1A2744')
ax.tick_params(colors='#44546A')
ax.grid(True, alpha=0.3, axis='y', color='#94A3B8')
ax.spines[['top', 'right']].set_visible(False)

plt.tight_layout()
plt.savefig('eve_sensitivity_results.png', dpi=300, bbox_inches='tight')
plt.show()

# ── Table 3: PV by bucket × scenario (printed) ───────────────────────────────
print("\n" + "=" * 95)
print("PV BY TIME BUCKET & SCENARIO")
print("=" * 95)
print(f"  {'Bucket':>6}  {'t_mid':>6}  {'CF':>12}  {'PV Base':>12}  "
      f"{'PV (a)+200':>12}  {'PV (b)−200':>12}  {'PV (c)Steep':>12}  {'PV (d)Flat':>12}")
print("  " + "-" * 91)
for _, row in repricing.iterrows():
    print(f"  {row['Bucket']:>6}  {row['Midpoint_Years']:>6.3f}  "
          f"{row['Total_CF']:>12,.2f}  {row['PV_Base']:>12,.2f}  "
          f"{row['PV_ScenA']:>12,.2f}  {row['PV_ScenB']:>12,.2f}  "
          f"{row['PV_ScenC']:>12,.2f}  {row['PV_ScenD']:>12,.2f}")
print("  " + "-" * 91)
print(f"  {'TOTAL':>6}  {'':>6}  {repricing['Total_CF'].sum():>12,.2f}  "
      f"{eve_base:>12,.2f}  {eve_a:>12,.2f}  {eve_b:>12,.2f}  "
      f"{eve_c:>12,.2f}  {eve_d:>12,.2f}")

print("\n" + "=" * 95)
print("ΔPV vs BASE BY TIME BUCKET & SCENARIO")
print("=" * 95)
print(f"  {'Bucket':>6}  {'t_mid':>6}  {'CF':>12}  {'PV Base':>12}  "
      f"{'ΔPV (a)+200':>13}  {'ΔPV (b)−200':>13}  {'ΔPV (c)Steep':>13}  {'ΔPV (d)Flat':>13}")
print("  " + "-" * 91)
for _, row in repricing.iterrows():
    print(f"  {row['Bucket']:>6}  {row['Midpoint_Years']:>6.3f}  "
          f"{row['Total_CF']:>12,.2f}  {row['PV_Base']:>12,.2f}  "
          f"{row['ΔPV_(a)']:>+13,.2f}  {row['ΔPV_(b)']:>+13,.2f}  "
          f"{row['ΔPV_(c)']:>+13,.2f}  {row['ΔPV_(d)']:>+13,.2f}")
print("  " + "-" * 91)
print(f"  {'TOTAL':>6}  {'':>6}  {repricing['Total_CF'].sum():>12,.2f}  "
      f"{eve_base:>12,.2f}  "
      f"{delta_a:>+13,.2f}  {delta_b:>+13,.2f}  "
      f"{delta_c:>+13,.2f}  {delta_d:>+13,.2f}")

# ── Plot 4: Waterfall — bucket ΔPV contributions per scenario ────────────────
# Presentation colour constants
WF_POS   = '#0D9488'   # teal  — positive ΔPV bucket
WF_NEG   = '#EF4444'   # red   — negative ΔPV bucket
WF_TOT_P = '#22C55E'   # green — total ΔEVE positive
WF_TOT_N = '#EF4444'   # red   — total ΔEVE negative
WF_CONN  = '#64748B'   # slate — connector lines
WF_BG    = '#F0F4F8'   # off-white — subplot background

fig, axes = plt.subplots(2, 2, figsize=(14, 8))
fig.patch.set_facecolor('#F0F4F8')

scenarios_wf = [
    ('(a) +200 bps Parallel',    'ΔPV_(a)', delta_a),
    ('(b) −200 bps Parallel',    'ΔPV_(b)', delta_b),
    ('(c) Steepener (Short Up)', 'ΔPV_(c)', delta_c),
    ('(d) Flattener',            'ΔPV_(d)', delta_d),
]

for ax, (label, col, total_delta) in zip(axes.flat, scenarios_wf):
    ax.set_facecolor(WF_BG)
    bucket_labels = repricing['Bucket'].tolist()
    values        = repricing[col].tolist()
    n             = len(values)

    # Compute floating bar bottoms (waterfall logic)
    bottoms  = []
    running  = 0.0
    for v in values:
        bottoms.append(running if v >= 0 else running + v)
        running += v

    # Bar colours: teal = positive ΔPV, red = negative ΔPV
    bar_colors = [WF_POS if v >= 0 else WF_NEG for v in values]

    # Plot incremental (bucket) bars
    ax.bar(range(n), [abs(v) for v in values], bottom=bottoms,
           color=bar_colors, edgecolor='white', linewidth=0.6,
           alpha=0.92, width=0.55)

    # Connector lines linking top of each bar to base of next
    cum = 0.0
    for i, v in enumerate(values[:-1]):
        cum += v
        ax.plot([i + 0.275, i + 0.725], [cum, cum],
                color=WF_CONN, linewidth=1.0, linestyle='--')

    # Total ΔEVE bar (rightmost)
    total_bottom = min(0.0, total_delta)
    total_color  = WF_TOT_P if total_delta >= 0 else WF_TOT_N
    ax.bar(n, abs(total_delta), bottom=total_bottom,
           color=total_color, edgecolor='white', linewidth=1.2,
           alpha=0.95, width=0.55)

    # Value labels inside each bucket bar
    for i, (v, bot) in enumerate(zip(values, bottoms)):
        mid = bot + abs(v) / 2
        ax.text(i, mid, f'{v:+.1f}', ha='center', va='center',
                fontsize=6.5, fontweight='bold', color='white')

    # Value label inside total bar
    ax.text(n, total_bottom + abs(total_delta) / 2,
            f'{total_delta:+.1f}', ha='center', va='center',
            fontsize=7.5, fontweight='bold', color='white')

    # Zero baseline
    ax.axhline(0, color='#44546A', linewidth=1.2)

    # X-axis: bucket labels + "TOTAL"
    ax.set_xticks(range(n + 1))
    ax.set_xticklabels(bucket_labels + ['TOTAL'],
                       rotation=40, ha='right', fontsize=8, color='#44546A')

    ax.set_title(f'{label}\nΔEVE = {total_delta:+,.1f}',
                 fontweight='bold', fontsize=9, color='#1A2744')
    ax.set_ylabel('ΔPV', fontsize=10, color='#44546A')
    ax.tick_params(colors='#44546A')
    ax.grid(True, alpha=0.25, axis='y', color='#94A3B8')
    ax.spines[['top', 'right']].set_visible(False)

# Shared legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor=WF_POS,   edgecolor='white', label='Positive ΔPV (EVE gain)'),
    Patch(facecolor=WF_NEG,   edgecolor='white', label='Negative ΔPV (EVE loss)'),
    Patch(facecolor=WF_TOT_P, edgecolor='white', label='Total ΔEVE (positive)'),
    Patch(facecolor=WF_TOT_N, edgecolor='white', label='Total ΔEVE (negative)'),
]
fig.legend(handles=legend_elements, loc='lower center', ncol=4,
           fontsize=9, framealpha=0.9, bbox_to_anchor=(0.5, -0.02))

plt.suptitle('Waterfall: Bucket-Level ΔPV Contributions to ΔEVE',
             fontweight='bold', fontsize=11, color='#1A2744', y=1.01)
plt.tight_layout()
plt.savefig('eve_waterfall.png', dpi=300, bbox_inches='tight')
plt.show()

# ── Plot 5: Rate Sensitivity Ladder ──────────────────────────────────────────
# Sweep parallel shifts from -300 to +300 bps; compute ΔEVE at each step
shifts_bps = np.arange(-300, 301, 50)
ladder_eve = []
for shift in shifts_bps:
    rates_shifted = np.maximum(base_rates + shift / 10_000, 0.0)
    df_shifted    = build_df_func(tenors_yrs, rates_shifted)
    eve_shifted   = sum(
        row['Total_CF'] * df_shifted(row['Midpoint_Years'])
        for _, row in repricing.iterrows()
    )
    ladder_eve.append(eve_shifted - eve_base)

fig, ax = plt.subplots(figsize=(10, 5))
fig.patch.set_facecolor('#F0F4F8')
ax.set_facecolor('#F0F4F8')

ax.plot(shifts_bps, ladder_eve, color='#1E3358', linewidth=2.5, zorder=3)
ax.fill_between(shifts_bps, ladder_eve, 0,
                where=[v < 0 for v in ladder_eve],
                alpha=0.15, color='#EF4444', label='EVE loss region')
ax.fill_between(shifts_bps, ladder_eve, 0,
                where=[v >= 0 for v in ladder_eve],
                alpha=0.15, color='#0D9488', label='EVE gain region')

# Mark the 4 Basel scenarios
scenario_pts = [
    ('(a) +200', 200,  delta_a, '#EF4444'),
    ('(b) −200', -200, delta_b, '#0D9488'),
    ('(c) Steep\n≈+80avg', None, delta_c, '#F59E0B'),
    ('(d) Flat\n≈+50avg',  None, delta_d, '#8B5CF6'),
]
for label, shift, delta, col in scenario_pts[:2]:
    ax.scatter(shift, delta, color=col, zorder=5, s=70)
    ax.annotate(label, xy=(shift, delta),
                xytext=(shift + 15, delta - 60),
                fontsize=8, color=col, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color=col, lw=1.2))

ax.axhline(0, color='#44546A', linewidth=1.0, linestyle='--')
ax.axvline(0, color='#44546A', linewidth=0.8, linestyle=':')
ax.set_xlabel('Parallel Rate Shift (bps)', fontsize=11, color='#44546A')
ax.set_ylabel('ΔEVE', fontsize=11, color='#44546A')
ax.set_title('Rate Sensitivity Ladder\nΔEVE across Parallel Rate Shifts (−300 to +300 bps)',
             fontweight='bold', fontsize=12, color='#1A2744')
ax.legend(fontsize=9, framealpha=0.9)
ax.tick_params(colors='#44546A')
ax.spines[['top', 'right']].set_visible(False)
ax.grid(True, alpha=0.25, color='#94A3B8')
plt.tight_layout()
plt.savefig('eve_rate_ladder.png', dpi=300, bbox_inches='tight')
plt.show()

# ── Plot 6: Tornado Chart ────────────────────────────────────────────────────
tornado_labels  = ['(a) +200 bps Parallel', '(b) −200 bps Parallel',
                   '(c) Steepener (Short Up)', '(d) Flattener']
tornado_deltas  = [delta_a, delta_b, delta_c, delta_d]

# Sort by absolute ΔEVE descending
order           = sorted(range(4), key=lambda i: abs(tornado_deltas[i]), reverse=True)
sorted_labels   = [tornado_labels[i] for i in order]
sorted_deltas   = [tornado_deltas[i] for i in order]
bar_cols        = ['#EF4444' if d < 0 else '#0D9488' for d in sorted_deltas]

fig, ax = plt.subplots(figsize=(10, 4))
fig.patch.set_facecolor('#F0F4F8')
ax.set_facecolor('#F0F4F8')

bars = ax.barh(range(len(sorted_labels)), sorted_deltas,
               color=bar_cols, edgecolor='white', linewidth=0.8,
               height=0.5, alpha=0.92)

# Value labels at bar tips
for i, (bar, val) in enumerate(zip(bars, sorted_deltas)):
    offset = 8 if val >= 0 else -8
    ha     = 'left' if val >= 0 else 'right'
    ax.text(val + offset, i, f'{val:+,.1f}',
            va='center', ha=ha, fontsize=10, fontweight='bold', color='#1A2744')

ax.axvline(0, color='#44546A', linewidth=1.2)
ax.set_yticks(range(len(sorted_labels)))
ax.set_yticklabels(sorted_labels, fontsize=10, color='#44546A')
ax.set_xlabel('ΔEVE', fontsize=11, color='#44546A')
ax.set_title('Tornado Chart — EVE Sensitivity by Scenario\n(ranked by |ΔEVE|)',
             fontweight='bold', fontsize=12, color='#1A2744')
ax.tick_params(colors='#44546A')
ax.spines[['top', 'right', 'left']].set_visible(False)
ax.grid(True, alpha=0.25, axis='x', color='#94A3B8')
ax.set_xlim(min(sorted_deltas) * 1.25, max(sorted_deltas) * 1.25)
plt.tight_layout()
plt.savefig('eve_tornado.png', dpi=300, bbox_inches='tight')
plt.show()

# %% [markdown]
# ## 8. Final Summary

# %%
print("\n" + "=" * 80)
print("PHASE 3 COMPLETE — EVE SENSITIVITY SUMMARY")
print("=" * 80)
print(f"\n  Base EVE:                   {eve_base:>12,.2f}")
print(f"\n  {'Scenario':<30}  {'EVE':>12}  {'ΔEVE':>10}  {'ΔEVE%':>8}")
print("  " + "-" * 64)
for _, row in eve_summary.iterrows():
    print(f"  {row['Scenario']:<30}  {row['EVE']:>12,.2f}  "
          f"{row['ΔEVE']:>+10,.2f}  {row['ΔEVE_%']:>+7.2f}%")
print(f"\n  Worst-case scenario:        {worst_scenario}")
print(f"  Binding ΔEVE:               {worst_delta:+,.2f}  ({worst_pct:+.4f}%)")
print(f"  Effective Duration:         {effective_duration:.4f} years")
print(f"  Effective Convexity:        {effective_convexity:.4f}")
print(f"\n  Output files:")
print(f"    eve_sensitivity_summary.csv")
print(f"    repricing_profile_with_eve.csv")
print(f"    shocked_yield_curves.csv")
print(f"    eve_key_metrics.csv")
print(f"    eve_sensitivity_results.png")
print(f"    eve_waterfall.png")
print(f"    eve_rate_ladder.png")
print(f"    eve_tornado.png")
print("\n" + "=" * 80)

# %%
