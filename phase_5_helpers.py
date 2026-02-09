# %% [markdown]
# # Phase 5: Shared Helpers
#
# ## QF609 Project #1 - IRRBB Modelling Framework
#
# Shared imports, data loading, and helper functions used by all Phase 5 sub-scripts.

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

# %%
# Load all Phase 1-4 CSV outputs
nmd_data = pd.read_csv('processed_nmd_data.csv', parse_dates=['Date'])
curve_data = pd.read_csv('processed_curve_data.csv')
core_noncore = pd.read_csv('core_noncore_split.csv')
survival_full = pd.read_csv('survival_curve_full.csv')
decay_params = pd.read_csv('decay_parameters.csv')
repricing_profile = pd.read_csv('repricing_profile.csv')
eve_summary_orig = pd.read_csv('eve_sensitivity_summary.csv')
nii_summary_orig = pd.read_csv('nii_sensitivity_summary.csv')

# Extract key parameters
current_balance = core_noncore[core_noncore['Component'] == 'Total Balance']['Amount'].values[0]
core_amount = core_noncore[core_noncore['Component'] == 'Core Deposits']['Amount'].values[0]
non_core_amount = core_noncore[core_noncore['Component'] == 'Non-Core Deposits']['Amount'].values[0]
core_ratio_primary = core_amount / current_balance
lambda_daily = decay_params[decay_params['Parameter'] == 'lambda_daily']['Value'].values[0]

# Curve data
tenors_years = curve_data['Tenor_Years'].values
base_rates = curve_data['ZeroRate'].values


# --- Helper 1: Build discount factor function (from phase_3) ---
def build_discount_function(tenors_yrs, zero_rates):
    """Build interpolated discount factor function using log-linear on DFs."""
    discount_factors = 1 / (1 + zero_rates) ** tenors_yrs
    log_df = np.log(discount_factors)
    interp_func = interp1d(tenors_yrs, log_df, kind='linear',
                          bounds_error=False, fill_value='extrapolate')

    def get_discount_factor(t):
        return np.exp(interp_func(t))

    return get_discount_factor


# --- Helper 2: Survival probability (from phase_1b/phase_2) ---
def get_survival(day, lambda_d):
    """S(t) = (1 - lambda_d)^t"""
    return (1 - lambda_d) ** day


# --- Helper 3: Slot cash flows into 11 buckets (from phase_2) ---
def slot_cashflows(balance, core_ratio, lambda_d):
    """
    Slot NMD balance into 11 IRRBB time buckets.

    Non-core -> O/N bucket. Core -> distributed by survival decay.
    Returns DataFrame with same structure as repricing_profile.
    """
    core_amt = balance * core_ratio
    non_core_amt = balance * (1 - core_ratio)

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

    rows = []
    for b in buckets:
        s_start = get_survival(b['Start_Days'], lambda_d)
        s_end = get_survival(b['End_Days'], lambda_d)
        core_cf = core_amt * (s_start - s_end)

        if b['Bucket'] == 'O/N':
            total_cf = non_core_amt + core_cf
            nc_cf = non_core_amt
        else:
            total_cf = core_cf
            nc_cf = 0.0

        rows.append({
            'Bucket': b['Bucket'],
            'Start_Days': b['Start_Days'],
            'End_Days': b['End_Days'],
            'Midpoint_Years': b['Midpoint_Years'],
            'Core_CF': core_cf,
            'Non_Core_CF': nc_cf,
            'Total_CF': total_cf
        })

    return pd.DataFrame(rows)


# --- Helper 4: Compute EVE (from phase_3) ---
def compute_eve(repricing_df, tenors_yrs, rates):
    """EVE = sum(CF * DF) using log-linear interpolated discount factors."""
    df_func = build_discount_function(tenors_yrs, rates)
    dfs = repricing_df['Midpoint_Years'].apply(df_func)
    pvs = repricing_df['Total_CF'] * dfs
    return pvs.sum()


# --- Helper 5: Compute NII (from phase_4) ---
def compute_nii(repricing_df, shock_func):
    """
    dNII = sum(CF * shock(t) * (1 - t)) for buckets with midpoint <= 1Y.

    shock_func: function that takes t_years and returns shock in decimal.
    """
    nii_buckets = repricing_df[repricing_df['Midpoint_Years'] <= 1.0].copy()
    delta_nii = 0.0
    for _, row in nii_buckets.iterrows():
        t = row['Midpoint_Years']
        cf = row['Total_CF']
        shock = shock_func(t)
        time_factor = 1 - t
        delta_nii += cf * shock * time_factor
    return delta_nii


# --- Helper 6: Shock functions (from phase_3/phase_4) ---
def shock_s1(t):
    """S1: +200bps parallel"""
    return 0.02

def shock_s2(t):
    """S2: -200bps parallel"""
    return -0.02

def shock_s3(t):
    """S3: Steepener - +200bps tapering to 0 at 10Y"""
    return 0.02 * max(1 - t / 10, 0)

def shock_s4(t):
    """S4: Flattener - +200bps to 0 at 5Y, then -100bps at 10Y"""
    if t <= 5:
        return 0.02 * (1 - t / 5)
    else:
        return -0.01 * (t - 5) / 5


def apply_shock_to_curve(tenors_yrs, base_rates_arr, shock_func):
    """Apply a shock function to the base zero rate curve."""
    shocked = np.array([base_rates_arr[i] + shock_func(tenors_yrs[i])
                       for i in range(len(tenors_yrs))])
    return np.maximum(shocked, 0)  # Zero floor


shock_funcs = {
    'S1: +200bps Parallel': shock_s1,
    'S2: -200bps Parallel': shock_s2,
    'S3: Steepener': shock_s3,
    'S4: Flattener': shock_s4
}
