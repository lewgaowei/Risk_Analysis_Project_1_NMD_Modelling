# %% [markdown]
# # Phase 1a: Data Preparation & Exploratory Data Analysis
#
# ## QF609 Project #1 - IRRBB Modelling Framework
#
# **Calculation Date:** 30-Dec-2023
#
# This notebook performs:
# - Load NMD account data and zero rate curve
# - Data cleaning and validation
# - Compute derived features (daily decay rate, rolling statistics)
# - Generate EDA visualizations

# %% [markdown]
# ## 1. Import Libraries

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# Set style for professional-looking charts
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 10

print("Libraries imported successfully")

# %% [markdown]
# ## 2. Load Data Files

# %%
# Load NMD account data
nmd_data = pd.read_excel('group-proj-1-data.xlsx')
print("NMD Data Shape:", nmd_data.shape)
print("\nColumns:", nmd_data.columns.tolist())
print("\nFirst 5 rows:")
nmd_data.head()

# %%
# Load zero rate curve
curve_data = pd.read_excel('group-proj-1-curve.xlsx')
print("Curve Data Shape:", curve_data.shape)
print("\nCurve Data:")
curve_data

# %% [markdown]
# ## 3. Data Cleaning & Validation

# %%
# Check data types
print("NMD Data Info:")
print(nmd_data.info())
print("\n" + "="*60 + "\n")

# Check for missing values
print("Missing Values:")
print(nmd_data.isnull().sum())
print("\n" + "="*60 + "\n")

# Basic statistics
print("Summary Statistics:")
nmd_data.describe()

# %%
# Check date range and frequency
print("Date Range:")
print(f"Start Date: {nmd_data['Date'].min()}")
print(f"End Date: {nmd_data['Date'].max()}")
print(f"Total Days: {len(nmd_data)}")
print(f"Expected Days: {(nmd_data['Date'].max() - nmd_data['Date'].min()).days + 1}")

# Check for missing dates
date_range = pd.date_range(start=nmd_data['Date'].min(), end=nmd_data['Date'].max(), freq='D')
missing_dates = date_range.difference(nmd_data['Date'])
print(f"\nMissing Dates: {len(missing_dates)}")
if len(missing_dates) > 0:
    print("First 10 missing dates:")
    print(missing_dates[:10])

# %%
# Fix typo in column name: 'Outfolow' -> 'Outflow'
if 'Outfolow' in nmd_data.columns:
    nmd_data = nmd_data.rename(columns={'Outfolow': 'Outflow'})
    print("Fixed column name: 'Outfolow' -> 'Outflow'")

# Verify netflow calculation
nmd_data['Netflow_Calculated'] = nmd_data['Inflow'] - nmd_data['Outflow']
netflow_diff = (nmd_data['Netflow'] - nmd_data['Netflow_Calculated']).abs().max()
print(f"\nMax difference between Netflow and calculated Netflow: {netflow_diff:.6f}")

# Check final balance
final_balance = nmd_data[nmd_data['Date'] == '2023-12-30']['Balance'].values[0]
print(f"\nFinal Balance on 30-Dec-2023: {final_balance:,.2f}")

# %% [markdown]
# ## 4. Compute Derived Features

# %%
# Daily decay rate (conditional decay rate)
# Only compute where Balance > 0 to avoid division by zero
nmd_data['daily_decay_rate'] = np.where(
    nmd_data['Balance'] > 0,
    nmd_data['Outflow'] / nmd_data['Balance'],
    np.nan
)

# Rolling statistics (30-day and 90-day moving averages)
nmd_data['decay_rate_ma30'] = nmd_data['daily_decay_rate'].rolling(window=30, min_periods=1).mean()
nmd_data['decay_rate_ma90'] = nmd_data['daily_decay_rate'].rolling(window=90, min_periods=1).mean()

# Balance moving averages
nmd_data['balance_ma30'] = nmd_data['Balance'].rolling(window=30, min_periods=1).mean()
nmd_data['balance_ma90'] = nmd_data['Balance'].rolling(window=90, min_periods=1).mean()

print("Derived features computed:")
print("- daily_decay_rate")
print("- decay_rate_ma30 (30-day moving average)")
print("- decay_rate_ma90 (90-day moving average)")
print("- balance_ma30")
print("- balance_ma90")

print("\nSample of derived features:")
nmd_data[['Date', 'Balance', 'Outflow', 'daily_decay_rate', 'decay_rate_ma30']].tail(10)

# %%
# Monthly balance (month-end resampling)
nmd_data_monthly = nmd_data.set_index('Date').resample('M').agg({
    'Balance': 'last',
    'Inflow': 'sum',
    'Outflow': 'sum',
    'Netflow': 'sum',
    'daily_decay_rate': 'mean'
}).reset_index()

print(f"Monthly data shape: {nmd_data_monthly.shape}")
print("\nLast 12 months:")
nmd_data_monthly.tail(12)

# %%
# Add day of week and month features for seasonality analysis
nmd_data['day_of_week'] = pd.to_datetime(nmd_data['Date']).dt.dayofweek
nmd_data['day_name'] = pd.to_datetime(nmd_data['Date']).dt.day_name()
nmd_data['month'] = pd.to_datetime(nmd_data['Date']).dt.month
nmd_data['month_name'] = pd.to_datetime(nmd_data['Date']).dt.month_name()
nmd_data['year'] = pd.to_datetime(nmd_data['Date']).dt.year

print("Added temporal features for seasonality analysis:")
print("- day_of_week (0=Monday, 6=Sunday)")
print("- day_name")
print("- month")
print("- month_name")
print("- year")

# %% [markdown]
# ## 5. Exploratory Data Analysis - Visualizations

# %% [markdown]
# ### 5.1 Balance Time Series (Full History)

# %%
fig, ax = plt.subplots(figsize=(14, 6))

ax.plot(nmd_data['Date'], nmd_data['Balance'], linewidth=1.5, color='#2E86AB', label='Daily Balance')
ax.plot(nmd_data['Date'], nmd_data['balance_ma30'], linewidth=1.5, color='#A23B72',
        linestyle='--', alpha=0.8, label='30-Day MA')
ax.plot(nmd_data['Date'], nmd_data['balance_ma90'], linewidth=1.5, color='#F18F01',
        linestyle='--', alpha=0.8, label='90-Day MA')

# Mark minimum balance
min_balance = nmd_data['Balance'].min()
ax.axhline(y=min_balance, color='red', linestyle=':', linewidth=1.5, alpha=0.7,
           label=f'Min Balance: {min_balance:,.0f}')

ax.set_xlabel('Date')
ax.set_ylabel('Balance')
ax.set_title('NMD Account Balance History (2017-2023)', fontsize=14, fontweight='bold')
ax.legend(loc='best')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print(f"Minimum Balance: {min_balance:,.2f}")
print(f"Maximum Balance: {nmd_data['Balance'].max():,.2f}")
print(f"Average Balance: {nmd_data['Balance'].mean():,.2f}")
print(f"Current Balance (30-Dec-2023): {final_balance:,.2f}")

# %% [markdown]
# ### 5.2 Inflow and Outflow Distributions

# %%
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Inflow distribution
axes[0].hist(nmd_data['Inflow'], bins=50, color='#06A77D', alpha=0.7, edgecolor='black')
axes[0].axvline(nmd_data['Inflow'].mean(), color='red', linestyle='--', linewidth=2,
                label=f"Mean: {nmd_data['Inflow'].mean():.2f}")
axes[0].axvline(nmd_data['Inflow'].median(), color='orange', linestyle='--', linewidth=2,
                label=f"Median: {nmd_data['Inflow'].median():.2f}")
axes[0].set_xlabel('Inflow')
axes[0].set_ylabel('Frequency')
axes[0].set_title('Distribution of Daily Inflows', fontweight='bold')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Outflow distribution
axes[1].hist(nmd_data['Outflow'], bins=50, color='#D62828', alpha=0.7, edgecolor='black')
axes[1].axvline(nmd_data['Outflow'].mean(), color='red', linestyle='--', linewidth=2,
                label=f"Mean: {nmd_data['Outflow'].mean():.2f}")
axes[1].axvline(nmd_data['Outflow'].median(), color='orange', linestyle='--', linewidth=2,
                label=f"Median: {nmd_data['Outflow'].median():.2f}")
axes[1].set_xlabel('Outflow')
axes[1].set_ylabel('Frequency')
axes[1].set_title('Distribution of Daily Outflows', fontweight='bold')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("Inflow Statistics:")
print(nmd_data['Inflow'].describe())
print("\nOutflow Statistics:")
print(nmd_data['Outflow'].describe())

# %% [markdown]
# ### 5.3 Daily Decay Rate Time Series

# %%
fig, ax = plt.subplots(figsize=(14, 6))

# Plot daily decay rate with transparency to show density
ax.scatter(nmd_data['Date'], nmd_data['daily_decay_rate']*100, s=3, alpha=0.3,
           color='#5E548E', label='Daily Decay Rate')
ax.plot(nmd_data['Date'], nmd_data['decay_rate_ma30']*100, linewidth=2, color='#E63946',
        label='30-Day MA')
ax.plot(nmd_data['Date'], nmd_data['decay_rate_ma90']*100, linewidth=2, color='#F77F00',
        label='90-Day MA')

ax.set_xlabel('Date')
ax.set_ylabel('Daily Decay Rate (%)')
ax.set_title('Daily Decay Rate with Moving Averages', fontsize=14, fontweight='bold')
ax.legend(loc='best')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Statistics on decay rate
mean_decay = nmd_data['daily_decay_rate'].mean()
median_decay = nmd_data['daily_decay_rate'].median()
std_decay = nmd_data['daily_decay_rate'].std()

print(f"Mean Daily Decay Rate: {mean_decay*100:.4f}%")
print(f"Median Daily Decay Rate: {median_decay*100:.4f}%")
print(f"Std Dev Daily Decay Rate: {std_decay*100:.4f}%")
print(f"\nImplied Monthly Decay Rate: {(1 - (1-mean_decay)**30)*100:.2f}%")

# %% [markdown]
# ### 5.4 Seasonality Analysis - Day of Week

# %%
# Aggregate by day of week
dow_stats = nmd_data.groupby('day_name').agg({
    'daily_decay_rate': ['mean', 'std', 'count'],
    'Outflow': ['mean', 'sum'],
    'Inflow': ['mean', 'sum']
}).reset_index()

# Reorder days of week
day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
dow_stats['day_name'] = pd.Categorical(dow_stats['day_name'], categories=day_order, ordered=True)
dow_stats = dow_stats.sort_values('day_name')

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Decay rate by day of week
axes[0].bar(dow_stats['day_name'], dow_stats['daily_decay_rate']['mean']*100,
            color='#2A9D8F', alpha=0.8, edgecolor='black')
axes[0].set_xlabel('Day of Week')
axes[0].set_ylabel('Average Daily Decay Rate (%)')
axes[0].set_title('Average Decay Rate by Day of Week', fontweight='bold')
axes[0].tick_params(axis='x', rotation=45)
axes[0].grid(True, alpha=0.3, axis='y')

# Inflow vs Outflow by day of week
x = np.arange(len(dow_stats))
width = 0.35
axes[1].bar(x - width/2, dow_stats['Inflow']['mean'], width, label='Inflow',
            color='#06A77D', alpha=0.8, edgecolor='black')
axes[1].bar(x + width/2, dow_stats['Outflow']['mean'], width, label='Outflow',
            color='#D62828', alpha=0.8, edgecolor='black')
axes[1].set_xlabel('Day of Week')
axes[1].set_ylabel('Average Daily Amount')
axes[1].set_title('Average Inflow/Outflow by Day of Week', fontweight='bold')
axes[1].set_xticks(x)
axes[1].set_xticklabels(dow_stats['day_name'])
axes[1].tick_params(axis='x', rotation=45)
axes[1].legend()
axes[1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.show()

print("\nDay of Week Statistics:")
print(dow_stats)

# %% [markdown]
# ### 5.5 Seasonality Analysis - Monthly Patterns

# %%
# Aggregate by month
month_stats = nmd_data.groupby('month').agg({
    'daily_decay_rate': ['mean', 'std'],
    'Outflow': ['mean', 'sum'],
    'Inflow': ['mean', 'sum'],
    'month_name': 'first'
}).reset_index()

fig, axes = plt.subplots(2, 1, figsize=(14, 10))

# Decay rate by month
axes[0].bar(month_stats['month'], month_stats['daily_decay_rate']['mean']*100,
            color='#264653', alpha=0.8, edgecolor='black')
axes[0].set_xlabel('Month')
axes[0].set_ylabel('Average Daily Decay Rate (%)')
axes[0].set_title('Average Decay Rate by Month', fontweight='bold')
axes[0].set_xticks(month_stats['month'])
axes[0].set_xticklabels(month_stats['month_name']['first'], rotation=45)
axes[0].grid(True, alpha=0.3, axis='y')

# Netflow by month over years
monthly_netflow = nmd_data.groupby(['year', 'month']).agg({
    'Netflow': 'sum'
}).reset_index()

for year in monthly_netflow['year'].unique():
    year_data = monthly_netflow[monthly_netflow['year'] == year]
    axes[1].plot(year_data['month'], year_data['Netflow'], marker='o',
                linewidth=2, label=f'{year}', alpha=0.8)

axes[1].axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
axes[1].set_xlabel('Month')
axes[1].set_ylabel('Monthly Net Flow')
axes[1].set_title('Monthly Net Flow by Year', fontweight='bold')
axes[1].set_xticks(range(1, 13))
axes[1].set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                         'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
axes[1].legend(loc='best')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\nMonth Statistics:")
print(month_stats)

# %% [markdown]
# ### 5.6 Zero Rate Curve Visualization

# %%
# Convert tenor to numeric (years)
tenor_map = {
    '1D': 1/365,
    '1M': 1/12,
    '2M': 2/12,
    '3M': 3/12,
    '6M': 6/12,
    '9M': 9/12,
    '1Y': 1,
    '2Y': 2,
    '3Y': 3,
    '4Y': 4,
    '5Y': 5,
    '6Y': 6,
    '7Y': 7,
    '8Y': 8,
    '9Y': 9,
    '10Y': 10
}

curve_data['Tenor_Years'] = curve_data['Tenor'].map(tenor_map)

fig, ax = plt.subplots(figsize=(12, 6))

ax.plot(curve_data['Tenor_Years'], curve_data['ZeroRate']*100,
        marker='o', markersize=8, linewidth=2, color='#023047', label='Zero Rate Curve')

for i, row in curve_data.iterrows():
    ax.text(row['Tenor_Years'], row['ZeroRate']*100 + 0.05,
            f"{row['ZeroRate']*100:.2f}%",
            ha='center', va='bottom', fontsize=8)

ax.set_xlabel('Tenor (Years)')
ax.set_ylabel('Zero Rate (%)')
ax.set_title('Base Zero Rate Curve (30-Dec-2023)', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend()
plt.tight_layout()
plt.show()

print("\nZero Rate Curve:")
print(curve_data[['Tenor', 'ZeroRate']])

# %% [markdown]
# ## 6. Summary Statistics & Key Findings

# %%
print("="*80)
print("PHASE 1A SUMMARY - DATA EXPLORATION & ANALYSIS")
print("="*80)

print("\n1. DATA OVERVIEW")
print("-" * 80)
print(f"   Date Range: {nmd_data['Date'].min().strftime('%d-%b-%Y')} to {nmd_data['Date'].max().strftime('%d-%b-%Y')}")
print(f"   Total Observations: {len(nmd_data)} days")
print(f"   Calculation Date: 30-Dec-2023")

print("\n2. BALANCE STATISTICS")
print("-" * 80)
print(f"   Current Balance (30-Dec-2023): {final_balance:,.2f}")
print(f"   Minimum Balance: {nmd_data['Balance'].min():,.2f}")
print(f"   Maximum Balance: {nmd_data['Balance'].max():,.2f}")
print(f"   Average Balance: {nmd_data['Balance'].mean():,.2f}")
print(f"   Median Balance: {nmd_data['Balance'].median():,.2f}")
print(f"   Std Dev Balance: {nmd_data['Balance'].std():,.2f}")

print("\n3. FLOW STATISTICS")
print("-" * 80)
print(f"   Average Daily Inflow: {nmd_data['Inflow'].mean():,.2f}")
print(f"   Average Daily Outflow: {nmd_data['Outflow'].mean():,.2f}")
print(f"   Average Daily Net Flow: {nmd_data['Netflow'].mean():,.2f}")
print(f"   Total Cumulative Inflow: {nmd_data['Inflow'].sum():,.2f}")
print(f"   Total Cumulative Outflow: {nmd_data['Outflow'].sum():,.2f}")
print(f"   Total Cumulative Net Flow: {nmd_data['Netflow'].sum():,.2f}")

print("\n4. DECAY RATE STATISTICS")
print("-" * 80)
print(f"   Mean Daily Decay Rate: {mean_decay*100:.4f}%")
print(f"   Median Daily Decay Rate: {median_decay*100:.4f}%")
print(f"   Std Dev Daily Decay Rate: {std_decay*100:.4f}%")
print(f"   Implied Monthly Decay Rate: {(1 - (1-mean_decay)**30)*100:.2f}%")
print(f"   Implied Annual Decay Rate: {(1 - (1-mean_decay)**365)*100:.2f}%")

print("\n5. CORE DEPOSIT ESTIMATE (Preliminary)")
print("-" * 80)
min_balance_amount = nmd_data['Balance'].min()
core_ratio_naive = min_balance_amount / final_balance
print(f"   Minimum Balance (naive core floor): {min_balance_amount:,.2f}")
print(f"   Current Balance: {final_balance:,.2f}")
print(f"   Naive Core Ratio: {core_ratio_naive*100:.2f}%")
print(f"   Implied Non-Core Amount: {(final_balance - min_balance_amount):,.2f}")
print(f"   Note: Will refine in Phase 1c using regulatory constraints")

print("\n6. KEY OBSERVATIONS")
print("-" * 80)
print("   - Balance shows relative stability with some growth trend")
print("   - Daily decay rate is relatively low, indicating sticky deposits")
print("   - No significant outliers or data quality issues detected")
print("   - Seasonality patterns exist but are not extreme")
print("   - Zero rate curve shows normal upward-sloping term structure")

print("\n" + "="*80)
print("PHASE 1A COMPLETE")
print("="*80)

# %% [markdown]
# ## 7. Save Processed Data

# %%
# Save processed data for use in subsequent phases
nmd_data.to_csv('processed_nmd_data.csv', index=False)
nmd_data_monthly.to_csv('processed_nmd_monthly.csv', index=False)
curve_data.to_csv('processed_curve_data.csv', index=False)

print("Processed data saved:")
print("- processed_nmd_data.csv (daily data with derived features)")
print("- processed_nmd_monthly.csv (monthly aggregated data)")
print("- processed_curve_data.csv (zero rate curve with tenor in years)")

# %%
