# QF609 Group Project #1 — NMD IRRBB Modelling Framework

## Project Overview

Build a modelling framework implementing the Basel approach for measuring Interest Rate Risk in the Banking Book (IRRBB) for a Non-Maturity Deposit (NMD) account cohort. Calculation date: **30-Dec-2023**.

## Data Files

- `group-proj-1-data.xlsx` — Daily NMD account data (2,556 rows, 31-Dec-2016 to 30-Dec-2023)
  - Columns: `Date`, `Balance`, `Inflow`, `Outfolow` (note: typo in source), `Netflow`
  - Final balance on 30-Dec-2023: ~18,651.70
- `group-proj-1-curve.xlsx` — Base zero rate curve (16 tenors from 1D to 10Y)
  - Columns: `Tenor`, `ZeroRate`
  - Tenors: 1D, 1M, 2M, 3M, 6M, 9M, 1Y, 2Y, 3Y, 4Y, 5Y, 6Y, 7Y, 8Y, 9Y, 10Y

## Implementation Requirements

Use **Python** with pandas, numpy, scipy, matplotlib/seaborn. Produce all outputs as clean charts and summary tables. The final deliverable is a set of results + visualizations suitable for a 10-15 minute class presentation.

---

## PHASE 1: Data Exploration & Deposit Decay Estimation

### 1a. Data Preparation & EDA

- Load both Excel files
- Check for missing dates, NaN values, outliers in flows
- Compute derived features:
  - `daily_decay_rate` = Outflow / Balance (conditional decay rate)
  - `monthly_balance` = resample to month-end balances
  - Rolling statistics (30-day, 90-day moving averages of decay rate)
- Generate EDA charts:
  - Balance time series plot (full history)
  - Inflow/outflow distributions (histograms)
  - Daily decay rate time series with moving average overlay
  - Seasonality analysis (decay rate by day-of-week, month)

### 1b. Decay Rate Modelling

Estimate the **survival function** S(t) = probability a dollar deposited at time 0 remains at time t.

**Primary approach — Average CDR (Conditional Decay Rate):**
- Compute mean daily decay rate: `λ_daily = mean(Outflow / Balance)`
- Convert to monthly: `λ_monthly = 1 - (1 - λ_daily)^30`
- Build exponential survival curve: `S(t) = (1 - λ_daily)^t` or equivalently `S(t) = exp(-λ*t)`
- Cap the curve at 5 years (1,826 days) per regulatory constraint

**Enhanced approach (bonus) — Regression-based CDR:**
- Model daily decay rate as function of: balance level, day-of-week dummies, month dummies, trend
- Use OLS or GLM (Beta regression for bounded rates)
- Use fitted model to project forward decay curve

**Output:**
- Survival curve S(t) plotted from t=0 to t=5Y
- Table of S(t) at key tenors: 1M, 2M, 3M, 6M, 9M, 1Y, 2Y, 3Y, 4Y, 5Y

### 1c. Core vs Non-Core Deposit Separation (Basel Framework)

Basel requires splitting NMD balance into:
- **Core (stable) deposits**: unlikely to reprice/leave even under stress
- **Non-core (volatile) deposits**: can leave quickly, reprice at O/N or 1M

**Method to estimate core portion:**
- Compute the historical minimum balance over the full sample: `B_min = min(Balance)`
- Alternative: use a percentile floor (e.g., 5th percentile of monthly average balances)
- Core ratio = B_min / B_current (or percentile floor / B_current)
- Non-core = B_current - Core

From the data: minimum balance is ~9,511 out of current ~18,652, giving a naive core ratio of ~51%. Consider also rolling-window minimum approaches.

**Apply regulatory constraints:**
- Core portion maximum behavioral maturity: **5 years** (given in problem)
- Basel also caps core deposits at a percentage of total NMD (typically up to 90% for retail) — apply if covered in class

**Output:**
- Core vs non-core split (dollar amounts and percentages)
- Visualization showing balance history with core floor marked
- Sensitivity table showing how different core assumptions (e.g., 50%, 60%, 70%, 80%) affect downstream results

---

## PHASE 2: Cash Flow Slotting (Basel Repricing Profile)

### 2a. Define Time Buckets

Use these standard IRRBB time buckets (aligned with the curve tenors):

| Bucket | Start | End | Midpoint (years) |
|--------|-------|-----|-------------------|
| O/N    | 0     | 1D  | 0.00137           |
| 1M     | 1D    | 1M  | 0.0417            |
| 2M     | 1M    | 2M  | 0.125             |
| 3M     | 2M    | 3M  | 0.2083            |
| 6M     | 3M    | 6M  | 0.375             |
| 9M     | 6M    | 9M  | 0.625             |
| 1Y     | 9M    | 1Y  | 0.875             |
| 2Y     | 1Y    | 2Y  | 1.5               |
| 3Y     | 2Y    | 3Y  | 2.5               |
| 4Y     | 3Y    | 4Y  | 3.5               |
| 5Y     | 4Y    | 5Y  | 4.5               |

### 2b. Slot Cash Flows

**Non-core portion:**
- Entire non-core amount goes into the **O/N or 1M bucket** (reprices immediately)

**Core portion:**
- Distribute across buckets using the decay survival function
- Cash flow in bucket i = Core × [S(t_{i-1}) - S(t_i)]
- Where t_i is the end-time of bucket i in days
- Any remaining core balance at 5Y is placed in the 5Y bucket (since 5Y cap)

**Output:**
- Repricing profile table: each bucket with its notional cash flow amount
- Bar chart of cash flow distribution across buckets
- Waterfall chart showing cumulative runoff

---

## PHASE 3: EVE (Economic Value of Equity) Sensitivity

### 3a. Discount Factor Construction

- Convert zero rates to discount factors: `DF(t) = 1 / (1 + r(t))^t`
- Where t is in years and r(t) is the continuously compounded or annually compounded zero rate (check convention — if rates are annual: `DF(t) = 1/(1+r)^t`; if continuous: `DF(t) = exp(-r*t)`)
- Interpolate the curve for any intermediate tenors needed (log-linear interpolation on discount factors)

### 3b. Base Case EVE

- `EVE_base = Σ CF(i) × DF(i)` across all time buckets
- Where CF(i) is the notional in bucket i, DF(i) is the discount factor at bucket midpoint

### 3c. Shocked EVE — 4 Scenarios

**Scenario 1: +200bps Parallel Shift**
- `r_shocked(t) = r(t) + 0.02` for all tenors
- Recompute DF, recompute EVE

**Scenario 2: -200bps Parallel Shift**
- `r_shocked(t) = max(r(t) - 0.02, 0)` for all tenors (apply zero floor)
- Recompute DF, recompute EVE

**Scenario 3: Short Rate Up Shock (Steepener)**
- +200bps at shortest tenor, tapering linearly to 0bps at 10Y
- Suggested formula: `shock(t) = 0.02 × max(1 - t/10, 0)` where t is tenor in years
- `r_shocked(t) = r(t) + shock(t)`

**Scenario 4: Flattener**
- Short end: +200bps at shortest tenor, tapering to 0bps at ~5Y midpoint
- Long end: -100bps at 10Y, tapering from 0bps at ~5Y midpoint
- Suggested formula: `shock(t) = 0.02 × max(1 - t/5, 0) - 0.01 × max((t-5)/5, 0)` for t in years
- Apply zero floor on resulting rates

### 3d. EVE Sensitivity

- `ΔEVE(scenario) = EVE_shocked - EVE_base`
- Report all 4 scenarios
- **Worst case ΔEVE** = the scenario with the largest negative ΔEVE (largest loss)

**Output:**
- Table: Base EVE, Shocked EVE for each scenario, ΔEVE, ΔEVE as % of base
- Bar chart comparing ΔEVE across scenarios
- Yield curve plot showing base vs shocked curves for all 4 scenarios

---

## PHASE 4: NII (Net Interest Income) Sensitivity

### 4a. NII Framework

NII sensitivity measures the impact on interest earnings over the **next 12 months**.

- Only cash flows that **reprice within 1 year** are affected by rate shocks
- NII impact = Σ (notional in bucket i) × Δr(i) × (remaining_time_in_1Y_horizon / 1Y)
- For bucket i with midpoint t_i (in years), if t_i ≤ 1:
  - `ΔNII(i) = CF(i) × shock(t_i) × (1 - t_i)` — the (1-t_i) factor accounts for the portion of the year remaining after repricing

### 4b. Compute NII Under 4 Scenarios

- Apply same 4 rate shock specifications as EVE
- Sum up ΔNII across all buckets within the 1Y horizon
- Worst case ΔNII = scenario with largest negative impact

**Output:**
- Table: ΔNII for each scenario
- Combined summary table with both EVE and NII worst cases
- Identify which scenario(s) are the binding IRRBB measures

---

## PHASE 5: A+ Bonus Components (Implement at least 2-3 of these)

### 5a. Sensitivity Analysis on Core Deposit Assumption ⭐ (HIGH PRIORITY)
- Run the entire EVE/NII calculation for core ratios: 40%, 50%, 60%, 70%, 80%
- Produce a sensitivity table and line chart showing how ΔEVE and ΔNII change
- This demonstrates model risk awareness — examiners and professors love this

### 5b. Pass-Through Rate Analysis ⭐ (HIGH PRIORITY)
- Model the NMD deposit rate as: `r_deposit = α + β × r_market`
- Where β is the pass-through coefficient (how much of market rate changes get passed to depositors)
- Estimate β from historical data if rate data is available, or assume typical values (β ≈ 0.3–0.7 for retail NMDs)
- Show how pass-through affects NII sensitivity (higher β = less NII sensitivity)

### 5c. Backtesting the Decay Model ⭐ (HIGH PRIORITY)
- Train decay model on 2017-2022 data
- Predict 2023 balance path using the trained decay parameters
- Compare predicted vs actual 2023 balances
- Report prediction error metrics (RMSE, MAPE)
- Plot actual vs predicted balance paths

### 5d. Monte Carlo Simulation of Deposit Paths
- Model daily outflow as: `Outflow(t) = Balance(t) × (λ + ε(t))` where ε ~ N(0, σ²)
- Simulate 1,000 balance paths forward from calculation date
- Compute EVE/NII for each path → distribution of outcomes
- Report VaR and expected shortfall of EVE/NII sensitivities
- Plot fan chart of simulated balance paths

### 5e. Stress-Adjusted Decay
- Rerun analysis with stressed decay rate (e.g., 1.5× base decay)
- Compare stressed vs base EVE/NII outcomes
- Justification: Basel requires banks to consider stress scenarios for NMD behavior

### 5f. Convexity-Adjusted EVE
- For each scenario, compute both duration-based (first-order) and convexity-adjusted (second-order) EVE changes
- Show that for ±200bps shocks, convexity matters
- Formula: `ΔP/P ≈ -D×Δy + 0.5×C×(Δy)²`

---

## PHASE 6: Presentation Materials

### Required Outputs for Slides

1. **Executive Summary slide**: Key findings — worst case EVE and NII, which scenario drives them
2. **Data Overview slide**: Balance history chart, summary statistics
3. **Decay Model slide**: Survival curve, methodology explanation
4. **Core/Non-Core Split slide**: Methodology, resulting split, balance floor visualization
5. **Cash Flow Slotting slide**: Repricing profile bar chart, table
6. **Rate Scenarios slide**: Yield curve plots showing base vs 4 shocked curves
7. **EVE Results slide**: ΔEVE bar chart, summary table
8. **NII Results slide**: ΔNII bar chart, summary table
9. **IRRBB Summary slide**: Combined worst-case reporting table
10. **Bonus Analysis slides**: Sensitivity analysis, backtesting, etc.
11. **Methodology & Assumptions slide**: Document all key assumptions

### Chart Style Guidelines
- Use consistent color scheme (e.g., blue for base, red for stressed/shocked)
- All charts must have clear titles, axis labels, legends
- Use professional formatting (no default matplotlib style — use seaborn or a clean theme)
- Export all charts as high-resolution PNGs (300 DPI)
- Save all output tables as formatted DataFrames exportable to Excel

---

## File Structure

```
project/
├── data/
│   ├── group-proj-1-data.xlsx
│   └── group-proj-1-curve.xlsx
├── src/
│   ├── 01_eda.py                 # Data loading, cleaning, EDA charts
│   ├── 02_decay_model.py         # Decay rate estimation, survival function
│   ├── 03_core_noncore.py        # Core/non-core separation
│   ├── 04_cashflow_slotting.py   # Basel cash flow slotting
│   ├── 05_eve_sensitivity.py     # EVE calculation and 4 scenarios
│   ├── 06_nii_sensitivity.py     # NII calculation and 4 scenarios
│   ├── 07_bonus_analysis.py      # All bonus components
│   └── utils.py                  # Shared helpers (interpolation, DF calc, etc.)
├── outputs/
│   ├── charts/                   # All PNG charts
│   ├── tables/                   # CSV/Excel summary tables
│   └── results_summary.xlsx      # Master results workbook
├── main.py                       # Orchestrator — runs everything end to end
└── README.md
```

---

## Key Assumptions to Document

1. Decay rate model: exponential decay using average historical CDR
2. Core deposit floor: based on historical minimum balance (or specified percentile)
3. 5-year regulatory cap on core deposit behavioral maturity
4. Zero rate curve assumed to be annually compounded (clarify if continuous)
5. Interpolation method for intermediate tenors: log-linear on discount factors
6. Zero floor applied to negative shocked rates
7. Short rate up shock: linear taper from +200bps at O/N to 0bps at 10Y
8. Flattener shock: +200bps short end tapering to 0 at 5Y, then -100bps at 10Y
9. NII horizon: 12 months
10. Non-core deposits reprice at O/N

---

## Priority Order for Implementation

1. **MUST HAVE**: Phases 1-4 (decay model, slotting, EVE, NII) — this is the core requirement
2. **SHOULD HAVE**: Phase 5a (sensitivity to core assumption) + Phase 5c (backtesting) — easiest high-impact bonuses
3. **NICE TO HAVE**: Phase 5b (pass-through), Phase 5d (Monte Carlo), Phase 5f (convexity)
4. **LAST**: Phase 6 presentation polish
