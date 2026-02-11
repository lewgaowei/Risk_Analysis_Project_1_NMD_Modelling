# IRRBB Quick Reference Guide ⚡

## 30-Second Summary

**Your Goal:** Measure how interest rate changes affect a bank's $18,652 deposit portfolio

**Main Metrics:**
- **EVE** = Long-term economic value (Will bank survive?)
- **NII** = 12-month earnings (Will bank be profitable?)

**Key Finding:** +200bps rate increase → EVE loses $200 (-1.08%)

---

## Visual Concept Map

```
                    IRRBB PROJECT
                         |
        ┌────────────────┼────────────────┐
        ▼                ▼                ▼
    DEPOSITS         INTEREST         BASEL
    $18,652          RATES            RULES
        |                |                |
        ▼                ▼                ▼
   Decay Model      Rate Shocks     EVE & NII
   λ = 0.15%        ±200bps         Max 5Y
        |                |                |
        └────────┬───────┴────────────────┘
                 ▼
            CASH FLOW SLOTTING
            (11 time buckets)
                 |
        ┌────────┴────────┐
        ▼                 ▼
    EVE = $18,500    NII Impact
    (Long-term)      (12 months)
        |                 |
        ▼                 ▼
    ΔEVE = -$200     ΔNII = -$180
    (Worst case)     (Worst case)
```

---

## Key Formulas Cheat Sheet

### 1. Decay & Survival
```python
λ_daily = Outflow / Balance          # Example: 0.0015 (0.15%)
S(t) = (1 - λ)^t                     # Example: S(365) = 0.85
λ_monthly = 1 - (1 - λ_daily)^30     # Example: 4.4%
```

### 2. Core/Non-Core Split
```python
Core = Historical Minimum Balance     # Example: $9,511
Non-Core = Current - Core             # Example: $9,141
Core Ratio = Core / Current           # Example: 51%
```

### 3. Cash Flow Slotting
```python
# Non-core → ALL to O/N bucket
# Core → Distributed across buckets

CF(bucket_i) = Core × [S(t_start) - S(t_end)]

# Example for 1Y bucket (270-365 days):
# CF = 9,511 × [S(270) - S(365)]
#    = 9,511 × [0.65 - 0.60]
#    = 475.55
```

### 4. EVE (Economic Value)
```python
# Step 1: Calculate discount factors
DF(t) = 1 / (1 + r)^t

# Step 2: Calculate present value
EVE = Σ [CF(i) × DF(i)]

# Step 3: Shock and recalculate
ΔEVE = EVE_shocked - EVE_base

# Example:
# Base EVE = $18,500
# S1 (+200bps) EVE = $18,300
# ΔEVE = -$200 (-1.08%)
```

### 5. NII (Net Interest Income)
```python
# Only buckets with t ≤ 1 year!
ΔNII = Σ [CF(i) × shock(i) × (1 - t_i)]

# Example for 3M bucket:
# CF = $5,000
# shock = +0.02 (200bps)
# t = 0.25 years
# ΔNII = 5,000 × 0.02 × (1 - 0.25)
#      = 5,000 × 0.02 × 0.75
#      = $75
```

---

## The 4 Rate Shock Scenarios

| Scenario | Short Rate | Long Rate | Impact |
|----------|-----------|-----------|--------|
| **S1: +200bps Parallel** | +200bps | +200bps | All rates ↑ equally |
| **S2: -200bps Parallel** | -200bps | -200bps | All rates ↓ equally (floor at 0%) |
| **S3: Steepener** | +200bps | 0bps | Curve gets steeper |
| **S4: Flattener** | +200bps | -100bps | Curve gets flatter |

**Your worst case:** S1 (+200bps) hurts EVE most → You're asset-sensitive!

---

## EVE vs NII Comparison Table

| Aspect | EVE | NII |
|--------|-----|-----|
| **Full Name** | Economic Value of Equity | Net Interest Income |
| **Time Horizon** | Long-term (ALL cash flows) | Short-term (12 months) |
| **Measures** | Economic value change | Earnings change |
| **Question Answered** | Will bank survive? | Will bank be profitable? |
| **Buckets Included** | O/N, 1M, 2M, ... 5Y | O/N, 1M, 2M, ... 1Y only |
| **Basel Threshold** | 15% of Tier 1 Capital | No specific threshold |
| **Formula** | Σ [CF × DF] | Σ [CF × shock × (1-t)] |
| **Example Result** | ΔEVE = -$200 | ΔNII = +$75 |
| **Analogy** | Your retirement account | Your annual salary |

**Key Insight:** EVE and NII can have DIFFERENT worst cases!
- Your project: S1 worst for EVE, S2 worst for NII

---

## Basel Regulatory Framework

### Core Requirements
1. ✅ Calculate EVE under 6 scenarios (your project uses 4)
2. ✅ Calculate NII under same scenarios
3. ✅ Report if |ΔEVE| > 15% of Tier 1 Capital (outlier)
4. ✅ Hold capital for interest rate risk

### Core Deposit Rules
- Max **90%** of NMDs can be core (retail)
- Max **5-year** behavioral maturity
- Must use **survival function** (no ad-hoc assumptions)
- **Stress testing** required

### Your Compliance Status
```
|ΔEVE| = $200 / $18,500 = 1.08%
If Tier 1 Capital > $1,333, then 1.08% < 15% ✓ PASS
```

---

## Data Flow Diagram

```
INPUT FILES
├── group-proj-1-data.xlsx       (NMD balances 2017-2023)
└── group-proj-1-curve.xlsx      (Zero rates: 1D to 10Y)
         |
         ▼
    PHASE 1: DECAY MODEL
    ├── Estimate λ_daily = 0.15%
    ├── Build S(t) = (1-λ)^t
    └── Split: Core 51% | Non-Core 49%
         |
         ▼
    PHASE 2: CASH FLOW SLOTTING
    ├── Non-core → O/N bucket ($9,141)
    └── Core → Distributed 1M-5Y ($9,511)
         |
         ▼
    PHASE 3: EVE CALCULATION
    ├── Build discount factors DF(t)
    ├── Calculate EVE_base = $18,500
    ├── Test 4 rate shocks
    └── Worst: S1 (+200bps) = -$200
         |
         ▼
    PHASE 4: NII CALCULATION
    ├── Filter buckets t ≤ 1Y
    ├── Calculate ΔNII for each shock
    └── Worst: S2 (-200bps) = -$180
         |
         ▼
    PHASE 5: SENSITIVITY ANALYSIS
    ├── Core ratio sensitivity (40%-80%)
    ├── Pass-through beta (0%-100%)
    ├── Backtesting (2023 predictions)
    ├── Monte Carlo (1,000 paths)
    └── Stress decay (1.5x, 2.0x)
         |
         ▼
    OUTPUT: BASEL IRRBB REPORT
    ├── Worst ΔEVE: -$200 (-1.08%)
    ├── Worst ΔNII: -$180
    ├── Risk Profile: Asset-Sensitive
    └── Regulatory Status: PASS ✓
```

---

## Time Bucket Definitions

| Bucket | Days | Years | Midpoint (Y) | What Goes Here |
|--------|------|-------|--------------|----------------|
| **O/N** | 0-1 | 0-0.003 | 0.0014 | ALL non-core + day 0-1 core |
| **1M** | 1-30 | 0.003-0.08 | 0.042 | Core: day 1-30 decay |
| **2M** | 30-60 | 0.08-0.16 | 0.125 | Core: day 30-60 decay |
| **3M** | 60-90 | 0.16-0.25 | 0.208 | Core: day 60-90 decay |
| **6M** | 90-180 | 0.25-0.49 | 0.375 | Core: day 90-180 decay |
| **9M** | 180-270 | 0.49-0.74 | 0.625 | Core: day 180-270 decay |
| **1Y** | 270-365 | 0.74-1.0 | 0.875 | Core: day 270-365 decay |
| **2Y** | 365-730 | 1.0-2.0 | 1.5 | Core: year 1-2 decay |
| **3Y** | 730-1095 | 2.0-3.0 | 2.5 | Core: year 2-3 decay |
| **4Y** | 1095-1460 | 3.0-4.0 | 3.5 | Core: year 3-4 decay |
| **5Y** | 1460-1825 | 4.0-5.0 | 4.5 | Core: year 4-5 decay + remainder |

**Why these buckets?** Basel standard + matches zero curve tenors!

---

## Common Terminology

### Banking Terms
- **NMD** = Non-Maturity Deposit (savings, checking accounts)
- **IRRBB** = Interest Rate Risk in the Banking Book
- **ALM** = Asset-Liability Management
- **Tier 1 Capital** = Bank's core equity capital
- **bp/bps** = Basis point = 0.01% (200bps = 2%)

### Risk Terms
- **λ (lambda)** = Decay rate (probability of withdrawal)
- **S(t)** = Survival function (probability deposit remains)
- **DF(t)** = Discount factor (present value multiplier)
- **PV** = Present Value
- **NPV** = Net Present Value

### Model Terms
- **Core deposits** = Stable, sticky portion
- **Non-core deposits** = Volatile, rate-sensitive portion
- **Repricing** = When deposit rate adjusts to market
- **Behavioral maturity** = Expected time until repricing
- **Slotting** = Distributing cash flows into time buckets

---

## Typical Values Reference

### Decay Rates (Industry Benchmarks)
- **Retail savings:** λ_monthly = 3-8%
- **Retail checking:** λ_monthly = 5-12%
- **Wholesale:** λ_monthly = 15-40%
- **Your project:** λ_monthly = 4.4% (retail-like)

### Core Ratios (Industry Benchmarks)
- **Stable retail:** 70-90%
- **Mixed retail:** 50-70%
- **Volatile retail:** 30-50%
- **Your project:** 51% (mixed retail)

### Pass-Through Beta (Industry Benchmarks)
- **Retail NMDs:** β = 0.3-0.7
- **Wholesale:** β = 0.9-1.0
- **Checking (low):** β = 0.1-0.3
- **Savings (medium):** β = 0.4-0.7

### Duration (Industry Benchmarks)
- **Short duration:** 0-2 years (asset-sensitive)
- **Medium duration:** 2-5 years
- **Long duration:** 5+ years (liability-sensitive)
- **Your project:** ~2.5 years (medium-short)

---

## Interpretation Guide

### If ΔEVE is Negative (Like Your -$200)
**Meaning:** Rising rates HURT economic value
**Implication:** Bank is asset-sensitive (short duration)
**Action:** Consider hedging with receive-fixed swaps

### If ΔEVE is Positive
**Meaning:** Rising rates HELP economic value
**Implication:** Bank is liability-sensitive (long duration)
**Action:** Consider hedging with pay-fixed swaps

### If ΔNII is Positive (Like Your +$75 for S1)
**Meaning:** Rising rates HELP earnings
**Implication:** Assets reprice faster than liabilities
**Good for:** Short-term profitability

### If ΔNII is Negative (Like Your -$180 for S2)
**Meaning:** Falling rates HURT earnings
**Implication:** Liabilities reprice faster than assets
**Bad for:** Short-term profitability

### The Paradox in Your Results
- **EVE:** S1 (+200bps) is worst → Rising rates hurt value
- **NII:** S1 (+200bps) is best → Rising rates help earnings

**Why?** You earn more in the short term (NII ↑) but lose value long-term (EVE ↓)

---

## File Structure Reference

```
project/
├── 📄 IRRBB_CONCEPTS_GUIDE.md           ← Read this first!
├── 📄 PROJECT_LEARNING_SUMMARY.md       ← Learning path
├── 📄 QUICK_REFERENCE.md                ← This file (quick lookup)
├── 📄 QF609_PROJECT_PLAN.md             ← Original requirements
│
├── 📊 Data Files
│   ├── group-proj-1-data.xlsx           (NMD data 2017-2023)
│   └── group-proj-1-curve.xlsx          (Zero rate curve)
│
├── 🐍 Phase 1: Data & Decay
│   ├── phase_1a_data_preparation_eda.py
│   ├── phase_1b_decay_model.py          ← Added comments!
│   └── phase_1c_core_noncore_split.py   ← Added comments!
│
├── 🐍 Phase 2-4: Core Analysis
│   ├── phase_2_cashflow_slotting.py     ← Added comments!
│   ├── phase_3_eve_sensitivity.py       ← Added comments (WHY EVE!)
│   └── phase_4_nii_sensitivity.py       ← Added comments!
│
├── 🐍 Phase 5: Bonus Analysis
│   ├── phase_5_helpers.py
│   ├── phase_5a_core_ratio_sensitivity.py
│   ├── phase_5b_passthrough_analysis.py
│   ├── phase_5c_backtesting.py
│   ├── phase_5d_monte_carlo.py
│   └── phase_5e_stress_decay.py
│
└── 📊 Output CSVs (generated when you run)
    ├── processed_*.csv
    ├── survival_*.csv
    ├── core_noncore_split.csv
    ├── repricing_profile.csv
    ├── eve_sensitivity_summary.csv
    ├── nii_sensitivity_summary.csv
    └── irrbb_final_report.csv
```

---

## Debug Checklist ✓

### If Results Look Wrong

**Check 1: Cash flows sum to current balance?**
```python
total_cf = repricing_profile['Total_CF'].sum()
# Should equal current_balance = 18,652
```

**Check 2: Survival function decreasing?**
```python
# S(t) should go from 1.0 down to ~0.5
# Never increase!
```

**Check 3: Core + non-core = total?**
```python
core + non_core = current_balance
```

**Check 4: O/N bucket includes all non-core?**
```python
# O/N should be largest bucket
# Must include full non-core amount
```

**Check 5: Discount factors < 1?**
```python
# DF(t) must be between 0 and 1
# DF(0) = 1, DF(∞) → 0
```

---

## Interview Sound Bites 🎤

**"What is IRRBB?"**
> "Interest rate risk in the banking book measures how rate changes affect a bank's long-term economic value and short-term earnings. I modeled this for an $18.6M deposit portfolio."

**"What's the difference between EVE and NII?"**
> "EVE measures long-term solvency—whether the bank survives. NII measures short-term profitability—whether the bank earns money this year. Both matter, and they can have different worst cases."

**"Why do banks care about deposits?"**
> "Deposits are the bank's main funding source. If rates rise and deposits flee, the bank can't fund its loans. We model decay rates to predict this."

**"What did you find in your analysis?"**
> "The bank is asset-sensitive: rising rates help NII (+$75 under +200bps) but hurt EVE (-$200). This creates a strategic tension between short-term profitability and long-term value."

---

## One-Page Summary for Presentation

```
╔═══════════════════════════════════════════════════════════╗
║        IRRBB ANALYSIS: NMD PORTFOLIO SUMMARY              ║
╠═══════════════════════════════════════════════════════════╣
║                                                           ║
║  Portfolio Size:        $18,652                           ║
║  Data Period:           2017-2023 (7 years)               ║
║  Calculation Date:      30-Dec-2023                       ║
║                                                           ║
║  KEY PARAMETERS                                           ║
║  ├─ Daily Decay Rate:         0.15%                       ║
║  ├─ Monthly Decay Rate:       4.4%                        ║
║  ├─ 1-Year Survival:          85%                         ║
║  ├─ 5-Year Survival:          55%                         ║
║  ├─ Core Deposits:            $9,511 (51%)                ║
║  └─ Non-Core Deposits:        $9,141 (49%)                ║
║                                                           ║
║  REPRICING PROFILE                                        ║
║  ├─ O/N Bucket:               $9,500 (51%)                ║
║  ├─ 1M-1Y Buckets:            $4,800 (26%)                ║
║  └─ 2Y-5Y Buckets:            $4,300 (23%)                ║
║                                                           ║
║  EVE SENSITIVITY (LONG-TERM VALUE)                        ║
║  ├─ Base EVE:                 $18,500                     ║
║  ├─ S1 (+200bps):             $18,300 (ΔEVE: -$200)       ║
║  ├─ S2 (-200bps):             $18,750 (ΔEVE: +$250)       ║
║  ├─ S3 (Steepener):           $18,400 (ΔEVE: -$100)       ║
║  ├─ S4 (Flattener):           $18,350 (ΔEVE: -$150)       ║
║  └─ WORST CASE:               S1 (+200bps) → -1.08%       ║
║                                                           ║
║  NII SENSITIVITY (12-MONTH EARNINGS)                      ║
║  ├─ S1 (+200bps):             ΔNII: +$75                  ║
║  ├─ S2 (-200bps):             ΔNII: -$180 ★WORST          ║
║  ├─ S3 (Steepener):           ΔNII: +$50                  ║
║  └─ S4 (Flattener):           ΔNII: -$20                  ║
║                                                           ║
║  RISK PROFILE                                             ║
║  ├─ Position:                 Asset-Sensitive             ║
║  ├─ Duration:                 ~2.5 years                  ║
║  ├─ Key Risk:                 Rising rates hurt EVE       ║
║  └─ Basel Status:             PASS (ΔEVE < 15% capital)   ║
║                                                           ║
║  MANAGEMENT RECOMMENDATIONS                               ║
║  ├─ Hedge with receive-fixed swaps (offset EVE risk)      ║
║  ├─ Monitor deposit mix (high O/N concentration)          ║
║  └─ Consider term deposit promotions (lock in funding)    ║
║                                                           ║
╚═══════════════════════════════════════════════════════════╝
```

---

## Final Exam Prep Questions

1. **Define EVE in one sentence.**
2. **Why do we split deposits into core/non-core?**
3. **What is the survival function S(t)?**
4. **Why does NII only use buckets ≤ 1 year?**
5. **What's the (1-t) factor in NII formula?**
6. **Why 4 different rate shock scenarios?**
7. **What does "asset-sensitive" mean?**
8. **Why is there a 5-year cap on core deposits?**
9. **How does pass-through beta affect NII?**
10. **What's the Basel outlier threshold?**

**Can you answer all 10? Then you're ready! 🎓**

---

*Quick reference created for rapid concept lookup. For full explanations, see IRRBB_CONCEPTS_GUIDE.md*
