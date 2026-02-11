# IRRBB Concepts Guide - QF609 Project #1

## 📚 Essential Concepts You Need to Understand

---

## 1. What is IRRBB (Interest Rate Risk in the Banking Book)?

### The Big Picture
Banks have two main "books":
- **Trading Book**: Securities held for short-term trading (mark-to-market daily)
- **Banking Book**: Long-term assets and liabilities (loans, deposits, etc.)

**IRRBB = Interest Rate Risk in the Banking Book**

### Why Do We Care?
When interest rates change, the economic value of a bank's assets and liabilities changes:
- If rates ↑ → Present value of future cash flows ↓
- If rates ↓ → Present value of future cash flows ↑

**Example:**
You have a deposit of $100,000 that will stay with you for 5 years. If interest rates suddenly jump from 2% to 4%, the present value of that deposit liability changes. This affects the bank's equity!

---

## 2. What are Non-Maturity Deposits (NMDs)?

### Definition
**NMDs** are deposits that:
- Have NO fixed maturity date (unlike a 5-year CD)
- Can be withdrawn anytime by the customer
- Examples: Savings accounts, checking accounts

### The Challenge
Since they have no maturity, how do we know when cash flows will occur?
- A 5-year loan → predictable cash flows
- A savings account → customer could withdraw tomorrow or stay for 10 years!

**This is why we need the decay model** → to estimate when deposits will leave.

---

## 3. Why Do We Calculate EVE? (Your Main Question!)

### EVE = Economic Value of Equity

**EVE measures the long-term economic value of the bank's equity under different interest rate scenarios.**

### The Formula
```
EVE = PV(Assets) - PV(Liabilities)
```

Where PV = Present Value (discounted cash flows)

### Why is EVE Important?

**1. Long-Term Solvency**
EVE tells us if the bank will still be solvent if rates change dramatically.

**Example:**
- Base case: EVE = $1,000,000
- Rates rise 200bps: EVE = $950,000
- Change: ΔEVE = -$50,000 (-5%)

This -$50,000 loss means the bank's long-term economic capital decreased by $50,000.

**2. Regulatory Requirement (Basel)**
Basel Committee requires banks to:
- Calculate EVE under 6 standardized shock scenarios
- Report if ΔEVE > 15% of Tier 1 Capital (outlier threshold)
- Banks must hold capital to cover interest rate risk

**3. Different from Book Value**
- **Book value** (accounting): What's on the balance sheet
- **Economic value** (EVE): True present value using current market rates

### EVE vs NII - What's the Difference?

| Metric | Time Horizon | What It Measures | Why It Matters |
|--------|--------------|------------------|----------------|
| **EVE** | Long-term (all cash flows) | Economic value change | Will the bank survive? (solvency) |
| **NII** | Short-term (12 months) | Earnings change | Will the bank be profitable this year? |

**Analogy:**
- **EVE** = Total net worth of your retirement portfolio
- **NII** = Your monthly income this year

Both matter! You need income today (NII) and wealth for the future (EVE).

---

## 4. The Decay Model - Why Do We Need It?

### The Problem
How long will a dollar deposited today stay in the bank?

### The Solution: Survival Function S(t)
**S(t) = Probability that $1 deposited at time 0 is still in the account at time t**

**Example:**
- S(0) = 100% (just deposited, it's definitely there)
- S(365 days) = 85% (85% chance it's still there after 1 year)
- S(1825 days) = 55% (55% chance it's still there after 5 years)

### The Decay Rate (λ lambda)
**λ = daily decay rate = Outflow / Balance**

This is the conditional probability of withdrawal:
```
S(t) = (1 - λ)^t
```

**Example:**
- λ_daily = 0.0015 (0.15% per day)
- After 1 year: S(365) = (1 - 0.0015)^365 = 0.578 = 57.8%
- Meaning: 42.2% of deposits decay over 1 year

### Why Exponential Decay?
We assume:
- Each day, there's a constant probability of withdrawal (λ)
- Withdrawals are independent of previous withdrawals
- This is the same math as radioactive decay in physics!

---

## 5. Core vs Non-Core Deposits

### Basel Regulatory Framework
Basel requires us to split NMDs into two buckets:

### Non-Core (Volatile) Deposits
**Definition:** Deposits that are rate-sensitive and can leave quickly
- **Behavior:** Customers shop around for best rates
- **Repricing:** Immediate (Overnight bucket)
- **Example:** Hot money, wholesale funding

### Core (Stable) Deposits
**Definition:** Deposits unlikely to leave even under stress
- **Behavior:** Sticky, relationship-based
- **Repricing:** Distributed over time (up to 5 years max)
- **Example:** Long-term customer checking accounts

### How to Estimate Core Floor?
**Method 1: Historical Minimum**
```
Core Floor = min(Balance over entire history)
```
Logic: The balance level that survived even during crisis periods

**Why 51% in Your Project?**
```
Minimum Balance = 9,511
Current Balance = 18,652
Core Ratio = 9,511 / 18,652 = 51%
```

### Regulatory Constraint
Basel caps core deposits at **maximum 5-year behavioral maturity** to prevent banks from assuming deposits will never leave.

---

## 6. Cash Flow Slotting

### What is It?
**Distributing the NMD balance across time buckets based on when we expect repricing.**

### The 11 Time Buckets
| Bucket | Midpoint | What Goes Here |
|--------|----------|----------------|
| O/N | 0 years | ALL non-core + tiny bit of core |
| 1M | 0.042 years | Core deposits expected to reprice in 1 month |
| 2M | 0.125 years | Core deposits expected to reprice in 2 months |
| ... | ... | ... |
| 5Y | 4.5 years | Remaining core deposits (5-year cap) |

### The Slotting Formula
For each bucket i:
```
Cash Flow(i) = Core × [S(t_start) - S(t_end)]
```

**Example for 1Y bucket (9M to 1Y):**
```
S(270 days) = 0.65
S(365 days) = 0.60
Cash Flow = 9,511 × (0.65 - 0.60) = 475.55
```

This means **$475.55 of core deposits will reprice between 9 months and 1 year.**

### Why Do This?
We need to know WHEN cash flows occur to:
1. Discount them properly (earlier CF = higher PV)
2. Calculate repricing risk (what's exposed to rate changes?)
3. Meet Basel reporting requirements

---

## 7. Rate Shock Scenarios

### Why 4 Scenarios?
Different rate movements affect the bank differently:

### Scenario 1: +200bps Parallel
- **What:** All rates ↑ by 200 basis points (2%)
- **Impact:** Hurts if bank has long-duration liabilities
- **Example:** 1Y rate goes from 3% → 5%

### Scenario 2: -200bps Parallel
- **What:** All rates ↓ by 200 basis points
- **Impact:** Hurts if bank has short-duration liabilities
- **Floor:** Rates can't go below 0% (zero floor)

### Scenario 3: Steepener (Short Rate Up)
- **What:** Short rates ↑ 200bps, long rates unchanged
- **Impact:** Yield curve gets steeper
- **Example:** 1M rate ↑ 200bps, 10Y rate ↑ 0bps

### Scenario 4: Flattener
- **What:** Short rates ↑ 200bps, long rates ↓ 100bps
- **Impact:** Yield curve gets flatter
- **Example:** 1M rate ↑ 200bps, 10Y rate ↓ 100bps

### Why These Specific Shocks?
Basel Committee chose these based on:
- Historical rate movements
- Worst-case scenarios observed globally
- Capturing both parallel and non-parallel shifts

---

## 8. Discount Factors and Present Value

### Why Discount?
**$100 received in 5 years is worth LESS than $100 today.**

### The Discount Factor Formula
```
DF(t) = 1 / (1 + r)^t
```

**Example:**
- Zero rate at 5Y = 4%
- DF(5) = 1 / (1.04)^5 = 0.8219
- $100 in 5 years = $100 × 0.8219 = $82.19 today

### Zero Rates
**Zero rate = Interest rate for a zero-coupon bond of maturity t**

Your curve has 16 points (1D, 1M, 2M, ... 10Y). We interpolate for intermediate tenors.

### Calculating EVE
```
EVE = Σ [Cash Flow(i) × Discount Factor(i)]
```

**Example:**
| Bucket | Cash Flow | DF | Present Value |
|--------|-----------|----|--------------|
| O/N | $9,000 | 0.9999 | $8,999.10 |
| 1M | $500 | 0.9960 | $498.00 |
| 1Y | $1,000 | 0.9650 | $965.00 |
| **Total EVE** | | | **$18,500** |

---

## 9. NII (Net Interest Income) Sensitivity

### What is NII?
**NII = Interest earned on assets - Interest paid on liabilities**

### 12-Month Horizon
NII only looks at **buckets that reprice within 1 year:**
- O/N, 1M, 2M, 3M, 6M, 9M, 1Y ✓
- 2Y, 3Y, 4Y, 5Y ✗ (not included)

### The NII Formula
```
ΔNII = Σ [CF × shock × (1 - t)]
```

Where:
- **CF** = Cash flow in bucket
- **shock** = Rate shock at that tenor
- **(1 - t)** = Remaining time in year

### Why the (1 - t) Factor?

**Example:** 3M bucket (t = 0.25 years)
- Reprices at 3 months
- Earns shocked rate for remaining 9 months (0.75 years)
- Time factor = 1 - 0.25 = 0.75

**Calculation:**
```
CF = $5,000
Shock = +2% (200bps)
Time factor = 0.75
ΔNII = $5,000 × 0.02 × 0.75 = $75
```

So you earn an **extra $75 in interest income** over the 12-month period.

---

## 10. Duration and Convexity

### Duration
**Duration = Sensitivity of price to interest rate changes**

```
Duration ≈ (PV_down - PV_up) / (2 × PV_base × Δy)
```

**Interpretation:**
- Duration of 3.5 years means:
  - 1% rate increase → ~3.5% decrease in value
  - 1% rate decrease → ~3.5% increase in value

### Convexity
**Convexity = How duration changes as rates change (2nd derivative)**

Captures non-linear effects. Important for large rate shocks like ±200bps.

---

## 11. Pass-Through Rate (Beta)

### What is It?
**β = How much of a market rate change is passed to depositors**

### The Model
```
Deposit Rate = α + β × Market Rate
```

**Example:**
- Market rate increases by 1%
- β = 0.5 (50% pass-through)
- Deposit rate increases by 0.5%

### Why Does It Matter?
Higher β → Bank passes rate changes to customers → **Less NII sensitivity**

**Example:**
- Rates ↑ 200bps, β = 0
  - Bank earns more, pays same → Big NII increase
- Rates ↑ 200bps, β = 1
  - Bank earns more, pays more → No NII change

### Typical Values
- **Retail NMDs:** β = 0.3 to 0.7 (partial pass-through)
- **Wholesale:** β = 0.9 to 1.0 (nearly full pass-through)

---

## 12. Backtesting

### What is It?
**Testing if our decay model predictions match reality.**

### The Process
1. **Train:** Use 2017-2022 data to estimate λ
2. **Test:** Predict 2023 balances using trained λ
3. **Compare:** Actual vs Predicted balances

### Metrics
- **RMSE** (Root Mean Square Error): Average prediction error
- **MAPE** (Mean Absolute Percentage Error): Percentage error

### Why Pure Decay Fails
The simple decay model assumes:
- No new deposits (inflows)
- Balance decays monotonically

Reality:
- Customers make new deposits daily!
- Balance fluctuates, doesn't just decay

**Solution:** Monthly re-anchoring (reset prediction each month)

---

## 13. Monte Carlo Simulation

### What is It?
**Running 1,000 different "what if" scenarios to see range of outcomes.**

### The Stochastic Model
```
Outflow(t) = Balance(t) × max(λ + ε, 0)
ε ~ N(0, σ²)
```

**ε = random shock** (some days more withdrawals, some days less)

### Why Do This?
1. **Quantify uncertainty:** What's the 5th percentile worst case?
2. **VaR (Value at Risk):** 95% confidence worst-case loss
3. **Expected Shortfall:** Average loss in worst 5% of cases

### Fan Chart
Shows range of possible balance paths:
- 5th to 95th percentile (90% confidence band)
- 25th to 75th percentile (50% confidence band)
- Median (50th percentile)

---

## 14. Key Banking Concepts

### Asset-Sensitive vs Liability-Sensitive

**Asset-Sensitive Bank:**
- Assets reprice faster than liabilities
- **Benefits** from rate increases (NII ↑)
- **Hurt** by rate decreases (NII ↓)

**Liability-Sensitive Bank:**
- Liabilities reprice faster than assets
- **Hurt** by rate increases (NII ↓)
- **Benefits** from rate decreases (NII ↑)

### Your Project
Large O/N bucket (non-core) → Significant repricing risk → Need to manage carefully!

---

## 15. Basel Regulatory Framework

### Why Basel?
International banking regulations to ensure financial stability after 2008 crisis.

### IRRBB Standards (Basel Committee BCBS 368)
Banks must:
1. Calculate **EVE** under 6 scenarios (you do 4)
2. Calculate **NII** under same scenarios
3. Report if outlier: **|ΔEVE| > 15% of Tier 1 Capital**
4. Hold capital for interest rate risk

### Core Deposit Rules
- Maximum 90% of NMDs can be core (retail)
- Maximum 5-year behavioral maturity
- Must use survival function (no ad-hoc assumptions)
- Stress testing required

---

## 16. Practical Interpretation

### How to Read Your Results

**EVE Table:**
```
Scenario          EVE      ΔEVE     ΔEVE_%
Base Case      18,500       0       0%
S1: +200bps    18,300    -200    -1.08%
S2: -200bps    18,750    +250    +1.35%
Worst Case: S1 (rates up hurt you)
```

**What This Means:**
- You're **asset-sensitive** (short duration)
- Rising rates → Economic value decreases
- Bank should consider:
  - Interest rate swaps to hedge
  - Adjust deposit pricing
  - Change asset/liability mix

**NII Table:**
```
Scenario         ΔNII
S1: +200bps    +150   (Good! Earn more)
S2: -200bps    -180   (Bad! Earn less)
Worst Case: S2 (rates down hurt earnings)
```

---

## 17. Your Project Flow Summary

```
Phase 1a: Data Exploration
   ↓
Phase 1b: Estimate λ (decay rate) → Build S(t)
   ↓
Phase 1c: Split into Core (51%) vs Non-Core (49%)
   ↓
Phase 2: Slot cash flows into 11 time buckets
   ↓
Phase 3: Calculate EVE base and 4 shocked scenarios
   ↓
Phase 4: Calculate NII for 12-month horizon
   ↓
Phase 5: Sensitivity analysis, backtesting, Monte Carlo
```

---

## 18. Key Formulas Cheat Sheet

```python
# Decay
λ_daily = mean(Outflow / Balance)
S(t) = (1 - λ_daily)^t

# Cash Flow Slotting
CF(bucket_i) = Core × [S(t_start) - S(t_end)]

# Discount Factor
DF(t) = 1 / (1 + r)^t

# EVE
EVE = Σ [CF(i) × DF(i)]
ΔEVE = EVE_shocked - EVE_base

# NII (12-month buckets only)
ΔNII = Σ [CF(i) × shock(i) × (1 - t_i)]

# Duration
D = (EVE_down - EVE_up) / (2 × EVE_base × Δy)

# Pass-Through
Effective_Shock = shock × (1 - β)
```

---

## 19. Common Mistakes to Avoid

1. **Confusing EVE and NII**
   - EVE = long-term value
   - NII = 12-month earnings

2. **Forgetting the zero floor**
   - Rates can't go negative (in most frameworks)

3. **Wrong time conversion**
   - 30 days ≠ 1 month exactly
   - 1 year = 365 days (or 360 for some conventions)

4. **Survival function confusion**
   - S(t) is always between 0 and 1
   - Must be decreasing (can't gain deposits in decay model)

5. **Bucket boundaries**
   - O/N bucket includes non-core + first day of core decay
   - Don't double-count!

---

## 20. Why This Matters for Your Career

### Skills You're Learning
✓ **Risk Management:** Basel IRRBB framework
✓ **Quantitative Modeling:** Survival analysis, Monte Carlo
✓ **Banking Operations:** NMD management, ALM (Asset-Liability Management)
✓ **Regulatory Compliance:** Basel standards
✓ **Python/Data Science:** pandas, numpy, visualization

### Career Applications
- **Risk Analyst** at banks
- **ALM (Asset-Liability Management)** specialist
- **Treasury** analyst
- **Risk Consulting** (Big 4, boutiques)
- **Central Bank** regulation

This project mirrors **real-world banking book risk management**!

---

## Questions for Self-Check

1. Why does the bank care about EVE?
2. What's the difference between core and non-core deposits?
3. Why can't all deposits be considered core?
4. Why do we need 4 different rate scenarios?
5. Why is NII only 12 months but EVE uses all cash flows?
6. What does a high O/N bucket percentage tell you about the bank's risk?
7. How does pass-through rate affect NII sensitivity?

**If you can answer these, you understand the project! 🎓**
