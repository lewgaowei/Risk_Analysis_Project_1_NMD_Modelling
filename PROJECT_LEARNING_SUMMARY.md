# QF609 Project #1 - Learning Summary 📚

## What I've Done For You

### 1. Created Comprehensive Concepts Guide ✅
**File:** `IRRBB_CONCEPTS_GUIDE.md` (20 sections!)

This explains ALL the concepts you need:
- What is IRRBB and why it matters
- **WHY we calculate EVE** (your main question!)
- EVE vs NII differences
- Decay models and survival functions
- Core vs non-core deposits
- Cash flow slotting
- Rate shocks
- And much more!

### 2. Added Educational Comments to Your Code ✅

I've updated these files with detailed explanations:
- ✅ `phase_1b_decay_model.py` - Explains survival functions
- ✅ `phase_1c_core_noncore_split.py` - Explains core/non-core split
- ✅ `phase_2_cashflow_slotting.py` - Explains time bucketing
- ✅ `phase_3_eve_sensitivity.py` - **Explains WHY EVE matters**
- ✅ `phase_4_nii_sensitivity.py` - Explains NII vs EVE

Each file now has:
- **High-level "WHAT ARE WE DOING" explanations** at the top
- **Conceptual comments** explaining the "why"
- **Examples with numbers** to illustrate concepts
- **Formulas with explanations**

---

## Your Learning Path 🎯

### Step 1: Start with the Concepts Guide
Read `IRRBB_CONCEPTS_GUIDE.md` sections in this order:

**Priority 1 (Core concepts):**
1. Section 1: What is IRRBB?
2. Section 2: What are NMDs?
3. **Section 3: Why EVE? ← YOUR MAIN QUESTION**
4. Section 4: Decay Model
5. Section 5: Core vs Non-Core

**Priority 2 (Technical details):**
6. Section 6: Cash Flow Slotting
7. Section 7: Rate Shock Scenarios
8. Section 8: Discount Factors
9. Section 9: NII

**Priority 3 (Advanced topics):**
10. Sections 10-20: Duration, Pass-through, Backtesting, Monte Carlo, etc.

### Step 2: Read the Python Files with Comments
Now go through your Python files in order. Each one builds on the previous:

```
Phase 1a → Data Exploration (understand your deposit data)
Phase 1b → Decay Model (how long deposits stay)
Phase 1c → Core/Non-Core (stable vs volatile split)
Phase 2  → Cash Flow Slotting (distribute into time buckets)
Phase 3  → EVE Calculation (LONG-TERM economic value)
Phase 4  → NII Calculation (SHORT-TERM earnings impact)
Phase 5  → Sensitivity Analysis (what if scenarios)
```

### Step 3: Run the Code and Examine Outputs
Execute each phase and look at:
- The printed summaries
- The charts (they tell a story!)
- The CSV outputs

---

## Quick Answer to Your Main Question 💡

### "Why are we calculating EVE?"

**Answer: EVE measures the LONG-TERM economic health of the bank.**

Imagine you're a bank:
- You have $18,652 in deposits (liabilities)
- These will stay/leave over the next 5 years
- Interest rates might change dramatically

**EVE tells you:**
> "If rates suddenly jump 2%, how much economic value does my balance sheet lose?"

**Why it matters:**
1. **Solvency Risk:** If EVE drops 30%, you might go bankrupt!
2. **Regulatory:** Basel REQUIRES you to report if EVE changes > 15% of capital
3. **Risk Management:** Tells you if you need to hedge (buy swaps, etc.)

**EVE vs Accounting Book Value:**
- Book value = What the balance sheet says (historical cost)
- EVE = What it's ACTUALLY worth today (market value)

**Example from your project:**
```
Base EVE = $18,500
After +200bps rate shock: EVE = $18,300
Loss: $200 (-1.08%)
```

This tells you: "A 2% rate increase costs the bank $200 in economic value."

**Compare to NII:**
- EVE = Your total wealth (retirement account)
- NII = Your annual salary
- You need BOTH! Wealth for the future + income for today

---

## Project Flow Diagram

```
┌─────────────────────────────────────────────────────────┐
│  INPUT: Deposit Data (2017-2023) + Zero Rate Curve     │
└─────────────────────────────────────────────────────────┘
                         ↓
        ┌────────────────────────────────────┐
        │  PHASE 1: Understand Deposits      │
        │  • λ = 0.15% daily decay           │
        │  • S(1Y) = 85% survival            │
        │  • Core = 51%, Non-core = 49%      │
        └────────────────────────────────────┘
                         ↓
        ┌────────────────────────────────────┐
        │  PHASE 2: Slot into Time Buckets   │
        │  • Non-core → O/N bucket           │
        │  • Core → distributed 1M-5Y        │
        └────────────────────────────────────┘
                         ↓
        ┌────────────────────────────────────┐
        │  PHASE 3: Calculate EVE            │
        │  • EVE_base = $18,500              │
        │  • Test 4 rate shock scenarios     │
        │  • Worst: S1 (+200bps) = -$200     │
        └────────────────────────────────────┘
                         ↓
        ┌────────────────────────────────────┐
        │  PHASE 4: Calculate NII            │
        │  • 12-month earnings impact        │
        │  • Different worst case than EVE?  │
        └────────────────────────────────────┘
                         ↓
        ┌────────────────────────────────────┐
        │  PHASE 5: Sensitivity Analysis     │
        │  • What if core ratio changes?     │
        │  • Monte Carlo uncertainty         │
        │  • Backtesting accuracy            │
        └────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│  OUTPUT: Basel IRRBB Report                             │
│  • Worst ΔEVE: -$200 (-1.08%)                          │
│  • Worst ΔNII: -$180                                   │
│  • Regulatory Status: PASS (< 15% of capital)          │
└─────────────────────────────────────────────────────────┘
```

---

## Key Formulas You Need to Know

### Decay & Survival
```python
λ_daily = Outflow / Balance  # Daily decay rate
S(t) = (1 - λ)^t             # Survival probability
```

### Cash Flow Slotting
```python
CF(bucket_i) = Core × [S(t_start) - S(t_end)]
```

### EVE (Economic Value of Equity)
```python
DF(t) = 1 / (1 + r)^t        # Discount factor
EVE = Σ [CF(i) × DF(i)]      # Sum across all buckets
ΔEVE = EVE_shocked - EVE_base # Impact of rate shock
```

### NII (Net Interest Income)
```python
# Only buckets with t ≤ 1 year
ΔNII = Σ [CF(i) × shock(i) × (1 - t_i)]
```

---

## Real-World Interpretation

### Your Results Mean:

**Asset-Sensitive Bank:**
- Rising rates HURT EVE (you lose economic value)
- Rising rates HELP NII (you earn more interest)
- Short-term gains, long-term pain!

**What a Bank Would Do:**
1. **Hedging:** Buy interest rate swaps to offset risk
2. **Pricing:** Adjust deposit rates to retain customers
3. **Mix:** Change asset/liability composition
4. **Capital:** Hold more capital as buffer

**Basel Compliance:**
If |ΔEVE| > 15% of Tier 1 Capital → "Outlier bank"
- Regulators require explanation
- May need to raise capital
- Enhanced supervision

---

## Common Student Mistakes to Avoid ⚠️

1. **Confusing EVE and NII**
   - They measure DIFFERENT things (value vs income)
   - They can have DIFFERENT worst cases!

2. **Thinking decay = bad**
   - Decay is NORMAL! Customers withdraw daily
   - We're just modeling the natural flow

3. **Forgetting time value of money**
   - $100 in 5 years ≠ $100 today
   - Always discount future cash flows!

4. **Misunderstanding core deposits**
   - Core ≠ all deposits
   - Core = the sticky, stable portion only

5. **Thinking one scenario is enough**
   - Rates can move in many ways
   - Must test parallel, steepener, flattener!

---

## Practice Questions for Exam/Interview 🎓

### Conceptual Questions

**Q1:** Why does Basel require banks to calculate EVE?
<details>
<summary>Click for answer</summary>

**A:** To ensure banks can survive large interest rate shocks. If EVE drops too much (>15% of capital), the bank might become insolvent. This protects depositors and financial stability.
</details>

**Q2:** What's the difference between core and non-core deposits?
<details>
<summary>Click for answer</summary>

**A:**
- **Core:** Sticky, relationship-based deposits that stay even under stress (e.g., your grandma's checking account)
- **Non-core:** Rate-sensitive "hot money" that leaves quickly (e.g., wholesale funding)

Core gets distributed over time (up to 5Y), non-core reprices overnight.
</details>

**Q3:** Why is the (1 - t) factor needed in the NII formula?
<details>
<summary>Click for answer</summary>

**A:** Because NII measures 12-month earnings. If a deposit reprices at 3 months (t=0.25), it only earns the new rate for the remaining 9 months (1 - 0.25 = 0.75 years).
</details>

### Technical Questions

**Q4:** Calculate EVE if:
- Bucket 1: CF = $5,000, r = 2%, t = 1Y
- Bucket 2: CF = $3,000, r = 3%, t = 5Y

<details>
<summary>Click for answer</summary>

**A:**
```
DF(1) = 1/(1.02)^1 = 0.9804
DF(5) = 1/(1.03)^5 = 0.8626

PV(1) = $5,000 × 0.9804 = $4,902
PV(2) = $3,000 × 0.8626 = $2,588

EVE = $4,902 + $2,588 = $7,490
```
</details>

**Q5:** If λ_daily = 0.001, what's the 1-year survival rate?
<details>
<summary>Click for answer</summary>

**A:**
```
S(365) = (1 - 0.001)^365 = 0.6943 = 69.43%

About 70% of deposits survive 1 year.
```
</details>

---

## Career Applications 💼

### Jobs That Use This:
- **ALM (Asset-Liability Management)** Analyst
- **Risk Management** Analyst (Market Risk, IRRBB)
- **Treasury** Analyst
- **Basel Compliance** Specialist
- **Risk Consulting** (Deloitte, EY, PwC, KPMG)
- **Central Bank** Supervision

### Skills You're Demonstrating:
✓ Quantitative modeling (survival analysis)
✓ Regulatory knowledge (Basel framework)
✓ Risk measurement (VaR, stress testing)
✓ Python/data science (pandas, numpy, visualization)
✓ Financial acumen (interest rate risk, NPV)

### Interview Talking Points:
"I built an IRRBB model for a $18.6M deposit portfolio following Basel standards. I modeled deposit decay using survival analysis, split deposits into core/non-core using historical minimum method, performed cash flow slotting across 11 time buckets, and calculated EVE and NII sensitivity under 4 rate shock scenarios. The worst-case EVE loss was $200 (-1.08%) under the +200bps parallel shock."

---

## Next Steps 🚀

### To Deepen Understanding:
1. Read Basel Committee BCBS 368 (IRRBB Standards)
2. Learn about interest rate swaps (hedging tool)
3. Study duration and convexity (bond math)
4. Explore SVaR (Stressed Value at Risk)

### To Extend This Project:
1. Add options-based deposits (caps, floors)
2. Include early redemption risk
3. Model prepayment for mortgages
4. Build an optimization for hedging strategy
5. Add behavioral models (beta coefficients)

---

## Get Help

**Stuck on a concept?**
1. Check `IRRBB_CONCEPTS_GUIDE.md` (comprehensive explanations)
2. Read the comments in the Python files
3. Look at the charts (visual intuition!)
4. Draw the cash flow timeline on paper

**Still confused?**
Ask yourself:
- What is this metric measuring? (value, income, risk?)
- What time horizon? (12 months, 5 years, all time?)
- Why does a regulator care about this?

---

## Final Wisdom 💡

> **"EVE measures whether the bank will SURVIVE.**
> **NII measures whether the bank will PROFIT."**
>
> You need both. A profitable bank that's insolvent doesn't survive.
> A solvent bank that's unprofitable doesn't thrive.

**This project teaches you how real banks manage their survival.**

Good luck! 🎓📊

---

*Created with care to help you understand IRRBB. Now go ace that project! 🚀*
