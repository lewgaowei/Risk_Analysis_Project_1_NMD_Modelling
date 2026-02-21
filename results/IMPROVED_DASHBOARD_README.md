# Advanced IRRBB Dashboard - Implementation Plan

## 📊 Project Overview

**Course**: QF609 (AY2025-2026) Group Project #1
**Topic**: Interest Rate Risk in the Banking Book (IRRBB) for Non-Maturity Deposits (NMD)
**Institution**: SMU Bank
**Calculation Date**: 31-Dec-2023
**Presentation Date**: Feb 23-24, 2026

---

## 🎯 Objectives

Build an advanced interactive dashboard to:
1. Estimate deposit decay profile for NMD accounts
2. Perform Basel-compliant deposit separation and cash flow slotting
3. Calculate EVE and NII sensitivities under 4 rate shock scenarios
4. Provide comprehensive risk analytics and model validation

---

## 📈 Key Improvements Over Original Dashboard

### **1. Real Data Integration**
- ✅ Use actual yield curve from `group-proj-1-curve.xlsx` (16 tenor points)
- ✅ Interpolation for missing tenors (cubic spline, linear)
- ✅ Full historical NMD analysis from `group-proj-1-data.xlsx` (2016-2023)
- ✅ Automatic data validation and cleaning

### **2. Advanced Decay Modeling**
- **Multiple Model Options**:
  - Exponential decay (standard Basel approach)
  - Logistic decay (S-curve behavioral patterns)
  - Weibull distribution (flexible hazard rates)
  - Custom piecewise decay functions
- **Statistical Calibration**:
  - Parameter estimation from historical data
  - Goodness-of-fit metrics (R², RMSE, AIC)
  - Model comparison framework
- **Behavioral Analysis**:
  - Volatility-adjusted core deposits
  - Seasonal pattern detection
  - Stability metrics

### **3. Basel-Compliant IRRBB Framework**

#### **Core/Non-Core Separation**
- Statistical methods (quantile-based, volatility-based)
- Regulatory constraints (5Y maximum behavioral maturity)
- Stable vs. unstable deposit identification

#### **Cash Flow Slotting**
- 9 time buckets: O/N, 1M, 2M, 3M, 6M, 9M, 1Y, 2Y, 3Y, 4Y, 5Y
- Midpoint allocation for discounting
- Residual balance treatment

### **4. Comprehensive Rate Shock Scenarios**

| Scenario | Description | Implementation |
|----------|-------------|----------------|
| **(a) +200bps Parallel** | Uniform upward shift | All tenors +200bps |
| **(b) -200bps Parallel** | Uniform downward shift | All tenors -200bps (floored at 0) |
| **(c) Short Rate Up** | Steepener shock | +200bps @ O/N, linear taper to 0 @ 5Y |
| **(d) Flattener** | Short up, long down | +200bps @ O/N to -100bps @ 5Y |

**Custom Features**:
- User-defined shock parameters
- Custom tenor-specific shocks
- Historical scenario replay

### **5. Risk Metrics Calculations**

#### **EVE (Economic Value of Equity)**
```
ΔEVE = PV(Assets)_shocked - PV(Liabilities)_shocked - [PV(Assets)_base - PV(Liabilities)_base]
ΔEVE% = ΔEVE / EVE_base × 100%
```
- Proper discount factor calculation: `DF(t) = 1 / (1 + r(t))^t`
- Zero-coupon bond pricing
- Duration and convexity analysis

#### **NII (Net Interest Income)**
```
NII = Σ(Cash Flow_i × Rate_i) over 1-year horizon
ΔNII = NII_shocked - NII_base
```
- 1-year earnings projection
- Repricing gap analysis
- Margin compression effects

### **6. Advanced Analytics & Visualization**

#### **Historical Analysis**
- Balance trend decomposition (trend, seasonal, residual)
- Inflow/outflow pattern analysis
- Volatility metrics (rolling std dev, VaR)
- Growth rate analysis

#### **Interactive Visualizations**
- Time-series plots with zoom/pan
- Survival curve animation
- 3D sensitivity surfaces
- Waterfall charts for risk attribution
- Heatmaps for scenario comparison

#### **Sensitivity Analysis**
- Parameter sensitivity (core %, decay rate)
- Grid-based sensitivity heatmaps
- Tornado diagrams (one-way sensitivities)
- Two-way sensitivity analysis

#### **Monte Carlo Simulation**
- Stochastic deposit balance paths
- Interest rate scenario generation
- Risk distribution (VaR, CVaR)
- Confidence intervals for EVE/NII

### **7. Model Validation & Backtesting**

- **In-Sample Fit**: Test decay model against 2016-2022 data
- **Out-of-Sample Test**: Validate on 2023 data
- **Stability Tests**: Rolling window analysis
- **Residual Diagnostics**: Autocorrelation, normality tests

### **8. Professional Features**

#### **Export Capabilities**
- **Excel Export**:
  - Multi-sheet workbook
  - Detailed calculations
  - Formatted tables
  - Charts and graphs
- **PDF Report**:
  - Executive summary
  - Model documentation
  - Risk metrics dashboard
  - Regulatory compliance checklist

#### **User Experience**
- Responsive layout (wide screen optimized)
- Loading indicators for heavy computations
- Error handling and validation
- Help tooltips and documentation
- Preset scenarios (conservative, moderate, aggressive)

---

## 🗂️ Dashboard Structure

### **Tab 1: Executive Overview**
- Key risk metrics at a glance
- Total balance, core/non-core split
- ΔEVE and ΔNII summary across all scenarios
- Worst-case scenario identification
- Quick action buttons (export, refresh)

### **Tab 2: Data Exploration**
- Raw data inspection
- Historical balance trends (2016-2023)
- Inflow/outflow analysis
- Descriptive statistics
- Data quality checks

### **Tab 3: Decay Model Calibration**
- Model selection (dropdown)
- Parameter estimation controls
- Goodness-of-fit statistics
- Model comparison table
- Survival curve visualization
- Historical fit overlay

### **Tab 4: Cash Flow Slotting**
- Core/non-core separation controls
- Time bucket allocation table
- Allocation bar chart
- Cumulative cash flow waterfall
- Maturity profile comparison

### **Tab 5: Yield Curve & Scenarios**
- Base curve visualization (actual data)
- Scenario selector with custom parameters
- Shocked curve comparison
- Rate change waterfall
- Scenario specification table

### **Tab 6: Risk Metrics**
- EVE calculation breakdown
- NII projection (1-year)
- Duration and convexity
- Scenario comparison table
- Worst-case highlighting
- Regulatory thresholds

### **Tab 7: Sensitivity & Simulation**
- Parameter sensitivity heatmaps
- Tornado charts
- Monte Carlo configuration
- Risk distribution plots
- VaR/CVaR metrics

### **Tab 8: Model Validation**
- Backtesting results
- Residual plots
- Statistical tests
- Rolling window analysis
- Model performance metrics

### **Tab 9: Export & Reports**
- Excel export button
- PDF report generation
- Custom report builder
- Data download options

---

## 🛠️ Technical Stack

### **Core Libraries**
- `streamlit` - Interactive web framework
- `pandas` - Data manipulation
- `numpy` - Numerical calculations
- `plotly` - Interactive charts
- `scipy` - Statistical functions, interpolation
- `openpyxl` - Excel export

### **Advanced Libraries**
- `statsmodels` - Statistical modeling
- `scikit-learn` - Model validation
- `reportlab` or `matplotlib` + `fpdf` - PDF generation
- `seaborn` - Statistical visualizations

---

## 📊 Data Files

1. **group-proj-1-data.xlsx**
   - 2,556 daily records (Dec 31, 2016 - Dec 30, 2023)
   - Columns: Date, Balance, Inflow, Outflow, Netflow

2. **group-proj-1-curve.xlsx**
   - 16 tenor points (1D to 10Y)
   - Zero rates (3.18% to 2.64%)

---

## 🚀 Implementation Phases

### **Phase 1: Foundation** ✅
- Load and validate data
- Curve interpolation
- Basic decay models

### **Phase 2: Core Functionality**
- Cash flow slotting
- All 4 shock scenarios
- EVE/NII calculations

### **Phase 3: Analytics**
- Historical analysis
- Sensitivity analysis
- Visualizations

### **Phase 4: Advanced Features**
- Monte Carlo simulation
- Model validation
- Backtesting

### **Phase 5: Polish**
- Export functionality
- UI/UX improvements
- Documentation

---

## 📝 Regulatory Compliance

### **Basel IRRBB Standards**
- ✅ 5-year maximum behavioral maturity cap
- ✅ Core/non-core separation
- ✅ Standardized shock scenarios
- ✅ EVE and NII metrics
- ✅ Worst-case reporting

### **Model Risk Management**
- ✅ Multiple model approaches
- ✅ Backtesting framework
- ✅ Parameter sensitivity
- ✅ Documentation and audit trail

---

## 🎓 Educational Value

### **Learning Outcomes**
1. Practical implementation of Basel IRRBB framework
2. Understanding NMD behavioral modeling
3. Interest rate risk measurement techniques
4. Model validation and backtesting
5. Professional dashboard development

---

## 📅 Timeline

| Milestone | Target | Status |
|-----------|--------|--------|
| Plan & Design | Feb 20 | ✅ Complete |
| Core Implementation | Feb 21 | 🔄 In Progress |
| Testing & Validation | Feb 22 | ⏳ Pending |
| Final Presentation | Feb 23-24 | ⏳ Pending |

---

## 👥 Usage Instructions

### **Running the Dashboard**
```bash
streamlit run improved_irrbb_dashboard.py
```

### **Key Features**
1. **Interactive Controls**: Use sliders and dropdowns in sidebar
2. **Model Comparison**: Toggle between different decay models
3. **Scenario Analysis**: Select shock scenarios and see live updates
4. **Export**: Download results as Excel or PDF

---

## 📚 References

- Basel Committee on Banking Supervision: "Interest rate risk in the banking book"
- BCBS 368: Standards for IRRBB
- QF609 Course Materials: IRRBB Framework and NMD Modeling

---

## 🔮 Future Enhancements

- [ ] Real-time data integration
- [ ] Machine learning for deposit forecasting
- [ ] Multi-currency support
- [ ] Portfolio-level IRRBB aggregation
- [ ] Stress testing framework
- [ ] API integration for automated reporting

---

**Last Updated**: February 20, 2026
**Version**: 2.0 (Advanced Implementation)
**File**: `improved_irrbb_dashboard.py`
