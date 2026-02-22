"""
QF609 Group Project #1 - Advanced IRRBB Dashboard
SMU Bank NMD Risk Analysis

Uses Phase 1-4 outputs:
- Core ratio from config.json (Phase 1c Detrended Regression)
- Survival curve from survival_curve_full_advanced.csv (Phase 1b Portfolio KM)
- Yield curve from processed_curve_data.csv (Phase 1a)
- 11 IRRBB buckets from repricing_profile.csv (Phase 2)
- EVE sensitivity (Phase 3) and NII sensitivity (Phase 4)

Run with: streamlit run improved_irrbb_dashboard.py
"""

import json
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.interpolate import interp1d
import warnings
warnings.filterwarnings('ignore')

# ===========================
# PAGE CONFIG
# ===========================

st.set_page_config(
    page_title="Advanced IRRBB Dashboard - SMU Bank",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===========================
# CUSTOM CSS
# ===========================

st.markdown("""
<style>
.big-font {
    font-size:24px !important;
    font-weight: bold;
    color: #1f77b4;
}
.metric-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 20px;
    border-radius: 15px;
    color: white;
    margin: 10px 0;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
}

/* Tab Styling */
.stTabs [data-baseweb="tab-list"] {
    gap: 4px;
    background-color: #f8f9fa;
    padding: 8px;
    border-radius: 10px;
}

.stTabs [data-baseweb="tab"] {
    height: 55px;
    background-color: #e9ecef;
    border-radius: 8px;
    padding: 12px 24px;
    color: #495057;
    font-weight: 600;
    font-size: 15px;
    border: 2px solid transparent;
    transition: all 0.3s ease;
}

.stTabs [data-baseweb="tab"]:hover {
    background-color: #dee2e6;
    border-color: #adb5bd;
    transform: translateY(-2px);
    box-shadow: 0 4px 8px rgba(0,0,0,0.1);
}

.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white !important;
    border-color: #667eea;
    box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
}

.stTabs [data-baseweb="tab-panel"] {
    padding-top: 20px;
}
</style>
""", unsafe_allow_html=True)

# ===========================
# BUCKET STRUCTURE (Phase 2: 11 IRRBB buckets)
# ===========================

BUCKET_STRUCTURE = {
    'O/N':  {'start_days': 0,    'end_days': 1,    'midpoint_years': 1/365},
    '1M':   {'start_days': 1,    'end_days': 30,   'midpoint_years': 0.0417},
    '2M':   {'start_days': 30,   'end_days': 60,   'midpoint_years': 0.125},
    '3M':   {'start_days': 60,   'end_days': 90,   'midpoint_years': 0.2083},
    '6M':   {'start_days': 90,   'end_days': 180,  'midpoint_years': 0.375},
    '9M':   {'start_days': 180,  'end_days': 270,  'midpoint_years': 0.625},
    '1Y':   {'start_days': 270,  'end_days': 365,  'midpoint_years': 0.875},
    '2Y':   {'start_days': 365,  'end_days': 730,  'midpoint_years': 1.5},
    '3Y':   {'start_days': 730,  'end_days': 1095, 'midpoint_years': 2.5},
    '4Y':   {'start_days': 1095, 'end_days': 1460, 'midpoint_years': 3.5},
    '5Y':   {'start_days': 1460, 'end_days': 1825, 'midpoint_years': 4.5},
}

# ===========================
# DATA LOADING FUNCTIONS
# ===========================

@st.cache_data
def load_nmd_data():
    """Load NMD historical data"""
    df = pd.read_excel('group-proj-1-data.xlsx')
    df.columns = ['Date', 'Balance', 'Inflow', 'Outflow', 'Netflow']
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    return df

@st.cache_data
def load_config():
    """Load core/non-core config from Phase 1c"""
    with open('config.json', 'r') as f:
        return json.load(f)

@st.cache_data
def load_yield_curve():
    """Load processed yield curve from Phase 1a"""
    curve = pd.read_csv('processed_curve_data.csv')
    curve = curve.rename(columns={'Tenor_Years': 'Years', 'ZeroRate': 'ZeroRate'})
    curve = curve.sort_values('Years')
    return curve

@st.cache_data
def load_survival_curve():
    """Load Portfolio KM survival curve from Phase 1b"""
    surv = pd.read_csv('survival_curve_full_advanced.csv')
    return surv

@st.cache_data
def build_curve_interpolator(curve_df):
    """Build linear interpolation function for yield curve (matching Phase 3)"""
    interpolator = interp1d(
        curve_df['Years'].values,
        curve_df['ZeroRate'].values,
        kind='linear',
        bounds_error=False,
        fill_value='extrapolate'
    )
    return interpolator

def build_survival_interpolator(surv_df):
    """Build interpolation function for Portfolio KM survival curve"""
    interpolator = interp1d(
        surv_df['Days'].values,
        surv_df['S(t)'].values,
        kind='linear',
        bounds_error=False,
        fill_value=(1.0, surv_df['S(t)'].values[-1])
    )
    return interpolator

# ===========================
# CASH FLOW SLOTTING FUNCTIONS
# ===========================

def allocate_cash_flows(balance, core_ratio, survival_func, bucket_structure):
    """
    Allocate NMD balance into 11 IRRBB buckets using Portfolio KM survival curve.
    Non-core -> O/N. Core -> 1M-5Y via S(t). Residual beyond 5Y -> 5Y bucket.
    """
    core_amount = balance * core_ratio
    non_core_amount = balance * (1 - core_ratio)

    core_alloc = {}
    non_core_alloc = {}

    for bucket_name, params in bucket_structure.items():
        if bucket_name == 'O/N':
            core_alloc[bucket_name] = 0.0
            non_core_alloc[bucket_name] = non_core_amount
            continue

        non_core_alloc[bucket_name] = 0.0
        s_start = float(survival_func(params['start_days']))
        s_end = float(survival_func(params['end_days']))
        core_alloc[bucket_name] = core_amount * (s_start - s_end)

    # Basel 5Y cap: residual surviving beyond 1825 days -> add to 5Y
    s_5y = float(survival_func(1825))
    core_alloc['5Y'] += core_amount * s_5y

    return core_alloc, non_core_alloc

# ===========================
# RISK CALCULATIONS
# ===========================

def apply_rate_shock(base_years, base_rates, scenario):
    """Apply rate shock scenarios matching Phase 3/4 definitions"""
    if scenario == "(a) +200bps Parallel":
        shocked_rates = base_rates + 0.02

    elif scenario == "(b) -200bps Parallel":
        shocked_rates = np.maximum(base_rates - 0.02, 0.0)

    elif scenario == "(c) Steepener (Short Up)":
        # +200bps at O/N, taper to 0 at 5Y
        taper = np.maximum(1 - base_years / 5.0, 0)
        shocked_rates = base_rates + 0.02 * taper

    elif scenario == "(d) Flattener":
        # Piecewise: +200bps at t=0, 0 at 2Y pivot, -100bps at 5Y
        flatten_shocks = np.where(
            base_years <= 2,
            0.02 * (1 - base_years / 2),
            -0.01 * (base_years - 2) / (5 - 2)
        )
        shocked_rates = np.maximum(base_rates + flatten_shocks, 0.0)

    else:
        shocked_rates = base_rates

    shocked_interpolator = interp1d(base_years, shocked_rates,
                                    kind='linear', bounds_error=False,
                                    fill_value='extrapolate')
    return shocked_interpolator, shocked_rates

def calculate_eve(cash_flows, midpoints, curve_interpolator):
    """EVE = sum(CF_i * DF_i) where DF = 1/(1+r)^t (Phase 3 formula)"""
    pv = 0.0
    for cf, t in zip(cash_flows, midpoints):
        if cf > 0:
            rate = max(float(curve_interpolator(t)), 0)
            df = 1.0 / ((1.0 + rate) ** t)
            pv += cf * df
    return pv

def calculate_nii(cash_flows, midpoints, curve_interpolator):
    """NII = sum(CF_i * r_i * (1 - t_i)) for t_i <= 1Y (Phase 4 formula)"""
    nii = 0.0
    for cf, t in zip(cash_flows, midpoints):
        if cf > 0 and t <= 1.0:
            rate = max(float(curve_interpolator(t)), 0)
            nii += cf * rate * (1 - t)
    return nii

def calculate_delta_nii(cash_flows, midpoints, base_interpolator, shocked_interpolator):
    """DNII = sum(CF_i * shock_i * (1 - t_i)) for t_i <= 1Y (Phase 4 formula)"""
    dnii = 0.0
    for cf, t in zip(cash_flows, midpoints):
        if cf > 0 and t <= 1.0:
            r_base = max(float(base_interpolator(t)), 0)
            r_shocked = max(float(shocked_interpolator(t)), 0)
            shock = r_shocked - r_base
            dnii += cf * shock * (1 - t)
    return dnii

# ===========================
# VISUALIZATION FUNCTIONS
# ===========================

def plot_balance_history(df):
    """Plot historical balance trend"""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df['Date'], y=df['Balance'],
        mode='lines', name='Balance',
        line=dict(color='#1f77b4', width=2),
        fill='tozeroy', fillcolor='rgba(31, 119, 180, 0.2)'
    ))
    fig.update_layout(
        title="Historical NMD Balance (2016-2023)",
        xaxis_title="Date", yaxis_title="Balance ($)",
        height=400, hovermode='x unified'
    )
    return fig

def plot_survival_curve(surv_df):
    """Plot Portfolio KM survival curve"""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=surv_df['Years'], y=surv_df['S(t)'] * 100,
        mode='lines', fill='tozeroy',
        line=dict(color='#2ecc71', width=3),
        name='Portfolio KM S(t)'
    ))
    fig.add_vline(x=5, line_dash="dash", line_color="red",
                  annotation_text="5Y Regulatory Cap")
    fig.update_layout(
        title="Deposit Survival Curve - Portfolio Kaplan-Meier",
        xaxis_title="Years", yaxis_title="% of Core Remaining",
        height=400, yaxis_range=[0, 105], xaxis_range=[0, 5.5]
    )
    return fig

def plot_cash_flow_allocation(slotting_df):
    """Plot cash flow allocation by bucket"""
    fig = go.Figure()
    colors = ['#e74c3c' if row['Time Bucket'] == 'O/N' else '#3498db'
              for _, row in slotting_df.iterrows()]
    fig.add_trace(go.Bar(
        x=slotting_df['Time Bucket'], y=slotting_df['Total_CF'],
        marker_color=colors,
        text=[f"${val:,.0f}<br>({pct:.1f}%)"
              for val, pct in zip(slotting_df['Total_CF'], slotting_df['CF_Percent'])],
        textposition='outside',
        hovertemplate='<b>%{x}</b><br>Cash Flow: $%{y:,.0f}<extra></extra>'
    ))
    fig.update_layout(
        title="Cash Flow Allocation by Time Bucket",
        xaxis_title="Time Bucket", yaxis_title="Cash Flow ($)",
        height=450, showlegend=False
    )
    return fig

def plot_yield_curves(curve_df, shocked_rates, scenario_name):
    """Plot base and shocked yield curves"""
    fig = go.Figure()
    years = curve_df['Years'].values
    base_rates = curve_df['ZeroRate'].values
    fig.add_trace(go.Scatter(
        x=years, y=base_rates * 100,
        mode='lines+markers', name='Base Curve',
        line=dict(color='#2c3e50', width=3), marker=dict(size=8)
    ))
    fig.add_trace(go.Scatter(
        x=years, y=shocked_rates * 100,
        mode='lines+markers', name=f'{scenario_name}',
        line=dict(color='#e74c3c', width=3, dash='dash'), marker=dict(size=8)
    ))
    fig.update_layout(
        title=f"Yield Curve Comparison - {scenario_name}",
        xaxis_title="Tenor (Years)", yaxis_title="Zero Rate (%)",
        height=450, hovermode='x unified',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(range=[-0.5, 10.5])
    )
    return fig

# ===========================
# MAIN APP
# ===========================

def main():
    # Header
    st.markdown('<p class="big-font">🏦 Advanced IRRBB Dashboard - SMU Bank</p>', unsafe_allow_html=True)
    st.markdown("**Non-Maturity Deposit (NMD) Analysis** | Calculation Date: 30-Dec-2023")

    # Load data
    try:
        df_nmd = load_nmd_data()
        config = load_config()
        curve_df = load_yield_curve()
        surv_df = load_survival_curve()
        curve_interpolator = build_curve_interpolator(curve_df)
        survival_interp = build_survival_interpolator(surv_df)
        CALC_DATE_BALANCE = config['current_balance']
    except Exception as e:
        st.error(f"Error loading data files: {e}")
        st.info("Ensure config.json, processed_curve_data.csv, survival_curve_full_advanced.csv, and group-proj-1-data.xlsx are present.")
        st.stop()

    # ===========================
    # SIDEBAR - Model Configuration (Phase 1-4 defaults, read-only)
    # ===========================
    st.sidebar.header("Model Configuration")

    st.sidebar.subheader("1. Core/Non-Core (Phase 1c)")
    st.sidebar.markdown(f"**Method:** {config['method']}")
    st.sidebar.markdown(f"**Core Ratio:** {config['core_ratio_pct']:.2f}%")
    st.sidebar.markdown(f"**Core Amount:** ${config['core_amount']:,.2f}")
    st.sidebar.markdown(f"**Non-Core Amount:** ${config['non_core_amount']:,.2f}")

    core_pct = config['core_ratio_pct'] / 100
    core_amount = CALC_DATE_BALANCE * core_pct
    non_core_amount = CALC_DATE_BALANCE * (1 - core_pct)

    st.sidebar.markdown("---")
    st.sidebar.subheader("2. Survival Model (Phase 1b)")
    st.sidebar.markdown("**Model:** Portfolio Kaplan-Meier")
    st.sidebar.markdown(f"**S(1Y):** {float(survival_interp(365)):.2%}")
    st.sidebar.markdown(f"**S(5Y):** {float(survival_interp(1825)):.2%}")

    st.sidebar.markdown("---")
    st.sidebar.subheader("3. Non-Core Allocation")
    st.sidebar.markdown("**Method:** 100% O/N (immediate repricing)")

    st.sidebar.markdown("---")
    st.sidebar.subheader("4. Rate Shock Scenario")
    scenario = st.sidebar.selectbox(
        "Select Scenario:",
        ["(a) +200bps Parallel", "(b) -200bps Parallel",
         "(c) Steepener (Short Up)", "(d) Flattener"]
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Yield Curve:** Phase 1a (processed_curve_data.csv)")
    st.sidebar.markdown("**Buckets:** 11 IRRBB (Phase 2)")

    # ===========================
    # CALCULATIONS
    # ===========================

    # Allocate cash flows using Portfolio KM survival
    core_alloc, non_core_alloc = allocate_cash_flows(
        CALC_DATE_BALANCE, core_pct, survival_interp, BUCKET_STRUCTURE
    )

    # Create slotting dataframe
    slotting_data = []
    for bucket_name, params in BUCKET_STRUCTURE.items():
        total_cf = core_alloc[bucket_name] + non_core_alloc[bucket_name]
        slotting_data.append({
            'Time Bucket': bucket_name,
            'Midpoint (Years)': params['midpoint_years'],
            'Core_CF': core_alloc[bucket_name],
            'Non_Core_CF': non_core_alloc[bucket_name],
            'Total_CF': total_cf,
            'CF_Percent': (total_cf / CALC_DATE_BALANCE) * 100
        })
    slotting_df = pd.DataFrame(slotting_data)

    # Apply rate shock
    base_years = curve_df['Years'].values
    base_rates = curve_df['ZeroRate'].values
    shocked_interpolator, shocked_rates = apply_rate_shock(base_years, base_rates, scenario)

    # Calculate EVE
    cash_flows = slotting_df['Total_CF'].values
    midpoints = slotting_df['Midpoint (Years)'].values

    eve_base = calculate_eve(cash_flows, midpoints, curve_interpolator)
    eve_shocked = calculate_eve(cash_flows, midpoints, shocked_interpolator)
    delta_eve = eve_shocked - eve_base
    delta_eve_pct = (delta_eve / eve_base) * 100

    # Calculate NII
    nii_base = calculate_nii(cash_flows, midpoints, curve_interpolator)
    nii_shocked = calculate_nii(cash_flows, midpoints, shocked_interpolator)
    delta_nii = calculate_delta_nii(cash_flows, midpoints, curve_interpolator, shocked_interpolator)
    delta_nii_pct = (delta_nii / nii_base) * 100 if nii_base != 0 else 0

    # ===========================
    # DISPLAY TABS
    # ===========================

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Overview",
        "📈 Historical Balance & Projection",
        "🔄 Survival Model",
        "💰 Cash Flow Slotting",
        "📉 Risk Metrics"
    ])

    with tab1:
        st.subheader("Key Metrics")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Total Balance",
                f"${CALC_DATE_BALANCE:,.0f}",
                f"As of 30-Dec-2023"
            )

        with col2:
            st.metric(
                "Core Deposit",
                f"${core_amount:,.0f}",
                f"{core_pct:.1%} (Detrended Regression)"
            )

        with col3:
            st.metric(
                "ΔEVE",
                f"${delta_eve:,.0f}",
                f"{delta_eve_pct:+.2f}%",
                delta_color="inverse"
            )

        with col4:
            st.metric(
                "ΔNII",
                f"${delta_nii:,.0f}",
                f"{delta_nii_pct:+.2f}%",
                delta_color="inverse"
            )

        st.markdown("---")

        col1, col2 = st.columns(2)

        with col1:
            st.plotly_chart(plot_cash_flow_allocation(slotting_df), use_container_width=True)

        with col2:
            st.plotly_chart(plot_yield_curves(curve_df, shocked_rates, scenario), use_container_width=True)

        # Summary table
        st.subheader("Cash Flow Summary")
        display_df = slotting_df.copy()
        display_df['Core_CF'] = display_df['Core_CF'].apply(lambda x: f"${x:,.2f}")
        display_df['Non_Core_CF'] = display_df['Non_Core_CF'].apply(lambda x: f"${x:,.2f}")
        display_df['Total_CF'] = display_df['Total_CF'].apply(lambda x: f"${x:,.2f}")
        display_df['CF_Percent'] = display_df['CF_Percent'].apply(lambda x: f"{x:.2f}%")
        st.dataframe(display_df, use_container_width=True, hide_index=True)

    with tab2:
        st.subheader("Historical Balance & Forward Projection")

        # Create projection based on Portfolio KM survival
        projection_days = np.arange(0, 5 * 365 + 1)
        core_projection = core_amount * np.array([float(survival_interp(d)) for d in projection_days])
        non_core_projection = np.full(len(projection_days), non_core_amount)
        total_projection = core_projection + non_core_projection

        from datetime import datetime, timedelta
        calc_date = datetime(2023, 12, 30)
        projection_dates = [calc_date + timedelta(days=int(d)) for d in projection_days]

        fig_projection = go.Figure()

        fig_projection.add_trace(go.Scatter(
            x=df_nmd['Date'], y=df_nmd['Balance'],
            mode='lines', name='Historical Balance',
            line=dict(color='#1f77b4', width=2),
            fill='tozeroy', fillcolor='rgba(31, 119, 180, 0.2)'
        ))

        fig_projection.add_trace(go.Scatter(
            x=projection_dates, y=total_projection,
            mode='lines', name='Total Projected',
            line=dict(color='#2ecc71', width=3, dash='solid')
        ))

        fig_projection.add_trace(go.Scatter(
            x=projection_dates, y=core_projection,
            mode='lines', name='Core Projected',
            line=dict(color='#9b59b6', width=2, dash='dash')
        ))

        fig_projection.add_trace(go.Scatter(
            x=projection_dates, y=non_core_projection,
            mode='lines', name='Non-Core (Stable)',
            line=dict(color='#e74c3c', width=2, dash='dot')
        ))

        fig_projection.add_shape(
            type="line", x0=calc_date, x1=calc_date, y0=0, y1=1,
            yref="paper", line=dict(color="gray", width=2, dash="dash")
        )
        fig_projection.add_annotation(
            x=calc_date, y=1, yref="paper",
            text="Calc Date (30-Dec-2023)",
            showarrow=False, yshift=10, font=dict(color="gray")
        )

        fig_projection.update_layout(
            title="Historical Balance (2016-2023) & Projected Balance (2024-2028) - Portfolio KM",
            xaxis_title="Date", yaxis_title="Balance ($)",
            height=500, hovermode='x unified',
            legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
        )

        st.plotly_chart(fig_projection, use_container_width=True)

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Starting Balance", f"${df_nmd['Balance'].iloc[0]:,.0f}", "31-Dec-2016")
        with col2:
            st.metric("Current Balance", f"${CALC_DATE_BALANCE:,.0f}", "30-Dec-2023")
        with col3:
            balance_5y = total_projection[-1]
            st.metric("Projected (5Y)", f"${balance_5y:,.0f}", "30-Dec-2028")
        with col4:
            decay_5y = ((balance_5y - CALC_DATE_BALANCE) / CALC_DATE_BALANCE) * 100
            st.metric("5Y Change", f"{decay_5y:+.1f}%", "vs Current")

        st.markdown("---")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### Core Deposit Projection")
            st.write(f"**Model:** Portfolio Kaplan-Meier (Phase 1b)")
            st.write(f"**Method:** Detrended Regression (Phase 1c)")
            st.write(f"**Initial Core:** ${core_amount:,.0f}")
            st.write(f"**Projected Core (5Y):** ${core_projection[-1]:,.0f}")
            core_decay_pct = ((core_projection[-1] - core_amount) / core_amount) * 100
            st.write(f"**Core Decay:** {core_decay_pct:+.1f}%")

        with col2:
            st.markdown("### Non-Core Deposit")
            st.write(f"**Allocation:** 100% O/N (immediate repricing)")
            st.write(f"**Initial Non-Core:** ${non_core_amount:,.0f}")
            st.write(f"**Assumption:** Stable over horizon")

        st.markdown("---")
        with st.expander("View Historical Raw Data"):
            st.dataframe(df_nmd, use_container_width=True)

    with tab3:
        st.subheader("Survival Model: Portfolio Kaplan-Meier (Phase 1b)")

        col1, col2 = st.columns([2, 1])

        with col1:
            st.plotly_chart(plot_survival_curve(surv_df), use_container_width=True)

        with col2:
            st.markdown("### Model Details")
            st.write("**Type:** Non-parametric (Portfolio KM)")
            st.write("**Source:** Phase 1b analysis")
            st.write("**Data:** Historical outflow observations")

            st.markdown("---")
            st.markdown("### Survival at Key Points")
            for years_pt in [0.25, 0.5, 1, 2, 3, 4, 5]:
                days_pt = years_pt * 365
                s_val = float(survival_interp(days_pt)) * 100
                st.write(f"**{years_pt}Y:** {s_val:.2f}%")

    with tab4:
        st.subheader("Cash Flow Slotting Details (11 IRRBB Buckets)")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Core Deposits")
            st.write(f"**Amount:** ${core_amount:,.2f}")
            st.write(f"**Percentage:** {core_pct:.1%}")
            st.write(f"**Method:** Detrended Regression")
            st.write(f"**Survival:** Portfolio KM")

        with col2:
            st.markdown("### Non-Core Deposits")
            st.write(f"**Amount:** ${non_core_amount:,.2f}")
            st.write(f"**Percentage:** {1 - core_pct:.1%}")
            st.write(f"**Allocation:** 100% O/N")

        st.markdown("---")

        # Waterfall chart
        fig_waterfall = go.Figure(go.Waterfall(
            name="Cash Flow", orientation="v",
            measure=["relative"] * len(slotting_df),
            x=slotting_df['Time Bucket'], y=slotting_df['Total_CF'],
            text=[f"${val:,.0f}" for val in slotting_df['Total_CF']],
            textposition="outside",
            connector={"line": {"color": "rgb(63, 63, 63)"}},
        ))
        fig_waterfall.update_layout(title="Cash Flow Waterfall by Bucket", height=400)
        st.plotly_chart(fig_waterfall, use_container_width=True)

    with tab5:
        st.subheader(f"Risk Metrics - {scenario}")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("EVE (Base)", f"${eve_base:,.0f}")
            st.metric("EVE (Shocked)", f"${eve_shocked:,.0f}")
            st.metric("ΔEVE", f"${delta_eve:,.0f}", f"{delta_eve_pct:+.2f}%")

        with col2:
            st.metric("NII (Base)", f"${nii_base:,.0f}")
            st.metric("NII (Shocked)", f"${nii_shocked:,.0f}")
            st.metric("ΔNII", f"${delta_nii:,.0f}", f"{delta_nii_pct:+.2f}%")

        with col3:
            st.markdown("### Risk Summary")
            if abs(delta_eve_pct) > 15:
                st.error(f"High EVE Risk: {delta_eve_pct:+.2f}%")
            elif abs(delta_eve_pct) > 10:
                st.warning(f"Moderate EVE Risk: {delta_eve_pct:+.2f}%")
            else:
                st.success(f"Low EVE Risk: {delta_eve_pct:+.2f}%")

        st.markdown("---")

        # All scenarios comparison
        st.subheader("All Scenarios Comparison")

        all_scenarios = [
            "(a) +200bps Parallel", "(b) -200bps Parallel",
            "(c) Steepener (Short Up)", "(d) Flattener"
        ]
        scenario_results = []

        for scen in all_scenarios:
            shock_interp, _ = apply_rate_shock(base_years, base_rates, scen)
            eve_s = calculate_eve(cash_flows, midpoints, shock_interp)
            nii_s = calculate_nii(cash_flows, midpoints, shock_interp)
            d_eve = eve_s - eve_base
            d_nii = calculate_delta_nii(cash_flows, midpoints, curve_interpolator, shock_interp)

            scenario_results.append({
                'Scenario': scen,
                'EVE': eve_s,
                'ΔEVE': d_eve,
                'ΔEVE %': (d_eve / eve_base) * 100,
                'NII': nii_s,
                'ΔNII': d_nii,
                'ΔNII %': (d_nii / nii_base) * 100 if nii_base != 0 else 0
            })

        comparison_df = pd.DataFrame(scenario_results)

        worst_eve_idx = comparison_df['ΔEVE'].idxmin()
        worst_nii_idx = comparison_df['ΔNII'].idxmin()

        st.dataframe(
            comparison_df.style.highlight_max(subset=['ΔEVE %', 'ΔNII %'], axis=0, color='lightcoral')
                              .highlight_min(subset=['ΔEVE %', 'ΔNII %'], axis=0, color='lightgreen')
                              .format({
                                  'EVE': '${:,.0f}',
                                  'ΔEVE': '${:,.0f}',
                                  'ΔEVE %': '{:+.2f}%',
                                  'NII': '${:,.0f}',
                                  'ΔNII': '${:,.0f}',
                                  'ΔNII %': '{:+.2f}%'
                              }),
            use_container_width=True
        )

        st.info(f"**Worst EVE Scenario:** {comparison_df.loc[worst_eve_idx, 'Scenario']} "
                f"(ΔEVE = ${comparison_df.loc[worst_eve_idx, 'ΔEVE']:,.0f}, "
                f"{comparison_df.loc[worst_eve_idx, 'ΔEVE %']:+.2f}%)")
        st.info(f"**Worst NII Scenario:** {comparison_df.loc[worst_nii_idx, 'Scenario']} "
                f"(ΔNII = ${comparison_df.loc[worst_nii_idx, 'ΔNII']:,.0f}, "
                f"{comparison_df.loc[worst_nii_idx, 'ΔNII %']:+.2f}%)")

        # Effective Duration
        st.markdown("---")
        st.subheader("Effective Duration")
        delta_y = 0.02  # 200bps
        eve_up = scenario_results[0]['EVE']   # (a) +200bps
        eve_dn = scenario_results[1]['EVE']   # (b) -200bps
        eff_duration = (eve_dn - eve_up) / (2 * eve_base * delta_y)
        st.metric("Effective Duration", f"{eff_duration:.2f} years")
        st.write(f"Formula: (EVE_down - EVE_up) / (2 x EVE_base x delta_y)")
        st.write(f"= ({eve_dn:,.0f} - {eve_up:,.0f}) / (2 x {eve_base:,.0f} x {delta_y})")

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: gray;'>
        <p>QF609 Group Project #1 - Advanced IRRBB for Non-Maturity Deposits</p>
        <p>SMU Bank | Calculation Date: 30-Dec-2023 | Model: Detrended Regression + Portfolio KM</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
