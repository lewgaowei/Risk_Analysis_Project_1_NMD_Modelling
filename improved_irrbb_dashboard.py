"""
QF609 Group Project #1 - Advanced IRRBB Dashboard
SMU Bank NMD Risk Analysis - Improved Version

Run with: streamlit run improved_irrbb_dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy import interpolate
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
# BUCKET STRUCTURE (FROM GROUP'S CSV)
# ===========================

BUCKET_STRUCTURE = {
    'O/N':  {'start_days': 0,    'end_days': 1,    'midpoint_years': 0.0027},
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
def load_yield_curve():
    """Load yield curve data"""
    curve = pd.read_excel('group-proj-1-curve.xlsx')
    curve.columns = ['Tenor', 'ZeroRate']
    return curve

def parse_tenor_to_years(tenor_str):
    """Convert tenor string to years"""
    tenor_str = str(tenor_str).strip()
    tenor_map = {
        '1D': 1/365, '1W': 7/365, '2W': 14/365,
        '1M': 1/12, '2M': 2/12, '3M': 3/12, '6M': 6/12, '9M': 9/12,
        '1Y': 1, '2Y': 2, '3Y': 3, '4Y': 4, '5Y': 5,
        '6Y': 6, '7Y': 7, '8Y': 8, '9Y': 9, '10Y': 10
    }
    if tenor_str in tenor_map:
        return tenor_map[tenor_str]
    else:
        # Try to parse as number
        try:
            return float(tenor_str.replace('Y', ''))
        except:
            return 1.0  # Default fallback

@st.cache_data
def build_curve_interpolator(curve_df):
    """Build interpolation function for yield curve"""
    curve_df['Years'] = curve_df['Tenor'].apply(parse_tenor_to_years)
    curve_df = curve_df.sort_values('Years')

    # Cubic spline interpolation
    interpolator = interpolate.CubicSpline(
        curve_df['Years'].values,
        curve_df['ZeroRate'].values,
        bc_type='natural'
    )
    return interpolator, curve_df

# ===========================
# DECAY MODEL FUNCTIONS
# ===========================

def exponential_decay(t_days, decay_rate_monthly):
    """Exponential decay: S(t) = exp(-λt)"""
    t_months = t_days / 30.0
    return np.exp(-decay_rate_monthly * t_months)

def logistic_decay(t_days, L=1.0, k=0.01, t0=180):
    """Logistic decay: S(t) = L / (1 + exp(k(t-t0)))"""
    return L / (1 + np.exp(k * (t_days - t0)))

def weibull_survival(t_days, lambda_param=500, k_param=1.5):
    """Weibull survival: S(t) = exp(-(t/λ)^k)"""
    return np.exp(-((t_days / lambda_param) ** k_param))

def custom_piecewise_decay(t_days):
    """Custom piecewise decay function"""
    t_years = t_days / 365.0
    if t_years < 0.5:
        return 1.0 - 0.3 * t_years
    elif t_years < 2.0:
        return 0.85 - 0.2 * (t_years - 0.5)
    else:
        return max(0.55 - 0.1 * (t_years - 2.0), 0.2)

def calculate_survival_curve(model_type, params, max_days=1825):
    """Calculate survival curve for given model and parameters"""
    days = np.arange(0, max_days + 1)

    if model_type == "Exponential":
        survival = exponential_decay(days, params['decay_rate'])
    elif model_type == "Logistic":
        survival = logistic_decay(days, params['L'], params['k'], params['t0'])
    elif model_type == "Weibull":
        survival = weibull_survival(days, params['lambda'], params['k'])
    else:  # Custom
        survival = np.array([custom_piecewise_decay(d) for d in days])

    return days, survival

# ===========================
# CASH FLOW SLOTTING FUNCTIONS
# ===========================

def allocate_core_deposits(core_amount, survival_func, bucket_structure):
    """Allocate core deposits across buckets based on survival function"""
    allocation = {}

    for bucket_name, params in bucket_structure.items():
        if bucket_name == 'O/N':
            allocation[bucket_name] = 0.0  # Core doesn't go to O/N
            continue

        start_days = params['start_days']
        end_days = params['end_days']

        # Calculate marginal decay in this bucket
        s_start = survival_func(start_days)
        s_end = survival_func(end_days)
        marginal_decay = s_start - s_end

        allocation[bucket_name] = core_amount * marginal_decay

    # Add residual to 5Y bucket
    s_5y = survival_func(1825)  # 5 years = 1825 days
    allocation['5Y'] += core_amount * s_5y

    return allocation

def allocate_non_core_deposits(non_core_amount, allocation_method, bucket_structure):
    """Allocate non-core deposits based on selected method"""
    allocation = {bucket: 0.0 for bucket in bucket_structure.keys()}

    if allocation_method == "100% O/N":
        allocation['O/N'] = non_core_amount
    elif allocation_method == "100% 1M":
        allocation['1M'] = non_core_amount
    elif allocation_method == "50% O/N, 50% 1M":
        allocation['O/N'] = non_core_amount * 0.5
        allocation['1M'] = non_core_amount * 0.5
    elif allocation_method == "Distributed (O/N to 3M)":
        allocation['O/N'] = non_core_amount * 0.6
        allocation['1M'] = non_core_amount * 0.25
        allocation['2M'] = non_core_amount * 0.10
        allocation['3M'] = non_core_amount * 0.05

    return allocation

# ===========================
# RISK CALCULATIONS
# ===========================

def apply_rate_shock(base_curve_df, scenario):
    """Apply rate shock scenarios"""
    years = base_curve_df['Years'].values
    base_rates = base_curve_df['ZeroRate'].values

    if scenario == "+200bps Parallel":
        shocked_rates = base_rates + 0.02

    elif scenario == "-200bps Parallel":
        shocked_rates = np.maximum(base_rates - 0.02, 0.0)

    elif scenario == "Short Rate Up":
        # +200bps at O/N, taper to 0 at 5Y
        shock = np.array([0.02 * max(0, (5 - y) / 5) for y in years])
        shocked_rates = base_rates + shock

    elif scenario == "Flattener":
        # +200bps at short end, -100bps at long end
        shock = np.array([0.02 - 0.03 * min(y / 5, 1.0) for y in years])
        shocked_rates = np.maximum(base_rates + shock, 0.0)

    else:  # Custom
        shocked_rates = base_rates  # Placeholder

    # Create interpolator for shocked curve
    shocked_interpolator = interpolate.CubicSpline(years, shocked_rates, bc_type='natural')

    return shocked_interpolator, shocked_rates

def calculate_discount_factor(rate, time_years):
    """Calculate discount factor: DF(t) = 1 / (1 + r)^t"""
    return 1.0 / ((1.0 + rate) ** time_years)

def calculate_eve(cash_flows, midpoints, curve_interpolator):
    """Calculate Economic Value of Equity (PV of cash flows)"""
    pv = 0.0

    for cf, t in zip(cash_flows, midpoints):
        if cf > 0:
            rate = curve_interpolator(t)
            df = calculate_discount_factor(rate, t)
            pv += cf * df

    return pv

def calculate_nii(cash_flows, midpoints, curve_interpolator, horizon_years=1.0):
    """Calculate Net Interest Income over horizon"""
    nii = 0.0

    for cf, t in zip(cash_flows, midpoints):
        if cf > 0 and t <= horizon_years:
            rate = curve_interpolator(t)
            nii += cf * rate

    return nii

# ===========================
# VISUALIZATION FUNCTIONS
# ===========================

def plot_balance_history(df):
    """Plot historical balance trend"""
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df['Date'],
        y=df['Balance'],
        mode='lines',
        name='Balance',
        line=dict(color='#1f77b4', width=2),
        fill='tozeroy',
        fillcolor='rgba(31, 119, 180, 0.2)'
    ))

    fig.update_layout(
        title="Historical NMD Balance (2016-2023)",
        xaxis_title="Date",
        yaxis_title="Balance ($)",
        height=400,
        hovermode='x unified'
    )

    return fig

def plot_survival_curve(days, survival, model_name):
    """Plot decay survival curve"""
    fig = go.Figure()

    years = days / 365.0

    fig.add_trace(go.Scatter(
        x=years,
        y=survival * 100,
        mode='lines',
        fill='tozeroy',
        line=dict(color='#2ecc71', width=3),
        name=f'{model_name} S(t)'
    ))

    fig.add_vline(x=5, line_dash="dash", line_color="red",
                  annotation_text="5Y Regulatory Cap")

    fig.update_layout(
        title=f"Deposit Survival Curve - {model_name} Model",
        xaxis_title="Years",
        yaxis_title="% of Core Remaining",
        height=400,
        yaxis_range=[0, 105],
        xaxis_range=[0, 5.5]
    )

    return fig

def plot_cash_flow_allocation(slotting_df):
    """Plot cash flow allocation by bucket"""
    fig = go.Figure()

    colors = ['#e74c3c' if row['Time Bucket'] == 'O/N' else '#3498db'
              for _, row in slotting_df.iterrows()]

    fig.add_trace(go.Bar(
        x=slotting_df['Time Bucket'],
        y=slotting_df['Total_CF'],
        marker_color=colors,
        text=[f"${val:,.0f}<br>({pct:.1f}%)"
              for val, pct in zip(slotting_df['Total_CF'], slotting_df['CF_Percent'])],
        textposition='outside',
        hovertemplate='<b>%{x}</b><br>Cash Flow: $%{y:,.0f}<extra></extra>'
    ))

    fig.update_layout(
        title="Cash Flow Allocation by Time Bucket",
        xaxis_title="Time Bucket",
        yaxis_title="Cash Flow ($)",
        height=450,
        showlegend=False
    )

    return fig

def plot_yield_curves(base_curve_df, shocked_rates, scenario_name):
    """Plot base and shocked yield curves"""
    fig = go.Figure()

    years = base_curve_df['Years'].values
    base_rates = base_curve_df['ZeroRate'].values

    fig.add_trace(go.Scatter(
        x=years,
        y=base_rates * 100,
        mode='lines+markers',
        name='Base Curve',
        line=dict(color='#2c3e50', width=3),
        marker=dict(size=8)
    ))

    fig.add_trace(go.Scatter(
        x=years,
        y=shocked_rates * 100,
        mode='lines+markers',
        name=f'{scenario_name}',
        line=dict(color='#e74c3c', width=3, dash='dash'),
        marker=dict(size=8)
    ))

    fig.update_layout(
        title=f"Yield Curve Comparison - {scenario_name}",
        xaxis_title="Tenor (Years)",
        yaxis_title="Zero Rate (%)",
        height=450,
        hovermode='x unified',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(range=[-0.5, 10.5])  # Show full range up to 10Y
    )

    return fig

# ===========================
# MAIN APP
# ===========================

def main():
    # Header
    st.markdown('<p class="big-font">🏦 Advanced IRRBB Dashboard - SMU Bank</p>', unsafe_allow_html=True)
    st.markdown("**Non-Maturity Deposit (NMD) Analysis** | Calculation Date: 31-Dec-2023")

    # Load data
    try:
        df_nmd = load_nmd_data()
        curve_raw = load_yield_curve()
        curve_interpolator, curve_df = build_curve_interpolator(curve_raw)
        CALC_DATE_BALANCE = df_nmd[df_nmd['Date'] == '2023-12-30']['Balance'].values[0]
    except Exception as e:
        st.error(f"⚠️ Error loading data files: {e}")
        st.info("Please ensure 'group-proj-1-data.xlsx' and 'group-proj-1-curve.xlsx' are in the working directory.")
        st.stop()

    # Sidebar
    st.sidebar.header("⚙️ Model Configuration")

    # Decay Model Selection
    st.sidebar.subheader("1️⃣ Decay Model")
    decay_model = st.sidebar.selectbox(
        "Select Model:",
        ["Exponential", "Logistic", "Weibull", "Custom Piecewise"]
    )

    # Model Parameters
    if decay_model == "Exponential":
        decay_rate = st.sidebar.slider("Monthly Decay Rate (%)", 0.1, 5.0, 0.5, 0.1) / 100
        model_params = {'decay_rate': decay_rate}
        survival_func = lambda t: exponential_decay(t, decay_rate)

    elif decay_model == "Logistic":
        L = st.sidebar.slider("Max Survival (L)", 0.5, 1.0, 1.0, 0.05)
        k = st.sidebar.slider("Steepness (k)", 0.001, 0.05, 0.01, 0.001)
        t0 = st.sidebar.slider("Inflection Point (days)", 90, 730, 180, 30)
        model_params = {'L': L, 'k': k, 't0': t0}
        survival_func = lambda t: logistic_decay(t, L, k, t0)

    elif decay_model == "Weibull":
        lambda_param = st.sidebar.slider("Scale (λ)", 100, 1000, 500, 50)
        k_param = st.sidebar.slider("Shape (k)", 0.5, 3.0, 1.5, 0.1)
        model_params = {'lambda': lambda_param, 'k': k_param}
        survival_func = lambda t: weibull_survival(t, lambda_param, k_param)

    else:  # Custom
        model_params = {}
        survival_func = lambda t: custom_piecewise_decay(t)

    # Core/Non-Core Split
    st.sidebar.subheader("2️⃣ Core/Non-Core Split")
    core_pct = st.sidebar.slider("Core Deposit %", 30.0, 90.0, 64.1, 0.5)

    core_amount = CALC_DATE_BALANCE * (core_pct / 100)
    non_core_amount = CALC_DATE_BALANCE - core_amount

    # Non-Core Allocation
    st.sidebar.subheader("3️⃣ Non-Core Allocation")
    non_core_method = st.sidebar.radio(
        "Method:",
        ["100% O/N", "100% 1M", "50% O/N, 50% 1M", "Distributed (O/N to 3M)"]
    )

    # Rate Shock Scenario
    st.sidebar.subheader("4️⃣ Rate Shock Scenario")
    scenario = st.sidebar.selectbox(
        "Select Scenario:",
        ["+200bps Parallel", "-200bps Parallel", "Short Rate Up", "Flattener"]
    )

    st.sidebar.markdown("---")

    # Export buttons
    if st.sidebar.button("📥 Export to Excel"):
        st.sidebar.info("Excel export coming soon!")

    # ===========================
    # CALCULATIONS
    # ===========================

    # Calculate survival curve
    days_range, survival_values = calculate_survival_curve(decay_model, model_params)

    # Allocate cash flows
    core_allocation = allocate_core_deposits(core_amount, survival_func, BUCKET_STRUCTURE)
    non_core_allocation = allocate_non_core_deposits(non_core_amount, non_core_method, BUCKET_STRUCTURE)

    # Create slotting dataframe
    slotting_data = []
    for bucket_name, params in BUCKET_STRUCTURE.items():
        total_cf = core_allocation[bucket_name] + non_core_allocation[bucket_name]
        slotting_data.append({
            'Time Bucket': bucket_name,
            'Midpoint (Years)': params['midpoint_years'],
            'Core_CF': core_allocation[bucket_name],
            'Non_Core_CF': non_core_allocation[bucket_name],
            'Total_CF': total_cf,
            'CF_Percent': (total_cf / CALC_DATE_BALANCE) * 100
        })

    slotting_df = pd.DataFrame(slotting_data)

    # Apply rate shock
    shocked_interpolator, shocked_rates = apply_rate_shock(curve_df, scenario)

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
    delta_nii = nii_shocked - nii_base
    delta_nii_pct = (delta_nii / nii_base) * 100 if nii_base != 0 else 0

    # ===========================
    # DISPLAY TABS
    # ===========================

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Overview",
        "📈 Historical Balance & Projection",
        "🔄 Decay Model",
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
                f"{core_pct:.1f}%"
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
        st.subheader("📋 Cash Flow Summary")
        display_df = slotting_df.copy()
        display_df['Core_CF'] = display_df['Core_CF'].apply(lambda x: f"${x:,.2f}")
        display_df['Non_Core_CF'] = display_df['Non_Core_CF'].apply(lambda x: f"${x:,.2f}")
        display_df['Total_CF'] = display_df['Total_CF'].apply(lambda x: f"${x:,.2f}")
        display_df['CF_Percent'] = display_df['CF_Percent'].apply(lambda x: f"{x:.2f}%")
        st.dataframe(display_df, use_container_width=True, hide_index=True)

    with tab2:
        st.subheader("Historical Balance & Forward Projection")

        # Create projection based on current allocation
        projection_years = 5
        projection_days = np.arange(0, projection_years * 365 + 1)

        # Core balance projection
        core_projection = core_amount * np.array([survival_func(d) for d in projection_days])

        # Non-core assumed constant (or user can choose decay)
        non_core_projection = np.full(len(projection_days), non_core_amount)

        # Total projection
        total_projection = core_projection + non_core_projection

        # Convert days to actual dates from calc date
        from datetime import datetime, timedelta
        calc_date = datetime(2023, 12, 30)
        projection_dates = [calc_date + timedelta(days=int(d)) for d in projection_days]

        # Create combined plot
        fig_projection = go.Figure()

        # Historical balance
        fig_projection.add_trace(go.Scatter(
            x=df_nmd['Date'],
            y=df_nmd['Balance'],
            mode='lines',
            name='Historical Balance',
            line=dict(color='#1f77b4', width=2),
            fill='tozeroy',
            fillcolor='rgba(31, 119, 180, 0.2)'
        ))

        # Projected total balance
        fig_projection.add_trace(go.Scatter(
            x=projection_dates,
            y=total_projection,
            mode='lines',
            name='Total Projected',
            line=dict(color='#2ecc71', width=3, dash='solid')
        ))

        # Projected core balance
        fig_projection.add_trace(go.Scatter(
            x=projection_dates,
            y=core_projection,
            mode='lines',
            name='Core Projected',
            line=dict(color='#9b59b6', width=2, dash='dash')
        ))

        # Projected non-core balance
        fig_projection.add_trace(go.Scatter(
            x=projection_dates,
            y=non_core_projection,
            mode='lines',
            name='Non-Core (Stable)',
            line=dict(color='#e74c3c', width=2, dash='dot')
        ))

        # Add vertical line at calculation date using shape instead of vline
        fig_projection.add_shape(
            type="line",
            x0=calc_date,
            x1=calc_date,
            y0=0,
            y1=1,
            yref="paper",
            line=dict(color="gray", width=2, dash="dash")
        )

        fig_projection.add_annotation(
            x=calc_date,
            y=1,
            yref="paper",
            text="Calc Date (30-Dec-2023)",
            showarrow=False,
            yshift=10,
            font=dict(color="gray")
        )

        fig_projection.update_layout(
            title=f"Historical Balance (2016-2023) & Projected Balance (2024-2028) - {decay_model} Model",
            xaxis_title="Date",
            yaxis_title="Balance ($)",
            height=500,
            hovermode='x unified',
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

        # Decay breakdown
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Core Deposit Projection")
            st.write(f"**Model:** {decay_model}")
            st.write(f"**Initial Core:** ${core_amount:,.0f}")
            st.write(f"**Projected Core (5Y):** ${core_projection[-1]:,.0f}")
            core_decay_pct = ((core_projection[-1] - core_amount) / core_amount) * 100
            st.write(f"**Core Decay:** {core_decay_pct:+.1f}%")

        with col2:
            st.markdown("### Non-Core Deposit")
            st.write(f"**Method:** {non_core_method}")
            st.write(f"**Initial Non-Core:** ${non_core_amount:,.0f}")
            st.write(f"**Assumption:** Stable over horizon")
            st.info("💡 Non-core assumed stable (can be made dynamic)")

        st.markdown("---")

        # Show data table
        with st.expander("📊 View Historical Raw Data"):
            st.dataframe(df_nmd, use_container_width=True)

    with tab3:
        st.subheader(f"Decay Model: {decay_model}")

        col1, col2 = st.columns([2, 1])

        with col1:
            st.plotly_chart(plot_survival_curve(days_range, survival_values, decay_model),
                          use_container_width=True)

        with col2:
            st.markdown("### Model Parameters")
            for key, value in model_params.items():
                if isinstance(value, float):
                    st.write(f"**{key}:** {value:.4f}")
                else:
                    st.write(f"**{key}:** {value}")

            st.markdown("---")
            st.markdown("### Survival at Key Points")
            for years in [0.25, 0.5, 1, 2, 3, 5]:
                days = years * 365
                s_val = survival_func(days) * 100
                st.write(f"**{years}Y:** {s_val:.2f}%")

    with tab4:
        st.subheader("Cash Flow Slotting Details")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Core Deposits")
            st.write(f"**Amount:** ${core_amount:,.2f}")
            st.write(f"**Percentage:** {core_pct:.1f}%")
            st.write(f"**Model:** {decay_model}")

        with col2:
            st.markdown("### Non-Core Deposits")
            st.write(f"**Amount:** ${non_core_amount:,.2f}")
            st.write(f"**Percentage:** {100-core_pct:.1f}%")
            st.write(f"**Method:** {non_core_method}")

        st.markdown("---")

        # Waterfall chart
        fig_waterfall = go.Figure(go.Waterfall(
            name="Cash Flow",
            orientation="v",
            measure=["relative"] * len(slotting_df),
            x=slotting_df['Time Bucket'],
            y=slotting_df['Total_CF'],
            text=[f"${val:,.0f}" for val in slotting_df['Total_CF']],
            textposition="outside",
            connector={"line": {"color": "rgb(63, 63, 63)"}},
        ))

        fig_waterfall.update_layout(
            title="Cash Flow Waterfall by Bucket",
            height=400
        )

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
                st.error(f"⚠️ High EVE Risk: {delta_eve_pct:+.2f}%")
            elif abs(delta_eve_pct) > 10:
                st.warning(f"⚠️ Moderate EVE Risk: {delta_eve_pct:+.2f}%")
            else:
                st.success(f"✅ Low EVE Risk: {delta_eve_pct:+.2f}%")

        st.markdown("---")

        # Scenario comparison
        st.subheader("All Scenarios Comparison")

        scenarios = ["+200bps Parallel", "-200bps Parallel", "Short Rate Up", "Flattener"]
        scenario_results = []

        for scen in scenarios:
            shock_interp, _ = apply_rate_shock(curve_df, scen)
            eve_s = calculate_eve(cash_flows, midpoints, shock_interp)
            nii_s = calculate_nii(cash_flows, midpoints, shock_interp)

            scenario_results.append({
                'Scenario': scen,
                'EVE': eve_s,
                'ΔEVE': eve_s - eve_base,
                'ΔEVE %': ((eve_s - eve_base) / eve_base) * 100,
                'NII': nii_s,
                'ΔNII': nii_s - nii_base,
                'ΔNII %': ((nii_s - nii_base) / nii_base) * 100 if nii_base != 0 else 0
            })

        comparison_df = pd.DataFrame(scenario_results)

        # Highlight worst case
        worst_eve_idx = comparison_df['ΔEVE %'].abs().idxmax()

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

        st.info(f"**Worst Case Scenario:** {comparison_df.loc[worst_eve_idx, 'Scenario']} with ΔEVE of {comparison_df.loc[worst_eve_idx, 'ΔEVE %']:+.2f}%")

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: gray;'>
        <p>QF609 Group Project #1 - Advanced IRRBB for Non-Maturity Deposits</p>
        <p>SMU Bank | Calculation Date: 31-Dec-2023 | Model: Basel Standardized Approach</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
