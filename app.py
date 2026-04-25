import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import os

# ================== Page Configuration ==================
st.set_page_config(page_title="QQQ Risk Predictor", layout="wide", page_icon="📈")

# ================== Custom CSS ==================
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 0rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #4B5563;
        text-align: center;
        margin-bottom: 2rem;
    }
    .card {
        background-color: #F9FAFB;
        border-radius: 1rem;
        padding: 1.5rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: white;
        border-radius: 0.75rem;
        padding: 0.75rem;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# ================== Title ==================
st.markdown('<div class="main-header">📊 QQQ Weekly Large Drawdown Risk Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Logistic Regression Model | 20% Drawdown Threshold | Cost‑Sensitive Threshold (C=0.01)</div>', unsafe_allow_html=True)

# ================== Load Model ==================
@st.cache_resource
def load_model():
    model = joblib.load('final_model.pkl')
    features = joblib.load('features.pkl')
    return model, features

@st.cache_data
def load_base_features():
    base = joblib.load('base_features.pkl')
    return base

try:
    model, features = load_model()
    base = load_base_features()
except FileNotFoundError as e:
    st.error(f"❌ Model files not found: {e}\nPlease ensure final_model.pkl, features.pkl, base_features.pkl are in the current directory.")
    st.stop()

base_vals = base.iloc[0].to_dict()

# ================== Sidebar ==================
with st.sidebar:
    st.markdown("## 📖 Model Overview")
    st.markdown("""
    - **Training period**: 2023-01-01 – 2025-12-31 (157 weeks)
    - **Features**: 11 (technical, sentiment, macro, interactions)
    - **Regularization**: L2 (C=0.01)
    - **Threshold**: Cost‑based (FN=10, FP=1) → avg. **0.47**
    - **Performance**: Recall **86.7%** | AUC **0.625**
    """)
    st.markdown("---")
    st.caption("Adjust the sliders below to see how the risk probability changes.")

# ================== Main Layout ==================
st.markdown("## 🔧 Adjust Market Features (based on a typical historical week)")
col_left, col_right = st.columns([1.2, 1], gap="medium")

with col_left:
    with st.container():
        st.markdown("### 📉 Technical Indicators")
        ma_bias = st.slider("MA_Bias (Price vs 20‑week MA)", -0.3, 0.3, base_vals['MA_Bias'], 0.01)
        atr = st.slider("ATR (Average True Range)", 0.0, 50.0, base_vals['ATR'], 0.5)
        rsi = st.slider("RSI (Relative Strength Index)", 0, 100, int(base_vals['RSI']), 1)
        vol_change = st.slider("Volume Change", -0.5, 1.0, base_vals['Volume_Change'], 0.05)

    with st.container():
        st.markdown("### 😊 Sentiment Indicators")
        sent_level = st.slider("Sentiment Level", 0.0, 1.0, base_vals['Sentiment_Level'], 0.01)
        sent_unc = st.slider("Sentiment Uncertainty", 0.0, 0.3, base_vals['Sentiment_Uncertainty'], 0.01)
        vol_spike = st.slider("Volume Spike (Post count ratio)", 0.5, 3.0, base_vals['Volume_Spike'], 0.05)

    with st.container():
        st.markdown("### 🏦 Macro Indicators")
        vix_trend = st.slider("VIX Trend", -2.0, 30.0, base_vals['VIX_Trend'], 0.5)
        yield_spread = st.slider("Yield Spread (10Y-2Y)", -1.0, 1.0, base_vals['yield_spread'], 0.05)

    with st.container():
        st.markdown("### ⚡ Interaction Features")
        risk_res = st.selectbox("Risk Resonance", [0, 1], index=int(base_vals['Risk_Resonance']))
        price_sent_div = st.selectbox("Price‑Sentiment Divergence", [0, 1], index=int(base_vals['Price_Sentiment_Divergence']))

# ================== Prediction ==================
X_new = pd.DataFrame([[
    ma_bias, atr, rsi, vol_change,
    sent_level, sent_unc, vol_spike,
    vix_trend, yield_spread,
    risk_res, price_sent_div
]], columns=features)

prob = model.predict_proba(X_new)[0][1]
threshold = 0.47
risk_text = "High Risk (Recommended to hedge)" if prob >= threshold else "Low Risk (Normal position)"
risk_color = "#D32F2F" if prob >= threshold else "#2E7D32"

with col_right:
    st.markdown("## 📈 Prediction Result")
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob*100,
        title={'text': "Risk Probability (%)", 'font': {'size': 20}},
        domain={'x': [0, 1], 'y': [0, 1]},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1},
            'bar': {'color': risk_color, 'thickness': 0.8},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, threshold*100], 'color': "#C8E6C9"},
                {'range': [threshold*100, 100], 'color': "#FFCDD2"}
            ],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': threshold*100
            }
        }
    ))
    fig.update_layout(height=380, margin=dict(t=50, b=20, l=20, r=20), paper_bgcolor="#F9FAFB", font=dict(color="black"))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(f"<h2 style='text-align: center; color: {risk_color};'>{risk_text}</h2>", unsafe_allow_html=True)
    st.markdown(f"<p style='text-align: center; font-size: 0.9rem;'>Decision threshold = {threshold:.0%} (cost‑based optimization)</p>", unsafe_allow_html=True)

# ================== Model Performance Summary ==================
st.markdown("---")
st.markdown("## 📋 Model Performance Summary")
col_met1, col_met2, col_met3 = st.columns(3)
with col_met1:
    st.markdown('<div class="metric-card"><h3>Recall</h3><p style="font-size:2rem; font-weight:bold;">86.7%</p><p style="color:#4B5563;">Capture of actual large drawdowns</p></div>', unsafe_allow_html=True)
with col_met2:
    st.markdown('<div class="metric-card"><h3>AUC</h3><p style="font-size:2rem; font-weight:bold;">0.625</p><p style="color:#4B5563;">Ranking ability</p></div>', unsafe_allow_html=True)
with col_met3:
    st.markdown('<div class="metric-card"><h3>Optimal Threshold</h3><p style="font-size:2rem; font-weight:bold;">0.47</p><p style="color:#4B5563;">Average across rolling windows</p></div>', unsafe_allow_html=True)

# ================== Investment Performance (Backtest) ==================
st.markdown("---")
st.markdown("## 💼 Investment Performance with Model Signals")
st.markdown("Based on a 57‑week out‑of‑sample backtest (2025–2026)")

# Try to show the cumulative return image if exists
image_path = "backtest_cumulative_return.png"
if os.path.exists(image_path):
    col_ret, col_curve = st.columns([1, 2])
    with col_ret:
        st.markdown("### Key Metrics")
        metrics = {
            "Annualised Return": ("15.11%", "5.73%"),
            "Maximum Drawdown": ("-21.46%", "-2.27%"),
            "Sharpe Ratio": ("0.73", "1.00"),
            "Calmar Ratio": ("0.70", "2.53"),
            "Win Rate": ("52.6%", "12.3%"),
        }
        for metric, (bench, strat) in metrics.items():
            st.markdown(f"**{metric}**  ")
            st.markdown(f"📈 Buy & Hold: {bench}  ")
            st.markdown(f"🛡️ Strategy: **{strat}**  ")
            st.markdown("---")
    with col_curve:
        st.markdown("### Cumulative Return Curves")
        st.image(image_path, caption="Blue: Buy & Hold | Orange: Model‑Based Strategy", use_column_width=True)
else:
    # Without image, just show key metrics in columns
    st.markdown("#### Key Performance Indicators (Backtest)")
    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric("Max Drawdown", "-2.27%", delta="-19.19%", delta_color="inverse")
    col_b.metric("Sharpe Ratio", "1.00", delta="+0.28")
    col_c.metric("Calmar Ratio", "2.53", delta="+1.82")
    col_d.metric("Annualised Return", "5.73%", delta="-9.38%")
    st.caption("Compared to Buy & Hold (QQQ). The model successfully avoided the sharp drawdown in April 2025.")

st.success("✅ The model‑based strategy avoids large drawdowns (e.g., April 2025) and delivers a **substantial improvement in risk‑adjusted returns** (Sharpe 0.73 → 1.00, Calmar 0.70 → 2.53).")

# ================== Footer ==================
st.markdown("---")
st.caption("⚠️ This tool is for demonstration purposes only. Always perform your own due diligence before making investment decisions.")
