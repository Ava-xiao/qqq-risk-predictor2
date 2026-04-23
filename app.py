import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go

# 页面配置
st.set_page_config(page_title="QQQ Risk Predictor", layout="wide")
st.title("📊 QQQ 下周大回撤风险预测")
st.markdown("**逻辑回归模型 | 20% 回撤阈值 | 成本法阈值优化 (C=0.01)**")

# ========== 加载模型和特征 ==========
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
    st.error(f"❌ 模型文件未找到: {e}\n请确保 final_model.pkl, features.pkl, base_features.pkl 在当前目录")
    st.stop()

base_vals = base.iloc[0].to_dict()

# ========== 侧边栏：模型说明 ==========
st.sidebar.header("📖 模型说明")
st.sidebar.markdown("""
- **训练数据**: 2023-2026 周度数据 (157周)
- **特征数**: 11个（技术+情绪+宏观+交互）
- **正则化**: L2, C=0.01
- **阈值**: 成本法 (FN=10, FP=1) 平均阈值 0.47
- **性能**: 召回率 86.7% | AUC 0.625
""")

# ========== 主区域：特征滑块 ==========
st.subheader("🔧 调整市场特征 (基于历史典型周)")
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("### 📉 技术指标")
    ma_bias = st.slider("MA_Bias (价格偏离20周均线)", -0.3, 0.3, base_vals['MA_Bias'], 0.01)
    atr = st.slider("ATR (平均真实波幅)", 0.0, 50.0, base_vals['ATR'], 0.5)
    rsi = st.slider("RSI (相对强弱指数)", 0, 100, int(base_vals['RSI']), 1)
    vol_change = st.slider("Volume_Change (成交量变化率)", -0.5, 1.0, base_vals['Volume_Change'], 0.05)

    st.markdown("### 😊 情绪指标")
    sent_level = st.slider("Sentiment_Level (情绪分数)", 0.0, 1.0, base_vals['Sentiment_Level'], 0.01)
    sent_unc = st.slider("Sentiment_Uncertainty (情绪不确定性)", 0.0, 0.3, base_vals['Sentiment_Uncertainty'], 0.01)
    vol_spike = st.slider("Volume_Spike (讨论量暴增)", 0.5, 3.0, base_vals['Volume_Spike'], 0.05)

    st.markdown("### 🏦 宏观指标")
    vix_trend = st.slider("VIX_Trend (VIX趋势)", -2.0, 30.0, base_vals['VIX_Trend'], 0.5)
    yield_spread = st.slider("Yield Spread (10Y-2Y)", -1.0, 1.0, base_vals['yield_spread'], 0.05)

    st.markdown("### ⚡ 交互特征")
    risk_res = st.selectbox("Risk_Resonance (风险共振)", [0, 1], index=int(base_vals['Risk_Resonance']))
    price_sent_div = st.selectbox("Price_Sentiment_Divergence (价格-情绪背离)", [0, 1], index=int(base_vals['Price_Sentiment_Divergence']))

# 构造特征向量（顺序必须与训练时一致）
X_new = pd.DataFrame([[
    ma_bias, atr, rsi, vol_change,
    sent_level, sent_unc, vol_spike,
    vix_trend, yield_spread,
    risk_res, price_sent_div
]], columns=features)

prob = model.predict_proba(X_new)[0][1]
threshold = 0.47   # 成本法平均阈值
pred = "高风险 (建议避险)" if prob >= threshold else "低风险 (正常持仓)"

with col2:
    st.subheader("📈 预测结果")
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob*100,
        title={'text': "风险概率 (%)"},
        domain={'x': [0,1], 'y': [0,1]},
        gauge={
            'axis': {'range': [0,100]},
            'bar': {'color': "darkred" if prob>=threshold else "steelblue"},
            'steps': [
                {'range': [0, threshold*100], 'color': "lightgreen"},
                {'range': [threshold*100, 100], 'color': "lightcoral"}
            ],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': threshold*100
            }
        }
    ))
    fig.update_layout(height=350, margin=dict(t=30,b=20,l=20,r=20))
    st.plotly_chart(fig, use_container_width=True)
    st.markdown(f"<h3 style='color:{'red' if pred.startswith('高') else 'green'};'>{pred}</h3>", unsafe_allow_html=True)
    st.caption(f"决策阈值: {threshold:.0%} (成本法优化)")

# ========== 特征重要性图 ==========
st.subheader("📊 特征重要性 (逻辑回归系数)")
st.image('feature_importance_color.png', use_column_width=True)

# ========== 成本-阈值曲线示意图 ==========
st.subheader("💰 成本法阈值优化")
st.image('cost_curve_example.png', caption="成本-阈值曲线示意图（基于验证集）", use_column_width=True)

# ========== 模型性能摘要 ==========
st.subheader("📋 模型性能摘要")
col3, col4, col5 = st.columns(3)
col3.metric("召回率 (Recall)", "86.7%")
col4.metric("AUC", "0.625")
col5.metric("最优阈值", "0.47")