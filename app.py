import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# 從 model.py 載入先前寫好的資產配置模型類別
from model import DualRegimeAllocationModel

st.set_page_config(page_title="雙重狀態資產配置模型", layout="wide")

st.title("📈 雙重狀態資產配置模型 (Dual-Regime Asset Allocation)")
st.caption("基於 Luo & Mulvey (2026) 論文實作之 SJM + XGBoost 雙重市場狀態動態資產配置框架")

# ==========================================
# 側邊欄：模型參數設定
# ==========================================
st.sidebar.header("⚙️ 模型參數設定")

jp_global = st.sidebar.slider("全域 Jump Penalty (λ_global)", 1.0, 50.0, 15.0, step=1.0)
jp_asset = st.sidebar.slider("資產 Jump Penalty (λ_asset)", 1.0, 50.0, 15.0, step=1.0)
prob_thresh = st.sidebar.slider("XGBoost 分類機率門檻", 0.5, 0.9, 0.7, step=0.05)
ewm_win = st.sidebar.slider("EWM 機率平滑視窗 (天)", 10, 126, 63, step=1)
tc_bps = st.sidebar.number_input("單邊交易成本 (bps)", min_value=0, max_value=100, value=10)

run_button = st.sidebar.button("🚀 執行模型回測")

# ==========================================
# 資料準備與模型執行
# ==========================================
@st.cache_data
def generate_mock_data():
    """生成模擬市場數據 (亦可改為載入真實 CSV/YFinance 資料)"""
    np.random.seed(2026)
    dates = pd.date_range('2020-01-01', '2025-12-31', freq='B')
    asset_names = ['LargeCap', 'MidCap', 'SmallCap', 'EAFE', 'Treasury', 
                   'Corporate', 'HighYield', 'REIT', 'Commodity', 'Gold', 'RiskFree']
    
    returns_data = np.random.normal(0.0003, 0.01, size=(len(dates), len(asset_names)))
    returns_df = pd.DataFrame(returns_data, index=dates, columns=asset_names)
    returns_df['RiskFree'] = 0.0001
    
    macro_data = np.random.normal(0, 1, size=(len(dates), 3))
    macro_df = pd.DataFrame(macro_data, index=dates, columns=['VIX', 'Yield_Curve', 'Inflation'])
    
    return returns_df, macro_df

returns_df, macro_df = generate_mock_data()

if run_button or 'results' not in st.session_state:
    with st.spinner("模型運算中 (SJM 識別 + XGBoost 預測)..."):
        model = DualRegimeAllocationModel(
            jump_penalty_global=jp_global,
            jump_penalty_asset=jp_asset,
            prob_threshold=prob_thresh,
            ewm_window=ewm_win
        )
        res_df, bmda_hist, bmga_hist = model.run_pipeline(returns_df, macro_df)
        st.session_state['results'] = res_df
        st.session_state['bmda'] = bmda_hist
        st.session_state['bmga'] = bmga_hist

res_df = st.session_state['results']

# ==========================================
# 績效指標與圖表展示
# ==========================================
cum_returns = (1 + res_df['Return']).cumprod()
sharpe = (res_df['Return'].mean() * 252) / (res_df['Return'].std() * np.sqrt(252))
max_dd = (cum_returns / cum_returns.cummax() - 1).min()
annual_ret = (cum_returns.iloc[-1] ** (252 / len(res_df))) - 1
avg_turnover = res_df['Turnover'].mean()

# 頂部 Key Metrics
col1, col2, col3, col4 = st.columns(4)
col1.metric("年化夏普值 (Sharpe Ratio)", f"{sharpe:.2f}")
col2.metric("最大回撤 (Max Drawdown)", f"{max_dd * 100:.2f}%")
col3.metric("年化報酬率 (Annualized Return)", f"{annual_ret * 100:.2f}%")
col4.metric("平均每日換手率 (Turnover)", f"{avg_turnover * 100:.2f}%")

st.markdown("---")

# 淨值曲線圖
st.subheader("📊 策略累積淨值曲線 (Cumulative Wealth Curve)")
fig_wealth = px.line(cum_returns, labels={"value": "累積淨值", "index": "日期"})
fig_wealth.update_layout(showlegend=False, height=450)
st.plotly_chart(fig_wealth, use_container_width=True)

# 動態資產池展示
col_a, col_b = st.columns(2)
with col_a:
    st.subheader("🛡️ 熊市防禦資產池 (BMDA)")
    latest_date = list(st.session_state['bmda'].keys())[-1]
    st.write(f"最新日期 ({latest_date.strftime('%Y-%m-%d')}) 選取資產：")
    st.success(", ".join(st.session_state['bmda'][latest_date]))

with col_b:
    st.subheader("🚀 牛市成長資產池 (BMGA)")
    st.write(f"最新日期 ({latest_date.strftime('%Y-%m-%d')}) 選取資產：")
    st.info(", ".join(st.session_state['bmga'][latest_date]))
