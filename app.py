import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from model import DualRegimeAllocationModel

st.set_page_config(page_title="雙重狀態資產配置模型", layout="wide")

st.title("📈 雙重狀態動態資產配置系統 (真實歷史數據回測)")
st.caption("基於 Luo & Mulvey (2026) 論文實作，結合 SJM 與 XGBoost 進行多空動態配置與牛熊市歷史視覺化。")

# 側邊欄設定
st.sidebar.header("📅 回測時間區間設定")
default_start = pd.to_datetime("2020-01-01")
default_end = pd.to_datetime("2025-12-31")

start_date = st.sidebar.date_input("回測開始日期", default_start)
end_date = st.sidebar.date_input("回測結束日期", default_end)

st.sidebar.header("⚙️ 模型參數設定")
jp_global = st.sidebar.slider("全域 Jump Penalty (λ_global)", 1.0, 50.0, 15.0, step=1.0)
jp_asset = st.sidebar.slider("資產 Jump Penalty (λ_asset)", 1.0, 50.0, 15.0, step=1.0)
prob_thresh = st.sidebar.slider("分類機率門檻", 0.5, 0.9, 0.7, step=0.05)
ewm_win = st.sidebar.slider("EWM 機率平滑視窗 (天)", 10, 126, 63, step=1)

run_button = st.sidebar.button("🚀 執行真實數據回測")

@st.cache_data(ttl=86400)
def fetch_real_market_data():
    tickers_map = {
        'S&P500': '^GSPC',
        'Nasdaq': '^IXIC',
        'Treasury': 'TLT',
        'Corporate': 'LQD',
        'HighYield': 'HYG',
        'Gold': 'GC=F',
        'Commodity': 'DBC',
        'REIT': 'VNQ'
    }
    macro_tickers = {'VIX': '^VIX'}
    all_tickers = list(tickers_map.values()) + list(macro_tickers.values())
    
    df_raw = yf.download(all_tickers, start="2018-01-01", progress=False)
    if isinstance(df_raw.columns, pd.MultiIndex):
        df_prices = df_raw['Adj Close'] if 'Adj Close' in df_raw.columns else df_raw['Close']
    else:
        df_prices = df_raw[['Close']]

    inv_tickers_map = {v: k for k, v in tickers_map.items()}
    inv_macro_map = {v: k for k, v in macro_tickers.items()}
    df_prices = df_prices.rename(columns={**inv_tickers_map, **inv_macro_map})
    
    valid_cols = [c for c in tickers_map.keys() if c in df_prices.columns]
    returns_df = df_prices[valid_cols].pct_change().dropna(how='all')
    returns_df['RiskFree'] = 0.0001
    
    macro_cols = [c for c in macro_tickers.keys() if c in df_prices.columns]
    macro_df = df_prices[macro_cols].reindex(returns_df.index).ffill().bfill()
    macro_df['Yield_Curve'] = 0.5 
    macro_df['Inflation'] = 2.0
    
    return returns_df.dropna(), macro_df.dropna()

with st.spinner("正在同步真實金融市場與總經歷史數據..."):
    raw_returns_df, raw_macro_df = fetch_real_market_data()

start_ts = pd.to_datetime(start_date)
end_ts = pd.to_datetime(end_date)
common_index = raw_returns_df.index.intersection(raw_macro_df.index)
clean_returns = raw_returns_df.loc[common_index]
clean_macro = raw_macro_df.loc[common_index]

mask = (clean_returns.index >= start_ts) & (clean_returns.index <= end_ts)
returns_df = clean_returns.loc[mask].copy()
macro_df = clean_macro.loc[mask].copy()

if returns_df.empty or len(returns_df) < 30:
    st.error("❌ 選擇的時間區間內真實資料不足，請擴大回測起訖日期！")
    st.stop()

if run_button or 'results' not in st.session_state:
    with st.spinner("模型運算中 (SJM + XGBoost)..."):
        model_instance = DualRegimeAllocationModel(
            jump_penalty_global=float(jp_global),
            jump_penalty_asset=float(jp_asset),
            prob_threshold=float(prob_thresh),
            ewm_window=int(ewm_win)
        )
        res_df, bmda_hist, bmga_hist = model_instance.run_pipeline(
            returns_df.copy(), macro_df.copy(), global_proxy_col='S&P500', riskfree_col='RiskFree'
        )
        st.session_state['results'] = res_df
        st.session_state['bmda'] = bmda_hist
        st.session_state['bmga'] = bmga_hist

res_df = st.session_state['results']

# 績效計算
cum_returns = (1 + res_df['Return']).cumprod()
sharpe = (res_df['Return'].mean() * 252) / (res_df['Return'].std() * np.sqrt(252)) if res_df['Return'].std() > 0 else 0
max_dd = (cum_returns / cum_returns.cummax() - 1).min()
annual_ret = (cum_returns.iloc[-1] ** (252 / len(res_df))) - 1 if len(res_df) > 0 else 0
avg_turnover = res_df['Turnover'].mean()

benchmark_returns = returns_df.loc[res_df.index, 'S&P500']
benchmark_cum = (1 + benchmark_returns).cumprod()
benchmark_sharpe = (benchmark_returns.mean() * 252) / (benchmark_returns.std() * np.sqrt(252)) if benchmark_returns.std() > 0 else 0
benchmark_max_dd = (benchmark_cum / benchmark_cum.cummax() - 1).min()
benchmark_annual_ret = (benchmark_cum.iloc[-1] ** (252 / len(benchmark_cum))) - 1 if len(benchmark_cum) > 0 else 0

st.subheader("📊 核心績效指標比較：雙重狀態策略 vs S&P 500")
col1, col2, col3, col4 = st.columns(4)
col1.metric("年化夏普值", f"{sharpe:.2f}", f"大盤基準: {benchmark_sharpe:.2f}")
col2.metric("最大回撤", f"{max_dd * 100:.2f}%", f"大盤基準: {benchmark_max_dd * 100:.2f}%")
col3.metric("年化報酬率", f"{annual_ret * 100:.2f}%", f"大盤基準: {benchmark_annual_ret * 100:.2f}%")
col4.metric("平均換手率", f"{avg_turnover * 100:.2f}%")

st.markdown("---")

# 淨值曲線對比圖
st.subheader("📈 累積淨值曲線對比")
comparison_df = pd.DataFrame({
    "雙重狀態動態配置策略": cum_returns,
    "S&P 500 (買入持有)": benchmark_cum
})
fig_wealth = px.line(comparison_df, labels={"value": "累積淨值", "index": "日期", "variable": "策略類型"})
fig_wealth.update_layout(height=450, template="plotly_dark", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
st.plotly_chart(fig_wealth, use_container_width=True)

# 牛熊市歷史狀態判定圖
st.subheader("🌍 全球市場多空狀態歷史判定圖 (Regime Shifting)")
st.caption("上方為 S&P 500 走勢，下方為模型預測的每日牛市機率，助您一眼識別歷史多空轉折點。")

fig_regime = make_subplots(
    rows=2, cols=1, 
    shared_xaxes=True, 
    vertical_spacing=0.08,
    row_heights=[0.7, 0.3],
    subplot_titles=("S&P 500 歷史走勢", "模型預測牛市機率 (Prob Bull)")
)

sp_prices = (1 + benchmark_returns).cumprod()
fig_regime.add_trace(
    go.Scatter(x=sp_prices.index, y=sp_prices, name="S&P 500 走勢", line=dict(color='#1f77b4', width=2)),
    row=1, col=1
)

bull_probs = res_df['Global_Bull_Prob']
fig_regime.add_trace(
    go.Scatter(x=bull_probs.index, y=bull_probs, name="牛市機率", line=dict(color='#ff7f0e', width=1.5)),
    row=2, col=1
)
fig_regime.add_hline(y=prob_thresh, line_dash="dash", line_color="gray", row=2, col=1, annotation_text="門檻線")

fig_regime.update_layout(
    template="plotly_dark",
    height=550,
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)
st.plotly_chart(fig_regime, use_container_width=True)

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
