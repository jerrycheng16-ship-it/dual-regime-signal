import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go

# 從 model.py 載入模型
from model import DualRegimeAllocationModel

st.set_page_config(page_title="雙重狀態資產配置模型 (真實數據版)", layout="wide")

st.title("📈 雙重狀態動態資產配置系統 (真實歷史數據回測)")
st.caption("基於 Luo & Mulvey (2026) 論文實作，串接 Yahoo Finance 真實資產與總經數據進行多空動態配置與效能評估。")

# ==========================================
# 側邊欄：模型與回測期間參數設定
# ==========================================
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

# ==========================================
# 資料準備：從 Yahoo Finance 抓取真實歷史數據
# ==========================================
@st.cache_data(ttl=86400)
def fetch_real_market_data():
    """透過 yfinance 抓取真實金融資產與總經指標歷史數據"""
    tickers_map = {
        'S&P500': '^GSPC',     # 大型股大盤代理
        'Nasdaq': '^IXIC',     # 科技成長股
        'Treasury': 'TLT',     # 美國 20 年期以上公債
        'Corporate': 'LQD',    # 投資級公司債
        'HighYield': 'HYG',    # 高收益債
        'Gold': 'GC=F',        # 黃金期貨
        'Commodity': 'DBC',    # 大宗商品
        'REIT': 'VNQ'          # 不動產信託
    }
    
    macro_tickers = {
        'VIX': '^VIX'          # 恐慌指數
    }

    all_tickers = list(tickers_map.values()) + list(macro_tickers.values())
    
    # 下載歷史資料
    df_raw = yf.download(all_tickers, start="2018-01-01", progress=False)
    
    # 相容 yfinance 多層欄位結構
    if isinstance(df_raw.columns, pd.MultiIndex):
        df_prices = df_raw['Adj Close'] if 'Adj Close' in df_raw.columns else df_raw['Close']
    else:
        df_prices = df_raw[['Close']]

    # 重新命名欄位回資產名稱
    inv_tickers_map = {v: k for k, v in tickers_map.items()}
    inv_macro_map = {v: k for k, v in macro_tickers.items()}
    
    rename_dict = {**inv_tickers_map, **inv_macro_map}
    df_prices = df_prices.rename(columns=rename_dict)
    
    # 計算日報酬率
    returns_df = df_prices[list(tickers_map.keys())].pct_change().dropna()
    returns_df['RiskFree'] = 0.0001  # 設定每日無風險利率約 2.5% 年化
    
    # 總經變數
    macro_df = df_prices[[k for k in macro_tickers.keys() if k in df_prices.columns]].dropna()
    # 補上簡單的殖利率曲線與通膨代理變數以符合模型維度
    macro_df['Yield_Curve'] = 0.5 
    macro_df['Inflation'] = 2.0
    
    return returns_df, macro_df

with st.spinner("正在透過 Yahoo Finance 同步真實全球金融市場與總經歷史數據..."):
    raw_returns_df, raw_macro_df = fetch_real_market_data()

# 根據使用者手動輸入的日期進行篩選
mask = (raw_returns_df.index >= pd.to_datetime(start_date)) & (raw_returns_df.index <= pd.to_datetime(end_date))
returns_df = raw_returns_df.loc[mask].dropna(how='all').copy()
macro_df = raw_macro_df.loc[mask].reindex(returns_df.index).ffill().bfill().copy()

if returns_df.empty or len(returns_df) < 50:
    st.error("❌ 選擇的時間區間內真實資料不足（或該區間為週末/假日），請調整回測起訖日期！")
    st.stop()

# 執行回測 Pipeline
if run_button or 'results' not in st.session_state:
    with st.spinner("真實數據模型運算中 (SJM 結構識別 + XGBoost 狀態預測)..."):
        model_instance = DualRegimeAllocationModel(
            jump_penalty_global=float(jp_global),
            jump_penalty_asset=float(jp_asset),
            prob_threshold=float(prob_thresh),
            ewm_window=int(ewm_win)
        )
        
        res_df, bmda_hist, bmga_hist = model_instance.run_pipeline(
            returns_df.copy(), 
            macro_df.copy(),
            global_proxy_col='S&P500',
            riskfree_col='RiskFree'
        )
        
        st.session_state['results'] = res_df
        st.session_state['bmda'] = bmda_hist
        st.session_state['bmga'] = bmga_hist

res_df = st.session_state['results']

# ==========================================
# 績效指標計算（策略 vs S&P 500 真實基準）
# ==========================================
cum_returns = (1 + res_df['Return']).cumprod()
sharpe = (res_df['Return'].mean() * 252) / (res_df['Return'].std() * np.sqrt(252)) if res_df['Return'].std() > 0 else 0
max_dd = (cum_returns / cum_returns.cummax() - 1).min()
annual_ret = (cum_returns.iloc[-1] ** (252 / len(res_df))) - 1 if len(res_df) > 0 else 0
avg_turnover = res_df['Turnover'].mean()

# S&P 500 (真實大盤基準)
benchmark_returns = returns_df.loc[res_df.index, 'S&P500']
benchmark_cum = (1 + benchmark_returns).cumprod()
benchmark_sharpe = (benchmark_returns.mean() * 252) / (benchmark_returns.std() * np.sqrt(252)) if benchmark_returns.std() > 0 else 0
benchmark_max_dd = (benchmark_cum / benchmark_cum.cummax() - 1).min()
benchmark_annual_ret = (benchmark_cum.iloc[-1] ** (252 / len(benchmark_cum))) - 1 if len(benchmark_cum) > 0 else 0

# 顯示核心績效對比
st.subheader("📊 真實數據績效比較：雙重狀態動態配置策略 vs S&P 500 (^GSPC)")
col1, col2, col3, col4 = st.columns(4)
col1.metric("年化夏普值 (Sharpe)", f"{sharpe:.2f}", f"大盤基準: {benchmark_sharpe:.2f}")
col2.metric("最大回撤 (Max Drawdown)", f"{max_dd * 100:.2f}%", f"大盤基準: {benchmark_max_dd * 100:.2f}%")
col3.metric("年化報酬率 (Annual Return)", f"{annual_ret * 100:.2f}%", f"大盤基準: {benchmark_annual_ret * 100:.2f}%")
col4.metric("平均每日換手率 (Turnover)", f"{avg_turnover * 100:.2f}%")

st.markdown("---")

# 累積淨值曲線對比圖
st.subheader("📈 真實歷史績效累積淨值曲線對比")
comparison_df = pd.DataFrame({
    "雙重狀態動態資產配置策略": cum_returns,
    "S&P 500 (真實買入持有 Benchmark)": benchmark_cum
})
fig_wealth = px.line(comparison_df, labels={"value": "累積淨值", "index": "日期", "variable": "策略類型"})
fig_wealth.update_layout(height=450, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
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

st.markdown("---")

# ==========================================
# 說明文件與真實資產清單 (README)
# ==========================================
st.header("📖 系統說明與真實資產清單 (README)")
st.markdown("本系統採用 **Yahoo Finance 真實歷史數據**（含 S&P 500、Nasdaq、TLT 長債、黃金等）進行回測。")

st.info("`#Real-Data` `#Yahoo-Finance` `#Quantitative-Strategy` `#Asset-Allocation` `#Python` `#Streamlit`")

with st.expander("📌 1. 真實數據資產清單與代理代碼", expanded=True):
    st.markdown("""
    系統回測時抓取的真實市場標的與代碼對應：
    - **S&P500** (`^GSPC`)：全域基準與大型股代理
    - **Nasdaq** (`^IXIC`)：科技成長股代理
    - **Treasury** (`TLT`)：美國 20 年期以上公債（防禦核心）
    - **Corporate** (`LQD`)：投資級公司債
    - **HighYield** (`HYG`)：高收益債
    - **Gold** (`GC=F`)：黃金期貨
    - **Commodity** (`DBC`)：大宗商品指數
    - **REIT** (`VNQ`)：不動產信託
    - **RiskFree**：固定無風險利率代理
    """)

with st.expander("🛠️ 2. 模型核心運作機制", expanded=False):
    st.markdown("""
    - **Statistical Jump Model (SJM)**：動態捕捉真實市場從多頭轉為空頭的結構性跳躍點。
    - **機器學習分類器**：預測未來市場狀態機率，並動態將資產分配至 **BMDA（防禦資產池）** 或 **BMGA（成長資產池）**。
    """)
