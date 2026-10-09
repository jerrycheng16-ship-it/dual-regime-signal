import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# 從 model.py 載入模型
from model import DualRegimeAllocationModel

st.set_page_config(page_title="雙重狀態資產配置模型", layout="wide")

st.title("📈 雙重狀態資產配置模型 (Dual-Regime Asset Allocation)")
st.caption("基於 Luo & Mulvey (2026) 論文實作之 SJM + XGBoost 雙重市場狀態動態資產配置框架（含 S&P 500 基準對比）")

# ==========================================
# 側邊欄：導覽分頁與模型參數設定
# ==========================================
st.sidebar.header("🧭 導覽選單")
nav_mode = st.sidebar.radio("選擇檢視模式", ["📊 模型回測與效能分析", "📖 系統說明與使用文件 (README)"])

if nav_mode == "📊 模型回測與效能分析":
    st.sidebar.markdown("---")
    st.sidebar.header("📅 回測時間區間設定")
    default_start = pd.to_datetime("2020-01-01")
    default_end = pd.to_datetime("2025-12-31")

    start_date = st.sidebar.date_input("回測開始日期", default_start)
    end_date = st.sidebar.date_input("回測結束日期", default_end)

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
        """生成模擬市場數據"""
        np.random.seed(2026)
        dates = pd.date_range('2015-01-01', '2026-12-31', freq='B')
        asset_names = ['LargeCap', 'MidCap', 'SmallCap', 'EAFE', 'Treasury', 
                       'Corporate', 'HighYield', 'REIT', 'Commodity', 'Gold', 'RiskFree']
        
        returns_data = np.random.normal(0.0003, 0.01, size=(len(dates), len(asset_names)))
        returns_df = pd.DataFrame(returns_data, index=dates, columns=asset_names)
        returns_df['RiskFree'] = 0.0001
        
        macro_data = np.random.normal(0, 1, size=(len(dates), 3))
        macro_df = pd.DataFrame(macro_data, index=dates, columns=['VIX', 'Yield_Curve', 'Inflation'])
        
        return returns_df, macro_df

    raw_returns_df, raw_macro_df = generate_mock_data()

    # 根據使用者手動輸入的日期進行篩選
    mask = (raw_returns_df.index >= pd.to_datetime(start_date)) & (raw_returns_df.index <= pd.to_datetime(end_date))
    returns_df = raw_returns_df.loc[mask].copy()
    macro_df = raw_macro_df.loc[mask].copy()

    if returns_df.empty:
        st.error("❌ 選擇的時間區間內沒有資料，請調整回測起訖日期！")
        st.stop()

    # 安全地執行 pipeline 避開型態衝突
    if run_button or 'results' not in st.session_state:
        with st.spinner("模型運算中 (SJM 識別 + XGBoost 預測)..."):
            model_instance = DualRegimeAllocationModel(
                jump_penalty_global=float(jp_global),
                jump_penalty_asset=float(jp_asset),
                prob_threshold=float(prob_thresh),
                ewm_window=int(ewm_win)
            )
            
            res_df, bmda_hist, bmga_hist = model_instance.run_pipeline(
                returns_df.copy(), 
                macro_df.copy()
            )
            
            st.session_state['results'] = res_df
            st.session_state['bmda'] = bmda_hist
            st.session_state['bmga'] = bmga_hist

    res_df = st.session_state['results']

    # ==========================================
    # 績效指標計算（策略 vs S&P 500 基準）
    # ==========================================
    cum_returns = (1 + res_df['Return']).cumprod()
    sharpe = (res_df['Return'].mean() * 252) / (res_df['Return'].std() * np.sqrt(252)) if res_df['Return'].std() > 0 else 0
    max_dd = (cum_returns / cum_returns.cummax() - 1).min()
    annual_ret = (cum_returns.iloc[-1] ** (252 / len(res_df))) - 1 if len(res_df) > 0 else 0
    avg_turnover = res_df['Turnover'].mean()

    # S&P 500 (LargeCap) 買入持有基準績效
    benchmark_returns = returns_df.loc[res_df.index, 'LargeCap']
    benchmark_cum = (1 + benchmark_returns).cumprod()
    benchmark_sharpe = (benchmark_returns.mean() * 252) / (benchmark_returns.std() * np.sqrt(252)) if benchmark_returns.std() > 0 else 0
    benchmark_max_dd = (benchmark_cum / benchmark_cum.cummax() - 1).min()
    benchmark_annual_ret = (benchmark_cum.iloc[-1] ** (252 / len(benchmark_cum))) - 1 if len(benchmark_cum) > 0 else 0

    # 顯示核心績效對比
    st.subheader("📊 核心績效指標比較：雙重狀態策略 vs S&P 500 (LargeCap)")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("年化夏普值 (Sharpe)", f"{sharpe:.2f}", f"基準: {benchmark_sharpe:.2f}")
    col2.metric("最大回撤 (Max Drawdown)", f"{max_dd * 100:.2f}%", f"基準: {benchmark_max_dd * 100:.2f}%")
    col3.metric("年化報酬率 (Annual Return)", f"{annual_ret * 100:.2f}%", f"基準: {benchmark_annual_ret * 100:.2f}%")
    col4.metric("平均每日換手率 (Turnover)", f"{avg_turnover * 100:.2f}%")

    st.markdown("---")

    # 累積淨值曲線對比圖
    st.subheader("📈 策略與 S&P 500 累積淨值曲線對比 (Cumulative Wealth Curve)")
    comparison_df = pd.DataFrame({
        "雙重狀態動態資產配置策略": cum_returns,
        "S&P 500 (買入持有 Benchmark)": benchmark_cum
    })
    fig_wealth = px.line(comparison_df, labels={"value": "累積淨值", "index": "日期", "variable": "策略類型"})
    fig_wealth.update_layout(height=450, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    st.plotly_chart(fig_wealth, use_container_width=True)

    # 資產池展示
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

else:
    # ==========================================
    # 說明文件與 Tag 區塊 (README Module)
    # ==========================================
    st.header("📖 系統說明與使用文件 (README)")
    st.markdown("歡迎使用 **雙重狀態動態資產配置系統**。本系統是基於學術論文實作的量化回測框架。")
    
    st.markdown("---")
    
    # 使用 Tag 區塊呈現說明文件標籤與核心摘要
    st.markdown("### 🏷️ 專案標籤與架構摘要")
    st.info("`#Quantitative-Finance` `#SJM-Model` `#XGBoost` `#Asset-Allocation` `#Python` `#Streamlit`")

    with st.expander("📌 1. 核心理論與設計理念", expanded=True):
        st.markdown("""
        - **基於論文實作**：本系統核心演算法改編自 **Luo & Mulvey (2026)** 的雙重狀態動態資產配置框架。
        - **Statistical Jump Model (SJM)**：利用動態規劃（Dynamic Programming）在歷史報酬率與總經序列中捕捉市場結構性的轉換點（Regime Shift）。
        - **機器學習預測**：結合 **XGBoost** 分類器與極簡技術指標/總經變數，預測未來市場狀態機率，並透過指數平滑（EWM）過濾雜訊。
        """)

    with st.expander("🛠️ 2. 側邊欄參數設定說明", expanded=False):
        st.markdown("""
        - **回測時間區間**：自由指定欲回測的起訖日期（例如 `2020-01-01` 至 `2025-12-31`）。
        - **全域 Jump Penalty ($\lambda_{global}$)**：控制全域市場狀態轉換的敏感度，數值越高狀態切換越平緩。
        - **資產 Jump Penalty ($\lambda_{asset}$)**：控制個別資產狀態切換的懲罰係數。
        - **XGBoost 機率門檻**：判定牛市或熊市觸發的機率分界線（預設 `0.7`）。
        - **EWM 平滑視窗**：對機率進行指數加權移動平均的天數（預設 `63` 天）。
        """)

    with st.expander("📊 3. 輸出指標與基準對比", expanded=False):
        st.markdown("""
        - **年化夏普值 (Sharpe Ratio)**：衡量風險調整後報酬率，並與 S&P 500 (LargeCap) 進行直接對比。
        - **最大回撤 (Max Drawdown)**：評估策略在空頭或震撼時期的防禦能力。
        - **動態資產池**：
          - **BMDA (熊市防禦資產池)**：自動納入防禦性資產與無風險利率。
          - **BMGA (牛市成長資產池)**：動態挑選具備上漲動能的風險資產。
        """)

    st.markdown("---")
    st.success("💡 **提示**：您可以隨時透過左側導覽選單切換回「模型回測與效能分析」介面進行互動模擬！")
