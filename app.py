import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBClassifier
    USE_XGB = True
except ImportError:
    from sklearn.ensemble import RandomForestClassifier as XGBClassifier
    USE_XGB = False

# ==========================================
# 1. 核心模型模組
# ==========================================
class StatisticalJumpModel:
    def __init__(self, n_clusters=2, jump_penalty=10.0):
        self.n_clusters = n_clusters
        self.jump_penalty = float(jump_penalty)
        self.centroids = None

    def fit_predict(self, X):
        X = np.asarray(X, dtype=np.float64)
        T, d = X.shape
        K = self.n_clusters
        
        np.random.seed(42)
        idx = np.random.choice(T, K, replace=False)
        centroids = X[idx].copy()
        prev_states = np.zeros(T, dtype=int)
        
        for iteration in range(20):
            emission_cost = cdist(X, centroids, metric='sqeuclidean')
            dp = np.zeros((T, K))
            pointers = np.zeros((T, K), dtype=int)
            dp[0] = emission_cost[0]
            
            for t in range(1, T):
                for k in range(K):
                    transition_costs = dp[t-1] + self.jump_penalty * (np.arange(K) != k)
                    best_prev = np.argmin(transition_costs)
                    dp[t, k] = transition_costs[best_prev] + emission_cost[t, k]
                    pointers[t, k] = best_prev
            
            states = np.zeros(T, dtype=int)
            states[-1] = np.argmin(dp[-1])
            for t in range(T - 2, -1, -1):
                states[t] = pointers[t + 1, states[t + 1]]
                
            if np.array_equal(states, prev_states):
                break
            prev_states = states.copy()
            
            for k in range(K):
                if np.sum(states == k) > 0:
                    centroids[k] = np.mean(X[states == k], axis=0)
                    
        self.centroids = centroids
        return states


def extract_minimalist_features(returns_df, riskfree_col='RiskFree'):
    # 排除無風險利率，只對有波動的風險資產計算特徵
    risky_cols = [c for c in returns_df.columns if c != riskfree_col]
    features = {}
    for col in risky_cols:
        r = returns_df[col]
        for hl in [1, 5, 10, 21]:
            alpha = 1 - np.exp(-np.log(2) / hl)
            mean_r = r.ewm(alpha=alpha).mean()
            downside_r = r.clip(upper=0)
            downside_std = np.sqrt((downside_r**2).ewm(alpha=alpha).mean())
            sortino = mean_r / (downside_std + 1e-6)
            
            features[f'{col}_mean_{hl}'] = mean_r
            features[f'{col}_downside_{hl}'] = downside_std
            features[f'{col}_sortino_{hl}'] = sortino
            
    df_feat = pd.DataFrame(features, index=returns_df.index).dropna()
    
    # 過濾掉變異數為 0 的欄位，避免標準化報錯
    df_feat = df_feat.loc[:, df_feat.var() > 1e-8]
    
    scaler = StandardScaler()
    scaled_vals = scaler.fit_transform(df_feat.values)
    return pd.DataFrame(scaled_vals, index=df_feat.index, columns=df_feat.columns)


class DualRegimeAllocationModel:
    def __init__(self, jump_penalty_global=20.0, jump_penalty_asset=20.0, prob_threshold=0.7, ewm_window=63):
        self.jp_global = float(jump_penalty_global)
        self.jp_asset = float(jump_penalty_asset)
        self.threshold = float(prob_threshold)
        self.ewm_window = int(ewm_window)
        
    def _create_classifier(self):
        if USE_XGB:
            return XGBClassifier(n_estimators=100, max_depth=3, random_state=42, eval_metric='logloss')
        else:
            return XGBClassifier(n_estimators=100, max_depth=3, random_state=42)

    def _safe_fit_predict_proba(self, X_train, y_train, X_all):
        unique_classes = np.unique(y_train)
        if len(unique_classes) < 2:
            y_train = y_train.copy()
            mid = len(y_train) // 2
            y_train[:mid] = 0
            y_train[mid:] = 1
            
        clf = self._create_classifier()
        clf.fit(X_train, y_train)
        probs = clf.predict_proba(X_all)
        return probs[:, 1] if probs.shape[1] > 1 else np.full(len(X_all), 0.5)

    def run_pipeline(self, returns_df, macro_df, global_proxy_col='S&P500', riskfree_col='RiskFree'):
        X_min = extract_minimalist_features(returns_df, riskfree_col=riskfree_col)
        X_comp = pd.concat([X_min, macro_df], axis=1).reindex(X_min.index).ffill().bfill()
        
        common_idx = X_comp.index.intersection(returns_df.index)
        X_comp = X_comp.loc[common_idx]
        returns_df_aligned = returns_df.loc[common_idx]
        
        risky_assets = [c for c in returns_df_aligned.columns if c not in [global_proxy_col, riskfree_col]]
        dates = common_idx
        
        global_min_feat = [c for c in X_comp.columns if global_proxy_col in c]
        if not global_min_feat:
            global_min_feat = list(X_comp.columns[:3])
            
        sjm_global = StatisticalJumpModel(n_clusters=2, jump_penalty=self.jp_global)
        global_states = sjm_global.fit_predict(X_comp[global_min_feat].values)
        
        sp_series = returns_df_aligned.loc[dates, global_proxy_col]
        sp_sma = sp_series.rolling(window=50, min_periods=1).mean()
        trend_label = (sp_series > sp_sma).astype(int).values
        
        ret_g0 = sp_series[global_states == 0].mean()
        ret_g1 = sp_series[global_states == 1].mean()
        sjm_bull_state = 1 if ret_g1 > ret_g0 else 0
        sjm_label = (global_states == sjm_bull_state).astype(int)
        
        global_regimes_label = np.where((trend_label == 1) | (sjm_label == 1), 1, 0)
        if len(np.unique(global_regimes_label)) < 2:
            global_regimes_label = (sp_series > sp_series.median()).astype(int).values

        asset_regimes_label = {}
        for a in risky_assets:
            a_min_feat = [c for c in X_comp.columns if a in c]
            if not a_min_feat:
                continue
            sjm_asset = StatisticalJumpModel(n_clusters=2, jump_penalty=self.jp_asset)
            a_states = sjm_asset.fit_predict(X_comp[a_min_feat].values)
            a_series = returns_df_aligned.loc[dates, a]
            a_sma = a_series.rolling(window=50, min_periods=1).mean()
            a_trend = (a_series > a_sma).astype(int).values
            
            ret_a0 = a_series[a_states == 0].mean()
            ret_a1 = a_series[a_states == 1].mean()
            a_bull_state = 1 if ret_a1 > ret_a0 else 0
            a_sjm = (a_states == a_bull_state).astype(int)
            
            a_label = np.where((a_trend == 1) | (a_sjm == 1), 1, 0)
            if len(np.unique(a_label)) < 2:
                a_label = (a_series > a_series.median()).astype(int).values
            asset_regimes_label[a] = a_label
            
        asset_regimes_label_df = pd.DataFrame(asset_regimes_label, index=dates)

        y_global = pd.Series(global_regimes_label, index=dates).shift(-1).ffill().fillna(0)
        X_train_g = X_comp.iloc[:-1].values
        y_train_g = y_global.iloc[:-1].values.astype(int)
        
        prob_global_bull = self._safe_fit_predict_proba(X_train_g, y_train_g, X_comp.values)
        
        pred_global_bear = (prob_global_bull < (1.0 - self.threshold)).astype(int)
        pred_global_bull = (prob_global_bull > self.threshold).astype(int)
        
        alpha_ewm = 2.0 / (self.ewm_window + 1.0)
        smoothed_prob_asset_bull = {}
        
        for a in risky_assets:
            if a not in asset_regimes_label_df.columns:
                continue
            y_asset = asset_regimes_label_df[a].shift(-1).ffill().fillna(0)
            X_train_a = X_comp.iloc[:-1].values
            y_train_a = y_asset.iloc[:-1].values.astype(int)
            
            raw_prob_bull = self._safe_fit_predict_proba(X_train_a, y_train_a, X_comp.values)
            prob_series = pd.Series(raw_prob_bull, index=dates)
            smoothed_prob_asset_bull[a] = prob_series.ewm(alpha=alpha_ewm).mean()
            
        smoothed_prob_asset_df = pd.DataFrame(smoothed_prob_asset_bull)

        bmda_sets = {}
        bmga_sets = {}
        
        for t in range(len(dates)):
            d = dates[t]
            is_global_bear = pred_global_bear[t] == 1
            is_global_bull = pred_global_bull[t] == 1
            
            if d in smoothed_prob_asset_df.index:
                asset_bulls = smoothed_prob_asset_df.loc[d] > self.threshold
                selected_risky = asset_bulls[asset_bulls].index.tolist()
            else:
                selected_risky = []
            
            if is_global_bear:
                selected_bmda = selected_risky.copy()
                selected_bmda.append(riskfree_col)
                bmda_sets[d] = selected_bmda
            else:
                bmda_sets[d] = [riskfree_col]
                
            if is_global_bull:
                selected_bmga = selected_risky.copy()
                if global_proxy_col not in selected_bmga:
                    selected_bmga.append(global_proxy_col)
                bmga_sets[d] = selected_bmga
            else:
                bmga_sets[d] = [global_proxy_col]

        portfolio_returns = []
        tc_rate = 0.0010
        prev_weights = pd.Series(0.0, index=returns_df_aligned.columns)
        
        for t in range(len(dates) - 1):
            d_today = dates[t]
            d_next = dates[t+1]
            
            if pred_global_bear[t] == 1:
                target_assets = bmda_sets.get(d_today, [riskfree_col])
            else:
                target_assets = bmga_sets.get(d_today, [global_proxy_col])
                
            target_weights = pd.Series(0.0, index=returns_df_aligned.columns)
            target_weights[target_assets] = 1.0 / len(target_assets)
            
            turnover = np.sum(np.abs(target_weights - prev_weights))
            tc = turnover * tc_rate
            
            r_next = returns_df_aligned.loc[d_next]
            port_r = np.sum(target_weights * r_next) - tc
            portfolio_returns.append({
                'Date': d_next, 
                'Return': port_r, 
                'Turnover': turnover,
                'Global_Bull_Prob': prob_global_bull[t]
            })
            prev_weights = target_weights.copy()
            
        res_df = pd.DataFrame(portfolio_returns).set_index('Date')
        return res_df, bmda_sets, bmga_sets


# ==========================================
# 2. Streamlit 介面與前端展示
# ==========================================
st.set_page_config(page_title="雙重狀態資產配置模型", layout="wide")

st.title("📈 雙重狀態動態資產配置系統 (真實歷史數據回測)")
st.caption("基於 Luo & Mulvey (2026) 論文實作，結合 SJM 與 XGBoost 進行多空動態配置與牛熊市歷史視覺化。")

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

st.subheader("📈 累積淨值曲線對比")
comparison_df = pd.DataFrame({
    "雙重狀態動態配置策略": cum_returns,
    "S&P 500 (買入持有)": benchmark_cum
})
fig_wealth = px.line(comparison_df, labels={"value": "累積淨值", "index": "日期", "variable": "策略類型"})
fig_wealth.update_layout(height=450, template="plotly_dark", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
st.plotly_chart(fig_wealth, use_container_width=True)

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

st.header("📖 系統說明與真實資產清單 (README)")
st.markdown("歡迎使用 **雙重狀態動態資產配置系統**。以下為本系統的核心架構、參數設定與真實資產清單說明。")

st.info("`#Real-Data` `#Yahoo-Finance` `#Quantitative-Strategy` `#Asset-Allocation` `#Python` `#Streamlit`")

with st.expander("📌 1. 真實數據資產清單與代理代碼", expanded=True):
    st.markdown("""
    系統回測時從 Yahoo Finance 抓取的真實市場標的與代碼對應：
    - **S&P500** (`^GSPC`)：全域基準與大型股大盤代理
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
