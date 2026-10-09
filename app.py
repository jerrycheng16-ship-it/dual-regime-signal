import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.spatial.distance import cdist

try:
    from xgboost import XGBClassifier
    USE_XGB = True
except ImportError:
    from sklearn.ensemble import RandomForestClassifier as XGBClassifier
    USE_XGB = False

# ==========================================
# 1. 核心模型模組 (原 model.py 內容)
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


def extract_minimalist_features(returns_df):
    features = {}
    for col in returns_df.columns:
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
            
    return pd.DataFrame(features, index=returns_df.index).dropna()


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
            y_train[0] = 1 if unique_classes[0] == 0 else 0
            
        clf = self._create_classifier()
        clf.fit(X_train, y_train)
        return clf.predict_proba(X_all)[:, 1]

    def run_pipeline(self, returns_df, macro_df, global_proxy_col='S&P500', riskfree_col='RiskFree'):
        X_min = extract_minimalist_features(returns_df)
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
        
        ret_g0 = returns_df_aligned.loc[dates, global_proxy_col][global_states == 0].mean()
        ret_g1 = returns_df_aligned.loc[dates, global_proxy_col][global_states == 1].mean()
        global_bull_state = 1 if ret_g1 > ret_g0 else 0
        global_regimes_label = (global_states == global_bull_state).astype(int)
        
        if len(np.unique(global_regimes_label)) < 2:
            global_regimes_label = (returns_df_aligned[global_proxy_col] > returns_df_aligned[global_proxy_col].median()).astype(int).values

        asset_regimes_label = {}
        for a in risky_assets:
            a_min_feat = [c for c in X_comp.columns if a in c]
            if not a_min_feat:
                continue
            sjm_asset = StatisticalJumpModel(n_clusters=2, jump_penalty=self.jp_asset)
            a_states = sjm_asset.fit_predict(X_comp[a_min_feat].values)
            
            ret_a0 = returns_df_aligned.loc[dates, a][a_states == 0].mean()
            ret_a1 = returns_df_aligned.loc[dates, a][a_states == 1].mean()
            a_bull_state = 1 if ret_a1 > ret_a0 else 0
            a_label = (a_states == a_bull_state).astype(int)
            if len(np.unique(a_label)) < 2:
                a_label = (returns_df_aligned[a] > returns_df_aligned[a].median()).astype(int).values
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
# 2. Streamlit 介面與前端展示 (原 app.py 內容)
# ==========================================
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
st.caption("上方為 S&P
