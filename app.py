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
    df_feat = df_feat.loc[:, df_feat.var() > 1e-8]
    
    scaler = StandardScaler()
    scaled_vals = scaler.fit_transform(df_feat.values)
    return pd.DataFrame(scaled_vals, index=df_feat.index, columns=df_feat.columns)


class DualRegimeAllocationModel:
    def __init__(self, jump_penalty_global=20.0, jump_penalty_asset=20.0, prob_threshold=0.7, ewm_window=63, rebalance_freq='Monthly', momentum_lookback=63, defense_mode='Strict'):
        self.jp_global = float(jump_penalty_global)
        self.jp_asset = float(jump_penalty_asset)
        self.threshold = float(prob_threshold)
        self.ewm_window = int(ewm_window)
        self.rebal_freq = rebalance_freq
        self.mom_lookback = int(momentum_lookback)
        self.defense_mode = defense_mode
        
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

        price_levels = (1 + returns_df_aligned[risky_assets]).cumprod()
        mom_matrix = price_levels / price_levels.shift(self.mom_lookback) - 1
        mom_matrix = mom_matrix.fillna(1.0)

        stock_like_assets = ['Nasdaq', 'S&P500']

        bmda_sets = {}
        bmga_sets = {}
        
        for t in range(len(dates)):
            d = dates[t]
            is_global_bear = pred_global_bear[t] == 1
            is_global_bull = pred_global_bull[t] == 1
            
            if d in smoothed_prob_asset_df.index:
                asset_bulls = smoothed_prob_asset_df.loc[d] > self.threshold
                ml_selected = asset_bulls[asset_bulls].index.tolist()
            else:
                ml_selected = []
            
            if d in mom_matrix.index:
                positive_mom_assets = mom_matrix.loc[d][mom_matrix.loc[d] > 0.0].index.tolist()
                filtered_risky = [a for a in ml_selected if a in positive_mom_assets]
            else:
                filtered_risky = ml_selected

            if is_global_bear:
                if self.defense_mode == 'Strict':
                    selected_bmda = [a for a in filtered_risky if a not in stock_like_assets]
                else:
                    selected_bmda = filtered_risky.copy()
                
                selected_bmda.append(riskfree_col)
                bmda_sets[d] = selected_bmda
            else:
                bmda_sets[d] = [riskfree_col]
                
            if is_global_bull:
                selected_bmga = filtered_risky.copy()
                if global_proxy_col not in selected_bmga and global_proxy_col in positive_mom_assets:
                    selected_bmga.append(global_proxy_col)
                if not selected_bmga or global_proxy_col not in selected_bmga:
                    selected_bmga = [global_proxy_col]
                bmga_sets[d] = selected_bmga
            else:
                bmga_sets[d] = [global_proxy_col]

        df_temp = pd.DataFrame(index=dates)
        if self.rebal_freq == 'Monthly':
            df_temp['key'] = df_temp.index.to_period('M')
            rebal_dates = set(df_temp.groupby('key').apply(lambda x: x.index[-1]))
        elif self.rebal_freq == 'Weekly':
            df_temp['key'] = df_temp.index.to_period('W')
            rebal_dates = set(df_temp.groupby('key').apply(lambda x: x.index[-1]))
        else:
            rebal_dates = set(dates)

        portfolio_returns = []
        tc_rate = 0.0010
        prev_weights = pd.Series(0.0, index=returns_df_aligned.columns)
        
        for t in range(len(dates) - 1):
            d_today = dates[t]
            d_next = dates[t+1]
            
            is_rebal_day = (t == 0) or (d_today in rebal_dates)
            
            if is_rebal_day:
                if pred_global_bear[t] == 1:
                    target_assets = bmda_sets.get(d_today, [riskfree_col])
                else:
                    target_assets = bmga_sets.get(d_today, [global_proxy_col])
                    
                target_weights = pd.Series(0.0, index=returns_df_aligned.columns)
                target_weights[target_assets] = 1.0 / len(target_assets)
            else:
                target_weights = prev_weights.copy()
            
            turnover = np.sum(np.abs(target_weights - prev_weights)) if is_rebal_day else 0.0
            tc = turnover * tc_rate
            
            r_next = returns_df_aligned.loc[d_next]
            port_r = np.sum(target_weights * r_next) - tc
            
            portfolio_returns.append({
                'Date': d_next, 
                'Return': port_r, 
                'Turnover': turnover,
                'Global_Bull_Prob': prob_global_bull[t],
                'Is_Rebal': is_rebal_day
            })
            prev_weights = target_weights.copy()
            
        res_df = pd.DataFrame(portfolio_returns).set_index('Date')
        return res_df, bmda_sets, bmga_sets, rebal_dates


# ==========================================
# 2. Streamlit 介面與前端展示
# ==========================================
st.set_page_config(page_title="雙重狀態資產配置系統", layout="wide")

st.title("📈 雙重狀態動態資產配置系統")
st.caption("結合 SJM、機器學習、絕對動量濾網與多空預測勝率追蹤。")

st.sidebar.header("📅 回測時間區間設定")
default_start = pd.to_datetime("2020-01-01")
default_end = pd.to_datetime("2025-12-31")

start_date = st.sidebar.date_input("回測開始日期", default_start)
end_date = st.sidebar.date_input("回測結束日期", default_end)

st.sidebar.header("⚙️ 模型與策略模式設定")
rebal_freq_option = st.sidebar.selectbox("資產調倉頻率", ["每月調整 (Monthly)", "每週調整 (Weekly)", "每日調整 (Daily)"], index=0)
freq_mapping = {"每月調整 (Monthly)": "Monthly", "每週調整 (Weekly)": "Weekly", "每日調整 (Daily)": "Daily"}
chosen_freq = freq_mapping[rebal_freq_option]

defense_option = st.sidebar.selectbox(
    "熊市防禦資產池模式", 
    [
        "絕對安全防禦 (Strict: 熊市嚴禁股票類資產)", 
        "動量優勢導向 (Flexible: 允許動量為正的股票)"
    ], 
    index=0
)
chosen_defense_mode = 'Strict' if "Strict" in defense_option else 'Flexible'

mom_lb = st.sidebar.slider("絕對動量回看天數 (Momentum Lookback)", 21, 126, 63, step=1)
jp_global = st.sidebar.slider("全域 Jump Penalty (λ_global)", 1.0, 50.0, 15.0, step=1.0)
jp_asset = st.sidebar.slider("資產 Jump Penalty (λ_asset)", 1.0, 50.0, 15.0, step=1.0)
prob_thresh = st.sidebar.slider("分類機率門檻", 0.5, 0.9, 0.7, step=0.05)
ewm_win = st.sidebar.slider("EWM 機率平滑視窗 (天)", 10, 126, 63, step=1)

run_button = st.sidebar.button("🚀 執行回測與分析")

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
    with st.spinner(f"模型運算中 (模式：{defense_option})..."):
        model_instance = DualRegimeAllocationModel(
            jump_penalty_global=float(jp_global),
            jump_penalty_asset=float(jp_asset),
            prob_threshold=float(prob_thresh),
            ewm_window=int(ewm_win),
            rebalance_freq=chosen_freq,
            momentum_lookback=int(mom_lb),
            defense_mode=chosen_defense_mode
        )
        res_df, bmda_hist, bmga_hist, rebal_dates = model_instance.run_pipeline(
            returns_df.copy(), macro_df.copy(), global_proxy_col='S&P500', riskfree_col='RiskFree'
        )
        st.session_state['results'] = res_df
        st.session_state['bmda'] = bmda_hist
        st.session_state['bmga'] = bmga_hist
        st.session_state['rebal_dates'] = rebal_dates

res_df = st.session_state['results']
bmda_hist = st.session_state['bmda']
bmga_hist = st.session_state['bmga']
rebal_dates = st.session_state['rebal_dates']

# ==========================================
# 3. 顯示網頁頂端：最近一期訊號與建議配置
# ==========================================
st.markdown("---")
st.subheader("🎯 最近一期市場訊號與建議配置")

latest_date = res_df.index[-1]
latest_prob = res_df.loc[latest_date, 'Global_Bull_Prob']
is_latest_bear = latest_prob < (1.0 - prob_thresh)
latest_state = "🐻 熊市防禦 (BMDA)" if is_latest_bear else "🚀 牛市成長 (BMGA)"
latest_assets = bmda_hist.get(latest_date, []) if is_latest_bear else bmga_hist.get(latest_date, [])

c1, c2, c3 = st.columns(3)
c1.metric("最新訊號日期 (生效日)", latest_date.strftime('%Y-%m-%d'))
c2.metric("模型牛市預測機率", f"{latest_prob:.4f}", latest_state)
c3.metric("建議配置資產池", ", ".join(latest_assets))
st.markdown("---")

# ==========================================
# 4. 績效與預測勝率計算
# ==========================================
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

# 計算多空預測勝率
correct_count = 0
total_count = 0
for d in res_df.index:
    prob = res_df.loc[d, 'Global_Bull_Prob']
    sp_ret = benchmark_returns.loc[d] if d in benchmark_returns.index else 0
    
    is_bull_pred = prob >= 0.5  # 以 0.5 作為多空分界
    is_market_up = sp_ret > 0
    is_market_down = sp_ret < 0
    
    if is_bull_pred and is_market_up:
        correct_count += 1
        total_count += 1
    elif not is_bull_pred and is_market_down:
        correct_count += 1
        total_count += 1
    elif sp_ret != 0:
        total_count += 1

win_rate = (correct_count / total_count) * 100 if total_count > 0 else 0

st.subheader(f"📊 核心績效指標與預測勝率 ({rebal_freq_option})")
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("年化夏普值", f"{sharpe:.2f}", f"大盤基準: {benchmark_sharpe:.2f}")
col2.metric("最大回撤", f"{max_dd * 100:.2f}%", f"大盤基準: {benchmark_max_dd * 10
