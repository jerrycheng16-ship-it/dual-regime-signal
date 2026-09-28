import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from scipy.spatial.distance import cdist

# ==========================================
# 1. Statistical Jump Model (SJM) 實作
# ==========================================
class StatisticalJumpModel:
    def __init__(self, n_clusters=2, jump_penalty=10.0):
        self.n_clusters = n_clusters
        self.jump_penalty = jump_penalty
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


# ==========================================
# 2. 特徵工程 (Feature Engineering)
# ==========================================
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


# ==========================================
# 3. 雙重狀態預測與 BMDA / BMGA 資產池建立
# ==========================================
class DualRegimeAllocationModel:
    def __init__(self, jump_penalty_global=20.0, jump_penalty_asset=20.0, prob_threshold=0.7, ewm_window=63):
        self.jp_global = jump_penalty_global
        self.jp_asset = jump_penalty_asset
        self.threshold = prob_threshold
        self.ewm_window = ewm_window
        
    def run_pipeline(self, returns_df, macro_df, global_proxy_col='LargeCap', riskfree_col='RiskFree'):
        # 1. 建立 Minimalist 特徵
        X_min = extract_minimalist_features(returns_df)
        
        # 2. 建立 Comprehensive 特徵 (修正新版 pandas ffill 寫法)
        X_comp = pd.concat([X_min, macro_df], axis=1).reindex(X_min.index).ffill().bfill()
        
        # 確保型態純粹為浮點數，避免 XGBoost / SciPy 轉型拋出 TypeError
        X_min_vals = X_min.astype(np.float64)
        X_comp_vals = X_comp.astype(np.float64)
        
        risky_assets = [c for c in returns_df.columns if c not in [global_proxy_col, riskfree_col]]
        dates = X_min.index
        
        # ----------------------------------------------------
        # Step 1: 全域與單一資產 SJM 標籤識別
        # ----------------------------------------------------
        global_min_feat = [c for c in X_min.columns if global_proxy_col in c]
        sjm_global = StatisticalJumpModel(n_clusters=2, jump_penalty=self.jp_global)
        global_states = sjm_global.fit_predict(X_min_vals[global_min_feat].values)
        
        ret_g0 = returns_df.loc[dates, global_proxy_col][global_states == 0].mean()
        ret_g1 = returns_df.loc[dates, global_proxy_col][global_states == 1].mean()
        global_bull_state = 1 if ret_g1 > ret_g0 else 0
        global_regimes_label = (global_states == global_bull_state).astype(int)
        
        asset_regimes_label = {}
        for a in risky_assets:
            a_min_feat = [c for c in X_min.columns if a in c]
            sjm_asset = StatisticalJumpModel(n_clusters=2, jump_penalty=self.jp_asset)
            a_states = sjm_asset.fit_predict(X_min_vals[a_min_feat].values)
            
            ret_a0 = returns_df.loc[dates, a][a_states == 0].mean()
            ret_a1 = returns_df.loc[dates, a][a_states == 1].mean()
            a_bull_state = 1 if ret_a1 > ret_a0 else 0
            asset_regimes_label[a] = (a_states == a_bull_state).astype(int)
            
        asset_regimes_label_df = pd.DataFrame(asset_regimes_label, index=dates)

        # ----------------------------------------------------
        # Step 2: XGBoost 預測與平滑處理
        # ----------------------------------------------------
        y_global = pd.Series(global_regimes_label, index=dates).shift(-1)
        
        # 轉成 numpy array 傳給 XGBoost 防止 Pandas 索引欄位引發 TypeError
        X_train_g = X_comp_vals.iloc[:-1].values
        y_train_g = y_global.iloc[:-1].values.astype(int)
        
        xgb_global = XGBClassifier(n_estimators=100, max_depth=3, random_state=42, eval_metric='logloss')
        xgb_global.fit(X_train_g, y_train_g)
        prob_global_bull = xgb_global.predict_proba(X_comp_vals.values)[:, 1]
        
        pred_global_bear = (prob_global_bull < (1 - self.threshold)).astype(int)
        pred_global_bull = (prob_global_bull > self.threshold).astype(int)
        
        alpha_ewm = 2.0 / (self.ewm_window + 1)
        smoothed_prob_asset_bull = {}
        
        for a in risky_assets:
            y_asset = asset_regimes_label_df[a].shift(-1)
            X_train_a = X_comp_vals.iloc[:-1].values
            y_train_a = y_asset.iloc[:-1].values.astype(int)
            
            xgb_asset = XGBClassifier(n_estimators=100, max_depth=3, random_state=42, eval_metric='logloss')
            xgb_asset.fit(X_train_a, y_train_a)
            raw_prob_bull = xgb_asset.predict_proba(X_comp_vals.values)[:, 1]
            
            prob_series = pd.Series(raw_prob_bull, index=dates)
            smoothed_prob_asset_bull[a] = prob_series.ewm(alpha=alpha_ewm).mean()
            
        smoothed_prob_asset_df = pd.DataFrame(smoothed_prob_asset_bull)

        # ----------------------------------------------------
        # 建立 BMDA 與 BMGA 動態資產池
        # ----------------------------------------------------
        bmda_sets = {}
        bmga_sets = {}
        
        for t in range(len(dates)):
            d = dates[t]
            is_global_bear = pred_global_bear[t] == 1
            is_global_bull = pred_global_bull[t] == 1
            
            asset_bulls = smoothed_prob_asset_df.loc[d] > self.threshold
            
            if is_global_bear:
                selected_bmda = asset_bulls[asset_bulls].index.tolist()
                selected_bmda.append(riskfree_col)
                bmda_sets[d] = selected_bmda
            else:
                bmda_sets[d] = [riskfree_col]
                
            if is_global_bull:
                selected_bmga = asset_bulls[asset_bulls].index.tolist()
                if global_proxy_col not in selected_bmga:
                    selected_bmga.append(global_proxy_col)
                bmga_sets[d] = selected_bmga
            else:
                bmga_sets[d] = [global_proxy_col]

        # ----------------------------------------------------
        # Step 3: 投資組合配置與回測
        # ----------------------------------------------------
        portfolio_returns = []
        tc_rate = 0.0010
        prev_weights = pd.Series(0.0, index=returns_df.columns)
        
        for t in range(len(dates) - 1):
            d_today = dates[t]
            d_next = dates[t+1]
            
            if pred_global_bear[t] == 1:
                target_assets = bmda_sets[d_today]
            else:
                target_assets = bmga_sets[d_today]
                
            target_weights = pd.Series(0.0, index=returns_df.columns)
            target_weights[target_assets] = 1.0 / len(target_assets)
            
            turnover = np.sum(np.abs(target_weights - prev_weights))
            tc = turnover * tc_rate
            
            r_next = returns_df.loc[d_next]
            port_r = np.sum(target_weights * r_next) - tc
            portfolio_returns.append({'Date': d_next, 'Return': port_r, 'Turnover': turnover})
            
            prev_weights = target_weights.copy()
            
        res_df = pd.DataFrame(portfolio_returns).set_index('Date')
        return res_df, bmda_sets, bmga_sets
