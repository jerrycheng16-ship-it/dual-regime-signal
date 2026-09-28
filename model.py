import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from scipy.spatial.distance import cdist

# ==========================================
# 1. Statistical Jump Model (SJM) 實作
# ==========================================
class StatisticalJumpModel:
    """
    論文 Step 1: Discrete-state Statistical Jump Model (SJM)
    使用 Dynamic Programming 求解帶有 Jump Penalty (\lambda) 的隱含狀態序列
    """
    def __init__(self, n_clusters=2, jump_penalty=10.0):
        self.n_clusters = n_clusters
        self.jump_penalty = jump_penalty
        self.centroids = None

    def fit_predict(self, X):
        T, d = X.shape
        K = self.n_clusters
        
        # 初始化 Centroids (使用標準 K-means 思路)
        np.random.seed(42)
        idx = np.random.choice(T, K, replace=False)
        centroids = X[idx]
        
        prev_states = np.zeros(T, dtype=int)
        
        # 座標下降法 (Coordinate Descent) 疊代求解
        for iteration in range(20):
            # 1. 估算發射代價 (Squared L2 Distance)
            emission_cost = cdist(X, centroids, metric='sqeuclidean') # Shape: (T, K)
            
            # 2. 動態規劃 (Dynamic Programming / Viterbi-like) 求最佳狀態序列
            dp = np.zeros((T, K))
            pointers = np.zeros((T, K), dtype=int)
            dp[0] = emission_cost[0]
            
            for t in range(1, T):
                for k in range(K):
                    # 從上一個狀態 k_prev 轉移到 k 的總代價
                    transition_costs = dp[t-1] + self.jump_penalty * (np.arange(K) != k)
                    best_prev = np.argmin(transition_costs)
                    dp[t, k] = transition_costs[best_prev] + emission_cost[t, k]
                    pointers[t, k] = best_prev
            
            # 回溯 (Backtracking)
            states = np.zeros(T, dtype=int)
            states[-1] = np.argmin(dp[-1])
            for t in range(T - 2, -1, -1):
                states[t] = pointers[t + 1, states[t + 1]]
                
            # 收斂條件檢查
            if np.array_equal(states, prev_states):
                break
            prev_states = states.copy()
            
            # 3. 更新 Centroids
            for k in range(K):
                if np.sum(states == k) > 0:
                    centroids[k] = np.mean(X[states == k], axis=0)
                    
        self.centroids = centroids
        return states


# ==========================================
# 2. 特徵工程 (Feature Engineering)
# ==========================================
def extract_minimalist_features(returns_df):
    """
    論文 Step 1: 僅使用報酬率相關指標 (Minimalist Feature Set) 作為 SJM 輸入
    包含：指數平滑平均報酬率、下行標準差、Sortino Ratio (半衰期 1, 5, 10, 21 天)
    """
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
    """
    論文完整的雙重狀態資產配置架構
    """
    def __init__(self, jump_penalty_global=20.0, jump_penalty_asset=20.0, prob_threshold=0.7, ewm_window=63):
        self.jp_global = jump_penalty_global
        self.jp_asset = jump_penalty_asset
        self.threshold = prob_threshold
        self.ewm_window = ewm_window
        
    def run_pipeline(self, returns_df, macro_df, global_proxy_col='LargeCap', riskfree_col='RiskFree'):
        """
        returns_df: 所有資產的日報酬率 (含 LargeCap 與 RiskFree)
        macro_df: 總體經濟指標 (VIX, 殖利率曲線斜率, 通膨代理等)
        """
        # 特徵提取
        X_min = extract_minimalist_features(returns_df)
        X_comp = pd.concat([X_min, macro_df], axis=1).reindex(X_min.index).fillna(method='ffill')
        
        risky_assets = [c for c in returns_df.columns if c not in [global_proxy_col, riskfree_col]]
        dates = X_min.index
        
        # ----------------------------------------------------
        # Step 1: 全域與單一資產 SJM 標籤識別
        # ----------------------------------------------------
        # 1. 全域狀態 (Global Regime)
        global_min_feat = [c for c in X_min.columns if global_proxy_col in c]
        sjm_global = StatisticalJumpModel(n_clusters=2, jump_penalty=self.jp_global)
        global_states = sjm_global.fit_predict(X_min[global_min_feat].values)
        
        # 映射狀態：平均報酬較高者為 Bull (1), 較低者為 Bear (0)
        ret_g0 = returns_df.loc[dates, global_proxy_col][global_states == 0].mean()
        ret_g1 = returns_df.loc[dates, global_proxy_col][global_states == 1].mean()
        global_bull_state = 1 if ret_g1 > ret_g0 else 0
        global_regimes_label = (global_states == global_bull_state).astype(int) # 1: Bull, 0: Bear
        
        # 2. 資產特定狀態 (Asset-Specific Regimes)
        asset_regimes_label = {}
        for a in risky_assets:
            a_min_feat = [c for c in X_min.columns if a in c]
            sjm_asset = StatisticalJumpModel(n_clusters=2, jump_penalty=self.jp_asset)
            a_states = sjm_asset.fit_predict(X_min[a_min_feat].values)
            
            ret_a0 = returns_df.loc[dates, a][a_states == 0].mean()
            ret_a1 = returns_df.loc[dates, a][a_states == 1].mean()
            a_bull_state = 1 if ret_a1 > ret_a0 else 0
            asset_regimes_label[a] = (a_states == a_bull_state).astype(int)
            
        asset_regimes_label_df = pd.DataFrame(asset_regimes_label, index=dates)

        # ----------------------------------------------------
        # Step 2: XGBoost 進行下一期 (t+1) 預測與指數平滑處理
        # ----------------------------------------------------
        # 預測標籤需前移一天 (Shift +1) 避免 Look-ahead bias
        y_global = pd.Series(global_regimes_label, index=dates).shift(-1)
        
        # 全域狀態 XGBoost 訓練與預測
        xgb_global = XGBClassifier(n_estimators=100, max_depth=3, random_state=42, eval_metric='logloss')
        xgb_global.fit(X_comp.iloc[:-1], y_global.iloc[:-1])
        prob_global_bull = xgb_global.predict_proba(X_comp)[:, 1]
        
        # 預測全域熊市/牛市 (帶有 0.7 閥值門檻)
        pred_global_bear = (prob_global_bull < (1 - self.threshold)).astype(int)
        pred_global_bull = (prob_global_bull > self.threshold).astype(int)
        
        # 單一資產狀態 XGBoost 訓練與預測 + 指數平滑 (EWM Smoothing)
        alpha_ewm = 2.0 / (self.ewm_window + 1)
        smoothed_prob_asset_bull = {}
        
        for a in risky_assets:
            y_asset = asset_regimes_label_df[a].shift(-1)
            xgb_asset = XGBClassifier(n_estimators=100, max_depth=3, random_state=42, eval_metric='logloss')
            xgb_asset.fit(X_comp.iloc[:-1], y_asset.iloc[:-1])
            raw_prob_bull = xgb_asset.predict_proba(X_comp)[:, 1]
            
            # 論文式 11, 12: 時間序列平滑過濾頻繁切換
            prob_series = pd.Series(raw_prob_bull, index=dates)
            smoothed_prob_asset_bull[a] = prob_series.ewm(alpha=alpha_ewm).mean()
            
        smoothed_prob_asset_df = pd.DataFrame(smoothed_prob_asset_bull)

        # ----------------------------------------------------
        # 建立 BMDA 與 BMGA 動態資產池 (論文式 9, 10)
        # ----------------------------------------------------
        bmda_sets = {} # 熊市防禦資產池
        bmga_sets = {} # 牛市成長資產池
        
        for t in range(len(dates)):
            d = dates[t]
            is_global_bear = pred_global_bear[t] == 1
            is_global_bull = pred_global_bull[t] == 1
            
            # 取得各風險資產預測是否為 Bull (平滑機率 > 0.7)
            asset_bulls = smoothed_prob_asset_df.loc[d] > self.threshold
            
            # 1. BMDA: 全域熊市 (Global Bear=1) 且 資產自身為牛市 (Asset Bull=1)
            if is_global_bear:
                selected_bmda = asset_bulls[asset_bulls].index.tolist()
                selected_bmda.append(riskfree_col) # RiskFree 永遠作為底線防禦
                bmda_sets[d] = selected_bmda
            else:
                bmda_sets[d] = [riskfree_col]
                
            # 2. BMGA: 全域牛市 (Global Bull=1) 且 資產自身為牛市 (Asset Bull=1)
            if is_global_bull:
                selected_bmga = asset_bulls[asset_bulls].index.tolist()
                if global_proxy_col not in selected_bmga:
                    selected_bmga.append(global_proxy_col) # LargeCap 作為預設成長資產
                bmga_sets[d] = selected_bmga
            else:
                bmga_sets[d] = [global_proxy_col]

        # ----------------------------------------------------
        # Step 3: 等權重投資組合配置與回測 (Extension 2: 0/1 BMDA-BMGA)
        # ----------------------------------------------------
        portfolio_returns = []
        tc_rate = 0.0010 # 交易成本 10 bps
        prev_weights = pd.Series(0.0, index=returns_df.columns)
        
        for t in range(len(dates) - 1):
            d_today = dates[t]
            d_next = dates[t+1]
            
            # 根據 t 日預測選擇 t+1 日的資產池
            if pred_global_bear[t] == 1:
                target_assets = bmda_sets[d_today]
            else:
                target_assets = bmga_sets[d_today]
                
            # 等權重 (Equal Weighting) 分配
            target_weights = pd.Series(0.0, index=returns_df.columns)
            target_weights[target_assets] = 1.0 / len(target_assets)
            
            # 計算交易成本 (Turnover * 10 bps)
            turnover = np.sum(np.abs(target_weights - prev_weights))
            tc = turnover * tc_rate
            
            # t+1 日投資組合總報酬
            r_next = returns_df.loc[d_next]
            port_r = np.sum(target_weights * r_next) - tc
            portfolio_returns.append({'Date': d_next, 'Return': port_r, 'Turnover': turnover})
            
            prev_weights = target_weights.copy()
            
        res_df = pd.DataFrame(portfolio_returns).set_index('Date')
        return res_df, bmda_sets, bmga_sets


# ==========================================
# 4. 假數據生成與範例執行 (Example Usage)
# ==========================================
if __name__ == '__main__':
    np.random.seed(2026)
    dates = pd.date_range('2020-01-01', '2025-12-31', freq='B')
    
    # 模擬 10 個風險資產 + 1 個無風險資產報酬率
    asset_names = ['LargeCap', 'MidCap', 'SmallCap', 'EAFE', 'Treasury', 
                   'Corporate', 'HighYield', 'REIT', 'Commodity', 'Gold', 'RiskFree']
    
    returns_data = np.random.normal(0.0003, 0.01, size=(len(dates), len(asset_names)))
    returns_df = pd.DataFrame(returns_data, index=dates, columns=asset_names)
    returns_df['RiskFree'] = 0.0001 # 無風險資產固定日報酬率
    
    # 模擬 3 個總體經濟特徵 (VIX, Yield Curve, Inflation)
    macro_data = np.random.normal(0, 1, size=(len(dates), 3))
    macro_df = pd.DataFrame(macro_data, index=dates, columns=['VIX', 'Yield_Curve', 'Inflation'])
    
    # 執行模型
    model = DualRegimeAllocationModel(jump_penalty_global=15.0, jump_penalty_asset=15.0)
    results, bmda_history, bmga_history = model.run_pipeline(returns_df, macro_df)
    
    # 輸出統計數據
    cum_returns = (1 + results['Return']).cumprod()
    sharpe = (results['Return'].mean() * 252) / (results['Return'].std() * np.sqrt(252))
    max_dd = (cum_returns / cum_returns.cummax() - 1).min()
    
    print("=" * 50)
    print("雙重狀態資產配置模型 (Dual-Regime Model) 回測結果：")
    print(f"年化夏普值 (Sharpe Ratio): {sharpe:.2f}")
    print(f"最大回撤 (Max Drawdown)  : {max_dd * 100:.2f}%")
    print(f"平均每日換手率 (Turnover): {results['Turnover'].mean() * 100:.2f}%")
    print("=" * 50)
