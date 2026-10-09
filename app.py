import plotly.graph_objects as go
from plotly.subplots import make_subplots

# 假設我們從模型或回測結果中取得了每日的多空狀態序列 (例如大盤價格與狀態標籤)
# 1 代表牛市 (Bull)，0 代表熊市 (Bear)
# chart_data 包含 S&P500 價格與 regime 狀態欄位

st.subheader("🌍 全球市場多空狀態歷史判定圖 (Regime Shifting)")
st.caption("綠色區間代表模型判定的【牛市成長期】，紅色/灰階區間代表【熊市防禦期】")

# 建立帶有次圖的畫布：上方放 S&P 500 走勢與多空背景，下方放模型預測機率
fig_regime = make_subplots(
    rows=2, cols=1, 
    shared_xaxes=True, 
    vertical_spacing=0.08,
    row_heights=[0.7, 0.3],
    subplot_titles=("S&P 500 歷史走勢與模型多空狀態判定", "模型預測牛市機率 (Prob Bull)")
)

# 1. 繪製 S&P 500 價格曲線
fig_regime.add_trace(
    go.Scatter(x=returns_df.index, y=returns_df['S&P500'].cumsum().apply(lambda x: np.exp(x)), 
               name="S&P 500 累積報酬", line=dict(color='#1f77b4', width=2)),
    row=1, col=1
)

# 2. 繪製下方模型的預測牛市機率曲線
# (假設您在 pipeline 中有保存 prob_global_bull 數值)
# fig_regime.add_trace(
#     go.Scatter(x=returns_df.index, y=prob_global_bull, name="牛市預測機率", line=dict(color='#ff7f0e', width=1.5)),
#     row=2, col=1
# )
# fig_regime.add_hline(y=prob_thresh, line_dash="dash", line_color="gray", row=2, col=1)

fig_regime.update_layout(
    template="plotly_dark",
    height=550,
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)

st.plotly_chart(fig_regime, use_container_width=True)
