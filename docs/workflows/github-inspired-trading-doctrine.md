# GitHub-Inspired Trading Doctrine

目的：把高星交易機器人的有效結構收斂成這台機器可執行、可驗證、可回滾的控制面，而不是直接照搬任何單一專案。

## 這次吸收的主來源

- `freqtrade/freqtrade`
  - 強項：dry-run 優先、完整回測、pair filtering、策略優化、lookahead/recursive bias 檢查
- `hummingbot/hummingbot`
  - 強項：市場微結構、maker / arbitrage / connector 架構、多 venue、自動化長時間運行
- `jesse-ai/jesse`
  - 強項：策略語法簡潔、研究/回測/實盤一體、Monte Carlo、ML pipeline、指標與風控內建
- `polakowo/vectorbt`
  - 強項：大量參數搜尋、快速回測、組合式訊號研究
- `nautechsystems/nautilus_trader`
  - 強項：事件驅動、決定論架構、production-grade engine、資料/執行一致性
- `robertmartin8/PyPortfolioOpt`
  - 強項：組合風險與權重分配，不只看單一幣對

## 收斂後的邏輯層

### 1. 研究層

- 先做多時間框架分析，不直接下單
- 先看趨勢、動能、波動、成交量與 breakout，而不是單點指標
- 用 sweep / backtest 尋找穩定區，不為了單次最佳化去追歷史峰值

### 2. 決策層

- 只接受「趨勢 + 動能 + 結構」同向的訊號
- 設定最低分數、最低 convergence、最低 ADX
- 不把 ML 當主引擎，只能做次級 gate 或排序加權

### 3. 風控層

- 單筆風險受 `max_account_risk_pct` 限制
- 每日交易次數受限
- 連續虧損、冷卻期、challenge drawdown 都是硬 gate
- 若交易所最小 notional 與風險 sizing 衝突，寧可不下單

### 4. 執行層

- 先 `analysis -> backtest -> live-readiness -> live-pilot`
- `live-readiness` 不通過時，不進入真單
- 能用 paper/testnet 就先用 paper/testnet
- 所有 live action 都要落 journal

### 5. 監控層

- 每次執行保留報表與 artifacts
- balance snapshots 持續寫入
- 以 challenge progress / drawdown / live order journal 觀察是否真的在成長
- 用 `n8n` 和定時器做低 token 自動化，不靠高頻對話輪詢

## 這套系統目前不照搬的東西

- 不照搬 Hummingbot 的高頻 maker / DEX gateway 複雜度
- 不照搬 Jesse 的全量 ML / editor / AI assistant 棧
- 不照搬 NautilusTrader 的重型 engine 與依賴
- 不做「為了追求高報酬」而放鬆實盤 gate

## 目前推薦的實操順序

1. `openclaw-quantctl validate-config --strategy-config config/strategy-stable-risk.yaml`
2. `python3 scripts/run_stability_workflow.py --strategy-config config/strategy-stable-risk.yaml`
3. `openclaw-quantctl analyze <SYMBOL> --market futures --interval 1h --render-chart`
4. `openclaw-quantctl backtest --strategy-config config/strategy-stable-risk.yaml`
5. `openclaw-quantctl live-readiness --strategy-config config/strategy-stable-risk.yaml`
6. 只有在 auth、challenge、risk gate 都通過時，才考慮 `live-pilot --execute`

## 成長獲利的正確定義

- 不是單次爆利
- 是 equity curve 持續上升、最大回撤受控、風險暴露可重現
- 所以這套系統用：
  - challenge progress
  - drawdown
  - balance snapshot curve
  - live order journal
  來判斷是否真的在變強

## Senior-Trader Checklist

- 每筆 TP / SL / manual close 都要先經過 `review-closed-trades`，再交給 `strategy-optimizer` 收斂。
- 收斂只允許動策略門檻，不允許碰 execution core、`live_trading` 底層開關、或 challenge / 槓桿基線。
- 新策略或新幣種先進 `paper/demo-first` cohort，不把單次表現直接升成 live 預設。
- 候選單除了分數，還要同時看 ADX、convergence、reward/risk、exchange minimum、既有持倉與 challenge 狀態。
- `allowed=false` 不是壞事；它代表 gate 仍在保護資金。

## 還沒放鬆的缺口

- 組合層風險上限仍需要獨立治理，不該只看單一標的。
- 不同 market regime 的停用/降權規則仍要持續細化。
- venue-specific slippage / fee calibration 仍要按標的與時段累積，而不是只靠固定假設。
- kill-switch 規則必須維持可審計，不要交給臨場情緒或聊天指令。
