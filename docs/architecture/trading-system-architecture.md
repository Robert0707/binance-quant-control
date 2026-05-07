# Trading System Architecture

目標：把整個交易專案整理成清楚分層，避免策略、排程、保護單、回測、優化、Hailo triage 混在一起。

This architecture follows the mainstream split used by mature bot frameworks:
universe/market selection, alpha generation, portfolio/risk construction,
pre-trade risk management, execution, and post-trade review are separate
responsibilities. A strategy is not allowed to trade just because a classifier
looks right; it must survive net expectancy, payoff, cost, and risk gates.

## 分層

### 1. Execution Layer

負責實際下單與保護單。

- `live_execution.py`
- `position_manager.py`
- `binance_api.py`

原則：
- 只執行已通過 gate 的動作
- 不自己決定要不要看新聞或改策略

### 2. Controller Layer

負責長跑流程協調，對應主流 bot 的 controller 概念。

- `autonomy.py`
- `run_autonomous_trader.py`

目前拆成兩條：
- `autonomous-trader.default.yaml`
  低頻巡航，30 分鐘級，負責選幣 / digest / 候選判斷
- `autonomous-guardian.default.yaml`
  高頻 guardian，5 分鐘級，只看持倉保護與風險
- `autonomous-live-lane.default.yaml`
  正式 live lane，30 分鐘級，允許自動新倉進場，但必須通過更嚴格 gate

### 3. Review And Optimization Layer

負責每一單結束後的檢討與策略收斂。

- `review-closed-trades`
- `payoff_objective.py`
- `strategy_optimizer.py`
- `strategy-hermes-pro.auto.yaml`

原則：
- 可以改策略參數
- 不直接改 execution 核心邏輯
- ranking 以 `expectancy_r`、`payoff_ratio`、PF、樣本數、成本壓力為主，
  win rate 只是輔助描述；`>=80%` 是 elite label，不是開單權限

### 4. Research And Macro Layer

負責低頻高價值判斷。

- `daily_digest.py`
- `strategy_analyzer.py`
- 新聞 / 巨鯨 / GitHub 基礎設施觀察

原則：
- 沒持倉、需要候選幣時才啟用
- 有 digest 快取就重用，避免浪費 token

### 5. Hailo Local Intelligence Layer

負責本地 triage，不直接下單。

- `projects/hailo-trading-triage`

Hailo 主要責任：
- 事件分類
- 圖像 / 圖表分類
- autonomy cycle / strategy optimizer / profit floor 狀態分流
- 只把高價值事件送到雲端

### 6. Official Engine / Backtest Layer

負責大規模回測與官方工具鏈。

- `external/freqtrade`
- `bin/openclaw-freqtradectl`

參考做法：
- Freqtrade Hyperopt / backtesting / protections
- 選幣與 whitelist 先由 digest 控制，再交給回測引擎驗證

## 交易節奏

1. 高頻 guardian 看持倉、保護單、停損停利、trailing。
2. 低頻 autonomy 看大局、選幣、候選與進場 gate。
3. formal live lane 只在無持倉且候選通過更嚴格 gate 時才自動新倉。
4. 成交後 review 自動寫檢討。
5. optimizer 根據已平倉結果收斂策略參數。
6. Hailo 在整條鏈上做本地 triage，雲端只看高價值事件。

## 利潤門檻

`min_expected_profit_usdt` 是進場最低目標，不是保證獲利。

目前用途：
- 若 TP1 扣估計雙邊手續費後，預估利潤低於門檻，就直接擋單。
- 這可避免小帳戶或低波動時做太小、太不划算的交易。

## Payoff-First Objective

`payoff_objective.py` 是研究、risk sweep、iteration、optimizer 共用的
收益目標函式。它刻意把 `expectancy_r`、`payoff_ratio`、PF、樣本數放在
headline win rate 前面，避免 80% 勝率但平均贏家太小的 cohort 被排到前面。

使用位置：
- `alpha_research.py`: ranking score 會加入 payoff objective
- `risk_combo_sweep.py`: recovery candidates 先按 payoff/expectancy 排序
- `high_win_iteration.py`: best alpha 與 expansion candidate 用同一套排序
- `strategy_optimizer.py`: closed-review cohort 不再用字串排序 promotion decision

## Exit Profile Layer

`exit_profiles.py` owns staged TP weights and runner stop behavior. This keeps
entry alpha separate from exit payoff construction:

- `balanced`: legacy middle ground.
- `payoff_runner`: lower first-scale-out, larger runner reserve, and profit
  locks after later targets. This is the default for the core expectancy
  research profile because the current blocker is weak average winner size.
- `capital_preservation`: earlier scale-out for defensive experiments.

`risk-combo-sweep` now searches exit profiles as part of the recovery grid. When
`--max-configs` is used, the bounded sampler preserves exit-profile coverage so
an interactive run does not accidentally test only the first few parameter
combinations.

Performance boundary:
- interactive sweeps should pass `--max-configs` and prefer `grid-mode fast`
  before long focused/standard sweeps
- long focused sweeps belong in background research, not in a blocking chat turn
- Hailo is useful for report triage/compression, but it does not accelerate
  pandas/backtest control flow; optimize the sweep budget and caching first

基本 promotion 目標：
- `trade_count >= 100`
- `win_rate >= 65%`
- `stop_loss_ratio <= 35%`
- `PF >= 1.5`
- `expectancy_r >= 0.10`
- `payoff_ratio >= 1.15`

## 不應混在一起的東西

- 保護單邏輯 不等於 策略優化
- Hailo triage 不等於 雲端大模型決策
- 回測引擎 不等於 live execution
- digest 候選選幣 不等於 最終下單許可
