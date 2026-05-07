# Strategy Convergence Validation

目的：把策略是否有效，從寬鬆的單一回測判斷，升級成多訊號、分層驗證、cohort 化的正式收斂制度。

## 三段式 Gate

## Route Quarantine Recovery Gate

`risk-combo-sweep` 是 quarantine route 的前置恢復實驗。它不寫入 closed-trade reviews，也不自動清 quarantine。

輸出分兩層：

- `recovery_gate`: 初篩，只檢查 full/test PF 是否回到指定目標，例如 `>0.8`，以及 test trade count 是否足夠。
- `robust_recovery_gate`: 放回 supervisor 前置條件，還會檢查 train PF、full PF、test PF、walk-forward 最低 PF、walk-forward 正報酬視窗數、route max drawdown、route max loss streak。

只有 `robust_recovery_gate.passed=true` 的組合，才可以進入人工檢討與小批 demo validation。`recovery_gate.passed=true` 但 `robust_recovery_gate.passed=false` 時，視為局部行情有效，不得放回 supervisor。

## Loss Diagnostics / Side Gate

`loss-diagnostics` 是正式虧損歸因入口：

```bash
openclaw-quantctl loss-diagnostics --min-bucket-trades 5 --top-n 20 --compact
```

它會輸出：

- route / side / source / symbol / exit reason
- score bin / convergence bin / R multiple bin
- PF、win/loss、net PnL、stop-loss ratio、high-conviction loss count
- `side_policy_recommendations`

`live-readiness` 與 `live-pilot` 會使用 route-side gate：

- route/side 樣本數小於 30 時，不因資料不足自動阻擋，但仍受 optimizer、route quarantine、ADX/convergence 等 gate 保護。
- route/side 樣本數達 30 且 PF < `0.8` 時阻擋。
- route/side 樣本數達 30 且 PF < `1.0` 且 net PnL < 0 時阻擋。

這比單看總 route 更細，避免空單或多單其中一側長期負期望時，被另一側或其它 symbol 掩蓋。

### Stage A: 快篩

用途：快速淘汰沒有基本正期望的策略。

門檻：
- `win_rate >= 65%`
- `profit_factor >= 1.2`
- `expectancy_r >= 0.05`
- `payoff_ratio >= 1.0`
- `trade_count >= 100`

結果：
- `screening_status=passed` 才可進下一階段

### Stage B: 長驗證

用途：確認策略不是只靠少量樣本或局部行情灌出高勝率。

門檻：
- `win_rate >= 70%`
- `profit_factor >= 1.5`
- `expectancy_r >= 0.10`
- `payoff_ratio >= 1.15`
- `simulated trades >= 100`
- `max_drawdown_pct <= route ceiling`
- `loss_streak <= route ceiling`

結果：
- `validation_status=passed`
- `promotion_decision=promote`

### Stage C: 卓越標記

用途：標出真正高品質的高信心策略，而不是一般策略的必要條件。

門檻：
- `win_rate >= 90%`
- `profit_factor >= 1.5`
- `trade_count >= 100`
- 同時滿足 route 的回撤與連敗限制

結果：
- `elite_status=elite_candidate`
- `promotion_decision=elite_candidate`

## Route 驗證標準

每個 route 在 `config/asset-routing.default.yaml` 內都自帶：
- `screening_min_win_rate`
- `screening_min_profit_factor`
- `screening_min_trades`
- `validation_min_win_rate`
- `validation_min_profit_factor`
- `validation_min_simulated_trades`
- `max_drawdown_pct`
- `max_loss_streak`
- `elite_*`

`route-symbol SYMBOL` 會直接把這些門檻輸出。

## 正式資料欄位

### Paper / Demo 單

每筆模擬單都應帶：
- `asset_class`
- `route_id`
- `strategy_profile`
- `cohort_id`
- `entry_reason_snapshot`
- `signal_scores`

### Closed Trade Review

每筆平倉 review 都應帶：
- `cohort_id`
- `rule_compliant`
- `false_positive_tag`
- `market_regime_tag`
- `signal_scores`

## Cohort 規則

`cohort_id` 定義：
- `asset_class:strategy_profile:market:interval`

目的：
- 不讓不同策略模板、不同幣類、不同節奏混在一起灌高勝率
- optimizer 以 cohort 為單位給出 `watchlist / promote / elite_candidate / reject`

## 主要輸出

### `route-symbol`
- 顯示 route 與 validation summary

### `paper-order`
- 寫入 signal snapshot 與 cohort id

### `review-closed-trades`
- 回寫 rule compliance 與 false positive tag

### `strategy-optimizer`
- 輸出：
  - `screening_status`
  - `validation_status`
  - `elite_status`
  - `promotion_decision`
  - `convergence_report`

### `backtest` / `backtest-sweep`
- 也會輸出 convergence 判定，供策略快篩與長驗證共用
