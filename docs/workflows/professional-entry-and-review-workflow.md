# Professional Entry And Review Workflow

目的：把交易程序升級成「程式先執行重複紀律，Hailo 做本地分流，雲端大模型只處理高價值脈絡」的專業交易桌流程。

## 不改的邊界

- 不改 `BINANCE_LIVE_TRADING_ENABLED`。
- testnet 寫入使用獨立 `BINANCE_TESTNET_TRADING_ENABLED` / `execution_mode=testnet_exploration` 邊界，不等於主網 live 放行。
- 不讓 Hailo 或雲端直接改 `symbol`、`qty`、`action`。
- 不讓每次新聞或單一分數直接觸發實單。

## 分層

### 1. 本地例行層

程式固定處理：

- `review-closed-trades --compact`
- `auto-pause-trading --compact`
- `positions --compact`
- `account --compact`
- `journal-summary`

這一層不需要雲端模型。

### 2. Hailo Triage 層

Hailo / 本地規則處理：

- 已平倉事件分類
- 策略 review 事件壓縮
- autonomy / optimizer 狀態分流
- 只把 high / critical 事件送往雲端候選流程

輸出只寫進 `projects/hailo-trading-triage/state/optimizer-queue/`。

### 3. 雲端脈絡層

雲端只處理：

- 新聞風險
- 巨鯨流向
- GitHub 交易基礎設施觀測
- digest 候選幣與方向
- strategy analyzer approve / watch

雲端輸出是 advisory，不直接下單。

### 4. Professional Entry Gate

新單進場必須通過四層：

- `execution_quality`: reward/risk、費用占比、滑價占比、net profit/risk。
- `market_state`: realized volatility、volume z-score、OBV、Bollinger bandwidth、spread。
- `signal_quality`: score、convergence、ADX、方向與 OBV 是否一致。
- `strategy_performance`: 最近已平倉 review 的 win rate、avg R、stop-loss ratio、loss streak、stop-loss cooldown。

任何一層失敗，`entry_gate.eligible=false`。

### 5. Closed-Trade Review And Optimizer

每筆 TP / SL / manual close 都進：

```bash
openclaw-quantctl review-closed-trades --compact
python3 scripts/run_strategy_optimizer.py --config config/strategy-optimizer.default.yaml
```

optimizer 會根據：

- win rate
- average R
- stop-loss ratio
- loss streak
- average win vs average loss

自動調整 `config/strategy-hermes-pro.auto.yaml` 的策略門檻，不改 execution 核心。

## 正式工作流

```bash
cd /home/robert/python/projects/binance-quant-control
python3 scripts/run_autonomous_trader.py --config config/autonomous-live-lane.default.yaml
```

## Testnet Exploration

```bash
cd /home/robert/python/projects/binance-quant-control
python3 scripts/run_autonomous_trader.py --config config/autonomous-testnet-explorer.default.yaml
```

這條 lane 會真的送 Binance futures testnet / demo API。它會放寬策略審核類 gate 來避免研究流程空轉，但仍保留 kill-switch、已有持倉、exchange filters、餘額與 futures 保護單。每筆 plan 都會輸出 `live_plan.sizing`，可看到該幣的建議槓桿、margin pct、margin USDT、風險 pct 與調整原因。

檢查輸出：

```bash
tail -n 1 state/autonomy/*-autonomous-cycle.json
```

重點欄位：

- `entry_gate.eligible`
- `entry_gate.reasons`
- `entry_gate.professional_gate.layers`
- `steps.review_closed_trades`
- `steps.hailo_triage`

## 成熟框架對齊

- 參考 Freqtrade 的 protections / backtesting / optimization 思路：用績效與風控 gate 控制交易頻率，不靠單一指標衝動進場。
- 對齊 Binance exchange filters 思路：交易前尊重最小數量、最小 notional、價格/數量精度、成本與滑價。
- 本專案仍保持自己的安全控制面：`live-readiness` / `entry_gate` / `live-pilot --execute` 分離。
