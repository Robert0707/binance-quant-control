# Autonomous AI Trader Workflow

目標：把例行監控、停損保護、已平倉檢討、Hailo 本地 triage、新聞/巨鯨/策略判斷整合成單一自動駕駛流程。

核心原則：
- 能用程式規則處理的事，先在本地完成。
- Hailo 只做本地事件分流與視覺/事件 triage，不假裝做雲端策略判斷。
- 雲端大模型只做高價值判斷：
  - digest 綜合判斷
  - 新聞風險
  - 巨鯨流向脈絡
  - 策略 analyzer approve / watch
- 不改 `live_trading` 底層邏輯，不重寫下單核心；只編排現有安全控制面。

自動巡航入口：
- `python3 scripts/run_autonomous_trader.py --config config/autonomous-trader.default.yaml`
- `python3 scripts/run_autonomous_trader.py --config config/autonomous-testnet-explorer.default.yaml`
- `python3 scripts/run_autonomous_trader.py --config config/autonomous-guardian.default.yaml`
- `python3 scripts/run_strategy_optimizer.py --config config/strategy-optimizer.default.yaml`

流程順序：
1. `review-closed-trades`
2. `auto-pause-trading`
3. `positions` / `account` / `journal-summary`
4. `hailo-trading-triage` 本地事件分流
5. `daily_digest` 看新聞、巨鯨、GitHub 基礎設施與候選幣排序
6. 若已有持倉：
   - 進入 `manage-open-positions`
   - 只處理保護單 / trailing stop 類型動作
   - guardian timer 可直接執行保護單調整，不需要人工再問一次
7. 若沒有持倉：
   - 根據 digest 選幣
   - 用 `live_execution` 產生 live plan
   - 只有在 digest action 與 strategy analyzer 都通過時，才可放行 live entry
8. 成交後：
   - `review-closed-trades` 自動補齊檢討
   - `strategy-optimizer` 依據已平倉結果自動收斂策略參數

預設安全值：
- `execute_live_entries: false`
- `execute_testnet_entries: true` 只在 `config/autonomous-testnet-explorer.default.yaml` 啟用
- 巡航版 `execute_position_protection: false`
- guardian 版 `execute_position_protection: true`
- `require_digest_action: pre_trade_notify`
- `require_strategy_analyzer_approval: true`

## Testnet Explorer

`config/autonomous-testnet-explorer.default.yaml` 是目前的積極迭代入口：

- 使用 Binance futures 24h quote volume 前 60 名當候選池，並合併手寫核心幣清單。
- 新聞觀測仍然每輪執行；高風險新聞會降低 sizing，而不是讓 testnet 研究完全停擺。
- Whale Alert 若不可用，降級成 `neutral`，不阻塞整輪流程。
- `execution_mode=testnet_exploration` 會把 optimizer / quarantine / route-side / historical bucket 這類策略審核降為 warning。
- kill-switch、已有持倉、交易所最小下單量、餘額不足、保護單提交仍然是硬限制。
- `allow_new_entries_with_open_positions=true` 時，未滿 `max_managed_positions` 仍會繼續找下一筆 testnet entry。
- 已持倉 symbol 會從新一輪 digest 候選池排除，避免同一幣被重複加倉。
- `operator-dashboard` 以客戶視角輸出 PnL、保護單覆蓋、虧損主因與下一步建議，不使用大模型。
- 移動止損只會在價格達到 activation 後替換既有保護單；未 armed 前不取消原 stop-loss / take-profit。
- 分段止盈不是平均切，也不是 TP1 全倉；新單會依風險/信心/新聞/幣種路由動態分配 TP 數量。
- 若已開倉保護單被舊 guardian 壓成全倉 TP，可用 `openclaw-quantctl repair-staged-tp --strategy-config ... --symbol ... --side BUY --execute --compact` 重建 stop-loss + 分段 TP。

每支幣的槓桿與艙位由 `symbol_sizing.py` 動態決定：

- BTC / ETH core：testnet 最高約 8x，基準 margin 約 48-50%，高信心與 top-10 流動性可略放大。
- Major alt：testnet 最高約 6x，基準 margin 約 38%，高波動或 top-60 尾端會自動縮小。
- Meme / high beta：testnet 最高約 5x，基準 margin 約 26%，高新聞風險或極端波動會壓到 1x 小艙。
- Unknown top-volume symbol：最高約 3x，基準 margin 約 18%，用小艙收集 testnet exchange feedback。
- 任一幣若 ADX / score / convergence 強，且波動受控，會提高一級槓桿或艙位；若波動、新聞或信心惡化則降槓桿與艙位。

分段止盈權重：
- 只有 1 個 TP：100%。
- 2 個 TP：一般約 30% / 70%；高風險新聞或高 beta 約 40% / 60%；高信心趨勢約 25% / 75%。
- 3 個 TP：一般約 30% / 35% / 35%；高信心趨勢約 25% / 35% / 40%；高風險或高 beta 約 45% / 35% / 20%。

推薦定時分層：
- `openclaw-binance-position-guardian.timer`: 每 5 分鐘，本地保護優先
- `openclaw-binance-testnet-explorer.timer`: 每 20 分鐘，積極 testnet 前 60 名候選掃描與下單
- `openclaw-binance-operator-dashboard.timer`: 每 15 分鐘，生成客戶視角 PnL / 風控 / 虧損原因摘要
- `openclaw-binance-autonomy.timer`: 每 30 分鐘，新聞 / 巨鯨 / 候選判斷
- `openclaw-binance-strategy-optimizer.timer`: 每 6 小時，收斂策略參數

輸出位置：
- `state/autonomy/*-autonomous-cycle.json`
- `state/operator-dashboard/*-operator-dashboard.json`
- `state/live-orders.jsonl` for Binance testnet execution journal

建議部署方式：
- 先讓 timer 跑 dry-run
- 確認幾輪 `entry_gate` 與 `position_management` 輸出都合理
- 再決定是否把 `execute_position_protection` 或 `execute_live_entries` 打開
