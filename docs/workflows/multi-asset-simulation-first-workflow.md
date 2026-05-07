# Multi-Asset Simulation-First Workflow

目的：把「不同幣種用不同策略、先模擬再收斂、最後才考慮實單」做成固定控制面，減少你一個指令一個動作的負擔。

## 核心原則

- 不改 `live_trading` 底層邏輯。
- 新策略先走 `paper/demo-first`，不直接升級 live。
- 每個 symbol 先分類，再決定策略與 review lane。
- 每筆模擬單、每筆已平倉 review 都要帶 `asset_class`、`route_id`、`review_lane`。

## 路由規則

- `btc_core`:
  `BTCUSDT`，使用 `config/strategy-btc-volatility.yaml`
- `eth_core`:
  `ETHUSDT`，使用 `config/strategy-eth-trend.yaml`
- `major_alt_trend`:
  `BNB/SOL/XRP/ADA/AVAX/NEAR/...`，使用 `config/strategy-major-alt-trend.yaml`
- `meme_high_beta`:
  `DOGE/PENGU/1000PEPE/TRUMP/WIF/...`，使用 `config/strategy-meme-momentum.yaml`
- `defensive_unknown`:
  未分類標的，使用 `config/strategy-defensive-default.yaml`

完整機器可讀版本在 `config/asset-routing.default.yaml`。

## 自動化分流

1. 資料收集
   `openclaw-quantctl analyze SYMBOL --market futures --interval 1h|4h`
2. 策略路由
   `openclaw-quantctl route-symbol SYMBOL`
3. 模擬單
   `openclaw-quantctl paper-order SYMBOL --side BUY|SELL --notional-usdt 3`
4. 持倉保護
   `openclaw-quantctl manage-position ...` 或 `trailing-update ...`
5. 回測與 sweep
   `openclaw-quantctl backtest ...`
   `openclaw-quantctl backtest-sweep`
6. 已平倉檢討
   `openclaw-quantctl review-closed-trades --compact`
7. 策略收斂
   `python3 scripts/run_strategy_optimizer.py --config config/strategy-optimizer.default.yaml`

## 一鍵巡航

```bash
cd /home/robert/python/projects/binance-quant-control
python3 scripts/run_autonomous_trader.py --config config/autonomous-simulation-lane.default.yaml
```

這條 lane 會：

- 保留 live entry 關閉
- 根據 digest 選幣
- 先做 `route-symbol`
- 用對應策略做分析與 gate
- gate 通過就自動記錄模擬單到 `state/paper-orders.jsonl`
- 後續交給 `review-closed-trades` 與 optimizer

## Demo API 邊界

- 如果你已提供 Binance demo/testnet API，系統仍維持 `BINANCE_USE_TESTNET=true`
- 這份 workflow 先以 paper journal 為主
- 等你確認 demo API lane 要正式接單，再單獨開一條 testnet execution lane，不和 main workflow 混在一起
