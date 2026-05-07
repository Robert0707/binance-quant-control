# Delivery Supervision Workflow

目的：把交付前的 500 次模擬單訓練，收斂成一條風控優先、可審計、可交給 Hermes 接手的流程。

## 核心邊界

- 只跑 paper / demo / market replay。
- 不呼叫 `execute_live_order`。
- 不自動打開 `BINANCE_LIVE_TRADING_ENABLED`。
- `strategy-optimizer` 沒有 `promote` / `elite_candidate` 時，live plan 會被全局 gate 擋下。
- route 最近績效惡化時，監督流程會停下並標記 quarantine，等人工檢討。

## 推薦 500 次入口

## 500 次前置風控實驗

若任何 route 被 quarantine，先不要跑 500 次，也不要清 quarantine。先跑：

```bash
openclaw-quantctl loss-diagnostics --min-bucket-trades 5 --top-n 20 --compact
```

先讀：

- `summary.profit_factor`
- `summary.stop_loss_ratio`
- `findings`
- `side_policy_recommendations`
- `worst_buckets`

若 `short-lane-underperforming`、`stop-loss-dominant`、或 route/side PF 低於 `0.8`，先修風控與 route/side gate，不要直接用 500 次 supervisor 追樣本。

```bash
openclaw-quantctl risk-combo-sweep \
  --routes btc-core,meme-high-beta,xau-macro \
  --max-symbols-per-route 1 \
  --limit 1000 \
  --target-profit-factor 0.8 \
  --min-test-trades 3 \
  --compact
```

判讀原則：

- `recovery_candidate_count` 只是初篩，代表 full/test PF 暫時回到目標，不代表可放回 supervisor。
- `robust_recovery_candidate_count` 才是放回前置條件，必須同時通過 train/full/test、walk-forward、回撤與連虧限制。
- `robust_recovery_candidate_count=0` 時，維持 quarantine，demo training 只應驗證 gate 是否正確跳過，不應寫入新的 closed-review 樣本。
- 新聞風險為 `high` 時，維持禁入；不要為了補樣本而強行做 demo/testnet entry。
- `risk-combo-sweep` 會測 side policy，包括 baseline、long-only、disable-shorts、shorts-extra-adx、shorts-extra-confirmation；若只靠禁空或加嚴空單讓局部 PF 變好，但 walk-forward 仍失敗，仍不可放回 supervisor。

## Route / Side 硬 Gate

`live-readiness` / `live-pilot` 會額外檢查 route-side closed-review 歷史：

- 樣本數達 30 後，route/side PF 低於 `0.8` 會阻擋。
- 樣本數達 30 且 route/side 淨 PnL 仍為負，也會阻擋正式 live plan。
- gate 結果在 `live_plan.challenge.route_side_risk`，包含 sample count、PF、net PnL、loss streak 與 reasons。

這個 gate 是為了防止「某條 route quarantine 被清掉，但同一個虧損方向又被 live plan 放回去」。

## 推薦 500 次入口

```bash
openclaw-quantctl delivery-supervisor \
  --cycles 50 \
  --training-rounds 10 \
  --symbols BTCUSDT,ETHUSDT,XAUTUSDT,BNBUSDT,SOLUSDT,XRPUSDT,ADAUSDT,AVAXUSDT,TRXUSDT,UNIUSDT,LINKUSDT,LTCUSDT,DOGEUSDT,1000PEPEUSDT,WIFUSDT \
  --mission-symbols-per-cycle 6 \
  --margin-notional-usdt 3 \
  --max-leverage 3 \
  --compact
```

這會產生 50 個監督 cycle，每個 cycle 做：

- digest / news risk 更新
- 多幣 mission 掃描與 paper order journal
- 10 輪 demo training
- strategy optimizer
- final convergence audit
- route-level PF / loss-streak quarantine

## 停止條件

- demo training 命令失敗
- 近 24 小時模擬 / replay PnL 觸發虧損上限
- route 最近 PF 或連敗觸發 quarantine
- optimizer 第一次 promote / elite_candidate，先停下等人工確認，不直接切 live

## 讀報告

報告位置：

```bash
state/delivery-supervision/*-delivery-supervision.json
```

重點欄位：

- `status`
- `stop_reasons`
- `cycles[].training.response`
- `cycles[].optimizer`
- `cycles[].route_risk.quarantined_routes`
- `cycles[].audit.findings`
- `live_guardrail.real_orders_sent_by_supervisor`

## 正式交付判準

- `doctor --compact` 是 ok
- Hailo triage 在 final audit 中可用
- closed reviews 至少 30 幣且核心 route 有 50+ cohort
- optimizer 不再 reject
- route quarantine 為空
- 最近 demo training 不再呈現系統性負期望
