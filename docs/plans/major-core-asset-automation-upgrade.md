# Major Core Asset Automation Upgrade

目標：把 BTC / ETH / XAU 主流資產做成「官方策略基底 + 本地收斂 + 一個指令總控」的完整工作流，不再需要一個指令一個動作。

## 新增內容

- `config/official-strategy-baselines.yaml`
  - 收錄 BTC / ETH / XAU 的公開策略基底與來源
- `config/strategy-xau-macro.yaml`
  - 新增 XAU macro paper-first lane
- `config/mission-control.default.yaml`
  - 定義一個總控入口的執行、排程、本地/雲端分工
- `openclaw-quantctl mission`
  - 一個入口輸入 symbols / target return / max leverage
- `scripts/run_trading_mission.py`
  - 給排程或外部 workflow 呼叫的 script 入口

## 這次順便整理出的缺陷

1. 新聞仍偏單一來源
- 目前 digest 主要還是單一 RSS feed
- 對 BTC / ETH / XAU 這種 macro 驅動資產還不夠

2. XAU 原生執行不在目前 core
- 這個 repo 是 Binance-first
- 所以 XAU 目前只能先走 `PAXGUSDT` proxy 研究路

3. 關鍵決策與高頻執行還沒有完全拆乾淨
- 高頻看單應留本地
- 新聞與最後核准應留雲端
- 這次已把邊界寫進 mission config 與 workflow

4. 多幣輸入時仍應只選一個最佳候選自動推進
- 避免一次多幣全開造成過度交易
- 先讓系統做選幣，再由同一條 lane 推進

## 建議排程

- `guardian`: 每 5 分鐘
- `scout / mission`: 每 30 分鐘
- `digest / macro`: 每 4 小時
- `review-closed-trades`: 每 2 小時
- `strategy-optimizer`: 每 6 小時

## 對你的操作影響

你接下來不需要再用這種節奏：

- 先問要做哪隻
- 再問用哪個策略
- 再問先模擬還是回測
- 再問要不要 live

而是先用：

```bash
openclaw-quantctl mission --symbols BTCUSDT,ETHUSDT,XAU --target-return-pct 8 --max-leverage 3
```

系統會自己完成前面的研究、分流、快篩與模擬單記錄。
