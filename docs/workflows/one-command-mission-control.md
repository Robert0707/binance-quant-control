# One-Command Mission Control

目標：把「我想做哪些幣、希望多少收益、最高槓桿多少」收斂成一個入口，讓機器人自動完成路由、分析、回測、模擬單、檢討與策略收斂。

## 指令

```bash
openclaw-quantctl mission \
  --symbols BTCUSDT,ETHUSDT,XAU \
  --target-return-pct 8 \
  --max-leverage 3
```

若要在既有 live 風控邊界內，讓它對最佳候選嘗試走正式 live plan：

```bash
openclaw-quantctl mission \
  --symbols BTCUSDT,ETHUSDT \
  --target-return-pct 8 \
  --max-leverage 3 \
  --execute-live
```

## 實際流程

1. 正規化 symbol。
2. 依 `config/asset-routing.default.yaml` 分到正確 route。
3. 載入該 route 的策略模板與官方基底來源。
4. 跑本地分析與 signal scoring。
5. 跑 route 專屬 backtest 與 convergence gate。
6. 選出最佳候選。
7. 候選通過門檻才自動寫入 `paper-order`。
8. 依最高原則，分析完成後一定先寫入 `paper-order` 做真市場模擬觀察；promotion gate 只影響能不能往 live 推進。
9. 若加上 `--execute-live`，也只有 promotion gate 通過時，才會在既有 `live-readiness` / execution core 邊界內試著走真單。
10. 任務最後自動跑 optimizer，產生新的策略收斂報告。

## 本地 / 雲端分工

本地：
- route-symbol
- analyze
- signal-score
- backtest
- paper-order
- review-closed-trades
- strategy-optimizer

雲端：
- 新聞風險整理
- whale context
- strategy analyzer 最終核准
- live 前最終判斷

原則：
- 雲端只做最後的高價值判斷。
- 大部分高頻動作都留在本地，減少 token 成本與不穩定性。

## XAU 說明

- `XAU` / `XAUUSD` 目前在這套 Binance-first repo 內會正規化成 `PAXGUSDT`
- 這代表先用 tokenized gold proxy 做研究與模擬驗證
- 若之後要做真正的 XAU spot / CFD / futures，需要新增外部 metals connector，不應硬塞進現有 Binance execution core
