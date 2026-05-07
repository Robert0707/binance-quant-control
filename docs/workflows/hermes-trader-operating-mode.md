# Hermes Trader Operating Mode

目的：把這台機器人在交易這條線上「現在能做什麼、該做什麼、不能做什麼」固定下來，避免後續 agent 又退回新手模式，或反過來放鬆到失控。

## Hermes 目前能做的事

- 定時或手動回寫 closed-trade review，針對每一筆 TP / SL / manual close 做策略檢討
- 定時收集 crypto 新聞摘要
- 監看 Whale Alert 巨鯨轉帳並做交易所流入 / 流出方向判讀
- 追蹤高星交易基礎設施專案：
  - `freqtrade/freqtrade`
  - `hummingbot/hummingbot`
  - `jesse-ai/jesse`
  - `ccxt/ccxt`
  - `bmoscon/cryptofeed`
  - `polakowo/vectorbt`
- 針對多個幣對做分析、排序、決策、paper-only pre-trade notification
- 用現有 `analysis -> backtest -> live-readiness -> live-pilot` 控制面執行，並保留 `Freqtrade` 當外部方法論 / 後續整合參考
- 產生成長報表、daily digest、journal、challenge snapshot

## Hermes 現在不該做的事

- 不該把排程研究 lane 拿去做高成本 `submit-analysis` 背景任務
- 不該在 Binance private auth 壞掉時開實單
- 不該因為一則新聞就追價
- 不該因為巨鯨單筆轉帳就直接反應成下單命令
- 不該在 `range` 結構裡硬做趨勢單
- 不該把 GitHub repo 活躍度誤認成價格方向本身

## 與新手模式不同的地方

- 高風險新聞不再一刀切全部 `no_trade`
  - 若內部訊號、巨鯨方向、外部上下文仍強烈一致，可降級成 `watchlist_only`
  - 只有在 `high news risk + whale opposed` 這類真正不對稱環境才直接 `no_trade`
- 內部分數不是唯一標準
  - `analysis score`
  - `convergence`
  - `ADX`
  - `whale alignment`
  - `news bias`
  - `GitHub observability`
  一起決定是否值得進一步行動
- 放寬的是「出手時機」，不是「生存底線」
  - 仍保留 hard stop-loss sizing
  - 仍保留 exchange minimum / margin checks
  - 仍保留 challenge drawdown gate
  - 仍保留 doctor / auth / execution readiness gate

## 推薦操作檔

- 保守生存基線：`config/strategy-stable-risk.yaml`
- 專業交易員模式：`config/strategy-hermes-pro.yaml`
- 超小資金試單：`config/strategy-live-pilot.yaml`

## 現在的真實邊界

- `doctor` 若 private auth 失敗，Hermes 可以分析、排序、通知，但不能進真單
- `n8n Slack` 若沒有 credentials 與 public webhook，Hermes 可以保留 workflow 與 readiness check，但不能說 Slack trigger 已完成
- 目前正式 execution boundary 仍是本專案既有的 Binance control plane；若未來真的切到 `Freqtrade`，必須是獨立整合專案，不可在文件先假設已完成

## 標準節奏

1. `doctor`
2. `review-closed-trades`
3. `daily digest`
4. `analysis / ranking`
5. `strategy analyzer`
6. `live-readiness`
7. 若 auth / risk / challenge 都過，再由既有 `live-pilot --execute` 邊界內執行
