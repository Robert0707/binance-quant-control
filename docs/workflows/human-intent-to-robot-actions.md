# Human Intent To Robot Actions

目的：把你常說的話，直接對應成機器人應執行的工作流，減少來回指揮成本。

## 常用語句對應

- `我加入了模擬單 / 測試新策略 / 進去模擬單做測試`
  機器人動作：進 simulation-first lane，先 `route-symbol`，分析完成後一定寫入 `paper-order`，再做 `backtest`、`review-closed-trades`

- `把幣種分類 / 不同幣種要用甚麼策略`
  機器人動作：查 `config/asset-routing.default.yaml`，回報 `asset_class`、`strategy_config`、`review_lane`

- `檢討整個流程 / 詳細分流`
  機器人動作：拆成資料收集、策略路由、模擬單、顧單、回測、檢討、優化七段

- `新幣到交易 / 任意幣丟進去 / 不要每隻幣改程式`
  機器人動作：優先跑 `openclaw-quantctl new-symbol-workflow --symbols SYMBOL --compact`，產出 `reject`、`research_candidate`、`near_ready_market_only`、`testnet_ready_candidate`，不為單一幣寫專用 Python

- `我時間沒有那多 / 不要一個指令一個動作`
  機器人動作：優先使用 `openclaw-quantctl mission --symbols ... --target-return-pct ... --max-leverage ...`，再用 optimizer 補策略收斂

## Prompt 資源

機器可讀資源：

- `config/operator-intent.default.yaml`
- `config/asset-routing.default.yaml`
- `config/mission-control.default.yaml`

CLI：

- `openclaw-quantctl route-intent "你的句子"`
- `openclaw-quantctl route-symbol SYMBOL`
- `openclaw-quantctl new-symbol-workflow --symbols SYMBOL --compact`
- `openclaw-quantctl mission --symbols BTCUSDT,ETHUSDT,XAU --target-return-pct 8 --max-leverage 3`
