# Freqtrade Engine Integration

目標：把 `freqtrade/freqtrade` 當成成熟交易引擎核心，讓 OpenClaw 這側只負責：

- 選幣與研究
- 風控 gate
- 排程與報表
- 硬體 / n8n / 通知整合

## 目前接法

- 官方 repo：`/home/robert/python/external/freqtrade`
- 官方 Docker image：`freqtradeorg/freqtrade:stable`
- 本機 wrapper：`/home/robert/python/bin/openclaw-freqtradectl`
- 本機 config：`/home/robert/python/external/freqtrade/user_data/config.openclaw.json`

## 為什麼選 Freqtrade 當核心

- 高星且長期維護
- 原生包含 dry-run / backtesting / hyperopt / plot-profit / lookahead-analysis / recursive-analysis
- 官方支援 Binance spot 與 futures
- 官方文件對 ARM64 明確建議走 Docker，符合這台 Pi 的穩定路線

## 跟現有 OpenClaw 量化面怎麼分工

### OpenClaw 保留

- `daily_digest.py`
- `n8n` digest workflow
- challenge / journal / growth report
- 系統 readiness / health / hardware wrappers

### Freqtrade 接手

- 交易引擎
- 回測命令
- data download
- strategy runtime
- 後續若 auth 穩定，可接 dry-run / trade

## 目前使用方式

### 基本健康檢查

```bash
/home/robert/python/bin/openclaw-freqtradectl health
```

### 用最新 digest 候選幣更新 Freqtrade whitelist

```bash
/home/robert/python/bin/openclaw-freqtradectl sync-whitelist
```

### 列出可用策略

```bash
/home/robert/python/bin/openclaw-freqtradectl list-strategies
```

### 下載資料

```bash
/home/robert/python/bin/openclaw-freqtradectl download-data --pairs NEAR/USDT:USDT DOGE/USDT:USDT -t 5m --days 14
```

### 跑回測

```bash
/home/robert/python/bin/openclaw-freqtradectl backtesting --timerange 20260101- --export trades
```

### 跑完整官方引擎 workflow

```bash
/home/robert/python/bin/openclaw-freqtrade-workflow --days 7
```

這條 workflow 會做：

- 重新產出最新 digest
- 把 digest 候選幣同步到 Freqtrade whitelist
- 下載官方引擎需要的歷史資料
- 跑一次官方 backtesting
- 更新 growth report

## 收斂原則

- 不直接把 Freqtrade live 打開
- 先用它取代「手搓交易核心」的部分
- 只有當 Binance private auth 穩定、`live-readiness` 恢復、系統 load 正常，才評估進一步啟用 dry-run 常駐或 live
