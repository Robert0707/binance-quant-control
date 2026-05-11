# Binance Quant Control

目標：把 Binance 量化分析、paper trading、testnet 驗證、OpenClaw workflow，收斂成可長期維護的安全控制面。

設計原則：
- 預設只做分析、回測思維、paper/testnet，不直接碰 mainnet live 下單。
- 所有交易建議都要先出分析報表、風控摘要、再由 workflow 或操作者決策。
- Binance API 用官方簽名規格直連，Blave alpha 當作可選加強，不綁死在單一資料源。
- Hailo/AI HAT 目前只保留 chart/image artifact hook；沒有合適的金融時序 HEF 之前，不做虛假的 edge inference 宣稱。

主要入口：
- `openclaw-quantctl doctor`
- `openclaw-quantctl env-template`
- `openclaw-quantctl review-closed-trades --compact`
- `openclaw-quantctl journal-summary`
- `python3 scripts/run_strategy_optimizer.py --config config/strategy-optimizer.default.yaml`
- professional entry / review workflow: `docs/workflows/professional-entry-and-review-workflow.md`
- strategy-only review lane: `docs/workflows/strategy-review-automation.md`
- senior-trader doctrine: `docs/workflows/github-inspired-trading-doctrine.md`
- `python3 scripts/run_autonomous_trader.py --config config/autonomous-trader.default.yaml`
- `python3 scripts/run_autonomous_trader.py --config config/autonomous-testnet-explorer.default.yaml --compact`
- `python3 scripts/run_autonomous_trader.py --config config/autonomous-guardian.default.yaml`
- `python3 scripts/run_autonomous_trader.py --config config/autonomous-simulation-lane.default.yaml`
- `python3 scripts/run_strategy_optimizer.py --config config/strategy-optimizer.default.yaml`
- `openclaw-quantctl analyze BTCUSDT --market futures --interval 1h --use-blave --render-chart`
- `openclaw-quantctl backtest --strategy-config config/strategy-live-pilot.yaml`
- `openclaw-quantctl backtest --strategy-config config/strategy-hermes-pro.yaml`
- `openclaw-quantctl backtest-sweep`
- `openclaw-quantctl alpha-research --config config/aggressive-alpha-research.default.yaml --compact`
- `python3 scripts/research_ocean_x_btc_evidence.py --optimize-core-whale-jump --symbols BTCUSDT,ETHUSDT,XAUTUSDT --interval 1h --target-win-rate 80 --min-train-trades 70 --min-test-trades 30 --min-profit-factor 1.5 --max-stop-loss-ratio 20 --max-configs 192`
- `openclaw-quantctl high-win-iteration --alpha-report state/core-10-mainstream-boundary-l1500-smoke/alpha-research-ranking.json --compact`
- `openclaw-quantctl high-win-converge --alpha-report state/core-10-mainstream-boundary-l1500-smoke/alpha-research-ranking.json --max-rounds 1 --compact`
- `openclaw-quantctl repository-audit --compact`
- `openclaw-quantctl professional-system-audit --compact`
- `openclaw-quantctl hermes-ai-trader --compact`
- `openclaw-quantctl challenge-init --strategy-config config/strategy-live-pilot.yaml --from-account`
- `openclaw-quantctl challenge-status --strategy-config config/strategy-live-pilot.yaml --refresh`
- `openclaw-quantctl build-analysis-spec BTCUSDT --market futures --interval 1h --use-blave --render-chart`
- `openclaw-quantctl submit-analysis BTCUSDT --market futures --interval 1h --use-blave --render-chart --run-now`
- `python3 scripts/submit_research_pack.py --scheduled --config config/research-pack.default.yaml`
- `openclaw-quantctl account --market spot|futures`
- `openclaw-quantctl positions --symbol BTCUSDT`
- `openclaw-quantctl paper-order BTCUSDT --market futures --side BUY --notional-usdt 200 --leverage 3`
- `openclaw-quantctl route-symbol BTCUSDT`
- `openclaw-quantctl route-intent "我加入了模擬單，測試新策略要先進去模擬單做測試"`
- `openclaw-quantctl mission --symbols BTCUSDT,ETHUSDT,XAU --target-return-pct 8 --max-leverage 3`
- `openclaw-quantctl delivery-supervisor --cycles 50 --training-rounds 10 --compact`
- `openclaw-quantctl live-readiness --strategy-config config/strategy-live-pilot.yaml`
- `openclaw-quantctl live-readiness --strategy-config config/strategy-hermes-pro.yaml`
- `openclaw-quantctl live-pilot --strategy-config config/strategy-live-pilot.yaml --execute`
- `openclaw-quantctl operator-dashboard --compact`
- `openclaw-quantctl external-context-key-status --compact`
- `openclaw-quantctl external-context --symbols BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,DOGEUSDT,TRXUSDT --compact`
- `openclaw-quantctl repair-staged-tp --strategy-config config/strategy-major-alt-trend.yaml --symbol APTUSDT --side BUY --execute --compact`

操作邊界：
- 排程自動化只允許 closed-trade review 與策略檢討，不允許定時 `submit-analysis`。
- `openclaw-quantctl analyze` / `submit-analysis` 是操作者手動研究入口，不是排程背景工作。
- `review-closed-trades` 是預設策略優化入口；它只回寫檢討資料，不改 execution 邏輯或 `live_trading` 底層設定。
- `strategy-optimizer` 只允許改 `config/strategy-hermes-pro.auto.yaml` 內白名單策略欄位：
  `risk.min_convergence`、`risk.min_score_long`、`risk.max_score_short`、`risk.cooldown_hours`、`risk.atr_stop_multiple`、`risk.min_adx`、`risk.trailing_callback_pct`、`risk.take_profit_r_multiples`。
- `strategy-optimizer` 不允許改 `defaults.*`、`execution.*`、`challenge.*`、`max_account_risk_pct`、槓桿上限、或任何 `BINANCE_LIVE_TRADING_ENABLED` 相關底層行為。
- `professional_entry_gate` 是正式 live lane 的上層入場制度；只做 gate / block / warning，不送單、不改下單底層。
- `trading_domain.py`、`portfolio_construction.py`、`feature_registry.py`、`signal_api.py`、`skipped_signal_journal.py` 是專業架構重構核心：broker-neutral 交易物件、組合風險目標、feature/label manifest hash、signal ledger 與拒單/跳過訊號記錄，尚未自動放行交易。
- `hermes-ai-trader` 是新的乾淨 AI Trader v2 控制面：標準 signal schema、event/plugin lifecycle、feature/model registry、structured committee、portfolio target/risk snapshot、Hailo 分工與 open-order gate；它不送單，只決定是否可進下一層 readiness。
- `delivery-supervisor` 是交付前長跑監督入口；只允許 paper/demo/market-replay，不會呼叫 live execution。若 optimizer 尚未 `promote` / `elite_candidate`，所有 live plan 也會被全局 promotion gate 擋下。
- `alpha-research` 是 aggressive research lane；只允許 backtest / paper / testnet 評估，`mainnet_live_allowed` 必須保持 false。它會用多策略族、slippage 情境、walk-forward robustness、out-of-sample return/drawdown 排名候選，不會直接升級到 mainnet live。

交付物：
- 公開市場資料分析引擎
- Binance read-only account / positions 檢查
- 可執行回測引擎
- paper-order journal
- live execution planner / risk gate / order journal
- funded challenge tracker with balance snapshots and drawdown stop
- OpenClaw taskctl workflow builder
- strategy-review-only scheduled guardrail lane
- `.env.example` + `strategy.example.yaml`
- venv 安裝腳本

執行證據落點：
- reports: `projects/binance-quant-control/reports/`
- state: `projects/binance-quant-control/state/`
- workflow specs: `~/.openclaw/runtime/binance-quant/task-specs/`
- workflow runs: `~/.openclaw/runtime/binance-quant/runs/`

Hermes / 維運文件：
- runbooks: `docs/runbooks/`
- external context API key setup: `docs/runbooks/external-context-api-keys.md`
- templates: `docs/templates/`
- Hermes 優化主計劃：`docs/plans/hermes-optimization-plan.md`
- 全自動 AI 交易員 workflow：`docs/workflows/autonomous-ai-trader.md`

安全邊界：
- `BINANCE_USE_TESTNET=true` 預設開啟
- `BINANCE_LIVE_TRADING_ENABLED=false` 預設關閉
- live trading 不做預設 workflow；需要額外明確開關與後續專門驗證
- `doctor` 需要同時通過 public connectivity 與 private API auth 檢查，才可考慮開 live
- `live-readiness` 只產生可執行計畫，不直接下單；`live-pilot --execute` 才會走真單提交
- 若啟用 challenge mode，系統會持續記錄 balance snapshots，並在達標或跌破回撤底線時自動封鎖新單
- 不處理提現權限，不建議 API key 開 withdraw

Hermes 交易員模式：
- `config/strategy-stable-risk.yaml` 是保守生存基線
- `config/strategy-hermes-pro.yaml` 是目前推薦的專業交易員模式
- `config/autonomous-trader.default.yaml` 是本地程式優先、關鍵判斷才交給 AI 的自動駕駛配置
- `config/autonomous-testnet-explorer.default.yaml` 是積極 Binance futures testnet 探索入口：前 60 名成交量、新聞/巨鯨觀測、逐幣動態槓桿與艙位、最多 5 筆 testnet 持倉
- `config/aggressive-alpha-research.default.yaml` 是高收益 alpha 掃描研究線：可動態引入 top futures quote-volume universe，使用 `config/strategy-aggressive-alpha-research.yaml` 的寬鬆研究假設，但結果只能進 backtest / paper / testnet 淘汰流程。
- `operator-dashboard` 是客戶視角的產品回饋入口：目前是否賺錢、保護單是否完整、歷史虧損主因、下一步調整建議；不呼叫大模型
- `repair-staged-tp` 可把舊式全倉 TP / trailing 保護重建為 stop-loss + 風控分段 TP，避免 TP1 一到全平
- `config/autonomous-guardian.default.yaml` 是高頻本地看單 / 保護單 guardian 配置
- `config/strategy-hermes-pro.auto.yaml` 會由策略收斂器自動生成
- `config/asset-routing.default.yaml` 是幣種分類與策略路由總表
- `config/operator-intent.default.yaml` 是操作者語句到機器人動作的提示詞資源
- `config/mission-control.default.yaml` 是一個指令總控與排程邊界配置
- `config/official-strategy-baselines.yaml` 是 BTC / ETH / XAU 公開策略基底來源
- `config/mainstream-risk-boundaries.default.yaml` 是主流交易 bot 風控邊界基準：exchange filters、pre-trade risk、triple-barrier、drawdown / stoploss guard、liquidity/funding/news veto。
- `config/high-win-iteration.default.yaml` 是嚴格 100 筆 / 80% 勝率 / <=20% 純停損 / PF>=1.5 的研究迭代控制器；只輸出下一輪研究命令與 gate，不開單、不改 live execution。
- `config/ocean-x-btc-eth-optimizer.default.yaml` 是 BTC/ETH TradingView-inspired / whale-jump proxy 收斂研究線；目前已改成 expectancy-first gate，不再用 `勝率>80%` 當開單硬條件。它只跑公開資料 train/test/walk-forward gate，不寫 live strategy、不開單。2026-05-04 完整 1h high-win 對照報告為 `reports/20260504T-btc-eth-tradingview-convergence-full/optimizer_summary.json`，結果 `candidate_count=0`；expectancy/R:R 對照報告為 `reports/20260504T-btc-eth-tradingview-expectancy-rr-1h-full/optimizer_summary.json`，仍未准 live，因目前候選卡在樣本、payoff 或 walk-forward expectancy。
- `high-win-converge` 是有上限的持續收斂 loop；預設 plan-only，加 `--execute-research` 才跑 backtest/research 批次，仍不開單、不啟用 mainnet。
- `repository-audit` 是專案檔案與架構盤點入口；預設跳過 `.env`、`state/`、`reports/`、快取與生成物，避免把秘密或噪音當成架構本體。
- `professional-system-audit` 是參考 TradingAgents / Lumibot / OctoBot / intelligent-trading-bot / AI-Trader 與主流風控後的專業交易系統成熟度 gate；只檢查架構、alpha 證據與 promotion blockers，不送單、不改 live 設定。
- `docs/architecture/repository-map.md` 是目前檔案邊界與大型模組壓力點索引。
- `docs/architecture/program-pruning-audit.md` 是架構清理與保留/刪除依據；目前刪除舊式 `targeted_volatile_sweep.py`，保留 n8n/Freqtrade/service 人工入口。
- `docs/architecture/professional-trading-system-blueprint.md` 是從找幣到交易的目標架構與 keep/refactor/rebuild 清單。
- `docs/architecture/hermes-ai-trader-v2.md` 是 Hermes AI Trader v2 的乾淨架構與 Hailo 分工。
- `docs/workflows/hermes-ai-trader-v2-workflow.md` 是一鍵 gate 工作流。
- `docs/workflows/market-bot-expectancy-research-pipeline.md` 是市面成熟 bot 風格的正期望值研究管線：feature dataset -> alpha research -> market-bot-gate -> Hermes gate -> live-readiness。
- `docs/workflows/new-symbol-to-trade-pipeline.md` 是從任意新幣到 paper/testnet 交易資格的完整流水線：intent -> route -> feature dataset -> expectancy research -> risk-combo matrix -> Hermes AI Trader -> readiness -> decision contract -> operator dashboard；不啟用 mainnet。
- `docs/workflows/professional-trading-lifecycle.md` 是專業交易生命週期的可重複工作流。
- `docs/workflows/strategy-convergence-validation.md` 是正式收斂 gate 與 cohort 規範
- `docs/workflows/mainstream-bot-risk-boundaries.md` 是這套邊界如何用在本專案的工作流
- `docs/workflows/one-command-mission-control.md` 是一個指令總控的使用方式
- `openclaw-quantctl delivery-supervisor --cycles 50 --training-rounds 10 --compact` 是 500 次 paper/demo 交付訓練的推薦入口
- 外部上下文不是只有價格：
  - 新聞風險
  - 巨鯨流向
  - GitHub 交易基礎設施觀測
  都會進入 digest 與排序決策
