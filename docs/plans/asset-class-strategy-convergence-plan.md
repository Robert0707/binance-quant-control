# Asset-Class Strategy Convergence Plan

> **For Hermes:** 依照本計劃逐步實作「不同資產類型走不同策略，但全部先進模擬單，直到結果收斂」的量化工作流。任何 live 執行路徑都不得被打開。

**Goal:** 為 BTC、ETH、黃金等不同資產類型建立分流策略，所有標的先以 paper/trade journal 模擬單持續驗證，直到 walk-forward / test 結果收斂後才考慮下一階段。

**Architecture:** 採用「資產分類器 → 策略路由器 → 模擬單執行 → 復盤與收斂評估」四層。BTC 類高波動資產、ETH 類中高波動趨勢資產、黃金類 session / macro 資產各自對應不同策略模板；live 執行維持關閉，僅允許 paper-order、backtest、review-closed-trades 與分析輸出。

**Tech Stack:** Python 3, `binance-quant-control`, `openclaw-quantctl`, local `paper-order`, `strategy_analyzer`, `backtest`, `backtest-sweep`, YAML config, JSON state files, pytest.

---

## 1. 目標定義

### 1.1 核心要求
1. **不同類型資產走不同策略**
   - BTC：高波動 / 假突破 / breakout-or-volatility lane
   - ETH：趨勢延續 / momentum lane
   - 黃金：session / mean-reversion / macro lane
2. **所有標的先模擬單**
   - 只允許 `paper-order` 與研究型 backtest
   - 不允許因優化而直接開 live
3. **結果要收斂才算可用**
   - train / test / walk-forward 不再明顯漂移
   - 收斂門檻需明確、可量化、可重跑
4. **保留交易邊界**
   - live execution boundaries 不改
   - kill-switch / pause-trading 繼續維持有效

### 1.2 成功標準
- 每個資產類型都有獨立策略模板
- 每個模板都有對應 paper 流程與回測驗證
- 任一模板若未收斂，仍只能留在 paper / research lane
- 不產生任何未授權 live 下單

---

## 2. 策略分流設計

### 2.1 資產分類器
建立一個 deterministic classifier，輸入 symbol 後輸出資產類型：

- `btc_like`
- `eth_like`
- `macro_gold_like`
- `alt_trend_like`
- `alt_mean_reversion_like`
- `unknown`

### 2.2 路由原則
- `btc_like` → breakout / volatility / fakeout filter
- `eth_like` → trend continuation / momentum filter
- `macro_gold_like` → session-aware / pullback / mean-reversion filter
- `unknown` → default conservative paper-only lane

### 2.3 路由輸出
路由器要回傳：
- `strategy_profile`
- `market`
- `interval`
- `paper_only=true`
- `convergence_gate`
- `risk_gate`

---

## 3. 收斂定義

### 3.1 必要條件
策略只有在以下條件持續成立時才算收斂候選：
- test set 交易數達到最低門檻
- walk-forward 各窗口表現穩定
- profit factor、win rate、average R 不再大幅飄移
- losing streak 在可接受範圍內

### 3.2 建議門檻
初版採用保守標準：
- test trades >= 5
- walk-forward windows >= 3
- test win rate >= 70%
- test profit factor >= 1.2
- walk-forward mean PF >= 1.1
- 最後兩輪參數改動後指標變動幅度 < 5%

### 3.3 不合格條件
以下任一條成立就繼續留在 paper：
- test 有明顯負期望
- walk-forward 明顯不穩
- 不同市場區段結果分裂
- 策略對單一行情片段過度依賴

---

## 4. 實作任務拆分

### Task 1: Define asset classes and routing rules
**Objective:** 建立資產分類與策略路由規則。

**Files:**
- Create: `src/binance_quant_control/asset_classification.py`
- Create: `tests/test_asset_classification.py`
- Modify: `src/binance_quant_control/strategy_analyzer.py`
- Modify: `src/binance_quant_control/analysis.py`

**Step 1: Write failing test**
```python
from binance_quant_control.asset_classification import classify_symbol

def test_classify_btc_eth_gold():
    assert classify_symbol("BTCUSDT") == "btc_like"
    assert classify_symbol("ETHUSDT") == "eth_like"
    assert classify_symbol("GC=F") == "macro_gold_like"
```

**Step 2: Run test to verify failure**
```bash
cd /home/robert/python/projects/binance-quant-control
.venv/bin/python -m pytest tests/test_asset_classification.py -v
```
Expected: FAIL — module/function missing.

**Step 3: Implement minimal classifier**
- BTC/ETH/GOLD 使用固定 mapping
- 其餘 symbol 用 deterministic fallback

**Step 4: Verify pass**
```bash
.venv/bin/python -m pytest tests/test_asset_classification.py -v
```

---

### Task 2: Create strategy templates per asset class
**Objective:** 每個資產類型對應一個獨立策略模板，不共用一個過度泛化的參數集。

**Files:**
- Create: `config/strategy-btc-volatility.yaml`
- Create: `config/strategy-eth-trend.yaml`
- Create: `config/strategy-gold-session.yaml`
- Modify: `src/binance_quant_control/strategy.py`
- Modify: `tests/test_strategy_config.py`

**Design notes:**
- BTC：更高 ADX / breakout / volatility gating
- ETH：中高 ADX / trend continuation / tighter convergence
- 黃金：session-aware / lower intraday noise / selective entry

**Verification:**
```bash
.venv/bin/python -m pytest tests/test_strategy_config.py -v
```

---

### Task 3: Add paper-only route enforcement
**Objective:** 確保所有新策略 lane 都只能走模擬單，不得繞過 live boundary。

**Files:**
- Modify: `src/binance_quant_control/cli.py`
- Modify: `src/binance_quant_control/live_execution.py` if present in execution path
- Modify: `src/binance_quant_control/backtest.py`
- Modify: `src/binance_quant_control/paper_order.py` or existing paper-order entrypoint
- Test: `tests/test_live_guardrails.py`

**Expected behavior:**
- strategy router may produce paper signal
- live execution remains blocked unless explicitly enabled
- kill-switch still blocks entries

**Verification:**
```bash
.venv/bin/python -m pytest tests/test_live_guardrails.py -v
/home/robert/python/ops/openclaw-runtime/openclaw-quantctl trading-control-status --compact
```

---

### Task 4: Add convergence evaluator
**Objective:** 把「結果收斂」變成可計算的 deterministic 指標。

**Files:**
- Create: `src/binance_quant_control/convergence.py`
- Create: `tests/test_convergence.py`
- Modify: `src/binance_quant_control/strategy_optimizer.py`
- Modify: `scripts/backtest_sweep.py`

**Convergence signals:**
- train/test PF gap
- walk-forward PF variance
- win-rate variance
- loss streak stability
- parameter stability across re-runs

**Verification:**
```bash
.venv/bin/python -m pytest tests/test_convergence.py -v
```

---

### Task 5: Make paper-order the default simulation lane
**Objective:** 所有候選策略先產生 paper 單與 journal 記錄，再做復盤。

**Files:**
- Modify: `src/binance_quant_control/cli.py`
- Modify: `src/binance_quant_control/order_journal.py`
- Modify: `src/binance_quant_control/daily_digest.py`
- Test: `tests/test_paper_order.py`

**Verification:**
```bash
/home/robert/python/ops/openclaw-runtime/openclaw-quantctl paper-order --help
.venv/bin/python -m pytest tests/test_paper_order.py -v
```

---

### Task 6: Add per-asset backtest lanes
**Objective:** BTC / ETH / gold 各自用自己的 sweep 參數與驗收門檻。

**Files:**
- Modify: `scripts/backtest_sweep.py`
- Modify: `src/binance_quant_control/strategy_optimizer.py`
- Modify: `config/strategy-optimizer.default.yaml`

**Lane ideas:**
- BTC lane: breakout / volatility regime
- ETH lane: trend continuation
- gold lane: session-aware / mean-reversion

**Verification:**
```bash
.venv/bin/python scripts/backtest_sweep.py
```

---

### Task 7: Document the operating rule
**Objective:** 把這條策略邊界寫進 workflow 文件，避免之後反覆口頭重講。

**Files:**
- Modify: `docs/workflows/strategy-review-automation.md`
- Create or modify: `docs/workflows/multi-asset-paper-first-workflow.md`

**Rule to document:**
- 不同資產走不同策略
- 所有策略先 paper
- 未收斂不得升級 live
- 每次 close trade 都要 review

---

## 5. 驗收清單

### 5.1 功能驗收
- [ ] BTC / ETH / 黃金各自有不同策略 lane
- [ ] 所有 lane 先走 paper-order
- [ ] live execution 邊界未被打開
- [ ] 收斂 evaluator 可重複計算
- [ ] 每個 lane 都有對應回測與復盤輸出

### 5.2 安全驗收
- [ ] `kill-switch` 仍有效
- [ ] `live_trading_enabled=false` 預設不變
- [ ] `paper-only=true` 預設不變
- [ ] 未通過收斂門檻不准升級

### 5.3 品質驗收
- [ ] 新增測試覆蓋策略分類與收斂判斷
- [ ] 所有變更可透過本地 CLI 驗證
- [ ] 回測輸出保持 compact 可讀

---

## 6. 預期執行順序
1. 先做資產分類與路由
2. 再做三條策略模板
3. 再加收斂評估
4. 再把 paper-order 設為預設驗證路
5. 最後補文件與測試

---

## 7. 結論
這條路線的核心不是「一個策略打天下」，而是：

- BTC 用 BTC 的打法
- ETH 用 ETH 的打法
- 黃金用黃金的打法
- 全部先 paper
- 直到數據收斂才允許往下一步

> **重要約束：在任何情況下，不因策略優化而放寬 live execution boundary。**
