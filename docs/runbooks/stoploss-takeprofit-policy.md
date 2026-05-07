# Stop-Loss Take-Profit Policy

目的：讓每筆交易的 SL / TP 管理可審計、可回放、低 token。

## 核心規則

- 每筆交易建立時就必須帶保護邏輯。
- 停損以帳戶可承受虧損金額為主，不以槓桿為主。
- TP 與 trailing 只能往保護收益方向演進，不能讓風險變大。

## 標準節奏

1. 建倉前確認：
   - `max_account_risk_pct`
   - stop distance
   - exchange minimum notional
2. 建倉後記錄：
   - entry
   - stop-loss
   - take-profit
   - leverage
   - planned account risk
3. 到達 TP1 行為：
   - 停損移到 breakeven 或 fee buffer 上方
4. 趨勢延續：
   - 保留 TP2
   - trailing 只做小幅收斂
5. 趨勢失真：
   - 收緊停損
   - 不擴大風險

## 異常處理

- 保護單缺失：先修保護，再談策略。
- 交易所最小 notional 與風控 sizing 衝突：放棄該單。
- 保護單與 journal 不一致：記錄 incident，再人工確認。
