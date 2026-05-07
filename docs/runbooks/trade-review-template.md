# Trade Review Template

每筆 TP / SL / manual close 後都要落一次 review。

## 必填欄位

- Symbol
- Side
- Opened at / Closed at
- Exit reason
- Realized PnL USDT / %
- Analysis score / bias / convergence
- Original stop-loss / take-profit
- 是否符合策略

## 復盤問題

1. 進場是否符合原始結構與規則？
2. 出場是策略內結果，還是執行 / 情緒 / 市場異常？
3. 風報比是否合理？
4. 是否有保護單錯誤、滑點、延遲、假突破？
5. 下一次只改哪一個參數或規則？

## 輸出格式

- `what_happened`: 一句話描述結果
- `root_cause`: 結構 / 風控 / 執行 / 市場雜訊
- `keep`: 下次維持不變的規則
- `change_next`: 下次只改一個規則或參數
- `confidence`: low / medium / high
