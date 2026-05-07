# Position Management

目的：把小倉 / 中倉 / 大倉、風險上限、移動停損與減倉規則固定成可執行流程。

## 原則

- 先算停損金額，再決定倉位。
- 不因槓桿看起來低就放大真實風險。
- 不加碼救單，不放寬原始停損。
- 有 open position 時，優先保護已存在風險，不追新的 setup。

## 倉位分級

- 小倉：試單 / 探路 / 結構未完全確認。
- 中倉：標準主力倉位，結構與風報比都達標。
- 大倉：高共振高延續 setup 才允許，且監控更嚴格。

## 操作順序

1. `openclaw-quantctl positions --compact`
2. `openclaw-quantctl account --market futures --compact`
3. `openclaw-quantctl journal-summary`
4. 若持倉已關閉，先 `openclaw-quantctl review-closed-trades --compact`
5. 若仍有持倉，只能走保護與檢討，不得開第二筆同向風險

## 允許的保護動作

- 到達 TP1 行為後，停損只允許往 breakeven 或鎖利方向移動。
- 趨勢轉弱時，優先收緊停損，不擴大風險。
- 若保護單遺失、重複、格式錯誤，立即停看修復，不做新單。

## 禁止事項

- 沒有 stop-loss / take-profit 的裸單。
- 平均攤平虧損倉位。
- 因情緒把停損往更遠處改。
- 因臨時對話直接改底層 execution 規則。
