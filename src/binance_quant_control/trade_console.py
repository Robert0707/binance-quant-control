from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs, urlparse

from .binance_api import BinanceAPIError, BinanceClient
from .config import Settings, load_settings
from .hermes_trade_loop import hermes_trade_status, run_hermes_trade_cycle
from .operator_dashboard import build_operator_dashboard
from .order_journal import read_closed_trade_reviews
from .trade_session import start_trade_session, stop_trade_session, trade_session_status


@dataclass(frozen=True, slots=True)
class TradeConsoleConfig:
    host: str = "127.0.0.1"
    port: int = 8765
    allow_order_actions: bool = False


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def build_equity_curve(limit: int = 200) -> list[dict[str, Any]]:
    rows = read_closed_trade_reviews()
    curve: list[dict[str, Any]] = []
    cumulative = 0.0
    for item in rows[-max(int(limit), 1) :]:
        pnl = _float(item.get("realized_pnl_usdt"))
        cumulative += pnl
        curve.append(
            {
                "timestamp": item.get("closed_at") or item.get("reviewed_at"),
                "symbol": item.get("symbol"),
                "side": item.get("side"),
                "realized_pnl_usdt": round(pnl, 8),
                "cumulative_pnl_usdt": round(cumulative, 8),
                "exit_reason": item.get("exit_reason"),
            }
        )
    return curve


def build_trade_console_snapshot(settings: Settings | None = None) -> dict[str, Any]:
    settings = settings or load_settings()
    dashboard = build_operator_dashboard(settings)
    return {
        "generated_at": _utc_now_iso(),
        "mode": {
            "use_testnet": settings.use_testnet,
            "live_trading_enabled": settings.live_trading_enabled,
            "testnet_trading_enabled": settings.testnet_trading_enabled,
            "mainnet_live_allowed": False,
        },
        "dashboard": {
            "status": dashboard.get("status"),
            "customer_summary": dashboard.get("customer_summary"),
            "positions": dashboard.get("positions") or [],
            "protective_orders": dashboard.get("protective_orders") or [],
            "execution_journal": dashboard.get("execution_journal"),
            "product_readiness": dashboard.get("product_readiness"),
            "candidate_pool": dashboard.get("candidate_pool"),
            "risk_combo_matrix": dashboard.get("risk_combo_matrix"),
            "operator_feedback": dashboard.get("operator_feedback") or [],
            "report_path": dashboard.get("report_path"),
        },
        "trade_session": trade_session_status(),
        "hermes_trade": hermes_trade_status(),
        "equity_curve": build_equity_curve(),
        "controls": {
            "start": {"method": "POST", "path": "/api/action/start"},
            "stop": {"method": "POST", "path": "/api/action/stop"},
            "cycle": {"method": "POST", "path": "/api/action/cycle"},
            "close_position": {
                "method": "POST",
                "path": "/api/action/close-position",
                "requires": ["symbol", "confirm=true"],
            },
        },
    }


def close_position_from_console(
    *,
    symbol: str,
    confirm: bool,
    settings: Settings | None = None,
) -> dict[str, Any]:
    if not confirm:
        return {
            "status": "blocked",
            "reason": "close-position-requires-confirm=true",
            "symbol": symbol.upper(),
        }
    settings = settings or load_settings()
    symbol_upper = symbol.strip().upper()
    if not symbol_upper:
        return {"status": "blocked", "reason": "missing-symbol"}

    with BinanceClient(settings) as client:
        raw_positions = client.positions(symbol_upper)
        position = next(
            (
                item
                for item in raw_positions
                if str(item.get("symbol") or "").upper() == symbol_upper
                and abs(_float(item.get("positionAmt"))) > 0
            ),
            None,
        )
        if not position:
            return {"status": "blocked", "reason": "position-not-found", "symbol": symbol_upper}
        qty = _float(position.get("positionAmt"))
        close_side = "SELL" if qty > 0 else "BUY"
        response = client.new_order(
            symbol_upper,
            close_side,
            "MARKET",
            quantity=abs(qty),
            reduce_only=True,
            market="futures",
        )
        cancelled_algo_orders = client.cancel_all_algo_orders(symbol_upper)
    return {
        "status": "submitted",
        "symbol": symbol_upper,
        "side": close_side,
        "quantity": abs(qty),
        "reduce_only": True,
        "market_close": response,
        "cancelled_algo_orders": cancelled_algo_orders,
    }


def run_trade_console_server(config: TradeConsoleConfig | None = None) -> ThreadingHTTPServer:
    config = config or TradeConsoleConfig()
    settings = load_settings()

    class Handler(BaseHTTPRequestHandler):
        server_version = "BinanceQuantTradeConsole/1.0"

        def log_message(self, format: str, *args: Any) -> None:
            return

        def _send_json(self, payload: dict[str, Any], status: int = HTTPStatus.OK) -> None:
            body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _send_html(self) -> None:
            body = TRADE_CONSOLE_HTML.encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _read_body(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length", "0") or 0)
            if length <= 0:
                return {}
            raw = self.rfile.read(length).decode("utf-8")
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                return {}
            return payload if isinstance(payload, dict) else {}

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path in {"/", "/console"}:
                self._send_html()
                return
            if parsed.path == "/api/snapshot":
                self._send_json(build_trade_console_snapshot(settings))
                return
            self._send_json({"status": "not_found", "path": parsed.path}, status=HTTPStatus.NOT_FOUND)

        def do_POST(self) -> None:
            parsed = urlparse(self.path)
            body = self._read_body()
            query = parse_qs(parsed.query)
            try:
                if parsed.path == "/api/action/start":
                    dry_run_only = not bool(body.get("execute_testnet", False))
                    self._send_json(
                        start_trade_session(
                            dry_run_only=dry_run_only,
                            reason=str(body.get("reason") or "trade console start"),
                        )
                    )
                    return
                if parsed.path == "/api/action/stop":
                    self._send_json(stop_trade_session(reason=str(body.get("reason") or "trade console stop")))
                    return
                if parsed.path == "/api/action/cycle":
                    self._send_json(
                        run_hermes_trade_cycle(
                            force=bool(body.get("force", True)),
                            execute_testnet_entries=bool(body.get("execute_testnet", True)),
                        )
                    )
                    return
                if parsed.path == "/api/action/close-position":
                    if not config.allow_order_actions:
                        self._send_json(
                            {"status": "blocked", "reason": "console-order-actions-disabled"},
                            status=HTTPStatus.FORBIDDEN,
                        )
                        return
                    symbol = str(body.get("symbol") or query.get("symbol", [""])[0])
                    confirm = bool(body.get("confirm", False))
                    self._send_json(close_position_from_console(symbol=symbol, confirm=confirm, settings=settings))
                    return
            except (BinanceAPIError, RuntimeError, ValueError) as exc:
                self._send_json({"status": "error", "error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                return
            self._send_json({"status": "not_found", "path": parsed.path}, status=HTTPStatus.NOT_FOUND)

    server = ThreadingHTTPServer((config.host, int(config.port)), Handler)
    return server


TRADE_CONSOLE_HTML = r"""<!doctype html>
<html lang="zh-Hant">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Binance Quant Console</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f6f7f9;
      --panel: #ffffff;
      --ink: #151922;
      --muted: #687080;
      --line: #d9dee7;
      --good: #0f8a5f;
      --bad: #b42318;
      --warn: #9a6700;
      --accent: #1b5f9e;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: var(--bg);
      color: var(--ink);
    }
    header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding: 14px 18px;
      border-bottom: 1px solid var(--line);
      background: var(--panel);
      position: sticky;
      top: 0;
      z-index: 2;
    }
    h1 { font-size: 18px; margin: 0; font-weight: 700; }
    main { padding: 16px; display: grid; gap: 14px; }
    .toolbar, .grid, .wide { display: grid; gap: 12px; }
    .toolbar { grid-template-columns: repeat(5, minmax(120px, 1fr)); }
    .grid { grid-template-columns: repeat(4, minmax(180px, 1fr)); }
    .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px;
      min-width: 0;
    }
    .metric { font-size: 24px; font-weight: 700; margin-top: 4px; }
    .label { color: var(--muted); font-size: 12px; }
    button {
      min-height: 38px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #fff;
      color: var(--ink);
      font-weight: 650;
      cursor: pointer;
    }
    button.primary { background: var(--accent); color: white; border-color: var(--accent); }
    button.danger { background: #fff4f2; color: var(--bad); border-color: #f1b8b2; }
    button:disabled { opacity: .55; cursor: wait; }
    table { width: 100%; border-collapse: collapse; font-size: 13px; }
    th, td { padding: 8px 6px; border-bottom: 1px solid var(--line); text-align: left; vertical-align: top; }
    th { color: var(--muted); font-size: 12px; font-weight: 650; }
    .good { color: var(--good); }
    .bad { color: var(--bad); }
    .warn { color: var(--warn); }
    .muted { color: var(--muted); }
    .feedback { margin: 0; padding-left: 18px; }
    .chart { width: 100%; height: 220px; display: block; border: 1px solid var(--line); border-radius: 6px; }
    pre { white-space: pre-wrap; overflow-wrap: anywhere; margin: 0; font-size: 12px; color: var(--muted); }
    @media (max-width: 900px) {
      .toolbar, .grid { grid-template-columns: 1fr 1fr; }
    }
    @media (max-width: 560px) {
      header { align-items: flex-start; gap: 8px; flex-direction: column; }
      .toolbar, .grid { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <header>
    <h1>Binance Quant Console</h1>
    <div class="muted" id="updated">--</div>
  </header>
  <main>
    <section class="toolbar">
      <button class="primary" onclick="act('start', {execute_testnet:true})">開始交易</button>
      <button onclick="act('start', {execute_testnet:false})">觀測模式</button>
      <button onclick="act('cycle', {force:true, execute_testnet:true})">跑一輪</button>
      <button class="danger" onclick="act('stop', {})">結束交易</button>
      <button onclick="refresh()">刷新</button>
    </section>

    <section class="grid">
      <div class="panel"><div class="label">模式</div><div class="metric" id="mode">--</div></div>
      <div class="panel"><div class="label">持倉</div><div class="metric" id="positionsCount">--</div></div>
      <div class="panel"><div class="label">未實現 PnL</div><div class="metric" id="openPnl">--</div></div>
      <div class="panel"><div class="label">候選通過</div><div class="metric" id="allowedCount">--</div></div>
    </section>

    <section class="panel">
      <div class="label">收益走勢</div>
      <canvas class="chart" id="equity" width="900" height="220"></canvas>
    </section>

    <section class="panel">
      <div class="label">持倉與平倉</div>
      <table>
        <thead><tr><th>幣</th><th>方向</th><th>數量</th><th>入場</th><th>標記</th><th>PnL</th><th>保護</th><th></th></tr></thead>
        <tbody id="positions"></tbody>
      </table>
    </section>

    <section class="panel">
      <div class="label">候選與 Gate</div>
      <pre id="candidates">--</pre>
    </section>

    <section class="panel">
      <div class="label">操作回饋</div>
      <ul class="feedback" id="feedback"></ul>
    </section>

    <section class="panel">
      <div class="label">最新動作</div>
      <pre id="lastAction">--</pre>
    </section>
  </main>
  <script>
    let busy = false;
    function cls(v) { return Number(v) >= 0 ? 'good' : 'bad'; }
    function fmt(v, n=4) {
      const x = Number(v);
      return Number.isFinite(x) ? x.toFixed(n) : '--';
    }
    async function post(path, body) {
      const res = await fetch('/api/action/' + path, {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(body || {})
      });
      return await res.json();
    }
    async function act(name, body) {
      if (busy) return;
      busy = true;
      document.querySelectorAll('button').forEach(b => b.disabled = true);
      try {
        const out = await post(name, body);
        document.getElementById('lastAction').textContent = JSON.stringify(out, null, 2);
        await refresh();
      } finally {
        busy = false;
        document.querySelectorAll('button').forEach(b => b.disabled = false);
      }
    }
    async function closePos(symbol) {
      if (!confirm('確認 reduce-only 平倉 ' + symbol + ' ?')) return;
      await act('close-position', {symbol, confirm: true});
    }
    function drawCurve(rows) {
      const c = document.getElementById('equity');
      const ctx = c.getContext('2d');
      ctx.clearRect(0, 0, c.width, c.height);
      ctx.strokeStyle = '#d9dee7';
      ctx.beginPath(); ctx.moveTo(35, 12); ctx.lineTo(35, 198); ctx.lineTo(880, 198); ctx.stroke();
      if (!rows || rows.length < 2) return;
      const vals = rows.map(r => Number(r.cumulative_pnl_usdt || 0));
      const min = Math.min(...vals), max = Math.max(...vals), span = Math.max(max - min, 0.0001);
      ctx.strokeStyle = '#1b5f9e'; ctx.lineWidth = 2; ctx.beginPath();
      vals.forEach((v, i) => {
        const x = 35 + (i / (vals.length - 1)) * 845;
        const y = 198 - ((v - min) / span) * 170;
        if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      });
      ctx.stroke();
      ctx.fillStyle = '#687080';
      ctx.fillText('max ' + max.toFixed(2), 42, 22);
      ctx.fillText('min ' + min.toFixed(2), 42, 194);
    }
    async function refresh() {
      const res = await fetch('/api/snapshot', {cache: 'no-store'});
      const data = await res.json();
      const d = data.dashboard || {};
      const s = d.customer_summary || {};
      document.getElementById('updated').textContent = data.generated_at || '--';
      document.getElementById('mode').textContent = data.mode && data.mode.use_testnet ? 'testnet' : 'mainnet-blocked';
      document.getElementById('positionsCount').textContent = s.open_position_count ?? 0;
      const pnl = Number(s.open_unrealized_pnl_usdt || 0);
      const pnlEl = document.getElementById('openPnl');
      pnlEl.textContent = fmt(pnl, 4);
      pnlEl.className = 'metric ' + cls(pnl);
      const cp = d.candidate_pool || {};
      document.getElementById('allowedCount').textContent = cp.readiness_allowed_count ?? 0;
      const protection = {};
      (d.protective_orders || []).forEach(p => protection[p.symbol] = p.coverage);
      document.getElementById('positions').innerHTML = (d.positions || []).map(p => {
        const val = Number(p.unrealized_pnl_usdt || 0);
        return `<tr><td>${p.symbol}</td><td>${p.side}</td><td>${p.quantity}</td><td>${p.entry_price}</td><td>${p.mark_price}</td><td class="${cls(val)}">${fmt(val, 4)}</td><td>${protection[p.symbol] || '--'}</td><td><button class="danger" onclick="closePos('${p.symbol}')">平倉</button></td></tr>`;
      }).join('') || '<tr><td colspan="8" class="muted">無持倉</td></tr>';
      const risk = d.risk_combo_matrix || {};
      document.getElementById('candidates').textContent = JSON.stringify({
        next_action: cp.next_action,
        readiness_allowed_count: cp.readiness_allowed_count,
        missing_horizons: cp.missing_horizons,
        risk_combo_status: risk.status,
        side_summary: risk.side_summary,
        product_readiness: d.product_readiness
      }, null, 2);
      document.getElementById('feedback').innerHTML = (d.operator_feedback || []).map(x => `<li>${x}</li>`).join('');
      drawCurve(data.equity_curve || []);
    }
    refresh();
    setInterval(refresh, 15000);
  </script>
</body>
</html>
"""
