from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlparse


@dataclass(frozen=True)
class AnalysisResult:
    verdict: str
    bias: str
    confidence: float
    regime: str
    notes: list[str]

    def as_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict,
            "bias": self.bias,
            "confidence": round(self.confidence, 3),
            "regime": self.regime,
            "notes": self.notes,
        }


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _compact_text(value: Any) -> str:
    text = str(value).strip()
    return text if text else "unknown"


def analyze_strategy_payload(payload: dict[str, Any]) -> dict[str, Any]:
    decision = payload.get("decision") or {}
    selected = decision.get("selected") or payload.get("selected") or {}
    news_risk = ((payload.get("news") or {}).get("risk") or {})
    whale = payload.get("whale") or {}
    doctor = payload.get("doctor") or {}
    github_summary = payload.get("github_summary") or {}

    direction = _compact_text(selected.get("direction") or decision.get("direction") or "neutral")
    symbol = _compact_text(selected.get("symbol") or payload.get("symbol") or "unknown")
    action = _compact_text(decision.get("action") or "stand_by")
    news_level = _compact_text(news_risk.get("risk_level") or "normal")
    whale_signal = _compact_text(whale.get("signal") or "neutral")
    doctor_state = _compact_text(doctor.get("overall") or "unknown")
    github_signal = _compact_text(github_summary.get("signal") or "mixed")

    score = float(selected.get("adjusted_score") or selected.get("composite_score") or selected.get("analysis_score") or 50.0)
    confidence = _clamp(score / 100.0)

    notes: list[str] = []
    if news_level == "high":
        confidence -= 0.18
        notes.append("high-event-risk")
    elif news_level == "elevated":
        confidence -= 0.08
        notes.append("elevated-news-risk")

    if whale_signal != "neutral":
        if (direction == "long" and whale_signal == "bullish") or (direction == "short" and whale_signal == "bearish"):
            confidence += 0.07
            notes.append("whale-aligned")
        else:
            confidence -= 0.07
            notes.append("whale-opposed")

    if doctor_state == "ok":
        confidence += 0.03
        notes.append("doctor-ok")
    elif doctor_state not in {"ok", "unknown"}:
        notes.append(f"doctor-{doctor_state}")

    if github_signal == "active":
        confidence += 0.02
        notes.append("github-observability-active")
    elif github_signal == "stale":
        confidence -= 0.03
        notes.append("github-observability-stale")

    if action in {"no_trade", "stand_by"}:
        verdict = "watch"
        regime = "risk_off" if news_level == "high" else "watchlist"
    elif action == "watchlist_only":
        verdict = "approve" if confidence >= 0.7 and news_level != "high" else "watch"
        regime = "aligned" if verdict == "approve" else "mixed"
    else:
        verdict = "approve" if confidence >= 0.55 else "watch"
        regime = "aligned" if confidence >= 0.6 else "mixed"

    if not selected:
        notes.append("no-selected-candidate")
        verdict = "watch"
        regime = "insufficient-data"
        confidence = min(confidence, 0.4)

    result = AnalysisResult(
        verdict=verdict,
        bias=direction,
        confidence=_clamp(confidence),
        regime=regime,
        notes=notes or ["compact-analysis-complete"],
    )
    return {
        "status": "ok",
        "symbol": symbol,
        "result": result.as_dict(),
    }


class StrategyAnalyzerHandler(BaseHTTPRequestHandler):
    server_version = "StrategyAnalyzer/1.0"

    def log_message(self, format: str, *args: Any) -> None:  # pragma: no cover
        return

    def _send_json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # pragma: no cover
        path = urlparse(self.path).path
        if path in {"/", "/health"}:
            self._send_json(200, {"status": "ok", "service": "strategy-analyzer"})
            return
        self._send_json(404, {"status": "error", "error": "not_found"})

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        if path != "/analyze":
            self._send_json(404, {"status": "error", "error": "not_found"})
            return
        length = int(self.headers.get("Content-Length", "0") or 0)
        raw = self.rfile.read(length).decode("utf-8") if length else "{}"
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            self._send_json(400, {"status": "error", "error": "invalid_json"})
            return
        if not isinstance(payload, dict):
            self._send_json(400, {"status": "error", "error": "expected_json_object"})
            return
        response = analyze_strategy_payload(payload)
        self._send_json(200, response)


def run_server(host: str, port: int) -> None:
    server = ThreadingHTTPServer((host, port), StrategyAnalyzerHandler)
    print(json.dumps({"status": "listening", "host": host, "port": port, "endpoint": f"http://{host}:{port}/analyze"}, separators=(",", ":")), flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:  # pragma: no cover
        pass
    finally:
        server.server_close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the compact strategy analyzer service.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8799)
    args = parser.parse_args()
    run_server(args.host, args.port)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
