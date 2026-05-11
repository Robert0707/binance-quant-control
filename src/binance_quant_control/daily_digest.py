from __future__ import annotations

import json
import os
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .asset_routing import resolve_symbol_route
from .candidate_universe import fetch_top_futures_symbols
from .config import PROJECT_ROOT, STATE_DIR, ensure_runtime_dirs, load_settings
from .external_context import build_external_context
from .side_risk_policy import evaluate_route_side_risk
from .signal_scoring import build_signal_scores

QUANTCTL = Path("/home/robert/.openclaw/bin/openclaw-quantctl")
DEFAULT_DIGEST_ROOT = STATE_DIR / "n8n-digests"
DEFAULT_NEWS_FEEDS = [
    "https://www.coindesk.com/arc/outboundfeeds/rss/",
    "https://cointelegraph.com/rss",
    "https://decrypt.co/feed",
]
DEFAULT_GITHUB_REPOS = [
    "ccxt/ccxt",
    "bmoscon/cryptofeed",
    "freqtrade/freqtrade",
    "hummingbot/hummingbot",
    "jesse-ai/jesse",
    "polakowo/vectorbt",
]
DEFAULT_ANALYSIS_LIMIT = 24
HIGH_RISK_KEYWORDS = {
    "cpi",
    "fomc",
    "fed",
    "inflation",
    "rate hike",
    "tariff",
    "war",
    "missile",
    "exploit",
    "hack",
    "liquidation",
    "lawsuit",
    "sec",
}
BULLISH_KEYWORDS = {
    "approval",
    "inflow",
    "accumulation",
    "adoption",
    "buyback",
    "partnership",
    "surge",
    "etf",
}
BEARISH_KEYWORDS = {
    "outflow",
    "dump",
    "selloff",
    "liquidation",
    "exploit",
    "hack",
    "crackdown",
    "ban",
}
EXCHANGE_KEYWORDS = {
    "binance",
    "coinbase",
    "kraken",
    "okx",
    "bybit",
    "bitfinex",
    "exchange",
}


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def now_stamp() -> str:
    return now_utc().strftime("%Y%m%dT%H%M%SZ")


def parse_iso_datetime(value: str) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def http_get_json(url: str, *, headers: dict[str, str] | None = None, timeout: int = 20) -> Any:
    request = urllib.request.Request(url, headers=headers or {})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def http_post_json(url: str, payload: dict[str, Any], *, timeout: int = 20) -> Any:
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        raw = response.read().decode("utf-8")
        return json.loads(raw) if raw else {}


def build_strategy_analyzer_payload(payload: dict[str, Any]) -> dict[str, Any]:
    decision = payload["decision"]
    return {
        "generated_at": payload["generated_at"],
        "project_root": payload["project_root"],
        "config": payload["config"],
        "doctor": {
            "overall": (payload.get("doctor") or {}).get("overall"),
            "warnings": (payload.get("doctor") or {}).get("warnings") or [],
        },
        "news": {
            "risk": payload["news"]["risk"],
        },
        "whale": {
            "signal": payload["whale"].get("signal"),
            "exchange_inflow_usd": payload["whale"].get("exchange_inflow_usd"),
            "exchange_outflow_usd": payload["whale"].get("exchange_outflow_usd"),
        },
        "github_summary": payload.get("github_summary") or {},
        "selected": decision.get("selected"),
        "decision": {
            "action": decision.get("action"),
            "reason": decision.get("reason"),
            "trade_mode": decision.get("trade_mode"),
            "should_notify": decision.get("should_notify"),
        },
        "ranked": payload["ranked"],
    }


def fetch_strategy_analyzer_summary(
    url: str,
    payload: dict[str, Any],
    *,
    timeout: int = 60,
) -> dict[str, Any]:
    if not url:
        return {
            "enabled": False,
            "available": False,
            "source": "strategy-analyzer",
            "reason": "STRATEGY_ANALYZER_URL not configured",
        }
    try:
        response = http_post_json(url, payload, timeout=timeout)
    except (urllib.error.URLError, TimeoutError, RuntimeError, ValueError) as exc:
        return {
            "enabled": True,
            "available": False,
            "source": url,
            "reason": str(exc),
        }
    if not isinstance(response, dict):
        return {
            "enabled": True,
            "available": False,
            "source": url,
            "reason": "strategy analyzer returned a non-object response",
        }
    return {
        "enabled": True,
        "available": True,
        "source": url,
        "result": response,
    }


def fetch_rss_feed(url: str, *, limit: int = 8) -> list[dict[str, Any]]:
    request = urllib.request.Request(url, headers={"User-Agent": "openclaw-n8n-digest"})
    with urllib.request.urlopen(request, timeout=20) as response:
        root = ET.fromstring(response.read())

    items: list[dict[str, Any]] = []
    for item in root.findall(".//channel/item"):
        title = (item.findtext("title") or "").strip()
        link = (item.findtext("link") or "").strip()
        published = (item.findtext("pubDate") or "").strip()
        if title:
            items.append({"title": title, "link": link, "published": published, "source": url})
        if len(items) >= limit:
            return items

    ns = {"atom": "http://www.w3.org/2005/Atom"}
    for entry in root.findall(".//atom:entry", ns):
        title = (entry.findtext("atom:title", default="", namespaces=ns) or "").strip()
        link_el = entry.find("atom:link", ns)
        link = (link_el.get("href") if link_el is not None else "") or ""
        published = (
            entry.findtext("atom:published", default="", namespaces=ns)
            or entry.findtext("atom:updated", default="", namespaces=ns)
        ).strip()
        if title:
            items.append({"title": title, "link": link.strip(), "published": published, "source": url})
        if len(items) >= limit:
            break
    return items


def collect_news_items(news_feeds: list[str], *, news_limit: int) -> list[dict[str, Any]]:
    feed_batches = [fetch_rss_feed(feed_url, limit=news_limit) for feed_url in news_feeds]
    items: list[dict[str, Any]] = []
    cursor = 0
    while len(items) < news_limit:
        added = False
        for batch in feed_batches:
            if cursor < len(batch):
                items.append(batch[cursor])
                added = True
                if len(items) >= news_limit:
                    break
        if not added:
            break
        cursor += 1
    return items


def assess_news_risk(items: list[dict[str, Any]]) -> dict[str, Any]:
    high_impact: list[dict[str, Any]] = []
    bullish = 0
    bearish = 0
    for item in items:
        title = str(item.get("title", "")).lower()
        if any(keyword in title for keyword in HIGH_RISK_KEYWORDS):
            high_impact.append(item)
        if any(keyword in title for keyword in BULLISH_KEYWORDS):
            bullish += 1
        if any(keyword in title for keyword in BEARISH_KEYWORDS):
            bearish += 1
    if len(high_impact) >= 2:
        risk_level = "high"
    elif high_impact:
        risk_level = "elevated"
    else:
        risk_level = "normal"
    bias = "bullish" if bullish > bearish else "bearish" if bearish > bullish else "neutral"
    return {
        "risk_level": risk_level,
        "bullish_count": bullish,
        "bearish_count": bearish,
        "high_impact_count": len(high_impact),
        "high_impact_titles": [item.get("title", "") for item in high_impact[:5]],
        "bias": bias,
    }


def summarize_whale_transactions(transactions: list[dict[str, Any]]) -> dict[str, Any]:
    inflow_usd = 0.0
    outflow_usd = 0.0
    largest: list[dict[str, Any]] = []
    for tx in transactions:
        amount_usd = float(tx.get("amount_usd") or 0.0)
        from_owner = str((tx.get("from") or {}).get("owner", "")).lower()
        to_owner = str((tx.get("to") or {}).get("owner", "")).lower()
        from_exchange = any(keyword in from_owner for keyword in EXCHANGE_KEYWORDS)
        to_exchange = any(keyword in to_owner for keyword in EXCHANGE_KEYWORDS)
        if to_exchange and not from_exchange:
            inflow_usd += amount_usd
        elif from_exchange and not to_exchange:
            outflow_usd += amount_usd
        largest.append(
            {
                "symbol": tx.get("symbol"),
                "amount_usd": round(amount_usd, 2),
                "from_owner": (tx.get("from") or {}).get("owner"),
                "to_owner": (tx.get("to") or {}).get("owner"),
                "transaction_type": tx.get("transaction_type"),
            }
        )
    if inflow_usd > outflow_usd * 1.25 and inflow_usd > 0:
        signal = "bearish"
    elif outflow_usd > inflow_usd * 1.25 and outflow_usd > 0:
        signal = "bullish"
    else:
        signal = "neutral"
    largest.sort(key=lambda item: item["amount_usd"], reverse=True)
    return {
        "transaction_count": len(transactions),
        "exchange_inflow_usd": round(inflow_usd, 2),
        "exchange_outflow_usd": round(outflow_usd, 2),
        "signal": signal,
        "largest": largest[:5],
    }


def fetch_whale_alert_summary(
    api_key: str,
    *,
    min_value_usd: int = 500000,
    lookback_seconds: int = 21600,
    limit: int = 10,
) -> dict[str, Any]:
    if not api_key:
        return {
            "enabled": False,
            "source": "whale-alert",
            "available": False,
            "reason": "WHALE_ALERT_API_KEY not configured",
        }

    query = urllib.parse.urlencode(
        {
            "api_key": api_key,
            "min_value": min_value_usd,
            "start": int(time.time()) - lookback_seconds,
            "limit": limit,
        }
    )
    url = f"https://api.whale-alert.io/v1/transactions?{query}"
    try:
        payload = http_get_json(url, headers={"User-Agent": "openclaw-n8n-digest"})
    except (urllib.error.URLError, TimeoutError, RuntimeError, ValueError) as exc:
        return {
            "enabled": True,
            "source": "whale-alert",
            "available": False,
            "reason": str(exc),
            "signal": "neutral",
            "exchange_inflow_usd": 0.0,
            "exchange_outflow_usd": 0.0,
            "transaction_count": 0,
            "largest": [],
        }
    transactions = payload.get("transactions") or []
    summary = summarize_whale_transactions(transactions)
    summary.update(
        {
            "enabled": True,
            "source": "whale-alert",
            "available": True,
            "result": payload.get("result", "unknown"),
        }
    )
    return summary


def fetch_github_observability(repos: list[str]) -> list[dict[str, Any]]:
    now = now_utc()
    items: list[dict[str, Any]] = []
    for repo in repos:
        url = f"https://api.github.com/repos/{repo}"
        payload = http_get_json(
            url,
            headers={
                "Accept": "application/vnd.github+json",
                "User-Agent": "openclaw-n8n-digest",
            },
        )
        updated_at = str(payload["updated_at"])
        updated_dt = parse_iso_datetime(updated_at)
        recency_days = (
            round((now - updated_dt).total_seconds() / 86400.0, 2)
            if updated_dt is not None
            else None
        )
        items.append(
            {
                "repo": payload["full_name"],
                "stars": payload["stargazers_count"],
                "open_issues": payload["open_issues_count"],
                "updated_at": updated_at,
                "recency_days": recency_days,
                "default_branch": payload["default_branch"],
                "html_url": payload["html_url"],
            }
        )
    return items


def summarize_github_observability(items: list[dict[str, Any]]) -> dict[str, Any]:
    repo_count = len(items)
    active_repo_count = sum(
        1 for item in items if item.get("recency_days") is not None and float(item["recency_days"]) <= 30.0
    )
    hot_repo_count = sum(
        1 for item in items if item.get("recency_days") is not None and float(item["recency_days"]) <= 7.0
    )
    total_stars = sum(int(item.get("stars") or 0) for item in items)
    if hot_repo_count >= 2 or active_repo_count >= max(1, repo_count - 1):
        signal = "active"
    elif active_repo_count >= 1:
        signal = "mixed"
    else:
        signal = "stale"
    return {
        "repo_count": repo_count,
        "active_repo_count": active_repo_count,
        "hot_repo_count": hot_repo_count,
        "total_stars": total_stars,
        "signal": signal,
        "top_repos": sorted(items, key=lambda item: int(item.get("stars") or 0), reverse=True)[:5],
    }


def run_compact_json(command: list[str], *, timeout: int = 300) -> dict[str, Any]:
    completed = subprocess.run(command, text=True, capture_output=True, timeout=timeout, check=False)
    stdout = (completed.stdout or "").strip()
    stderr = (completed.stderr or "").strip()
    if completed.returncode != 0:
        raise RuntimeError(stderr or stdout or f"command failed: {' '.join(command)}")
    try:
        return json.loads(stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"command returned invalid JSON: {' '.join(command)}") from exc


def _candidate_quality(
    latest: dict[str, Any],
    analysis: dict[str, Any],
    *,
    route: Any,
    trade_plan: dict[str, Any],
    news_risk: dict[str, Any] | None = None,
    side: str = "BUY",
) -> dict[str, Any]:
    signal_scores = build_signal_scores(
        route=route,
        latest=latest,
        analysis=analysis,
        trade_plan=trade_plan,
        news_risk=news_risk,
        side=side,
    )
    return {
        "score": signal_scores["composite_convergence_score"],
        "price_structure_score": signal_scores["price_structure_score"],
        "flow_score": signal_scores["flow_score"],
        "event_risk_score": signal_scores["event_risk_score"],
        "execution_quality_score": signal_scores["execution_quality_score"],
        "strategy_fit_score": signal_scores["strategy_fit_score"],
        "adx": round(float(latest.get("adx") or 0.0), 4),
        "volume_zscore_20": round(float(latest.get("volume_zscore_20") or 0.0), 4),
        "realized_vol_20": round(float(latest.get("realized_vol_20") or 0.0), 6),
    }


def run_quant_analysis(symbol: str, market: str, interval: str) -> dict[str, Any]:
    payload = run_compact_json(
        [str(QUANTCTL), "analyze", symbol.upper(), "--market", market, "--interval", interval, "--compact"],
        timeout=600,
    )
    latest = payload.get("latest") or {}
    analysis = payload.get("analysis") or {}
    bias = str(analysis.get("bias") or "neutral")
    adx = float(latest.get("adx") or 0.0)
    score = float(analysis.get("score") or 0.0)
    convergence = float(analysis.get("convergence") or 0.0)
    direction = "long" if "long" in bias else "short" if "short" in bias else "neutral"
    side = "SELL" if direction == "short" else "BUY"
    route = resolve_symbol_route(str(payload.get("symbol", symbol.upper())))
    quality = _candidate_quality(
        latest,
        analysis,
        route=route,
        trade_plan=payload.get("trade_plan") or {},
        side=side,
    )
    composite = round((score * convergence * 0.4) + (float(quality["score"]) * 0.6) + min(adx, 25.0) * 0.25, 3)
    return {
        "symbol": payload.get("symbol", symbol.upper()),
        "market": payload.get("market", market),
        "interval": interval,
        "direction": direction,
        "composite_score": composite,
        "candidate_quality": quality,
        "signal_scores": quality,
        "analysis": analysis,
        "latest": latest,
        "trade_plan": payload.get("trade_plan") or {},
        "run_id": payload.get("run_id"),
        "route_id": route.route_id,
        "asset_class": route.asset_class,
        "validation": route.validation.to_dict(),
    }


def candidate_is_tradeable(item: dict[str, Any]) -> bool:
    analysis = item.get("analysis") or {}
    latest = item.get("latest") or {}
    quality = item.get("candidate_quality")
    if quality is None:
        route = resolve_symbol_route(str(item.get("symbol") or "BTCUSDT"))
        quality = _candidate_quality(
            latest,
            analysis,
            route=route,
            trade_plan=item.get("trade_plan") or {},
            side="SELL" if str(item.get("direction") or "") == "short" else "BUY",
        )
    direction = str(item.get("direction") or "neutral")
    if direction not in {"long", "short"}:
        return False
    if float(analysis.get("convergence") or 0.0) < 0.55:
        return False
    if float(latest.get("adx") or 0.0) < 12.0:
        return False
    if float(latest.get("realized_vol_20") or 0.0) > 2.2:
        return False
    if float(quality.get("score") or 0.0) < 45.0:
        return False
    return True


def rank_candidates(analyses: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    tradeable = [item for item in analyses if candidate_is_tradeable(item)]
    long_candidates = [item for item in tradeable if item.get("direction") == "long"]
    short_candidates = [item for item in tradeable if item.get("direction") == "short"]
    neutral_candidates = [item for item in analyses if item.get("direction") == "neutral"]
    for bucket in (long_candidates, short_candidates, neutral_candidates):
        bucket.sort(
            key=lambda item: (
                float((item.get("candidate_quality") or {}).get("score") or 0.0),
                float(item.get("composite_score") or 0.0),
            ),
            reverse=True,
        )
    return {
        "long": long_candidates,
        "short": short_candidates,
        "neutral": neutral_candidates,
    }


def choose_strategy(candidate: dict[str, Any], news_risk: str) -> str:
    latest = candidate.get("latest") or {}
    adx = float(latest.get("adx") or 0.0)
    score = float((candidate.get("analysis") or {}).get("score") or 0.0)
    if news_risk == "high":
        return "risk_off"
    if adx >= 18 and score >= 66:
        return "trend_following"
    if adx >= 14 and score >= 58:
        return "breakout"
    return "mean_reversion"


def adjusted_candidate_score(
    candidate: dict[str, Any],
    *,
    news_risk: dict[str, Any],
    whale_summary: dict[str, Any],
    github_summary: dict[str, Any],
) -> tuple[float, str, list[str], dict[str, Any]]:
    direction = str(candidate.get("direction") or "neutral")
    analysis = candidate.get("analysis") or {}
    base_score = float(candidate.get("composite_score") or 0.0)
    convergence = float(analysis.get("convergence") or 0.0)
    analysis_score = float(analysis.get("score") or 0.0)
    adjusted = base_score
    notes: list[str] = []
    route_id = str(candidate.get("route_id") or "")
    side = "BUY" if direction == "long" else "SELL" if direction == "short" else "UNKNOWN"
    route_side_feedback: dict[str, Any] = {}

    if route_id and side in {"BUY", "SELL"}:
        feedback = evaluate_route_side_risk(route_id=route_id, side=side)
        route_side_feedback = feedback.to_dict()
        if feedback.sample_count >= feedback.min_samples:
            if feedback.net_pnl_usdt < 0.0 and feedback.profit_factor < 0.8:
                adjusted -= 14.0
                notes.append("route-side-negative-expectancy")
            elif feedback.net_pnl_usdt < 0.0 and feedback.profit_factor < 1.0:
                adjusted -= 7.0
                notes.append("route-side-still-negative")
            elif feedback.net_pnl_usdt > 0.0 and feedback.profit_factor >= 1.2:
                adjusted += 5.0
                notes.append("route-side-positive-feedback")
            if feedback.stop_loss_ratio >= 70.0 and feedback.profit_factor < 1.0:
                adjusted -= 6.0
                notes.append("route-side-stop-loss-heavy")
            if feedback.loss_streak >= 3:
                adjusted -= 3.0
                notes.append("route-side-loss-streak")
        elif feedback.sample_count > 0 and feedback.net_pnl_usdt < 0.0:
            if feedback.profit_factor < 0.5:
                adjusted -= 18.0
                notes.append("route-side-early-negative-expectancy")
            elif feedback.profit_factor < 0.8:
                adjusted -= 9.0
                notes.append("route-side-early-weak-feedback")

    whale_signal = str(whale_summary.get("signal") or "neutral")
    alignment = "neutral"
    if whale_signal != "neutral":
        if (direction == "long" and whale_signal == "bullish") or (direction == "short" and whale_signal == "bearish"):
            adjusted += 5.0
            alignment = "aligned"
            notes.append("whale-aligned")
        else:
            adjusted -= 7.0
            alignment = "opposed"
            notes.append("whale-opposed")

    news_bias = str(news_risk.get("bias") or "neutral")
    if news_bias != "neutral":
        if (direction == "long" and news_bias == "bullish") or (direction == "short" and news_bias == "bearish"):
            adjusted += 2.0
            notes.append("news-bias-aligned")
        else:
            adjusted -= 3.0
            notes.append("news-bias-opposed")

    risk_level = str(news_risk.get("risk_level") or "normal")
    if risk_level == "high":
        adjusted -= 8.0
        notes.append("high-news-risk")
    elif risk_level == "elevated":
        adjusted -= 4.0
        notes.append("elevated-news-risk")

    github_signal = str(github_summary.get("signal") or "mixed")
    if github_signal == "active":
        adjusted += 1.5
        notes.append("github-stack-active")
    elif github_signal == "stale":
        adjusted -= 1.5
        notes.append("github-stack-stale")

    if analysis_score >= 82 and convergence >= 0.82:
        adjusted += 3.0
        notes.append("internal-conviction-strong")
    elif analysis_score < 58 or convergence < 0.58:
        adjusted -= 4.0
        notes.append("internal-conviction-weak")

    return round(adjusted, 3), alignment, notes, route_side_feedback


def build_decision(
    ranked: dict[str, list[dict[str, Any]]],
    news_risk: dict[str, Any],
    whale_summary: dict[str, Any],
    github_summary: dict[str, Any],
    doctor: dict[str, Any],
) -> dict[str, Any]:
    candidates = [*ranked["long"], *ranked["short"]]
    evaluated: list[dict[str, Any]] = []
    for candidate in candidates:
        adjusted_score, alignment, context_notes, route_side_feedback = adjusted_candidate_score(
            candidate,
            news_risk=news_risk,
            whale_summary=whale_summary,
            github_summary=github_summary,
        )
        enriched = dict(candidate)
        enriched["adjusted_score"] = adjusted_score
        enriched["external_alignment"] = alignment
        enriched["context_notes"] = context_notes
        enriched["route_side_feedback"] = route_side_feedback
        evaluated.append(enriched)
    evaluated.sort(key=lambda item: float(item.get("adjusted_score") or 0.0), reverse=True)
    selected = evaluated[0] if evaluated else None

    trade_mode = "paper_only" if doctor.get("warnings") else "guarded_live_candidate"
    if not selected:
        return {
            "action": "stand_by",
            "reason": "no_ranked_candidates",
            "trade_mode": trade_mode,
            "selected": None,
            "should_notify": False,
        }

    strategy = choose_strategy(selected, news_risk["risk_level"])
    adjusted_score = float(selected.get("adjusted_score") or 0.0)
    alignment = str(selected.get("external_alignment") or "neutral")
    risk_level = str(news_risk.get("risk_level") or "normal")
    context_notes = [str(item) for item in (selected.get("context_notes") or [])]
    negative_feedback = any(
        item
        in {
            "route-side-negative-expectancy",
            "route-side-still-negative",
            "route-side-stop-loss-heavy",
            "route-side-early-negative-expectancy",
            "route-side-early-weak-feedback",
        }
        for item in context_notes
    )
    if alignment == "opposed" and risk_level == "high":
        action = "no_trade"
        reason = "high_event_risk_and_external_misalignment"
    elif risk_level == "high" and negative_feedback:
        action = "stand_by"
        reason = "high_event_risk_and_negative_feedback"
    elif adjusted_score >= 72 and strategy != "risk_off" and alignment != "opposed":
        action = "pre_trade_notify"
        reason = "top_ranked_candidate"
    elif adjusted_score >= 64 or (strategy == "risk_off" and not negative_feedback):
        action = "watchlist_only"
        reason = "guarded_context_candidate"
    else:
        action = "stand_by"
        reason = "insufficient_adjusted_edge"
    return {
        "action": action,
        "reason": reason,
        "trade_mode": trade_mode,
        "selected": {
            "symbol": selected["symbol"],
            "route_id": selected.get("route_id"),
            "asset_class": selected.get("asset_class"),
            "direction": selected["direction"],
            "composite_score": selected["composite_score"],
            "adjusted_score": selected["adjusted_score"],
            "analysis_score": (selected.get("analysis") or {}).get("score"),
            "convergence": (selected.get("analysis") or {}).get("convergence"),
            "strategy": strategy,
            "external_alignment": alignment,
            "context_notes": selected.get("context_notes") or [],
            "route_side_feedback": selected.get("route_side_feedback") or {},
            "price_structure_score": (selected.get("candidate_quality") or {}).get("price_structure_score"),
            "flow_score": (selected.get("candidate_quality") or {}).get("flow_score"),
            "event_risk_score": (selected.get("candidate_quality") or {}).get("event_risk_score"),
            "execution_quality_score": (selected.get("candidate_quality") or {}).get("execution_quality_score"),
            "strategy_fit_score": (selected.get("candidate_quality") or {}).get("strategy_fit_score"),
            "validation": selected.get("validation"),
        },
        "should_notify": action == "pre_trade_notify",
    }


def apply_strategy_analyzer_to_decision(
    decision: dict[str, Any],
    strategy_analysis: dict[str, Any],
) -> dict[str, Any]:
    if not strategy_analysis.get("available"):
        return decision
    result = (strategy_analysis.get("result") or {}).get("result") or {}
    verdict = str(result.get("verdict") or "watch")
    confidence = float(result.get("confidence") or 0.0)
    updated = dict(decision)
    selected = dict(updated.get("selected") or {}) if updated.get("selected") else None
    updated["strategy_analyzer"] = {
        "verdict": verdict,
        "confidence": round(confidence, 3),
        "regime": result.get("regime"),
        "notes": result.get("notes") or [],
    }
    if selected is None:
        return updated
    if verdict == "approve" and confidence >= 0.62 and updated["action"] == "watchlist_only":
        updated["action"] = "pre_trade_notify"
        updated["reason"] = "strategy_analyzer_promoted_candidate"
        updated["should_notify"] = True
    elif verdict == "watch" and confidence < 0.48 and updated["action"] == "pre_trade_notify":
        updated["action"] = "watchlist_only"
        updated["reason"] = "strategy_analyzer_downgraded_candidate"
        updated["should_notify"] = False
    return updated


def format_telegram_text(payload: dict[str, Any]) -> str:
    decision = payload["decision"]
    news_risk = payload["news"]["risk"]
    whale = payload["whale"]
    lines = [
        "Binance Quant Daily Digest",
        f"generated_at: {payload['generated_at']}",
        f"decision: {decision['action']}",
        f"trade_mode: {decision['trade_mode']}",
        f"news_risk: {news_risk['risk_level']} ({news_risk['high_impact_count']} high-impact headlines)",
        f"whale_signal: {whale.get('signal', 'unavailable')}",
        f"github_stack: {payload['github_summary'].get('signal', 'unknown')}",
    ]
    selected = decision.get("selected")
    if selected:
        lines.extend(
            [
                f"candidate: {selected['symbol']} {selected['direction']}",
                f"strategy: {selected['strategy']}",
                f"composite_score: {selected['composite_score']}",
                f"adjusted_score: {selected.get('adjusted_score')}",
                f"analysis_score: {selected['analysis_score']} convergence={selected['convergence']}",
            ]
        )
    else:
        lines.append(f"reason: {decision['reason']}")
    return "\n".join(lines)


def maybe_send_telegram(payload: dict[str, Any]) -> dict[str, Any]:
    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "").strip()
    if not payload["decision"]["should_notify"]:
        return {"sent": False, "reason": "decision does not require notification"}
    if not token or not chat_id:
        return {"sent": False, "reason": "TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID missing"}

    encoded_token = urllib.parse.quote(token, safe="")
    url = f"https://api.telegram.org/bot{encoded_token}/sendMessage"
    response = http_post_json(url, {"chat_id": chat_id, "text": payload["telegram_text"]})
    return {"sent": True, "response": response}


def load_config(config_path: Path) -> dict[str, Any]:
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Config must be a JSON object: {config_path}")
    return payload


def build_digest(config: dict[str, Any]) -> dict[str, Any]:
    ensure_runtime_dirs()
    configured_symbols = [str(symbol).upper() for symbol in config.get("symbols") or ["BTCUSDT", "ETHUSDT", "SOLUSDT"]]
    excluded_symbols = {str(symbol).upper() for symbol in config.get("exclude_symbols") or []}
    symbols = list(configured_symbols)
    analysis_limit = int(config.get("analysis_limit") or DEFAULT_ANALYSIS_LIMIT)
    market = str(config.get("market") or "futures")
    interval = str(config.get("interval") or "4h")
    include_top_volume = bool(config.get("include_top_futures_volume", False))
    top_volume_limit = int(config.get("top_futures_volume_limit") or 60)
    universe_symbols: list[dict[str, Any]] = []
    if include_top_volume and market == "futures":
        try:
            universe = fetch_top_futures_symbols(
                load_settings(),
                limit=top_volume_limit,
                include_symbols=configured_symbols,
            )
            universe_symbols = [item.to_dict() for item in universe]
            symbols = [item.symbol for item in universe]
        except Exception as exc:
            universe_symbols = [{"status": "unavailable", "reason": str(exc)}]
    if excluded_symbols:
        symbols = [symbol for symbol in symbols if symbol not in excluded_symbols]
    symbols = symbols[:analysis_limit]
    news_feeds = [str(url) for url in (config.get("news_feeds") or DEFAULT_NEWS_FEEDS)]
    news_limit = int(config.get("news_limit") or 8)
    whale_api_key = str(os.getenv("WHALE_ALERT_API_KEY", "")).strip()
    whale_min_value_usd = int(config.get("whale_min_value_usd") or 500000)
    whale_limit = int(config.get("whale_limit") or 10)
    strategy_analyzer_url = str(
        config.get("strategy_analyzer_url") or os.getenv("STRATEGY_ANALYZER_URL", "")
    ).strip()
    github_repos = [str(repo) for repo in (config.get("github_observability_repos") or DEFAULT_GITHUB_REPOS)]
    digest_root = Path(config.get("digest_root") or DEFAULT_DIGEST_ROOT).expanduser().resolve()
    digest_root.mkdir(parents=True, exist_ok=True)

    news_items = collect_news_items(news_feeds, news_limit=news_limit)
    news_risk = assess_news_risk(news_items)

    whale_summary = fetch_whale_alert_summary(
        whale_api_key,
        min_value_usd=whale_min_value_usd,
        limit=whale_limit,
    )
    github_observability = fetch_github_observability(github_repos)
    github_summary = summarize_github_observability(github_observability)
    external_context = build_external_context(
        symbols,
        config_path=config.get("external_context_config") or None,
    )
    doctor = run_compact_json([str(QUANTCTL), "doctor", "--compact"], timeout=300)
    analyses = [run_quant_analysis(symbol, market, interval) for symbol in symbols]
    ranked = rank_candidates(analyses)
    decision = build_decision(ranked, news_risk, whale_summary, github_summary, doctor)

    payload = {
        "generated_at": now_utc().replace(microsecond=0).isoformat(),
        "project_root": str(PROJECT_ROOT),
        "config": {
            "symbols": symbols,
            "configured_symbols": configured_symbols,
            "exclude_symbols": sorted(excluded_symbols),
            "include_top_futures_volume": include_top_volume,
            "top_futures_volume_limit": top_volume_limit,
            "universe_symbols": universe_symbols[:top_volume_limit],
            "analysis_limit": analysis_limit,
            "market": market,
            "interval": interval,
            "news_feeds": news_feeds,
            "news_limit": news_limit,
            "whale_min_value_usd": whale_min_value_usd,
            "whale_limit": whale_limit,
            "digest_root": str(digest_root),
            "strategy_analyzer_url": strategy_analyzer_url,
            "external_context_config": config.get("external_context_config")
            or "config/external-context.default.yaml",
        },
        "doctor": doctor,
        "news": {
            "items": news_items,
            "risk": news_risk,
        },
        "whale": whale_summary,
        "github_observability": github_observability,
        "github_summary": github_summary,
        "external_context": external_context,
        "analyses": analyses,
        "ranked": {
            "long": ranked["long"][:3],
            "short": ranked["short"][:3],
            "neutral": ranked["neutral"][:3],
        },
        "decision": decision,
    }
    payload["strategy_analysis"] = fetch_strategy_analyzer_summary(
        strategy_analyzer_url,
        build_strategy_analyzer_payload(payload),
    )
    payload["decision"] = apply_strategy_analyzer_to_decision(payload["decision"], payload["strategy_analysis"])
    payload["telegram_text"] = format_telegram_text(payload)
    payload["telegram"] = maybe_send_telegram(payload)

    output_path = digest_root / f"{now_stamp()}-daily-digest.json"
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    payload["output_path"] = str(output_path)
    return payload
