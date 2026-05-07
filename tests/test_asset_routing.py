from __future__ import annotations

from binance_quant_control.asset_routing import resolve_symbol_route


def test_doge_uses_dedicated_market_bot_route() -> None:
    route = resolve_symbol_route("DOGEUSDT")

    assert route.route_id == "doge-meme-high-beta"
    assert route.asset_class == "meme_high_beta"
    assert route.interval == "4h"
    assert route.review_lane == "doge-market-bot-review"


def test_other_meme_symbols_stay_on_broad_meme_route() -> None:
    route = resolve_symbol_route("WIFUSDT")

    assert route.route_id == "meme-high-beta"
    assert route.asset_class == "meme_high_beta"
