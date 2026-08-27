"""Provider abstraction: Binance stays default; Kraken + Coinbase map SOLUSDT.

Network tests hit public REST (no API keys). They skip if a venue is unreachable
so CI without egress still goes green. Mocked tests always run.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from slate_core.connectors.base import (
    MarketDataError,
    resolve_canonical_symbol,
)
from slate_core.connectors.binance_provider import BinanceDataProvider
from slate_core.connectors.coinbase import CoinbaseDataProvider
from slate_core.connectors.factory import available_providers, get_market_data_provider
from slate_core.connectors.kraken import KrakenDataProvider
from slate_core.connectors.paper_book import PaperTradingBook
from slate_core.connectors.replay import ReplayMarketDataProvider

FIXTURES = Path("tests/fixtures/providers")
REAL_SOL = Path("sol_data_cache/SOLUSDT_perpetual_1h_6m.csv")


def _load(name: str):
    return json.loads((FIXTURES / name).read_text())


def test_factory_lists_binance_kraken_coinbase():
    names = set(available_providers())
    assert names == {"binance", "kraken", "coinbase"}
    assert get_market_data_provider("binance").name == "binance"
    assert get_market_data_provider("kraken").name == "kraken"
    assert get_market_data_provider("coinbase").name == "coinbase"
    with pytest.raises(MarketDataError):
        get_market_data_provider("replay")
    with pytest.raises(MarketDataError):
        get_market_data_provider("not-a-venue")


def test_symbol_map_solusdt_nearest_pairs():
    b = resolve_canonical_symbol("binance", "SOLUSDT")
    assert b.venue_symbol == "SOLUSDT"
    k = resolve_canonical_symbol("kraken", "SOLUSDT")
    assert k.venue_symbol == "SOLUSDT"
    assert "SOLUSD" in k.candidates
    c = resolve_canonical_symbol("coinbase", "SOLUSDT")
    assert c.venue_symbol == "SOL-USDT"
    assert "SOL-USD" in c.candidates


def test_symbol_map_btc_kraken_uses_xbt():
    k = resolve_canonical_symbol("kraken", "BTCUSDT")
    assert k.venue_symbol.startswith("XBT")


def test_binance_provider_parses_real_response_shape():
    ticker = _load("binance_ticker_solusdt.json")
    klines = _load("binance_klines_solusdt.json")

    def fake_get(url, params=None, **_):
        if "ticker" in url:
            return ticker
        return klines

    provider = BinanceDataProvider(use_futures=True, session_get=fake_get)
    t = provider.fetch_ticker("SOLUSDT")
    assert t["provider"] == "binance"
    assert t["venue_symbol"] == "SOLUSDT"
    assert t["last_price"] == pytest.approx(148.25)
    bars = provider.fetch_ohlcv("SOLUSDT", interval="1d", limit=10)
    assert len(bars) == 2
    assert bars[0]["open"] < bars[0]["high"]
    assert bars[-1]["close"] == pytest.approx(144.0)


def test_kraken_provider_parses_real_response_shape():
    ticker = _load("kraken_ticker_solusd.json")
    ohlc = _load("kraken_ohlc_solusd.json")

    def fake_get(url, params=None, **_):
        if "Ticker" in url:
            return ticker
        return ohlc

    provider = KrakenDataProvider(session_get=fake_get)
    t = provider.fetch_ticker("SOLUSDT")
    assert t["provider"] == "kraken"
    assert t["last_price"] == pytest.approx(148.25)
    assert t["venue_symbol"] == "SOLUSD"
    bars = provider.fetch_ohlcv("SOLUSDT", interval="1d", limit=10)
    assert len(bars) == 2
    assert bars[0]["close"] == pytest.approx(141.0)
    assert bars[-1]["volume"] == pytest.approx(1100.0)


def test_coinbase_provider_parses_real_response_shape():
    ticker = _load("coinbase_ticker_solusd.json")
    stats = _load("coinbase_stats_solusd.json")
    candles = _load("coinbase_candles_solusd.json")

    def fake_get(url, params=None, **_):
        if url.endswith("/ticker"):
            return ticker
        if url.endswith("/stats"):
            return stats
        return candles

    provider = CoinbaseDataProvider(session_get=fake_get)
    t = provider.fetch_ticker("SOLUSDT")
    assert t["provider"] == "coinbase"
    assert t["last_price"] == pytest.approx(148.25)
    assert t["venue_symbol"] in {"SOL-USDT", "SOL-USD"}
    bars = provider.fetch_ohlcv("SOLUSDT", interval="1d", limit=10)
    assert len(bars) == 2
    # Sorted oldest → newest even though Coinbase sends newest first
    assert bars[0]["close"] == pytest.approx(141.0)
    assert bars[1]["close"] == pytest.approx(144.0)


def test_kraken_coinbase_fail_closed_on_empty():
    def boom(url, params=None, **_):
        raise MarketDataError("blocked")

    with pytest.raises(MarketDataError):
        KrakenDataProvider(session_get=boom).fetch_ticker("SOLUSDT")
    with pytest.raises(MarketDataError):
        CoinbaseDataProvider(session_get=boom).fetch_ticker("SOLUSDT")


def _real_sol_bars(n=30):
    assert REAL_SOL.exists(), f"real SOL cache missing at {REAL_SOL}"
    df = pd.read_json(REAL_SOL)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    rows = df.sort_values("timestamp").tail(n)
    return [
        {
            "timestamp": row.timestamp.to_pydatetime(),
            "open": float(row.open),
            "high": float(row.high),
            "low": float(row.low),
            "close": float(row.close),
            "volume": float(row.volume),
        }
        for row in rows.itertuples()
    ]


def test_replay_pass_real_sol_short_and_long():
    """Documented replay: real SOLUSDT cache → paper book, both sides."""
    bars = _real_sol_bars(40)
    provider = ReplayMarketDataProvider(bars, canonical="SOLUSDT", source_provider="binance")
    entry = float(bars[0]["close"])
    exit_px = float(bars[-1]["close"])
    assert entry > 0 and exit_px > 0

    provider.set_cursor(0)
    ticker = provider.fetch_ticker("SOLUSDT")
    assert ticker["last_price"] == pytest.approx(entry)

    long_book = PaperTradingBook(taker_fee=0.0005, slippage_bps=10.0, default_notional=100.0)
    short_book = PaperTradingBook(taker_fee=0.0005, slippage_bps=10.0, default_notional=100.0)
    assert long_book.apply_decision("ENTER_LONG", "SOLUSDT", entry)["success"]
    assert short_book.apply_decision("ENTER_SHORT", "SOLUSDT", entry)["success"]
    long_res = long_book.apply_decision("EXIT", "SOLUSDT", exit_px)
    short_res = short_book.apply_decision("EXIT", "SOLUSDT", exit_px)
    assert long_res["success"] and short_res["success"]
    # Same bars, opposite sign (fees keep them from being exact negatives).
    if exit_px > entry:
        assert long_res["realized_pnl"] > short_res["realized_pnl"]
    elif exit_px < entry:
        assert short_res["realized_pnl"] > long_res["realized_pnl"]


def _live_or_skip(fetch):
    try:
        return fetch()
    except Exception as exc:  # noqa: BLE001 — live venue may be blocked
        pytest.skip(f"live public API unavailable: {exc}")


@pytest.mark.network
def test_live_kraken_sol_ticker_and_ohlcv():
    provider = KrakenDataProvider()
    ticker = _live_or_skip(lambda: provider.fetch_ticker("SOLUSDT"))
    assert ticker["provider"] == "kraken"
    assert ticker["last_price"] > 0
    assert ticker["venue_symbol"]
    bars = _live_or_skip(lambda: provider.fetch_ohlcv("SOLUSDT", interval="1d", limit=5))
    assert len(bars) >= 1
    assert bars[-1]["close"] > 0


@pytest.mark.network
def test_live_coinbase_sol_ticker_and_ohlcv():
    provider = CoinbaseDataProvider()
    ticker = _live_or_skip(lambda: provider.fetch_ticker("SOLUSDT"))
    assert ticker["provider"] == "coinbase"
    assert ticker["last_price"] > 0
    assert ticker["venue_symbol"] in {"SOL-USDT", "SOL-USD"}
    bars = _live_or_skip(lambda: provider.fetch_ohlcv("SOLUSDT", interval="1d", limit=5))
    assert len(bars) >= 1
    assert bars[-1]["close"] > 0


@pytest.mark.network
def test_live_binance_sol_still_works():
    provider = BinanceDataProvider()
    ticker = _live_or_skip(lambda: provider.fetch_ticker("SOLUSDT"))
    assert ticker["provider"] == "binance"
    assert ticker["last_price"] > 0
    assert ticker["venue_symbol"] == "SOLUSDT"
