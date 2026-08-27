"""Provider abstraction: Binance stays default; Kraken + Coinbase + Deribit.

Network tests hit public REST (no API keys). They skip if a venue is unreachable
so CI without egress still goes green. Mocked tests always run.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

from slate_core.connectors.base import (
    MarketDataError,
    resolve_canonical_symbol,
)
from slate_core.connectors.binance_provider import BinanceDataProvider
from slate_core.connectors.coinbase import MAX_CANDLES, CoinbaseDataProvider
from slate_core.connectors.deribit import MAX_CANDLES as DERIBIT_MAX_CANDLES
from slate_core.connectors.deribit import DeribitDataProvider
from slate_core.connectors.factory import available_providers, get_market_data_provider
from slate_core.connectors.kraken import KrakenDataProvider
from slate_core.connectors.paper_book import PaperTradingBook
from slate_core.connectors.replay import ReplayMarketDataProvider

FIXTURES = Path("tests/fixtures/providers")
REAL_SOL = Path("sol_data_cache/SOLUSDT_perpetual_1h_6m.csv")


def _load(name: str):
    return json.loads((FIXTURES / name).read_text())


def test_factory_lists_binance_kraken_coinbase_deribit():
    names = set(available_providers())
    assert names == {"binance", "kraken", "coinbase", "deribit"}
    assert get_market_data_provider("binance").name == "binance"
    assert get_market_data_provider("kraken").name == "kraken"
    assert get_market_data_provider("coinbase").name == "coinbase"
    assert get_market_data_provider("deribit").name == "deribit"
    with pytest.raises(MarketDataError):
        get_market_data_provider("replay")
    with pytest.raises(MarketDataError):
        get_market_data_provider("not-a-venue")


def test_factory_default_stays_binance(monkeypatch):
    """Mailbox: do not flip the default because Binance 451s on some egress."""
    monkeypatch.delenv("SLATE_DATA_PROVIDER", raising=False)
    assert get_market_data_provider().name == "binance"
    monkeypatch.setenv("SLATE_DATA_PROVIDER", "deribit")
    assert get_market_data_provider().name == "deribit"
    monkeypatch.delenv("SLATE_DATA_PROVIDER", raising=False)
    assert get_market_data_provider().name == "binance"


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


def test_symbol_map_deribit_btc_eth_perps_not_sol():
    btc = resolve_canonical_symbol("deribit", "BTCUSDT")
    assert btc.venue_symbol == "BTC-PERPETUAL"
    assert "BTC_USDC-PERPETUAL" in btc.candidates
    eth = resolve_canonical_symbol("deribit", "ETHUSDT")
    assert eth.venue_symbol == "ETH-PERPETUAL"
    assert "ETH_USDC-PERPETUAL" in eth.candidates
    # SOLUSDT is not in the Deribit table — adapter must fail closed, not invent.
    sol = resolve_canonical_symbol("deribit", "SOLUSDT")
    assert sol.venue_symbol == "SOLUSDT"
    assert "SOL_USDC-PERPETUAL" not in sol.candidates


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


def _parse_coinbase_bound(value: str) -> int:
    return int(datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp())


def _synthetic_coinbase_candles(count: int, now_ts: int | None = None):
    # Newest first, Exchange API shape: [time, low, high, open, close, volume]
    now_ts = now_ts or int(datetime.now(timezone.utc).timestamp())
    return [
        [now_ts - i * 86400, 1.0, 2.0, 1.5, 1.6 + i * 0.001, 10.0]
        for i in range(count)
    ]


def test_coinbase_limit_over_350_paginates_start_end():
    """limit>350 must page with start/end and return exactly `limit` bars."""
    universe = _synthetic_coinbase_candles(500)
    calls = []

    def fake_get(url, params=None, **_):
        assert url.endswith("/candles")
        params = params or {}
        calls.append(dict(params))
        start = _parse_coinbase_bound(params["start"])
        end = _parse_coinbase_bound(params["end"])
        matched = [row for row in universe if start <= row[0] <= end]
        matched.sort(key=lambda row: row[0], reverse=True)
        return matched[:MAX_CANDLES]

    provider = CoinbaseDataProvider(session_get=fake_get, page_sleep=0)
    bars = provider.fetch_ohlcv("SOLUSDT", interval="1d", limit=400)
    assert len(bars) == 400
    assert len(calls) >= 2
    assert all("start" in c and "end" in c for c in calls)
    assert all(c.get("granularity") == 86400 for c in calls)
    timestamps = [b["timestamp"] for b in bars]
    assert timestamps == sorted(timestamps)
    assert timestamps[0] < timestamps[-1]


def test_coinbase_limit_over_350_fail_closed_when_short():
    """Do not silently return 350 when the caller asked for more."""
    universe = _synthetic_coinbase_candles(MAX_CANDLES)
    calls = []

    def fake_get(url, params=None, **_):
        params = params or {}
        calls.append(dict(params))
        start = _parse_coinbase_bound(params["start"])
        end = _parse_coinbase_bound(params["end"])
        matched = [row for row in universe if start <= row[0] <= end]
        matched.sort(key=lambda row: row[0], reverse=True)
        return matched[:MAX_CANDLES]

    provider = CoinbaseDataProvider(session_get=fake_get, page_sleep=0)
    with pytest.raises(MarketDataError, match="350"):
        provider.fetch_ohlcv("SOLUSDT", interval="1d", limit=400)
    assert len(calls) >= 1
    assert "start" in calls[0] and "end" in calls[0]


def test_kraken_coinbase_fail_closed_on_empty():
    def boom(url, params=None, **_):
        raise MarketDataError("blocked")

    with pytest.raises(MarketDataError):
        KrakenDataProvider(session_get=boom).fetch_ticker("SOLUSDT")
    with pytest.raises(MarketDataError):
        CoinbaseDataProvider(session_get=boom).fetch_ticker("SOLUSDT")


def _deribit_instruments():
    return _load("deribit_instruments_futures.json")


def _deribit_fake_get(ticker=None, chart=None, instruments=None):
    ticker = ticker if ticker is not None else _load("deribit_ticker_btc_perpetual.json")
    chart = chart if chart is not None else _load("deribit_chart_btc_perpetual.json")
    instruments = instruments if instruments is not None else _deribit_instruments()

    def fake_get(url, params=None, **_):
        if "get_instruments" in url:
            return instruments
        if url.endswith("/public/ticker") or "/public/ticker" in url:
            return ticker
        if "get_tradingview_chart_data" in url:
            return chart
        raise MarketDataError(f"unexpected deribit url {url}")

    return fake_get


def test_deribit_provider_parses_real_response_shape():
    provider = DeribitDataProvider(session_get=_deribit_fake_get())
    t = provider.fetch_ticker("BTCUSDT")
    assert t["provider"] == "deribit"
    assert t["venue_symbol"] == "BTC-PERPETUAL"
    assert t["last_price"] == pytest.approx(79501.5)
    assert t["last_price"] != 50000.0  # not the old synthetic default
    bars = provider.fetch_ohlcv("BTCUSDT", interval="1d", limit=10)
    assert len(bars) == 2
    assert bars[0]["close"] == pytest.approx(79100.0)
    assert bars[1]["close"] == pytest.approx(79495.0)
    assert bars[0]["timestamp"] < bars[1]["timestamp"]


def test_deribit_maps_eth_and_accepts_linear_fallback_name():
    instruments = _deribit_instruments()
    # Inverse ETH missing → linear ETH_USDC-PERPETUAL must be tried.
    instruments = {
        "jsonrpc": "2.0",
        "result": [
            row
            for row in instruments["result"]
            if row["instrument_name"] != "ETH-PERPETUAL"
        ],
    }
    ticker = _load("deribit_ticker_btc_perpetual.json")
    ticker = {
        **ticker,
        "result": {**ticker["result"], "instrument_name": "ETH_USDC-PERPETUAL", "last_price": 2625.5},
    }

    def fake_get(url, params=None, **_):
        if "get_instruments" in url:
            return instruments
        if "/public/ticker" in url:
            assert (params or {}).get("instrument_name") == "ETH_USDC-PERPETUAL"
            return ticker
        raise MarketDataError(f"unexpected deribit url {url}")

    provider = DeribitDataProvider(session_get=fake_get)
    t = provider.fetch_ticker("ETHUSDT")
    assert t["venue_symbol"] == "ETH_USDC-PERPETUAL"
    assert t["last_price"] == pytest.approx(2625.5)


def test_deribit_solusdt_fail_closed_no_invented_book():
    """Deribit lists SOL_USDC-PERPETUAL live; this pass still refuses SOLUSDT."""
    calls = []

    def fake_get(url, params=None, **_):
        calls.append(url)
        raise AssertionError("SOLUSDT must fail before any Deribit HTTP call")

    provider = DeribitDataProvider(session_get=fake_get)
    with pytest.raises(MarketDataError, match="SOLUSDT"):
        provider.fetch_ticker("SOLUSDT")
    with pytest.raises(MarketDataError, match="SOLUSDT"):
        provider.fetch_ohlcv("SOLUSDT", interval="1d", limit=5)
    assert calls == []


def test_deribit_options_out_of_scope():
    provider = DeribitDataProvider(session_get=_deribit_fake_get())
    with pytest.raises(MarketDataError, match="option"):
        provider.fetch_ticker("BTC-27JUN26-100000-C")


def test_deribit_missing_last_price_fail_closed():
    ticker = _load("deribit_ticker_btc_perpetual.json")
    ticker = {**ticker, "result": {**ticker["result"], "last_price": None}}
    provider = DeribitDataProvider(session_get=_deribit_fake_get(ticker=ticker))
    with pytest.raises(MarketDataError, match="last_price"):
        provider.fetch_ticker("BTCUSDT")


def test_deribit_jsonrpc_error_fail_closed():
    def fake_get(url, params=None, **_):
        if "get_instruments" in url:
            return _deribit_instruments()
        return {
            "jsonrpc": "2.0",
            "error": {"code": -32602, "message": "Invalid params"},
        }

    provider = DeribitDataProvider(session_get=fake_get)
    with pytest.raises(MarketDataError, match="Invalid params"):
        provider.fetch_ticker("BTCUSDT")


def test_deribit_no_synthetic_50k_on_blocked_venue():
    def boom(url, params=None, **_):
        raise MarketDataError("blocked")

    with pytest.raises(MarketDataError):
        DeribitDataProvider(session_get=boom).fetch_ticker("BTCUSDT")


def test_deribit_defaults_to_production_public_host():
    provider = DeribitDataProvider(session_get=_deribit_fake_get())
    assert provider.base_url == "https://www.deribit.com/api/v2"
    assert "test.deribit.com" not in provider.base_url


def _deribit_chart_page(rows, start_ms, end_ms):
    matched = [row for row in rows if start_ms <= row[0] <= end_ms]
    matched.sort()
    return {
        "jsonrpc": "2.0",
        "result": {
            "status": "ok",
            "ticks": [row[0] for row in matched],
            "open": [row[1] for row in matched],
            "high": [row[2] for row in matched],
            "low": [row[3] for row in matched],
            "close": [row[4] for row in matched],
            "volume": [row[5] for row in matched],
        },
    }


def test_deribit_limit_over_page_cap_paginates_start_end():
    """limit>5000 must page with start/end and return exactly `limit` bars."""
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    # Align to daily ticks so window math stays deterministic.
    now_ms = now_ms - (now_ms % 86_400_000)
    universe = [
        [now_ms - i * 86_400_000, 1.0, 2.0, 0.5, 1.6 + i * 0.001, 10.0]
        for i in range(5500)
    ]
    calls = []

    def fake_get(url, params=None, **_):
        params = params or {}
        if "get_instruments" in url:
            return _deribit_instruments()
        assert "get_tradingview_chart_data" in url
        calls.append(dict(params))
        start = int(params["start_timestamp"])
        end = int(params["end_timestamp"])
        page = [
            row for row in universe if start <= row[0] <= end
        ]
        page.sort()
        page = page[-DERIBIT_MAX_CANDLES:]
        return _deribit_chart_page(page, start, end)

    provider = DeribitDataProvider(session_get=fake_get, page_sleep=0)
    bars = provider.fetch_ohlcv("BTCUSDT", interval="1d", limit=5200)
    assert len(bars) == 5200
    assert len(calls) >= 2
    assert all("start_timestamp" in c and "end_timestamp" in c for c in calls)
    assert all(c.get("resolution") == "1D" for c in calls)
    timestamps = [b["timestamp"] for b in bars]
    assert timestamps == sorted(timestamps)
    assert timestamps[0] < timestamps[-1]


def test_deribit_limit_over_page_cap_fail_closed_when_short():
    """Do not silently return 5000 when the caller asked for more."""
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    now_ms = now_ms - (now_ms % 86_400_000)
    universe = [
        [now_ms - i * 86_400_000, 1.0, 2.0, 0.5, 1.6, 10.0]
        for i in range(DERIBIT_MAX_CANDLES)
    ]
    calls = []

    def fake_get(url, params=None, **_):
        params = params or {}
        if "get_instruments" in url:
            return _deribit_instruments()
        calls.append(dict(params))
        start = int(params["start_timestamp"])
        end = int(params["end_timestamp"])
        page = [row for row in universe if start <= row[0] <= end]
        page.sort()
        page = page[-DERIBIT_MAX_CANDLES:]
        return _deribit_chart_page(page, start, end)

    provider = DeribitDataProvider(session_get=fake_get, page_sleep=0)
    with pytest.raises(MarketDataError, match="5000"):
        provider.fetch_ohlcv("BTCUSDT", interval="1d", limit=5200)
    assert len(calls) >= 1
    assert "start_timestamp" in calls[0] and "end_timestamp" in calls[0]


def test_deribit_unsupported_4h_fail_closed():
    provider = DeribitDataProvider(session_get=_deribit_fake_get())
    with pytest.raises(MarketDataError, match="unsupported interval"):
        provider.fetch_ohlcv("BTCUSDT", interval="4h", limit=10)


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


@pytest.mark.network
def test_live_deribit_btc_perpetual_ticker_and_ohlcv():
    provider = DeribitDataProvider()
    ticker = _live_or_skip(lambda: provider.fetch_ticker("BTCUSDT"))
    assert ticker["provider"] == "deribit"
    assert ticker["last_price"] > 0
    assert ticker["last_price"] != 50000.0
    assert ticker["venue_symbol"] in {"BTC-PERPETUAL", "BTC_USDC-PERPETUAL"}
    bars = _live_or_skip(lambda: provider.fetch_ohlcv("BTCUSDT", interval="1d", limit=5))
    assert len(bars) >= 1
    assert bars[-1]["close"] > 0
