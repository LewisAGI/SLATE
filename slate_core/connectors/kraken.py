"""Kraken public market-data adapter.

Public REST (no API key):

- Ticker: ``GET https://api.kraken.com/0/public/Ticker?pair=SOLUSD``
- OHLC:   ``GET https://api.kraken.com/0/public/OHLC?pair=SOLUSD&interval=1440``

Canonical ``SOLUSDT`` tries ``SOLUSDT`` then ``SOLUSD``. BTC is ``XBTUSDT`` /
``XBTUSD``. Paper only — this module never places orders.

Kraken pair names in the JSON body can differ from the request (``XXBTZUSD``).
We accept whatever key comes back in ``result``.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from slate_core.config.constants import KRAKEN_API_BASE

from .base import INTERVAL_MINUTES, MarketDataError, MarketDataProvider
from .http_util import get_json

# Kraken OHLC interval is minutes. 4h=240 exists; 6h does not — nearest 240.
_KRAKEN_INTERVAL = {
    "1m": 1,
    "5m": 5,
    "15m": 15,
    "30m": 30,
    "1h": 60,
    "4h": 240,
    "6h": 240,
    "1d": 1440,
}


class KrakenDataProvider(MarketDataProvider):
    name = "kraken"

    def __init__(self, session_get=None):
        self._get_json = session_get or get_json
        self.base_url = KRAKEN_API_BASE

    def _public(self, path: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        payload = self._get_json(f"{self.base_url}{path}", params=params)
        if not isinstance(payload, dict):
            raise MarketDataError("kraken: unexpected payload")
        errors = payload.get("error") or []
        if errors:
            raise MarketDataError(f"kraken error: {errors}")
        result = payload.get("result")
        if not isinstance(result, dict) or not result:
            raise MarketDataError("kraken: empty result")
        return result

    def _pair_payload(self, result: Dict[str, Any]) -> Dict[str, Any]:
        # Drop metadata keys like "last" on OHLC responses.
        pairs = {k: v for k, v in result.items() if k != "last" and isinstance(v, (dict, list))}
        if not pairs:
            raise MarketDataError("kraken: no pair data in result")
        key = next(iter(pairs))
        return {"venue_symbol": key, "data": pairs[key]}

    def fetch_ticker(self, symbol: str) -> Dict[str, Any]:
        resolved = self.resolve_symbol(symbol)
        last_error: Optional[Exception] = None
        for venue_symbol in resolved.candidates:
            try:
                result = self._public("/0/public/Ticker", {"pair": venue_symbol})
                parsed = self._pair_payload(result)
            except MarketDataError as exc:
                last_error = exc
                continue
            data = parsed["data"]
            if not isinstance(data, dict):
                last_error = MarketDataError(f"kraken ticker not a dict for {venue_symbol}")
                continue
            # c = [last price, lot volume]; v = volume [today, 24h];
            # h/l = high/low [today, 24h]; o = today open; b/a = bid/ask.
            try:
                last = float(data["c"][0])
                if last <= 0:
                    raise ValueError("non-positive last")
                open_px = float(data.get("o") or last)
                change = ((last - open_px) / open_px * 100.0) if open_px else 0.0
                high = data.get("h") or [0, 0]
                low = data.get("l") or [0, 0]
                vol = data.get("v") or [0, 0]
                bid = data.get("b") or [0]
                ask = data.get("a") or [0]
            except (KeyError, TypeError, ValueError, IndexError) as exc:
                last_error = MarketDataError(f"kraken ticker parse {venue_symbol}: {exc}")
                continue
            return {
                "symbol": resolved.canonical,
                "venue_symbol": parsed["venue_symbol"],
                "provider": self.name,
                "last_price": last,
                "bid_price": float(bid[0]),
                "ask_price": float(ask[0]),
                "volume_24h": float(vol[1] if len(vol) > 1 else vol[0]),
                "change_24h": change,
                "high_24h": float(high[1] if len(high) > 1 else high[0]),
                "low_24h": float(low[1] if len(low) > 1 else low[0]),
                "timestamp": datetime.now(timezone.utc),
            }
        raise MarketDataError(
            f"kraken: no real ticker for {symbol} (tried {list(resolved.candidates)}): {last_error}"
        )

    def fetch_ohlcv(self, symbol: str, interval: str = "1d", limit: int = 100) -> List[Dict[str, Any]]:
        resolved = self.resolve_symbol(symbol)
        minutes = _KRAKEN_INTERVAL.get(interval) or INTERVAL_MINUTES.get(interval)
        if minutes is None:
            raise MarketDataError(f"kraken: unsupported interval {interval}")
        last_error: Optional[Exception] = None
        for venue_symbol in resolved.candidates:
            try:
                result = self._public(
                    "/0/public/OHLC",
                    {"pair": venue_symbol, "interval": int(minutes)},
                )
                parsed = self._pair_payload(result)
            except MarketDataError as exc:
                last_error = exc
                continue
            rows = parsed["data"]
            if not isinstance(rows, list) or not rows:
                last_error = MarketDataError(f"kraken empty OHLC for {venue_symbol}")
                continue
            # Kraken returns oldest→newest; trim to `limit` most recent.
            trimmed = rows[-int(limit) :]
            bars: List[Dict[str, Any]] = []
            for row in trimmed:
                # [time, open, high, low, close, vwap, volume, count]
                bars.append({
                    "timestamp": datetime.fromtimestamp(int(row[0]), tz=timezone.utc),
                    "open": float(row[1]),
                    "high": float(row[2]),
                    "low": float(row[3]),
                    "close": float(row[4]),
                    "volume": float(row[6]),
                    "symbol": resolved.canonical,
                    "venue_symbol": parsed["venue_symbol"],
                    "provider": self.name,
                })
            if bars:
                return bars
        raise MarketDataError(
            f"kraken: no real OHLCV for {symbol} (tried {list(resolved.candidates)}): {last_error}"
        )

    async def get_ticker(self, symbol: str) -> Dict[str, Any]:
        return await asyncio.to_thread(self.fetch_ticker, symbol)

    async def get_ohlcv(
        self, symbol: str, interval: str = "1d", limit: int = 100
    ) -> List[Dict[str, Any]]:
        return await asyncio.to_thread(self.fetch_ohlcv, symbol, interval, limit)
