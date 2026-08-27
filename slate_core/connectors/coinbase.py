"""Coinbase Exchange public market-data adapter.

Public REST (no API key):

- Ticker: ``GET https://api.exchange.coinbase.com/products/SOL-USD/ticker``
- Stats:  ``GET https://api.exchange.coinbase.com/products/SOL-USD/stats``
- Candles: ``GET .../products/SOL-USD/candles?granularity=86400``

Canonical ``SOLUSDT`` tries ``SOL-USDT`` then ``SOL-USD``. Coinbase does not
list USDT-M perps the way Binance does; the nearest spot pair is the data
path. Paper only — this module never places orders.

Candle array is ``[time, low, high, open, close, volume]``, newest first.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from slate_core.config.constants import COINBASE_EXCHANGE_API_BASE

from .base import MarketDataError, MarketDataProvider
from .http_util import get_json

# Coinbase granularities are seconds. No native 4h — use 1h (3600) rather than
# invent aggregated bars.
_COINBASE_GRANULARITY = {
    "1m": 60,
    "5m": 300,
    "15m": 900,
    "1h": 3600,
    "4h": 3600,
    "6h": 21600,
    "1d": 86400,
}


class CoinbaseDataProvider(MarketDataProvider):
    name = "coinbase"

    def __init__(self, session_get=None):
        self._get_json = session_get or get_json
        self.base_url = COINBASE_EXCHANGE_API_BASE

    def _get(self, path: str, params: Optional[Dict[str, Any]] = None) -> Any:
        return self._get_json(f"{self.base_url}{path}", params=params)

    def fetch_ticker(self, symbol: str) -> Dict[str, Any]:
        resolved = self.resolve_symbol(symbol)
        last_error: Optional[Exception] = None
        for venue_symbol in resolved.candidates:
            try:
                ticker = self._get(f"/products/{venue_symbol}/ticker")
                stats = self._get(f"/products/{venue_symbol}/stats")
            except MarketDataError as exc:
                last_error = exc
                continue
            if not isinstance(ticker, dict) or "price" not in ticker:
                last_error = MarketDataError(f"coinbase ticker missing price for {venue_symbol}")
                continue
            try:
                last = float(ticker["price"])
                if last <= 0:
                    raise ValueError("non-positive last")
                open_px = float((stats or {}).get("open") or last)
                change = ((last - open_px) / open_px * 100.0) if open_px else 0.0
            except (TypeError, ValueError) as exc:
                last_error = MarketDataError(f"coinbase ticker parse {venue_symbol}: {exc}")
                continue
            stats = stats if isinstance(stats, dict) else {}
            return {
                "symbol": resolved.canonical,
                "venue_symbol": venue_symbol,
                "provider": self.name,
                "last_price": last,
                "bid_price": float(ticker.get("bid") or 0),
                "ask_price": float(ticker.get("ask") or 0),
                "volume_24h": float(ticker.get("volume") or stats.get("volume") or 0),
                "change_24h": change,
                "high_24h": float(stats.get("high") or 0),
                "low_24h": float(stats.get("low") or 0),
                "timestamp": datetime.now(timezone.utc),
            }
        raise MarketDataError(
            f"coinbase: no real ticker for {symbol} (tried {list(resolved.candidates)}): {last_error}"
        )

    def fetch_ohlcv(self, symbol: str, interval: str = "1d", limit: int = 100) -> List[Dict[str, Any]]:
        resolved = self.resolve_symbol(symbol)
        granularity = _COINBASE_GRANULARITY.get(interval)
        if granularity is None:
            raise MarketDataError(f"coinbase: unsupported interval {interval}")
        last_error: Optional[Exception] = None
        for venue_symbol in resolved.candidates:
            try:
                data = self._get(
                    f"/products/{venue_symbol}/candles",
                    {"granularity": granularity},
                )
            except MarketDataError as exc:
                last_error = exc
                continue
            if not isinstance(data, list) or not data:
                last_error = MarketDataError(f"coinbase empty candles for {venue_symbol}")
                continue
            # Newest first → oldest first, then trim.
            rows = sorted(data, key=lambda r: r[0])
            trimmed = rows[-int(limit) :]
            bars: List[Dict[str, Any]] = []
            for row in trimmed:
                bars.append({
                    "timestamp": datetime.fromtimestamp(int(row[0]), tz=timezone.utc),
                    "low": float(row[1]),
                    "high": float(row[2]),
                    "open": float(row[3]),
                    "close": float(row[4]),
                    "volume": float(row[5]),
                    "symbol": resolved.canonical,
                    "venue_symbol": venue_symbol,
                    "provider": self.name,
                })
            if bars:
                return bars
        raise MarketDataError(
            f"coinbase: no real OHLCV for {symbol} (tried {list(resolved.candidates)}): {last_error}"
        )

    async def get_ticker(self, symbol: str) -> Dict[str, Any]:
        return await asyncio.to_thread(self.fetch_ticker, symbol)

    async def get_ohlcv(
        self, symbol: str, interval: str = "1d", limit: int = 100
    ) -> List[Dict[str, Any]]:
        return await asyncio.to_thread(self.fetch_ohlcv, symbol, interval, limit)
