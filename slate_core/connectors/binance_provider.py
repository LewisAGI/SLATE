"""Binance public market-data adapter (spot or USDT-M perp).

Wraps the same public REST the existing ``BinanceFetcher`` uses. Does **not**
replace ``binance_spot`` / ``binance_usdt_perpetual``; those stay as-is.
Fails closed — no $50,000 synthetic ticker.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from slate_core.config.constants import BINANCE_API_BASE, BINANCE_FUTURES_API_BASE

from .base import MarketDataError, MarketDataProvider, ResolvedSymbol
from .http_util import get_json

_BINANCE_INTERVALS = {
    "1m": "1m",
    "5m": "5m",
    "15m": "15m",
    "30m": "30m",
    "1h": "1h",
    "4h": "4h",
    "6h": "6h",
    "1d": "1d",
}


class BinanceDataProvider(MarketDataProvider):
    name = "binance"

    def __init__(self, use_futures: bool = True, session_get=None):
        # Default futures: SLATE's primary book is SOLUSDT perp.
        self.use_futures = use_futures
        self._get_json = session_get or get_json
        self.base_url = BINANCE_FUTURES_API_BASE if use_futures else BINANCE_API_BASE
        self.ticker_path = "/fapi/v1/ticker/24hr" if use_futures else "/api/v3/ticker/24hr"
        self.klines_path = "/fapi/v1/klines" if use_futures else "/api/v3/klines"

    def _request(self, path: str, params: Optional[Dict[str, Any]] = None) -> Any:
        return self._get_json(f"{self.base_url}{path}", params=params)

    def _native(self, symbol: str) -> ResolvedSymbol:
        return self.resolve_symbol(symbol)

    def fetch_ticker(self, symbol: str) -> Dict[str, Any]:
        resolved = self._native(symbol)
        last_error: Optional[Exception] = None
        for venue_symbol in resolved.candidates:
            try:
                data = self._request(self.ticker_path, {"symbol": venue_symbol})
            except MarketDataError as exc:
                last_error = exc
                continue
            if not isinstance(data, dict) or "lastPrice" not in data:
                last_error = MarketDataError(f"binance ticker missing lastPrice for {venue_symbol}")
                continue
            last = float(data["lastPrice"])
            if last <= 0:
                last_error = MarketDataError(f"binance ticker lastPrice<=0 for {venue_symbol}")
                continue
            return {
                "symbol": resolved.canonical,
                "venue_symbol": venue_symbol,
                "provider": self.name,
                "last_price": last,
                "bid_price": float(data.get("bidPrice") or 0),
                "ask_price": float(data.get("askPrice") or 0),
                "volume_24h": float(data.get("volume") or 0),
                "change_24h": float(data.get("priceChangePercent") or 0),
                "high_24h": float(data.get("highPrice") or 0),
                "low_24h": float(data.get("lowPrice") or 0),
                "timestamp": datetime.now(timezone.utc),
            }
        raise MarketDataError(
            f"binance: no real ticker for {symbol} (tried {list(resolved.candidates)}): {last_error}"
        )

    def fetch_ohlcv(self, symbol: str, interval: str = "1d", limit: int = 100) -> List[Dict[str, Any]]:
        resolved = self._native(symbol)
        mapped = _BINANCE_INTERVALS.get(interval, interval)
        last_error: Optional[Exception] = None
        for venue_symbol in resolved.candidates:
            try:
                data = self._request(
                    self.klines_path,
                    {"symbol": venue_symbol, "interval": mapped, "limit": min(int(limit), 1500)},
                )
            except MarketDataError as exc:
                last_error = exc
                continue
            if not isinstance(data, list) or not data:
                last_error = MarketDataError(f"binance empty klines for {venue_symbol}")
                continue
            bars: List[Dict[str, Any]] = []
            for row in data:
                bars.append({
                    "timestamp": datetime.fromtimestamp(int(row[0]) / 1000, tz=timezone.utc),
                    "open": float(row[1]),
                    "high": float(row[2]),
                    "low": float(row[3]),
                    "close": float(row[4]),
                    "volume": float(row[5]),
                    "symbol": resolved.canonical,
                    "venue_symbol": venue_symbol,
                    "provider": self.name,
                })
            if bars:
                return bars
        raise MarketDataError(
            f"binance: no real OHLCV for {symbol} (tried {list(resolved.candidates)}): {last_error}"
        )

    async def get_ticker(self, symbol: str) -> Dict[str, Any]:
        return await asyncio.to_thread(self.fetch_ticker, symbol)

    async def get_ohlcv(
        self, symbol: str, interval: str = "1d", limit: int = 100
    ) -> List[Dict[str, Any]]:
        return await asyncio.to_thread(self.fetch_ohlcv, symbol, interval, limit)
