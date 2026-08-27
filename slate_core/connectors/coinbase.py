"""Coinbase Exchange public market-data adapter.

Public REST (no API key):

- Ticker: ``GET https://api.exchange.coinbase.com/products/SOL-USD/ticker``
- Stats:  ``GET https://api.exchange.coinbase.com/products/SOL-USD/stats``
- Candles: ``GET .../products/SOL-USD/candles?granularity=86400``

Stay on the Exchange API host (``api.exchange.coinbase.com``). Do **not**
switch to Advanced Trade v3. Each candles request returns at most
``MAX_CANDLES`` (350) bars; ``limit > 350`` must paginate with ``start`` /
``end`` or fail closed. Never silently return fewer bars than requested.

Canonical ``SOLUSDT`` tries ``SOL-USDT`` then ``SOL-USD``. Coinbase does not
list USDT-M perps the way Binance does; the nearest spot pair is the data
path. Paper only — this module never places orders.

Candle array is ``[time, low, high, open, close, volume]``, newest first.
"""

from __future__ import annotations

import asyncio
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from slate_core.config.constants import COINBASE_EXCHANGE_API_BASE

from .base import MarketDataError, MarketDataProvider
from .http_util import get_json

# Exchange API hard cap. Public REST is 10 req/s per IP — page sleep stays under that.
MAX_CANDLES = 350
_DEFAULT_PAGE_SLEEP = 0.12

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


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class CoinbaseDataProvider(MarketDataProvider):
    name = "coinbase"

    def __init__(self, session_get=None, page_sleep: float = _DEFAULT_PAGE_SLEEP):
        self._get_json = session_get or get_json
        self.base_url = COINBASE_EXCHANGE_API_BASE
        self._page_sleep = float(page_sleep)

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

    def _candles_to_bars(
        self,
        rows: List[Any],
        *,
        canonical: str,
        venue_symbol: str,
    ) -> List[Dict[str, Any]]:
        bars: List[Dict[str, Any]] = []
        for row in rows:
            bars.append({
                "timestamp": datetime.fromtimestamp(int(row[0]), tz=timezone.utc),
                "low": float(row[1]),
                "high": float(row[2]),
                "open": float(row[3]),
                "close": float(row[4]),
                "volume": float(row[5]),
                "symbol": canonical,
                "venue_symbol": venue_symbol,
                "provider": self.name,
            })
        return bars

    def _fetch_candle_page(
        self,
        venue_symbol: str,
        granularity: int,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
    ) -> List[Any]:
        params: Dict[str, Any] = {"granularity": granularity}
        if start is not None or end is not None:
            if start is None or end is None:
                raise MarketDataError("coinbase: start and end must be sent together")
            params["start"] = _iso(start)
            params["end"] = _iso(end)
        data = self._get(f"/products/{venue_symbol}/candles", params)
        if not isinstance(data, list):
            raise MarketDataError(f"coinbase candles not a list for {venue_symbol}")
        if len(data) > MAX_CANDLES:
            # Venue violated its own cap — do not silently keep a truncated view.
            raise MarketDataError(
                f"coinbase: candles page returned {len(data)} > {MAX_CANDLES} for {venue_symbol}"
            )
        return data

    def _fetch_candles_paginated(
        self,
        venue_symbol: str,
        granularity: int,
        limit: int,
    ) -> List[Any]:
        """Walk backwards with start/end windows of at most MAX_CANDLES bars.

        Fails closed if pagination cannot assemble ``limit`` unique bars.
        """
        collected: Dict[int, Any] = {}
        end = datetime.now(timezone.utc)
        window = timedelta(seconds=granularity * MAX_CANDLES)
        max_pages = (int(limit) + MAX_CANDLES - 1) // MAX_CANDLES + 1
        pages = 0

        while len(collected) < limit and pages < max_pages:
            if pages and self._page_sleep > 0:
                time.sleep(self._page_sleep)
            start = end - window
            page = self._fetch_candle_page(
                venue_symbol, granularity, start=start, end=end
            )
            pages += 1
            if not page:
                break
            for row in page:
                collected[int(row[0])] = row
            oldest = min(int(row[0]) for row in page)
            next_end = datetime.fromtimestamp(oldest - 1, tz=timezone.utc)
            if next_end >= end:
                break
            end = next_end
            if len(page) < MAX_CANDLES:
                break  # history exhausted on this product

        rows = [collected[k] for k in sorted(collected)]
        if len(rows) < limit:
            raise MarketDataError(
                f"coinbase: requested {limit} candles for {venue_symbol} but only "
                f"got {len(rows)} (Exchange API cap {MAX_CANDLES}/request; "
                f"pagination exhausted after {pages} page(s))"
            )
        return rows[-limit:]

    def fetch_ohlcv(self, symbol: str, interval: str = "1d", limit: int = 100) -> List[Dict[str, Any]]:
        resolved = self.resolve_symbol(symbol)
        granularity = _COINBASE_GRANULARITY.get(interval)
        if granularity is None:
            raise MarketDataError(f"coinbase: unsupported interval {interval}")
        try:
            limit_n = int(limit)
        except (TypeError, ValueError) as exc:
            raise MarketDataError(f"coinbase: invalid limit {limit!r}") from exc
        if limit_n <= 0:
            raise MarketDataError("coinbase: limit must be positive")

        last_error: Optional[Exception] = None
        for venue_symbol in resolved.candidates:
            try:
                if limit_n <= MAX_CANDLES:
                    data = self._fetch_candle_page(venue_symbol, granularity)
                    if not data:
                        raise MarketDataError(f"coinbase empty candles for {venue_symbol}")
                    # Newest first → oldest first, then trim.
                    rows = sorted(data, key=lambda r: r[0])[-limit_n:]
                else:
                    rows = self._fetch_candles_paginated(
                        venue_symbol, granularity, limit_n
                    )
            except MarketDataError as exc:
                last_error = exc
                continue
            bars = self._candles_to_bars(
                rows, canonical=resolved.canonical, venue_symbol=venue_symbol
            )
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
