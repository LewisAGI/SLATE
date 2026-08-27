"""Replay a recorded real OHLCV series through the provider interface.

Used by tests and paper dry-runs when live keys / network are absent.
Bars must come from a real cache or a previously fetched public response —
this class does not generate prices.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Sequence

from .base import MarketDataError, MarketDataProvider


class ReplayMarketDataProvider(MarketDataProvider):
    name = "replay"

    def __init__(
        self,
        bars: Sequence[Dict[str, Any]],
        *,
        canonical: str = "SOLUSDT",
        venue_symbol: str = "SOLUSDT",
        source_provider: str = "replay",
    ):
        if not bars:
            raise MarketDataError("replay provider requires at least one real bar")
        self._bars = [dict(b) for b in bars]
        self._canonical = canonical
        self._venue_symbol = venue_symbol
        self._source = source_provider
        self._cursor = len(self._bars) - 1

    def set_cursor(self, index: int) -> None:
        if not (0 <= index < len(self._bars)):
            raise MarketDataError(f"replay cursor {index} out of range")
        self._cursor = index

    def _bar_at(self, index: int) -> Dict[str, Any]:
        bar = self._bars[index]
        close = float(bar["close"])
        if close <= 0:
            raise MarketDataError("replay bar has non-positive close — not real data")
        return bar

    def fetch_ticker(self, symbol: str) -> Dict[str, Any]:
        bar = self._bar_at(self._cursor)
        close = float(bar["close"])
        open_px = float(bar.get("open") or close)
        change = ((close - open_px) / open_px * 100.0) if open_px else 0.0
        ts = bar.get("timestamp")
        if isinstance(ts, str):
            ts = datetime.fromisoformat(ts)
        return {
            "symbol": self._canonical,
            "venue_symbol": self._venue_symbol,
            "provider": self.name,
            "source_provider": self._source,
            "last_price": close,
            "bid_price": close,
            "ask_price": close,
            "volume_24h": float(bar.get("volume") or 0),
            "change_24h": change,
            "high_24h": float(bar.get("high") or close),
            "low_24h": float(bar.get("low") or close),
            "timestamp": ts or datetime.utcnow(),
        }

    def fetch_ohlcv(self, symbol: str, interval: str = "1d", limit: int = 100) -> List[Dict[str, Any]]:
        del interval  # replay is already sampled
        window = self._bars[max(0, self._cursor + 1 - int(limit)) : self._cursor + 1]
        out: List[Dict[str, Any]] = []
        for bar in window:
            rec = dict(bar)
            rec.setdefault("symbol", self._canonical)
            rec.setdefault("venue_symbol", self._venue_symbol)
            rec.setdefault("provider", self.name)
            out.append(rec)
        return out

    async def get_ticker(self, symbol: str) -> Dict[str, Any]:
        return self.fetch_ticker(symbol)

    async def get_ohlcv(
        self, symbol: str, interval: str = "1d", limit: int = 100
    ) -> List[Dict[str, Any]]:
        return self.fetch_ohlcv(symbol, interval=interval, limit=limit)
