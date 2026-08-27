"""Shared market-data / paper-execution provider contract.

SLATE historically hardcoded Binance + SOLUSDT and paper-entered LONG only.
This module is the venue-agnostic surface:

- Canonical symbols (``SOLUSDT``, ``BTCUSDT``, …) map to the nearest listed
  pair on each venue (Kraken ``SOLUSD`` / ``SOLUSDT``, Coinbase ``SOL-USD``).
- Providers fetch **real** public market data. They must not invent prices.
- Orders stay paper-only. Live keys are not required for the data path.

Existing Binance connectors keep working; they are wrapped, not replaced.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Iterable, List, Literal, Mapping, Optional, Sequence
import os

PositionSide = Literal["long", "short"]
DecisionType = Literal["ENTER_LONG", "ENTER_SHORT", "EXIT", "HOLD"]

# Canonical SLATE symbol → ordered venue-native candidates (first is preferred).
# Kraken BTC is XBT; Coinbase uses hyphenated products. USDT pair first, USD
# fallback — Coinbase/Kraken often list SOL-USD more reliably than SOL-USDT.
VENUE_SYMBOL_CANDIDATES: Dict[str, Dict[str, List[str]]] = {
    "binance": {
        "SOLUSDT": ["SOLUSDT"],
        "BTCUSDT": ["BTCUSDT"],
        "ETHUSDT": ["ETHUSDT"],
    },
    "kraken": {
        "SOLUSDT": ["SOLUSDT", "SOLUSD"],
        "BTCUSDT": ["XBTUSDT", "XBTUSD", "XXBTZUSD"],
        "ETHUSDT": ["ETHUSDT", "ETHUSD"],
    },
    "coinbase": {
        "SOLUSDT": ["SOL-USDT", "SOL-USD"],
        "BTCUSDT": ["BTC-USDT", "BTC-USD"],
        "ETHUSDT": ["ETH-USDT", "ETH-USD"],
    },
}

INTERVAL_MINUTES = {
    "1m": 1,
    "5m": 5,
    "15m": 15,
    "30m": 30,
    "1h": 60,
    "4h": 240,
    "6h": 360,
    "1d": 1440,
}


@dataclass(frozen=True)
class ResolvedSymbol:
    """Canonical SLATE symbol mapped onto a venue-native pair."""

    canonical: str
    venue: str
    venue_symbol: str
    candidates: Sequence[str] = field(default_factory=tuple)


@dataclass
class NormalizedTicker:
    """Common ticker shape used by paper execution and MarketDataManager."""

    symbol: str
    venue_symbol: str
    provider: str
    last_price: float
    bid_price: float = 0.0
    ask_price: float = 0.0
    volume_24h: float = 0.0
    change_24h: float = 0.0
    high_24h: float = 0.0
    low_24h: float = 0.0
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "venue_symbol": self.venue_symbol,
            "provider": self.provider,
            "last_price": self.last_price,
            "bid_price": self.bid_price,
            "ask_price": self.ask_price,
            "volume_24h": self.volume_24h,
            "change_24h": self.change_24h,
            "high_24h": self.high_24h,
            "low_24h": self.low_24h,
            "timestamp": self.timestamp,
        }


@dataclass
class OHLCVBar:
    """One real OHLCV bar. Never synthesized."""

    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    symbol: str = ""
    venue_symbol: str = ""
    provider: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
            "symbol": self.symbol,
            "venue_symbol": self.venue_symbol,
            "provider": self.provider,
        }


class MarketDataError(RuntimeError):
    """Raised when a provider cannot return real market data."""


class MarketDataProvider(ABC):
    """Venue adapter: real public market data + paper-only execution book.

    Implementations must fail closed (raise ``MarketDataError``) instead of
    inventing prices. Paper fills live on ``PaperTradingBook``, not the venue.
    """

    name: str = "unknown"

    def resolve_symbol(self, canonical: str) -> ResolvedSymbol:
        """Map SOLUSDT (etc.) to the nearest listed pair. No network."""
        return resolve_canonical_symbol(self.name, canonical)

    @abstractmethod
    async def get_ticker(self, symbol: str) -> Dict[str, Any]:
        """Return a NormalizedTicker-compatible dict for ``symbol``."""

    @abstractmethod
    async def get_ohlcv(
        self,
        symbol: str,
        interval: str = "1d",
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Return oldest→newest real OHLCV bars. Empty list is a hard miss."""

    async def get_candles(
        self,
        symbol: str,
        interval: str = "1h",
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Alias used by legacy Binance connectors."""
        return await self.get_ohlcv(symbol, interval=interval, limit=limit)


def resolve_canonical_symbol(venue: str, canonical: str) -> ResolvedSymbol:
    """Return preferred venue-native symbol plus fallbacks.

    Unknown venues / symbols pass the canonical through unchanged so Binance
    SOLUSDT keep working even if a caller invents a pair.
    """
    venue_key = (venue or "").strip().lower()
    symbol = (canonical or "").strip().upper().replace("-", "")
    # Coinbase-style incoming "SOL-USD" → try as-is via original too.
    table = VENUE_SYMBOL_CANDIDATES.get(venue_key, {})
    candidates = table.get(symbol)
    if not candidates:
        # Already venue-native (e.g. SOL-USD, SOLUSD, XBTUSD)
        native = canonical.strip() if canonical else symbol
        return ResolvedSymbol(
            canonical=symbol or native,
            venue=venue_key or venue,
            venue_symbol=native,
            candidates=(native,),
        )
    return ResolvedSymbol(
        canonical=symbol,
        venue=venue_key,
        venue_symbol=candidates[0],
        candidates=tuple(candidates),
    )


def infer_entry_side(source: Any) -> PositionSide:
    """Read LONG/SHORT from existing discovery/signal fields. Default long.

    Does not invent a strategy — only interprets fields the rest of SLATE
    already writes (``entry_type``, signed ``signal``, ``validation_details``).
    Unknown / two-way labels (``LONG_SHORT``, ``MARKET_NEUTRAL``) stay long so
    current long-only callers do not flip.
    """
    tokens: List[Any] = []
    if source is None:
        return "long"
    if isinstance(source, str):
        tokens.append(source)
    elif isinstance(source, Mapping):
        tokens.extend(
            source.get(k)
            for k in ("entry_type", "side", "position_side", "signal", "decision_type")
        )
        details = source.get("validation_details") or source.get("strategy_design") or {}
        if isinstance(details, Mapping):
            tokens.extend(
                details.get(k)
                for k in ("entry_type", "side", "position_side", "signal")
            )
    else:
        for attr in ("entry_type", "side", "position_side", "signal", "decision_type"):
            tokens.append(getattr(source, attr, None))
        details = getattr(source, "validation_details", None) or {}
        if isinstance(details, Mapping):
            tokens.extend(
                details.get(k)
                for k in ("entry_type", "side", "position_side", "signal")
            )

    for raw in tokens:
        if raw is None:
            continue
        if isinstance(raw, (int, float)) and not isinstance(raw, bool):
            if raw < 0:
                return "short"
            if raw > 0:
                return "long"
            continue
        text = str(raw).strip().upper()
        if text in {"SHORT", "ENTER_SHORT", "SELL", "-1"}:
            return "short"
        if text in {"LONG", "ENTER_LONG", "BUY", "1"}:
            return "long"
    return "long"


def decision_type_for_side(side: PositionSide) -> DecisionType:
    return "ENTER_SHORT" if side == "short" else "ENTER_LONG"


def default_provider_name() -> str:
    return os.environ.get("SLATE_DATA_PROVIDER", "binance").strip().lower() or "binance"


def bars_to_records(bars: Iterable[OHLCVBar]) -> List[Dict[str, Any]]:
    return [b.to_dict() if isinstance(b, OHLCVBar) else dict(b) for b in bars]
