"""Deribit public market-data adapter (perpetual / linear only).

Public REST (no API key, no auth headers, production host by default):

- Ticker:  ``GET https://www.deribit.com/api/v2/public/ticker?instrument_name=``
- Charts:  ``GET .../public/get_tradingview_chart_data`` (start/end/resolution)
- Listed:  ``GET .../public/get_instruments?currency=any&kind=future``

JSON-RPC wraps every body in ``{jsonrpc, result, error}``. Fail closed on
error, missing ``last_price``, ``status=no_data``, or empty candles. Never
invent a ticker (no synthetic $50k).

Canonical map (this pass):

- ``BTCUSDT`` → ``BTC-PERPETUAL``, then linear ``BTC_USDC-PERPETUAL``
- ``ETHUSDT`` → ``ETH-PERPETUAL``, then linear ``ETH_USDC-PERPETUAL``
- ``SOLUSDT`` and other alts → ``MarketDataError`` (do not invent a SOL book)
- Options (``BTC-27JUN26-100000-C``) are out of scope

Deribit ``get_tradingview_chart_data`` returns at most ~5000 bars per page
(observed 5001 inclusive). ``limit > 5000`` must paginate with start/end or
fail closed. Paper only — this module never places orders.
"""

from __future__ import annotations

import asyncio
import re
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Set

from slate_core.config.constants import DERIBIT_API_BASE

from .base import (
    MarketDataError,
    MarketDataProvider,
    ResolvedSymbol,
    VENUE_SYMBOL_CANDIDATES,
    resolve_canonical_symbol,
)
from .http_util import get_json

# Observed production cap: 5001 ticks (5000 intervals + current bar).
MAX_CANDLES = 5000
_DEFAULT_PAGE_SLEEP = 0.12

# Documented resolution enum. 4h (240) is not listed — fail closed rather
# than invent aggregated bars or silently serve 3h/6h as 4h.
_DERIBIT_RESOLUTION = {
    "1m": "1",
    "5m": "5",
    "15m": "15",
    "30m": "30",
    "1h": "60",
    "6h": "360",
    "1d": "1D",
}

_OPTION_NAME = re.compile(
    r"^[A-Z0-9]+-\d{1,2}[A-Z]{3}\d{2}-\d+-[CP]$",
    re.IGNORECASE,
)
_BTC_ETH_PERP_PREFIXES = ("BTC-", "BTC_", "ETH-", "ETH_")


def _is_btc_eth_perpetual(instrument_name: str) -> bool:
    name = (instrument_name or "").strip().upper()
    if not name.endswith("PERPETUAL"):
        return False
    return name.startswith(_BTC_ETH_PERP_PREFIXES)


def _looks_like_option(instrument_name: str) -> bool:
    return bool(_OPTION_NAME.match((instrument_name or "").strip().upper()))


def _resolution_ms(resolution: str) -> int:
    if resolution == "1D":
        return 86_400_000
    return int(resolution) * 60_000


class DeribitDataProvider(MarketDataProvider):
    name = "deribit"

    def __init__(
        self,
        session_get=None,
        page_sleep: float = _DEFAULT_PAGE_SLEEP,
        base_url: Optional[str] = None,
    ):
        self._get_json = session_get or get_json
        # Production public host. Testnet is https://test.deribit.com/api/v2/
        # and is opt-in via base_url only — never the default.
        self.base_url = (base_url or DERIBIT_API_BASE).rstrip("/")
        self._page_sleep = float(page_sleep)
        self._listed_perpetuals: Optional[Set[str]] = None

    def resolve_symbol(self, canonical: str) -> ResolvedSymbol:
        """Map BTCUSDT/ETHUSDT onto listed perps. Fail closed for alts/options."""
        raw = (canonical or "").strip()
        if _looks_like_option(raw):
            raise MarketDataError(
                f"deribit: options are out of scope this pass ({raw!r})"
            )
        resolved = resolve_canonical_symbol(self.name, raw)
        table = VENUE_SYMBOL_CANDIDATES.get(self.name, {})
        if resolved.canonical in table:
            return resolved
        native = resolved.venue_symbol
        if _is_btc_eth_perpetual(native):
            return ResolvedSymbol(
                canonical=resolved.canonical,
                venue=self.name,
                venue_symbol=native,
                candidates=(native,),
            )
        raise MarketDataError(
            f"deribit: no listed BTC/ETH perpetual for {raw!r} "
            "(SOLUSDT and other alts are fail-closed; options are out of scope)"
        )

    def _rpc(self, path: str, params: Optional[Dict[str, Any]] = None) -> Any:
        payload = self._get_json(f"{self.base_url}{path}", params=params)
        if not isinstance(payload, dict):
            raise MarketDataError("deribit: unexpected payload")
        error = payload.get("error")
        if error:
            raise MarketDataError(f"deribit error: {error}")
        if "result" not in payload:
            raise MarketDataError("deribit: missing result")
        result = payload["result"]
        if result is None:
            raise MarketDataError("deribit: empty result")
        return result

    def _listed_active_perpetuals(self) -> Set[str]:
        """Cache active future perpetuals from public/get_instruments."""
        if self._listed_perpetuals is not None:
            return self._listed_perpetuals
        result = self._rpc(
            "/public/get_instruments",
            {"currency": "any", "kind": "future"},
        )
        if not isinstance(result, list) or not result:
            raise MarketDataError("deribit: get_instruments returned no futures")
        names: Set[str] = set()
        for row in result:
            if not isinstance(row, dict):
                continue
            name = row.get("instrument_name")
            if not name or not row.get("is_active"):
                continue
            if row.get("kind") == "option":
                continue
            if not str(name).upper().endswith("PERPETUAL"):
                continue
            names.add(str(name))
        if not names:
            raise MarketDataError("deribit: no active perpetuals listed")
        self._listed_perpetuals = names
        return names

    def _candidates_that_exist(self, candidates: Sequence[str]) -> List[str]:
        listed = self._listed_active_perpetuals()
        existing = [name for name in candidates if name in listed]
        if not existing:
            raise MarketDataError(
                f"deribit: none of {list(candidates)} are active listed perpetuals"
            )
        return existing

    def fetch_ticker(self, symbol: str) -> Dict[str, Any]:
        resolved = self.resolve_symbol(symbol)
        last_error: Optional[Exception] = None
        try:
            venue_names = self._candidates_that_exist(resolved.candidates)
        except MarketDataError as exc:
            raise MarketDataError(
                f"deribit: no real ticker for {symbol}: {exc}"
            ) from exc
        for venue_symbol in venue_names:
            try:
                data = self._rpc(
                    "/public/ticker", {"instrument_name": venue_symbol}
                )
            except MarketDataError as exc:
                last_error = exc
                continue
            if not isinstance(data, dict):
                last_error = MarketDataError(
                    f"deribit ticker not a dict for {venue_symbol}"
                )
                continue
            if data.get("kind") == "option":
                last_error = MarketDataError(
                    f"deribit: options are out of scope ({venue_symbol})"
                )
                continue
            returned_name = data.get("instrument_name") or venue_symbol
            if _looks_like_option(str(returned_name)):
                last_error = MarketDataError(
                    f"deribit: options are out of scope ({returned_name})"
                )
                continue
            try:
                last = data.get("last_price")
                if last is None:
                    raise ValueError("missing last_price")
                last_px = float(last)
                if last_px <= 0:
                    raise ValueError("non-positive last_price")
            except (TypeError, ValueError) as exc:
                last_error = MarketDataError(
                    f"deribit ticker parse {venue_symbol}: {exc}"
                )
                continue
            stats = data.get("stats") if isinstance(data.get("stats"), dict) else {}
            ts = data.get("timestamp")
            if isinstance(ts, (int, float)):
                stamp = datetime.fromtimestamp(ts / 1000.0, tz=timezone.utc)
            else:
                stamp = datetime.now(timezone.utc)
            return {
                "symbol": resolved.canonical,
                "venue_symbol": str(returned_name),
                "provider": self.name,
                "last_price": last_px,
                "bid_price": float(data.get("best_bid_price") or 0),
                "ask_price": float(data.get("best_ask_price") or 0),
                "volume_24h": float(stats.get("volume") or 0),
                "change_24h": float(stats.get("price_change") or 0),
                "high_24h": float(stats.get("high") or 0),
                "low_24h": float(stats.get("low") or 0),
                "timestamp": stamp,
            }
        raise MarketDataError(
            f"deribit: no real ticker for {symbol} "
            f"(tried {list(resolved.candidates)}): {last_error}"
        )

    def _chart_to_bars(
        self,
        result: Dict[str, Any],
        *,
        canonical: str,
        venue_symbol: str,
    ) -> List[Dict[str, Any]]:
        status = result.get("status")
        ticks = result.get("ticks")
        if status == "no_data" or ticks == []:
            return []
        if status not in {None, "ok"}:
            raise MarketDataError(
                f"deribit chart status {status!r} for {venue_symbol}"
            )
        opens = result.get("open")
        highs = result.get("high")
        lows = result.get("low")
        closes = result.get("close")
        volumes = result.get("volume")
        if not isinstance(ticks, list) or not ticks:
            raise MarketDataError(f"deribit empty chart ticks for {venue_symbol}")
        columns = (opens, highs, lows, closes, volumes)
        if any(not isinstance(col, list) for col in columns):
            raise MarketDataError(
                f"deribit chart missing OHLC columns for {venue_symbol}"
            )
        if any(len(col) != len(ticks) for col in columns):
            raise MarketDataError(
                f"deribit chart column length mismatch for {venue_symbol}"
            )
        bars: List[Dict[str, Any]] = []
        for i, tick in enumerate(ticks):
            bars.append({
                "timestamp": datetime.fromtimestamp(
                    int(tick) / 1000.0, tz=timezone.utc
                ),
                "open": float(opens[i]),
                "high": float(highs[i]),
                "low": float(lows[i]),
                "close": float(closes[i]),
                "volume": float(volumes[i]),
                "symbol": canonical,
                "venue_symbol": venue_symbol,
                "provider": self.name,
            })
        bars.sort(key=lambda b: b["timestamp"])
        return bars

    def _fetch_chart_page(
        self,
        venue_symbol: str,
        resolution: str,
        start_ms: int,
        end_ms: int,
        *,
        canonical: str,
    ) -> List[Dict[str, Any]]:
        result = self._rpc(
            "/public/get_tradingview_chart_data",
            {
                "instrument_name": venue_symbol,
                "start_timestamp": int(start_ms),
                "end_timestamp": int(end_ms),
                "resolution": resolution,
            },
        )
        if not isinstance(result, dict):
            raise MarketDataError(f"deribit chart not an object for {venue_symbol}")
        return self._chart_to_bars(
            result, canonical=canonical, venue_symbol=venue_symbol
        )

    def _fetch_charts_paginated(
        self,
        venue_symbol: str,
        resolution: str,
        limit: int,
        *,
        canonical: str,
    ) -> List[Dict[str, Any]]:
        """Walk backwards in MAX_CANDLES windows. Fail closed if short."""
        collected: Dict[int, Dict[str, Any]] = {}
        end_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
        step_ms = _resolution_ms(resolution)
        window_ms = step_ms * MAX_CANDLES
        max_pages = (int(limit) + MAX_CANDLES - 1) // MAX_CANDLES + 1
        pages = 0

        while len(collected) < limit and pages < max_pages:
            if pages and self._page_sleep > 0:
                time.sleep(self._page_sleep)
            start_ms = end_ms - window_ms
            page = self._fetch_chart_page(
                venue_symbol,
                resolution,
                start_ms,
                end_ms,
                canonical=canonical,
            )
            pages += 1
            if not page:
                break
            for bar in page:
                collected[int(bar["timestamp"].timestamp())] = bar
            oldest_ms = int(min(bar["timestamp"].timestamp() for bar in page) * 1000)
            next_end = oldest_ms - 1
            if next_end >= end_ms:
                break
            end_ms = next_end
            if len(page) < MAX_CANDLES:
                break  # history exhausted

        bars = [collected[k] for k in sorted(collected)]
        if len(bars) < limit:
            raise MarketDataError(
                f"deribit: requested {limit} candles for {venue_symbol} but only "
                f"got {len(bars)} (chart cap {MAX_CANDLES}/request; "
                f"pagination exhausted after {pages} page(s))"
            )
        return bars[-limit:]

    def fetch_ohlcv(
        self, symbol: str, interval: str = "1d", limit: int = 100
    ) -> List[Dict[str, Any]]:
        resolved = self.resolve_symbol(symbol)
        resolution = _DERIBIT_RESOLUTION.get(interval)
        if resolution is None:
            raise MarketDataError(
                f"deribit: unsupported interval {interval} "
                f"(documented resolutions: {sorted(_DERIBIT_RESOLUTION)})"
            )
        try:
            limit_n = int(limit)
        except (TypeError, ValueError) as exc:
            raise MarketDataError(f"deribit: invalid limit {limit!r}") from exc
        if limit_n <= 0:
            raise MarketDataError("deribit: limit must be positive")

        try:
            venue_names = self._candidates_that_exist(resolved.candidates)
        except MarketDataError as exc:
            raise MarketDataError(
                f"deribit: no real OHLCV for {symbol}: {exc}"
            ) from exc

        last_error: Optional[Exception] = None
        for venue_symbol in venue_names:
            try:
                if limit_n <= MAX_CANDLES:
                    end_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
                    start_ms = end_ms - (_resolution_ms(resolution) * limit_n)
                    bars = self._fetch_chart_page(
                        venue_symbol,
                        resolution,
                        start_ms,
                        end_ms,
                        canonical=resolved.canonical,
                    )
                    if not bars:
                        raise MarketDataError(
                            f"deribit empty chart for {venue_symbol}"
                        )
                    bars = bars[-limit_n:]
                else:
                    bars = self._fetch_charts_paginated(
                        venue_symbol,
                        resolution,
                        limit_n,
                        canonical=resolved.canonical,
                    )
            except MarketDataError as exc:
                last_error = exc
                continue
            if bars:
                return bars
        raise MarketDataError(
            f"deribit: no real OHLCV for {symbol} "
            f"(tried {list(resolved.candidates)}): {last_error}"
        )

    async def get_ticker(self, symbol: str) -> Dict[str, Any]:
        return await asyncio.to_thread(self.fetch_ticker, symbol)

    async def get_ohlcv(
        self, symbol: str, interval: str = "1d", limit: int = 100
    ) -> List[Dict[str, Any]]:
        return await asyncio.to_thread(self.fetch_ohlcv, symbol, interval, limit)
