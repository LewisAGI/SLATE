"""Build a market-data provider by name.

    SLATE_DATA_PROVIDER=binance|kraken|coinbase|deribit|replay

Default is Binance (existing path). Kraken, Coinbase, and Deribit use public
REST and need no API keys for data. ``replay`` is constructed explicitly with
recorded real bars — ``get_market_data_provider("replay")`` raises so we never
silently invent a book.
"""

from __future__ import annotations

from typing import Dict, Iterable, Optional, Type

from .base import MarketDataError, MarketDataProvider, default_provider_name
from .binance_provider import BinanceDataProvider
from .coinbase import CoinbaseDataProvider
from .deribit import DeribitDataProvider
from .kraken import KrakenDataProvider

_PROVIDERS: Dict[str, Type[MarketDataProvider]] = {
    "binance": BinanceDataProvider,
    "kraken": KrakenDataProvider,
    "coinbase": CoinbaseDataProvider,
    "deribit": DeribitDataProvider,
}


def available_providers() -> Iterable[str]:
    return tuple(_PROVIDERS.keys())


def get_market_data_provider(
    name: Optional[str] = None,
    **kwargs,
) -> MarketDataProvider:
    key = (name or default_provider_name()).strip().lower()
    if key == "replay":
        raise MarketDataError(
            "replay provider must be constructed with recorded real bars "
            "(ReplayMarketDataProvider), not via the name factory"
        )
    cls = _PROVIDERS.get(key)
    if cls is None:
        raise MarketDataError(
            f"unknown data provider {key!r}; choose one of {sorted(_PROVIDERS)}"
        )
    return cls(**kwargs)
