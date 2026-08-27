"""Venue connectors: Binance (existing) plus Kraken / Coinbase data adapters.

Paper trading only. Public market data. No synthetic prices on the new path.
"""

from .base import (
    MarketDataError,
    MarketDataProvider,
    infer_entry_side,
    resolve_canonical_symbol,
)
from .factory import available_providers, get_market_data_provider
from .paper_book import PaperPosition, PaperTradingBook

__all__ = [
    "MarketDataError",
    "MarketDataProvider",
    "PaperPosition",
    "PaperTradingBook",
    "available_providers",
    "get_market_data_provider",
    "infer_entry_side",
    "resolve_canonical_symbol",
]
