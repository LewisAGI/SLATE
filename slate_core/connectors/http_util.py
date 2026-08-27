"""Tiny public-REST helper for venue adapters. No synthetic payloads."""

from __future__ import annotations

from typing import Any, Dict, Optional
import requests

from .base import MarketDataError

_DEFAULT_HEADERS = {
    "Accept": "application/json",
    "User-Agent": "SLATE/2.0 paper-data",
}


def get_json(
    url: str,
    params: Optional[Dict[str, Any]] = None,
    timeout: float = 10.0,
    headers: Optional[Dict[str, str]] = None,
) -> Any:
    merged = dict(_DEFAULT_HEADERS)
    if headers:
        merged.update(headers)
    try:
        response = requests.get(url, params=params, timeout=timeout, headers=merged)
    except requests.RequestException as exc:
        raise MarketDataError(f"request failed: {url}: {exc}") from exc
    if response.status_code != 200:
        raise MarketDataError(
            f"{url} returned HTTP {response.status_code}: {response.text[:200]}"
        )
    try:
        return response.json()
    except ValueError as exc:
        raise MarketDataError(f"{url} returned non-JSON") from exc
