"""Long and short paper positions share one execution path.

Does not invent a strategy: side is read from existing discovery fields
(``entry_type`` / signed ``signal``). Longs must keep working.
"""

from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from slate_core.autonomous.config import (
    AutonomousConfig,
    Discovery,
    DiscoveryCategory,
)
from slate_core.autonomous.trading_executor import TradingExecutor
from slate_core.connectors.base import infer_entry_side
from slate_core.connectors.paper_book import PaperTradingBook
from slate_core.connectors.replay import ReplayMarketDataProvider

_REAL_SOL = Path("sol_data_cache/SOLUSDT_perpetual_1h_6m.csv")


def _replay_from_real_sol() -> ReplayMarketDataProvider:
    assert _REAL_SOL.exists(), f"real SOL cache missing at {_REAL_SOL}"
    df = pd.read_json(_REAL_SOL)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    row = df.sort_values("timestamp").iloc[-1]
    bar = {
        "timestamp": row["timestamp"].to_pydatetime() if hasattr(row["timestamp"], "to_pydatetime") else row["timestamp"],
        "open": float(row["open"]),
        "high": float(row["high"]),
        "low": float(row["low"]),
        "close": float(row["close"]),
        "volume": float(row["volume"]),
    }
    return ReplayMarketDataProvider([bar], canonical="SOLUSDT", source_provider="binance")


def _discovery(**overrides) -> Discovery:
    defaults = dict(
        question="edge?",
        answer="existing signal",
        category=DiscoveryCategory.STRATEGY_EDGE,
        confidence=0.9,
        novelty_score=0.5,
        profitability_score=0.8,
        symbol="SOLUSDT",
        timeframe="1d",
        regime_conditions={},
        total_return_pct=8.0,
        sharpe_ratio=1.2,
        max_drawdown_pct=5.0,
        win_rate=0.6,
        profit_factor=1.5,
        transaction_costs_usdt=2.0,
        profit_after_costs=50.0,
        realistic_edge=True,
        discovery_method="test",
    )
    defaults.update(overrides)
    return Discovery(**defaults)


def test_infer_entry_side_reads_existing_fields_only():
    assert infer_entry_side(None) == "long"
    assert infer_entry_side(_discovery()) == "long"  # default keeps longs
    assert infer_entry_side(_discovery(entry_type="SHORT")) == "short"
    assert infer_entry_side(_discovery(entry_type="LONG")) == "long"
    assert infer_entry_side(_discovery(validation_details={"signal": -1})) == "short"
    assert infer_entry_side(_discovery(validation_details={"entry_type": "SHORT"})) == "short"
    assert infer_entry_side({"entry_type": "LONG_SHORT"}) == "long"
    assert infer_entry_side({"entry_type": "MARKET_NEUTRAL"}) == "long"
    assert infer_entry_side(SimpleNamespace(entry_type="SELL")) == "short"


def test_paper_book_long_and_short_open_close_same_path():
    book = PaperTradingBook(
        initial_capital=10_000,
        taker_fee=0.0005,
        slippage_bps=0.0,  # isolate PnL sign
        default_notional=100.0,
    )
    opened = book.apply_decision("ENTER_SHORT", "SOLUSDT", 100.0)
    assert opened["success"] is True
    assert opened["side"] == "short"
    assert "SOLUSDT" in book.positions
    assert book.positions["SOLUSDT"].side == "short"

    book.mark_to_market("SOLUSDT", 90.0)
    assert book.positions["SOLUSDT"].unrealized_pnl() == pytest.approx(10.0, rel=1e-6)

    closed = book.apply_decision("EXIT", "SOLUSDT", 90.0)
    assert closed["success"] is True
    assert closed["action"] == "close"
    assert closed["realized_pnl"] > 0
    assert "SOLUSDT" not in book.positions


def test_paper_book_long_still_profits_when_price_rises():
    book = PaperTradingBook(taker_fee=0.0, slippage_bps=0.0, default_notional=100.0)
    book.apply_decision("ENTER_LONG", "SOLUSDT", 100.0)
    book.apply_decision("EXIT", "SOLUSDT", 110.0)
    assert book.realized_pnl == pytest.approx(10.0, rel=1e-6)


def test_paper_book_short_loses_when_price_rises_symmetric_to_long():
    """Same move, opposite side → opposite PnL (zero costs)."""
    long_book = PaperTradingBook(taker_fee=0.0, slippage_bps=0.0, default_notional=100.0)
    short_book = PaperTradingBook(taker_fee=0.0, slippage_bps=0.0, default_notional=100.0)
    long_book.apply_decision("ENTER_LONG", "SOLUSDT", 100.0)
    short_book.apply_decision("ENTER_SHORT", "SOLUSDT", 100.0)
    long_book.apply_decision("EXIT", "SOLUSDT", 110.0)
    short_book.apply_decision("EXIT", "SOLUSDT", 110.0)
    assert long_book.realized_pnl == pytest.approx(10.0, rel=1e-6)
    assert short_book.realized_pnl == pytest.approx(-10.0, rel=1e-6)
    assert long_book.realized_pnl == pytest.approx(-short_book.realized_pnl, rel=1e-6)


def test_paper_book_risk_caps_identical_for_both_sides():
    book = PaperTradingBook(
        initial_capital=10_000,
        max_position_frac=0.03,
        max_leverage=3,
        default_notional=10_000,  # ask for more than the 3% cap
    )
    sized = book.position_notional()
    assert sized == pytest.approx(300.0)  # 3% of 10k
    book.open_position("SOLUSDT", "short", 50.0, notional=10_000)
    pos = book.positions["SOLUSDT"]
    assert pos.quantity * pos.entry_price == pytest.approx(300.0, rel=1e-4)


@pytest.mark.asyncio
async def test_executor_opens_short_when_discovery_says_short():
    provider = _replay_from_real_sol()
    executor = TradingExecutor(AutonomousConfig(), provider=provider)

    decisions = await executor.evaluate_discoveries_for_trading(
        [_discovery(entry_type="SHORT")]
    )
    assert len(decisions) == 1
    assert decisions[0].decision_type == "ENTER_SHORT"
    assert decisions[0].side == "short"

    result = await executor.execute_paper_trade(decisions[0])
    assert result["success"] is True
    assert result["side"] == "short"
    assert executor.book.positions["SOLUSDT"].side == "short"

    # Close on the same path
    exit_decision = decisions[0]
    exit_decision.decision_type = "EXIT"
    closed = await executor.execute_paper_trade(exit_decision)
    assert closed["success"] is True
    assert closed["action"] == "close"
    assert "SOLUSDT" not in executor.book.positions


@pytest.mark.asyncio
async def test_executor_long_path_unchanged_without_entry_type():
    provider = _replay_from_real_sol()
    executor = TradingExecutor(AutonomousConfig(), provider=provider)

    decisions = await executor.evaluate_discoveries_for_trading([_discovery()])
    assert len(decisions) == 1
    assert decisions[0].decision_type == "ENTER_LONG"
    assert decisions[0].side == "long"

    result = await executor.execute_paper_trade(decisions[0])
    assert result["success"] is True
    assert result["side"] == "long"
    assert executor.book.positions["SOLUSDT"].side == "long"


@pytest.mark.asyncio
async def test_executor_exits_before_flipping_long_to_short():
    provider = _replay_from_real_sol()
    executor = TradingExecutor(AutonomousConfig(), provider=provider)
    await executor.execute_paper_trade(
        (await executor.evaluate_discoveries_for_trading([_discovery(entry_type="LONG")]))[0]
    )
    assert executor.book.positions["SOLUSDT"].side == "long"

    flip = await executor.evaluate_discoveries_for_trading(
        [_discovery(entry_type="SHORT")]
    )
    assert flip[0].decision_type == "EXIT"
    closed = await executor.execute_paper_trade(flip[0])
    assert closed["action"] == "close"
    assert "SOLUSDT" not in executor.book.positions
