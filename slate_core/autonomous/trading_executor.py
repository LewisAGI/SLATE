"""
SLATE Autonomous Trading Executor

Makes real trading decisions in paper trading mode.
Integrates with discoveries to execute autonomous trading decisions.

Longs and shorts share one path: infer side from existing discovery fields
(``entry_type`` / signed ``signal``), size with the same risk caps, open/close
on ``PaperTradingBook``, mark PnL with the perpetual sign convention.
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
from dataclasses import dataclass

from .config import Discovery, AutonomousConfig
from slate_core.connectors.base import (
    decision_type_for_side,
    infer_entry_side,
)
from slate_core.connectors.factory import get_market_data_provider
from slate_core.connectors.paper_book import PaperTradingBook

logger = logging.getLogger(__name__)


@dataclass
class TradingDecision:
    """Autonomous trading decision."""
    decision_type: str  # "ENTER_LONG", "ENTER_SHORT", "EXIT", "HOLD"
    symbol: str
    confidence: float
    reason: str
    discovery: Discovery
    paper_execution: bool  # Always True - safety constraint
    timestamp: datetime
    side: str = "long"  # 'long' or 'short' — explicit, not inferred later

    def to_dict(self):
        return {
            'decision_type': self.decision_type,
            'side': self.side,
            'symbol': self.symbol,
            'confidence': self.confidence,
            'reason': self.reason,
            'discovery_summary': self.discovery.answer[:100] if self.discovery else None,
            'paper_execution': self.paper_execution,
            'timestamp': self.timestamp.isoformat()
        }


class TradingExecutor:
    """
    Execute autonomous trading decisions in paper trading mode.

    SAFETY:
    - ONLY paper trading (never real money)
    - All decisions logged and reviewable
    - Transaction costs always applied
    - Position sizing risk-managed
    - Shorts use the same book, costs, and risk caps as longs
    """

    def __init__(
        self,
        config: AutonomousConfig,
        provider=None,
        book: Optional[PaperTradingBook] = None,
    ):
        self.config = config

        self.paper_exchange = None
        self.provider = provider
        if self.provider is None:
            try:
                self.provider = get_market_data_provider(getattr(config, "data_provider", None))
                self.paper_exchange = self.provider
                logger.info(
                    "Paper trading data provider initialized: %s",
                    getattr(self.provider, "name", type(self.provider).__name__),
                )
            except Exception as exc:  # noqa: BLE001 — keep executor up without a venue
                logger.warning("Market data provider not available - paper trading limited: %s", exc)
        else:
            self.paper_exchange = self.provider

        self.book = book or PaperTradingBook(
            taker_fee=config.taker_fee,
            slippage_bps=config.base_slippage_bps,
            max_position_frac=0.03,
            max_leverage=3,
            max_positions=config.max_positions,
            default_notional=100.0,
        )

        self.decision_history = []
        # Back-compat alias: callers still read executor.paper_positions
        self.paper_positions = self.book.positions

        logger.info("Trading Executor initialized in PAPER_TRADING mode")

    def _decision_for_discovery(self, discovery: Discovery, decision_score: float) -> TradingDecision:
        side = infer_entry_side(discovery)
        existing = self.book.positions.get(discovery.symbol)
        if existing is not None and existing.side != side:
            decision_type = "EXIT"
            reason = (
                f"Opposite side ({side}) vs open {existing.side}; "
                f"closing before flip. {discovery.answer[:80]}"
            )
        else:
            decision_type = decision_type_for_side(side)
            reason = f"Profitable strategy discovered: {discovery.answer[:100]}"
        return TradingDecision(
            decision_type=decision_type,
            side=side,
            symbol=discovery.symbol,
            confidence=decision_score,
            reason=reason,
            discovery=discovery,
            paper_execution=True,
            timestamp=datetime.now(),
        )

    async def evaluate_discoveries_for_trading(self, discoveries: List[Discovery]) -> List[TradingDecision]:
        """
        Evaluate discoveries and make trading decisions.

        This is where autonomous trading decisions are made.
        Multiple discoveries are analyzed and prioritized.
        Side comes from the discovery's existing entry_type / signal — not a
        new strategy, and not hardcoded LONG.
        """
        decisions = []

        for discovery in discoveries:
            # Skip if not realistic edge
            if not discovery.realistic_edge:
                logger.debug(f"Skipping {discovery.symbol}: not realistic edge")
                continue

            # Skip if confidence too low
            if discovery.confidence < self.config.min_confidence_to_store:
                logger.debug(f"Skipping {discovery.symbol}: confidence {discovery.confidence:.2f} too low")
                continue

            # Skip if drawdown too high
            if discovery.max_drawdown_pct > self.config.max_drawdown_pct:
                logger.debug(f"Skipping {discovery.symbol}: drawdown {discovery.max_drawdown_pct:.1f}% too high")
                continue

            # Calculate decision score
            decision_score = self._calculate_decision_score(discovery)

            # Make trading decision
            if decision_score > 0.7:  # High confidence threshold
                decision = self._decision_for_discovery(discovery, decision_score)
                decisions.append(decision)

                logger.info(f"🎯 Trading decision: {decision.decision_type} {decision.symbol}")
                logger.info(f"   Side: {decision.side}")
                logger.info(f"   Confidence: {decision.confidence:.1%}")
                logger.info(f"   Reason: {decision.reason}")

        return decisions

    def _calculate_decision_score(self, discovery: Discovery) -> float:
        """
        Calculate trading decision score from discovery metrics.

        Combines multiple factors into a single confidence score.
        """
        score = 0.0

        # Profitability after costs (40% weight)
        if discovery.profit_after_costs > 0:
            profitability_score = min(discovery.profitability_score, 1.0)
            score += profitability_score * 0.40

        # Risk-adjusted returns (25% weight)
        if discovery.sharpe_ratio > 0.5:
            sharpe_score = min(discovery.sharpe_ratio / 2.0, 1.0)
            score += sharpe_score * 0.25

        # Win rate (20% weight)
        if discovery.win_rate > 0.5:
            win_score = (discovery.win_rate - 0.5) * 2  # Scale 0.5-1.0 to 0.0-1.0
            score += win_score * 0.20

        # Discovery confidence (15% weight)
        score += discovery.confidence * 0.15

        return min(score, 1.0)  # Cap at 1.0

    async def _current_price(self, symbol: str) -> float:
        if not self.provider:
            return 0.0
        ticker = await self.provider.get_ticker(symbol)
        if not ticker:
            return 0.0
        return float(ticker.get("last_price") or 0.0)

    async def execute_paper_trade(self, decision: TradingDecision) -> Dict[str, Any]:
        """
        Execute a trading decision in paper trading mode.

        Longs and shorts go through ``PaperTradingBook.apply_decision``.
        """
        logger.info(f"📊 Executing PAPER trade: {decision.decision_type} {decision.symbol}")

        if decision.decision_type == "HOLD":
            return {
                "success": True,
                "paper_trade": True,
                "action": "hold",
                "symbol": decision.symbol,
                "side": decision.side,
            }

        if not self.provider:
            logger.error("Paper data provider not available - cannot execute trade")
            return {"success": False, "error": "no_exchange"}

        try:
            current_price = await self._current_price(decision.symbol)
            if current_price <= 0.0:
                logger.error(f"Cannot get price for {decision.symbol}")
                return {"success": False, "error": "no_price"}

            execution_result = self.book.apply_decision(
                decision.decision_type,
                decision.symbol,
                current_price,
            )
            execution_result.setdefault("symbol", decision.symbol)
            execution_result.setdefault("side", decision.side)
            execution_result["decision_type"] = decision.decision_type
            execution_result["decision_confidence"] = decision.confidence
            if decision.discovery:
                execution_result["discovery_sharpe"] = decision.discovery.sharpe_ratio
                execution_result["discovery_profit"] = decision.discovery.profit_after_costs
            execution_result["timestamp"] = datetime.now().isoformat()
            execution_result["provider"] = getattr(self.provider, "name", "unknown")

            if execution_result.get("success"):
                logger.info(
                    "✅ Paper trade executed: %s %s $%.2f",
                    execution_result.get("side"),
                    execution_result.get("symbol"),
                    execution_result.get("position_value_usdt")
                    or execution_result.get("notional")
                    or 0.0,
                )
                if "transaction_costs_usdt" in execution_result:
                    logger.info(
                        "   Transaction costs: $%.4f USDT",
                        execution_result["transaction_costs_usdt"],
                    )
                if execution_result.get("action") == "close":
                    logger.info(
                        "   Realized PnL: $%.4f USDT",
                        execution_result.get("realized_pnl", 0.0),
                    )

            self.paper_positions = self.book.positions
            self.decision_history.append(decision)
            return execution_result

        except Exception as e:
            logger.error(f"Error executing paper trade: {e}", exc_info=True)
            return {"success": False, "error": str(e)}

    def get_paper_positions(self) -> Dict[str, Any]:
        """Get current paper trading positions."""
        snap = self.book.snapshot()
        # Preserve the previous keys plus side/PnL now that shorts exist.
        snap["positions"] = {
            symbol: {
                "entry_price": pos.entry_price,
                "quantity": pos.quantity,
                "side": pos.side,
                "unrealized_pnl": pos.unrealized_pnl(),
                "entry_time": pos.entry_time.isoformat(),
            }
            for symbol, pos in self.book.positions.items()
        }
        return snap

    def get_decision_history(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get recent trading decisions."""
        recent_decisions = self.decision_history[-limit:]
        return [decision.to_dict() for decision in recent_decisions]

    def get_statistics(self) -> Dict[str, Any]:
        """Get trading executor statistics."""
        return {
            "total_decisions": len(self.decision_history),
            "active_positions": len(self.book.positions),
            "mode": "PAPER_TRADING_ONLY",
            "connector_available": self.provider is not None,
            "provider": getattr(self.provider, "name", None),
            "recent_decisions": self.get_decision_history(limit=5),
            "paper_positions": self.get_paper_positions(),
        }
