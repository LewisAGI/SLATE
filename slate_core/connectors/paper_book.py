"""Venue-agnostic paper book: longs and shorts share one path.

Perpetual backtests already use ``signal ∈ {+1, −1}`` with mirrored PnL:

    long  PnL = (exit − entry) * qty
    short PnL = (entry − exit) * qty

The autonomous paper executor previously stored a side-less bag and always
emitted ENTER_LONG. This book is that missing execution layer — same sizing,
fees, slippage, and risk caps on both sides. Paper only; no live orders.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Literal, Optional

from .base import DecisionType, PositionSide

Side = PositionSide


@dataclass
class PaperFill:
    symbol: str
    side: Side
    action: Literal["open", "close", "hold", "already_open"]
    quantity: float
    price: float
    notional: float
    fee: float
    slippage: float
    realized_pnl: float = 0.0
    paper_trade: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "side": self.side,
            "action": self.action,
            "quantity": self.quantity,
            "price": self.price,
            "notional": self.notional,
            "fee": self.fee,
            "slippage": self.slippage,
            "realized_pnl": self.realized_pnl,
            "paper_trade": True,
        }


@dataclass
class PaperPosition:
    """One-way paper position. Quantity is always positive; side is explicit."""

    symbol: str
    side: Side
    quantity: float
    entry_price: float
    mark_price: float
    entry_time: datetime = field(default_factory=datetime.now)
    fees_paid: float = 0.0
    realized_pnl: float = 0.0

    def unrealized_pnl(self) -> float:
        if self.quantity <= 0:
            return 0.0
        if self.side == "long":
            return (self.mark_price - self.entry_price) * self.quantity
        return (self.entry_price - self.mark_price) * self.quantity

    def signed_quantity(self) -> float:
        return self.quantity if self.side == "long" else -self.quantity

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "side": self.side,
            "quantity": self.quantity,
            "signed_quantity": self.signed_quantity(),
            "entry_price": self.entry_price,
            "mark_price": self.mark_price,
            "unrealized_pnl": self.unrealized_pnl(),
            "realized_pnl": self.realized_pnl,
            "fees_paid": self.fees_paid,
            "entry_time": self.entry_time.isoformat(),
        }


class PaperTradingBook:
    """Single open/close/mark path used by both longs and shorts."""

    def __init__(
        self,
        initial_capital: float = 10_000.0,
        taker_fee: float = 0.0005,
        slippage_bps: float = 10.0,
        max_position_frac: float = 0.03,
        max_leverage: int = 3,
        max_positions: int = 5,
        default_notional: float = 100.0,
    ):
        self.initial_capital = float(initial_capital)
        self.cash = float(initial_capital)
        self.taker_fee = float(taker_fee)
        self.slippage_bps = float(slippage_bps)
        self.max_position_frac = float(max_position_frac)
        self.max_leverage = int(max_leverage)
        self.max_positions = int(max_positions)
        self.default_notional = float(default_notional)
        self.positions: Dict[str, PaperPosition] = {}
        self.fills: list[PaperFill] = []
        self.realized_pnl: float = 0.0

    # ------------------------------------------------------------------
    # Sizing / risk (identical for long and short)
    # ------------------------------------------------------------------
    @property
    def equity(self) -> float:
        return self.cash + sum(p.unrealized_pnl() for p in self.positions.values())

    def position_notional(self, requested: Optional[float] = None) -> float:
        """Risk-capped notional. Same cap on both sides."""
        cap = self.equity * self.max_position_frac * max(self.max_leverage, 1)
        # Perpetual backtest also clamps to max_position_frac of capital
        # without leverage when that is tighter — keep the conservative min.
        conservative = self.equity * self.max_position_frac
        allowed = min(cap, conservative) if self.max_position_frac < 1 else cap
        want = self.default_notional if requested is None else float(requested)
        return max(0.0, min(want, allowed, self.cash))

    def _costs(self, notional: float) -> tuple[float, float]:
        fee = abs(notional) * self.taker_fee
        slip = abs(notional) * (self.slippage_bps / 10_000.0)
        return fee, slip

    def _fill_price(self, raw_price: float, side: Side, opening: bool) -> float:
        """Adverse slippage: buy above / sell below mid, same as perps backtest."""
        slip = raw_price * (self.slippage_bps / 10_000.0)
        opening_long = opening and side == "long"
        closing_short = (not opening) and side == "short"
        if opening_long or closing_short:
            return raw_price + slip  # buying the asset
        return raw_price - slip  # selling the asset

    # ------------------------------------------------------------------
    # Same-path open / close
    # ------------------------------------------------------------------
    def open_position(
        self,
        symbol: str,
        side: Side,
        price: float,
        notional: Optional[float] = None,
    ) -> Dict[str, Any]:
        if price <= 0:
            return {"success": False, "error": "invalid_price"}
        existing = self.positions.get(symbol)
        if existing is not None:
            if existing.side == side:
                return {
                    "success": True,
                    "action": "already_open",
                    "paper_trade": True,
                    **existing.to_dict(),
                }
            return {"success": False, "error": "opposite_side_open", "side": existing.side}
        if len(self.positions) >= self.max_positions:
            return {"success": False, "error": "max_positions"}

        sized = self.position_notional(notional)
        if sized <= 0:
            return {"success": False, "error": "no_buying_power"}

        fill_px = self._fill_price(price, side, opening=True)
        qty = sized / fill_px
        fee, slip = self._costs(sized)
        total_cost = fee + slip
        if total_cost > self.cash:
            return {"success": False, "error": "insufficient_cash_for_costs"}

        self.cash -= total_cost
        pos = PaperPosition(
            symbol=symbol,
            side=side,
            quantity=qty,
            entry_price=fill_px,
            mark_price=fill_px,
            fees_paid=total_cost,
        )
        self.positions[symbol] = pos
        fill = PaperFill(
            symbol=symbol,
            side=side,
            action="open",
            quantity=qty,
            price=fill_px,
            notional=sized,
            fee=fee,
            slippage=slip,
        )
        self.fills.append(fill)
        return {
            "success": True,
            "paper_trade": True,
            "action": "open",
            **fill.to_dict(),
            "entry_price": fill_px,
            "position_value_usdt": sized,
            "transaction_costs_usdt": total_cost,
        }

    def close_position(self, symbol: str, price: float) -> Dict[str, Any]:
        pos = self.positions.get(symbol)
        if pos is None:
            return {"success": False, "error": "no_position"}
        if price <= 0:
            return {"success": False, "error": "invalid_price"}

        fill_px = self._fill_price(price, pos.side, opening=False)
        notional = fill_px * pos.quantity
        fee, slip = self._costs(notional)
        if pos.side == "long":
            pnl = (fill_px - pos.entry_price) * pos.quantity
        else:
            pnl = (pos.entry_price - fill_px) * pos.quantity
        pnl -= fee + slip
        self.cash += pnl
        self.realized_pnl += pnl
        fill = PaperFill(
            symbol=symbol,
            side=pos.side,
            action="close",
            quantity=pos.quantity,
            price=fill_px,
            notional=notional,
            fee=fee,
            slippage=slip,
            realized_pnl=pnl,
        )
        self.fills.append(fill)
        closed = pos.to_dict()
        del self.positions[symbol]
        return {
            "success": True,
            "paper_trade": True,
            "action": "close",
            **fill.to_dict(),
            "exit_price": fill_px,
            "realized_pnl": pnl,
            "closed_position": closed,
        }

    def apply_decision(
        self,
        decision_type: DecisionType,
        symbol: str,
        price: float,
        notional: Optional[float] = None,
    ) -> Dict[str, Any]:
        """One entry point for ENTER_LONG / ENTER_SHORT / EXIT / HOLD."""
        if decision_type == "HOLD":
            return {"success": True, "paper_trade": True, "action": "hold", "symbol": symbol}
        if decision_type == "EXIT":
            return self.close_position(symbol, price)
        if decision_type not in ("ENTER_LONG", "ENTER_SHORT"):
            return {"success": False, "error": "unknown_decision", "decision_type": decision_type}

        side: Side = "short" if decision_type == "ENTER_SHORT" else "long"
        existing = self.positions.get(symbol)
        if existing is not None and existing.side != side:
            closed = self.close_position(symbol, price)
            opened = self.open_position(symbol, side, price, notional)
            opened["flipped"] = True
            opened["prior_close"] = closed
            return opened
        return self.open_position(symbol, side, price, notional)

    def mark_to_market(self, symbol: str, price: float) -> Optional[PaperPosition]:
        pos = self.positions.get(symbol)
        if pos is None:
            return None
        pos.mark_price = float(price)
        return pos

    def snapshot(self) -> Dict[str, Any]:
        return {
            "mode": "PAPER_TRADING_ONLY",
            "cash": self.cash,
            "equity": self.equity,
            "realized_pnl": self.realized_pnl,
            "active_positions": len(self.positions),
            "positions": {sym: pos.to_dict() for sym, pos in self.positions.items()},
        }
