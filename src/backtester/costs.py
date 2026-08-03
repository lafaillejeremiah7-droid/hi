"""Cost model for backtesting.

Models realistic trading costs including:
- Volatility-based slippage (0.5-2 points on NAS100)
- Configurable commission per round trip
"""

import numpy as np


class CostModel:
    """Models trading costs including slippage and commissions.

    Slippage is modeled as a function of volatility: higher volatility
    means worse fills. Commission is a fixed cost per round trip.

    Attributes:
        base_slippage_points: Base slippage in index points.
        commission_per_round_trip: Commission cost per trade (entry + exit).
        point_value: Dollar value per point of the instrument.
    """

    def __init__(
        self,
        base_slippage_points: float = 1.0,
        commission_per_round_trip: float = 4.50,
        point_value: float = 20.0,
    ):
        """Initialize cost model.

        Args:
            base_slippage_points: Base slippage in points (scales with volatility).
            commission_per_round_trip: Round trip commission in dollars.
            point_value: Dollar value per point move.
        """
        self.base_slippage_points = base_slippage_points
        self.commission_per_round_trip = commission_per_round_trip
        self.point_value = point_value

    def apply_slippage(
        self,
        price: float,
        direction: int,
        volatility: float,
        avg_volatility: float,
    ) -> float:
        """Apply slippage to an execution price.

        Slippage scales with current volatility relative to average.
        Higher volatility means more slippage (worse fills).
        Range: 0.5x to 2x of base_slippage_points.

        Args:
            price: Intended execution price.
            direction: 1 for buy, -1 for sell.
            volatility: Current bar's true range or ATR.
            avg_volatility: Average volatility over recent period.

        Returns:
            Adjusted execution price after slippage.
        """
        if avg_volatility == 0:
            vol_ratio = 1.0
        else:
            vol_ratio = volatility / avg_volatility

        # Scale slippage between 0.5x and 2.0x based on volatility ratio
        slip_multiplier = max(0.5, min(2.0, vol_ratio))
        slippage = self.base_slippage_points * slip_multiplier

        # Slippage always works against the trader
        if direction == 1:  # Buying: slippage increases price
            return price + slippage
        else:  # Selling: slippage decreases price
            return price - slippage

    def apply_commission(self) -> float:
        """Calculate commission cost for a round trip trade.

        Returns:
            Total commission cost in dollar terms.
        """
        return self.commission_per_round_trip

    def total_cost_per_trade(
        self,
        entry_price: float,
        exit_price: float,
        direction: int,
        entry_volatility: float,
        exit_volatility: float,
        avg_volatility: float,
    ) -> float:
        """Calculate total cost for a complete trade (entry + exit).

        Returns the total cost in points that should be subtracted
        from gross P&L.

        Args:
            entry_price: Raw entry price.
            exit_price: Raw exit price.
            direction: 1 for long, -1 for short.
            entry_volatility: Volatility at entry.
            exit_volatility: Volatility at exit.
            avg_volatility: Average volatility for reference.

        Returns:
            Total cost in points (always positive).
        """
        # Entry slippage
        actual_entry = self.apply_slippage(
            entry_price, direction, entry_volatility, avg_volatility
        )
        # Exit slippage (opposite direction)
        actual_exit = self.apply_slippage(
            exit_price, -direction, exit_volatility, avg_volatility
        )

        # Slippage cost in points
        slippage_cost = abs(actual_entry - entry_price) + abs(actual_exit - exit_price)

        # Commission in points
        commission_points = self.commission_per_round_trip / self.point_value

        return slippage_cost + commission_points

    def partial_exit_cost(
        self,
        entry_price: float,
        exit_price: float,
        direction: int,
        entry_volatility: float,
        exit_volatility: float,
        avg_volatility: float,
        fraction: float = 0.5,
    ) -> float:
        """Calculate cost for a partial position exit.

        Used when closing a fraction of the position (e.g., 50% partial close).
        Entry slippage is proportioned to the fraction being closed.
        Commission is split proportionally.

        Args:
            entry_price: Raw entry price.
            exit_price: Raw exit price at partial close.
            direction: 1 for long, -1 for short.
            entry_volatility: Volatility at entry.
            exit_volatility: Volatility at exit.
            avg_volatility: Average volatility for reference.
            fraction: Fraction of position being closed (0-1).

        Returns:
            Cost in points for the partial exit (always positive).
        """
        # Entry slippage (proportioned)
        actual_entry = self.apply_slippage(
            entry_price, direction, entry_volatility, avg_volatility
        )
        # Exit slippage (proportioned)
        actual_exit = self.apply_slippage(
            exit_price, -direction, exit_volatility, avg_volatility
        )

        # Slippage cost scaled by fraction
        slippage_cost = (abs(actual_entry - entry_price) + abs(actual_exit - exit_price)) * fraction

        # Commission proportional to fraction
        commission_points = (self.commission_per_round_trip / self.point_value) * fraction

        return slippage_cost + commission_points

    @classmethod
    def from_config(cls, config: dict) -> "CostModel":
        """Create CostModel from configuration dict.

        Args:
            config: Dictionary with keys: slippage_points,
                    commission_per_round_trip, point_value.

        Returns:
            Configured CostModel instance.
        """
        return cls(
            base_slippage_points=config.get("slippage_points", 1.0),
            commission_per_round_trip=config.get("commission_per_round_trip", 4.50),
            point_value=config.get("point_value", 20.0),
        )
