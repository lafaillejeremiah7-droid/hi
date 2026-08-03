"""Base strategy abstract class.

Defines the interface that all trading strategies must implement.
"""

from abc import ABC, abstractmethod
from typing import Any

import pandas as pd


class BaseStrategy(ABC):
    """Abstract base class for trading strategies.

    All strategies must implement signal generation and stop/target
    placement methods. Parameters should be stored in self.params
    for optimization support.
    """

    def __init__(self, params: dict[str, Any] | None = None):
        """Initialize strategy with parameters.

        Args:
            params: Dictionary of strategy parameters. If None,
                    uses default parameters from the strategy.
                    If provided, merges with defaults (custom values override).
        """
        defaults = self.default_params()
        if params:
            defaults.update(params)
        self.params = defaults

    @abstractmethod
    def default_params(self) -> dict[str, Any]:
        """Return default parameters for this strategy.

        Returns:
            Dictionary of parameter name -> default value.
        """
        ...

    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals from preprocessed data.

        Args:
            df: Preprocessed DataFrame with OHLCV and indicator columns.

        Returns:
            DataFrame with at minimum a 'signal' column containing:
            1 = long entry, -1 = short entry, 0 = flat/no signal.
            May include additional columns like 'signal_strength'.
        """
        ...

    @abstractmethod
    def get_stop_loss(self, df: pd.DataFrame, idx: int, direction: int) -> float:
        """Calculate stop loss price for a trade entry.

        Args:
            df: Full DataFrame with indicator data.
            idx: Index position where the trade is entered.
            direction: 1 for long, -1 for short.

        Returns:
            Stop loss price level.
        """
        ...

    @abstractmethod
    def get_take_profit(self, df: pd.DataFrame, idx: int, direction: int, feature_zscore: float | None = None) -> float:
        """Calculate take profit price for a trade entry.

        Args:
            df: Full DataFrame with indicator data.
            idx: Index position where the trade is entered.
            direction: 1 for long, -1 for short.
            feature_zscore: Optional Z-score of active feature at entry.
                If |feature_zscore| >= threshold, use extended TP multiplier.

        Returns:
            Take profit price level.
        """
        ...

    def get_param_ranges(self) -> dict[str, list]:
        """Return parameter ranges for grid search optimization.

        Override in subclass to define parameter search space.

        Returns:
            Dictionary of parameter name -> list of values to search.
        """
        return {}

    @property
    def name(self) -> str:
        """Return strategy name."""
        return self.__class__.__name__
