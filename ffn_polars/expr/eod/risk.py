import polars as pl
import numpy as np
from datetime import date, datetime
from typing import Union, Callable
import math
from ffn_polars.utils.guardrails import guard_expr
from ffn_polars.utils.decorators import auto_alias
from ffn_polars.utils.typing import ExprOrStr
from ffn_polars.registry import register


@register(namespace="eod")
@guard_expr("self", expected_dtype=pl.Float64)
@auto_alias("ulcer_index")
def ulcer_index(self: ExprOrStr) -> pl.Expr:
    """
    Returns a Polars expression to compute the Ulcer Index from a price series.

    Formula:
        1. Compute cumulative max of prices.
        2. Calculate drawdowns: ((price - cummax) / cummax) * 100.
        3. Square drawdowns, take mean, then square root.
    """
    cummax = self.cum_max()
    drawdown_pct = ((self - cummax) / cummax) * 100
    squared_drawdowns = drawdown_pct.pow(2)

    return squared_drawdowns.mean().sqrt()


@register(namespace="eod")
@guard_expr("self", expected_dtype=pl.Float64)
@auto_alias("ulcer_performance_index")
def ulcer_performance_index(
    self: ExprOrStr, rf: Union[float, str] = 0.0, n: int = None
) -> pl.Expr:
    """
    Returns a Polars expression to compute Ulcer Performance Index (UPI).

    UPI = mean(excess returns) / ulcer index
    Must be used inside `.select()` or `.with_columns()`

    Args:
        self: column with prices
        rf: either a float (annualized) or a column name containing RF series
        n: required if rf is float and nonzero
    """
    if isinstance(rf, float):
        if rf != 0 and n is None:
            raise ValueError("nperiods must be set when rf is a non-zero float")

        excess_returns = self.ffn.to_returns() - (rf / n if rf != 0 else 0)

    elif isinstance(rf, str):
        # Subtract column rf from returns
        excess_returns = self.ffn.to_returns() - pl.col(rf)
    else:
        raise TypeError("rf must be a float or a string (column name)")

    return excess_returns.mean() / self.ffn.ulcer_index()


@register(namespace="eod")
@guard_expr("self", expected_dtype=pl.Float64)
@auto_alias("drawdowns")
def to_drawdown_series(self: ExprOrStr) -> pl.Expr:
    """
    Calculates the `drawdown <https://www.investopedia.com/terms/d/drawdown.asp>`_ series.

    This returns a series representing a drawdown.
    When the price is at all time highs, the drawdown
    is 0. However, when prices are below high water marks,
    the drawdown series = current / hwm - 1

    The max drawdown can be obtained by simply calling .min()
    on the result (since the drawdown series is negative)

    Method ignores all gaps of NaN's in the price series.

    Args:
        self: prices

    """
    prices_clean = self.forward_fill()
    hwm = prices_clean.cum_max()
    return prices_clean / hwm - 1


@register(namespace="eod")
@guard_expr("self", expected_dtype=pl.Float64)
@auto_alias("max_drawdown")
def calc_max_drawdown(self: ExprOrStr) -> pl.Expr:
    """
    Calculates the max drawdown of a price series. If you want the
    actual drawdown series, please use to_drawdown_series.
    """
    return self.ffn.to_drawdown_series().min()
