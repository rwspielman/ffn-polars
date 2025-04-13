import polars as pl
import numpy as np
from datetime import date, datetime
from typing import Union, Callable
import math
from ffn_polars.utils.guardrails import guard_expr
from ffn_polars.utils.decorators import auto_alias
from ffn_polars.utils.typing import ExprOrStr

try:
    from ffn_polars import _rust

    _HAS_RUST = hasattr(_rust, "prob_mom")
except ImportError:
    _rust = None
    _HAS_RUST = False


@guard_expr("self", expected_dtype=pl.Float64)
@auto_alias("sortino_ratio")
def sortino_ratio(
    self: ExprOrStr, rf: float = 0.0, n: int = 252, annualize: bool = True
) -> pl.Expr:
    rf_periodic = rf / n
    excess = self - rf_periodic

    downside = excess.map_elements(
        lambda x: min(x, 0.0), return_dtype=pl.Float64
    )  # TODO: vectorize
    downside_std = downside.std(ddof=1)

    sortino = (
        pl.when(downside_std.is_not_null() & (downside_std != 0.0))
        .then(excess.mean() / downside_std)
        .otherwise(None)
    )

    if annualize:
        sortino *= n**0.5

    return sortino


@guard_expr("self", expected_dtype=pl.Float64)
@auto_alias("calmar_ratio")
def calc_calmar_ratio(self: ExprOrStr, date_col: str) -> pl.Expr:
    """
    Returns a Polars expression to compute the Calmar ratio: CAGR / |Max Drawdown|

    Args:
        prices_col (str): Column name of price series
        date_col (str): Column name of date series

    Returns:
        pl.Expr: Calmar ratio expression
    """
    cagr_expr = self.ffn.calc_cagr(date_col=date_col)
    max_dd_expr = self.ffn.calc_max_drawdown().abs()

    return cagr_expr / max_dd_expr


@guard_expr("self", expected_dtype=pl.Float64)
@auto_alias("sharpe")
def calc_sharpe(
    self: str,
    rf: Union[float, str] = 0.0,
    n: int = 252,
    annualize: bool = True,
) -> pl.Expr:
    """
    Polars expression that computes the Sharpe ratio in-place without aliasing.
    """
    excess_expr = self.ffn.to_excess_returns(rf, n)

    sharpe_expr = (
        (excess_expr.mean() / excess_expr.std(ddof=1)) * math.sqrt(n)
        if annualize
        else (excess_expr.mean() / excess_expr.std(ddof=1))
    )

    return sharpe_expr


@guard_expr("self", expected_dtype=pl.Float64)
@auto_alias("risk_return_ratio")
def calc_risk_return_ratio(self) -> pl.Expr:
    """
    Calculates the return / risk ratio. Basically the
    `Sharpe ratio <https://www.investopedia.com/terms/s/sharperatio.asp>`_ without factoring in the `risk-free rate <https://www.investopedia.com/terms/r/risk-freerate.asp>`_.
    """
    return calc_sharpe(self)


@guard_expr("self", expected_dtype=pl.Float64)
@guard_expr("benchmark", expected_dtype=pl.Float64)
@auto_alias("ir")
def calc_information_ratio(self: ExprOrStr, benchmark: ExprOrStr) -> pl.Expr:
    """
    Returns a Polars expression that computes the Information Ratio.

    Args:
        returns_col: name of the column with asset returns
        benchmark_col: name of the column with benchmark returns
    """
    diff = self - benchmark

    return (diff.mean() / diff.std(ddof=1)).fill_nan(0.0).fill_null(0.0)


@guard_expr("self", expected_dtype=pl.Float64)
@guard_expr("b", expected_dtype=pl.Float64)
@auto_alias("prob_mom")
def calc_prob_mom(self: ExprOrStr, b: ExprOrStr) -> pl.Expr:
    """
    Polars expression that computes probabilistic momentum between two return columns.
    If Rust plugin is available, uses it. Otherwise, falls back to a Polars map_batches version.
    """

    name1 = self.meta.output_name()
    name2 = b.meta.output_name()
    if _HAS_RUST:
        return pl.struct([name1, name2]).map_batches(
            lambda s: _rust.prob_mom(s.struct.field(name1), s.struct.field(name2)),
            return_dtype=pl.Float64,
        )

    # fallback: pure Polars map
    diff = self - b
    ir = (diff.mean() / diff.std()).alias("information_ratio")
    n = pl.count().alias("n_obs")

    return (
        pl.struct([ir, n])
        .map_batches(
            lambda s: pl.Series([_prob_mom_cdf(s[0])]),
            return_dtype=pl.Float64,
        )
        .alias("prob_momentum")
    )


def _prob_mom_cdf(stats: dict) -> float | None:
    from scipy.stats import t

    ir = stats.get("information_ratio")
    n = stats.get("n_obs")
    if ir is None or n is None or n <= 1:
        return None
    return t.cdf(ir, df=n - 1)
