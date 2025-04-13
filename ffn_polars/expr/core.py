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

TRADING_DAYS_PER_YEAR = 252


@guard_expr("self", expected_dtype=pl.Float64)
@auto_alias("returns")
def to_returns(self: ExprOrStr) -> pl.Expr:
    """
    Calculates the simple arithmetic returns of a price series.

    Formula is: (t1 / t0) - 1

    Args:
        * prices: Expects a price series

    """
    return self / self.shift(1) - 1


@guard_expr("self", expected_dtype=pl.Float64)
@auto_alias("log_returns")
def to_log_returns(self: ExprOrStr) -> pl.Expr:
    """
    Calculates the log returns of a price series.

    Formula is: ln(p1/p0)

    Args:
        * prices: Expects a price series

    """
    return (self / self.shift(1)).log()


@guard_expr("self", expected_dtype=pl.Float64, required_substring="returns")
@auto_alias("price_index")
def to_price_index(self, start=100):
    """
    Returns a price index given a series of returns.

    Args:
        * returns: Expects a return series
        * start (number): Starting level

    Assumes arithmetic returns.

    Formula is: cumprod (1+r)
    """
    return (self.fill_null(0.0) + 1).cum_prod() * start


@guard_expr("self", expected_dtype=pl.Float64)
@auto_alias("rebased")
def rebase(self, value=100):
    """
    Rebase a price series to a given value.
    Args:
        * prices: Expects a price series
        * value (number): Value to rebase to
    Formula is: (p / p0) * value
    """
    return self / self.first() * value


@guard_expr("self", expected_dtype=pl.Float64)
@auto_alias("drawdowns")
def to_drawdown_series(self):
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
        * prices (Series or DataFrame): Series of prices.

    """
    prices_clean = self.forward_fill()
    hwm = prices_clean.cum_max()
    return prices_clean / hwm - 1


@guard_expr("self", expected_dtype=pl.Float64)
@guard_expr("date_col", expected_dtype=pl.Datetime)
@auto_alias("mtd")
def calc_mtd(self: pl.Expr, date_col: ExprOrStr = "Date") -> pl.Expr:
    """
    Calculate Month-To-Date return using daily prices only.

    Logic:
    - Latest price = last row
    - Reference price = last price from previous month
    - MTD = (latest / reference) - 1
    """
    prices = self
    latest_date = date_col.max()

    # Extract month & year
    latest_month = latest_date.dt.month()
    latest_year = latest_date.dt.year()

    return pl.when(True).then(
        prices.filter(
            (date_col.dt.month() != latest_month) | (date_col.dt.year() != latest_year)
        )
        .last()
        .pipe(lambda ref: prices.last() / ref - 1)
    )


@auto_alias("ytd")
@guard_expr("self", expected_dtype=pl.Float64)
@guard_expr("date_col", expected_dtype=pl.Datetime)
def calc_ytd(self: pl.Expr, date_col: ExprOrStr = "Date") -> pl.Expr:
    """
    Calculate Year-To-Date (YTD) return using daily prices.

    Logic:
    - Identify current year from latest date
    - First price = first row of current year
    - Latest price = most recent row
    - YTD = (latest / first_of_year) - 1

    Assumes `date_col` is sorted ascending.
    """

    latest_date_expr = date_col.max()
    current_year = latest_date_expr.dt.year()

    # Filter to current year only
    current_year_prices = self.filter(date_col.dt.year() == current_year)

    return (current_year_prices.last() / current_year_prices.first()) - 1


@guard_expr("self", expected_dtype=pl.Float64)
@auto_alias("max_drawdown")
def calc_max_drawdown(self):
    """
    Calculates the max drawdown of a price series. If you want the
    actual drawdown series, please use to_drawdown_series.
    """
    return self.ffn.to_drawdown_series().min()


@guard_expr("self", expected_dtype=pl.Datetime)
@auto_alias("year_frac")
def year_frac(self) -> pl.Expr:
    """
    Returns a Polars expression that computes the year fraction between the first and last date
    in a column, assuming average year length (365.25 days = 31557600 seconds).
    """
    return (
        self.cast(pl.Datetime).last() - self.cast(pl.Datetime).first()
    ).dt.total_seconds() / 31_557_600


@guard_expr("self", expected_dtype=pl.Datetime)
@guard_expr("date_col", expected_dtype=pl.Datetime)
@auto_alias("cagr")
def calc_cagr(self: ExprOrStr, date_col: ExprOrStr) -> pl.Expr:
    """
    Calculates the `CAGR (compound annual growth rate) <https://www.investopedia.com/terms/c/cagr.asp>`_ for a given price series.

    Args:
        * prices (pandas.Series): A Series of prices.
    Returns:
        * float -- cagr.

    """
    return ((self.last() / self.first()) ** (1 / year_frac(date_col)) - 1).alias("cagr")


@guard_expr("date_col", expected_dtype=pl.Datetime)
@auto_alias("inferred_freq")
def infer_freq(self: ExprOrStr) -> pl.Expr:
    """
    Infers human-readable calendar frequency label from a datetime column.
    Works best for: yearly, quarterly, monthly, weekly, daily.

    Returns: "yearly", "quarterly", "monthly", "weekly", "daily", or None
    """
    deltas = (
        (
            self.cast(pl.Datetime).sort().diff().dt.total_nanoseconds().cast(pl.Float64)
            / 86400
            / 1_000_000_000
        )  # convert to float days
        .drop_nulls()
        .alias("delta_days")
    )

    std_expr = deltas.std().alias("delta_std")
    mode_expr = deltas.mode().first().alias("mode_days")

    return (
        pl.struct(
            [
                std_expr,
                mode_expr,
            ]
        )
        .map_batches(_map_mode_days_with_tolerance, return_dtype=pl.Utf8)
        .alias("freq")
    )


def _map_mode_days_with_tolerance(batch: pl.Series) -> pl.Series:
    row = batch[0]  # pl.Struct
    std = row["delta_std"]
    mode = row["mode_days"]

    if std is None or std > 1.0:
        return pl.Series(["unknown"])

    d = mode
    if abs(d - 365.25) < 5:
        return pl.Series(["yearly"])
    elif abs(d - 91) <= 3:
        return pl.Series(["quarterly"])
    elif abs(d - 30) <= 3:
        return pl.Series(["monthly"])
    elif abs(d - 7) <= 1:
        return pl.Series(["weekly"])
    elif abs(d - 1) <= 0.1:
        return pl.Series(["daily"])
    else:
        return pl.Series(["unknown"])


@guard_expr("self", expected_dtype=pl.Float64)
@auto_alias("deannualized")
def deannualize(self: ExprOrStr, n: int) -> pl.Expr:
    """
    Returns a Polars expression that converts annualized returns to periodic returns.

    Args:
        col (str): Name of the column containing annualized returns
        nperiods (int): Number of periods per year (e.g., 252 for daily)
    """
    return (self + 1.0) ** (1.0 / n) - 1.0


@guard_expr("self", expected_dtype=pl.Float64)
@auto_alias("excess")
def to_excess_returns(self: ExprOrStr, rf: Union[float, str], n: int) -> pl.Expr:
    """
    Returns a Polars expression that computes excess returns.

    Args:
        returns_col (str): Column with actual returns
        rf (float | str): Either an annualized float or column name with risk-free returns
        nperiods (int): Number of periods per year (used for deannualization if rf is a float)
    """
    if isinstance(rf, float):
        if rf == 0:
            return self
        else:
            return self - ((1 + rf) ** (1 / n) - 1)
    elif isinstance(rf, str):
        return self - pl.col(rf)
    else:
        raise TypeError("rf must be either a float or a column name string")


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
    excess_expr = to_excess_returns(self, rf, n)

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


@guard_expr("self", expected_dtype=pl.Float64)
@auto_alias("total_return")
def calc_total_return(self: ExprOrStr) -> pl.Expr:
    """
    Calculates the total return of a series.

    last / first - 1
    """
    return (self.last() / self.first()) - 1


@guard_expr("self", expected_dtype=pl.Float64)
@guard_expr("durations", expected_dtype=pl.Float64)
@auto_alias("annualized")
def annualize(
    self: ExprOrStr, durations: ExprOrStr, one_year: float = 365.0
) -> pl.Expr:
    """
    Returns a Polars expression to annualize returns given durations.

    Args:
        returns_col (str): Name of the column with returns (e.g., 0.05 = 5%).
        durations_col (str): Name of the column with durations (e.g., days held).
        one_year (float): Number of periods in a year (default 365.0 for days).

    Returns:
        pl.Expr: Expression computing annualized return.
    """
    return (1.0 + self) ** (one_year / durations) - 1.0


@guard_expr("self", expected_dtype=pl.Datetime)
@auto_alias("nperiods")
def infer_nperiods(self: ExprOrStr, annualization_factor: int | None = None) -> pl.Expr:
    af = annualization_factor or TRADING_DAYS_PER_YEAR

    return (
        self.cast(pl.Datetime)
        .sort()
        .diff()
        .dt.total_nanoseconds()
        .drop_nulls()
        .map_batches(lambda s: _infer_from_deltas(s, af), return_dtype=pl.Int64)
    )


def _infer_from_deltas(series: pl.Series, af: int) -> pl.Series:
    seconds = series / 1_000_000_000
    valid = seconds.filter((seconds > 1e-9) & seconds.is_finite())

    if valid.len() < 1:
        return pl.Series([None])

    std = valid.std()
    if std > 1e-3:
        return pl.Series([None])

    dt = valid.mode().to_numpy()[0]

    # Calendar-style tolerances
    if abs(dt - 365 * 86400) <= 86400:  # yearly range: 360-366d
        return pl.Series([1])
    elif abs(dt - 91 * 86400) <= 3 * 86400:
        return pl.Series([4])
    elif abs(dt - 30 * 86400) <= 3 * 86400:
        return pl.Series([12])
    elif abs(dt - 7 * 86400) <= 60:
        return pl.Series([52])
    elif abs(dt - 86400) <= 10:
        return pl.Series([af])
    elif abs(dt - 3600) <= 5:
        return pl.Series([af * 24])
    elif abs(dt - 60) <= 1:
        return pl.Series([af * 24 * 60])
    elif abs(dt - 1) <= 0.1:
        return pl.Series([af * 24 * 60 * 60])
    elif dt > 0:
        return pl.Series([round(af * 24 * 60 * 60 / dt)])  # ✅ rounded tick-level
    else:
        return pl.Series([None])


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
        price_col: column with prices
        rf: either a float (annualized) or a column name containing RF series
        nperiods: required if rf is float and nonzero
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


core_funcs = {
    "to_returns": to_returns,
    "to_log_returns": to_log_returns,
    "to_price_index": to_price_index,
    "rebase": rebase,
    "to_drawdown_series": to_drawdown_series,
    "calc_mtd": calc_mtd,
    "calc_ytd": calc_ytd,
    "calc_max_drawdown": calc_max_drawdown,
    "year_frac": year_frac,
    "calc_cagr": calc_cagr,
    "infer_freq": infer_freq,
    "annualize": annualize,
    "deannualize": deannualize,
    "to_excess_returns": to_excess_returns,
    "calc_sharpe": calc_sharpe,
    "calc_risk_return_ratio": calc_risk_return_ratio,
    "calc_information_ratio": calc_information_ratio,
    "calc_prob_mom": calc_prob_mom,
    "calc_total_return": calc_total_return,
    "infer_nperiods": infer_nperiods,
    "sortino_ratio": sortino_ratio,
    "calc_calmar_ratio": calc_calmar_ratio,
    "ulcer_index": ulcer_index,
    "ulcer_performance_index": ulcer_performance_index,
    "calc_ulcer_index": ulcer_index,
    "calc_ulcer_performance_index": ulcer_performance_index,
    "calc_prob_mom_cdf": _prob_mom_cdf,
}
