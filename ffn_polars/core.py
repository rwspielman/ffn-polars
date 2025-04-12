import polars as pl
import numpy as np
from datetime import date, datetime
from typing import Union, Callable
import math

try:
    from . import _rust

    _HAS_RUST = hasattr(_rust, "prob_mom")
except ImportError:
    _rust = None
    _HAS_RUST = False

TRADING_DAYS_PER_YEAR = 252


def to_return(prices):
    """
    Calculates the simple arithmetic returns of a price series.

    Formula is: (t1 / t0) - 1

    Args:
        * prices: Expects a price series

    """
    return pl.col(prices) / pl.col(prices).shift(1) - 1


def to_log_returns(prices):
    """
    Calculates the log returns of a price series.

    Formula is: ln(p1/p0)

    Args:
        * prices: Expects a price series

    """
    return (pl.col(prices) / pl.col(prices).shift(1)).log()


def to_price_index(returns, start=100):
    """
    Returns a price index given a series of returns.

    Args:
        * returns: Expects a return series
        * start (number): Starting level

    Assumes arithmetic returns.

    Formula is: cumprod (1+r)
    """
    return (pl.col(returns).fill_null(0.0) + 1).cum_prod() * start


def rebase(prices, value=100):
    """
    Rebase a price series to a given value.
    Args:
        * prices: Expects a price series
        * value (number): Value to rebase to
    Formula is: (p / p0) * value
    """
    return pl.col(prices) / pl.col(prices).first() * value


def to_drawdown_series(prices):
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
    prices_clean = pl.col(prices).forward_fill()
    hwm = prices_clean.cum_max()
    return prices_clean / hwm - 1


def calc_mtd(daily_prices: pl.Series, monthly_prices: pl.Series | None = None) -> float:
    """
    Calculates MTD return from sorted price series.

    :param daily_prices: Polars Series of daily prices, ordered by Date ascending.
    :param monthly_prices: Optional Polars Series of month-end prices, also ordered.
    :return: float (MTD return)
    """
    last = daily_prices[-1]
    if monthly_prices is None or monthly_prices.len() <= 1:
        first = daily_prices[0]
    else:
        first = monthly_prices[-2]

    return (last / first) - 1


def calc_ytd(daily_prices: pl.Series, yearly_prices: pl.Series | None = None) -> float:
    """
    Calculates year-to-date return from sorted price series.

    :param daily_prices: Polars Series of daily prices, sorted ascending.
    :param yearly_prices: Optional Polars Series of year-end prices, sorted ascending.
    :return: float (YTD return)
    """
    last = daily_prices[-1]
    if yearly_prices is None or yearly_prices.len() <= 1:
        first = daily_prices[0]
    else:
        first = yearly_prices[-2]

    return (last / first) - 1


def calc_max_drawdown(prices):
    """
    Calculates the max drawdown of a price series. If you want the
    actual drawdown series, please use to_drawdown_series.
    """
    return ((pl.col(prices) / pl.col(prices).cum_max()).min() - 1).alias("max_drawdown")


def year_frac(date_col: str) -> pl.Expr:
    """
    Returns a Polars expression that computes the year fraction between the first and last date
    in a column, assuming average year length (365.25 days = 31557600 seconds).
    """
    return (
        pl.col(date_col).cast(pl.Datetime).last()
        - pl.col(date_col).cast(pl.Datetime).first()
    ).dt.total_seconds() / 31_557_600


def calc_cagr(prices, dt):
    """
    Calculates the `CAGR (compound annual growth rate) <https://www.investopedia.com/terms/c/cagr.asp>`_ for a given price series.

    Args:
        * prices (pandas.Series): A Series of prices.
    Returns:
        * float -- cagr.

    """
    return (
        (pl.col(prices).last() / pl.col(prices).first()) ** (1 / year_frac(dt)) - 1
    ).alias("cagr")


def infer_freq(date_col: str) -> pl.Expr:
    """
    Infers human-readable calendar frequency label from a datetime column.
    Works best for: yearly, quarterly, monthly, weekly, daily.

    Returns: "yearly", "quarterly", "monthly", "weekly", "daily", or None
    """
    deltas = (
        (
            pl.col(date_col)
            .cast(pl.Datetime)
            .sort()
            .diff()
            .dt.total_nanoseconds()
            .cast(pl.Float64)
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


def deannualize(col: str, nperiods: int) -> pl.Expr:
    """
    Returns a Polars expression that converts annualized returns to periodic returns.

    Args:
        col (str): Name of the column containing annualized returns
        nperiods (int): Number of periods per year (e.g., 252 for daily)
    """
    return (pl.col(col) + 1.0) ** (1.0 / nperiods) - 1.0


def to_excess_returns(
    returns_col: str, rf: Union[float, str], nperiods: int
) -> pl.Expr:
    """
    Returns a Polars expression that computes excess returns.

    Args:
        returns_col (str): Column with actual returns
        rf (float | str): Either an annualized float or column name with risk-free returns
        nperiods (int): Number of periods per year (used for deannualization if rf is a float)
    """
    if isinstance(rf, float):
        if rf == 0:
            return pl.col(returns_col).alias(returns_col)
        else:
            return (pl.col(returns_col) - ((1 + rf) ** (1 / nperiods) - 1)).alias(
                returns_col
            )
    elif isinstance(rf, str):
        return (pl.col(returns_col) - pl.col(rf)).alias(returns_col)
    else:
        raise TypeError("rf must be either a float or a column name string")


def calc_sharpe(
    returns_col: str,
    rf: Union[float, str] = 0.0,
    nperiods: int = 252,
    annualize: bool = True,
) -> pl.Expr:
    """
    Polars expression that computes the Sharpe ratio in-place without aliasing.
    """
    excess_expr = to_excess_returns(returns_col, rf, nperiods)

    sharpe_expr = (
        (excess_expr.mean() / excess_expr.std(ddof=1)) * math.sqrt(nperiods)
        if annualize
        else (excess_expr.mean() / excess_expr.std(ddof=1))
    ).alias(f"{returns_col}_sharpe")

    return sharpe_expr


def calc_risk_return_ratio(returns):
    """
    Calculates the return / risk ratio. Basically the
    `Sharpe ratio <https://www.investopedia.com/terms/s/sharperatio.asp>`_ without factoring in the `risk-free rate <https://www.investopedia.com/terms/r/risk-freerate.asp>`_.
    """
    return calc_sharpe(returns)


def calc_information_ratio(returns_col: str, benchmark_col: str) -> pl.Expr:
    """
    Returns a Polars expression that computes the Information Ratio.

    Args:
        returns_col: name of the column with asset returns
        benchmark_col: name of the column with benchmark returns
    """
    diff = pl.col(returns_col) - pl.col(benchmark_col)

    return (
        (diff.mean() / diff.std(ddof=1))
        .fill_nan(0.0)
        .fill_null(0.0)
        .alias(f"{returns_col}_ir")
    )


def calc_prob_mom(a: str, b: str) -> pl.Expr:
    """
    Polars expression that computes probabilistic momentum between two return columns.
    If Rust plugin is available, uses it. Otherwise, falls back to a Polars map_batches version.
    """
    if _HAS_RUST:
        return (
            pl.struct([pl.col(a), pl.col(b)])
            .map_batches(
                lambda s: _rust.prob_mom(s.struct.field(a), s.struct.field(b)),
                return_dtype=pl.Float64,
            )
            .alias("prob_momentum")
        )

    # fallback: pure Polars map
    diff = pl.col(a) - pl.col(b)
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


def calc_total_return(prices):
    """
    Calculates the total return of a series.

    last / first - 1
    """
    return (pl.col(prices).last() / pl.col(prices).first()) - 1


def annualize(returns_col: str, durations_col: str, one_year: float = 365.0) -> pl.Expr:
    """
    Returns a Polars expression to annualize returns given durations.

    Args:
        returns_col (str): Name of the column with returns (e.g., 0.05 = 5%).
        durations_col (str): Name of the column with durations (e.g., days held).
        one_year (float): Number of periods in a year (default 365.0 for days).

    Returns:
        pl.Expr: Expression computing annualized return.
    """
    return (
        (1.0 + pl.col(returns_col)) ** (one_year / pl.col(durations_col)) - 1.0
    ).alias("annualized")


def infer_nperiods(col: str, annualization_factor: int | None = None) -> pl.Expr:
    af = annualization_factor or TRADING_DAYS_PER_YEAR

    return (
        pl.col(col)
        .cast(pl.Datetime)
        .sort()
        .diff()
        .dt.total_nanoseconds()
        .drop_nulls()
        .map_batches(lambda s: _infer_from_deltas(s, af), return_dtype=pl.Int64)
        .alias("nperiods")
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


def sortino_ratio(
    returns_col: str, rf: float = 0.0, nperiods: int = 252, annualize: bool = True
) -> pl.Expr:
    rf_periodic = rf / nperiods
    excess = pl.col(returns_col) - rf_periodic

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
        sortino *= nperiods**0.5

    return sortino.alias("sortino_ratio")


def calc_calmar_ratio(prices_col: str, date_col: str) -> pl.Expr:
    """
    Returns a Polars expression to compute the Calmar ratio: CAGR / |Max Drawdown|

    Args:
        prices_col (str): Column name of price series
        date_col (str): Column name of date series

    Returns:
        pl.Expr: Calmar ratio expression
    """
    cagr_expr = calc_cagr(prices_col, date_col)
    max_dd_expr = calc_max_drawdown(prices_col).abs()

    return (cagr_expr / max_dd_expr).alias("calmar_ratio")


def ulcer_index(price_col: str) -> pl.Expr:
    """
    Returns a Polars expression to compute the Ulcer Index from a price series.

    Formula:
        1. Compute cumulative max of prices.
        2. Calculate drawdowns: ((price - cummax) / cummax) * 100.
        3. Square drawdowns, take mean, then square root.
    """
    cummax = pl.col(price_col).cum_max()
    drawdown_pct = ((pl.col(price_col) - cummax) / cummax) * 100
    squared_drawdowns = drawdown_pct.pow(2)

    return squared_drawdowns.mean().sqrt().alias("ulcer_index")


def ulcer_performance_index(
    price_col: str, rf: Union[float, str] = 0.0, nperiods: int = None
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
        if rf != 0 and nperiods is None:
            raise ValueError("nperiods must be set when rf is a non-zero float")

        excess_returns = to_return(price_col) - (rf / nperiods if rf != 0 else 0)

    elif isinstance(rf, str):
        # Subtract column rf from returns
        excess_returns = to_return(price_col) - pl.col(rf)
    else:
        raise TypeError("rf must be a float or a string (column name)")

    return (excess_returns.mean() / ulcer_index(price_col)).alias(
        "ulcer_performance_index"
    )


def resample_returns(
    returns: Union[pl.Series, pl.DataFrame],
    func: Callable[[pl.DataFrame], Union[float, int, pl.Series, pl.DataFrame]],
    seed: int = 0,
    num_trials: int = 100,
) -> pl.DataFrame:
    """
    Resample Polars returns and apply a stat function to each sample.

    Avoids `to_dict()` when possible.
    """
    if isinstance(returns, pl.Series):
        returns = returns.to_frame()
    elif not isinstance(returns, pl.DataFrame):
        raise TypeError("returns must be a Polars Series or DataFrame")

    n = returns.height
    rng = np.random.default_rng(seed)

    results = []
    for i in range(num_trials):
        sample_indices = rng.integers(0, n, size=n)
        sample = returns[sample_indices]
        result = func(sample)

        if isinstance(result, (float, int)):
            results.append({"trial": i, "stat": result})

        elif isinstance(result, pl.Series):
            results.append({"trial": i, result.name: result.item()})

        elif isinstance(result, pl.DataFrame):
            if result.height != 1:
                raise ValueError("func must return a DataFrame with exactly one row")
            values = result.row(0)
            keys = result.columns
            results.append({"trial": i, **dict(zip(keys, values))})

        else:
            raise TypeError(f"Unsupported return type from func: {type(result)}")

    return pl.DataFrame(results)
