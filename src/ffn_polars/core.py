import polars as pl
from datetime import date, datetime
from typing import Union
import math


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


def calc_mtd(daily_prices, monthly_prices):
    """
    Calculates mtd return of a price series.
    Use daily_prices if prices are only available from same month
    else use monthly_prices
    """
    if monthly_prices is None:
        return pl.col(daily_prices).last() / pl.col(daily_prices).first() - 1
    else:
        return (
            pl.when(pl.col(monthly_prices).len() == 1)
            .then(pl.col(daily_prices).last() / pl.col(daily_prices).first() - 1)
            .otherwise(
                pl.col(daily_prices).last() / pl.col(monthly_prices).tail(2).first() - 1
            )
        )


def calc_ytd(daily_prices, yearly_prices):
    """
    Calculates ytd return of a price series.
    Use daily_prices if prices are only available from same year
    else use yearly_prices
    """
    if len(yearly_prices) == 1:
        return daily_prices.last() / daily_prices.first() - 1
    else:
        return daily_prices.last() / yearly_prices.tail().first() - 1


def calc_max_drawdown(prices):
    """
    Calculates the max drawdown of a price series. If you want the
    actual drawdown series, please use to_drawdown_series.
    """
    return (pl.col(prices) / pl.col(prices).cum_max()).min() - 1


def drawdown_details(drawdown):
    raise NotImplementedError("Not implemented yet")


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
    return (pl.col(prices).last() / pl.col(prices).first()) ** (1 / year_frac(dt)) - 1


def infer_freq(date_col: str) -> pl.Expr:
    """
    Returns a Polars expression that infers frequency from a datetime column.
    Output: estimated nperiods per year (252, 52, 12, 4, 1), or None
    """
    diffs = pl.col(date_col).cast(pl.Datetime).diff().dt.total_days().drop_nulls()

    # Take the most common delta (mode of the differences)
    most_common_delta = diffs.mode().first()

    return (
        pl.when(most_common_delta == 1)
        .then(pl.lit(252))
        .when(most_common_delta == 7)
        .then(pl.lit(52))
        .when((most_common_delta >= 28) & (most_common_delta <= 31))
        .then(pl.lit(12))
        .when((most_common_delta >= 89) & (most_common_delta <= 92))
        .then(pl.lit(4))
        .when((most_common_delta >= 360) & (most_common_delta <= 366))
        .then(pl.lit(1))
        .otherwise(pl.lit(None))
        .alias("nperiods")
    )


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
