from typing import Callable, Union

import polars as pl
from polars._typing import IntoExpr

from ffn_polars.registry import register
from ffn_polars.utils.decorators import auto_alias
from ffn_polars.utils.guardrails import guard_expr


@register(namespace="eod")
@guard_expr("self", expected_dtype=pl.Float64)
@auto_alias("returns")
def to_returns(self) -> pl.Expr:
    """
    Calculates the simple arithmetic returns of a price series.

    Formula is: (t1 / t0) - 1

    """
    return self / self.shift(1) - 1


@register(namespace="eod")
@guard_expr("self", expected_dtype=pl.Float64)
@auto_alias("log_returns")
def to_log_returns(self) -> pl.Expr:
    """
    Calculates the log returns of a price series.

    Formula is: ln(p1/p0)

    """
    return (self / self.shift(1)).log()


@register(namespace="eod")
@guard_expr("self", expected_dtype=pl.Float64)
@guard_expr("date_col", expected_dtype=pl.Datetime)
@auto_alias("mtd")
def calc_mtd(self, date_col: IntoExpr = "Date") -> pl.Expr:
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


@register(namespace="eod")
@auto_alias("ytd")
@guard_expr("self", expected_dtype=pl.Float64)
@guard_expr("date_col", expected_dtype=pl.Datetime)
def calc_ytd(self, date_col: IntoExpr = "Date") -> pl.Expr:
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


@register(namespace="eod")
@guard_expr("self", expected_dtype=pl.Datetime)
@guard_expr("date_col", expected_dtype=pl.Datetime)
@auto_alias("cagr")
def calc_cagr(self, date_col: IntoExpr) -> pl.Expr:
    """
    Calculates the `CAGR (compound annual growth rate) <https://www.investopedia.com/terms/c/cagr.asp>`_ for a given price series.

    Returns:
        * float -- cagr.

    """
    return (self.last() / self.first()) ** (1 / date_col.ffn.year_frac()) - 1


@register(namespace="eod")
@guard_expr("self", expected_dtype=pl.Float64)
@auto_alias("excess")
def to_excess_returns(self, rf: Union[float, str], n: int) -> pl.Expr:
    """
    Returns a Polars expression that computes excess returns.

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


@register(namespace="eod")
@guard_expr("self", expected_dtype=pl.Float64, required_substring="returns")
@auto_alias("price_index")
def to_price_index(self, start=100):
    """
    Returns a price index given a series of returns.

    Assumes arithmetic returns.

    Formula is: cumprod (1+r)
    """
    return (self.fill_null(0.0) + 1).cum_prod() * start


@register(namespace="eod")
@guard_expr("self", expected_dtype=pl.Float64)
@auto_alias("rebased")
def rebase(self, value=100):
    """
    Rebase a price series to a given value.

    Formula is: (p / p0) * value
    """
    return self / self.first() * value


@register(namespace="eod")
@guard_expr("self", expected_dtype=pl.Float64)
@auto_alias("total_return")
def calc_total_return(self) -> pl.Expr:
    """
    Calculates the total return of a series.

    last / first - 1
    """
    return (self.last() / self.first()) - 1
