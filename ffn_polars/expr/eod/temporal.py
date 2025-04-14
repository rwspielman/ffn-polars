import polars as pl
from polars._typing import IntoExpr

from ffn_polars.config import TRADING_DAYS_PER_YEAR
from ffn_polars.registry import register
from ffn_polars.utils.decorators import auto_alias
from ffn_polars.utils.guardrails import guard_expr


@register(namespace="eod")
@guard_expr("self", expected_dtype=pl.Float64)
@auto_alias("deannualized")
def deannualize(self, n: int) -> pl.Expr:
    """
    Returns a Polars expression that converts annualized returns to periodic returns.

    Args:
        self: column containing annualized returns
        n: Number of periods per year (e.g., 252 for daily)
    """
    return (self + 1.0) ** (1.0 / n) - 1.0


@register(namespace="eod")
@guard_expr("self", expected_dtype=pl.Float64)
@guard_expr("durations", expected_dtype=pl.Float64)
@auto_alias("annualized")
def annualize(self, durations: IntoExpr, one_year: float = 365.0) -> pl.Expr:
    """
    Returns a Polars expression to annualize returns given durations.

    Args:
        self: Name of the column with returns (e.g., 0.05 = 5%).
        durations: Name of the column with durations (e.g., days held).
        one_year: Number of periods in a year (default 365.0 for days).

    Returns:
        pl.Expr: Expression computing annualized return.
    """
    return (1.0 + self) ** (one_year / durations) - 1.0


@register(namespace="eod")
@guard_expr("self", expected_dtype=pl.Datetime)
@auto_alias("nperiods")
def infer_nperiods(self, annualization_factor: int | None = None) -> pl.Expr:
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


@register(namespace="eod")
@guard_expr("date_col", expected_dtype=pl.Datetime)
@auto_alias("inferred_freq")
def infer_freq(self) -> pl.Expr:
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


@register(namespace="eod")
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
