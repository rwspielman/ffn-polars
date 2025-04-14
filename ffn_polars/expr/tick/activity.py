import polars as pl
from ffn_polars.utils import auto_alias, guard_expr, ExprOrStr
from ffn_polars.expr.tick.utils import SCALE
from ffn_polars.registry import register


@register(namespace="tick")
@guard_expr("self", expected_dtype=pl.Datetime)
@auto_alias("trade_rate")
def calc_trade_rate(self: ExprOrStr, per: str = "ms") -> pl.Expr:
    """
    Calculates trade rate as number of trades per second.

    Assumes `self` is a timestamp column from tick data.

    Returns:
        An expression representing trades per second:
        (count of rows) / (max(timestamp) - min(timestamp)).seconds

    Example:
        df.group_by("ticker").agg(
            pl.col("timestamp").calc_trade_rate()
        )
    """
    scale = SCALE.get(per)
    return pl.count().cast(pl.Float64) / (
        (self.last() - self.first()).dt.total_nanoseconds() / scale
    )


@register(namespace="tick")
@guard_expr("self", expected_dtype=pl.Datetime)
@auto_alias("inter_trade_time")
def calc_inter_trade_time(self: ExprOrStr, per: str = "s") -> pl.Expr:
    """
    Calculates the average time between consecutive trades.

    Args:
        self: Timestamp column
        per: Time unit — "s", "ms", "us", or "ns"

    Returns:
        Float expression representing mean inter-trade time in desired unit

    Example:
        df.group_by("ticker").agg(
            pl.col("timestamp").calc_inter_trade_time(per="ms")
        )
    """
    scale = SCALE.get(per)
    if scale is None:
        raise ValueError(f"Unsupported time unit: {per}")

    return self.diff().dt.total_nanoseconds().mean() / scale


@register(namespace="tick")
@guard_expr("self", expected_dtype=pl.Datetime)
@auto_alias("burstiness")
def calc_burstiness(self: ExprOrStr, per: str = "s") -> pl.Expr:
    """
    Calculates burstiness as std(inter-trade time) / mean(inter-trade time).

    Args:
        self: timestamp column
        per: time unit ("s", "ms", "us", "ns")

    Returns:
        Float representing burstiness of trading activity.
    """
    scale = SCALE.get(per)
    if scale is None:
        raise ValueError(f"Unsupported unit: {per}")

    itt_ns = self.diff().dt.total_nanoseconds()
    return itt_ns.std().cast(pl.Float64) / itt_ns.mean().cast(pl.Float64)
