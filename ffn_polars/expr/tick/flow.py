import polars as pl
from ffn_polars.utils import auto_alias, guard_expr, ExprOrStr
from ffn_polars.expr.tick.utils import SCALE
from ffn_polars.registry import register


@register(namespace="tick")
@guard_expr("self", expected_dtype=pl.Int64)
@guard_expr("ts", expected_dtype=pl.Datetime)
@auto_alias("volume_rate")
def calc_volume_rate(self: ExprOrStr, ts: ExprOrStr, per: str = "s") -> pl.Expr:
    """
    Calculates volume traded per unit time.

    Args:
        self: Numeric column of trade volumes
        ts: Datetime column
        per: "s", "ms", "us", or "ns" (default "s")

    Returns:
        Float expression representing volume per unit time

    Example:
        df.group_by("ticker").agg(
            calc_volume_rate("volume", "timestamp", per="ms")
        )
    """
    scale = SCALE.get(per)
    if scale is None:
        raise ValueError(f"Unsupported time unit: {per}")

    return self.sum().cast(pl.Float64) / (
        (ts.max() - ts.min()).dt.total_nanoseconds() / scale
    )


@register(namespace="tick")
@guard_expr("self", expected_dtype=pl.Float64)
@auto_alias("order_flow_imbalance")
def calc_order_flow_imbalance(self: ExprOrStr) -> pl.Expr:
    """
    Calculates Order Flow Imbalance (OFI) as the sum of signed volume.

    Assumes volume is signed:
        +V = buyer-initiated
        -V = seller-initiated

    Returns:
        Float (positive = net buying, negative = net selling)

    Example:
        df.group_by("ticker").agg(
            pl.col("signed_volume").calc_order_flow_imbalance()
        )
    """
    return self.sum().cast(pl.Float64)


@register(namespace="tick")
@guard_expr("self", expected_dtype=pl.Float64)
@guard_expr("volume", expected_dtype=pl.Float64)
@auto_alias("traded_value")
def calc_traded_value(self: ExprOrStr, volume: ExprOrStr) -> pl.Expr:
    """
    Calculates traded value (price × volume sum).

    Args:
        self: Float column of trade prices
        volume: Numeric column of trade volumes

    Returns:
        Float: total traded value (dollar volume)

    Example:
        df.group_by("ticker").agg(
            calc_traded_value("price", "volume")
        )
    """
    return (self * volume).sum().cast(pl.Float64)


@register(namespace="tick")
@guard_expr("price", expected_dtype=pl.Float64)
@guard_expr("volume", expected_dtype=pl.Float64)
@auto_alias("vwap")
def calc_vwap(self: ExprOrStr, volume: ExprOrStr) -> pl.Expr:
    """
    Calculates volume-weighted average price (VWAP).

    Formula:
        VWAP = sum(price * volume) / sum(volume)

    Returns:
        A float expression representing VWAP

    Example:
        df.group_by("ticker").agg(
            calc_vwap("price", "volume")
        )
    """
    return (self * volume).sum() / volume.sum().cast(pl.Float64)
