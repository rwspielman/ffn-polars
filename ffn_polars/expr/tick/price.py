import polars as pl
from ffn_polars.utils import auto_alias, guard_expr, ExprOrStr
from ffn_polars.expr.tick.utils import SCALE
from ffn_polars.registry import register


@register(namespace="tick")
@guard_expr("self", expected_dtype=pl.Float64)
@auto_alias("micro_returns")
def calc_micro_returns(self: ExprOrStr) -> pl.Expr:
    """
    Calculates log returns at tick level:
        log(p_t) - log(p_{t-1})

    Returns:
        Tick-level log return series
    """
    return self.log() - self.log().shift(1)


@register(namespace="tick")
@guard_expr("self", expected_dtype=pl.Float64)
@auto_alias("price_volatility_ratio")
def calc_price_volatility_ratio(self: ExprOrStr) -> pl.Expr:
    """
    Computes the coefficient of variation:
        std(price) / mean(price)

    Returns:
        Float: unitless relative volatility
    """
    return self.std().cast(pl.Float64) / self.mean().cast(pl.Float64)


@register(namespace="tick")
@guard_expr("price", expected_dtype=pl.Float64)
@guard_expr("volume", expected_dtype=pl.Float64)
@auto_alias("price_impact")
def calc_price_impact(self: ExprOrStr, volume: ExprOrStr) -> pl.Expr:
    """
    Calculates absolute price impact:
        (last price - first price) / sum(volume)

    Assumes volume is unsigned and price is a float column.

    Returns:
        Float representing absolute price impact per unit volume

    Example:
        df.group_by("ticker").agg(
            calc_price_impact("price", "volume")
        )
    """
    return (self.last() - self.first()) / volume.sum().cast(pl.Float64)
