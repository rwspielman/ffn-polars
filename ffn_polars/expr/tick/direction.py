import polars as pl
from ffn_polars.utils import auto_alias, guard_expr, ExprOrStr
from ffn_polars.expr.tick.utils import SCALE
from ffn_polars.registry import register


@register(namespace="tick")
@guard_expr("self", expected_dtype=pl.Float64)
@auto_alias("direction")
def tick_rule(self: ExprOrStr) -> pl.Expr:
    """
    Infers trade direction using the tick rule:
        +1 if price > prev_price
        -1 if price < prev_price
         0 otherwise
    """
    return (
        pl.when(self > self.shift(1))
        .then(1)
        .when(self < self.shift(1))
        .then(-1)
        .otherwise(0)
    )


@register(namespace="tick")
@guard_expr("self", expected_dtype=pl.Datetime)
@guard_expr("price", expected_dtype=pl.Float64)
def apply_tick_rule_to_volume(self: ExprOrStr, price: ExprOrStr) -> pl.Expr:
    """
    Applies the tick rule to volume data.
    Args:
        self: Volume column
        price: Price column
    Returns:
        Volume with sign based on tick rule
    """
    return self.cast(pl.Float64) * price.ffn.tick_rule()


@register(namespace="tick")
@guard_expr("self", expected_dtype=pl.Float64)
@auto_alias("tick_imbalance")
def calc_tick_imbalance(self: ExprOrStr) -> pl.Expr:
    """
    Calculates tick imbalance using signed volume.

    Args:
        self: signed tick rule / direction column

    Returns:
        Float between -1 and 1
    """
    return self.sum().cast(pl.Float64) / self.len().cast(pl.Float64)
