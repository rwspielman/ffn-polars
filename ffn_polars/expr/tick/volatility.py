import polars as pl
from ffn_polars.utils import auto_alias, guard_expr, ExprOrStr
from ffn_polars.expr.tick.utils import SCALE
from ffn_polars.registry import register


@register(namespace="tick")
@guard_expr("self", expected_dtype=pl.Float64)
@auto_alias("realized_volatility")
def calc_realized_volatility(self: ExprOrStr) -> pl.Expr:
    """
    Calculates realized volatility (non-annualized) from a price series.

    Formula:
        sqrt(Σ (log(p_t) - log(p_{t-1}))^2)

    Assumes `self` is a price series. Use inside `.select()` or `.group_by().agg()`.

    Returns:
        Realized volatility over the window

    Example:
        df.group_by("ticker").agg(
            pl.col("price").calc_realized_volatility()
        )
    """
    log_returns = self.log() - self.log().shift(1)
    return (log_returns**2).sum().sqrt()
