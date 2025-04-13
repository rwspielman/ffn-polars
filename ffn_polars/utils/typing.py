from typing import Union
import polars as pl

ExprOrStr = Union[str, pl.Expr]


def resolve_expr(expr: ExprOrStr) -> pl.Expr:
    return pl.col(expr) if isinstance(expr, str) else expr
