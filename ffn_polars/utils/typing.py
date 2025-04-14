import polars as pl


def resolve_expr(expr) -> pl.Expr:
    return pl.col(expr) if isinstance(expr, str) else expr
