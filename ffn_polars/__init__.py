import polars as pl
from ffn_polars.registry import (
    FFN_REGISTRY,
    FFN_TICK_REGISTRY,
    FFN_EOD_REGISTRY,
    FFN_DF_REGISTRY,
)
from ffn_polars.registry import import_all_expr_modules

EXPR_MODULES = [FFN_REGISTRY]
DF_EXPR_MODULES = [FFN_DF_REGISTRY]

import_all_expr_modules()


@pl.api.register_expr_namespace("ffn")
class FFNNamespace:
    def __init__(self, expr: pl.Expr):
        self._expr = expr

    def __getattr__(self, name):
        for registry in EXPR_MODULES:
            if name in registry:
                return lambda *args, **kwargs: registry[name](
                    self._expr, *args, **kwargs
                )
        raise AttributeError(f"'ffn' has no method '{name}'")

    @staticmethod
    def extract_all_alias_suffixes() -> set[str]:
        suffixes = set()
        for registry in EXPR_MODULES:
            for fn in registry.values():
                actual_fn = getattr(fn, "__wrapped__", fn)
                suffix = getattr(actual_fn, "_alias_suffix", None)
                if suffix:
                    suffixes.add(suffix)
        return suffixes


@pl.api.register_expr_namespace("ffn_tick")
class FFNTickNamespace:
    def __init__(self, expr: pl.Expr):
        self._expr = expr

    def __getattr__(self, name):
        if name in FFN_TICK_REGISTRY:
            return lambda *args, **kwargs: FFN_TICK_REGISTRY[name](
                self._expr, *args, **kwargs
            )
        raise AttributeError(f"'ffn_tick' has no method '{name}'")


@pl.api.register_expr_namespace("ffn_eod")
class FFNEODNamespace:
    def __init__(self, expr: pl.Expr):
        self._expr = expr

    def __getattr__(self, name):
        if name in FFN_EOD_REGISTRY:
            return lambda *args, **kwargs: FFN_EOD_REGISTRY[name](
                self._expr, *args, **kwargs
            )
        raise AttributeError(f"'ffn_eod' has no method '{name}'")


@pl.api.register_dataframe_namespace("ffn")
class FFNDataFrameNamespace:
    def __init__(self, df: pl.DataFrame):
        self._df = df

    def __getattr__(self, name):
        for registry in DF_EXPR_MODULES:
            if name in registry:
                return lambda *args, **kwargs: registry[name](self._df, *args, **kwargs)
        raise AttributeError(f"'ffn' has no method '{name}'")
