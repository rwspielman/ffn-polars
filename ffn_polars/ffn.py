import polars as pl
from .expr import expr_funcs
from .df import df_funcs

EXPR_MODULES = [expr_funcs]
DF_EXPR_MODULES = [df_funcs]


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
        for registry in EXPR_MODULES:  # add others like perf_funcs
            for fn in registry.values():
                actual_fn = getattr(fn, "__wrapped__", fn)
                suffix = getattr(actual_fn, "_alias_suffix", None)
                if suffix:
                    suffixes.add(suffix)
        return suffixes


@pl.api.register_dataframe_namespace("ffn")
class FFNDataFrameNamespace:
    def __init__(self, df: pl.DataFrame):
        self._df = df

    def __getattr__(self, name):
        for registry in DF_EXPR_MODULES:
            if name in registry:
                return lambda *args, **kwargs: registry[name](self._df, *args, **kwargs)
        raise AttributeError(f"'ffn' has no method '{name}'")
