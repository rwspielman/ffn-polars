import importlib
import pkgutil
from collections.abc import Iterable

FFN_REGISTRY = {}
FFN_DF_REGISTRY = {}
FFN_TICK_REGISTRY = {}
FFN_EOD_REGISTRY = {}


def register(namespace="tick", kind="expr", alias=None):
    def decorator(func):
        aliases = (
            alias
            if isinstance(alias, Iterable) and not isinstance(alias, str)
            else [alias or func.__name__]
        )

        for name in aliases:
            if kind == "expr":
                FFN_REGISTRY[name] = func
                if namespace == "tick":
                    FFN_TICK_REGISTRY[name] = func
                elif namespace == "eod":
                    FFN_EOD_REGISTRY[name] = func
            elif kind == "df":
                FFN_DF_REGISTRY[name] = func
            else:
                raise ValueError(f"Unsupported kind: {kind}")
        return func

    return decorator


def import_all_modules_in(*packages):
    for pkg in packages:
        for _, modname, ispkg in pkgutil.walk_packages(
            pkg.__path__, pkg.__name__ + "."
        ):
            if not ispkg:
                importlib.import_module(modname)


def import_all_expr_modules():
    import ffn_polars.df
    import ffn_polars.expr

    import_all_modules_in(ffn_polars.expr, ffn_polars.df)
