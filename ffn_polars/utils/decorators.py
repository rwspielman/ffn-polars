from functools import wraps

import polars as pl


def auto_alias(suffix: str):
    def decorator(func):
        @wraps(func)
        def wrapper(self: pl.Expr, *args, **kwargs):
            result = func(self, *args, **kwargs)

            if not isinstance(result, pl.Expr):
                return result

            try:
                base_name = self.meta.output_name()
                from ffn_polars import FFNNamespace

                known_suffixes = FFNNamespace.extract_all_alias_suffixes()

                # Only remove suffixes if they were auto-added before
                for sfx in known_suffixes:
                    if base_name.endswith(f"_{sfx}"):
                        base_name = base_name[: -len(sfx) - 1]
                        break

                if not base_name:
                    return result

                if base_name.endswith(f"_{suffix}"):
                    alias_name = base_name  # already suffixed
                else:
                    alias_name = f"{base_name}_{suffix}"

                return result.alias(alias_name)
            except Exception as e:
                return result

        wrapper._alias_suffix = suffix
        return wrapper

    return decorator
