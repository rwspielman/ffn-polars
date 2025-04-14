import warnings
from enum import Enum, auto
from functools import wraps
from typing import Any

import polars as pl

from .typing import resolve_expr


# --- Mode config ---
class GuardMode(Enum):
    WARN = auto()
    STRICT = auto()
    SILENT = auto()


GUARD_CONFIG = {"mode": GuardMode.WARN}


def set_guard_mode(mode: GuardMode):
    GUARD_CONFIG["mode"] = mode


def _guard_mode(strict: bool | None) -> GuardMode:
    if strict is True:
        return GuardMode.STRICT
    elif strict is False:
        return GuardMode.WARN
    else:
        return GUARD_CONFIG["mode"]


# --- Internal ---
def _emit_guard(reason, param_name, root, dtype, override_mode=None):
    msg = f"[guard] Column '{root}' (param '{param_name}') {reason}."
    mode = override_mode or GUARD_CONFIG["mode"]

    if mode == GuardMode.STRICT:
        raise ValueError(msg)
    elif mode == GuardMode.WARN:
        warnings.warn(msg, stacklevel=4)


def _validate_expr(
    expr: pl.Expr, param: str, checks: dict[str, Any], override_mode=None
):
    try:
        root = expr.meta.root_names()[0]
        dtype = expr.meta.output_type()
    except Exception:
        _emit_guard(
            "could not be inspected", param, "unknown", "unknown", override_mode
        )
        return

    if "expected_dtype" in checks and dtype != checks["expected_dtype"]:
        _emit_guard(
            f"has dtype {dtype}, expected {checks['expected_dtype']}",
            param,
            root,
            dtype,
            override_mode,
        )

    if "required_substring" in checks and checks["required_substring"] not in root:
        _emit_guard(
            f"name does not contain '{checks['required_substring']}'",
            param,
            root,
            dtype,
            override_mode,
        )


# --- Decorator ---
def guard_expr(
    param: str,
    *,
    expected_dtype=None,
    required_substring=None,
    strict: bool | None = None,
):
    def decorator(func):
        if not hasattr(func, "_guard_checks"):
            func._guard_checks = []

        func._guard_checks.append(
            {
                "param": param,
                "checks": {
                    "expected_dtype": expected_dtype,
                    "required_substring": required_substring,
                },
                "strict": strict,
            }
        )

        @wraps(func)
        def wrapper(self: pl.Expr | str, *args, **kwargs):
            from inspect import signature

            sig = signature(func)
            param_names = list(sig.parameters.keys())

            # Bind all inputs for lookup and update
            bound_args = dict(zip(param_names, (self, *args)))
            bound_args.update(kwargs)

            # Prepare values to pass to the actual function
            resolved_self = self
            resolved_args = list(args)
            resolved_kwargs = dict(kwargs)

            for guard in getattr(func, "_guard_checks", []):
                param = guard["param"]
                checks = guard["checks"]
                strict_mode = guard["strict"]

                if param in bound_args:
                    original = bound_args[param]

                    # Only resolve strings and pl.Expr; leave bools, ints, etc. untouched
                    if isinstance(original, (str, pl.Expr)):
                        expr = resolve_expr(original)

                        # Pass resolved expr to actual call
                        if param == "self":
                            resolved_self = expr
                        elif param in param_names[: len(args)]:
                            index = param_names.index(param) - 1  # account for self
                            resolved_args[index] = expr
                        else:
                            resolved_kwargs[param] = expr

                        _validate_expr(
                            expr,
                            param,
                            checks,
                            override_mode=_guard_mode(strict_mode),
                        )

            return func(resolved_self, *resolved_args, **resolved_kwargs)

        return wrapper

    return decorator
