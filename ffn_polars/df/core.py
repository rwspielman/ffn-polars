from typing import Callable, Union

import numpy as np
import polars as pl

from ffn_polars.registry import register


@register(kind="df")
def resample_returns(
    self,
    func: Callable[[pl.DataFrame], Union[float, int, pl.Series, pl.DataFrame]],
    seed: int = 0,
    num_trials: int = 100,
) -> pl.DataFrame:
    """
    Resample Polars returns and apply a stat function to each sample.

    Avoids `to_dict()` when possible.
    """

    n = self.height
    rng = np.random.default_rng(seed)

    results = []
    for i in range(num_trials):
        sample_indices = rng.integers(0, n, size=n)
        sample = self[sample_indices]
        result = func(sample)

        if isinstance(result, (float, int)):
            results.append({"trial": i, "stat": result})

        elif isinstance(result, pl.Series):
            results.append({"trial": i, result.name: result.item()})

        elif isinstance(result, pl.DataFrame):
            if result.height != 1:
                raise ValueError("func must return a DataFrame with exactly one row")
            values = result.row(0)
            keys = result.columns
            results.append({"trial": i, **dict(zip(keys, values))})

        else:
            raise TypeError(f"Unsupported return type from func: {type(result)}")

    return pl.DataFrame(results)
