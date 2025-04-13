import polars as pl
import ffn_polars as ffn
import numpy as np


def test_resample_returns_df_func():
    returns = pl.DataFrame({"r": [0.01, -0.02, 0.015, 0.005]})

    def return_full_stat(df):
        return df.select(
            [pl.col("r").mean().alias("mean"), pl.col("r").std().alias("std")]
        )

    result = returns.ffn.resample_returns(return_full_stat, num_trials=4, seed=1)

    assert result.shape == (4, 3)
    assert {"trial", "mean", "std"} == set(result.columns)
