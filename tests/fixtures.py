import polars as pl
from pytest import fixture

import ffn_polars


@fixture
def df() -> pl.DataFrame:
    try:
        df = pl.read_csv("tests/data/test_data.csv", try_parse_dates=True)
    except FileNotFoundError as e:
        try:
            df = pl.read_csv("data/test_data.csv", try_parse_dates=True)
        except FileNotFoundError as e2:
            raise (str(e2))
    df = df.with_columns(pl.col("Date").str.strptime(pl.Date, "%m/%d/%Y").alias("Date"))
    return df


@fixture
def ts(df) -> pl.Series:
    return df.select(pl.col("Date"), pl.col("AAPL")).to_series()
