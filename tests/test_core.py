from .fixtures import df, ts
import polars as pl
import ffn_polars as ffn
from datetime import datetime, date
import pytest
import math


def aae(actual: float, expected: float, places: int = 3):
    """Assert that two floats are equal up to a specified number of decimal places."""
    rounded_actual = round(actual, places)
    rounded_expected = round(expected, places)
    assert rounded_actual == rounded_expected, (
        f"Assertion failed: {rounded_actual} != {rounded_expected} "
        f"(rounded to {places} places)"
    )


def test_to_returns(df):
    data = df
    actual = data.select(ffn.to_return("AAPL"))

    assert actual.height == data.height
    assert actual["AAPL"][0] is None
    aae(actual["AAPL"][1], -0.019)
    aae(actual["AAPL"][9], -0.022)


def test_to_log_returns(df):
    data = df
    actual = data.select(ffn.to_log_returns("AAPL"))

    assert actual.height == data.height
    assert actual["AAPL"][0] is None
    aae(actual["AAPL"][1], -0.019)
    aae(actual["AAPL"][9], -0.022)


def test_to_price_index(df):
    data = df
    actual = data.select(ffn.to_return("AAPL").alias("AAPL_ret")).select(
        ffn.to_price_index("AAPL_ret").alias("AAPL"),
    )

    assert actual.height == data.height
    aae(actual["AAPL"][0], 100)
    aae(actual["AAPL"][9], 91.366, 3)

    actual = data.select(ffn.to_return("AAPL").alias("AAPL_ret")).select(
        ffn.to_price_index("AAPL_ret", start=1).alias("AAPL"),
    )

    assert actual.height == data.height
    aae(actual["AAPL"][0], 1, 3)
    aae(actual["AAPL"][9], 0.914, 3)


def test_rebase(df):
    data = df
    actual = data.select(ffn.rebase("AAPL").alias("AAPL"))

    assert actual.height == data.height
    aae(actual["AAPL"][0], 100, 3)
    aae(actual["AAPL"][9], 91.366, 3)


def test_to_drawdown_series_df(df):
    data = df
    actual = data.select(
        ffn.to_drawdown_series("AAPL").alias("AAPL"),
    )

    assert len(actual) == len(data)
    aae(actual["AAPL"][0], 0, 3)
    aae(actual["AAPL"][1], -0.019, 3)
    aae(actual["AAPL"][9], -0.086, 3)


def test_max_drawdown_df(df):
    data = df
    data = data[0:10]
    actual = data.select(
        ffn.calc_max_drawdown("AAPL").alias("AAPL"),
        ffn.calc_max_drawdown("MSFT").alias("MSFT"),
        ffn.calc_max_drawdown("C").alias("C"),
    )

    aae(actual["AAPL"][0], -0.086, 3)
    aae(actual["MSFT"][0], -0.048, 3)
    aae(actual["C"][0], -0.033, 3)


def test_year_frac(df):
    actual = df.select(ffn.year_frac("Date"))
    # not exactly the same as excel but close enough
    aae(actual["Date"][0], 9.9986, 4)


def test_cagr_df(df):
    data = df
    actual = data.select(
        ffn.calc_cagr("AAPL", "Date").alias("AAPL"),
        ffn.calc_cagr("MSFT", "Date").alias("MSFT"),
        ffn.calc_cagr("C", "Date").alias("C"),
    )
    aae(actual["AAPL"][0], 0.440, 3)
    aae(actual["MSFT"][0], 0.041, 3)
    aae(actual["C"][0], -0.205, 3)


@pytest.mark.parametrize(
    "interval_days,expected",
    [
        (1, 252),  # daily
        (7, 52),  # weekly
        (30, 12),  # monthly-ish
        (90, 4),  # quarterly-ish
        (365, 1),  # yearly-ish
        (5, None),  # unsupported interval
    ],
)
def test_infer_freq_expr(interval_days, expected):
    df = pl.DataFrame(
        {
            "Date": pl.date_range(
                start=date(2020, 1, 1),
                interval=f"{interval_days}d",
                eager=True,
                end=date(2022, 1, 1),
            )
        }
    )

    result = df.select(ffn.infer_freq("Date")).item()
    assert result == expected


@pytest.mark.parametrize(
    "r_annual, nperiods, expected",
    [
        (0.12, 12, (1 + 0.12) ** (1 / 12) - 1),
        (0.10, 252, (1 + 0.10) ** (1 / 252) - 1),
    ],
)
def test_deannualize(r_annual, nperiods, expected):
    df = pl.DataFrame({"r": [r_annual]})
    result = df.select(ffn.deannualize("r", nperiods)).item()
    assert math.isclose(result, expected, rel_tol=1e-9)


@pytest.mark.parametrize(
    "returns, rf, nperiods, expected",
    [
        # float RF
        ([0.05], 0.12, 12, 0.05 - ((1 + 0.12) ** (1 / 12) - 1)),
        ([0.03], 0.0, 252, 0.03),
    ],
)
def test_to_excess_returns_expr_with_float_rf(returns, rf, nperiods, expected):
    df = pl.DataFrame({"returns": returns})
    result = df.select(
        ffn.to_excess_returns("returns", rf=rf, nperiods=nperiods)
    ).item()
    assert math.isclose(result, expected, rel_tol=1e-10)


def test_to_excess_returns_expr_with_column_rf():
    df = pl.DataFrame({"returns": [0.05, 0.03, 0.04], "rf": [0.01, 0.01, 0.01]})
    result = df.select(ffn.to_excess_returns("returns", rf="rf", nperiods=252))[
        "returns"
    ].to_list()
    expected = [0.04, 0.02, 0.03]
    for r, e in zip(result, expected):
        assert math.isclose(r, e, rel_tol=1e-10)


def test_to_excess_returns_expr_raises_on_bad_rf_type():
    df = pl.DataFrame({"returns": [0.01, 0.02]})
    with pytest.raises(TypeError):
        df.select(ffn.to_excess_returns("returns", rf=[0.01], nperiods=252))


def test_calc_risk_return_ratio_simple_case():
    returns = pl.DataFrame({"returns": [0.01, 0.02, -0.01, 0.03, -0.02]})
    expected_ratio = returns.mean() / returns.std() * math.sqrt(252)
    result = returns.select(ffn.calc_risk_return_ratio("returns").alias("ratio"))
    aae(result["ratio"][0], expected_ratio["returns"][0])


def test_calc_sharpe_expr_basic():
    df = pl.DataFrame({"returns": [0.01, 0.02, -0.01, 0.005]})
    result = df.select(ffn.calc_sharpe("returns", rf=0.0, nperiods=252, annualize=True))
    sharpe = result["returns_sharpe"][0]
    assert isinstance(sharpe, float)
    assert sharpe != 0


@pytest.mark.parametrize(
    "returns, benchmark",
    [
        ([0.02, 0.01, -0.005, 0.003], [0.015, 0.008, -0.003, 0.002]),
        ([0.03, 0.02, 0.01], [0.01, 0.015, 0.005]),
        ([0.01, 0.01, 0.01], [0.01, 0.01, 0.01]),  # std = 0 → IR = 0
    ],
)
def test_calc_information_ratio_expr(returns, benchmark):
    df = pl.DataFrame({"r": returns, "b": benchmark})
    result = df.select(ffn.calc_information_ratio("r", "b"))["r_ir"][0]

    # Compute expected IR manually using standard Python
    diffs = [r - b for r, b in zip(returns, benchmark)]
    mean_diff = sum(diffs) / len(diffs)
    std_diff = (sum((x - mean_diff) ** 2 for x in diffs) / (len(diffs) - 1)) ** 0.5
    expected = 0.0 if std_diff == 0 else mean_diff / std_diff

    assert math.isclose(result, expected, rel_tol=1e-9)


def is_close(a: float, b: float, rtol: float = 1e-6) -> bool:
    return abs(a - b) <= rtol * max(abs(a), abs(b))


def test_annualize_basic():
    df = pl.DataFrame(
        {
            "returns": [0.05, 0.10, 0.25],
            "durations": [30, 90, 180],
        }
    )

    result = df.with_columns(ffn.annualize("returns", "durations"))

    expected = [
        (1 + 0.05) ** (365.0 / 30) - 1,
        (1 + 0.10) ** (365.0 / 90) - 1,
        (1 + 0.25) ** (365.0 / 180) - 1,
    ]

    for actual, exp in zip(result["annualized"], expected):
        assert is_close(actual, exp)


def test_annualize_zero_return():
    df = pl.DataFrame(
        {
            "returns": [0.0],
            "durations": [60],
        }
    )
    result = df.with_columns(ffn.annualize("returns", "durations"))
    assert result["annualized"][0] == 0.0


def test_annualize_one_day_duration():
    df = pl.DataFrame(
        {
            "returns": [0.01],
            "durations": [1],
        }
    )
    result = df.with_columns(ffn.annualize("returns", "durations"))
    expected = (1 + 0.01) ** 365.0 - 1
    assert is_close(result["annualized"][0], expected)


def test_annualize_large_return_long_duration():
    df = pl.DataFrame(
        {
            "returns": [10.0],
            "durations": [730],
        }
    )
    result = df.with_columns(ffn.annualize("returns", "durations"))
    expected = (1 + 10.0) ** (365.0 / 730) - 1
    assert is_close(result["annualized"][0], expected)
