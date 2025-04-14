import math
from datetime import date, datetime, timedelta

import numpy as np
import polars as pl
import pytest

import ffn_polars as ffn
from ffn_polars.config import TRADING_DAYS_PER_YEAR
from tests.fixtures import df, ts
from tests.utils import aae


def test_to_returns(df):
    data = df
    actual = data.select(pl.col("AAPL").ffn.to_returns())

    assert actual.height == data.height
    assert actual["AAPL_returns"][0] is None
    aae(actual["AAPL_returns"][1], -0.019)
    aae(actual["AAPL_returns"][9], -0.022)


def test_to_log_returns(df):
    data = df
    actual = data.select(pl.col("AAPL").ffn.to_log_returns())

    assert actual.height == data.height
    assert actual["AAPL_log_returns"][0] is None
    aae(actual["AAPL_log_returns"][1], -0.019)
    aae(actual["AAPL_log_returns"][9], -0.022)


def test_to_price_index(df):
    data = df
    actual = data.select(pl.col("AAPL").ffn.to_returns().ffn.to_price_index())

    assert actual.height == data.height
    aae(actual["AAPL_price_index"][0], 100)
    aae(actual["AAPL_price_index"][9], 91.366, 3)

    actual = data.select(pl.col("AAPL").ffn.to_returns().ffn.to_price_index(start=1))


    assert actual.height == data.height
    aae(actual["AAPL_price_index"][0], 1, 3)
    aae(actual["AAPL_price_index"][9], 0.914, 3)


def test_rebase(df):
    data = df
    actual = data.select(pl.col("AAPL").ffn.rebase())

    assert actual.height == data.height
    aae(actual["AAPL_rebased"][0], 100, 3)
    aae(actual["AAPL_rebased"][9], 91.366, 3)


def test_to_drawdown_series_df(df):
    data = df
    actual = data.select(
        pl.col("AAPL").ffn.to_drawdown_series()
    )

    assert len(actual) == len(data)
    aae(actual["AAPL_drawdowns"][0], 0, 3)
    aae(actual["AAPL_drawdowns"][1], -0.019, 3)
    aae(actual["AAPL_drawdowns"][9], -0.086, 3)


def test_calc_mtd_basic():
    df = pl.DataFrame({
        "Date": pl.date_range(date(2024,3,1), date(2024,4,10), eager=True),
        "AAPL": [100.0 + i for i in range(41)]  # AAPL ends at 140, starts at 100
    })

    result = df.select(
        pl.col("AAPL").ffn.calc_mtd(date_col="Date").alias("AAPL_mtd")
    )

    # Latest: 140, last price of previous month (March): 130
    expected = (140 / 130) - 1  # ≈ 0.0769
    actual = result["AAPL_mtd"][0]
    aae(actual, expected, 4)

def test_calc_ytd_basic():
    df = pl.DataFrame({
        "Date": pl.date_range(date(2024,1,1), date(2024,4,9), eager=True),
        "AAPL": [100.0 + i for i in range(100)]  # Starts at 100, ends at 199
    })

    result = df.select(
        pl.col("AAPL").ffn.calc_ytd(date_col="Date").alias("AAPL_ytd")
    )

    expected = (199 / 100) - 1  # = 0.99
    actual = result["AAPL_ytd"][0]
    aae(actual, expected, 4)

def test_max_drawdown_df(df):
    data = df
    data = data[0:10]
    actual = data.select(
        pl.col("AAPL").ffn.calc_max_drawdown().alias("AAPL"),
        pl.col("MSFT").ffn.calc_max_drawdown().alias("MSFT"),
        pl.col("C").ffn.calc_max_drawdown().alias("C"),
    )

    aae(actual["AAPL"][0], -0.086, 3)
    aae(actual["MSFT"][0], -0.048, 3)
    aae(actual["C"][0], -0.033, 3)


def test_year_frac(df):
    actual = df.select(pl.col("Date").ffn.year_frac())
    # not exactly the same as excel but close enough
    aae(actual["Date_year_frac"][0], 9.9986, 4)


def test_cagr_df(df):
    data = df
    actual = data.select(
        pl.col("AAPL").ffn.calc_cagr(date_col="Date"),
        pl.col("MSFT").ffn.calc_cagr(date_col="Date"),
        pl.col("C").ffn.calc_cagr(date_col="Date"),
    )
    aae(actual["AAPL_cagr"][0], 0.440, 3)
    aae(actual["MSFT_cagr"][0], 0.041, 3)
    aae(actual["C_cagr"][0], -0.205, 3)


def make_df(n: int, freq: str | timedelta, start: str = "2024-01-01 00:00:00", value: bool = True) -> pl.DataFrame:
    start_dt = datetime.strptime(start, "%Y-%m-%d %H:%M:%S")

    if isinstance(freq, timedelta):
        delta = freq
    else:
        match freq:
            case "1s": delta = timedelta(seconds=1)
            case "1m": delta = timedelta(minutes=1)
            case "1h": delta = timedelta(hours=1)
            case "1d": delta = timedelta(days=1)
            case "1mo": delta = timedelta(days=31)
            case "1y": delta = timedelta(days=366)
            case "100ms": delta = timedelta(milliseconds=100)
            case _: raise ValueError(f"Unsupported freq: {freq}")

    timestamps = [start_dt + i * delta for i in range(n)]
    data = {"Date": timestamps}
    if value:
        data["value"] = np.random.randn(n)
    return pl.DataFrame(data)

@pytest.mark.parametrize(
    "delta,expected",
    [
        (timedelta(days=1), "daily"),
        (timedelta(days=7), "weekly"),
        (timedelta(days=30), "monthly"),
        (timedelta(days=91), "quarterly"),
        (timedelta(days=365), "yearly"),
    ]
)
def test_infer_freq_labels(delta, expected):
    df = make_df(10, delta)
    result = df.select(pl.col("Date").ffn.infer_freq()).item()
    assert result == expected

def test_infer_freq_unknown():
    # Irregular intervals: 1d, 5d, 20d...
    dt = datetime(2024, 1, 1)
    dates = [dt + timedelta(days=x) for x in [0, 1, 6, 26, 55]]
    df = pl.DataFrame({"Date": dates})
    result = df.select(pl.col("Date").ffn.infer_freq()).item()
    assert result == "unknown"

@pytest.mark.parametrize(
    "r_annual, nperiods, expected",
    [
        (0.12, 12, (1 + 0.12) ** (1 / 12) - 1),
        (0.10, 252, (1 + 0.10) ** (1 / 252) - 1),
    ],
)
def test_deannualize(r_annual, nperiods, expected):
    df = pl.DataFrame({"r": [r_annual]})
    result = df.select(pl.col("r").ffn.deannualize(n=nperiods)).item()
    aae(result, expected, 8)


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
    result = df.select(pl.col("returns").ffn.to_excess_returns(rf=rf, n=nperiods)).item()
    aae(result, expected, 8)


def test_to_excess_returns_expr_with_column_rf():
    df = pl.DataFrame({"returns": [0.05, 0.03, 0.04], "rf": [0.01, 0.01, 0.01]})
    result = df.select(pl.col("returns").ffn.to_excess_returns(rf="rf", n=252))[
        "returns_excess"
    ].to_list()
    expected = [0.04, 0.02, 0.03]
    for r, e in zip(result, expected):
        aae(r, e, 8)


def test_to_excess_returns_expr_raises_on_bad_rf_type():
    df = pl.DataFrame({"returns": [0.01, 0.02]})
    with pytest.raises(TypeError):
        df.select(pl.col("returns").ffn.to_excess_returns(rf=[0.01], nperiods=252))


def test_calc_risk_return_ratio_simple_case():
    returns = pl.DataFrame({"returns": [0.01, 0.02, -0.01, 0.03, -0.02]})
    expected_ratio = returns.mean() / returns.std() * math.sqrt(252)
    result = returns.select(pl.col("returns").ffn.calc_risk_return_ratio())
    aae(result["returns_risk_return_ratio"][0], expected_ratio["returns"][0])


def test_calc_sharpe_expr_basic():
    df = pl.DataFrame({"returns": [0.01, 0.02, -0.01, 0.005]})
    result = df.select(pl.col("returns").ffn.calc_sharpe(rf=0.0, n=252, annualize=True))
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
    result = df.select(pl.col("r").ffn.calc_information_ratio(benchmark="b"))["r_ir"][0]

    # Compute expected IR manually using standard Python
    diffs = [r - b for r, b in zip(returns, benchmark)]
    mean_diff = sum(diffs) / len(diffs)
    std_diff = (sum((x - mean_diff) ** 2 for x in diffs) / (len(diffs) - 1)) ** 0.5
    expected = 0.0 if std_diff == 0 else mean_diff / std_diff

    aae(result, expected, 8)


def test_calc_prob_mom_expr():
    df = pl.DataFrame({
        "a": [0.02, 0.01, -0.01, 0.03],
        "b": [0.01, 0.00,  0.01, 0.01],
    })

    out = df.select(pl.col("a").ffn.calc_prob_mom(b="b"))
    val = out.item()
    assert 0 <= val <= 1

def test_calc_total_return_simple():
    df = pl.DataFrame(
        {
            "Date": ["2023-01-01", "2023-01-02", "2023-01-03"],
            "AAPL": [100.0, 110.0, 130.0],
        }
    )

    result = df.select(pl.col("AAPL").ffn.calc_total_return()).item()
    expected = (130.0 / 100.0) - 1
    aae(result, expected, 6)


def test_calc_total_return_flat():
    df = pl.DataFrame(
        {
            "Date": ["2023-01-01", "2023-01-02", "2023-01-03"],
            "AAPL": [100.0, 100.0, 100.0],
        }
    )

    result = df.select(pl.col("AAPL").ffn.calc_total_return()).item()
    assert result == 0.0


def test_calc_total_return_negative():
    df = pl.DataFrame(
        {
            "Date": ["2023-01-01", "2023-01-02", "2023-01-03"],
            "AAPL": [100.0, 90.0, 80.0],
        }
    )

    result = df.select(pl.col("AAPL").ffn.calc_total_return()).item()
    expected = (80.0 / 100.0) - 1
    aae(result, expected, 6)


def is_close(a: float, b: float, rtol: float = 1e-6) -> bool:
    return abs(a - b) <= rtol * max(abs(a), abs(b))


def test_annualize_basic():
    df = pl.DataFrame(
        {
            "returns": [0.05, 0.10, 0.25],
            "durations": [30, 90, 180],
        }
    )

    result = df.select(pl.col("returns").ffn.annualize(durations="durations"))

    expected = [
        (1 + 0.05) ** (365.0 / 30) - 1,
        (1 + 0.10) ** (365.0 / 90) - 1,
        (1 + 0.25) ** (365.0 / 180) - 1,
    ]

    for actual, exp in zip(result["returns_annualized"], expected):
        assert is_close(actual, exp)


def test_annualize_zero_return():
    df = pl.DataFrame(
        {
            "returns": [0.0],
            "durations": [60],
        }
    )
    result = df.select(pl.col("returns").ffn.annualize(durations="durations"))
    assert result["returns_annualized"][0] == 0.0


def test_annualize_one_day_duration():
    df = pl.DataFrame(
        {
            "returns": [0.01],
            "durations": [1],
        }
    )
    result = df.select(pl.col("returns").ffn.annualize(durations="durations"))
    expected = (1 + 0.01) ** 365.0 - 1
    assert is_close(result["returns_annualized"][0], expected)


def test_annualize_large_return_long_duration():
    df = pl.DataFrame(
        {
            "returns": [10.0],
            "durations": [730],
        }
    )
    result = df.select(pl.col("returns").ffn.annualize(durations="durations"))
    expected = (1 + 10.0) ** (365.0 / 730) - 1
    assert is_close(result["returns_annualized"][0], expected)


def test_sortino_ratio(df):
    rf = 0.0
    nperiods = 1

    df = df.with_columns(pl.col("AAPL").ffn.to_returns().alias("returns"))
    er = df.with_columns([(pl.col("returns") - rf / nperiods).alias("excess")])

    # replicate: negative_returns = np.minimum(er[1:], 0.0)
    excess = [x for x in er["excess"].to_list() if x is not None]
    negative_returns = [
        min(x, 0.0) for x in excess[1:]
    ]  # skip first just like original test
    downside_std = (
        sum(
            (x - sum(negative_returns) / len(negative_returns)) ** 2
            for x in negative_returns
        )
        / (len(negative_returns) - 1)
    ) ** 0.5
    expected = (sum(excess) / len(excess)) / downside_std * (nperiods**0.5)

    # actual from Polars expr
    result = df.select(pl.col("returns").ffn.sortino_ratio(rf=rf, n=nperiods))
    actual = result["returns_sortino_ratio"][0]

    aae(actual, expected, places=3)

def test_calc_calmar_ratio_expr():
    df = pl.DataFrame({
        "Date": pl.date_range(
            start=datetime(2020, 1, 1),
            end=datetime(2020, 4, 1),
            interval="1mo",
            eager=True
        ),
        "price": [100, 90, 120, 140]
    })

    result = df.select(pl.col("price").ffn.calc_calmar_ratio(date_col="Date")).item()

    cagr = df.select(pl.col("price").ffn.calc_cagr(date_col="Date")).item()
    # max_dd = abs(df.select(ffn.calc_max_drawdown("price"))["max_drawdown"].item())
    max_dd = abs(df.select(pl.col("price").ffn.calc_max_drawdown()).item())
    expected = cagr / max_dd

    aae(result, expected, 4)

def test_ulcer_index_known_example():
    # Prices with known drawdowns
    df = pl.DataFrame({
        "Date": ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"],
        "price": [100.0, 90.0, 95.0, 85.0, 80.0],
    })

    # Drawdowns from peak:
    # 0%, -10%, -5%, -15%, -20%
    # Ulcer Index = sqrt(mean([0^2, 10^2, 5^2, 15^2, 20^2])) = sqrt(250) ≈ 15.81
    result = df.select(pl.col("price").ffn.ulcer_index()).item()
    aae(result, np.sqrt((0**2 + 10**2 + 5**2 + 15**2 + 20**2) / 5), 4)


# @pytest.mark.parametrize("rf,nperiods,expected", [
#     (0.0, None, 0.0681),  # baseline, UPI = mean return / ulcer index
#     (0.02, 252, 0.0497),  # constant RF
# ])
# def test_ulcer_performance_index_float_rf(rf, nperiods, expected):
#     df = pl.DataFrame({
#         "price": [100, 95, 97, 85, 80, 82],
#     })
#
#     result = df.select(
#         ffn.ulcer_performance_index("price", rf=rf, nperiods=nperiods)
#     ).item()
#
#     aae(result, expected, 3)

# def test_ulcer_performance_index_float_rf():
#     df = pl.DataFrame({
#         "price": [100, 95, 97, 85, 80, 82],
#     })
#
#     # Compute returns manually
#     returns = df.select(
#         pl.col("price").pct_change()
#     ).to_series().drop_nulls()
#
#     # Compute ulcer index
#     cummax = df["price"].cum_max()
#     drawdowns = ((df["price"] - cummax) / cummax) * 100
#     ulcer = np.sqrt(np.mean(drawdowns.to_numpy()**2))
#
#     expected_upi = returns.mean() / ulcer
#
#     result = df.select(
#         ffn.ulcer_performance_index("price", rf=0.0, nperiods=252)
#     ).item()
#
#     aae(result, expected_upi, 4)


@pytest.mark.parametrize(
    "prices, rf, nperiods",
    [
        ([100, 95, 97, 85, 80, 82], 0.0, None),          # baseline: rf = 0
        ([100, 95, 97, 85, 80, 82], 0.03, 252),           # constant rf
        ([100, 105, 110, 115, 120, 125], 0.01, 252),      # rising prices, rf > 0
    ]
)
def test_ulcer_performance_index_float_rf(prices, rf, nperiods):
    df = pl.DataFrame({"price": prices})

    # Compute returns manually
    returns = df.select(
        pl.col("price").pct_change()
    ).to_series().drop_nulls()

    # Adjust for rf if needed
    if rf and nperiods:
        returns = returns - (rf / nperiods)

    # Compute ulcer index
    cummax = df["price"].cum_max()
    drawdowns = ((df["price"] - cummax) / cummax) * 100
    ulcer = np.sqrt(np.mean(drawdowns.to_numpy() ** 2))

    if ulcer < 1e-8:
        expected_upi = np.inf
    else:
        expected_upi = np.mean(returns.to_numpy()) / ulcer

    result = df.select(pl.col("price").ffn.ulcer_performance_index(rf=rf, n=nperiods)).item()

    if np.isnan(expected_upi):
        assert np.isinf(result)
    else:
        aae(result, expected_upi, 4)


def test_ulcer_performance_index_column_rf():
    df = pl.DataFrame({
        "price": [100, 95, 97, 85, 80, 82],
        "rf_col": [0.0001] * 6,  # daily constant RF series
    })

    result = df.select(pl.col("price").ffn.ulcer_performance_index(rf="rf_col")).item()

    manual_rf_adjusted_returns = df.select(
        (pl.col("price").pct_change() - pl.col("rf_col")).mean()
    ).item()

    ui = df.select(
        (((pl.col("price") - pl.col("price").cum_max()) / pl.col("price").cum_max()) * 100).pow(2).mean().sqrt()
    ).item()

    aae(result, manual_rf_adjusted_returns / ui, 4)

def test_invalid_rf_type_raises():
    df = pl.DataFrame({"price": [100, 95, 97, 85]})
    with pytest.raises(TypeError):
        df.select(pl.col("price").ffn.ulcer_performance_index(rf=[0.01]))

def test_missing_nperiods_raises():
    df = pl.DataFrame({"price": [100, 95, 97, 85]})
    with pytest.raises(ValueError):
        df.select(pl.col("price").ffn.ulcer_performance_index(rf=0.03, n=None))

@pytest.mark.parametrize(
    "freq, expected",
    [
        ("1d", TRADING_DAYS_PER_YEAR),
        ("1h", TRADING_DAYS_PER_YEAR * 24),
        ("1m", TRADING_DAYS_PER_YEAR * 24 * 60),
        ("1s", TRADING_DAYS_PER_YEAR * 24 * 60 * 60),
        ("100ms", TRADING_DAYS_PER_YEAR * 24 * 60 * 60 * 10),
    ]
)
def test_infer_nperiods_known_freqs(freq, expected):
    df = make_df(10, freq)
    result = df.select(pl.col("Date").ffn.infer_nperiods()).item()
    assert result == expected

def test_infer_nperiods_monthly_and_yearly():
    df_monthly = make_df(10, "1mo")
    df_yearly = make_df(10, "1y")

    assert df_monthly.select(pl.col("Date").ffn.infer_nperiods()).item() == 12
    assert df_yearly.select(pl.col("Date").ffn.infer_nperiods()).item() == 1

def test_infer_nperiods_irregular_returns_none():
    dt = datetime.strptime("2024-01-01 00:00:00", "%Y-%m-%d %H:%M:%S")
    # Irregular intervals: 1s, 2s, 3s, 5s...
    deltas = [1, 2, 3, 5, 8]
    timestamps = [dt]
    for delta in deltas:
        timestamps.append(timestamps[-1] + timedelta(seconds=delta))

    df = pl.DataFrame({
        "Date": timestamps,
        "value": np.random.randn(len(timestamps)),
    })

    result = df.select(pl.col("Date").ffn.infer_nperiods()).item()
    assert result is None
