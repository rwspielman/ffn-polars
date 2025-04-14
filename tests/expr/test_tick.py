import pytest
import polars as pl
from datetime import datetime, date, timedelta
from tests.utils import aae
from math import log, sqrt
import statistics


@pytest.mark.parametrize(
    "per, expected",
    [
        ("s", 3 / 10),
        ("ms", 3 / 10_000),
        ("us", 3 / 10_000_000),
        ("ns", 3 / 10_000_000_000),
    ],
)
def test_calc_trade_rate_select(per, expected):
    df = pl.DataFrame(
        {
            "timestamp": [
                datetime(2023, 1, 1, 9, 30, 0),
                datetime(2023, 1, 1, 9, 30, 5),
                datetime(2023, 1, 1, 9, 30, 10),
            ]
        }
    )
    result = df.select(pl.col("timestamp").ffn.calc_trade_rate(per=per).alias("rate"))
    aae(result.item(), expected, 9)


@pytest.mark.parametrize(
    "per, expected",
    [
        ("s", {"AAPL": 3 / 10, "MSFT": 2 / 10}),
        ("ms", {"AAPL": 3 / 10_000, "MSFT": 2 / 10_000}),
        ("us", {"AAPL": 3 / 10_000_000, "MSFT": 2 / 10_000_000}),
        ("ns", {"AAPL": 3 / 10_000_000_000, "MSFT": 2 / 10_000_000_000}),
    ],
)
def test_calc_trade_rate_group_by(per, expected):
    df = pl.DataFrame(
        {
            "timestamp": [
                datetime(2023, 1, 1, 9, 30, 0),
                datetime(2023, 1, 1, 9, 30, 5),
                datetime(2023, 1, 1, 9, 30, 10),
                datetime(2023, 1, 1, 9, 30, 0),
                datetime(2023, 1, 1, 9, 30, 10),
            ],
            "ticker": ["AAPL", "AAPL", "AAPL", "MSFT", "MSFT"],
        }
    )
    result = (
        df.group_by("ticker")
        .agg(pl.col("timestamp").ffn.calc_trade_rate(per=per).alias("rate"))
        .sort("ticker")
    )
    for row in result.iter_rows(named=True):
        aae(row["rate"], expected[row["ticker"]], 9)


@pytest.mark.parametrize(
    "per, expected",
    [
        ("s", 5.0),
        ("ms", 5_000.0),
        ("us", 5_000_000.0),
        ("ns", 5_000_000_000.0),
    ],
)
def test_calc_inter_trade_time_select(per, expected):
    df = pl.DataFrame(
        {
            "timestamp": [
                datetime(2023, 1, 1, 9, 30, 0),
                datetime(2023, 1, 1, 9, 30, 5),
                datetime(2023, 1, 1, 9, 30, 10),
            ]
        }
    )

    result = df.select(
        pl.col("timestamp").ffn.calc_inter_trade_time(per=per).alias("itt")
    )
    aae(result.item(), expected, 9)


@pytest.mark.parametrize(
    "per, expected",
    [
        ("s", {"AAPL": 5.0, "MSFT": 10.0}),
        ("ms", {"AAPL": 5_000.0, "MSFT": 10_000.0}),
        ("us", {"AAPL": 5_000_000.0, "MSFT": 10_000_000.0}),
        ("ns", {"AAPL": 5_000_000_000.0, "MSFT": 10_000_000_000.0}),
    ],
)
def test_calc_inter_trade_time_group_by(per, expected):
    df = pl.DataFrame(
        {
            "timestamp": [
                datetime(2023, 1, 1, 9, 30, 0),
                datetime(2023, 1, 1, 9, 30, 5),
                datetime(2023, 1, 1, 9, 30, 10),
                datetime(2023, 1, 1, 9, 30, 0),
                datetime(2023, 1, 1, 9, 30, 10),
            ],
            "ticker": ["AAPL", "AAPL", "AAPL", "MSFT", "MSFT"],
        }
    )

    result = (
        df.group_by("ticker")
        .agg(pl.col("timestamp").ffn.calc_inter_trade_time(per=per).alias("itt"))
        .sort("ticker")
    )

    for row in result.iter_rows(named=True):
        aae(row["itt"], expected[row["ticker"]], 9)


@pytest.mark.parametrize(
    "per, expected",
    [
        ("s", 600 / 10),
        ("ms", 600 / 10_000),
        ("us", 600 / 10_000_000),
        ("ns", 600 / 10_000_000_000),
    ],
)
def test_calc_volume_rate_select(per, expected):
    df = pl.DataFrame(
        {
            "timestamp": [
                datetime(2023, 1, 1, 9, 30, 0),
                datetime(2023, 1, 1, 9, 30, 5),
                datetime(2023, 1, 1, 9, 30, 10),
            ],
            "volume": [100, 200, 300],
        }
    )

    result = df.select(
        pl.col("volume").ffn.calc_volume_rate(ts="timestamp", per=per).alias("vrate")
    )

    aae(result.item(), expected, 9)


@pytest.mark.parametrize(
    "per, expected",
    [
        ("s", {"AAPL": 600 / 10, "MSFT": 400 / 10}),
        ("ms", {"AAPL": 600 / 10_000, "MSFT": 400 / 10_000}),
        ("us", {"AAPL": 600 / 10_000_000, "MSFT": 400 / 10_000_000}),
        ("ns", {"AAPL": 600 / 10_000_000_000, "MSFT": 400 / 10_000_000_000}),
    ],
)
def test_calc_volume_rate_group_by(per, expected):
    df = pl.DataFrame(
        {
            "timestamp": [
                datetime(2023, 1, 1, 9, 30, 0),
                datetime(2023, 1, 1, 9, 30, 5),
                datetime(2023, 1, 1, 9, 30, 10),
                datetime(2023, 1, 1, 9, 30, 0),
                datetime(2023, 1, 1, 9, 30, 10),
            ],
            "volume": [100, 200, 300, 200, 200],
            "ticker": ["AAPL", "AAPL", "AAPL", "MSFT", "MSFT"],
        }
    )

    result = (
        df.group_by("ticker")
        .agg(
            pl.col("volume")
            .ffn.calc_volume_rate(ts="timestamp", per=per)
            .alias("vrate")
        )
        .sort("ticker")
    )

    for row in result.iter_rows(named=True):
        aae(row["vrate"], expected[row["ticker"]], 9)


def test_calc_realized_volatility_select():
    df = pl.DataFrame({"price": [100.0, 101.0, 98.0, 99.0]})

    # Manual log return vol:
    log_returns = [log(101 / 100), log(98 / 101), log(99 / 98)]
    expected = sqrt(sum(r**2 for r in log_returns))

    result = df.select(pl.col("price").ffn.calc_realized_volatility())

    aae(result.item(), expected, 9)


def test_calc_realized_volatility_grouped_by_ticker():
    df = pl.DataFrame(
        {
            "price": [100.0, 101.0, 98.0, 200.0, 210.0],
            "ticker": ["AAPL", "AAPL", "AAPL", "MSFT", "MSFT"],
        }
    )

    expected = {
        "AAPL": sqrt(sum(r**2 for r in [log(101 / 100), log(98 / 101)])),
        "MSFT": sqrt(sum(r**2 for r in [log(210 / 200)])),
    }

    result = (
        df.group_by("ticker")
        .agg(pl.col("price").ffn.calc_realized_volatility())
        .sort("ticker")
    )

    for row in result.iter_rows(named=True):
        aae(row["price_realized_volatility"], expected[row["ticker"]], 9)


def test_tick_rule_direction():
    df = pl.DataFrame(
        {
            "price": [100.0, 101.0, 101.0, 100.0, 102.0],
        }
    )

    result = df.select(pl.col("price").ffn.tick_rule().alias("direction"))[
        "direction"
    ].to_list()

    expected = [0, 1, 0, -1, 1]
    assert result == expected


def test_apply_tick_rule_to_volume():
    df = pl.DataFrame(
        {
            "timestamp": pl.date_range(
                date(2023, 1, 1), date(2023, 1, 5), interval="1d", eager=True
            ),
            "price": [100.0, 101.0, 101.0, 100.0, 102.0],
            "volume": [100, 200, 150, 300, 400],
        }
    )

    result = df.select(
        pl.col("volume")
        .ffn.apply_tick_rule_to_volume(price="price")
        .alias("signed_volume")
    )["signed_volume"].to_list()

    expected = [0.0, 200.0, 0.0, -300.0, 400.0]
    assert result == expected


def test_calc_tick_imbalance_signed_volume():
    df = pl.DataFrame({"signed_volume": [1, 1, -1, -1]})  # Net = 0

    result = df.select(
        pl.col("signed_volume").ffn.calc_tick_imbalance().alias("imbalance")
    )

    expected = 0.0
    aae(result.item(), expected, 9)


def test_calc_vwap_select():
    df = pl.DataFrame(
        {
            "price": [100.0, 101.0, 102.0],
            "volume": [200, 300, 500],
        }
    )

    weighted_sum = 100 * 200 + 101 * 300 + 102 * 500
    expected = weighted_sum / (200 + 300 + 500)

    result = df.select(pl.col("price").ffn.calc_vwap(volume="volume"))

    aae(result.item(), expected, 9)


def test_calc_vwap_group_by_ticker():
    df = pl.DataFrame(
        {
            "price": [100.0, 101.0, 102.0, 200.0, 201.0],
            "volume": [200, 300, 500, 100, 100],
            "ticker": ["AAPL", "AAPL", "AAPL", "MSFT", "MSFT"],
        }
    )

    aapl_vwap = (100 * 200 + 101 * 300 + 102 * 500) / (200 + 300 + 500)
    msft_vwap = (200 * 100 + 201 * 100) / (100 + 100)

    expected = {
        "AAPL": aapl_vwap,
        "MSFT": msft_vwap,
    }

    result = (
        df.group_by("ticker")
        .agg(pl.col("price").ffn.calc_vwap(volume="volume"))
        .sort("ticker")
    )

    for row in result.iter_rows(named=True):
        aae(row["price_vwap"], expected[row["ticker"]], 9)


@pytest.mark.parametrize(
    "per, expected",
    [
        ("s", 5 / 5),  # ITTs: [5, 5, 5] → std = 0, mean = 5 → burst = 0
        ("ms", 5_000 / 5_000),
        ("us", 5_000_000 / 5_000_000),
        ("ns", 5_000_000_000 / 5_000_000_000),
    ],
)
def test_calc_burstiness_regular(per, expected):
    base = datetime(2023, 1, 1, 9, 30, 0)
    df = pl.DataFrame(
        {
            "timestamp": [
                base,
                base + timedelta(seconds=5),
                base + timedelta(seconds=10),
                base + timedelta(seconds=15),
            ]
        }
    )

    result = df.select(pl.col("timestamp").ffn.calc_burstiness(per=per))

    aae(result.item(), 0.0, 9)


def test_calc_burstiness_grouped():
    df = pl.DataFrame(
        {
            "timestamp": [
                datetime(2023, 1, 1, 9, 30, 0),
                datetime(2023, 1, 1, 9, 30, 1),
                datetime(2023, 1, 1, 9, 30, 3),
                datetime(2023, 1, 1, 9, 30, 0),
                datetime(2023, 1, 1, 9, 30, 5),
            ],
            "ticker": ["AAPL", "AAPL", "AAPL", "MSFT", "MSFT"],
        }
    )

    # AAPL ITTs: [1, 2] → mean = 1.5, std = ~0.7071
    # MSFT ITT: [5] → std = 0, mean = 5 → burst = 0
    result = (
        df.group_by("ticker")
        .agg(pl.col("timestamp").ffn.calc_burstiness(per="s"))
        .sort("ticker")
    )

    expected = {
        "AAPL": 0.70710678 / 1.5,
        "MSFT": None,
    }

    for row in result.iter_rows(named=True):
        actual = row["timestamp_burstiness"]
        expected_value = expected[row["ticker"]]
        if expected_value is None:
            assert actual is None, f"Expected None for {row['ticker']}, got {actual}"
        else:
            assert (
                actual is not None
            ), f"Expected {expected_value} for {row['ticker']}, got None"
            aae(actual, expected_value, 7)


def test_calc_burstiness_nonzero():
    df = pl.DataFrame(
        {
            "timestamp": [
                datetime(2023, 1, 1, 9, 30, 0),
                datetime(2023, 1, 1, 9, 30, 3),
                datetime(2023, 1, 1, 9, 30, 7),
                datetime(2023, 1, 1, 9, 30, 8),
            ]
        }
    )

    # ITTs: [3, 4, 1]
    itts = [3, 4, 1]
    mean_itt = sum(itts) / len(itts)
    std_itt = statistics.stdev(itts)
    expected = std_itt / mean_itt

    result = df.select(pl.col("timestamp").ffn.calc_burstiness(per="s"))

    aae(result.item(), expected, 9)


def test_calc_price_impact_select():
    df = pl.DataFrame({"price": [100.0, 101.0, 102.0], "volume": [200, 300, 500]})

    # ΔP = 102 - 100 = 2
    # sum(volume) = 1000
    expected = 2 / 1000

    result = df.select(pl.col("price").ffn.calc_price_impact(volume="volume"))

    aae(result.item(), expected, 9)


def test_calc_price_impact_group_by():
    df = pl.DataFrame(
        {
            "price": [100.0, 101.0, 102.0, 200.0, 198.0],
            "volume": [200, 300, 500, 100, 100],
            "ticker": ["AAPL", "AAPL", "AAPL", "MSFT", "MSFT"],
        }
    )

    expected = {
        "AAPL": (102.0 - 100.0) / (200 + 300 + 500),
        "MSFT": (198.0 - 200.0) / (100 + 100),
    }

    result = (
        df.group_by("ticker")
        .agg(
            pl.col("price").ffn.calc_price_impact(volume="volume").alias("price_impact")
        )
        .sort("ticker")
    )

    for row in result.iter_rows(named=True):
        aae(row["price_impact"], expected[row["ticker"]], 9)


def test_order_flow_imbalance_select():
    df = pl.DataFrame({"signed_volume": [100.0, -50.0, 200.0, -100.0]})

    expected = sum([100.0, -50.0, 200.0, -100.0])  # = 150.0

    result = df.select(pl.col("signed_volume").ffn.calc_order_flow_imbalance())

    aae(result.item(), expected, 9)


def test_order_flow_imbalance_grouped():
    df = pl.DataFrame(
        {
            "signed_volume": [100.0, -50.0, 200.0, -100.0, -150.0],
            "ticker": ["AAPL", "AAPL", "AAPL", "MSFT", "MSFT"],
        }
    )

    expected = {
        "AAPL": 100.0 - 50.0 + 200.0,  # = 250
        "MSFT": -100.0 + -150.0,  # = -250
    }

    result = (
        df.group_by("ticker")
        .agg(pl.col("signed_volume").ffn.calc_order_flow_imbalance())
        .sort("ticker")
    )

    for row in result.iter_rows(named=True):
        aae(row["signed_volume_order_flow_imbalance"], expected[row["ticker"]], 9)


def test_calc_traded_value_select():
    df = pl.DataFrame({"price": [100.0, 101.0, 102.0], "volume": [10, 20, 30]})

    expected = 100 * 10 + 101 * 20 + 102 * 30

    result = df.select(pl.col("price").ffn.calc_traded_value(volume="volume"))

    aae(result.item(), expected, 9)


def test_calc_traded_value_group_by():
    df = pl.DataFrame(
        {
            "price": [100.0, 101.0, 102.0, 200.0, 205.0],
            "volume": [10, 20, 30, 50, 50],
            "ticker": ["AAPL", "AAPL", "AAPL", "MSFT", "MSFT"],
        }
    )

    expected = {"AAPL": 100 * 10 + 101 * 20 + 102 * 30, "MSFT": 200 * 50 + 205 * 50}

    result = (
        df.group_by("ticker")
        .agg(
            pl.col("price").ffn.calc_traded_value(volume="volume").alias("traded_value")
        )
        .sort("ticker")
    )

    for row in result.iter_rows(named=True):
        aae(row["traded_value"], expected[row["ticker"]], 9)


def test_calc_micro_returns_select():
    df = pl.DataFrame({"price": [100.0, 105.0, 110.0]})

    result = df.select(pl.col("price").ffn.calc_micro_returns())[
        "price_micro_returns"
    ].to_list()

    expected = [None, log(105 / 100), log(110 / 105)]

    for r, e in zip(result, expected):
        if r is None:
            assert e is None
        else:
            aae(r, e, 9)


def test_calc_price_volatility_ratio_select():
    prices = [100.0, 102.0, 101.0, 103.0]
    df = pl.DataFrame({"price": prices})

    expected = statistics.stdev(prices) / statistics.mean(prices)

    result = df.select(pl.col("price").ffn.calc_price_volatility_ratio())

    aae(result.item(), expected, 9)
