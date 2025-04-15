import numpy as np
import polars as pl

from .common import EXCHANGE_IDS, EXCHANGE_WEIGHTS, assign_tape, get_trading_window_ns
from .market_events import inject_event_windows, is_within_events


def generate_quote_conditions(n: int) -> list[str]:
    return [
        "19" if np.random.rand() < 0.8 else str(np.random.choice(range(1, 20)))
        for _ in range(n)
    ]


def generate_quote_indicators(n: int) -> list[str]:
    return [
        "" if np.random.rand() < 0.9 else str(np.random.choice([4, 12, 13]))
        for _ in range(n)
    ]


def simulate_quotes(
    ticker: str,
    day_index: int = 0,
    ticks_per_day: int = 250_000,
    base_price: float = 276.0,
    avg_spread: float = 0.02,
    seed: int = 42,
) -> pl.DataFrame:
    np.random.seed(seed + day_index)

    # Timestamps for the day
    start_ns, end_ns = get_trading_window_ns(day_index)
    timestamps = np.linspace(start_ns, end_ns, ticks_per_day).astype(np.int64)

    # Inject market open/close and news event windows
    open_burst = [(start_ns, start_ns + int(30 * 60 * 1e9))]  # First 30 mins
    close_burst = [(end_ns - int(30 * 60 * 1e9), end_ns)]  # Last 30 mins
    news_bursts = inject_event_windows(
        (start_ns, end_ns),
        day_index=day_index,
        num_events=2,
        min_duration_ns=int(5 * 60 * 1e9),
        max_duration_ns=int(30 * 60 * 1e9),
        probability=0.3,
        seed=seed,
    )
    all_bursts = open_burst + close_burst + news_bursts
    in_burst = is_within_events(timestamps, all_bursts)

    # Price walk with volatility bursts
    volatility = np.where(in_burst, 0.03, 0.01)
    price = np.round(np.cumsum(np.random.normal(0, volatility)) + base_price, 2)

    # Spread = microstructure noise
    spread = np.abs(np.random.normal(avg_spread, avg_spread / 3, size=ticks_per_day))
    spread[in_burst] *= 1.5  # wider spreads in bursts
    ask_price = np.round(price + spread, 2)
    bid_price = np.round(price, 2)

    # Lot sizes (1 = 100 shares)
    ask_size = np.random.poisson(2, ticks_per_day)
    bid_size = np.random.poisson(2, ticks_per_day)
    ask_size[ask_size == 0] = 1
    bid_size[bid_size == 0] = 1

    # Exchange ID per side
    ask_ex = np.random.choice(EXCHANGE_IDS, size=ticks_per_day, p=EXCHANGE_WEIGHTS)
    bid_ex = np.random.choice(EXCHANGE_IDS, size=ticks_per_day, p=EXCHANGE_WEIGHTS)
    tape = [assign_tape(eid) for eid in ask_ex]

    return pl.DataFrame(
        {
            "ticker": [ticker] * ticks_per_day,
            "ask_exchange": ask_ex,
            "ask_price": ask_price,
            "ask_size": ask_size,
            "bid_exchange": bid_ex,
            "bid_price": bid_price,
            "bid_size": bid_size,
            "conditions": generate_quote_conditions(ticks_per_day),
            "indicators": generate_quote_indicators(ticks_per_day),
            "participant_timestamp": timestamps,
            "sequence_number": np.arange(1_000, 1_000 + ticks_per_day),
            "sip_timestamp": timestamps
            + np.random.randint(5_000, 50_000, size=ticks_per_day),
            "tape": tape,
            "trf_timestamp": timestamps
            + np.random.randint(10_000, 100_000, size=ticks_per_day),
        }
    )
