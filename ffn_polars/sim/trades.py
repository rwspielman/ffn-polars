import numpy as np
import polars as pl

from .common import assign_tape, get_trading_window_ns
from .market_events import inject_event_windows, is_within_events


def simulate_trades(
    ticker: str,
    quotes: pl.DataFrame,
    day_index: int = 0,
    target_volume: int = 10_000_000,
    avg_trade_size: int = 100,
    seed: int = 42,
) -> pl.DataFrame:
    np.random.seed(seed + day_index)

    # Extract timestamps and price bands from quotes
    quotes = quotes.sort("participant_timestamp")
    quote_ts = quotes["participant_timestamp"].to_numpy()
    ask_px = quotes["ask_price"].to_numpy()
    bid_px = quotes["bid_price"].to_numpy()
    ask_size = quotes["ask_size"].to_numpy()
    bid_size = quotes["bid_size"].to_numpy()
    ask_ex = quotes["ask_exchange"].to_numpy()
    bid_ex = quotes["bid_exchange"].to_numpy()

    # Determine how many trades to simulate
    n_trades = int(target_volume / avg_trade_size)

    # Sample trade times within quote time bounds
    trade_ts = np.sort(np.random.choice(quote_ts, size=n_trades, replace=True))

    # Inject regime volatility into trades
    start_ns, end_ns = get_trading_window_ns(day_index)
    open_burst = [(start_ns, start_ns + int(30 * 60 * 1e9))]
    close_burst = [(end_ns - int(30 * 60 * 1e9), end_ns)]
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
    in_burst = is_within_events(trade_ts, all_bursts)

    # Random trade direction: 1 = buy (hit ask), -1 = sell (hit bid)
    direction = np.random.choice([-1, 1], size=n_trades)
    price = np.interp(trade_ts, quote_ts, ask_px) * (direction == 1) + np.interp(
        trade_ts, quote_ts, bid_px
    ) * (direction == -1)
    price = np.round(price, 2)

    # Volume is scaled by quote size
    size = np.where(
        direction == 1,
        np.interp(trade_ts, quote_ts, ask_size),
        np.interp(trade_ts, quote_ts, bid_size),
    )
    size *= np.where(in_burst, 2.0, 1.0)
    size = np.clip(size, 1, 10)  # in lots
    size = (size * 100).astype(int)

    # Trade ID = sequential
    ids = np.arange(1_000_000, 1_000_000 + n_trades)
    sequence_number = np.arange(2_000_000, 2_000_000 + n_trades)

    # Assign exchanges based on direction
    exchange = np.where(
        direction == 1,
        np.interp(trade_ts, quote_ts, ask_ex),
        np.interp(trade_ts, quote_ts, bid_ex),
    ).astype(int)

    tape = [assign_tape(eid) for eid in exchange]

    return pl.DataFrame(
        {
            "ticker": [ticker] * n_trades,
            "conditions": ["0"] * n_trades,  # all regular for now
            "correction": [None] * n_trades,
            "exchange": exchange,
            "id": ids,
            "participant_timestamp": trade_ts,
            "price": price,
            "sequence_number": sequence_number,
            "sip_timestamp": trade_ts
            + np.random.randint(5_000, 200_000, size=n_trades),
            "size": size,
            "tape": tape,
            "trf_id": np.random.choice([0, 1, 2], size=n_trades),
            "trf_timestamp": trade_ts
            + np.random.randint(10_000, 400_000, size=n_trades),
        }
    )
