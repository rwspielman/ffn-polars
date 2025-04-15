import numpy as np

EXCHANGES = [
    {"id": 12, "name": "Nasdaq"},
    {"id": 10, "name": "NYSE"},
    {"id": 11, "name": "NYSE Arca"},
    {"id": 8, "name": "Cboe EDGX"},
    {"id": 2, "name": "Nasdaq BX"},
    {"id": 1, "name": "AMEX"},
    {"id": 201, "name": "FINRA NYSE TRF"},
    {"id": 202, "name": "FINRA Nasdaq TRF Carteret"},
]

EXCHANGE_IDS = [e["id"] for e in EXCHANGES]
EXCHANGE_WEIGHTS = [0.25, 0.2, 0.15, 0.15, 0.1, 0.05, 0.05, 0.05]


def assign_tape(exchange_id: int) -> int:
    if exchange_id in [10, 11, 1]:
        return 1
    elif exchange_id in [12, 2]:
        return 3
    elif exchange_id in [201, 202]:
        return np.random.choice([1, 2, 3])
    else:
        return 2


def get_trading_window_ns(day_offset: int = 0) -> tuple[int, int]:
    from datetime import datetime, timedelta

    import pytz

    eastern = pytz.timezone("US/Eastern")
    base_date = datetime.strptime("2024-04-12", "%Y-%m-%d") + timedelta(days=day_offset)
    start_dt = eastern.localize(
        datetime.combine(base_date, datetime.strptime("09:30", "%H:%M").time())
    )
    end_dt = eastern.localize(
        datetime.combine(base_date, datetime.strptime("16:00", "%H:%M").time())
    )
    return int(start_dt.timestamp() * 1e9), int(end_dt.timestamp() * 1e9)
