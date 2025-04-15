from typing import List, Tuple

import numpy as np


def inject_event_windows(
    trading_ns: Tuple[int, int],
    day_index: int,
    num_events: int,
    min_duration_ns: int,
    max_duration_ns: int,
    probability: float = 0.5,
    seed: int = 42,
) -> List[Tuple[int, int]]:
    np.random.seed(seed + day_index)
    start_ns, end_ns = trading_ns
    duration_range = max_duration_ns - min_duration_ns
    total_events = (
        0 if np.random.rand() > probability else np.random.randint(1, num_events + 1)
    )

    event_windows = []
    for _ in range(total_events):
        duration = min_duration_ns + np.random.randint(duration_range)
        start_time = start_ns + np.random.randint(0, (end_ns - start_ns - duration))
        event_windows.append((start_time, start_time + duration))
    return event_windows


def is_within_events(ts: np.ndarray, events: List[Tuple[int, int]]) -> np.ndarray:
    mask = np.zeros_like(ts, dtype=bool)
    for start, end in events:
        mask |= (ts >= start) & (ts < end)
    return mask
