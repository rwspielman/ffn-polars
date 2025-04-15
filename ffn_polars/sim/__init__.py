import polars as pl

from .quotes import simulate_quotes
from .trades import simulate_trades


from .plugins.registry import PLUGIN_REGISTRY


def apply_plugins(
    df: pl.DataFrame, plugins: list[dict], plugin_type: str
) -> pl.DataFrame:
    for p in plugins:
        plugin = PLUGIN_REGISTRY.get(p["name"])
        if plugin and plugin.type == plugin_type:
            df = plugin.apply(df, params=p.get("params", {}))
    return df


def simulate_market(
    ticker: str,
    days: int = 1,
    quote_ticks_per_day: int = 500_000,
    trade_volume_per_day: int = 10_000_000,
    seed: int = 42,
    plugins: list[dict] = None,
):
    plugins = plugins or []

    quotes_all = []
    trades_all = []

    for day in range(days):
        quotes = simulate_quotes(
            ticker=ticker, day_index=day, ticks_per_day=quote_ticks_per_day, seed=seed
        )
        quotes = apply_plugins(quotes, plugins, "quotes")

        trades = simulate_trades(
            ticker=ticker,
            quotes=quotes,
            day_index=day,
            target_volume=trade_volume_per_day,
            seed=seed,
        )
        trades = apply_plugins(trades, plugins, "trades")

        quotes_all.append(quotes)
        trades_all.append(trades)

    return (
        pl.concat(quotes_all).sort("participant_timestamp"),
        pl.concat(trades_all).sort("participant_timestamp"),
    )
