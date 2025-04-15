from typing import Literal, Protocol
import polars as pl

PluginType = Literal["quotes", "trades"]


class MicrostructurePlugin(Protocol):
    name: str
    type: PluginType

    def apply(self, df: pl.DataFrame, *, params: dict) -> pl.DataFrame: ...
