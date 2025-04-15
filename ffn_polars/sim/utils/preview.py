from typing import Callable, List

import polars as pl

from ffn_polars.sim.plugins.registry import PLUGIN_REGISTRY

# Type alias
InspectorFn = Callable[[pl.DataFrame, pl.DataFrame], None]


def preview_plugin_effect(
    df: pl.DataFrame,
    plugin_name: str,
    *,
    params: dict = None,
    inspectors: List[InspectorFn] = None,
) -> None:
    params = params or {}
    inspectors = inspectors or [summary_diff, numeric_change, column_changes]

    plugin = PLUGIN_REGISTRY.get(plugin_name)
    if plugin is None:
        print(f"❌ Plugin '{plugin_name}' not found.")
        return

    df_before = df.clone()
    df_after = plugin.apply(df.clone(), params=params)

    print(f"\n🧩 Plugin: {plugin_name}")
    print(f"📄 Type: {plugin.type}")
    print(f"📘 Description: {plugin.__doc__ or 'No docstring'}")
    print(f"🔧 Params: {params}")

    for inspect in inspectors:
        print("\n" + "-" * 40)
        inspect(df_before, df_after)


def summary_diff(df_before: pl.DataFrame, df_after: pl.DataFrame) -> None:
    print("📊 Shape Change:")
    print(f"Rows: {df_before.height} → {df_after.height}")
    print(f"Cols: {df_before.width} → {df_after.width}")


def numeric_change(df_before: pl.DataFrame, df_after: pl.DataFrame) -> None:
    print("📈 Numeric Column Change:")
    for col in df_before.columns:
        if df_before[col].dtype in (pl.Float64, pl.Int64):
            try:
                before_mean = df_before[col].mean()
                after_mean = df_after[col].mean()
                change = after_mean - before_mean
                pct = (change / before_mean) * 100 if before_mean else 0
                print(f"{col:<20} Δ {change:.5f} ({pct:+.2f}%)")
            except:
                continue


def column_changes(df_before: pl.DataFrame, df_after: pl.DataFrame) -> None:
    print("🔍 Column Differences:")
    added = set(df_after.columns) - set(df_before.columns)
    removed = set(df_before.columns) - set(df_after.columns)
    common = set(df_after.columns) & set(df_before.columns)
    type_changed = [
        col for col in common if df_before[col].dtype != df_after[col].dtype
    ]

    if added:
        print(f"🆕 Added: {added}")
    if removed:
        print(f"❌ Removed: {removed}")
    if type_changed:
        print(f"🔁 Type Changed: {type_changed}")
