"""Reusable helpers for the Chapter 2 assignment notebook.

The public WIKI Prices data set is intentionally not committed because it is
roughly 1.8 GB as CSV. Set ``WIKI_DATA_PATH`` to the extracted CSV or Parquet
file before running the notebook.
"""

from __future__ import annotations

import gc
import os
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import polars as pl


class SvgFigure:
    """Small HTML wrapper that renders an SVG in Jupyter and on GitHub."""

    def __init__(self, svg: str, label: str) -> None:
        self.svg = svg
        self.label = label

    def _repr_html_(self) -> str:
        return self.svg

    def __repr__(self) -> str:
        return self.label


def locate_wiki_data() -> Path:
    """Locate the WIKI Prices CSV or Parquet file."""
    candidates = []
    if os.environ.get("WIKI_DATA_PATH"):
        candidates.append(Path(os.environ["WIKI_DATA_PATH"]).expanduser())
    candidates.extend(
        [
            Path("data/WIKI_PRICES.parquet"),
            Path("data/WIKI_PRICES.csv"),
            Path("WIKI_PRICES.parquet"),
            Path("WIKI_PRICES.csv"),
        ]
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        "WIKI Prices was not found. Set WIKI_DATA_PATH to WIKI_PRICES.csv or "
        "WIKI_PRICES.parquet before running this notebook."
    )


def load_wiki_prices() -> pl.DataFrame:
    """Load the rows needed by the three data-backed exercises.

    Full AAPL and MSFT histories support Exercises 1 and 2. Exercise 4 starts
    in 2014, so older rows for every other symbol are excluded to keep the
    executed notebook practical on an ordinary laptop.
    """
    path = locate_wiki_data()
    if path.suffix.lower() == ".parquet":
        frame = pl.scan_parquet(path)
    else:
        frame = pl.scan_csv(path, try_parse_dates=True, infer_schema_length=10_000)

    rename = {}
    if "ticker" in frame.columns:
        rename["ticker"] = "symbol"
    if "date" in frame.columns:
        rename["date"] = "timestamp"
    if "ex-dividend" in frame.columns:
        rename["ex-dividend"] = "ex_dividend"
    frame = frame.rename(rename)
    if frame.schema["timestamp"] == pl.String:
        frame = frame.with_columns(pl.col("timestamp").str.to_date())
    frame = frame.filter(
        pl.col("symbol").is_in(["AAPL", "MSFT"])
        | (pl.col("timestamp") >= date(2014, 1, 1))
    )
    return frame.collect().sort(["symbol", "timestamp"])


def check_ohlc_invariants(
    df: pl.DataFrame,
    open_col: str = "open",
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
    volume_col: str = "volume",
    rtol: float = 4 * np.finfo(np.float64).eps,
) -> pl.DataFrame:
    """Return pass rates for the six standard OHLCV invariants."""
    results: list[dict[str, Any]] = []

    def at_least(greater: str, lesser: str) -> pl.Expr:
        slack = rtol * pl.max_horizontal(pl.col(greater).abs(), pl.col(lesser).abs())
        return pl.col(greater) >= pl.col(lesser) - slack

    def add(name: str, condition: pl.Expr, columns: list[str]) -> None:
        applicable = df.drop_nulls(columns)
        if applicable.height:
            results.append(
                {
                    "check": name,
                    "valid_pct": applicable.select(condition.mean()).item() * 100,
                    "applicable_rows": applicable.height,
                    "total_rows": df.height,
                }
            )

    add("high_gte_low", at_least(high_col, low_col), [high_col, low_col])
    add("high_gte_open", at_least(high_col, open_col), [high_col, open_col])
    add("high_gte_close", at_least(high_col, close_col), [high_col, close_col])
    add("low_lte_open", at_least(open_col, low_col), [low_col, open_col])
    add("low_lte_close", at_least(close_col, low_col), [low_col, close_col])
    add("volume_non_negative", pl.col(volume_col) >= 0, [volume_col])
    return pl.DataFrame(results)


SCENARIOS = {
    "Empirical 2010-2020": {
        "cause_share": 0.257,
        "acquisition_share": 0.655,
        "other_share": 0.088,
        "cause_return": -0.60,
        "acquisition_return": 0.25,
        "other_return": 0.0,
    }
}
OUTCOMES = ("cause", "acquisition", "other")


def expected_terminal(params: dict[str, float]) -> float:
    """Calculate the share-weighted terminal return for one scenario."""
    return sum(params[f"{name}_share"] * params[f"{name}_return"] for name in OUTCOMES)


def prepare_survivorship_data(wiki: pl.DataFrame) -> dict[str, Any]:
    """Build the repaired return panel and exit labels used in Exercise 4."""
    window_start = date.fromisoformat("2014-01-01")
    window_end = wiki["timestamp"].max()
    if hasattr(window_end, "date"):
        window_end = window_end.date()
    first_month = window_start + timedelta(days=30)

    panel = wiki.filter(
        (pl.col("timestamp") >= window_start) & (pl.col("timestamp") <= window_end)
    )
    universe = (
        panel.group_by("symbol")
        .agg(pl.col("timestamp").min().alias("first_date"))
        .filter(pl.col("first_date") <= first_month)["symbol"]
        .to_list()
    )
    panel = panel.filter(pl.col("symbol").is_in(universe))
    returns = (
        panel.sort(["symbol", "timestamp"])
        .with_columns(
            (pl.col("adj_close") / pl.col("adj_close").shift(1).over("symbol") - 1).alias("ret"),
            pl.col("volume").shift(1).over("symbol").alias("prev_volume"),
        )
        .drop_nulls("ret")
        .with_columns(
            ((pl.col("ret") > 1.0) | (pl.col("prev_volume") == 0)).alias("suspect")
        )
    )

    coverage = returns.group_by("timestamp").agg(pl.len().alias("quoted"))
    real_sessions = coverage.filter(pl.col("quoted") >= 0.2 * len(universe))["timestamp"]
    returns = returns.filter(pl.col("timestamp").is_in(real_sessions))

    lifespans = panel.group_by("symbol").agg(pl.col("timestamp").max().alias("last_date"))
    exit_threshold = window_end - timedelta(days=15)
    leavers = lifespans.filter(pl.col("last_date") < exit_threshold)
    survivors = lifespans.filter(pl.col("last_date") >= exit_threshold)
    return {
        "window_start": window_start,
        "window_end": window_end,
        "first_month": first_month,
        "panel": panel,
        "universe": universe,
        "returns": returns,
        "window_lifespans": lifespans,
        "leavers": leavers,
        "survivors": survivors,
    }


def build_panel_matrices(
    returns: pl.DataFrame,
    window_lifespans: pl.DataFrame,
    survivors: pl.DataFrame,
    drop_suspect: bool = True,
) -> dict[str, Any]:
    """Precompute portfolio returns and the exit schedule."""
    frame = returns
    if drop_suspect:
        frame = frame.with_columns(
            pl.when(pl.col("suspect")).then(None).otherwise(pl.col("ret")).alias("ret")
        )
    symbols = np.array(sorted(frame["symbol"].unique().to_list()))
    sessions = np.array(sorted(frame["timestamp"].unique().to_list()), dtype="datetime64[D]")
    row_sessions = np.array(frame["timestamp"].to_list(), dtype="datetime64[D]")
    row_symbols = np.array(frame["symbol"].to_list())
    session_index = np.searchsorted(sessions, row_sessions)
    symbol_index = np.searchsorted(symbols, row_symbols)
    ret = np.full((len(sessions), len(symbols)), np.nan, dtype=np.float64)
    ret[session_index, symbol_index] = np.array(frame["ret"].to_list(), dtype=np.float64)
    last = dict(zip(window_lifespans["symbol"], window_lifespans["last_date"], strict=True))
    last_session = np.array([np.datetime64(last[symbol]) for symbol in symbols])
    is_survivor = np.isin(symbols, np.array(survivors["symbol"].to_list()))
    alive = sessions[:, None] <= last_session[None, :]
    held = np.vstack([np.ones((1, len(symbols)), dtype=bool), alive[:-1]])
    exits = (~alive) & held & ~is_survivor[None, :]
    contribution = np.nan_to_num(ret) * held
    n_held_universe = np.maximum(held.sum(axis=1), 1)
    n_held_survivors = np.maximum(held[:, is_survivor].sum(axis=1), 1)
    exit_session, exit_symbol = np.nonzero(exits)
    return {
        "symbols": symbols,
        "sessions": sessions,
        "is_survivor": is_survivor,
        "n_held_universe": n_held_universe,
        "base_universe": contribution.sum(axis=1) / n_held_universe,
        "base_survivors": contribution[:, is_survivor].sum(axis=1) / n_held_survivors,
        "exit_session": exit_session,
        "exit_symbol": exit_symbol,
    }


def universe_path(matrices: dict[str, Any], terminal: float | np.ndarray) -> np.ndarray:
    """Return a $100-based path for the full starting universe."""
    terminal_by_exit = (
        np.full(matrices["exit_symbol"].size, terminal)
        if np.isscalar(terminal)
        else np.asarray(terminal)[matrices["exit_symbol"]]
    )
    daily = matrices["base_universe"].copy()
    np.add.at(
        daily,
        matrices["exit_session"],
        terminal_by_exit / matrices["n_held_universe"][matrices["exit_session"]],
    )
    return 100.0 * np.cumprod(1.0 + daily)


def survivor_path(matrices: dict[str, Any]) -> np.ndarray:
    """Return a $100-based path for the survivors-only portfolio."""
    return 100.0 * np.cumprod(1.0 + matrices["base_survivors"])


def survivorship_chart(paths: dict[str, dict[str, Any]], empirical_terminal: float) -> SvgFigure:
    """Create a dependency-free two-panel SVG of the cumulative paths."""
    width, height = 1000, 430
    margin_left, margin_top, panel_width, panel_height, gap = 65, 55, 420, 300, 65
    colors = {"survivors": "#b45309", "full_zero": "#2563eb", "full_empirical": "#059669"}
    labels = {
        "survivors": "Survivors only",
        "full_zero": "Full universe - 0% terminal",
        "full_empirical": f"Full universe - {empirical_terminal:.1%} terminal",
    }
    all_values = np.concatenate(
        [np.asarray(item[key]) for item in paths.values() for key in colors]
    )
    y_min, y_max = float(np.nanmin(all_values)), float(np.nanmax(all_values))
    padding = max((y_max - y_min) * 0.08, 1.0)
    y_min, y_max = y_min - padding, y_max + padding

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="500" y="25" text-anchor="middle" font-family="sans-serif" font-size="18" font-weight="bold">Survivors-only versus full-universe portfolios</text>',
    ]
    for panel_index, bucket in enumerate(("High liquidity", "Low liquidity")):
        x0 = margin_left + panel_index * (panel_width + gap)
        y0 = margin_top
        parts.append(f'<rect x="{x0}" y="{y0}" width="{panel_width}" height="{panel_height}" fill="none" stroke="#444"/>')
        parts.append(f'<text x="{x0 + panel_width / 2}" y="{y0 - 12}" text-anchor="middle" font-family="sans-serif" font-size="15">{bucket}</text>')
        item = paths[bucket]
        n = len(item["sessions"])
        step = max(1, n // 700)
        indices = list(range(0, n, step))
        if indices[-1] != n - 1:
            indices.append(n - 1)
        for key, color in colors.items():
            values = np.asarray(item[key], dtype=float)
            points = []
            for index in indices:
                x = x0 + panel_width * index / max(n - 1, 1)
                y = y0 + panel_height * (y_max - values[index]) / (y_max - y_min)
                points.append(f"{x:.1f},{y:.1f}")
            parts.append(f'<polyline points="{" ".join(points)}" fill="none" stroke="{color}" stroke-width="2"/>')
        parts.append(f'<text x="{x0 + panel_width / 2}" y="{y0 + panel_height + 28}" text-anchor="middle" font-family="sans-serif" font-size="12">Trading session</text>')
        if panel_index == 0:
            parts.append(f'<text x="18" y="{y0 + panel_height / 2}" transform="rotate(-90 18 {y0 + panel_height / 2})" text-anchor="middle" font-family="sans-serif" font-size="12">Portfolio value, starting at 100</text>')
    legend_x, legend_y = 675, 382
    for offset, key in enumerate(colors):
        y = legend_y + 16 * offset
        parts.append(f'<line x1="{legend_x}" y1="{y}" x2="{legend_x + 24}" y2="{y}" stroke="{colors[key]}" stroke-width="3"/>')
        parts.append(f'<text x="{legend_x + 30}" y="{y + 4}" font-family="sans-serif" font-size="11">{labels[key]}</text>')
    parts.append("</svg>")
    return SvgFigure("".join(parts), "Two-panel survivorship-bias cumulative-return chart")


def time_operation(function: Callable[[], Any], runs: int, warmup: bool) -> tuple[float, Any]:
    """Time a callable and return the mean duration and final result."""
    if warmup:
        function()
    timings = []
    result = None
    for _ in range(runs):
        start = time.perf_counter()
        result = function()
        timings.append(time.perf_counter() - start)
    return float(np.mean(timings)), result


def time_write(function: Callable[[], Any]) -> tuple[float, Any]:
    gc.collect()
    return time_operation(function, runs=1, warmup=False)


def time_read(function: Callable[[], Any]) -> tuple[float, Any]:
    gc.collect()
    return time_operation(function, runs=3, warmup=True)


def force_materialize_polars(frame: pl.DataFrame | pl.LazyFrame) -> pl.DataFrame:
    """Force every column of a Polars frame to be read."""
    if isinstance(frame, pl.LazyFrame):
        frame = frame.collect()
    expressions = []
    for name, dtype in frame.schema.items():
        if dtype.is_numeric():
            expressions.append(pl.col(name).sum())
        elif dtype == pl.String:
            expressions.append(pl.col(name).str.len_bytes().sum())
        else:
            expressions.append(pl.col(name).count())
    if expressions:
        frame.select(expressions)
    return frame


def estimate_memory_mb(frame: pl.DataFrame | pd.DataFrame) -> float:
    """Estimate the in-memory size of a Polars or pandas frame."""
    if isinstance(frame, pl.DataFrame):
        return frame.estimated_size("mb")
    return frame.memory_usage(deep=True).sum() / 1_000_000


def generate_ohlcv_data(n_symbols: int = 100, n_rows: int = 10_000, seed: int = 42) -> pl.DataFrame:
    """Generate a deterministic one-million-row one-minute OHLCV panel."""
    rng = np.random.default_rng(seed)
    total_rows = n_symbols * n_rows
    session_bars = 390
    n_sessions = int(np.ceil(n_rows / session_bars))
    days = np.busday_offset(np.datetime64("2024-01-02", "D"), np.arange(n_sessions), roll="forward")
    opens = days.astype("datetime64[m]") + np.timedelta64(570, "m")
    grid = (opens[:, None] + np.arange(session_bars, dtype="timedelta64[m]")[None, :]).ravel()[:n_rows]
    timestamps = np.tile(grid.astype("datetime64[us]"), n_symbols)
    symbol_names = np.array([f"SYM_{i:03d}" for i in range(n_symbols)])
    symbols = np.repeat(symbol_names, n_rows)
    base_prices = 100.0 + rng.normal(size=n_symbols) * 10.0
    returns = rng.normal(size=(n_symbols, n_rows)) * 0.001
    prices = (base_prices[:, None] * np.exp(np.cumsum(returns, axis=1))).ravel()
    noise = np.abs(rng.normal(size=total_rows)) * 0.002
    open_prices = prices * (1.0 - noise * 0.5)
    close_prices = prices * (1.0 + noise * 0.5)
    high_prices = np.maximum.reduce([open_prices, close_prices, prices * (1.0 + noise)])
    low_prices = np.minimum.reduce([open_prices, close_prices, prices * (1.0 - noise)])
    volume = np.exp(rng.normal(size=total_rows) * 0.5 + 10.0).astype(np.int64)
    return pl.DataFrame(
        {
            "timestamp": timestamps,
            "symbol": symbols,
            "open": open_prices,
            "high": high_prices,
            "low": low_prices,
            "close": close_prices,
            "volume": volume,
            "vwap": (high_prices + low_prices + close_prices) / 3.0,
            "num_trades": (volume / 1000 + rng.integers(10, 100, total_rows)).astype(np.int32),
        }
    )


def prepare_storage_benchmark() -> dict[str, Any]:
    """Create the synthetic panel and a temporary output directory."""
    benchmark_dir = Path(".benchmark_output")
    benchmark_dir.mkdir(exist_ok=True)
    return {
        "BENCHMARK_DIR": benchmark_dir,
        "ACTIVE_SCALE": "L",
        "ohlcv_df": generate_ohlcv_data(),
    }
