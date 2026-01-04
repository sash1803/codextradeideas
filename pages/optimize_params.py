"""
Parameter optimizer for Level Mapping Strategy (XO)

- Optimizes core risk/logic parameters for each timeframe.
- Uses total_R from backtest_xo_logic as the objective.
- Aggregates R over a basket of symbols so results generalise better.

HOW TO USE
----------
1. Put this file in the same folder as `live_trade_ideas.py`.
2. Edit SYMBOLS and TIMEFRAMES below as you like.
3. Edit CORE_PARAM_GRID to tighten/widen the search ranges.
4. Run:  python optimize_params.py
"""

from itertools import product
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple

import numpy as np

# Import from your existing app module
from live_trade_ideas import (
    load_ohlcv,          # host data loader
    resample_ohlcv,
    composite_profile,
    single_prints,
    drop_incomplete_bar,
    backtest_xo_logic,
)

# ------------------------------------------------------------
# Config: symbols, timeframes, and the parameter grid
# ------------------------------------------------------------

# Symbols to optimise over (edit to match what you actually trade)
SYMBOLS: List[str] = [
    "BTCUSDT",
    "ETHUSDT",
    "SOLUSDT",
    "AVAXUSDT",
]

# Timeframes to optimise (must match what your app expects)
TIMEFRAMES: List[str] = ["1h", "4h", "12h", "1d"]

# Number of candles to load per symbol/timeframe
HISTORY_LIMIT: int = 1500

# Composite settings (kept fixed in this script; you can tweak if you want)
LOOKBACK_DAYS: int = 35
COMP_BINS: int = 240

# Minimum total number of trades per param set (across all symbols) to accept.
MIN_TRADES: int = 10

# Core parameter grid.
# Keep this small at first; you can widen once everything works.
CORE_PARAM_GRID: Dict[str, List[Any]] = {
    "ema_fast":      [10, 12, 16],
    "ema_slow":      [21, 34],
    "atr_period":    [10, 14],
    "atr_mult":      [1.0, 1.25, 1.5],
    "min_stop_bps":  [15, 25],
    "min_tp1_bps":   [40, 60],
    "min_rr":        [1.0, 1.3],
    "hold_bars":     [1, 2],
    "retest_bars":   [4, 6],
    "tol_bps":       [3, 5],   # level tolerance in basis points
    # Advanced but still cheap:
    "tp1_exit_pct":  [50, 100],
    "max_pending_bars": [0, 10],
    # Extra ATR buffer beyond zone to SL (0 = off)
    "extra_atr_buffer": [0.0, 0.5],
}


@dataclass
class ParamResult:
    tf: str
    params: Dict[str, Any]
    total_R: float
    total_trades: int
    hit_rate_R: float


# ------------------------------------------------------------
# Data prep
# ------------------------------------------------------------

def load_and_prepare(symbol: str, tf: str, limit: int, drop_last: bool = True):
    """
    Wrapper around your load_ohlcv + resampling, mirroring run_for_symbol().
    """
    df = load_ohlcv(symbol, tf, limit=limit, category="linear")
    if "time" in df.columns:
        df = df.set_index("time")
    df.index = np.array(df.index, dtype="datetime64[ns]")
    df = df[["open", "high", "low", "close", "volume"]].dropna()
    df = df.sort_index()

    df = resample_ohlcv(df, tf)
    if drop_last:
        df = drop_incomplete_bar(df, tf)
    return df


def make_params(tf: str,
                overrides: Dict[str, Any],
                sp_bands,
                lookback_days: int,
                comp_bins: int) -> Dict[str, Any]:
    """
    Build the params dict expected by backtest_xo_logic, using
    overrides from grid + sensible defaults from your Streamlit UI.
    """
    tol_bps = overrides.get("tol_bps", 5)

    params: Dict[str, Any] = {
        "tf_for_stats": tf,

        # Core knobs
        "ema_fast":      int(overrides.get("ema_fast", 12)),
        "ema_slow":      int(overrides.get("ema_slow", 21)),
        "atr_period":    int(overrides.get("atr_period", 14)),
        "hold_bars":     int(overrides.get("hold_bars", 2)),
        "retest_bars":   int(overrides.get("retest_bars", 6)),
        "tol":           float(tol_bps) / 10000.0,
        "atr_mult":      float(overrides.get("atr_mult", 1.5)),
        "min_stop_bps":  int(overrides.get("min_stop_bps", 25)),
        "min_tp1_bps":   int(overrides.get("min_tp1_bps", 50)),
        "min_rr":        float(overrides.get("min_rr", 1.2)),
        "entry_style":   "Limit-on-retest",  # you can change this to "Stop-through" if you prefer

        # Take-profit behaviour
        "tp1_exit_pct":  int(overrides.get("tp1_exit_pct", 100)),

        # Volume profile / composite
        "lookback_days": int(lookback_days),
        "comp_bins":     int(comp_bins),

        # Trigger types
        "use_val_reclaim": True,
        "use_vah_break":   True,
        "use_sp_cont":     True,
        "sp_bands":        sp_bands,

        # Pending order behaviour
        "max_pending_bars":      int(overrides.get("max_pending_bars", 0)),
        "invalidate_on_regime_flip": True,

        # Extra ATR buffer on SL
        "_extra_atr_buffer": float(overrides.get("extra_atr_buffer", 0.0)),

        # No advanced filters in optimisation (keeps it simple).
        "block_countertrend_entries": False,
        "filters_func": None,
    }
    return params


# ------------------------------------------------------------
# Optimisation core
# ------------------------------------------------------------

def optimise_for_timeframe(tf: str,
                           symbols: List[str],
                           param_grid: Dict[str, List[Any]],
                           history_limit: int,
                           lookback_days: int,
                           comp_bins: int,
                           min_trades: int) -> ParamResult:
    """
    For a given timeframe, run a grid search over `param_grid`.
    Objective: maximise total_R summed across all `symbols`.
    """
    print(f"\n=== Optimising timeframe {tf} ===")

    # Pre-load data + composites per symbol (so we don't reload per combo)
    symbol_data: Dict[str, Any] = {}
    for sym in symbols:
        try:
            df = load_and_prepare(sym, tf, limit=history_limit, drop_last=True)
            if df.empty:
                print(f"  [WARN] {sym} @ {tf}: empty DF, skipping.")
                continue
            comp = composite_profile(df, days=lookback_days, n_bins=comp_bins)
            sp_bands = single_prints(comp["hist"])
            symbol_data[sym] = (df, comp, sp_bands)
            print(f"  Loaded {sym} @ {tf}: {len(df)} bars.")
        except Exception as e:
            print(f"  [ERROR] Failed to load {sym} @ {tf}: {e}")

    if not symbol_data:
        raise RuntimeError(f"No valid data loaded for timeframe {tf}.")

    # Prepare grid
    keys = list(param_grid.keys())
    value_lists = [param_grid[k] for k in keys]
    total_combos = 1
    for vs in value_lists:
        total_combos *= len(vs)
    print(f"  Searching {total_combos} parameter combinations...")

    best: ParamResult | None = None
    tried = 0

    for combo_vals in product(*value_lists):
        overrides = dict(zip(keys, combo_vals))
        tried += 1

        # Aggregate results across all symbols
        total_R = 0.0
        total_trades = 0
        hit_rates = []

        for sym, (df, comp, sp_bands) in symbol_data.items():
            params = make_params(tf, overrides, sp_bands, lookback_days, comp_bins)
            try:
                trades_df, summary, _, _ = backtest_xo_logic(df, comp, params, force_eod_close=False)
            except Exception as e:
                print(f"    [ERROR] {sym} @ {tf} combo {overrides}: {e}")
                total_R = float("-inf")
                break

            total_R += float(summary.get("total_R", 0.0))
            total_trades += int(summary.get("trades", 0))
            hr = summary.get("hit_rate_R", np.nan)
            if np.isfinite(hr):
                hit_rates.append(hr)

        if total_trades < min_trades:
            # Too few trades to trust the stats; skip
            continue

        avg_hit_rate = float(np.mean(hit_rates)) if hit_rates else float("nan")
        result = ParamResult(tf=tf,
                             params=make_params(tf, overrides, sp_bands=None,
                                                lookback_days=lookback_days,
                                                comp_bins=comp_bins),
                             total_R=total_R,
                             total_trades=total_trades,
                             hit_rate_R=avg_hit_rate)

        if (best is None) or (result.total_R > best.total_R):
            best = result
            print(
                f"  [NEW BEST] combo #{tried} @ {tf}: "
                f"total_R={result.total_R:.2f}, trades={result.total_trades}, "
                f"hit_rate_R={result.hit_rate_R:.2%}"
            )

    if best is None:
        raise RuntimeError(f"No valid parameter set found for timeframe {tf} (min_trades={min_trades}).")

    return best


def optimise_all_timeframes():
    """
    Run optimisation for all TIMEFRAMES and print a summary table.
    """
    results: List[ParamResult] = []
    for tf in TIMEFRAMES:
        best_tf = optimise_for_timeframe(
            tf=tf,
            symbols=SYMBOLS,
            param_grid=CORE_PARAM_GRID,
            history_limit=HISTORY_LIMIT,
            lookback_days=LOOKBACK_DAYS,
            comp_bins=COMP_BINS,
            min_trades=MIN_TRADES,
        )
        results.append(best_tf)

    print("\n======= BEST PARAMS PER TIMEFRAME =======")
    for r in results:
        print(f"\nTimeframe: {r.tf}")
        print(f"  total_R      : {r.total_R:.2f}")
        print(f"  total_trades : {r.total_trades}")
        print(f"  hit_rate_R   : {r.hit_rate_R:.2%}")
        print("  Params to use:")
        for k, v in sorted(r.params.items()):
            print(f"    {k}: {v}")


if __name__ == "__main__":
    optimise_all_timeframes()
