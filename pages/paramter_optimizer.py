from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from dataclasses import dataclass
from itertools import product
from typing import Dict, Any, List, Tuple

# ⬇️ Adjust this import if your file has a different name
from live_trade_ideas import (
    load_ohlcv,
    resample_ohlcv,
    composite_profile,
    single_prints,
    drop_incomplete_bar,
    backtest_xo_logic,
)


# ============================
# Dataclass for results
# ============================

@dataclass
class ParamResult:
    tf: str
    overrides: Dict[str, Any]
    total_R: float
    total_trades: int
    hit_rate_R: float


# ============================
# Helper functions
# ============================

def _opt_load_and_prepare(
    symbol: str,
    tf: str,
    limit: int,
    drop_last: bool = True,
) -> pd.DataFrame:
    """
    Wrapper around load_ohlcv + resampling, mirroring the main strategy page.
    """
    df = load_ohlcv(symbol, tf, limit=limit, category="linear")

    if "time" in df.columns:
        df = df.set_index("time")

    df = df.sort_index()
    cols = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
    df = df[cols].dropna()

    df = resample_ohlcv(df, tf)
    if drop_last:
        df = drop_incomplete_bar(df, tf)

    return df


def _opt_make_params(
    tf: str,
    overrides: Dict[str, Any],
    lookback_days: int,
    comp_bins: int,
    sp_bands: List[Tuple[float, float]],
) -> Dict[str, Any]:
    """
    Build the params dict expected by backtest_xo_logic, using `overrides`
    for the knobs we want to optimise and sensible defaults for the rest.
    """
    tol_bps = int(overrides.get("tol_bps", 5))

    params: Dict[str, Any] = {
        "tf_for_stats": tf,

        # Core knobs (optimised)
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

        # Entry / exit behaviour
        "entry_style":   "Limit-on-retest",   # you can change to "Stop-through" if you prefer
        "tp1_exit_pct":  int(overrides.get("tp1_exit_pct", 100)),

        # Extra ATR buffer on SL
        "_extra_atr_buffer": float(overrides.get("extra_atr_buffer", 0.0)),

        # Composite / volume profile
        "lookback_days": int(lookback_days),
        "comp_bins":     int(comp_bins),

        # Triggers
        "use_val_reclaim": True,
        "use_vah_break":   True,
        "use_sp_cont":     True,
        "sp_bands":        sp_bands,

        # Pending order behaviour
        "max_pending_bars": int(overrides.get("max_pending_bars", 0)),
        "invalidate_on_regime_flip": True,

        # Filters: off inside optimiser to keep search clean & fast
        "block_countertrend_entries": False,
        "filters_func": None,
    }

    return params


def _optimise_for_timeframe(
    tf: str,
    symbols: List[str],
    param_grid: Dict[str, List[Any]],
    history_limit: int,
    lookback_days: int,
    comp_bins: int,
    min_trades: int,
) -> ParamResult:
    """
    For a given timeframe, run a grid search over `param_grid`.

    Objective: maximise total_R summed across all `symbols`.
    """
    st.write(f"### Optimising timeframe `{tf}`")

    # --- Pre-load data + composites per symbol (avoid reloading per combo) ---
    symbol_data: Dict[str, Any] = {}
    for sym in symbols:
        try:
            df = _opt_load_and_prepare(sym, tf, limit=history_limit, drop_last=True)
            if df.empty:
                st.warning(f"{sym} @ {tf}: empty data, skipping.")
                continue

            comp = composite_profile(df, days=lookback_days, n_bins=comp_bins)
            sp_bands = single_prints(comp["hist"])

            symbol_data[sym] = (df, comp, sp_bands)
            st.write(f"- Loaded `{sym}` @ `{tf}` with {len(df)} bars.")
        except Exception as e:
            st.error(f"Failed to load {sym} @ {tf}: {e}")

    if not symbol_data:
        raise RuntimeError(f"No valid data loaded for timeframe {tf}.")

    # --- Build cartesian grid of parameter combinations ---
    keys = list(param_grid.keys())
    value_lists = [param_grid[k] for k in keys]
    total_combos = 1
    for vs in value_lists:
        total_combos *= max(1, len(vs))

    st.write(f"- Searching **{total_combos}** parameter combinations...")

    progress = st.progress(0.0, text=f"{tf}: starting grid search...")
    best: ParamResult | None = None
    tried = 0

    for combo_vals in product(*value_lists):
        overrides = dict(zip(keys, combo_vals))
        tried += 1

        total_R = 0.0
        total_trades = 0
        hit_rates: List[float] = []

        # Evaluate one combo across all symbols
        for sym, (df, comp, sp_bands) in symbol_data.items():
            params = _opt_make_params(tf, overrides, lookback_days, comp_bins, sp_bands)
            try:
                trades_df, summary, _, _ = backtest_xo_logic(
                    df,
                    comp,
                    params,
                    force_eod_close=False,
                )
            except Exception as e:
                st.warning(f"Error for {sym} @ {tf} with {overrides}: {e}")
                total_R = float("-inf")
                break

            total_R += float(summary.get("total_R", 0.0))
            total_trades += int(summary.get("trades", 0))
            hr = summary.get("hit_rate_R", np.nan)
            if np.isfinite(hr):
                hit_rates.append(hr)

        # Ensure we have enough trades to trust this combo
        if total_trades < min_trades:
            progress.progress(
                tried / total_combos,
                text=f"{tf}: skipping low-trade combo {tried}/{total_combos}",
            )
            continue

        avg_hit_rate = float(np.mean(hit_rates)) if hit_rates else float("nan")
        result = ParamResult(
            tf=tf,
            overrides=overrides,
            total_R=total_R,
            total_trades=total_trades,
            hit_rate_R=avg_hit_rate,
        )

        if (best is None) or (result.total_R > best.total_R):
            best = result
            progress.progress(
                tried / total_combos,
                text=(
                    f"{tf}: new best combo {tried}/{total_combos} — "
                    f"total_R={result.total_R:.2f}, "
                    f"trades={result.total_trades}, "
                    f"hit={result.hit_rate_R:.2%}"
                ),
            )
        else:
            progress.progress(
                tried / total_combos,
                text=f"{tf}: tested {tried}/{total_combos} combos...",
            )

    progress.empty()

    if best is None:
        raise RuntimeError(
            f"No valid parameter set found for timeframe {tf} (min_trades={min_trades})."
        )

    st.success(
        f"Best for {tf}: total_R={best.total_R:.2f}, "
        f"trades={best.total_trades}, hit_rate_R={best.hit_rate_R:.2%}"
    )

    return best


def _parse_list(text: str, cast_type):
    vals: List[Any] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            vals.append(cast_type(part))
        except ValueError:
            # ignore bad entries
            continue
    return vals


# ============================
# Streamlit UI
# ============================

st.title("📊 Parameter Optimiser — Level Mapping Strategy")

st.markdown(
    """
This page runs a **grid search** over your core parameters and finds the
combination that maximises **total R** across a basket of symbols, per timeframe.

Keep the grids **small at first** (e.g. 2–3 values per parameter), then widen if needed.
"""
)

# --- Top-level settings ---
with st.expander("Data & timeframe settings", expanded=True):
    syms_text = st.text_input(
        "Symbols to optimise over (comma-separated)",
        value="BTCUSDT,ETHUSDT,SOLUSDT,AVAXUSDT",
        help="All these symbols are used when scoring a parameter combo.",
    )

    tf_choices = ["1h", "4h", "12h", "1d"]
    tfs_selected = st.multiselect(
        "Timeframes to optimise",
        options=tf_choices,
        default=["1h", "4h", "12h"],
    )

    history_limit = st.number_input(
        "Bars to load per symbol/timeframe",
        min_value=300,
        max_value=5000,
        value=1500,
        step=100,
    )

    lookback_days_opt = st.number_input(
        "Composite lookback days",
        min_value=10,
        max_value=90,
        value=35,
        step=1,
        help="Used for rolling composite in the backtest.",
    )

    comp_bins_opt = st.number_input(
        "Composite bins",
        min_value=60,
        max_value=400,
        value=240,
        step=20,
        help="Number of volume-profile bins.",
    )

    min_trades = st.number_input(
        "Minimum total trades per parameter combo (across all symbols)",
        min_value=1,
        max_value=500,
        value=10,
        step=1,
        help="Combos with fewer trades than this are ignored.",
    )

# --- Parameter grid ---
with st.expander("Parameter grid (advanced)", expanded=False):
    st.caption(
        "Enter comma-separated lists. The optimiser will test **all combinations**."
    )

    col1, col2 = st.columns(2)

    ema_fast_text = col1.text_input("EMA fast values", "10,12,16")
    ema_slow_text = col2.text_input("EMA slow values", "21,34")

    atr_period_text = col1.text_input("ATR period values", "10,14")
    atr_mult_text   = col2.text_input("ATR multiple for stops", "1.0,1.25,1.5")

    min_stop_bps_text = col1.text_input("Min stop distance (bps)", "15,25")
    min_tp1_bps_text  = col2.text_input("Min TP1 distance (bps)", "40,60")

    min_rr_text       = col1.text_input("Min R/R to TP1", "1.0,1.3")
    hold_bars_text    = col2.text_input("Hold bars past level", "1,2")

    retest_bars_text  = col1.text_input("Retest window (bars)", "4,6")
    tol_bps_text      = col2.text_input("Level tolerance (bps)", "3,5")

    tp1_exit_pct_text = col1.text_input("TP1 exit percent", "50,100")
    max_pending_bars_text = col2.text_input(
        "Max pending bars (0 = never timeout)",
        "0,10",
    )

    extra_atr_buf_text = st.text_input("Extra ATR buffer on SL", "0.0,0.5")

    # Build the grid dict
    param_grid: Dict[str, List[Any]] = {
        "ema_fast":         _parse_list(ema_fast_text, int) or [12],
        "ema_slow":         _parse_list(ema_slow_text, int) or [21],
        "atr_period":       _parse_list(atr_period_text, int) or [14],
        "atr_mult":         _parse_list(atr_mult_text, float) or [1.5],
        "min_stop_bps":     _parse_list(min_stop_bps_text, int) or [25],
        "min_tp1_bps":      _parse_list(min_tp1_bps_text, int) or [50],
        "min_rr":           _parse_list(min_rr_text, float) or [1.2],
        "hold_bars":        _parse_list(hold_bars_text, int) or [2],
        "retest_bars":      _parse_list(retest_bars_text, int) or [6],
        "tol_bps":          _parse_list(tol_bps_text, int) or [5],
        "tp1_exit_pct":     _parse_list(tp1_exit_pct_text, int) or [100],
        "max_pending_bars": _parse_list(max_pending_bars_text, int) or [0],
        "extra_atr_buffer": _parse_list(extra_atr_buf_text, float) or [0.0],
    }

    total_combos = 1
    for v in param_grid.values():
        total_combos *= max(1, len(v))
    st.write(f"Total combinations to test **per timeframe**: `{total_combos}`")


run_opt_btn = st.button("🚀 Run optimiser", type="primary")

if run_opt_btn:
    symbols = [s.strip().upper() for s in syms_text.split(",") if s.strip()]

    if not symbols:
        st.error("Please provide at least one symbol.")
    elif not tfs_selected:
        st.error("Please select at least one timeframe.")
    else:
        all_results: List[ParamResult] = []

        for tf in tfs_selected:
            try:
                best = _optimise_for_timeframe(
                    tf=tf,
                    symbols=symbols,
                    param_grid=param_grid,
                    history_limit=int(history_limit),
                    lookback_days=int(lookback_days_opt),
                    comp_bins=int(comp_bins_opt),
                    min_trades=int(min_trades),
                )
                all_results.append(best)
            except Exception as e:
                st.error(f"Optimisation failed for timeframe {tf}: {e}")

        if all_results:
            st.subheader("Best parameter combos per timeframe")

            summary_rows = []
            for r in all_results:
                summary_rows.append(
                    {
                        "Timeframe": r.tf,
                        "total_R": r.total_R,
                        "total_trades": r.total_trades,
                        "hit_rate_R": r.hit_rate_R,
                        "overrides": r.overrides,
                    }
                )

            summary_df = pd.DataFrame(summary_rows).set_index("Timeframe")
            st.dataframe(summary_df)

            # Ready-to-paste dict for your main strategy page
            lines = ["OPTIMISED_PARAM_OVERRIDES = {"]
            for r in all_results:
                lines.append(f"    '{r.tf}': {r.overrides},")
            lines.append("}")
            st.code("\n".join(lines), language="python")
