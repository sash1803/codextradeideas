# pages/01_Live_Bybit_MTF_OB.py
# Bybit live data -> MTF OB pipeline -> Backtesting with TP floors (TP1>=1R, TP2>=2R)

from __future__ import annotations

import time
from typing import Dict, List, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import streamlit as st

import requests, time, math
from typing import List

from mtf_ob_report_app import (
    DEFAULTS,
    apply_mtf_confluence,
    bias_label,
    compute_tradeplan_for_side,
    detect_ob_zones,
    enrich_ob_context,
    evaluate_mitigation,
    fvg_mask,
    ob_backtest_stats,
    ob_backtest_tf,
    tpo_profile,
    pivots,
)

class BybitHTTPError(RuntimeError):
    """Generic Bybit HTTP/retCode error."""

class BybitForbidden(RuntimeError):
    """Raised on HTTP 403 (IP blocked / forbidden)."""


UA_HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; StreamlitApp/1.0; +https://streamlit.io)"
}


matplotlib.use("Agg")

st.set_page_config(page_title="MTF OB Trade Plan — Bybit + Backtest", layout="centered")

import os

def _safe_secret(key: str, default: str | None = None):
    try:
        # Accessing st.secrets triggers parsing; wrap it entirely
        return st.secrets[key]
    except Exception:
        return os.environ.get(key, default)
    
BYBIT_BASE = _safe_secret("BYBIT_BASE", "https://api.bybit.com")

TF_TO_BYBIT = {"Weekly": "W", "Daily": "D", "4H": "240", "1H": "60", "30m": "30", "15m": "15"}
ORDER_TFS = ["Weekly", "Daily", "4H", "1H", "30m", "15m"]


# -----------------------------------------------------------------------------
# UI Defaults
# -----------------------------------------------------------------------------
if "ui_settings" not in st.session_state:
    st.session_state.ui_settings = DEFAULTS.copy()
cfg = {**DEFAULTS, **st.session_state.ui_settings}

st.title("Multi-Timeframe OB Trade Plan — Bybit Live Data + Backtest")

with st.sidebar:
    st.subheader("Data Source")
    st.caption("Bybit public API (no key).")

    category = st.selectbox(
        "Market",
        ["perps", "spot"],
        index=0,
        help=(
            "Pick data source type: 'linear' = USDT/USDC-margined perpetual futures; "
            "'spot' = cash market."
        ),
    )

    symbol = st.text_input(
        "Symbol",
        value="BTCUSDT",
        help="Trading pair to fetch (e.g., BTCUSDT, ETHUSDT, SOLUSDT).",
    )

    per_tf_bars = st.number_input(
        "Bars per timeframe",
        min_value=200,
        max_value=2000,
        value=2000,
        step=100,
        help=(
            "History depth per TF (Weekly/Daily/4H/1H/30m/15m). "
            "More bars = deeper detection & backtests."
        ),
    )

    st.markdown("---")
    st.subheader("Detection Preset")

    preset = st.selectbox(
        "Preset",
        ["Balanced (recommended)", "Conservative", "Aggressive"],
        index=0,
        help="Sets sensible defaults for detection strictness (min score & displacement).",
    )

    if preset == "Balanced (recommended)":
        min_score = 3.0
        displacement_k = 1.5
    elif preset == "Conservative":
        min_score = 3.5
        displacement_k = 1.8
    else:
        min_score = 2.5
        displacement_k = 1.2

    adaptive_disp = st.checkbox(
        "Adaptive displacement",
        value=False,
        help="If on, raises/lowers displacement requirement based on recent volatility/quantiles.",
    )

    displacement_mult = st.slider(
        "Adaptive floor ×",
        min_value=0.6,
        max_value=1.8,
        value=1.0,
        step=0.05,
        help="Multiplier for the adaptive displacement floor. Larger = stricter impulse requirement.",
    )

    pivot_left = st.number_input(
        "Pivot left",
        min_value=2,
        max_value=10,
        value=3,
        help="Bars required on the left to confirm a swing pivot. Higher = fewer, stronger pivots.",
    )

    pivot_right = st.number_input(
        "Pivot right",
        min_value=2,
        max_value=10,
        value=3,
        help="Bars required on the right to confirm a swing pivot. Higher = fewer, stronger pivots.",
    )

    fvg_span = st.selectbox(
        "FVG span",
        [3, 5],
        index=0,
        help="Window size for fair-value gap detection used to enrich OB scoring/context.",
    )

    iou_thresh = st.slider(
        "IoU dedupe threshold",
        min_value=0.30,
        max_value=0.95,
        value=0.50,
        help=(
            "Overlap cutoff to merge overlapping OBs on the same TF. Higher keeps more distinct "
            "zones; lower merges more."
        ),
    )

    st.markdown("---")
    st.subheader("Stops & Targets")

    stop_buffer_mode = st.selectbox(
        "Stop buffer mode",
        ["Absolute", "% of entry", "ATR14 × multiple"],  # <-- use this exact string
        index=2,
    )

    stop_buffer_abs = st.number_input(
        "Absolute buffer (if Absolute)",
        value=0.02,
        step=0.0001,
        format="%.6f",
        help="Only used in 'Absolute' mode. Fixed currency added beyond the far side of the zone.",
    )

    stop_buffer_pct = st.number_input(
        "Percent buffer (0.01 = 1%)",
        value=0.0,
        step=0.001,
        format="%.4f",
        help="Only used in 'Percent' mode. % of entry added beyond the far side of the zone.",
    )

    if "atr14_mults" not in st.session_state:
        st.session_state["atr14_mults"] = {
            "Weekly": 1.5,
            "Daily": 1.5,
            "4H": 1.2,
            "1H": 1.0,
            "30m": 1.0,
            "15m": 1.0,
        }

    atr_mults = st.session_state["atr14_mults"].copy()
    for tf in ["Weekly", "Daily", "4H", "1H", "30m", "15m"]:
        atr_mults[tf] = st.number_input(
            f"ATR14 × — {tf}",
            value=float(atr_mults[tf]),
            step=0.05,
            help=(
                "Only used in 'ATR14 × multiple' mode. Wider = more room beyond the zone, lower R, "
                "fewer noise stops."
            ),
        )
    st.session_state["atr14_mults"] = atr_mults

    target_policy = st.selectbox(
        "Target policy",
        ["Opposing OB mids", "2R/3R", "Structure (swing highs/lows)"],
        index=0,
        help=(
            "How TPs are chosen before floors. Regardless, TP1≥1R and TP2≥2R are enforced."
        ),
    )

    st.markdown("---")
    fetch_btn = st.button(
        "Fetch & Run",
        type="primary",
        help="Download candles, detect OBs, build plans.",
    )


# -----------------------------------------------------------------------------
# Data fetching
# -----------------------------------------------------------------------------
@st.cache_data(ttl=180)
def cached_bybit(category, symbol, interval, bars):
    return get_bybit_ohlcv(category, symbol, interval, bars)

def _bybit_klines_once(
    *, category: str, symbol: str, interval: str,
    start: int | None, end: int | None, limit: int = 500
) -> List[list]:
    url = f"{BYBIT_BASE}/v5/market/kline"
    params = {"category": category, "symbol": symbol, "interval": interval, "limit": str(limit)}
    if start is not None: params["start"] = str(start)
    if end is not None: params["end"] = str(end)
    r = requests.get(url, params=params, headers=UA_HEADERS, timeout=20)
    if r.status_code == 403:
        raise BybitForbidden(f"403 Forbidden: {r.text}")
    if r.status_code >= 400:
        raise BybitHTTPError(f"{r.status_code}: {r.text}")
    data = r.json()
    if data.get("retCode") != 0:
        # 10001/10006 etc — surface clearly
        raise BybitHTTPError(f"Bybit retCode={data.get('retCode')} retMsg={data.get('retMsg')}")
    return data.get("result", {}).get("list", []) or []


def get_bybit_ohlcv(category: str, symbol: str, interval: str, bars: int) -> pd.DataFrame:
    got: List[list] = []
    end_ms: int | None = None
    tries = 0
    page = min(500, max(1, bars))           # smaller page to be friendlier
    max_tries = 10

    while len(got) < bars and tries < max_tries:
        try:
            rows = _bybit_klines_once(
                category=category, symbol=symbol, interval=interval,
                start=None, end=end_ms, limit=page
            )
        except BybitForbidden as e:
            # Bubble up; our outer code will switch to fallback provider
            raise
        except BybitHTTPError as e:
            # brief backoff on transient HTTP / retCode errors
            sleep = 0.4 * (2 ** tries)
            time.sleep(min(3.0, sleep))
            tries += 1
            if tries >= max_tries:
                raise
            continue

        if not rows:
            break

        got.extend(rows)
        oldest = min(int(r[0]) for r in rows)
        end_ms = oldest - 1
        tries += 1
        time.sleep(0.12)  # gentle pacing

    if not got:
        return pd.DataFrame(columns=["time","open","high","low","close","volume"])

    got = sorted(got, key=lambda r: int(r[0]))
    df = pd.DataFrame(got, columns=["start","open","high","low","close","volume","turnover"])
    for c in ["open","high","low","close","volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    ts = pd.to_datetime(df["start"].astype(np.int64), unit="ms", utc=True).dt.tz_convert("UTC").dt.tz_localize(None)
    out = (pd.DataFrame({"time":ts,"open":df["open"],"high":df["high"],"low":df["low"],"close":df["close"],"volume":df["volume"]})
           .dropna().drop_duplicates(subset="time").sort_values("time").reset_index(drop=True))
    if len(out) > bars:
        out = out.iloc[-bars:].reset_index(drop=True)
    return out
# -----------------------------------------------------------------------------
# Pipeline
# -----------------------------------------------------------------------------
def _recompute_rr(entry: float, sl: float, tp: float | None) -> float | None:
    if tp is None:
        return None
    risk = abs(entry - sl)
    return round(abs(tp - entry) / risk, 2) if risk > 0 else None


def run_pipeline(dfs_by_tf: Dict[str, pd.DataFrame]) -> Tuple[dict, dict, dict, dict]:
    detected: Dict[str, Tuple[list, list]] = {}
    profiles: Dict[str, object] = {}
    biases: Dict[str, str] = {}

    for tf, df in dfs_by_tf.items():
        if df.empty or len(df) < 50:
            detected[tf] = ([], [])
            continue

        bulls, bears, _, _ = detect_ob_zones(
            df,
            use_wicks=cfg["use_wicks"],
            pivot_left=pivot_left,
            pivot_right=pivot_right,
            displacement_k=displacement_k,
            adaptive_disp=adaptive_disp,
            displacement_mult=displacement_mult,
            min_score=min_score,
            fvg_span=fvg_span,
            iou_thresh=iou_thresh,
            return_debug=False,
        )

        bulls = evaluate_mitigation(df, bulls, "bullish")
        bears = evaluate_mitigation(df, bears, "bearish")

        prof = tpo_profile(df)
        b_fvg, s_fvg = fvg_mask(df, span=fvg_span)
        bias = bias_label(df, ma=50)

        bulls2, bears2, _ = enrich_ob_context(df, bulls, bears, prof, b_fvg, s_fvg)

        detected[tf] = (bulls2, bears2)
        profiles[tf] = prof
        biases[tf] = bias

    apply_mtf_confluence(detected, ORDER_TFS)

    plans: Dict[str, dict] = {}
    for tf in ORDER_TFS:
        df = dfs_by_tf.get(tf, pd.DataFrame())
        bulls, bears = detected.get(tf, ([], []))

        if df.empty:
            plans[tf] = {"bullish": [], "bearish": []}
            continue

        stop_atr_mult = st.session_state["atr14_mults"].get(tf, 1.0)

        piv_hi, piv_lo = pivots(df, left=pivot_left, right=pivot_right)

        bull_plans = compute_tradeplan_for_side(
            "bullish", bulls, bears, stop_buffer=stop_buffer_abs, tf_name=tf,
            tf_bias=biases.get(tf,"neutral"), target_policy=target_policy, df=df,
            piv_hi=piv_hi, piv_lo=piv_lo,  # <-- pass these so “Structure” works
            stop_buffer_mode=stop_buffer_mode, stop_buffer_abs=stop_buffer_abs,
            stop_buffer_pct=stop_buffer_pct, stop_buffer_atr=stop_atr_mult
        )

        bear_plans = compute_tradeplan_for_side(
            "bearish", bulls, bears, stop_buffer=stop_buffer_abs, tf_name=tf,
            tf_bias=biases.get(tf,"neutral"), target_policy=target_policy, df=df,
            piv_hi=piv_hi, piv_lo=piv_lo,  # <-- pass these so “Structure” works
            stop_buffer_mode=stop_buffer_mode, stop_buffer_abs=stop_buffer_abs,
            stop_buffer_pct=stop_buffer_pct, stop_buffer_atr=stop_atr_mult
        )

        # Safeguard: recompute R and ensure TP floors for display
        for rows, side in [(bull_plans, "bullish"), (bear_plans, "bearish")]:
            for r in rows:
                risk = abs(r.entry - r.sl)
                if risk > 0 and r.tp1 is not None and r.tp2 is not None:
                    # Ensure TP1 >= 1R and TP2 >= 2R
                    if side == "bullish":
                        r.tp1 = max(r.tp1, r.entry + 1.0 * risk)
                        r.tp2 = max(r.tp2, max(r.tp1, r.entry + 2.0 * risk))
                    else:
                        r.tp1 = min(r.tp1, r.entry - 1.0 * risk)
                        r.tp2 = min(r.tp2, min(r.tp1, r.entry - 2.0 * risk))
                    r.rr1 = _recompute_rr(r.entry, r.sl, r.tp1)
                    r.rr2 = _recompute_rr(r.entry, r.sl, r.tp2)

        plans[tf] = {"bullish": bull_plans, "bearish": bear_plans}

    return detected, plans, profiles, biases


# -----------------------------------------------------------------------------
# Fetch & Plans
# -----------------------------------------------------------------------------
if fetch_btn:
    st.write(f"Fetching Bybit data for **{symbol}** ({category}) …")

    dfs: Dict[str, pd.DataFrame] = {}
    errs: List[str] = []

    for tf in ORDER_TFS:
        try:
            dfs[tf] = get_bybit_ohlcv(
                category=category,
                symbol=symbol,
                interval=TF_TO_BYBIT[tf],
                bars=int(per_tf_bars),
            )
        except Exception as e:
            dfs[tf] = pd.DataFrame(columns=["time", "open", "high", "low", "close", "volume"])
            errs.append(f"{tf}: {e}")

    if errs:
        with st.expander("Fetch warnings / errors"):
            for e in errs:
                st.warning(e)

    detected, plans, profiles, biases = run_pipeline(dfs)
    st.session_state.update({"dfs": dfs, "detected": detected, "biases": biases, "plans": plans})

    st.subheader("Trade Plans (per TF)")
    for tf in ORDER_TFS:
        st.markdown(f"### {tf} — Bias: **{biases.get(tf, 'neutral')}**")

        bull = plans.get(tf, {}).get("bullish", [])
        bear = plans.get(tf, {}).get("bearish", [])

        def _table(rows):
            if not rows:
                return pd.DataFrame(
                    columns=["entry", "sl", "tp1", "tp2", "rr1", "rr2", "reasons", "invalidations"]
                )
            return pd.DataFrame(
                [
                    {
                        "entry": r.entry,
                        "sl": r.sl,
                        "tp1": r.tp1,
                        "tp2": r.tp2,
                        "rr1": r.rr1,
                        "rr2": r.rr2,
                        "reasons": "• " + " | ".join(r.reasons),
                        "invalidations": "• " + " | ".join(r.invalidations),
                    }
                    for r in rows
                ]
            )

        st.caption("Bullish")
        st.dataframe(_table(bull), use_container_width=True)

        st.caption("Bearish")
        st.dataframe(_table(bear), use_container_width=True)

else:
    st.info("Set your market + symbol, then click **Fetch & Run**. Backtesting below unlocks afterwards.")


# -----------------------------------------------------------------------------
# Backtesting
# -----------------------------------------------------------------------------
st.markdown("---")
st.header("Backtesting")

bt_disabled = not all(k in st.session_state for k in ["dfs", "detected"])
if bt_disabled:
    st.info("Load live data first to enable backtesting.")
else:
    dfs = st.session_state.get("dfs", {})
    detected = st.session_state.get("detected", {})

    colA, colB, colC = st.columns([1.1, 1, 1])
    with colA:
        tfs_to_test = st.multiselect(
            "Timeframes to backtest",
            ["Weekly", "Daily", "4H", "1H", "30m", "15m"],
            default=["4H", "1H", "30m", "15m"],
            help="Select which TFs to simulate trades on, using the detected OBs for each.",
        )

    with colB:
        rr_min = st.number_input(
            "RR min (TP2 target)",
            min_value=1.0,
            max_value=5.0,
            value=2.0,
            step=0.25,
            help=(
                "Minimum R multiple enforced for TP2 in backtests "
                "(alongside global TP floors: TP1≥1R, TP2≥2R)."
            ),
        )

        expiry_bars = st.number_input(
            "Signal expiry (bars)",
            min_value=5,
            max_value=500,
            value=30,
            step=5,
            help=("Max bars after idea creation to get filled at entry. Otherwise the idea expires (NoFill)."),
        )

    with colC:
        max_hold_bars = st.number_input(
            "Max hold (bars)",
            min_value=10,
            max_value=2000,
            value=300,
            step=10,
            help=("Max bars to hold after fill. If neither TP2 nor BE/SL hit, exit at last bar (TimedOut)."),
        )

        risk_cash = st.number_input(
            "Risk per trade (cash)",
            min_value=1.0,
            value=1000.0,
            step=50.0,
            help="Dollar risk per trade used for converting R results to $ in stats (does not affect fills/exits).",
        )

    show_equity = st.checkbox(
        "Show equity (R) curve per TF",
        value=False,
        help="If on, plots cumulative R for closed trades (TP2/SL/BE) per timeframe.",
    )

    run_bt = st.button(
        "Run Backtest",
        type="primary",
        disabled=bt_disabled,
        help="Simulate entries/exits with BE after TP1.",
    )

    if run_bt:
        agg_stats_rows: List[dict] = []

        for tf in tfs_to_test:
            df = dfs.get(tf, pd.DataFrame())
            bulls, bears = detected.get(tf, ([], []))

            if df.empty or (not bulls and not bears):
                st.warning(f"{tf}: no data or no detected zones; skipping.")
                continue

            # Use SL buffer settings consistent with Ideas
            stop_atr_mult = st.session_state["atr14_mults"].get(tf, 1.0)

            trades = ob_backtest_tf(
                df,
                bulls,
                bears,
                rr_min=float(rr_min),
                expiry_bars=int(expiry_bars),
                max_hold_bars=int(max_hold_bars),
                stop_buffer_mode=stop_buffer_mode,
                stop_buffer_abs=stop_buffer_abs,
                stop_buffer_pct=stop_buffer_pct,
                stop_buffer_atr=stop_atr_mult,
            )

            stats = ob_backtest_stats(trades, risk_cash=float(risk_cash))

            # ------- Per-TF rendering -------
            st.subheader(f"{tf} — Backtest")

            # Ensure DataFrame
            if not isinstance(trades, pd.DataFrame):
                try:
                    trades = pd.DataFrame(trades)
                except Exception:
                    trades = pd.DataFrame()

            # Stats table
            stat_df = pd.DataFrame(
                [
                    {
                        "Trades": stats.get("trades", 0),
                        "Wins": stats.get("wins", 0),
                        "Losses": stats.get("losses", 0),
                        "BE": stats.get("breakevens", 0),
                        "Win% (All)": round(stats.get("win_rate_pct", 0.0), 2),
                        "Win% (W/L only)": round(stats.get("win_over_wl_pct", 0.0), 2),
                        "PF": round(stats.get("profit_factor", 0.0), 2),
                        "Avg R": round(stats.get("avg_R", 0.0), 3),
                        "Expectancy R": round(stats.get("expectancy_R", 0.0), 3),
                        "Med R": round(stats.get("median_R", 0.0), 3),
                        "P25 R": round(stats.get("p25_R", 0.0), 3),
                        "P75 R": round(stats.get("p75_R", 0.0), 3),
                        "Max DD (R)": round(stats.get("max_drawdown_R", 0.0), 3),
                        "Gross R": round(stats.get("gross_R", 0.0), 3),
                        "Gross $": stats.get("gross_$", 0.0),
                        "Avg $": stats.get("avg_$", 0.0),
                        "Exp $": stats.get("exp_$", 0.0),
                    }
                ]
            )
            st.dataframe(stat_df, use_container_width=True)

            # Trades table
            show_cols = [
                c
                for c in [
                    "side",
                    "created_idx",
                    "fill_idx",
                    "exit_idx",
                    "status",
                    "entry",
                    "sl",
                    "tp1",
                    "tp2",
                    "exit_price",
                    "R",
                    "bars_held",
                ]
                if c in trades.columns
            ]
            st.caption("Trades")

            if not trades.empty and show_cols:
                st.dataframe(trades[show_cols].reset_index(drop=True), use_container_width=True, height=240)
            else:
                st.info("No trades or no standard columns to display.")

            # Equity curve (optional)
            if show_equity and not trades.empty:
                status_col = next((c for c in trades.columns if c.lower() == "status"), None)
                if status_col is None:
                    status_col = next((c for c in trades.columns if c.lower() in {"state", "result"}), None)

                have_R = "R" in trades.columns
                sort_cols = [c for c in ["exit_idx", "fill_idx"] if c in trades.columns]

                if status_col and have_R:
                    status_vals = trades[status_col].astype(str).str.upper()
                    mask_closed = status_vals.isin(["TP2", "SL", "BE", "CLOSED", "EXIT"])
                    closed = trades[mask_closed].copy()

                    if not closed.empty:
                        if sort_cols:
                            closed = closed.sort_values(sort_cols)
                        eq = closed["R"].cumsum().reset_index(drop=True)

                        fig, ax = plt.subplots(figsize=(7.2, 3.2))
                        ax.plot(eq.values)
                        ax.set_title(f"Equity (R) — {tf}")
                        ax.set_xlabel("Closed trade #")
                        ax.set_ylabel("Cumulative R")
                        st.pyplot(fig, clear_figure=True)
                    else:
                        st.info("No closed trades (TP2/SL/BE) to plot.")
                else:
                    st.info("Equity curve skipped (missing columns like status/R).")

            # Aggregate
            agg_stats_rows.append({"TF": tf, **stats})

        # Aggregate table
        if agg_stats_rows:
            st.markdown("### Aggregate (by TF)")
            agg_df = pd.DataFrame(agg_stats_rows)
            keep = [
                "TF",
                "trades",
                "wins",
                "losses",
                "breakevens",
                "win_rate_pct",
                "win_over_wl_pct",
                "profit_factor",
                "avg_R",
                "expectancy_R",
                "median_R",
                "p25_R",
                "p75_R",
                "max_drawdown_R",
                "gross_R",
                "gross_$",
                "avg_$",
                "exp_$",
            ]
            keep = [c for c in keep if c in agg_df.columns]
            st.dataframe(agg_df[keep], use_container_width=True)
