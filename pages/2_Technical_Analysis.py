# pages/2_Technical_Analysis.py
# Self-contained Technical Analysis page (CryptoCred-style)
# - Clean chart: shows labelled S/R levels and range (no trade ideas)
# - Separate per-idea charts: each long/short on its own figure
# - Valid ideas must meet minimum R/R (default 2.0) using TP1
# - Per-TF data tables:
#     • Key Levels (S/R) with rationale
#     • Valid Trade Ideas with an Explanation column
# - CSV uploads per TF, robust time parsing, PNG downloads, CSV export
# - Fully integrated backtesting UI (per timeframe)

from __future__ import annotations

import io, re
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

import time
import requests
import os

import streamlit as st
import pandas as pd
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import mplfinance as mpf

from typing import Dict, List

# Always defined so later code never crashes when CSV path isn’t used
stored_uploads: Dict[str, bytes] = {}
bulk_files = []
w_csv = d_csv = h4_csv = h1_csv = m30_csv = m15_csv = None


# =========================
# Backtest (self-contained)
# =========================
# ---------- Idea type (what your idea generator must return) ----------
@dataclass(frozen=True)
class TAIdea:
    side: str           # "long" or "short"
    entry: float
    sl: float
    tp1: float          # +1R
    tp2: float          # final take profit (>= 2R plan)
    created_idx: int    # index (in df) when the idea is created
    note: str = ""      # optional explanation

# ---------- ADAPTER: tell the backtester how to get ideas at each bar ----------
def _ideas_at_bar_adapter(df_sub: pd.DataFrame) -> list[TAIdea]:
    ideas: list[TAIdea] = []
    n = len(df_sub)
    if n < 120:
        return ideas

    def _ema(arr: np.ndarray, span: int) -> np.ndarray:
        return pd.Series(arr).ewm(span=span, adjust=False).mean().values

    def _atr14(df: pd.DataFrame) -> float:
        h, l, c1 = df["high"].values, df["low"].values, df["close"].shift(1).values
        tr = np.maximum.reduce([np.abs(h - l), np.abs(h - c1), np.abs(l - c1)])
        atr = pd.Series(tr).rolling(14).mean().values
        return float(atr[-1]) if np.isfinite(atr[-1]) else float(np.nan)

    def _pivots(df: pd.DataFrame, L: int = 3, R: int = 3):
        h, l = df["high"].values, df["low"].values
        n = len(df)
        ph = np.zeros(n, dtype=bool)
        pl = np.zeros(n, dtype=bool)
        for i in range(L, n - R):
            if all(h[i] > h[i-j] for j in range(1, L+1)) and all(h[i] > h[i+j] for j in range(1, R+1)):
                ph[i] = True
            if all(l[i] < l[i-j] for j in range(1, L+1)) and all(l[i] < l[i+j] for j in range(1, R+1)):
                pl[i] = True
        return ph, pl

    def _nearest_levels(df: pd.DataFrame, ph: np.ndarray, pl: np.ndarray, ref_px: float, max_lookback: int = 600):
        start = max(0, len(df) - max_lookback)
        highs  = df["high"].values[start:]; lows = df["low"].values[start:]
        ph_i   = np.where(ph[start:])[0];   pl_i  = np.where(pl[start:])[0]
        res_levels = [float(highs[i]) for i in ph_i if highs[i] > ref_px]
        sup_levels = [float(lows[i])  for i in pl_i if lows[i]  < ref_px]
        res = sorted(res_levels, key=lambda x: x - ref_px)[:1]
        sup = sorted(sup_levels, key=lambda x: ref_px - x)[:1]
        return (res[0] if res else None), (sup[0] if sup else None)

    c, h, l = df_sub["close"].values, df_sub["high"].values, df_sub["low"].values
    i0 = n - 1

    ema20 = _ema(c, 20); ema50 = _ema(c, 50)
    atr   = _atr14(df_sub)
    if not np.isfinite(atr) or atr <= 0: return ideas

    ph, pl = _pivots(df_sub, L=3, R=3)
    px, hi0, lo0 = float(c[i0]), float(h[i0]), float(l[i0])

    nearest_res, nearest_sup = _nearest_levels(df_sub, ph, pl, ref_px=px, max_lookback=800)
    near_thresh = 0.35 * atr
    sl_buf      = 0.40 * atr

    # ----- LONG -----
    if ema20[i0] > ema50[i0] and nearest_sup is not None and (px - nearest_sup) <= near_thresh and (px >= nearest_sup):
        entry = float(nearest_sup)
        sl    = float(entry - sl_buf)
        if lo0 > sl:
            risk = entry - sl
            if risk > 0:
                tp1 = entry + risk  # +1R
                if nearest_res is not None:
                    rr_to_level = (nearest_res - entry) / risk
                    tp2 = float(nearest_res) if rr_to_level >= 2.0 else float(entry + 2.0 * risk)
                else:
                    tp2 = float(entry + 2.0 * risk)
                rr2 = (tp2 - entry) / risk
                if rr2 >= 2.0:
                    ideas.append(TAIdea(
                        side="long", entry=entry, sl=sl, tp1=float(tp1), tp2=float(tp2), created_idx=i0,
                        note=f"Uptrend (EMA20>EMA50), near support @ {entry:.4f}; TP1=+1R, TP2=nearest R or 2R."
                    ))

    # ----- SHORT -----
    if ema20[i0] < ema50[i0] and nearest_res is not None and (nearest_res - px) <= near_thresh and (px <= nearest_res):
        entry = float(nearest_res)
        sl    = float(entry + sl_buf)
        if hi0 < sl:
            risk = sl - entry
            if risk > 0:
                tp1 = entry - risk  # +1R
                if nearest_sup is not None:
                    rr_to_level = (entry - nearest_sup) / risk
                    tp2 = float(nearest_sup) if rr_to_level >= 2.0 else float(entry - 2.0 * risk)
                else:
                    tp2 = float(entry - 2.0 * risk)
                rr2 = (entry - tp2) / risk
                if rr2 >= 2.0:
                    ideas.append(TAIdea(
                        side="short", entry=entry, sl=sl, tp1=float(tp1), tp2=float(tp2), created_idx=i0,
                        note=f"Downtrend (EMA20<EMA50), near resistance @ {entry:.4f}; TP1=+1R, TP2=nearest S or 2R."
                    ))

    # If we somehow created >2, keep at most one per side (closest to price)
    if len(ideas) > 2:
        longs  = [x for x in ideas if x.side == "long"]
        shorts = [x for x in ideas if x.side == "short"]
        pick = []
        if longs:  pick.append(min(longs,  key=lambda z: abs(px - z.entry)))
        if shorts: pick.append(min(shorts, key=lambda z: abs(px - z.entry)))
        return pick
    return ideas

# ---------- Simulation helpers ----------
def _bar_hits_level(high: float, low: float, level: float) -> bool:
    return (low <= level <= high)

def _simulate_limit_then_sl_tp(
    df: pd.DataFrame,
    idea: TAIdea,
    start_idx: int,
    expiry_bars: int = 30,
    max_hold_bars: int = 200,
    move_to_be_on_tp1: bool = True,
):
    """
    1) Wait for LIMIT fill at entry (within expiry_bars).
    2) After fill, if TP1 (+1R) is hit, move SL to entry from the NEXT bar.
    3) Exit on first touch of BE/SL or final TP2 (within max_hold_bars).

    Returns status in {"NoFill","BadRisk","TimedOut","SL","BE","TP"}.
    """
    side   = idea.side.lower()
    entry  = float(idea.entry)
    sl0    = float(idea.sl)
    tp1    = float(idea.tp1)
    tp2    = float(idea.tp2)
    i0     = int(start_idx) + 1
    n      = len(df)

    # 1) wait for fill
    fill_idx = None
    for i in range(i0, min(n, i0 + expiry_bars + 1)):
        hi = float(df["high"].iloc[i]); lo = float(df["low"].iloc[i])
        if _bar_hits_level(hi, lo, entry):
            fill_idx = i
            break
    if fill_idx is None:
        return {"status":"NoFill","filled":False,"fill_idx":None,"exit_idx":None,"exit_price":None,"R":0.0,"bars_held":0}

    # risk
    risk = abs(entry - sl0)
    if risk <= 0:
        return {"status":"BadRisk","filled":False,"fill_idx":fill_idx,"exit_idx":None,"exit_price":None,"R":0.0,"bars_held":0}

    # state
    be_armed = False
    be_armed_at = None
    sl_cur = sl0
    exit_idx = exit_px = None
    status = None

    for j in range(fill_idx, min(n, fill_idx + max_hold_bars + 1)):
        hi = float(df["high"].iloc[j]); lo = float(df["low"].iloc[j])

        if not be_armed and move_to_be_on_tp1:
            # Pre-BE: SL has priority if ambiguity occurs
            hit_sl  = _bar_hits_level(hi, lo, sl_cur)
            hit_tp2 = _bar_hits_level(hi, lo, tp2)
            hit_tp1 = _bar_hits_level(hi, lo, tp1)

            if hit_sl and (hit_tp1 or hit_tp2):
                status, exit_idx, exit_px = "SL", j, sl_cur; break
            if hit_sl:
                status, exit_idx, exit_px = "SL", j, sl_cur; break
            if hit_tp2:
                status, exit_idx, exit_px = "TP", j, tp2; break
            if hit_tp1:
                be_armed = True
                be_armed_at = j
                continue  # BE active from NEXT bar
        else:
            # Post-BE: SL is moved to entry from the bar AFTER tp1 was hit
            if j > be_armed_at:
                sl_cur = entry

            hit_be  = _bar_hits_level(hi, lo, sl_cur)  # entry as SL
            hit_tp2 = _bar_hits_level(hi, lo, tp2)

            # If both inside same bar, take BE (conservative)
            if hit_be and hit_tp2:
                status, exit_idx, exit_px = "BE", j, entry; break
            if hit_tp2:
                status, exit_idx, exit_px = "TP", j, tp2; break
            if hit_be:
                status, exit_idx, exit_px = "BE", j, entry; break

    if exit_idx is None:
        return {
            "status": "TimedOut", "filled": True, "fill_idx": fill_idx,
            "exit_idx": None, "exit_price": None, "R": 0.0,
            "bars_held": (min(n, fill_idx + max_hold_bars + 1) - fill_idx)
        }

    # R-multiple
    if side == "long":
        R = (exit_px - entry) / risk
    else:
        R = (entry - exit_px) / risk
    if status == "BE":
        R = 0.0

    return {
        "status": status,
        "filled": True,
        "fill_idx": fill_idx,
        "exit_idx": exit_idx,
        "exit_price": exit_px,
        "R": float(R),
        "bars_held": int(exit_idx - fill_idx),
        "be_armed": bool(be_armed),
        "be_armed_at": int(be_armed_at) if be_armed_at is not None else None,
    }

def _rolling_backtest(
    df: pd.DataFrame,
    rr_min: float = 2.0,
    expiry_bars: int = 30,
    max_hold_bars: int = 200,
    warmup_bars: int = 100,   # <- NEW
) -> pd.DataFrame:
    """
    Replays the dataset bar-by-bar, asking the adapter for ideas at each bar.
    warmup_bars: minimum completed bars before we begin generating ideas.
    """
    n = len(df)
    warmup_bars = int(max(0, warmup_bars))
    if n < warmup_bars + 3:
        return pd.DataFrame([])

    seen_keys = set()
    rows = []

    start_t = max(1, min(n - 3, warmup_bars))
    for t in range(start_t, n - 2):
        df_sub = df.iloc[:t+1].copy()
        new_ideas = _ideas_at_bar_adapter(df_sub)

        for idea in new_ideas:
            risk = abs(float(idea.entry) - float(idea.sl))
            if risk <= 0:
                continue

            # Support both the new (tp1/tp2) and old (tp) idea shapes
            tp2 = getattr(idea, "tp2", None)
            tp  = getattr(idea, "tp",  None)
            rr_planned = abs((tp2 if tp2 is not None else tp) - idea.entry) / risk
            if rr_planned < rr_min:
                continue

            key = (
                idea.side,
                round(float(idea.entry), 6),
                round(float(idea.sl), 6),
                round(float(getattr(idea, "tp1", getattr(idea, "tp", np.nan))), 6),
                round(float(tp2 if tp2 is not None else getattr(idea, "tp", np.nan)), 6),
                int(idea.created_idx),
            )
            if key in seen_keys:
                continue
            seen_keys.add(key)

            sim = _simulate_limit_then_sl_tp(
                df,
                idea,
                start_idx=t,
                expiry_bars=expiry_bars,
                max_hold_bars=max_hold_bars,
                # if your simulator doesn't take this kw yet, remove it here
                move_to_be_on_tp1=True
            )

            rows.append({
                "created_time": df["time"].iloc[int(idea.created_idx)] if "time" in df.columns else int(idea.created_idx),
                "created_idx": int(idea.created_idx),
                "side": idea.side,
                "entry": float(idea.entry),
                "sl": float(idea.sl),
                "tp1": float(getattr(idea, "tp1", np.nan)),
                "tp2": float(tp2 if tp2 is not None else getattr(idea, "tp", np.nan)),
                "rr_planned": round(float(rr_planned), 2),
                "note": getattr(idea, "note", ""),
                **sim,
                "exit_time": (df["time"].iloc[sim["exit_idx"]] if ("time" in df.columns and sim.get("exit_idx") is not None) else None)
            })

    return pd.DataFrame(rows)

def _compute_stats(trades: pd.DataFrame, r_cash: float = 1000.0) -> dict:
    if trades.empty:
        return {
            "trades": 0, "wins": 0, "losses": 0, "breakevens": 0,
            "win_rate": 0.0, "win_rate_excl_be": 0.0,
            "avg_R": 0.0, "avg_$": 0.0,
            "expectancy": 0.0, "expectancy_$": 0.0,
            "profit_factor": 0.0,
            "median_R": 0.0, "p25_R": 0.0, "p75_R": 0.0,
            "max_drawdown_R": 0.0, "max_drawdown_$": 0.0,
            "sharpe_R": 0.0,
            "cum_pnl_$": 0.0,
        }

    status_up = trades["status"].astype(str).str.upper()
    closed_all = trades[status_up.isin(["TP", "SL", "BE", "BREAKEVEN", "B/E"])].copy()
    closed_wo_be = trades[status_up.isin(["TP", "SL"])].copy()

    wins = (closed_all["R"] > 0).sum()
    losses = (closed_all["R"] < 0).sum()
    breakevens = len(closed_all) - wins - losses

    win_rate_all = wins / max(1, len(closed_all)) * 100.0
    win_rate_excl_be = wins / max(1, wins + losses) * 100.0

    avg_R = closed_all["R"].mean() if len(closed_all) else 0.0
    expectancy = avg_R

    gross_win = closed_wo_be.loc[closed_wo_be["R"] > 0, "R"].sum()
    gross_loss = -closed_wo_be.loc[closed_wo_be["R"] < 0, "R"].sum()
    profit_factor = (gross_win / gross_loss) if gross_loss > 0 else np.nan

    median_R = closed_all["R"].median() if len(closed_all) else 0.0
    p25 = closed_all["R"].quantile(0.25) if len(closed_all) else 0.0
    p75 = closed_all["R"].quantile(0.75) if len(closed_all) else 0.0

    # Equity and drawdown in R
    eq_R = closed_all.sort_values("exit_idx")["R"].cumsum().reset_index(drop=True)
    roll_max_R = eq_R.cummax()
    dd_R = eq_R - roll_max_R
    max_dd_R = dd_R.min() if len(dd_R) else 0.0

    sharpe = (closed_all["R"].mean() / (closed_all["R"].std(ddof=1) + 1e-12) * np.sqrt(12)) if len(closed_all) >= 2 else 0.0

    # $ conversions with fixed 1R = r_cash
    avg_usd = avg_R * r_cash
    expectancy_usd = expectancy * r_cash
    cum_pnl_usd = float(eq_R.iloc[-1] * r_cash) if len(eq_R) else 0.0
    max_dd_usd = max_dd_R * r_cash

    return {
        "trades": int(len(closed_wo_be)),
        "wins": int(wins),
        "losses": int(losses),
        "breakevens": int(breakevens),
        "win_rate": float(round(win_rate_all, 1)),
        "win_rate_excl_be": float(round(win_rate_excl_be, 1)),
        "avg_R": float(round(avg_R, 3)),
        "avg_$": float(round(avg_usd, 2)),
        "expectancy": float(round(expectancy, 3)),
        "expectancy_$": float(round(expectancy_usd, 2)),
        "profit_factor": float(round(profit_factor, 3)) if np.isfinite(profit_factor) else float("nan"),
        "median_R": float(round(median_R, 3)),
        "p25_R": float(round(p25, 3)),
        "p75_R": float(round(p75, 3)),
        "max_drawdown_R": float(round(max_dd_R, 3)),
        "max_drawdown_$": float(round(max_dd_usd, 2)),
        "sharpe_R": float(round(sharpe, 3)),
        "cum_pnl_$": float(round(cum_pnl_usd, 2)),
    }

def _plot_equity_curve(trades: pd.DataFrame, title: str = "Equity (R) Curve"):
    closed = trades[trades["status"].isin(["TP","SL","BE"])].sort_values("exit_idx")  # <-- include BE (0R step)
    if closed.empty:
        st.info("No closed trades to plot.")
        return
    eq = closed["R"].cumsum()
    fig, ax = plt.subplots(figsize=(7.4, 3.6))
    ax.plot(eq.values)
    ax.set_title(title)
    ax.set_xlabel("Closed trades (chronological)")
    ax.set_ylabel("Cumulative R")
    st.pyplot(fig)

# ---------- Streamlit UI hook (put this in your TA page’s content area) ----------
def render_ta_backtest_ui(df: pd.DataFrame, key_prefix: str):
    st.subheader("📈 Backtest (TA ideas) — Beta")
    st.caption("Replays historical bars, generates the ideas you *would have* recommended at each step, simulates limit fills, and records TP/SL outcomes. Only ideas with R/R ≥ 2 are included.")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        rr_min = st.number_input("Min R/R", value=2.0, step=0.5, min_value=1.0, format="%.1f", key=f"{key_prefix}_rr")
    with c2:
        expiry_bars = st.number_input("Idea expiry (bars)", value=30, step=5, min_value=1, key=f"{key_prefix}_exp")
    with c3:
        max_hold_bars = st.number_input("Max hold (bars)", value=200, step=10, min_value=10, key=f"{key_prefix}_hold")
    with c4:
        warmup_bars = st.number_input(                     # <-- NEW
            "Min bars before ideas",
            value=100, step=10, min_value=10,
            key=f"{key_prefix}_warmup"
        )

    if st.button("Run backtest", use_container_width=True, key=f"{key_prefix}_run"):
        with st.spinner("Running backtest over historical bars…"):
            trades = _rolling_backtest(df, rr_min=rr_min, expiry_bars=expiry_bars, max_hold_bars=max_hold_bars,warmup_bars=int(warmup_bars))

        if trades.empty:
            st.warning("No trades generated by the rules over the chosen data.")
            return

        stats = _compute_stats(trades)
        st.markdown("#### Summary")
        st.write({
            "Trades (TP/SL only)": stats["trades"],
            "Wins": stats["wins"],
            "Losses": stats["losses"],
            "Breakevens": stats["breakevens"],
            "Win rate %": stats["win_rate"],                 # includes BE in denom
            "Win rate % (W/(W+L))": stats["win_rate_excl_be"],
            "Avg R / $": f'{stats["avg_R"]} / ${stats["avg_$"]}',
            "Expectancy (R / $)": f'{stats["expectancy"]} / ${stats["expectancy_$"]}',
            "Cumulative P&L ($)": stats["cum_pnl_$"],
            "Profit factor": stats["profit_factor"],
            "Median R": stats["median_R"],
            "R p25 / p75": f'{stats["p25_R"]} / {stats["p75_R"]}',
            "Max drawdown (R / $)": f'{stats["max_drawdown_R"]} / ${stats["max_drawdown_$"]}',
            "Sharpe (per 12 trades)": stats["sharpe_R"],
        })

        _plot_equity_curve(trades, title="Equity (R) Curve")

        show_cols = ["created_time", "side", "entry", "sl", "tp1", "tp2",
             "rr_planned", "status", "R", "bars_held", "exit_time", "note"]
        st.markdown("#### Trades")
        st.dataframe(trades[show_cols].reset_index(drop=True), use_container_width=True, key=f"{key_prefix}_trades_df")

        st.download_button("⬇️ Download trades (CSV)",
                           data=trades.to_csv(index=False).encode("utf-8"),
                           file_name="ta_backtest_trades.csv",
                           mime="text/csv",
                           key=f"{key_prefix}_dl")


# ---------------------- Page setup ----------------------
st.set_page_config(page_title="Technical Analysis", layout="wide")

# ---------- Consistent canvas for ALL charts ----------
CANVAS_W_PX = 1100
CANVAS_H_PX = 620
CANVAS_DPI  = 120

SUPPORTED_TFS = ["Weekly","Daily","4H","1H","30m","15m"]

# Default ATR14 multiples per TF
ATR14_MULT = {
    "Weekly": 0.60, "Daily": 0.80, "4H": 1.00,
    "1H": 1.20, "30m": 1.40, "15m": 1.60,
}

def _safe_secret(key: str, default: str | None = None):
    try:
        # Accessing st.secrets triggers parsing; wrap it entirely
        return st.secrets[key]
    except Exception:
        return os.environ.get(key, default)
    
BYBIT_BASE = _safe_secret("BYBIT_BASE", "https://api.bybit.com")
TF_TO_BYBIT = {"Weekly": "W", "Daily": "D", "4H": "240", "1H": "60", "30m": "30", "15m": "15"}
ORDER_TFS = ["Weekly", "Daily", "4H", "1H", "30m", "15m"]


# Color palette for overlays
PALETTE = {
    "sr_support": "#3b82f6",
    "sr_resist":  "#ef4444",
    "range_lo":   "#9333ea",
    "range_hi":   "#9333ea",
    "range_mid":  "#6366f1",
    "entry":      "#0ea5e9",
    "sl":         "#ef4444",
    "tp":         "#22c55e",
    "sig_up":     "#10b981",
    "sig_dn":     "#f97316",
    "label_bg":   "#ffffffcc",
    "label_ec":   "#666666",
}

plt.rcParams.update({"font.size": 10, "axes.titlesize": 12, "axes.labelsize": 10, "legend.fontsize": 9})

# =========================
# ------- Data types ------
# =========================
@dataclass
class TAPlan:
    setup: str                 # e.g., "SR Flip Long", "Range Reversion Short", "SFP Long"
    entry: float
    sl: float
    tp1: Optional[float]
    tp2: Optional[float]
    rr1: Optional[float]
    rr2: Optional[float]
    reasons: List[str]
    signal_index: Optional[int] = None  # where the signal fired (index in df)

# =========================
# -------- Helpers --------
# =========================
def _pivot_masks(df: pd.DataFrame, left: int = 3, right: int = 3) -> tuple[np.ndarray, np.ndarray]:
    h, l = df["high"].values, df["low"].values
    n = len(df); sh = np.zeros(n, bool); sl = np.zeros(n, bool)
    for i in range(left, n - right):
        if all(h[i] > h[i - j] for j in range(1, left + 1)) and all(h[i] > h[i + j] for j in range(1, right + 1)):
            sh[i] = True
        if all(l[i] < l[i - j] for j in range(1, left + 1)) and all(l[i] < l[i + j] for j in range(1, right + 1)):
            sl[i] = True
    return sh, sl

def _dedupe_levels(levels: list[float], tol_ratio: float = 0.0015) -> list[float]:
    if not levels: return []
    levels = sorted(set(float(x) for x in levels if np.isfinite(x)))
    out = []
    for lv in levels:
        if not out or abs(lv - out[-1]) > max(abs(out[-1]), 1.0) * tol_ratio:
            out.append(lv)
    return out

def nearest_sr_levels(df: pd.DataFrame, *, left: int = 3, right: int = 3) -> tuple[list[float], list[float]]:
    """Return up to one Support and one Resistance (closest to last close)."""
    if len(df) < left + right + 3:
        return [], []
    sh, sl = _pivot_masks(df, left=left, right=right)
    pivot_highs = [float(df["high"].iloc[i]) for i in np.where(sh)[0]]
    pivot_lows  = [float(df["low"].iloc[i])  for i in np.where(sl)[0]]

    res_levels = _dedupe_levels(pivot_highs)
    sup_levels = _dedupe_levels(pivot_lows)

    last = float(df["close"].iloc[-1])
    # nearest above/below last close
    res_above = sorted([y for y in res_levels if y >= last], key=lambda y: y - last)
    sup_below = sorted([y for y in sup_levels if y <= last], key=lambda y: last - y)

    R = [res_above[0]] if res_above else ( [res_levels[0]] if res_levels else [] )
    S = [sup_below[0]] if sup_below else ( [sup_levels[-1]] if sup_levels else [] )
    return S[:1], R[:1]

def draw_nearest_sr(ax, df, dark: bool = False):
    S, R = nearest_sr_levels(df)
    if not (S or R): return
    x0, x1 = -0.5, len(df) - 0.5
    lab_fc = "#0E1117" if dark else "white"
    lab_ec = "#D0D4DA" if dark else "#333333"
    def _tag(y, text):
        ax.text(x1, y, text, va="bottom", ha="right", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.25", fc=lab_fc, ec=lab_ec, lw=1.0, alpha=0.98), zorder=60)
    for i, y in enumerate(S, 1):
        ax.hlines(y, x0, x1, colors="#2E8B57", linestyles="-", linewidth=1.6, zorder=55)
        _tag(y, f"S{i}")
    for i, y in enumerate(R, 1):
        ax.hlines(y, x0, x1, colors="#C23B22", linestyles="-", linewidth=1.6, zorder=55)
        _tag(y, f"R{i}")

def ta_is_still_valid(df: pd.DataFrame, plan: TAPlan, *, require_entry_touch: bool = False) -> bool:
    """Return True if the idea hasn’t been stopped or fully completed (TP2)
    after the signal formed. Optional: require a retest/touch of entry."""
    n = len(df)
    if n == 0:
        return False
    start = max(0, (plan.signal_index or 0))
    highs = df["high"].values[start:]
    lows  = df["low"].values[start:]
    if highs.size == 0 or lows.size == 0:
        # No bars after signal; treat as pending (valid)
        return True

    entry = float(plan.entry)
    sl    = float(plan.sl)
    tp2   = float(plan.tp2) if plan.tp2 is not None else None
    is_long = ("Long" in plan.setup)

    # Optional: require entry touch post-signal (pending until touched)
    if require_entry_touch:
        touched = np.any((lows <= entry) & (highs >= entry))
        if not touched:
            return True  # pending = still valid

    # Earliest SL hit
    sl_hits = np.where(lows <= sl)[0] if is_long else np.where(highs >= sl)[0]
    first_sl = int(sl_hits[0]) if sl_hits.size else None

    # Earliest TP2 hit (treat as “completed” if it happened first)
    first_tp2 = None
    if tp2 is not None:
        tp2_hits = np.where(highs >= tp2)[0] if is_long else np.where(lows <= tp2)[0]
        first_tp2 = int(tp2_hits[0]) if tp2_hits.size else None

    # If SL came first (or SL but no TP2) → invalid
    if first_sl is not None and (first_tp2 is None or first_sl <= first_tp2):
        return False
    # If TP2 came first (or TP2 but no SL) → completed
    if first_tp2 is not None and (first_sl is None or first_tp2 < first_sl):
        return False

    # Otherwise still valid (may have hit TP1 or be pending)
    return True

def infer_tf_from_name(name: str) -> Optional[str]:
    s = name.lower().replace(" ", "_")
    patterns = [
        (r"(?:^|_|-)(1w|w|weekly|week)(?:_|-|\.|$)", "Weekly"),
        (r"(?:^|_|-)(1d|d|daily|day)(?:_|-|\.|$)",   "Daily"),
        (r"(?:^|_|-)(4h|h4)(?:_|-|\.|$)",            "4H"),
        (r"(?:^|_|-)(240m?|240min)(?:_|-|\.|$)",     "4H"),
        (r"(?:^|_|-)(1h|h1)(?:_|-|\.|$)",            "1H"),
        (r"(?:^|_|-)(60m?|60min)(?:_|-|\.|$)",       "1H"),
        (r"(?:^|_|-)(30m?|m30)(?:_|-|\.|$)",         "30m"),
        (r"(?:^|_|-)(15m?|m15)(?:_|-|\.|$)",         "15m"),
    ]
    for pat, tf in patterns:
        if re.search(pat, s): return tf
    return None

def _parse_time_column(raw, *, dayfirst: Optional[bool] = None,
                       epoch_unit: Optional[str] = None,
                       force_utc: bool = True,
                       tz_naive: bool = True) -> pd.Series:
    s = pd.Series(raw)
    sn = pd.to_numeric(s, errors="coerce")

    if sn.notna().mean() > 0.95:
        vals = sn.dropna()
        med = float(vals.median()) if len(vals) else 0.0
        unit = None
        if epoch_unit:
            m = {"s":"s","seconds":"s","ms":"ms","milliseconds":"ms","us":"us","microseconds":"us","ns":"ns","nanoseconds":"ns","excel":"excel"}
            unit = m.get(epoch_unit.lower(), None)
        if unit is None:
            if 20000 < med < 80000: unit = "excel"
            elif med > 1e18: unit = "ns"
            elif med > 1e15: unit = "us"
            elif med > 1e10: unit = "ms"
            elif med > 1e9: unit = "s"
            else: unit = "s"
        if unit == "excel":
            dt_vals = pd.to_datetime(vals, unit="D", origin="1899-12-30", utc=force_utc, errors="coerce")
        else:
            dt_vals = pd.to_datetime(vals, unit=unit, utc=force_utc, errors="coerce")
        out = pd.Series(index=s.index, dtype="datetime64[ns, UTC]" if force_utc else "datetime64[ns]")
        out.loc[vals.index] = dt_vals
        dt = out  # <- FIX: ensure we return full-aligned series
    else:
        def try_parse(dfirst: bool):
            return pd.to_datetime(s, errors="coerce", utc=force_utc, dayfirst=dfirst, infer_datetime_format=True)
        if dayfirst is None:
            dt0 = try_parse(False)
            better = dt0
            if (dt0.isna().mean() > 0.2) or s.astype(str).str.contains(r"/").mean() > 0.2:
                dt1 = try_parse(True)
                if dt1.isna().mean() < dt0.isna().mean(): better = dt1
            dt = better
        else:
            dt = try_parse(dayfirst)
        if dt.isna().any():
            for fmt in ("%Y-%m-%d %H:%M:%S","%Y-%m-%d %H:%M","%m/%d/%Y %H:%M:%S","%d/%m/%Y %H:%M:%S",
                        "%m/%d/%Y %H:%M","%d/%m/%Y %H:%M","%Y-%m-%dT%H:%M:%S.%fZ","%Y-%m-%dT%H:%M:%S%z"):
                try:
                    dt2 = pd.to_datetime(s, format=fmt, errors="coerce", utc=force_utc)
                    dt = dt.fillna(dt2)
                except Exception:
                    pass

    if tz_naive and hasattr(dt.dt, "tz_localize"):
        dt = dt.dt.tz_convert("UTC").dt.tz_localize(None)
    return dt

def load_csv(file_like_or_bytes,
             *, dayfirst: Optional[bool] = None,
             epoch_unit: Optional[str] = None,
             tz_naive: bool = True) -> pd.DataFrame:
    if isinstance(file_like_or_bytes, (bytes, bytearray)):
        bio = io.BytesIO(file_like_or_bytes)
    else:
        try:
            data = file_like_or_bytes.getvalue()
        except Exception:
            try:
                data = file_like_or_bytes.read()
            except Exception:
                data = None
        if data is None:
            raise ValueError("Could not read uploaded file.")
        bio = io.BytesIO(data)

    df = pd.read_csv(bio)
    cols = {c.lower().strip(): c for c in df.columns}
    def pick(*names):
        for n in names:
            for k,v in cols.items():
                if k == n.lower(): return v
        return None

    ts_col = pick("timestamp","time","date","datetime","open time","time (utc)")
    o_col  = pick("open","o"); h_col  = pick("high","h"); l_col  = pick("low","l"); c_col  = pick("close","c")
    v_col  = pick("volume","v","vol")

    if ts_col is None or o_col is None or h_col is None or l_col is None or c_col is None:
        raise ValueError("CSV must include time, open, high, low, close (volume optional).")

    t = _parse_time_column(df[ts_col], dayfirst=dayfirst, epoch_unit=epoch_unit, force_utc=True, tz_naive=tz_naive)

    out = pd.DataFrame({
        "time":  t,
        "open":  pd.to_numeric(df[o_col], errors="coerce"),
        "high":  pd.to_numeric(df[h_col], errors="coerce"),
        "low":   pd.to_numeric(df[l_col], errors="coerce"),
        "close": pd.to_numeric(df[c_col], errors="coerce"),
        "volume": pd.to_numeric(df[v_col], errors="coerce") if v_col else np.nan,
    })
    out = out.dropna(subset=["time","open","high","low","close"]).drop_duplicates(subset="time").sort_values("time").reset_index(drop=True)
    return out

def pivots(df: pd.DataFrame, left=3, right=3) -> Tuple[np.ndarray, np.ndarray]:
    highs = df["high"].values
    lows  = df["low"].values
    n = len(df)
    swing_high = np.zeros(n, dtype=bool)
    swing_low  = np.zeros(n, dtype=bool)
    for i in range(left, n - right):
        if all(highs[i] > highs[i - j] for j in range(1, left + 1)) and \
           all(highs[i] > highs[i + j] for j in range(1, right + 1)):
            swing_high[i] = True
        if all(lows[i] < lows[i - j] for j in range(1, left + 1)) and \
           all(lows[i] < lows[i + j] for j in range(1, right + 1)):
            swing_low[i] = True
    return swing_high, swing_low

def atr(df: pd.DataFrame, period=14) -> pd.Series:
    h, l, c = df["high"], df["low"], df["close"].shift(1)
    tr = pd.concat([(h - l).abs(), (h - c).abs(), (l - c).abs()], axis=1).max(axis=1)
    return tr.rolling(period).mean().bfill()

from typing import List

def _bybit_klines_once(
    *,
    category: str,
    symbol: str,
    interval: str,
    start: int | None,
    end: int | None,
    limit: int = 1000,
) -> List[list]:
    url = f"{BYBIT_BASE}/v5/market/kline"
    params = {"category": category, "symbol": symbol, "interval": interval, "limit": str(limit)}
    if start is not None:
        params["start"] = str(start)
    if end is not None:
        params["end"] = str(end)

    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    data = r.json()

    if data.get("retCode") != 0:
        raise RuntimeError(f"Bybit error {data.get('retCode')}: {data.get('retMsg')}")

    return data.get("result", {}).get("list", []) or []


def get_bybit_ohlcv(category: str, symbol: str, interval: str, bars: int) -> pd.DataFrame:
    got: List[list] = []
    end_ms: int | None = None
    tries = 0

    page = min(1000, max(1, bars))
    while len(got) < bars and tries < 10:
        rows = _bybit_klines_once(
            category=category,
            symbol=symbol,
            interval=interval,
            start=None,
            end=end_ms,
            limit=page,
        )
        if not rows:
            break

        got.extend(rows)
        oldest = min(int(r[0]) for r in rows)
        end_ms = oldest - 1
        tries += 1
        time.sleep(0.12)

    if not got:
        return pd.DataFrame(columns=["time", "open", "high", "low", "close", "volume"])

    got = sorted(got, key=lambda r: int(r[0]))
    df = pd.DataFrame(got, columns=["start", "open", "high", "low", "close", "volume", "turnover"])

    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    ts = (
        pd.to_datetime(df["start"].astype(np.int64), unit="ms", utc=True)
          .dt.tz_convert("UTC")
          .dt.tz_localize(None)
    )

    out = (
        pd.DataFrame(
            {
                "time": ts,
                "open": df["open"],
                "high": df["high"],
                "low": df["low"],
                "close": df["close"],
                "volume": df["volume"],
            }
        )
        .dropna()
        .drop_duplicates(subset="time")
        .sort_values("time")
        .reset_index(drop=True)
    )

    if len(out) > bars:
        out = out.iloc[-bars:].reset_index(drop=True)

    return out


# =========================
# ---- Level extraction ----
# =========================
def ta_equal_price_clusters(level_prices: np.ndarray, tol_ratio: float=0.0015, min_touches: int=2) -> List[Tuple[float,int]]:
    """
    Cluster prices that are 'equal' within a tolerance ratio.
    Returns list of (level_price, touches) sorted by price.
    """
    if len(level_prices) == 0: return []
    vals = np.sort(level_prices)
    groups: List[List[float]] = []
    cur = [vals[0]]
    for x in vals[1:]:
        tol = max(abs(cur[-1]), 1.0) * tol_ratio
        if abs(x - cur[-1]) <= tol:
            cur.append(x)
        else:
            if len(cur) >= min_touches: groups.append(cur[:])
            cur = [x]
    if len(cur) >= min_touches: groups.append(cur[:])
    result = [(float(np.mean(g)), len(g)) for g in groups]
    result.sort(key=lambda t: t[0])
    return result

def ta_extract_levels(df: pd.DataFrame, left: int=3, right: int=3, tol_ratio: float=0.0015, min_touches: int=2):
    sh, sl = pivots(df, left=left, right=right)
    highs = df["high"].values
    lows  = df["low"].values
    resist = ta_equal_price_clusters(highs[sh], tol_ratio, min_touches)  # list[(price,touches)]
    supp   = ta_equal_price_clusters(lows[sl],  tol_ratio, min_touches)
    return supp, resist

def _next_levels_above_below(levels: List[Tuple[float,int]], price: float, above=True, n=2) -> List[float]:
    if above:
        vals = [lv for lv,_ in levels if lv > price]; vals.sort(); return vals[:n]
    vals = [lv for lv,_ in levels if lv < price]; vals.sort(reverse=True); return vals[:n]

def _closest_level_touch(df: pd.DataFrame, level: float, i0: int, i1: int) -> Optional[int]:
    highs = df["high"].values; lows = df["low"].values
    i1 = min(i1, len(df)-1)
    for i in range(i0, i1+1):
        if lows[i] <= level <= highs[i]:
            return i
    return None

def _rr(entry: float, sl: float, tp: float) -> Optional[float]:
    risk = abs(entry - sl)
    if risk <= 0: return None
    return round(abs(tp - entry) / risk, 2)

# =========================
# ---- Signal detection ----
# =========================
def ta_detect_sr_flip(df: pd.DataFrame, supp, resist, side: str, lookback: int, retest_window: int):
    """SR Flip: break through a level, then retest and accept on the other side."""
    c = df["close"].values
    end = len(df)-1
    start = max(1, end - lookback)

    if side == "long":
        for level,_s in reversed(resist):  # break ABOVE resistance -> retest -> accept
            i_break = None
            for i in range(start+1, end+1):
                if c[i-1] <= level < c[i]:
                    i_break = i; break
            if i_break is None: continue
            i_touch = _closest_level_touch(df, level, i_break+1, i_break+retest_window)
            if i_touch is None: continue
            if c[i_touch] >= level:
                return (i_touch, float(level), "SR Flip Long",
                        [f"Broke above resistance {level:.6f} then reclaimed on retest."])
    else:
        for level,_s in reversed(supp):  # break BELOW support -> retest -> reject
            i_break = None
            for i in range(start+1, end+1):
                if c[i-1] >= level > c[i]:
                    i_break = i; break
            if i_break is None: continue
            i_touch = _closest_level_touch(df, level, i_break+1, i_break+retest_window)
            if i_touch is None: continue
            if c[i_touch] <= level:
                return (i_touch, float(level), "SR Flip Short",
                        [f"Broke below support {level:.6f} then rejected on retest."])
    return None

def ta_detect_range_reversion(df: pd.DataFrame, lookback: int, dev_window: int=60):
    """
    Define range on lookback, then detect a deviation + close back inside.
    Returns: (i_sig, level(LO/HI), mid, (rng_low,rng_high), name, reasons)
    """
    end = len(df)-1; start = max(0, end - lookback)
    rng_low = float(df["low"].iloc[start:end+1].min())
    rng_high= float(df["high"].iloc[start:end+1].max())
    mid = (rng_low + rng_high)/2.0

    for i in range(max(start+1, end-dev_window), end+1):
        lo = float(df["low"].iloc[i]); hi = float(df["high"].iloc[i]); cl = float(df["close"].iloc[i])
        if lo < rng_low and cl > rng_low:
            return (i, rng_low, mid, (rng_low, rng_high), "Range Reversion Long",
                    [f"Deviation below range low {rng_low:.6f} then close back inside."])
        if hi > rng_high and cl < rng_high:
            return (i, rng_high, mid, (rng_low, rng_high), "Range Reversion Short",
                    [f"Deviation above range high {rng_high:.6f} then close back inside."])
    return None

def ta_detect_sfp(df: pd.DataFrame, left: int=3, right: int=3, lookback: int=200):
    """
    Swing Failure Pattern: take out a swing (wick through) but close back inside.
    Returns one of: (i_sig, level, "SFP Long"/"SFP Short", reasons)
    """
    sh, sl = pivots(df, left=left, right=right)
    highs = df["high"].values; lows = df["low"].values; close = df["close"].values
    end = len(df)-1; start = max(0, end - lookback)
    swing_high_idxs = np.where(sh)[0]
    swing_low_idxs  = np.where(sl)[0]

    # Bearish SFP (of swing high)
    for j in reversed(swing_high_idxs):
        if j <= start: break
        H = highs[j]
        for i in range(j+1, end+1):
            if highs[i] > H and close[i] < H:
                return (i, float(H), "SFP Short", [f"Wick through prior swing high {H:.6f} then close back below."])

    # Bullish SFP (of swing low)
    for j in reversed(swing_low_idxs):
        if j <= start: break
        L = lows[j]
        for i in range(j+1, end+1):
            if lows[i] < L and close[i] > L:
                return (i, float(L), "SFP Long", [f"Wick through prior swing low {L:.6f} then close back above."])
    return None

# =========================
# ---- Plan construction ---
# =========================
def calc_stop_buffer(df: pd.DataFrame, entry: float, side: str, mode: str,
                     abs_buf: float, pct_buf: float, atr_mult: float) -> float:
    if mode == "Absolute": return float(abs_buf)
    if mode == "% of entry": return float(entry) * (float(pct_buf)/100.0)
    a = float(atr(df, 14).iloc[-1])
    if np.isfinite(a) and a > 0: return float(atr_mult) * a
    # fallback
    return float(abs_buf) if abs_buf > 0 else float(entry) * 0.002

# --- Trade plan settings (safe defaults + pull from session if present)
target_policy    = st.session_state.get("ta_target_policy", "Opposing OB mids")
stop_buffer_mode = st.session_state.get("ta_stop_mode", "Absolute")               # "Absolute" | "% of entry" | "ATR14 × multiple"
stop_buffer_abs  = float(st.session_state.get("ta_stop_abs", 0.02))
stop_buffer_pct  = float(st.session_state.get("ta_stop_pct", 0.0))                # percent (not 0-1)
stop_buffer_atr  = float(st.session_state.get("ta_stop_atr", 1.0))

with st.sidebar:
    target_policy    = st.selectbox("Target policy", ["Opposing OB mids", "2R/3R", "Structure (swing highs/lows)"], index=0, key="ta_target_policy")
    stop_buffer_mode = st.selectbox("Stop buffer mode", ["Absolute", "% of entry", "ATR14 × multiple"], index=0, key="ta_stop_mode")
    if stop_buffer_mode == "Absolute":
        stop_buffer_abs = st.number_input("Absolute buffer", value=stop_buffer_abs, step=0.01, format="%.6f", key="ta_stop_abs")
        stop_buffer_pct = 0.0; stop_buffer_atr = 0.0
    elif stop_buffer_mode == "% of entry":
        stop_buffer_pct = st.number_input("% of entry", value=stop_buffer_pct, step=0.05, format="%.2f", key="ta_stop_pct")
        stop_buffer_abs = 0.0; stop_buffer_atr = 0.0
    else:
        stop_buffer_atr = st.number_input("ATR14 × multiple", value=stop_buffer_atr, step=0.05, format="%.2f", key="ta_stop_atr")
        stop_buffer_abs = 0.0; stop_buffer_pct = 0.0

def _targets_2r3r(side: str, entry: float, sl: float) -> tuple[float, float]:
    risk = abs(entry - sl)
    if risk <= 0:
        # fall back to 1R/2R to stay safe
        return (entry + (risk if side=="long" else -risk),
                entry + (2*risk if side=="long" else -2*risk))
    if side == "long":
        return (entry + 2.0 * risk, entry + 3.0 * risk)
    else:
        return (entry - 2.0 * risk, entry - 3.0 * risk)


def ta_build_plans(df: pd.DataFrame, *,
                   target_policy: str = "Structure (swing highs/lows)",
                   stop_mode: str="ATR14 × multiple", stop_abs: float=0.02, stop_pct: float=0.25, stop_atr: float=0.30,
                   left: int=3, right: int=3, tol_ratio: float=0.0015, min_touches: int=2,
                   lookback_levels: int=400, lookback_signals: int=240, retest_window: int=60):

    """
    Returns: (plans, ctx) where ctx holds:
      - supports: List[(price,touches)]
      - resistances: List[(price,touches)]
      - range: Optional[dict(low,high,mid)]
    """
    plans: List[TAPlan] = []

    # 1) Compute levels on a larger history window to get decent structure
    df_levels = df.tail(lookback_levels)
    supp, resist = ta_extract_levels(df_levels, left, right, tol_ratio, min_touches)

    # 2) SR flip long/short
    for side in ("long","short"):
        sr = ta_detect_sr_flip(df, supp, resist, side=side, lookback=lookback_signals, retest_window=retest_window)
        if sr:
            i_sig, level, name, why = sr
            entry = level
            buf   = calc_stop_buffer(df=df, entry=entry, side=("bullish" if side=="long" else "bearish"),
                                     mode=stop_mode, abs_buf=stop_abs, pct_buf=stop_pct, atr_mult=stop_atr)
            sl    = entry - buf if side=="long" else entry + buf
            if side == "long":
                if target_policy == "2R/3R":
                    tp1, tp2 = _targets_2r3r("long", entry, sl)
                else:
                    ups = _next_levels_above_below(resist, entry, above=True, n=2)
                    tp1 = ups[0] if len(ups)>=1 else (entry + 1.0*(entry-sl))
                    tp2 = ups[1] if len(ups)>=2 else (entry + 2.0*(entry-sl))
                    if tp2 < tp1: tp1, tp2 = tp2, tp1

            else:
                if target_policy == "2R/3R":
                    tp1, tp2 = _targets_2r3r("short", entry, sl)
                else:
                    downs = _next_levels_above_below(supp, entry, above=False, n=2)
                    tp1 = downs[0] if len(downs)>=1 else (entry - 1.0*(sl-entry))
                    tp2 = downs[1] if len(downs)>=2 else (entry - 2.0*(sl-entry))
                    if tp2 > tp1: tp1, tp2 = tp2, tp1

            plans.append(TAPlan(
                setup=name, entry=round(entry,6), sl=round(sl,6),
                tp1=round(float(tp1),6), tp2=round(float(tp2),6),
                rr1=_rr(entry, sl, float(tp1)), rr2=_rr(entry, sl, float(tp2)),
                reasons=why + [f"Stop beyond level ({stop_mode}). Targets = next S/R; 1R/2R fallback."],
                signal_index=i_sig
            ))

    # 3) Range deviation & reclaim
    range_ctx = None
    rng = ta_detect_range_reversion(df, lookback=lookback_signals, dev_window=60)
    if rng:
        i_sig, level, mid, (rng_low, rng_high), name, why = rng
        long_side = ("Long" in name)
        entry = level
        buf   = calc_stop_buffer(df=df, entry=entry, side=("bullish" if long_side else "bearish"),
                                 mode=stop_mode, abs_buf=stop_abs, pct_buf=stop_pct, atr_mult=stop_atr)
        sl    = entry - buf if long_side else entry + buf
        if target_policy == "2R/3R":
            tp1, tp2 = _targets_2r3r("long" if long_side else "short", entry, sl)
        else:
            tp1, tp2 = mid, (rng_high if long_side else rng_low)
            if long_side and tp2 < tp1: tp1, tp2 = tp2, tp1
            if not long_side and tp2 > tp1: tp1, tp2 = tp2, tp1
        if long_side and tp2 < tp1: tp1, tp2 = tp2, tp1
        if not long_side and tp2 > tp1: tp1, tp2 = tp2, tp1
        plans.append(TAPlan(
            setup=name, entry=round(entry,6), sl=round(sl,6),
            tp1=round(float(tp1),6), tp2=round(float(tp2),6),
            rr1=_rr(entry, sl, float(tp1)), rr2=_rr(entry, sl, float(tp2)),
            reasons=why + ["TP1 at midline, TP2 at opposite bound."],
            signal_index=i_sig
        ))
        range_ctx = {"low":rng_low, "high":rng_high, "mid":mid}

    # 4) SFP (optional additional idea)
    sfp = ta_detect_sfp(df, left=left, right=right, lookback=min(lookback_signals, 300))
    if sfp:
        i_sig, level, name, why = sfp
        long_side = ("Long" in name)
        entry = level
        buf   = calc_stop_buffer(df=df, entry=entry, side=("bullish" if long_side else "bearish"),
                                 mode=stop_mode, abs_buf=stop_abs, pct_buf=stop_pct, atr_mult=stop_atr)
        sl    = entry - buf if long_side else entry + buf
        if long_side:
            if target_policy == "2R/3R":
                tp1, tp2 = _targets_2r3r("long", entry, sl)
            else:
                ups = _next_levels_above_below(ta_equal_price_clusters(df["high"].values, tol_ratio, 2), entry, above=True, n=2)
                tp1 = ups[0] if len(ups)>=1 else (entry + 1.0*(entry-sl))
                tp2 = ups[1] if len(ups)>=2 else (entry + 2.0*(entry-sl))
                if tp2 < tp1: tp1, tp2 = tp2, tp1
        else:
            if target_policy == "2R/3R":
                tp1, tp2 = _targets_2r3r("short", entry, sl)
            else:
                downs = _next_levels_above_below(ta_equal_price_clusters(df["low"].values, tol_ratio, 2), entry, above=False, n=2)
                tp1 = downs[0] if len(downs)>=1 else (entry - 1.0*(sl-entry))
                tp2 = downs[1] if len(downs)>=2 else (entry - 2.0*(sl-entry))
                if tp2 > tp1: tp1, tp2 = tp2, tp1

        plans.append(TAPlan(
            setup=name, entry=round(entry,6), sl=round(sl,6),
            tp1=round(float(tp1),6), tp2=round(float(tp2),6),
            rr1=_rr(entry, sl, float(tp1)), rr2=_rr(entry, sl, float(tp2)),
            reasons=why + ["Stop beyond swept level; targets at next S/R."],
            signal_index=i_sig
        ))

    ctx = {"supports": supp, "resistances": resist, "range": range_ctx}
    return plans, ctx

# =========================
# ---- Charting (TA) ------
# =========================
def _mpf_style(dark: bool):
    if dark:
        mc = mpf.make_marketcolors(up="#3CCB7F", down="#FF6B6B", edge="inherit", wick="inherit", volume="inherit")
        return mpf.make_mpf_style(base_mpf_style="nightclouds", marketcolors=mc, facecolor="#0E1117",
                                  edgecolor="#0E1117", gridcolor="#2A2A2A",
                                  rc={"axes.labelcolor":"#EAECEE","xtick.color":"#EAECEE","ytick.color":"#EAECEE"})
    mc = mpf.make_marketcolors(up="#00A86B", down="#D62728", edge="inherit", wick="inherit", volume="inherit")
    return mpf.make_mpf_style(base_mpf_style="yahoo", marketcolors=mc)

def _label_right(ax, y: float, text: str, color: str):
    x_right = ax.get_xlim()[1]
    ax.text(x_right, y, f" {text} ", va="center", ha="left", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.2", fc=PALETTE["label_bg"], ec=PALETTE["label_ec"]),
            color=color)

def _nearest_two_sr_from_ctx(df: pd.DataFrame, ctx: Dict) -> Tuple[List[Tuple[float,int]], List[Tuple[float,int]]]:
    """Pick at most two supports (<= last) and two resistances (>= last), closest to last price."""
    last = float(df["close"].iloc[-1])
    supp = ctx.get("supports", [])
    resist = ctx.get("resistances", [])
    # supports below/equal last, nearest first
    s_cands = sorted([(lv,t) for lv,t in supp if lv <= last], key=lambda p: last - p[0])[:2]
    # resistances above/equal last, nearest first
    r_cands = sorted([(lv,t) for lv,t in resist if lv >= last], key=lambda p: p[0] - last)[:2]
    return s_cands, r_cands

def ta_clean_chart_png(df: pd.DataFrame, ctx: Dict, title: str, dark_charts: bool=False) -> bytes:
    """Clean chart with important lines (labelled), NO trade ideas."""
    dfp = df.copy().set_index("time")[["open","high","low","close"]]
    dfp = dfp.rename(columns={"open":"Open","high":"High","low":"Low","close":"Close"})
    style = _mpf_style(dark_charts)

    fig, axlist = mpf.plot(dfp, type="candle", style=style, volume=False, returnfig=True, figratio=(16,9), figscale=1.1)
    fig.set_size_inches(CANVAS_W_PX/CANVAS_DPI, CANVAS_H_PX/CANVAS_DPI); ax = axlist[0]
    pal = PALETTE
    x0 = -0.2; x1 = len(dfp) - 0.8

    # S/R lines: **at most two each, nearest to last price**
    supports, resistances = _nearest_two_sr_from_ctx(df, ctx)
    for i,(lvl, s) in enumerate(supports, 1):
        ax.hlines(lvl, x0, x1, colors=pal["sr_support"], linestyles=":", lw=1.4, alpha=0.95)
        _label_right(ax, lvl, f"Support #{i} (touches={s})", pal["sr_support"])
    for i,(lvl, s) in enumerate(resistances, 1):
        ax.hlines(lvl, x0, x1, colors=pal["sr_resist"],  linestyles=":", lw=1.4, alpha=0.95)
        _label_right(ax, lvl, f"Resistance #{i} (touches={s})", pal["sr_resist"])

    # Range bounds (if present)
    rng = ctx.get("range")
    if rng:
        ax.hlines(rng["low"],  x0, x1, colors=pal["range_lo"],  linestyles="--", lw=1.4);  _label_right(ax, rng["low"],  "Range Low",  pal["range_lo"])
        ax.hlines(rng["high"], x0, x1, colors=pal["range_hi"],  linestyles="--", lw=1.4);  _label_right(ax, rng["high"], "Range High", pal["range_hi"])
        ax.hlines(rng["mid"],  x0, x1, colors=pal["range_mid"], linestyles="-.", lw=1.2);  _label_right(ax, rng["mid"],  "Midline",    pal["range_mid"])

    # Axis cosmetics
    ax.set_title(title); ax.set_xlabel("Time"); ax.set_ylabel("Price")
    fig.subplots_adjust(left=0.065, right=0.995, top=0.92, bottom=0.12)
    locator = mdates.AutoDateLocator(minticks=4, maxticks=8); ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator)); fig.autofmt_xdate()

    out = io.BytesIO()
    fig.savefig(out, format="png", dpi=CANVAS_DPI, bbox_inches=None, pad_inches=0)
    plt.close(fig); out.seek(0)
    return out.getvalue()

def ta_idea_chart_png(df: pd.DataFrame, plan: TAPlan, ctx: Dict, title: str, dark_charts: bool=False) -> bytes:
    """Chart for a SINGLE idea (entry/SL/TPs), plus S/R + range context."""
    dfp = df.copy().set_index("time")[["open","high","low","close"]]
    dfp = dfp.rename(columns={"open":"Open","high":"High","low":"Low","close":"Close"})
    style = _mpf_style(dark_charts)

    fig, axlist = mpf.plot(dfp, type="candle", style=style, volume=False, returnfig=True, figratio=(16,9), figscale=1.1)
    fig.set_size_inches(CANVAS_W_PX/CANVAS_DPI, CANVAS_H_PX/CANVAS_DPI); ax = axlist[0]
    pal = PALETTE
    x0 = -0.2; x1 = len(dfp) - 0.8

    # Context lines (limit to at most two nearest S & R)
    supports    = sorted(ctx.get("supports", []), key=lambda t: -t[1])[:2]
    resistances = sorted(ctx.get("resistances", []), key=lambda t: -t[1])[:2]

    for (lvl, _s) in supports:    ax.hlines(lvl, x0, x1, colors=pal["sr_support"], linestyles=":", lw=1.0, alpha=0.8)
    for (lvl, _s) in resistances: ax.hlines(lvl, x0, x1, colors=pal["sr_resist"],  linestyles=":", lw=1.0, alpha=0.8)
    rng = ctx.get("range")
    if rng:
        ax.hlines(rng["low"],  x0, x1, colors=pal["range_lo"],  linestyles="--", lw=1.1)
        ax.hlines(rng["high"], x0, x1, colors=pal["range_hi"],  linestyles="--", lw=1.1)
        ax.hlines(rng["mid"],  x0, x1, colors=pal["range_mid"], linestyles="-.", lw=1.0)

    # Idea lines
    ax.hlines(plan.entry, x0, x1, colors=pal["entry"], linestyles="-",  lw=1.8)
    ax.hlines(plan.sl,    x0, x1, colors=pal["sl"],    linestyles="--", lw=1.6)
    if plan.tp1 is not None: ax.hlines(plan.tp1, x0, x1, colors=pal["tp"], linestyles=":",  lw=1.8)
    if plan.tp2 is not None: ax.hlines(plan.tp2, x0, x1, colors=pal["tp"], linestyles=":",  lw=1.2)
    if plan.signal_index is not None and 0 <= plan.signal_index < len(dfp):
        ax.axvline(plan.signal_index, color=(pal["sig_up"] if "Long" in plan.setup else pal["sig_dn"]),
                   linestyle="--", lw=1.2, alpha=0.9)
        ax.text(plan.signal_index, df["high"].iloc[-1], f"{plan.setup}",
                rotation=90, va="top", ha="center", fontsize=8,
                bbox=dict(boxstyle="round,pad=0.2", fc=pal["label_bg"], ec=pal["label_ec"]))

    # Labels on the right
    _label_right(ax, plan.entry, f"Entry ({plan.setup})", pal["entry"])
    _label_right(ax, plan.sl,    f"Stop", pal["sl"])
    if plan.tp1 is not None: _label_right(ax, plan.tp1, f"TP1 (R/R={plan.rr1})", pal["tp"])
    if plan.tp2 is not None and plan.rr2 is not None: _label_right(ax, plan.tp2, f"TP2 (R/R={plan.rr2})", pal["tp"])

    # Legend blocks
    from matplotlib.lines import Line2D
    legend_elems = [
        Line2D([0],[0], color=pal["entry"], lw=2.0, label="Entry"),
        Line2D([0],[0], color=pal["sl"],    lw=2.0, linestyle="--", label="Stop"),
        Line2D([0],[0], color=pal["tp"],    lw=2.0, linestyle=":",  label="Targets"),
    ]
    ax.legend(handles=legend_elems, loc="upper left", framealpha=0.96)

    ax.set_title(title); ax.set_xlabel("Time"); ax.set_ylabel("Price")
    fig.subplots_adjust(left=0.065, right=0.995, top=0.92, bottom=0.12)
    locator = mdates.AutoDateLocator(minticks=4, maxticks=8); ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator)); fig.autofmt_xdate()

    out = io.BytesIO()
    fig.savefig(out, format="png", dpi=CANVAS_DPI, bbox_inches=None, pad_inches=0)
    plt.close(fig); out.seek(0)
    return out.getvalue()

# =========================
# ---- Tables & rationales -
# =========================
def build_levels_table(ctx: Dict, last_price: float) -> pd.DataFrame:
    """Assemble a Key Levels (S/R) table with short rationales."""
    supp: List[Tuple[float,int]] = sorted(ctx.get("supports", []), key=lambda t: abs(t[0]-last_price))[:5]
    res:  List[Tuple[float,int]] = sorted(ctx.get("resistances", []), key=lambda t: abs(t[0]-last_price))[:5]
    rng = ctx.get("range")
    rows = []
    def rationale(kind: str, lvl: float, touches: int) -> str:
        bits = [f"Well-defined {kind} ({touches} touches)"]
        # proximity to last price
        rel = "below" if lvl < last_price else "above"
        bits.append(f"{abs((lvl-last_price)/max(abs(last_price),1e-9))*100:.1f}% {rel} last price")
        # range context
        if rng:
            h = rng["high"] - rng["low"]
            if h > 0:
                if abs(lvl - rng["low"]) / h < 0.1: bits.append("near Range Low")
                if abs(lvl - rng["high"]) / h < 0.1: bits.append("near Range High")
                if abs(lvl - rng["mid"]) / h < 0.05: bits.append("near Midline")
        return "; ".join(bits)
    for lvl, t in supp:
        rows.append({"Type":"Support","Level":round(lvl,6),"Touches":t,"Explanation":rationale("support", lvl, t)})
    for lvl, t in res:
        rows.append({"Type":"Resistance","Level":round(lvl,6),"Touches":t,"Explanation":rationale("resistance", lvl, t)})
    df = pd.DataFrame(rows)
    if not df.empty:
        # Order: by Type then by distance to price
        df["Dist"] = (df["Level"] - last_price).abs()
        df = df.sort_values(["Type","Dist"]).drop(columns=["Dist"]).reset_index(drop=True)
    return df

def build_ideas_table(plans: List[TAPlan], min_rr: float) -> pd.DataFrame:
    rows = []
    for p in plans:
        expl = "; ".join(p.reasons)
        rr_ok = (p.rr1 is not None and p.rr1 >= float(min_rr))
        if rr_ok:
            rows.append({
                "Setup": p.setup,
                "Entry": p.entry, "SL": p.sl,
                "TP1": p.tp1, "R/R1": p.rr1,
                "TP2": p.tp2, "R/R2": p.rr2,
                "Explanation": expl
            })
    return pd.DataFrame(rows)

# =========================
# ------- Sidebar UI ------
# =========================
st.sidebar.header("Technical Analysis")

# ---- Data source selector ----
source = st.sidebar.radio(
    "Data source",
    ["Bybit (live)", "CSV uploads"],
    index=0,
    help="Use Bybit public API for live OHLCV, or upload your own CSVs."
)

if source == "Bybit (live)":
    st.sidebar.subheader("Bybit")
    category = st.sidebar.selectbox(
        "Market",
        ["perps", "spot"],
        index=0,
        help="perps = USDT/USDC perps; spot = cash market."
    )
    symbol = st.sidebar.text_input(
        "Symbol",
        value="BTCUSDT",
        help="e.g., BTCUSDT, ETHUSDT, SOLUSDT"
    )
    per_tf_bars = st.sidebar.number_input(
        "Bars per timeframe",
        min_value=200,
        max_value=2000,
        value=1000,
        step=100,
        help="History depth fetched for each TF (Weekly→15m)."
    )
else:
    if source == "CSV uploads":
        stored_uploads.clear()  # reset

    bulk_files = st.sidebar.file_uploader(
        "Bulk CSV upload (weekly / daily / 4h / 1h / 30m / 15m in file name)",
        type=["csv"], accept_multiple_files=True
    )

    # Optional per-TF overrides
    with st.sidebar.expander("Optional per-TF overrides"):
        w_csv  = st.file_uploader("Weekly · 1W", type=["csv"], key="csv_w")
        d_csv  = st.file_uploader("Daily · 1D",  type=["csv"], key="csv_d")
        h4_csv = st.file_uploader("4H · 240m",   type=["csv"], key="csv_4h")
        h1_csv = st.file_uploader("1H · 60m",    type=["csv"], key="csv_1h")
        m30_csv= st.file_uploader("30m",         type=["csv"], key="csv_30m")
        m15_csv= st.file_uploader("15m",         type=["csv"], key="csv_15m")

    # Fill stored_uploads only when CSV mode is active
    unmatched_files = []
    if bulk_files:
        for i, uf in enumerate(bulk_files):
            tf = infer_tf_from_name(uf.name)
            b = _bytes(uf)
            if not b:
                continue
            if tf:
                stored_uploads[tf] = b
            else:
                unmatched_files.append((i, uf))

    # apply per-TF overrides last so they win
    for tf, uf in {
        "Weekly": w_csv, "Daily": d_csv, "4H": h4_csv,
        "1H": h1_csv, "30m": m30_csv, "15m": m15_csv
    }.items():
        b = _bytes(uf) if uf else None
        if b:
            stored_uploads[tf] = b



def _bytes(uf):
    if not uf: return None
    try: return uf.getvalue()
    except Exception:
        try: return uf.read()
        except Exception: return None

stored_uploads: Dict[str, bytes] = {}
unmatched_files = []

if bulk_files:
    for i, uf in enumerate(bulk_files):
        tf = infer_tf_from_name(uf.name)
        b = _bytes(uf)
        if not b: continue
        if tf: stored_uploads[tf] = b
        else: unmatched_files.append((i, uf))

if unmatched_files:
    st.sidebar.info("Some files need a timeframe mapping:")
    for i, uf in unmatched_files:
        sel = st.sidebar.selectbox(f"Timeframe for {uf.name}", SUPPORTED_TFS, index=None, placeholder="Choose…", key=f"map_{i}")
        if sel:
            b = _bytes(uf)
            if b: stored_uploads[sel] = b

# Single TF overrides take precedence
for tf, uf in {"Weekly":w_csv,"Daily":d_csv,"4H":h4_csv,"1H":h1_csv,"30m":m30_csv,"15m":m15_csv}.items():
    b = _bytes(uf)
    if b: stored_uploads[tf] = b

# Time parsing options
with st.sidebar.expander("Time parsing"):
    dayfirst_ui = st.checkbox("DD/MM/YYYY", value=False)
    epoch_ui = st.selectbox("Numeric time column is…",
                            ["Auto-detect","seconds","milliseconds","microseconds","nanoseconds","Excel serial days"], index=0)
    tz_naive_ui = st.checkbox("Make UTC tz-naive (recommended for charts)", value=True)
epoch_map = {"Auto-detect":None,"seconds":"s","milliseconds":"ms","microseconds":"us","nanoseconds":"ns","Excel serial days":"excel"}
epoch_kw = epoch_map[epoch_ui]

pair = st.sidebar.text_input("Pair symbol", value="XRPUSD")
dark_charts = st.sidebar.checkbox("Dark charts", value=False)

# TA Parameters
st.sidebar.subheader("TA Parameters")
ta_left   = st.sidebar.slider("Pivot left (levels)", 2, 6, 3)
ta_right  = st.sidebar.slider("Pivot right (levels)", 2, 6, 3)
ta_min_t  = st.sidebar.slider("Min touches to form a level", 1, 5, 2)
ta_tol_ppm= st.sidebar.slider("Equal-level tolerance (‰ of price)", 0.5, 4.0, 1.5, 0.1)
ta_tol    = ta_tol_ppm / 1000.0
ta_lookL  = st.sidebar.slider("Lookback for levels (bars)", 150, 800, 400, 10)
ta_lookS  = st.sidebar.slider("Lookback for signals (bars)", 60, 400, 240, 10)
ta_retest = st.sidebar.slider("Retest window after breakout (bars)", 10, 150, 60, 5)

# Stops
st.sidebar.subheader("Stops")
stop_mode = st.sidebar.radio("Stop mode", ["Absolute","% of entry","ATR14 × multiple"], horizontal=True, index=2)
if stop_mode == "Absolute":
    stop_abs = st.sidebar.number_input("Absolute buffer", value=0.02, step=0.01, format="%.6f"); stop_pct = 0.0; stop_atr = 0.0
elif stop_mode == "% of entry":
    stop_pct = st.sidebar.number_input("Percent of entry (%)", value=0.25, step=0.05, format="%.2f"); stop_abs = 0.0; stop_atr = 0.0

# Validation
st.sidebar.subheader("Validation")
min_rr_required = st.sidebar.number_input("Minimum R/R (using TP1)", value=2.0, step=0.5, min_value=1.0)
min_bars_ta = st.sidebar.number_input(
    "Min bars for TA (not backtest)", value=80, min_value=20, step=10
)

# =========================
# --------- Body ----------
# =========================
st.title("📊 Technical Analysis")
st.caption("Clean chart with labelled S/R & range; separate per-idea charts for valid setups (R/R ≥ threshold). Includes per-TF data tables with explanations. Backtesting is available per timeframe. Based on rules found in TA video by CryptoCred https://youtu.be/azB9Q_9MYsI?si=GPzjDs_tgB4SBAdv")

# seed containers for parsed data
if "ta_dfs" not in st.session_state: st.session_state.ta_dfs = None
if "ta_tfs" not in st.session_state: st.session_state.ta_tfs = None


# --- Load on click ---
# --- Load on click ---
if st.button("Generate (Technical Analysis)"):
    try:
        if source == "Bybit (live)":
            st.info(f"Fetching Bybit data for {symbol} ({category})…")
            dfs: Dict[str, pd.DataFrame] = {}
            errs: list[str] = []
            for tf in ORDER_TFS:
                try:
                    dfs[tf] = get_bybit_ohlcv(
                        category=category,
                        symbol=symbol,
                        interval=TF_TO_BYBIT[tf],
                        bars=int(per_tf_bars),
                    )
                except Exception as e:
                    errs.append(f"{tf}: {e}")
                    dfs[tf] = pd.DataFrame(columns=["time","open","high","low","close","volume"])
            # Keep only TFs that returned data
            dfs = {tf: df for tf, df in dfs.items() if not df.empty}
            if not dfs:
                st.error("No data returned from Bybit. Check symbol/market.")
                st.stop()
            ordered = [tf for tf in ORDER_TFS if tf in dfs]
            st.session_state.ta_dfs = dfs
            st.session_state.ta_tfs = ordered
            if errs:
                st.warning("Some timeframes failed:\n" + "\n".join(errs))
            st.success("Live data loaded. Scroll down to view analysis and run backtests.")
        else:
            # CSV path (original behavior)
            if not stored_uploads:
                st.error("Upload at least one CSV (Weekly/Daily/4H/1H/30m/15m) in the sidebar.")
                st.stop()
            dfs = {
                tf: load_csv(b,
                             dayfirst=st.session_state.get("dayfirst_ui", False),
                             epoch_unit=epoch_kw,  # your mapping is fine
                             tz_naive=st.session_state.get("tz_naive_ui", True))
                for tf, b in stored_uploads.items()
            }
            st.session_state.ta_dfs = dfs
            st.session_state.ta_tfs = [tf for tf in ORDER_TFS if tf in dfs]
            st.success("CSV data loaded. Scroll down to view analysis and run backtests.")
    except Exception as e:
        st.error(f"Data load error: {e}")
        st.stop()



# --- Render on every run if we have data (KEEP ONLY THIS ONE COPY) ---
dfs = st.session_state.ta_dfs
ordered_tfs = st.session_state.ta_tfs
export_rows = []


if dfs and ordered_tfs:
    st.caption("Detected timeframes: " + ", ".join(ordered_tfs))

    # 🔽 Pick ONE timeframe to display
    selected_tf = st.selectbox(
        "Timeframe",
        options=ordered_tfs,
        index=0,
        key="ta_tf_select",
        help="Choose which timeframe to analyze and backtest."
    )

    tf = selected_tf
    st.subheader(tf)
    df = dfs[tf]

    if len(df) < int(min_bars_ta):
        st.info(f"Not enough bars for {tf} (need ≥ {int(min_bars_ta)}).")
        st.stop()

    # Per-TF ATR input (lives in the sidebar, unique key per TF)
    if stop_mode == "ATR14 × multiple":
        default_atr = ATR14_MULT.get(tf, 1.0)
        stop_atr_tf = st.sidebar.number_input(
            f"ATR14 multiple — {tf}",
            value=float(default_atr),
            step=0.05,
            format="%.2f",
            key=f"atr_{tf}"
        )
    else:
        stop_atr_tf = 0.0

    # 👉 Everything below is exactly what you had inside the old loop body
    # (build plans, plot main chart, per-idea charts/tables, backtest for this TF, etc.)
    plans, ctx = ta_build_plans(
        df,
        target_policy=target_policy,
        stop_mode=stop_buffer_mode,                                  # <- correct name
        stop_abs=stop_buffer_abs if stop_buffer_mode == "Absolute" else 0.0,
        stop_pct=stop_buffer_pct if stop_buffer_mode == "% of entry" else 0.0,
        stop_atr=stop_buffer_atr if stop_buffer_mode == "ATR14 × multiple" else 0.0,
        left=ta_left, right=ta_right, tol_ratio=ta_tol, min_touches=ta_min_t,
        lookback_levels=ta_lookL, lookback_signals=ta_lookS, retest_window=ta_retest
    )
    last_price = float(df["close"].iloc[-1])

    # Key Levels table
    st.markdown("**Key Levels (S/R)** — highest-confidence clusters (sorted by proximity).")
    levels_df = build_levels_table(ctx, last_price)
    if not levels_df.empty:
        st.dataframe(levels_df, use_container_width=True)
    else:
        st.info("No strong S/R clusters found under current parameters.")

    # Clean chart (no ideas)
    clean_img = ta_clean_chart_png(df, ctx, f"{pair} — {tf} (Clean: S/R + Range)", dark_charts=dark_charts)
    st.image(clean_img, use_container_width=True)
    st.download_button(f"⬇️ {tf} clean chart (PNG)", data=clean_img, file_name=f"{pair}_{tf}_TA_clean.png")

    # Valid ideas table
    valid_plans = [
        p for p in plans
        if (p.rr1 is not None and p.rr1 >= float(min_rr_required) and ta_is_still_valid(df, p, require_entry_touch=True))
    ]
    st.markdown(f"**Valid Trade Ideas (R/R ≥ {min_rr_required})** — each includes a brief explanation of why it’s attractive.")
    ideas_df = build_ideas_table(valid_plans, min_rr=min_rr_required)
    if not ideas_df.empty:
        st.dataframe(ideas_df, use_container_width=True)
    else:
        st.info(f"No valid ideas (R/R ≥ {min_rr_required}) under current parameters.")

    # Per-idea charts
    for i, p in enumerate(valid_plans, 1):
        idea_img = ta_idea_chart_png(df, p, ctx, f"{pair} — {tf} — Idea #{i}: {p.setup}", dark_charts=dark_charts)
        st.image(idea_img, use_container_width=True, caption=f"Idea #{i}: {p.setup} — RR1={p.rr1}, RR2={p.rr2}")
        st.download_button(
            f"⬇️ {tf} idea #{i} (PNG)",
            data=idea_img,
            file_name=f"{pair}_{tf}_TA_idea_{i}.png"
        )
        export_rows.append({
            "TF": tf, "Setup": p.setup,
            "Entry": p.entry, "SL": p.sl,
            "TP1": p.tp1, "R/R1": p.rr1,
            "TP2": p.tp2, "R/R2": p.rr2,
            "Why": "; ".join(p.reasons)
        })

    # Backtest for this TF — unique key per TF so widgets don’t collide
    with st.expander(f"Backtest this timeframe ({tf})", expanded=False):
        render_ta_backtest_ui(df, key_prefix=f"bt_{tf}")

    st.divider()

if export_rows:
    out = pd.DataFrame(export_rows)
    st.download_button("⬇️ Export Technical Analysis (valid ideas only, CSV)",
                    data=out.to_csv(index=False).encode("utf-8"),
                    file_name=f"{pair}_technical_analysis_valid_ideas.csv",
                    mime="text/csv")
else:
    st.info("Load data with **Generate (Technical Analysis)** in the header first.")


    

