from __future__ import annotations
# pages/level_mapping_strategy.py
# Level Mapping Strategy — Trade Calculation Correct + Visuals + Entry Styles
# ---------------------------------------------------------------------------
# What this page includes:
# - Correct, transparent trade math (all knobs passed via `params`)
# - backtest_xo_logic(..., force_eod_close=False) -> (trades_df, summary, open_pos, pending_list)
# - Entry styles: Stop-through, Limit-on-retest, Market-on-trigger
# - TP1 = FULL EXIT at TP1 (simple & conservative)
# - Optional end-of-data close (off by default) to allow active positions
# - Active (open) + Pending tables
# - Per-symbol charts: Cumulative R & R distribution
# - All-symbol charts: Total R per symbol (bar), Cumulative R over time (daily)
# - Diagnostics expander to see last 10 bars + regimes + levels
#
# Notes:
# - This page expects a host app function `load_ohlcv(symbol, tf, limit=..., category=...)`
#   that returns OHLCV with either a DatetimeIndex or a 'time' column (UTC).



from matplotlib.ticker import MaxNLocator
from matplotlib.dates import AutoDateLocator, ConciseDateFormatter

import math
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
import pandas as pd
import streamlit as st
import html as _html

st.set_page_config(page_title="Level Mapping Strategy — XO", layout="wide")

# Optional third-party dependency (safe import)
try:
    import requests
except Exception:
    requests = None

# ---- Optional hook to host data loader ----
try:
    from mtf_ob_report_app import load_ohlcv as _load_ohlcv
    load_ohlcv = _load_ohlcv
except Exception:
    # Define a placeholder fallback so the app still runs without crashing
    def load_ohlcv(symbol, tf, limit=1500, category="linear"):
        raise RuntimeError("load_ohlcv() not found. Ensure mtf_ob_report_app.py is in the same directory.")


# ----------------- Small helpers -----------------

def _ema(series, span):
    return series.ewm(span=span, adjust=False).mean()

def htf_ema_state(df: pd.DataFrame) -> dict:
    def _state(x: pd.DataFrame) -> dict:
        if x is None or x.empty:
            return {"bull_ok": False, "bear_cross": True}

        e12 = _ema(x["close"], 12)
        e25 = _ema(x["close"], 25)

        bull_ok = (x["close"].iloc[-1] > e12.iloc[-1]) and (x["close"].iloc[-1] > e25.iloc[-1])

        # bearish cross event on the last bar (needs at least 2 points)
        bear_cross = False
        if len(x) >= 2:
            bear_cross = (e12.iloc[-2] > e25.iloc[-2]) and (e12.iloc[-1] <= e25.iloc[-1])

        return {"bull_ok": bool(bull_ok), "bear_cross": bool(bear_cross)}

    # Ensure resample-safe index (cheap)
    g = df.copy()
    if not isinstance(g.index, pd.DatetimeIndex):
        g.index = pd.to_datetime(g.index, utc=True)
    else:
        g.index = pd.to_datetime(g.index, utc=True)

    w  = _resample_ohlc(g, "7D")  # or "W-SUN" if you want calendar weeks
    d3 = _resample_ohlc(g, "3D")

    return {"weekly": _state(w), "d3": _state(d3)}


def _esc(x: str) -> str:
    return _html.escape(str(x), quote=True)

def _period_minutes(tf: str) -> Optional[int]:
    return {"5m":5,"15m":15,"30m":30,"1h":60,"4h":240,"12h":720,"1d":1440,"1W":10080}.get(tf)

def timeframe_to_pandas_freq(tf: str) -> str:
    return {"5m":"5T","15m":"15T","30m":"30T","1h":"1H","4h":"4H","12h":"12H","1d":"1D","1W":"7D"}.get(tf, "1H")

def resample_ohlcv(df: pd.DataFrame, tf: str) -> pd.DataFrame:
    if df.empty:
        return df
    freq = timeframe_to_pandas_freq(tf)
    di = pd.DatetimeIndex(pd.to_datetime(df.index, utc=True))
    g = df.copy()
    g.index = di
    o = g["open"].resample(freq, label="right", closed="right").first()
    h = g["high"].resample(freq, label="right", closed="right").max()
    l = g["low"].resample(freq, label="right", closed="right").min()
    c = g["close"].resample(freq, label="right", closed="right").last()
    v = g["volume"].resample(freq, label="right", closed="right").sum()
    out = pd.DataFrame({"open":o,"high":h,"low":l,"close":c,"volume":v}).dropna()
    return out

def drop_incomplete_bar(df: pd.DataFrame, tf: str) -> pd.DataFrame:
    """Only drop the last row if its timestamp is in the future (UTC)."""
    if df.empty:
        return df
    last_ts = pd.to_datetime(df.index[-1])
    if last_ts.tz is None:
        last_ts = last_ts.tz_localize("UTC")
    else:
        last_ts = last_ts.tz_convert("UTC")
    now = pd.Timestamp.now(tz="UTC")
    if last_ts > now:
        return df.iloc[:-1].copy()
    return df

def ema_regime(df: pd.DataFrame, fast: int = 12, slow: int = 21) -> pd.DataFrame:
    e1 = df["close"].ewm(span=fast, adjust=False).mean()
    e2 = df["close"].ewm(span=slow, adjust=False).mean()
    spread = e1 - e2
    slope = spread.diff()
    angle = np.degrees(np.arctan((slope / np.maximum(df["close"], 1e-9)).rolling(3).mean()))
    regime = np.where((e1>e2) & (angle>5), "uptrend",
             np.where((e1<e2) & (angle<-5), "downtrend", "range"))
    out = df.copy()
    out["ema12"], out["ema21"] = e1, e2
    out["ema_angle_deg"], out["regime"] = angle, regime
    return out


def _resample_ohlc(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    o = df["open"].resample(rule, label="right", closed="right").first()
    h = df["high"].resample(rule, label="right", closed="right").max()
    l = df["low"].resample(rule, label="right", closed="right").min()
    c = df["close"].resample(rule, label="right", closed="right").last()
    v = df["volume"].resample(rule, label="right", closed="right").sum()
    return pd.DataFrame({"open":o,"high":h,"low":l,"close":c,"volume":v}).dropna()




def volume_profile(df: pd.DataFrame, n_bins: int = 240) -> pd.DataFrame:
    lo, hi = df["close"].min(), df["close"].max()
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return pd.DataFrame({"price":[],"vol":[]})
    bins = np.linspace(lo, hi, n_bins+1)
    idx = np.digitize(df["close"].values, bins) - 1
    idx = np.clip(idx, 0, len(bins)-2)
    vol_hist = np.zeros(len(bins)-1, dtype=float)
    np.add.at(vol_hist, idx, df["volume"].values)
    centers = (bins[:-1]+bins[1:])/2
    return pd.DataFrame({"price": centers, "vol": vol_hist})

def value_area(vprof: pd.DataFrame, coverage: float = 0.70) -> Tuple[float, float, float]:
    if vprof.empty:
        return np.nan, np.nan, np.nan
    vp = vprof.sort_values("price").reset_index(drop=True)
    total = vp["vol"].sum()
    if total <= 0:
        return np.nan, np.nan, np.nan
    poc_i = int(vp["vol"].values.argmax())
    used = float(vp["vol"].iloc[poc_i])
    L, R = poc_i-1, poc_i+1
    chosen = {poc_i}
    while used < coverage*total and (L>=0 or R<len(vp)):
        left = vp["vol"].iloc[L] if L>=0 else -1
        right = vp["vol"].iloc[R] if R<len(vp) else -1
        if right>=left:
            chosen.add(R); used += max(right,0); R+=1
        else:
            chosen.add(L); used += max(left,0); L-=1
    vah = vp["price"].iloc[max(chosen)]
    val = vp["price"].iloc[min(chosen)]
    poc = vp["price"].iloc[poc_i]
    return val, poc, vah

def composite_profile(df, days=35, n_bins=240, va_cov: float = 0.68) -> Dict[str, Any]:
    window = df[df.index >= (df.index.max() - pd.Timedelta(days=days))].copy()
    if window.empty:
        return {"VAL": np.nan, "POC": np.nan, "VAH": np.nan, "hist": pd.DataFrame({"price":[],"vol":[]})}
    vp = volume_profile(window, n_bins=n_bins)
    val, poc, vah = value_area(vp, va_cov)
    return {"VAL": val, "POC": poc, "VAH": vah, "hist": vp}

@st.cache_data(show_spinner=False)
def rolling_composite_profile_cached(df, days, n_bins, min_bars, va_cov):
    return rolling_composite_profile(df, days=days, n_bins=n_bins, min_bars=min_bars, va_cov=va_cov)


def rolling_composite_profile(
    df: pd.DataFrame,
    days: int = 35,
    n_bins: int = 240,
    min_bars: int = 20,
    va_cov: float = 0.68,
) -> pd.DataFrame:
    """
    Per-bar composite profile using ONLY data up to each bar's timestamp.
    Returns DataFrame with index == df.index and columns ['VAL','POC','VAH'].
    """
    out = pd.DataFrame(index=df.index, columns=["VAL", "POC", "VAH"], dtype=float)
    if df.empty:
        return out

    idx = df.index
    lookback = pd.Timedelta(days=days)

    for i, ts in enumerate(idx):
        start_ts = ts - lookback
        j0 = idx.searchsorted(start_ts)
        window = df.iloc[j0:i+1]
        if len(window) < min_bars:
            continue

        vp = volume_profile(window, n_bins=n_bins)
        val, poc, vah = value_area(vp, va_cov)
        out.at[ts, "VAL"] = float(val)
        out.at[ts, "POC"] = float(poc)
        out.at[ts, "VAH"] = float(vah)

    return out


def single_prints(vprof: pd.DataFrame, q: float = 0.15) -> List[Tuple[float,float]]:
    if vprof.empty:
        return []
    thresh = vprof["vol"].quantile(q)
    low = vprof[vprof["vol"]<=thresh].copy()
    if low.empty:
        return []
    prices = low["price"].sort_values().values
    blocks, cur = [], []
    if len(vprof["price"]) < 2:
        return []
    step = np.diff(vprof["price"]).mean()
    for p in prices:
        if not cur:
            cur = [p]; continue
        if abs((p - cur[-1]) - step) < step*0.01:
            cur.append(p)
        else:
            blocks.append((min(cur), max(cur))); cur=[p]
    if cur:
        blocks.append((min(cur), max(cur)))
    return [(a,b) for (a,b) in blocks if b-a > step*1.5]

def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    tr = np.maximum(high-low, np.maximum((high-close.shift()).abs(), (low-close.shift()).abs()))
    return tr.rolling(period).mean()

# ----------------- Trigger logic -----------------

class RetestTriggerState:
    def __init__(self, side: str, hold_bars: int, retest_window: int, tolerance: float):
        self.side = side  # "up" or "down"
        self.hold_bars = int(hold_bars)
        self.retest_window = int(retest_window)
        self.tol = float(tolerance)

        self.run_len = 0
        self.armed = False
        self.window_left = 0
        self.level_at_arm = np.nan

    def reset(self):
        self.run_len = 0
        self.armed = False
        self.window_left = 0
        self.level_at_arm = np.nan

    def update(self, close: float, high: float, low: float, level: float) -> bool:
        """Return True only on the bar where the retest condition is actually observed."""
        if not np.isfinite(level):
            self.reset()
            return False

        # Hold condition (cross + hold)
        cond = (close > level) if self.side == "up" else (close < level)
        self.run_len = (self.run_len + 1) if cond else 0

        just_armed = False
        if (not self.armed) and (self.run_len == self.hold_bars):
            self.armed = True
            self.window_left = self.retest_window
            self.level_at_arm = float(level)
            just_armed = True  # retest window starts NEXT bar

        if self.armed and (not just_armed):
            eps = self.level_at_arm * self.tol

            if self.side == "up":
                ok = (low <= (self.level_at_arm + eps)) and (close > (self.level_at_arm - eps))
            else:
                ok = (high >= (self.level_at_arm - eps)) and (close < (self.level_at_arm + eps))

            self.window_left -= 1
            if ok:
                # Trigger happens NOW (no lookahead)
                self.reset()
                return True

            if self.window_left <= 0:
                self.reset()

        return False


class LevelTriggerCfg:
    def __init__(self, hold_bars: int = 2, retest_window: int = 6, tolerance: float = 0.0005):
        self.hold_bars = int(hold_bars)
        self.retest_window = int(retest_window)
        self.tolerance = float(tolerance)

def breakout_or_deviation(df: pd.DataFrame, level: float, side: str, retest: int, tol_bps: float):
    """Return 'continuation'|'deviation'|None; evaluated using ONLY bars available in df (no forward peeking)."""
    if df is None or df.empty or (not np.isfinite(level)):
        return None

    closes = df["close"].to_numpy()
    highs  = df["high"].to_numpy()
    lows   = df["low"].to_numpy()

    eps = level * (float(tol_bps) / 1e4)
    n = max(1, int(retest))

    if side == "up":
        broke = closes > (level + eps)
        # breakout EVENT = False -> True transition
        events = broke & np.r_[False, ~broke[:-1]]
        if not events.any():
            return None

        ib = int(np.flatnonzero(events)[-1])  # most recent breakout index
        if ib + 1 >= len(df):
            return None

        post_closes = closes[ib+1:]
        post_lows   = lows[ib+1:]

        # Deviation only once it's actually happened (known now)
        if post_closes[0] < (level - eps):
            return "deviation"

        # Continuation only once we have enough post-breakout bars
        if len(post_lows) >= n and (post_lows[:n] >= (level - eps)).all():
            return "continuation"

        return None

    else:  # side == "down"
        broke = closes < (level - eps)
        events = broke & np.r_[False, ~broke[:-1]]
        if not events.any():
            return None

        ib = int(np.flatnonzero(events)[-1])
        if ib + 1 >= len(df):
            return None

        post_closes = closes[ib+1:]
        post_highs  = highs[ib+1:]

        if post_closes[0] > (level + eps):
            return "deviation"

        if len(post_highs) >= n and (post_highs[:n] <= (level + eps)).all():
            return "continuation"

        return None


def cvd_proxy(df: pd.DataFrame) -> pd.Series:
    sign = np.where(df["close"].diff().fillna(0)>=0, 1, -1)
    return pd.Series((sign * df["volume"].values).cumsum(), index=df.index, name="cvd_proxy")

def level_to_level_plan_safe(
    entry_px: float,
    side: str,
    targets: List[float],
    atrv: float,
    atr_mult_tp: float,
    min_stop_bps: int,
    min_tp1_bps: int,
    atr_mult_sl: Optional[float] = None,
) -> Dict[str, float]:
    assert side in ("long", "short")
    sign = 1 if side == "long" else -1

    atr_mult_sl = float(atr_mult_tp if atr_mult_sl is None else atr_mult_sl)

    sl_raw = entry_px - sign * (atr_mult_sl * atrv)
    min_stop_abs = entry_px * (min_stop_bps / 10000.0)
    sl = sl_raw if abs(entry_px - sl_raw) >= min_stop_abs else entry_px - sign * min_stop_abs

    min_tp1_abs = entry_px * (min_tp1_bps / 10000.0)

    def _first_far_enough(ts):
        for t in ts:
            if (
                np.isfinite(t)
                and abs(t - entry_px) >= min_tp1_abs
                and (sign * (t - entry_px) > 0)
            ):
                return float(t)
        return None

    tp1 = _first_far_enough(targets)
    if tp1 is None:
        tp1 = entry_px + sign * max(min_tp1_abs, atr_mult_tp * atrv)

    ts_same_side = [t for t in targets if np.isfinite(t) and sign * (t - entry_px) > 0]
    if ts_same_side:
        tp2 = max(ts_same_side) if side == "long" else min(ts_same_side)
        if sign * (tp2 - tp1) <= 0:
             tp2 = tp1 + sign * max(min_tp1_abs, 2 * atr_mult_tp * atrv)
    else:
         tp2 = tp1 + sign * max(min_tp1_abs, 2 * atr_mult_tp * atrv)

    return {"sl": float(sl), "tp1": float(tp1), "tp2": float(tp2)}

def _rr(entry: float, sl: float, tp: float) -> float:
    risk = abs(entry - sl)
    return (abs(tp - entry) / risk) if risk > 0 and np.isfinite(tp) else np.nan

# ----------------- Backtest & live calculation -----------------

def refresh_pending_order_pre_activation_xo(
    pending_order: Dict[str, Any],
    i_prev: int,
    df: pd.DataFrame,
    VAL_p: float, POC_p: float, VAH_p: float,
    mon_h: float, mon_l: float,
    lookback_deviation: int,
    min_stop_bps: int,
    min_tp1_bps: int,
    atr_buf_x: float = 0.0,
) -> Dict[str, Any]:

    side = str(pending_order.get("side","")).lower()
    if side not in ("long","short"):
        return pending_order

    level = float(pending_order.get("signal_level", np.nan))
    if not np.isfinite(level):
        return pending_order

    entry = float(pending_order["entry"])

    # structural SL from sweep (history-only)
    if side == "long":
        structure_sl = sweep_cluster_low(df, level, end_idx=i_prev, max_lookback=lookback_deviation)
        if not np.isfinite(structure_sl):
            return pending_order
        # enforce min stop
        min_stop_dist = entry * (min_stop_bps/10000.0)
        if (entry - structure_sl) < min_stop_dist:
            structure_sl = entry - min_stop_dist

        min_tp_dist = entry * (min_tp1_bps / 10000.0)
        tp1, tp2 = pick_targets_long(entry, [POC_p, VAH_p, mon_h], min_tp_dist)

    else:
        structure_sl = sweep_cluster_high(df, level, end_idx=i_prev, max_lookback=lookback_deviation)
        if not np.isfinite(structure_sl):
            return pending_order
        min_stop_dist = entry * (min_stop_bps/10000.0)
        if (structure_sl - entry) < min_stop_dist:
            structure_sl = entry + min_stop_dist

        min_tp_dist = entry * (min_tp1_bps / 10000.0)
        tp1, tp2 = pick_targets_short(entry, [POC_p, VAL_p, mon_l], min_tp_dist)

    atrv = float(df["atr"].iloc[i_prev]) if ("atr" in df.columns) else np.nan
    if atr_buf_x > 0 and np.isfinite(atrv):
        if side == "long":
            structure_sl = float(structure_sl) - atr_buf_x * atrv
        else:
            structure_sl = float(structure_sl) + atr_buf_x * atrv

    # fallback tp2
    risk = abs(entry - structure_sl)
    if np.isfinite(risk) and risk > 0 and not np.isfinite(tp2) and np.isfinite(tp1):
        tp2 = (entry + 2*risk) if side=="long" else (entry - 2*risk)

    # optional clamp to signal bar extreme (still history-only)
    sig_low  = float(pending_order.get("signal_low",  np.nan))
    sig_high = float(pending_order.get("signal_high", np.nan))
    if side == "long" and np.isfinite(sig_low):
        structure_sl = min(structure_sl, sig_low)
    if side == "short" and np.isfinite(sig_high):
        structure_sl = max(structure_sl, sig_high)

    pending_order["sl"]  = float(structure_sl)
    pending_order["tp1"] = float(tp1) if np.isfinite(tp1) else float("nan")
    pending_order["tp2"] = float(tp2) if np.isfinite(tp2) else float("nan")
    pending_order["last_refresh_ts"] = df.index[i_prev]
    return pending_order


def should_invalidate_pending(
    ts: pd.Timestamp,
    row_prev: pd.Series,
    pending_order: Dict[str, Any],
    df: pd.DataFrame,
    params: Dict[str, Any],
) -> bool:
    """
    Called at the START of bar `ts` (before it can fill).
    Must only use information available up to the CLOSE of `ts_prev`.
    Pass row_prev = df.loc[ts_prev].
    """

    # --- 1) Time-out: pending too long without filling ---
    max_pending_bars = int(params.get("max_pending_bars", 0))  # 0 = disabled
    if max_pending_bars > 0:
        try:
            created_ts = pending_order.get("created_ts")
            if created_ts is not None:
                created_idx = df.index.get_loc(created_ts)
                now_idx = df.index.get_loc(ts)
                if (now_idx - created_idx) >= max_pending_bars:
                    return True
        except Exception:
            pass

    # --- 2) Optional: invalidate on EMA regime flip (use PREV bar regime only) ---
    if params.get("invalidate_on_regime_flip", False):
        try:
            regime_prev = row_prev.get("regime", None)
        except Exception:
            regime_prev = None

        side = str(pending_order.get("side", "")).lower()
        if regime_prev is not None:
            if side == "long" and regime_prev == "downtrend":
                return True
            if side == "short" and regime_prev == "uptrend":
                return True

    # --- 3) Invalidate if price closes back through the signal level (setup failed) ---
    sig_level = float(pending_order.get("signal_level", np.nan))
    tol = float(params.get("tol", 0.0))  # already in decimal (bps/10000)
    side = str(pending_order.get("side","")).lower()

    if np.isfinite(sig_level):
        if side == "long":
            # reclaim failed: close back below level (with tol)
            if float(row_prev["close"]) < sig_level * (1.0 - tol):
                return True
        elif side == "short":
            # breakdown failed: close back above level (with tol)
            if float(row_prev["close"]) > sig_level * (1.0 + tol):
                return True

    return False

def _rr_gate_tp(tp1: float, tp2: float, tp1_enabled: bool) -> float:
    # If TP1 is disabled, gate on TP2 instead
    if tp1_enabled and np.isfinite(tp1):
        return float(tp1)
    return float(tp2) if np.isfinite(tp2) else float("nan")

def get_monday_range_completed(df: pd.DataFrame, current_time: pd.Timestamp) -> tuple[float, float]:
    current_time = pd.to_datetime(current_time, utc=True)
    idx = pd.to_datetime(df.index, utc=True)

    this_mon_start = (current_time - pd.Timedelta(days=current_time.dayofweek)).normalize()
    this_mon_end   = this_mon_start + pd.Timedelta(days=1)

    mon_start = (this_mon_start - pd.Timedelta(days=7)) if current_time < this_mon_end else this_mon_start
    mon_end   = mon_start + pd.Timedelta(days=1)

    # RIGHT-LABELED candles: exclude boundary at mon_start (belongs to prior day)
    monday_data = df.loc[(idx > mon_start) & (idx <= mon_end)]
    if monday_data.empty:
        return np.nan, np.nan

    return float(monday_data["high"].max()), float(monday_data["low"].min())

def get_monthly_open(df: pd.DataFrame, current_time: pd.Timestamp) -> float:
    current_time = pd.to_datetime(current_time, utc=True)
    idx = pd.to_datetime(df.index, utc=True)

    month_start = current_time.normalize().replace(day=1)

    loc = idx.searchsorted(month_start, side="left")
    if loc < len(df):
        ts = idx[loc]
        if ts.year == current_time.year and ts.month == current_time.month:
            return float(df.iloc[loc]["open"])

    month_data = df[(idx.year == current_time.year) & (idx.month == current_time.month)]
    if not month_data.empty:
        return float(month_data.iloc[0]["open"])

    return np.nan

def sweep_cluster_low(df: pd.DataFrame, level: float, end_idx: int, max_lookback: int):
    start = max(0, end_idx - max_lookback + 1)
    j = end_idx
    min_low = float("inf")
    seen = False
    while j >= start:
        lo = float(df["low"].iloc[j])
        cl = float(df["close"].iloc[j])
        if (lo < level) or (cl < level):
            seen = True
            min_low = min(min_low, lo)
            j -= 1
            continue
        if seen:
            break
        j -= 1
    return (min_low if seen else np.nan)

def sweep_cluster_high(df: pd.DataFrame, level: float, end_idx: int, max_lookback: int):
    start = max(0, end_idx - max_lookback + 1)
    j = end_idx
    max_hi = -float("inf")
    seen = False
    while j >= start:
        hi = float(df["high"].iloc[j])
        cl = float(df["close"].iloc[j])
        if (hi > level) or (cl > level):
            seen = True
            max_hi = max(max_hi, hi)
            j -= 1
            continue
        if seen:
            break
        j -= 1
    return (max_hi if seen else np.nan)

def pick_targets_long(entry_px: float, candidates: list[float], min_tp_dist: float = 0.0):
    ups = sorted({float(x) for x in candidates if pd.notna(x) and np.isfinite(x) and float(x) > entry_px + min_tp_dist})
    tp1 = ups[0] if len(ups) >= 1 else np.nan
    tp2 = ups[1] if len(ups) >= 2 else np.nan
    return tp1, tp2

def pick_targets_short(entry_px: float, candidates: list[float], min_tp_dist: float = 0.0):
    dns = sorted({float(x) for x in candidates if pd.notna(x) and np.isfinite(x) and float(x) < entry_px - min_tp_dist}, reverse=True)
    tp1 = dns[0] if len(dns) >= 1 else np.nan
    tp2 = dns[1] if len(dns) >= 2 else np.nan
    return tp1, tp2


def backtest_xo_logic(df: pd.DataFrame, comp: Dict[str, Any], rng: Dict[str, Any], params: Dict[str, Any], force_eod_close: bool = False):
    """
    XO LOGIC UPDATE:
    1. Levels: Monday Range (High/Low), Monthly Open, Composite VAL/VAH.
    2. Trigger: DEVIATION & RECLAIM (Trap).
    3. Stop Loss: Structural.
    4. Integrated Filters via params["filters_func"]
    """
    df = df.copy()

    entry_style = str(params.get("entry_style", "Market-on-trigger"))
    tol = float(params.get("tol", 0.0))
    
    # Ensure index is datetime
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, utc=True)
    elif df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")

    # Calculate indicators
    df["atr"] = atr(df, params["atr_period"])
    df["cvd_proxy"] = cvd_proxy(df)
    df = ema_regime(df, params["ema_fast"], params["ema_slow"])

    trades: List[Dict[str, Any]] = []
    open_pos: Optional[Dict[str, Any]] = None
    pending_order: Optional[Dict[str, Any]] = None

    # Rolling Composite Profile
    lookback_days = int(params.get("lookback_days", 35))
    comp_bins     = int(params.get("comp_bins", 240))
    va_cov        = float(params.get("va_cov", 0.68))
    rolling_comp = rolling_composite_profile_cached(df[["close", "volume"]], lookback_days, comp_bins, 20, va_cov)
    
    val_series = rolling_comp["VAL"].reindex(df.index)
    poc_series = rolling_comp["POC"].reindex(df.index)
    vah_series = rolling_comp["VAH"].reindex(df.index)

    # Param extraction
    min_rr = float(params["min_rr"])
    tp1_exit_pct = int(params.get("tp1_exit_pct", 100))
    tp1_exit_frac = tp1_exit_pct / 100.0
    tp1_enabled = tp1_exit_frac > 0.0
    min_tp1_bps = int(params.get("min_tp1_bps", 50))
    min_stop_bps = int(params.get("min_stop_bps", 20))
    atr_mult_tp = float(params.get("atr_mult_tp", params.get("atr_mult", 1.5)))
    atr_mult_sl = float(params.get("atr_mult_sl", params.get("atr_mult", 1.5)))
    
    # XO specific params
    lookback_deviation = int(params.get("lookback_deviation", 5))
    
    # Retrieve Filter Function
    filters_func = params.get("filters_func", None)

    # HELPER METHODS (Nested)
    def _planned_entry(level: float, direction: str, entry_style: str, tol: float) -> tuple[str, float]:
        # direction: "long" or "short"
        if entry_style == "Market-on-trigger":
            return "market", float("nan")  # caller uses o_cur
        if entry_style == "Stop-through":
            return ("stop", level * (1.0 + tol)) if direction == "long" else ("stop", level * (1.0 - tol))
        # Limit-on-retest
        return ("limit", level * (1.0 - tol)) if direction == "long" else ("limit", level * (1.0 + tol))

    def _dedupe_levels(levels, tol_bps: float = 2.0):
        lv = sorted([float(x) for x in levels if pd.notna(x)])
        out = []
        for x in lv:
            if not out:
                out.append(x)
                continue
            tol = abs(out[-1]) * (tol_bps / 10000.0)
            if abs(x - out[-1]) > tol:
                out.append(x)
        return out

    def _fill_px_for_gap(entry_kind: str, side: str, epx: float, o_cur: float) -> float:
        # Optional but strongly recommended (handles open already beyond stop/limit)
        if entry_kind == "stop":
            return max(epx, o_cur) if side == "long" else min(epx, o_cur)
        if entry_kind == "limit":
            return min(epx, o_cur) if side == "long" else max(epx, o_cur)
        return epx

    def _manage_open_pos(open_pos: Dict[str, Any], ts: pd.Timestamp, hi_cur: float, lo_cur: float):
        side = open_pos["side"]
        sl   = float(open_pos["sl"])
        tp1  = float(open_pos.get("tp1", np.nan))
        tp2  = float(open_pos.get("tp2", np.nan))

        sl_hit  = (lo_cur <= sl) if side == "long" else (hi_cur >= sl)
        tp1_hit = tp1_enabled and np.isfinite(tp1) and ((hi_cur >= tp1) if side == "long" else (lo_cur <= tp1))
        tp2_hit = np.isfinite(tp2) and ((hi_cur >= tp2) if side == "long" else (lo_cur <= tp2))

        hit = None
        if sl_hit:
            hit = ("SL", sl)
        else:
            tp1_full_exit = (tp1_exit_frac >= 0.999)
            if tp1_full_exit:
                if tp1_hit: hit = ("TP1", tp1)
                elif tp2_hit: hit = ("TP2", tp2)
            else:
                if tp2_hit: hit = ("TP2", tp2)
                elif tp1_hit: hit = ("TP1", tp1)

        if not hit:
            return open_pos, False

        kind, exit_price = hit
        entry = float(open_pos["entry"])
        qty0  = float(open_pos.get("qty", 1.0))
        risk_abs = abs(entry - sl) or float("nan")

        def _append_leg(exit_kind, exit_px, leg_qty):
            pnl_px = ((exit_px - entry) if side == "long" else (entry - exit_px)) * leg_qty
            pnl_R  = (((exit_px - entry) / risk_abs) if side == "long" else ((entry - exit_px) / risk_abs)) * leg_qty
            trades.append({
                "entry_time": open_pos["entry_time"], "exit_time": ts,
                "side": side, "reason": open_pos["reason"], "exit_kind": exit_kind,
                "entry": entry, "sl": sl, "tp1": tp1, "tp2": tp2,
                "exit": float(exit_px), "pnl": float(pnl_px), "pnl_R": float(pnl_R),
                "qty": float(leg_qty),
                "entry_kind": open_pos.get("entry_kind", "market"),
            })

        if (kind == "TP2" and (0.0 < tp1_exit_frac < 1.0) and tp1_hit and np.isfinite(tp1) and np.isfinite(tp2)):
            take_qty = qty0 * tp1_exit_frac
            rem_qty  = qty0 - take_qty
            _append_leg("TP1_part", tp1, take_qty)
            _append_leg("TP2", tp2, rem_qty)
            return None, True

        if kind in ("SL", "TP2"):
            _append_leg(kind, exit_price, qty0)
            return None, True

        if tp1_exit_frac <= 0.0: return open_pos, False
        if tp1_exit_frac >= 1.0:
            _append_leg("TP1", exit_price, qty0)
            return None, True

        take_qty = qty0 * tp1_exit_frac
        rem_qty  = qty0 - take_qty
        _append_leg("TP1_part", exit_price, take_qty)
        open_pos["qty"] = float(rem_qty)
        open_pos["tp1"] = float("nan")
        return open_pos, False


    idx = df.index
    for i in range(len(df)):
        ts = idx[i]
        if i < lookback_deviation + 2: continue

        row = df.iloc[i]
        hi_cur, lo_cur = float(row["high"]), float(row["low"])
        o_cur = float(row["open"])
        close_cur = float(row["close"])

        ts_prev  = idx[i-1]
        row_prev = df.iloc[i-1]

        # --- Levels at T-1 ---
        mon_h, mon_l = get_monday_range_completed(df, ts_prev)
        m_open       = get_monthly_open(df, ts_prev)
        VAL_p = float(val_series.iloc[i-1]) if pd.notna(val_series.iloc[i-1]) else np.nan
        VAH_p = float(vah_series.iloc[i-1]) if pd.notna(vah_series.iloc[i-1]) else np.nan
        POC_p = float(poc_series.iloc[i-1]) if pd.notna(poc_series.iloc[i-1]) else np.nan

        support_levels    = _dedupe_levels([mon_l, m_open, VAL_p])
        resistance_levels = _dedupe_levels([mon_h, m_open, VAH_p])

        exited_this_bar = False

        # --- 1) Manage existing position ---
        if open_pos is not None:
            open_pos, exited_this_bar = _manage_open_pos(open_pos, ts, hi_cur, lo_cur)

        # --- pending order lifecycle (only when no open position) ---
        if open_pos is None and pending_order is not None:

            # 1) invalidate BEFORE activation (uses ONLY row_prev + params)
            if should_invalidate_pending(ts, row_prev, pending_order, df, params):
                pending_order = None
            else:
                # 2) refresh SL/TPs using ONLY info up to ts_prev
                pending_order = refresh_pending_order_pre_activation_xo(
                    pending_order,
                    i_prev=i-1,
                    df=df,
                    VAL_p=VAL_p, POC_p=POC_p, VAH_p=VAH_p,
                    mon_h=mon_h, mon_l=mon_l,
                    lookback_deviation=lookback_deviation,
                    min_stop_bps=min_stop_bps,
                    min_tp1_bps=min_tp1_bps,
                    atr_buf_x=float(params.get("_extra_atr_buffer", 0.0)),  # <-- add
                )

                # cancel pending if plan becomes invalid
                side = str(pending_order.get("side","")).lower()
                entry = float(pending_order["entry"])
                sl    = float(pending_order["sl"])
                tp1   = float(pending_order.get("tp1", np.nan))
                tp2   = float(pending_order.get("tp2", np.nan))

                tp_gate = tp1 if (tp1_enabled and np.isfinite(tp1)) else tp2
                risk = abs(entry - sl)
                reward = (tp_gate - entry) if side == "long" else (entry - tp_gate)

                if (not np.isfinite(tp_gate)) or (risk <= 0) or (reward <= 0) or ((reward / risk) < min_rr):
                    pending_order = None
                    continue


                # 3) check activation on CURRENT bar (hi/lo), fill at entry price
                ek = pending_order.get("entry_kind", "")
                side = str(pending_order.get("side", "")).lower()
                epx = float(pending_order["entry"])

                filled = False
                if ek == "stop":
                    filled = (hi_cur >= epx) if side == "long" else (lo_cur <= epx)
                elif ek == "limit":
                    filled = (lo_cur <= epx) if side == "long" else (hi_cur >= epx)

                if filled:
                    fill_px = _fill_px_for_gap(ek, side, epx, o_cur)
                    open_pos = {
                        "side": side,
                        "entry_kind": ek,
                        "entry_time": ts,
                        "entry": float(fill_px),
                        "sl": float(pending_order["sl"]),
                        "tp1": float(pending_order["tp1"]),
                        "tp2": float(pending_order["tp2"]),
                        "reason": pending_order.get("reason", ""),
                        "qty": 1.0,
                    }
                    pending_order = None

                    # ✅ handle same-bar SL/TP
                    open_pos, _exited = _manage_open_pos(open_pos, ts, hi_cur, lo_cur)
                    if _exited:
                        # open_pos will already be None if fully exited
                        continue

            # if you had a pending, don't arm a second one on the same bar
            if pending_order is not None:
                continue

        if exited_this_bar: continue

        # --- 2) Detect + Enter new setups ---
        if open_pos is None:
            close_prev  = float(row_prev["close"])
            low_prev    = float(row_prev["low"])
            high_prev   = float(row_prev["high"])
            close_prev2 = float(df["close"].iloc[i-2])

            eps_level = lambda lvl: abs(float(lvl)) * tol
            def _held(side: str, lvl: float) -> bool:
                if i < hold_bars:
                    return False
                window = df.iloc[i-hold_bars:i]
                if side == "long":
                    return bool((window["close"] > (lvl + eps_level(lvl))).all())
                return bool((window["close"] < (lvl - eps_level(lvl))).all())

            # ---- LONG ----
            for level in support_levels:
                level = float(level)
                eps = eps_level(level)
                reclaim = (close_prev > (level + eps)) and ((low_prev < (level - eps)) or (close_prev2 <= (level - eps)))
                if not (reclaim and _held("long", level)):
                    continue

                retest_ok = (lo_cur <= (level + eps)) and (close_cur > (level - eps))
                if not retest_ok:
                    continue

                structure_sl = sweep_cluster_low(df, level, end_idx=i-1, max_lookback=lookback_deviation)
                if not np.isfinite(structure_sl): continue

                # decide planned entry
                if entry_style == "Market-on-trigger":
                    entry_kind = "market"
                    entry_px = o_cur
                else:
                    entry_kind, entry_px = _planned_entry(level, "long", entry_style, tol)

                atr_buf_x = float(params.get("_extra_atr_buffer", 0.0))
                atrv_prev = float(row_prev.get("atr", np.nan))
                if atr_buf_x > 0 and np.isfinite(atrv_prev):
                    structure_sl = float(structure_sl) - atr_buf_x * atrv_prev

                # then compute structure_sl / min_stop / tp1,tp2 / min_tp / min_rr using entry_px (NOT o_cur)

                min_stop_dist = entry_px * (min_stop_bps / 10000.0)
                risk = entry_px - structure_sl
                if risk < min_stop_dist:
                    structure_sl = entry_px - min_stop_dist
                    risk = entry_px - structure_sl

                if risk <= 0: continue

                min_tp_dist = entry_px * (min_tp1_bps / 10000.0)
                tp1, tp2 = pick_targets_long(entry_px, [POC_p, VAH_p, mon_h], min_tp_dist)

                # If TP1 is enabled, require TP1 to exist
                if tp1_enabled and not np.isfinite(tp1):
                    continue

                # Ensure TP2 fallback exists
                if not np.isfinite(tp2):
                    tp2 = entry_px + 2.0 * risk

                tp_gate = tp1 if (tp1_enabled and np.isfinite(tp1)) else tp2
                if not np.isfinite(tp_gate):
                    continue

                min_tp_dist = entry_px * (min_tp1_bps / 10000.0)
                if abs(tp_gate - entry_px) < min_tp_dist:
                    continue

                reward = tp_gate - entry_px
                if (reward / risk) < min_rr:
                    continue


                # --- CHECK FILTERS ---
                if filters_func is not None:
                    # Signature: (i_ts, side, entry_px, reason, df_in, comp_in, rng_in, row_now)
                    # Use df, comp, and rng passed into backtest_xo_logic
                    comp_now = {"VAL": VAL_p, "POC": POC_p, "VAH": VAH_p}

                    df_hist = df.loc[:ts_prev]
                    rng_now = rolling_swing_range(df_hist, lookback=max(150, int(params.get("comp_bins", 240))//2))

                    if not filters_func(ts_prev, "long", entry_px, "XO Reclaim", df, comp_now, rng_now, row_prev):
                        continue

                if entry_style == "Market-on-trigger":
                    open_pos = {
                        "side": "long", "entry_time": ts, "entry": float(entry_px), "sl": float(structure_sl),
                        "tp1": float(tp1), "tp2": float(tp2), "reason": f"XO Reclaim {level:.2f}",
                        "qty": 1.0, "entry_kind": "market",
                    }
                    break
                else:
                    # create pending order instead of opening immediately
                    if entry_style == "Stop-through":
                        entry_kind = "stop"
                        pend_entry = level * (1.0 + tol)
                    else:  # "Limit-on-retest"
                        entry_kind = "limit"
                        pend_entry = level * (1.0 - tol)

                    pending_order = {
                        "side": "long",
                        "entry_kind": entry_kind,
                        "entry": float(entry_px),
                        "sl": float(structure_sl),
                        "tp1": float(tp1),
                        "tp2": float(tp2),
                        "reason": f"XO Reclaim {level:.2f}",
                        "created_ts": ts,
                        "signal_ts": ts_prev,
                        "signal_level": float(level),
                        "signal_low": float(low_prev),
                        "signal_high": float(high_prev),
                    }

                    # Try same-bar activation
                    ek = pending_order["entry_kind"]
                    side = str(pending_order.get("side", "")).lower()
                    epx  = float(pending_order["entry"])

                    filled = False
                    if ek == "stop":
                        filled = (hi_cur >= epx) if side == "long" else (lo_cur <= epx)
                    elif ek == "limit":
                        filled = (lo_cur <= epx) if side == "long" else (hi_cur >= epx)

                    if filled:
                        fill_px = _fill_px_for_gap(ek, side, epx, o_cur)
                        open_pos = {
                            "side": side,
                            "entry_kind": ek,
                            "entry_time": ts,
                            "entry": float(fill_px),
                            "sl": float(pending_order["sl"]),
                            "tp1": float(pending_order["tp1"]),
                            "tp2": float(pending_order["tp2"]),
                            "reason": pending_order.get("reason", ""),
                            "qty": 1.0,
                        }
                        pending_order = None

                        # same-bar SL/TP handling (your existing behavior)
                        open_pos, _exited = _manage_open_pos(open_pos, ts, hi_cur, lo_cur)
                        if _exited:
                            continue


                    break


            # ---- SHORT ----
            if open_pos is None and pending_order is None:
                for level in resistance_levels:
                    level = float(level)
                    eps = eps_level(level)
                    lost = (close_prev < (level - eps)) and ((high_prev > (level + eps)) or (close_prev2 >= (level + eps)))
                    if not (lost and _held("short", level)):
                        continue

                    retest_ok = (hi_cur >= (level - eps)) and (close_cur < (level + eps))
                    if not retest_ok:
                        continue

                    structure_sl = sweep_cluster_high(df, level, end_idx=i-1, max_lookback=lookback_deviation)
                    if not np.isfinite(structure_sl): continue

                    # decide planned entry
                    if entry_style == "Market-on-trigger":
                        entry_kind = "market"
                        entry_px = o_cur
                    else:
                        entry_kind, entry_px = _planned_entry(level, "short", entry_style, tol)

                    # then compute structure_sl / min_stop / tp1,tp2 / min_tp / min_rr using entry_px (NOT o_cur)

                    atr_buf_x = float(params.get("_extra_atr_buffer", 0.0))
                    atrv_prev = float(row_prev.get("atr", np.nan))
                    if atr_buf_x > 0 and np.isfinite(atrv_prev):
                        structure_sl = float(structure_sl) + atr_buf_x * atrv_prev

                    min_stop_dist = entry_px * (min_stop_bps / 10000.0)
                    risk = structure_sl - entry_px
                    if risk < min_stop_dist:
                        structure_sl = entry_px + min_stop_dist
                        risk = structure_sl - entry_px

                    if risk <= 0: continue

                    min_tp_dist = entry_px * (min_tp1_bps / 10000.0)
                    tp1, tp2 = pick_targets_short(entry_px, [POC_p, VAL_p, mon_l], min_tp_dist)

                    if tp1_enabled and not np.isfinite(tp1):
                        continue

                    if not np.isfinite(tp2):
                        tp2 = entry_px - 2.0 * risk

                    tp_gate = tp1 if (tp1_enabled and np.isfinite(tp1)) else tp2
                    if not np.isfinite(tp_gate):
                        continue

                    min_tp_dist = entry_px * (min_tp1_bps / 10000.0)
                    if abs(tp_gate - entry_px) < min_tp_dist:
                        continue

                    reward = entry_px - tp_gate
                    if (reward / risk) < min_rr:
                        continue


                    # --- CHECK FILTERS ---
                    if filters_func is not None:
                        comp_now = {"VAL": VAL_p, "POC": POC_p, "VAH": VAH_p}

                        df_hist = df.loc[:ts_prev]
                        rng_now = rolling_swing_range(df_hist, lookback=max(150, int(params.get("comp_bins", 240))//2))

                        if not filters_func(ts_prev, "short", entry_px, "XO Lost", df, comp_now, rng_now, row_prev):
                            continue

                    if entry_style == "Market-on-trigger":
                        open_pos = {
                            "side": "short", "entry_time": ts, "entry": float(entry_px), "sl": float(structure_sl),
                            "tp1": float(tp1), "tp2": float(tp2), "reason": f"XO Lost {level:.2f}",
                            "qty": 1.0, "entry_kind": "market",
                        }
                        break
                    else:
                        if entry_style == "Stop-through":
                            entry_kind = "stop"
                            pend_entry = level * (1.0 - tol)
                        else:  # "Limit-on-retest"
                            entry_kind = "limit"
                            pend_entry = level * (1.0 + tol)

                        pending_order = {
                            "side": "short",
                            "entry_kind": entry_kind,
                            "entry": float(entry_px),
                            "sl": float(structure_sl),
                            "tp1": float(tp1),
                            "tp2": float(tp2),
                            "reason": f"XO Lost {level:.2f}",
                            "created_ts": ts,
                            "signal_ts": ts_prev,
                            "signal_level": float(level),
                            "signal_low": float(low_prev),
                            "signal_high": float(high_prev),
                        }

                        # Try same-bar activation
                        ek = pending_order["entry_kind"]
                        side = str(pending_order.get("side", "")).lower()
                        epx  = float(pending_order["entry"])

                        filled = False
                        if ek == "stop":
                            filled = (hi_cur >= epx) if side == "long" else (lo_cur <= epx)
                        elif ek == "limit":
                            filled = (lo_cur <= epx) if side == "long" else (hi_cur >= epx)

                        if filled:
                            fill_px = _fill_px_for_gap(ek, side, epx, o_cur)
                            open_pos = {
                                "side": side,
                                "entry_kind": ek,
                                "entry_time": ts,
                                "entry": float(fill_px),
                                "sl": float(pending_order["sl"]),
                                "tp1": float(pending_order["tp1"]),
                                "tp2": float(pending_order["tp2"]),
                                "reason": pending_order.get("reason", ""),
                                "qty": 1.0,
                            }
                            pending_order = None

                            # same-bar SL/TP handling (your existing behavior)
                            open_pos, _exited = _manage_open_pos(open_pos, ts, hi_cur, lo_cur)
                            if _exited:
                                continue


                        break


        if open_pos is not None and open_pos.get("entry_time") == ts:
            if open_pos.get("entry_kind") == "market":
                open_pos, _ = _manage_open_pos(open_pos, ts, hi_cur, lo_cur)

    # EOD Close
    if open_pos is not None and force_eod_close:
        ts_end = df.index[-1]
        entry, side, sl, qty = float(open_pos["entry"]), open_pos["side"], float(open_pos["sl"]), float(open_pos.get("qty", 1.0))
        tp1, tp2 = float(open_pos.get("tp1", np.nan)), float(open_pos.get("tp2", np.nan))
        exit_price = float(df["close"].iloc[-1])
        risk_abs = abs(entry - sl) or float("nan")
        pnl_R = ( (exit_price - entry) / risk_abs if side == "long" else (entry - exit_price) / risk_abs ) * qty
        pnl_px = ( (exit_price - entry) if side == "long" else (entry - exit_price) ) * qty
        trades.append({
            "entry_time": open_pos["entry_time"], "entry_kind": open_pos.get("entry_kind", "market"), "exit_time": ts_end,
            "side": side, "reason": open_pos["reason"], "exit_kind": "EOD",
            "entry": entry, "sl": sl, "tp1": tp1, "tp2": tp2,
            "exit": exit_price, "pnl": float(pnl_px), "pnl_R": float(pnl_R), "qty": float(qty),
        })
        open_pos = None

    trades_df = pd.DataFrame(trades)
    expected_cols = ["entry_time","exit_time","side","reason","exit_kind","entry_kind","entry","sl","tp1","tp2","exit","pnl","pnl_R","qty"]
    if trades_df.empty: trades_df = pd.DataFrame(columns=expected_cols)
    else:
        for c in expected_cols:
            if c not in trades_df.columns: trades_df[c] = np.nan

    if not trades_df.empty:
        delta = (pd.to_datetime(trades_df["exit_time"], utc=True) - pd.to_datetime(trades_df["entry_time"], utc=True))
        trades_df["hold_minutes"] = delta.dt.total_seconds() / 60.0
        period_mins = _period_minutes(params.get("tf_for_stats", "1h")) or 60
        trades_df["hold_bars"] = trades_df["hold_minutes"] / period_mins
        r_series = trades_df["pnl_R"]
        summary = {
            "trades": int(len(trades_df)),
            "hit_rate": float((trades_df["pnl"] > 0).mean()),
            "total_pnl": float(trades_df["pnl"].sum()),
            "avg_pnl": float(trades_df["pnl"].mean()),
            "median_pnl": float(trades_df["pnl"].median()),
            "total_R": float(r_series.fillna(0).sum()),
            "avg_R": float(r_series.dropna().mean()) if r_series.notna().any() else float("nan"),
            "median_R": float(r_series.dropna().median()) if r_series.notna().any() else float("nan"),
            "hit_rate_R": float((r_series.dropna() > 0).mean()) if r_series.notna().any() else float("nan"),
        }
    else:
        summary = {"trades": 0, "hit_rate": float("nan"), "total_pnl": 0.0, "avg_pnl": float("nan"), "median_pnl": float("nan"), "total_R": 0.0, "avg_R": float("nan"), "median_R": float("nan"), "hit_rate_R": float("nan")}

    pending_list = [pending_order] if pending_order is not None else []
    return trades_df, summary, open_pos, pending_list

# ----------------- UI -----------------

st.title("Level Mapping Strategy — XO")

# --- Ensure Telegram sender is defined before UI uses it ---


if "_send_tg" not in globals():
    def _send_tg(tkn: str, chat_id: str, text: str) -> None:
        try:
            if not requests:
                st.warning("requests not available: cannot send Telegram messages.")
                return
            url = f"https://api.telegram.org/bot{tkn}/sendMessage"
            payload = {"chat_id": chat_id, "text": text, "parse_mode": "HTML", "disable_web_page_preview": True}
            requests.post(url, json=payload, timeout=10)
        except Exception as e:
            st.warning(f"Telegram send failed: {e}")
# --- Telegram de-dupe sets (session) ---
if "tg_seen_pending" not in st.session_state:
    st.session_state["tg_seen_pending"] = set()   # (symbol, created_ts, entry)
if "tg_seen_active" not in st.session_state:
    st.session_state["tg_seen_active"] = set()    # (symbol, entry_time, entry)
if "tg_seen_closed" not in st.session_state:
    st.session_state["tg_seen_closed"] = set()    # (symbol, entry_time, exit_time, exit_kind, exit)
if "tg_hits_seen" not in st.session_state:
    st.session_state["tg_hits_seen"] = set()

with st.sidebar:
    st.subheader("Inputs")
    predefined_symbols = [
    "BTCUSDT","ETHUSDT","SOLUSDT","XRPUSDT","AVAXUSDT",
    "LTCUSDT","ZECUSDT", "ASTERUSDT", "AAVEUSDT", "HYPEUSDT", "XLMUSDT", "DOGEUSDT", "BNBUSDT",
    "TRXUSDT", "ADAUSDT", "LINKUSDT", "HBARUSDT", "UNIUSDT", "DOTUSDT", "SHIB1000USDT", "ENAUSDT", "LDOUSDT", 
    "WIFUSDT", "FARTCOINUSDT", "TRUMPUSDT", "XVGUSDT", "MONUSDT"
    ]
    symbols_selected = st.multiselect(
    "Symbols", options=predefined_symbols, default=predefined_symbols,
    help="Predefined list (all selected). Deselect any. Use the box below to add more for this run."
    )
    extra_symbols = st.text_input(
    "Add symbols (comma-separated)", value="",
    help="Type one or more tickers separated by commas (e.g., BTCUSDT,ETHUSDT). They will be added for this run."
    )
    def _parse_syms(s):
        return [t.strip().upper() for t in s.split(",") if t.strip()]
    symbols_final = sorted(set(symbols_selected + _parse_syms(extra_symbols)))
    tf = st.selectbox("Timeframe", ["5m","15m","30m","1h","4h","12h","1d","1W"], index=5, help="Bar interval for resampling. Shorter TF = faster signals and smaller ATR; affects hold/retest windows and R/R.")
    lookback_days = st.slider("Composite days", 21, 49, 35, 1, help="Days of history to build the volume profile (VAL/POC/VAH). Longer = steadier levels, fewer triggers.")
    n_bins = st.slider("Profile bins", 120, 400, 240, 20, help="Number of histogram bins in the volume profile. Higher = finer granularity; can slightly shift VAL/POC/VAH.")

    # Risk/logic knobs
    with st.expander("Risk/logic knobs", expanded=False):

        atr_mult = st.number_input("ATR stop multiple", value=1.5, min_value=0.5, max_value=10.0, step=0.25, help="Stop distance = ATR × this multiple. Larger = wider stop → lower R/R; Smaller = tighter stop → higher R/R but easier stopouts.")
        min_stop_bps = st.number_input("Min stop distance (bps)", value=25, min_value=0, max_value=500, step=5, help="Hard minimum stop distance in basis points. Prevents unrealistically tight stops; can reduce R/R and block ideas.")
        min_tp1_bps  = st.number_input("Min TP1 distance (bps)",  value=50, min_value=0, max_value=5000, step=10, help="Minimum distance from entry to TP1 (bps). Ensures TP1 is not too close; affects R/R gating.")
        min_rr       = st.number_input("Min R/R (to TP1)",        value=1.2, min_value=0.5, max_value=5.0, step=0.1, help="Trades are only armed if R/R(entry→TP1 vs entry→SL) ≥ this. Raise for pickier ideas; lower to allow more.")
        va_cov = st.slider("Value Area coverage", 0.60, 0.75, 0.68, 0.01)
        ema_fast = st.number_input("EMA fast", 5, 50, 12, 1, help="Fast EMA for regime. Smaller = more reactive regimes → more SP continuation signals.")
        ema_slow = st.number_input("EMA slow", 10, 100, 21, 1, help="Slow EMA for regime. Larger = steadier regimes → fewer SP continuation signals.")
        atr_period = st.number_input("ATR period", 5, 50, 14, 1, help="ATR lookback. Shorter = more responsive stops/targets; Longer = smoother stops.")

        lookback_deviation = st.number_input(
            "Deviation lookback (bars)",
            min_value=2, max_value=100, value=5, step=1,
            help="How far back to search for the sweep cluster low/high to place structural SL."
        )
        hold_bars = st.number_input("Hold bars past level", 1, 10, 3, 1, help="After crossing VAL/VAH, # of consecutive closes beyond the level required before a retest counts. Higher = stricter triggers.")
        retest_bars = st.number_input("Retest window (bars)", 1, 20, 8, 1, help="Bars allowed for the retest after the hold. If no retest within this window → no order is armed.")
        tol_bps = st.number_input("Level tolerance (bps)", 0, 50, 10, 1, help="Tolerance applied to entry relative to VAL/VAH/SP bands. Larger = entry further from level; can change R/R and fill probability.")

        entry_style = st.selectbox(
        "Entry style",
        ["Stop-through", "Limit-on-retest", "Market-on-trigger"],
        index=1,
        help="How to place entries when a trigger fires."
        )

        tp1_exit_pct = st.select_slider(
            "TP1 exit %",
            options=[0, 25, 50, 75, 100],
            value=100,
            help="Percentage of the position to close at TP1; the remainder rides toward TP2/SL."
        )

        drop_last_incomplete = st.checkbox("Drop last incomplete candle", True, help="If checked, removes a last bar whose timestamp is in the future (still forming). Prevents using incomplete bars for signals.")
    
    with st.expander("Optional Filters (advanced)", expanded=False):
        st.caption("These gate entries.")
        col1, col2 = st.columns(2)
        use_htf_conf = col1.checkbox("Require HTF confluence (Monthly/Weekly)", value=True)
        htf_tol_bps  = col1.slider("HTF confluence tolerance (bps)", 2, 60, 25, 1)
        use_accept   = col1.checkbox("Require acceptance at level", value=False)
        accept_bars  = col1.slider("Bars of acceptance (proxy)", 1, 6, 2, 1)
        use_htf_ema_gate = col1.checkbox("Require HTF EMA(12/25) bull intact", value=False,
                                 help="Weekly & 3-Day: price > both EMAs, no fresh bearish cross.")
        avoid_mid    = col2.checkbox("Avoid mid-range entries", value=True)
        mid_min_bps  = col2.slider("Min distance from mid (bps)", 10, 200, 40, 5)
        no_fade      = col2.checkbox("No shorts into support / longs into resistance", value=True)
        opp_buf_bps  = col2.slider("Opposite-level buffer (bps)", 5, 60, 25, 1)
        block_countertrend_entries    = col2.checkbox("Block entries against EMA regime (no longs in downtrend, no shorts in uptrend)", value=True)

        st.markdown("---")
        st.caption("Flow confirmation proxies (no DOM needed)")
        col3, col4 = st.columns(2)
        use_relvol  = col3.checkbox("Require relative volume spike", value=False)
        relvol_x    = col3.slider("RelVol threshold (× avg)", 12, 400, 180, 5, help="Value ×100; e.g. 180 = 1.8×")
        use_wick    = col4.checkbox("Allow wick/engulf proxy", value=False)
        wick_min_bp = col4.slider("Min wick size (bps) at level", 0, 200, 30, 5)

        st.markdown("---")
        add_atr_buf = st.checkbox("Add ATR buffer beyond zone to SL", value=False)
        atr_buf_x   = st.slider("ATR buffer (× ATR)", 0.0, 1.5, 0.5, 0.1) if add_atr_buf else 0.0

        use_mid_gate = st.checkbox("Mid-gate is crisper, avoid-mid is spacing", value=False)

    with st.expander("Pending order behaviour"):
        col1, col2 = st.columns(2)

        max_pending_bars = col1.number_input(
            "Max bars to keep a pending order alive (0 = never timeout)",
            min_value=0,
            max_value=50,
            value=6,
            step=1,
        )
        invalidate_on_regime_flip = col2.checkbox(
            "Cancel pending orders if EMA regime flips against them",
            value=True,
        )


    # --- Telegram Alerts (sidebar) ---
    with st.expander("Telegram Alerts", expanded=False):
        enable_tg = st.checkbox("Enable Telegram alerts", value=False)

        # Default Telegram credentials
        default_tg_token = "8400576570:AAEg2DScBGEkCavTne9yLRqFERNgEeIiVDo"
        default_chat_id = "-4676287176"

        tg_token = st.text_input(
            "Bot Token",
            type="password",
            value=default_tg_token,
            help="BotFather token (default prefilled for your bot)."
        )
        tg_chat_id = st.text_input(
            "Chat ID (user/group/channel)",
            value=default_chat_id,
            help="Telegram Chat ID (default prefilled for your alert group)."
        )

        notify_preview = st.checkbox("Notify on PREVIEWED trades (pending orders)", value=True)
        notify_active = st.checkbox("Notify on ACTIVE trade (currently open)", value=True)
        notify_closed = st.checkbox("Notify on CLOSED trades (TP/SL fills)", value=True)
        closed_types = st.multiselect(
            "Closed types to notify",
            ["TP1","TP2","SL"],
            default=["TP1","TP2","SL"],
            help="Choose which exit kinds trigger a Telegram notification."
        )
        notify_hits = st.checkbox("Notify on ACTIVE hits (SL/TP1/TP2)", value=True)



    run_btn = st.button("Run", use_container_width=True)

# ---- Symbol picker (after sidebar so symbols_final exists) ----
picker_opts = (["ALL", "Pending", "Active"] + symbols_final) if symbols_final else ["ALL", "Pending", "Active"]
_selection_key = "sym_picker_selection"

try:
    selection = st.segmented_control(
        "Output view",
        picker_opts,
        selection_mode="single",
        default="ALL",
        key=_selection_key,
    )
except Exception:
    selection = st.radio("Output view", picker_opts, horizontal=True, index=0, key=_selection_key)

st.caption("Choose **ALL** for cumulative charts, **Pending**/**Active** to see cross-symbol tables, or a single symbol for detailed output.")



def _load(symbol: str, tf: str) -> pd.DataFrame:
    if load_ohlcv is None:
        st.error("No data loader found. Please expose `load_ohlcv(symbol, timeframe)` in your host app.")
        st.stop()

    df = load_ohlcv(symbol, tf, limit=1500, category="linear")
    if df is None or df.empty:
        return pd.DataFrame(columns=["open","high","low","close","volume"])

    if "time" in df.columns:
        t = df["time"]
        # robust: handle ms vs s vs already-ISO
        if np.issubdtype(t.dtype, np.number):
            med = float(pd.to_numeric(t, errors="coerce").dropna().median()) if t.notna().any() else 0.0
            unit = "ms" if med > 1e12 else "s"
            idx = pd.to_datetime(t, unit=unit, utc=True, errors="coerce")
        else:
            idx = pd.to_datetime(t, utc=True, errors="coerce")

        df = df.drop(columns=["time"]).set_index(idx)
        df = df[~df.index.duplicated(keep="last")].sort_index()
    if "volume" not in df.columns:
        df["volume"] = 0.0
    return df



def _fmt_ddhhmm(minutes):
    """Format minutes as dd:hh:mm (zero-padded). Returns '—' if missing."""
    try:
        if minutes is None or not np.isfinite(minutes) or minutes < 0:
            return "—"
        minutes = int(round(float(minutes)))
        days, rem = divmod(minutes, 1440)
        hours, mins = divmod(rem, 60)
        return f"{days:02d}:{hours:02d}:{mins:02d}"
    except Exception:
        return "—"


# ---- Compute & Render helpers (no trade-logic changes) ----
def rolling_swing_range(df: pd.DataFrame, lookback: int = 150) -> dict:
    w = df.tail(lookback)
    if w.empty: return {"hi": np.nan, "lo": np.nan, "mid": np.nan}
    r_hi, r_lo = float(w["high"].max()), float(w["low"].min())
    return {"hi": r_hi, "lo": r_lo, "mid": (r_hi + r_lo)/2.0}

def equal_extrema_mask(series: pd.Series, tol_bps: float) -> pd.Series:
    # identifies clusters of equal highs/lows within tolerance
    vals = series.values
    tol = np.maximum(np.abs(vals)*tol_bps/1e4, 1e-12)
    eq = np.zeros_like(vals, dtype=bool)
    for i in range(1, len(vals)):
        if abs(vals[i]-vals[i-1]) <= max(tol[i], tol[i-1]):
            eq[i] = eq[i-1] = True
    return pd.Series(eq, index=series.index)

def is_sfp(df: pd.DataFrame, prior_extreme: float, side: str, lookback: int = 40) -> bool:
    try:
        w = df.tail(lookback)
        if w.empty or not (prior_extreme == prior_extreme):  # NaN check
            return False
        r = w.iloc[-1]
        if side == "short":  # SFP high
            return bool((r["high"] > prior_extreme) and (r["close"] < prior_extreme))
        else:                # SFP low
            return bool((r["low"]  < prior_extreme) and (r["close"] > prior_extreme))
    except Exception:
        pass
    return False


def compression_squeeze(df: pd.DataFrame, lookback=50, pct=0.15) -> bool:
    try:
        m = df["close"].rolling(20).mean()
        sd = df["close"].rolling(20).std()
        bbw = 2*sd / (m.replace(0, np.nan).abs())
        thr = np.nanpercentile(bbw.dropna().values, int(pct*100)) if bbw.notna().any() else np.inf
        return bool(bbw.iloc[-1] <= thr)
    except Exception:
        pass
    return False

def cvd_divergence(df: pd.DataFrame, lookback=60, near="resistance") -> bool:
    try:
        if "cvd_proxy" not in df.columns:
            return False
        w = df.tail(int(lookback)).dropna(subset=["cvd_proxy","close"])
        if len(w) < 8:
            return False
        t = np.arange(len(w))
        b_price = np.polyfit(t, w["close"].values, 1)[0]
        b_cvd   = np.polyfit(t, w["cvd_proxy"].values, 1)[0]
        return (b_cvd > 0 and b_price <= 0) if near=="resistance" else (b_cvd < 0 and b_price >= 0)
    except Exception:
        pass
    return False

def oi_spike(df: pd.DataFrame, mult=1.8) -> bool:
    try:
        if "open_interest" not in df.columns:
            return False
        ma = df["open_interest"].rolling(50, min_periods=10).mean()
        if not np.isfinite(ma.iloc[-1]) or ma.iloc[-1] <= 0:
            return False
        return bool(df["open_interest"].iloc[-1] >= mult*ma.iloc[-1])
    except Exception:
        pass
    return False


def expansion_accept(df: pd.DataFrame, key_level: float, side: str, bars=2, tol_bps=0.0) -> bool:
    try:
        eps = key_level*(tol_bps/1e4)
        w = df.tail(int(max(1, bars)))
        if w.empty:
            return False
        if side == "long":
            return bool((w["close"] > key_level+eps).all())
        else:
            return bool((w["close"] < key_level-eps).all())
    except Exception:
        pass
    return False

# Advanced toggles (not currently wired to UI; keep defined to avoid NameError)
use_breakout_logic = False
sfp_only = False
use_comp_expansion = False
use_cvd_absorption = False
use_oi_spike_reduce = False
naked_poc_guard = False
use_btc_guard = False

def build_filters_func(symbol: str, df: pd.DataFrame, n_bins: int, cfg: Dict[str, Any]):
    use_htf_conf = bool(cfg.get("use_htf_conf", False))
    htf_tol_bps  = float(cfg.get("htf_tol_bps", 0))
    use_accept   = bool(cfg.get("use_accept", False))
    accept_bars  = int(cfg.get("accept_bars", 0))
    avoid_mid    = bool(cfg.get("avoid_mid", False))
    mid_min_bps  = float(cfg.get("mid_min_bps", 0))
    no_fade      = bool(cfg.get("no_fade", False))
    opp_buf_bps  = float(cfg.get("opp_buf_bps", 0))
    block_countertrend_entries = bool(cfg.get("block_countertrend_entries", False))

    use_relvol   = bool(cfg.get("use_relvol", False))
    relvol_x     = float(cfg.get("relvol_x", 0))
    use_wick     = bool(cfg.get("use_wick", False))
    wick_min_bp  = float(cfg.get("wick_min_bp", 0))

    use_htf_ema_gate = bool(cfg.get("use_htf_ema_gate", False))
    use_mid_gate = bool(cfg.get("use_mid_gate", False))

    va_cov_local = float(cfg.get("va_cov", 0.68))


    # Return None if no filters are enabled
    enabled = any([
        use_htf_conf,
        use_accept,
        avoid_mid,
        no_fade,
        use_relvol,
        use_wick,
        use_htf_ema_gate,
        use_mid_gate,
        block_countertrend_entries,
        # advanced toggles (currently false in your code)
        use_breakout_logic, sfp_only, use_comp_expansion, use_cvd_absorption,
        use_oi_spike_reduce, naked_poc_guard, use_btc_guard,
    ])
    if not enabled:
        return None

    def _bps(a, b):
        try:
            a = float(a); b = float(b)
            return abs((a - b) / a) * 1e4 if a != 0 else np.inf
        except Exception:
            return np.inf

    def _nearest_bps(px, levels: dict):
        vals = [levels.get("VAL"), levels.get("POC"), levels.get("VAH")]
        vals = [v for v in vals if np.isfinite(v)]
        return min((_bps(px, v) for v in vals), default=np.inf)

    def _opp_nearest_bps(side: str, px: float, comp_levels: dict):
        if side == "long":
            vals = [comp_levels.get("POC"), comp_levels.get("VAH")]
        else:
            vals = [comp_levels.get("POC"), comp_levels.get("VAL")]
        vals = [v for v in vals if np.isfinite(v)]
        return min((_bps(px, v) for v in vals), default=np.inf)

    def _accepted(side: str, i_ts: pd.Timestamp, comp_in: dict, df_local: pd.DataFrame) -> bool:
        if not use_accept or accept_bars <= 0:
            return True
        try:
            ix = df_local.index.get_loc(i_ts)
            w  = df_local.iloc[max(0, ix - accept_bars + 1):ix + 1]
            if w.empty:
                return False
            val = comp_in.get("VAL", np.nan)
            vah = comp_in.get("VAH", np.nan)
            if side == "long":
                return (w["close"] > val).all() if np.isfinite(val) else True
            else:
                return (w["close"] < vah).all() if np.isfinite(vah) else True
        except Exception:
            return True  # fail-open

    def _relvol_ok(i_ts: pd.Timestamp, df_local: pd.DataFrame) -> bool:
        if not use_relvol:
            return True
        try:
            vol = df_local["volume"]
            ma  = vol.rolling(50, min_periods=10).mean()
            v   = float(vol.loc[i_ts])
            m   = float(ma.loc[i_ts])
            thr = float(relvol_x) / 100.0
            return (m > 0) and (v >= thr * m)
        except Exception:
            return False

    def _wick_ok(side: str, i_ts: pd.Timestamp, entry: float, df_local: pd.DataFrame) -> bool:
        if not use_wick:
            return True
        try:
            r = df_local.loc[i_ts]
            hi, lo, op, cl = float(r["high"]), float(r["low"]), float(r["open"]), float(r["close"])
            up_w = max(0.0, hi - max(op, cl))
            dn_w = max(0.0, min(op, cl) - lo)
            w_bps = lambda x: (x / entry) * 1e4 if entry > 0 else 0.0
            return (w_bps(dn_w) >= float(wick_min_bp)) if side == "long" else (w_bps(up_w) >= float(wick_min_bp))
        except Exception:
            return False

    def _filters_func(i_ts, side, entry_px, reason, df_in, comp_in, rng_in, row_now) -> bool:
        # enforce history-only
        try:
            df_hist = df_in.loc[:i_ts]
        except Exception:
            df_hist = df_in
        if df_hist is None or df_hist.empty:
            return False

        # 1) HTF confluence
        if use_htf_conf:
            htf_w = composite_profile(df_hist, days=84,  n_bins=n_bins, va_cov=va_cov_local)
            htf_m = composite_profile(df_hist, days=168, n_bins=n_bins, va_cov=va_cov_local)
            ok_w = _nearest_bps(entry_px, htf_w) <= float(htf_tol_bps)
            ok_m = _nearest_bps(entry_px, htf_m) <= float(htf_tol_bps)
            if not (ok_w or ok_m):
                return False

        # 2) acceptance
        if not _accepted(side, i_ts, comp_in, df_hist):
            return False

        # 3) avoid mid
        if avoid_mid and np.isfinite(comp_in.get("VAL", np.nan)) and np.isfinite(comp_in.get("VAH", np.nan)):
            mid = 0.5 * (float(comp_in["VAL"]) + float(comp_in["VAH"]))
            if _bps(entry_px, mid) < float(mid_min_bps):
                return False

        # 4) no fade
        if no_fade and _opp_nearest_bps(side, entry_px, comp_in) < float(opp_buf_bps):
            return False

        # 5) flow proxies
        if not _relvol_ok(i_ts, df_hist):
            return False
        if not _wick_ok(side, i_ts, entry_px, df_hist):
            return False

        # 6) HTF EMA gate (longs)
        if use_htf_ema_gate and side == "long":
            st_htf = htf_ema_state(df_hist)
            bull_intact_now = (st_htf["weekly"]["bull_ok"] and not st_htf["weekly"]["bear_cross"]) and \
                              (st_htf["d3"]["bull_ok"] and not st_htf["d3"]["bear_cross"])
            if not bull_intact_now:
                return False

        # 7) countertrend block
        if block_countertrend_entries:
            regime_now = row_now.get("regime", None)
            if side == "long" and regime_now == "downtrend":
                return False
            if side == "short" and regime_now == "uptrend":
                return False

        # mid-gate (optional)
        if use_mid_gate and np.isfinite(rng_in.get("mid", np.nan)):
            if side == "long" and entry_px < rng_in["mid"]:
                return False
            if side == "short" and entry_px > rng_in["mid"]:
                return False

        return True

    return _filters_func


def prepare_symbol_context(symbol: str, tf_now: str) -> Dict[str, Any]:
    raw = _load(symbol, tf_now)
    df = raw
    want = _period_minutes(tf_now)
    if want and len(raw) >= 5:
        med = raw.index.to_series().diff().dropna().median()
        have = int(round(med.total_seconds() / 60)) if pd.notna(med) else None
        if have is None or have != want:
            df = resample_ohlcv(raw, tf_now)

    if drop_last_incomplete:
        df = drop_incomplete_bar(df, tf_now)

    comp = composite_profile(df, days=lookback_days, n_bins=n_bins, va_cov=va_cov)
    rng = rolling_swing_range(df, lookback=max(150, n_bins // 2))
    sp_bands = single_prints(comp["hist"])
    return {"df": df, "comp": comp, "rng": rng, "sp_bands": sp_bands}


def build_strategy_params(symbol: str, df: pd.DataFrame, sp_bands: List[Tuple[float, float]]) -> Dict[str, Any]:
    params = {
        "tf_for_stats": tf,
        "ema_fast": int(ema_fast), "ema_slow": int(ema_slow),
        "atr_period": int(atr_period),
        "hold_bars": int(hold_bars), "retest_bars": int(retest_bars),
        "tol_bps": float(tol_bps),
        "tol": float(tol_bps) / 10000.0,
        "use_val_reclaim": True,
        "use_vah_break": True,
        "use_sp_cont": True,
        "sp_bands": sp_bands,
        "atr_mult": float(atr_mult),
        "min_stop_bps": int(min_stop_bps),
        "min_tp1_bps": int(min_tp1_bps),
        "atr_mult_tp": float(atr_mult),
        "atr_mult_sl": float(atr_mult),
        "min_rr": float(min_rr),
        "entry_style": entry_style,
        "tp1_exit_pct": int(tp1_exit_pct),
        "lookback_days": int(lookback_days),
        "comp_bins": int(n_bins),
        "max_pending_bars": int(max_pending_bars),
        "invalidate_on_regime_flip": bool(invalidate_on_regime_flip),
        "_extra_atr_buffer": float(atr_buf_x) if add_atr_buf else 0.0,
        "block_countertrend_entries": bool(block_countertrend_entries),
        "va_cov": float(va_cov),
        "lookback_deviation": int(lookback_deviation),
    }
    filter_cfg = {
        "use_htf_conf": use_htf_conf,
        "htf_tol_bps": htf_tol_bps,
        "use_accept": use_accept,
        "accept_bars": accept_bars,
        "avoid_mid": avoid_mid,
        "mid_min_bps": mid_min_bps,
        "no_fade": no_fade,
        "opp_buf_bps": opp_buf_bps,
        "block_countertrend_entries": block_countertrend_entries,
        "use_relvol": use_relvol,
        "relvol_x": relvol_x,
        "use_wick": use_wick,
        "wick_min_bp": wick_min_bp,
        "use_htf_ema_gate": use_htf_ema_gate,
        "use_mid_gate": use_mid_gate,
        "va_cov": float(va_cov),
    }
    params["filters_func"] = build_filters_func(symbol, df, n_bins, filter_cfg)
    return params


def notify_closed_trades(symbol: str, trades_df: pd.DataFrame, tf_now: str) -> None:
    if not (enable_tg and tg_token and tg_chat_id and notify_closed and not trades_df.empty):
        return

    last_trade = trades_df.iloc[-1].to_dict()
    ek = str(last_trade.get("exit_kind", ""))
    ek_filter = "TP1" if ek.startswith("TP1") else ek
    if ek_filter not in closed_types:
        return

    key = (symbol, str(last_trade.get("entry_time")), str(last_trade.get("exit_time")), ek, float(last_trade.get("exit", np.nan)))
    if key in st.session_state["tg_seen_closed"]:
        return

    msg = "\n".join([
        f"✅ <b>CLOSED</b> — <b>{symbol}</b> ({_esc(tf_now)})",
        f"Side: <b>{_esc(last_trade.get('side','')).upper()}</b>  ·  {_esc(last_trade.get('reason',''))}",
        f"Exit: <b>{float(last_trade.get('exit')):.4f}</b>  ·  Kind: <b>{_esc(ek)}</b>",
        f"PnL_R: <b>{float(last_trade.get('pnl_R', 0.0)):.2f}R</b>",
    ])
    _send_tg(tg_token, tg_chat_id, msg)
    st.session_state["tg_seen_closed"].add(key)


def _val_line(comp: Dict[str, Any]) -> str:
    vals = []
    try:
        for k in ("VAL", "POC", "VAH"):
            v = comp.get(k, float("nan"))
            if np.isfinite(v):
                vals.append(f"{k}: <b>{v:.4f}</b>")
    except Exception:
        pass
    return " · ".join(vals) if vals else ""


def notify_live_states(symbol: str, tf_now: str, comp: Dict[str, Any], df: pd.DataFrame, pending_list: List[Dict[str, Any]], active_open: Optional[Dict[str, Any]]):
    if not (("enable_tg" in globals() and enable_tg) and tg_token and tg_chat_id):
        return

    if ("notify_preview" in globals() and notify_preview) and pending_list:
        for p in pending_list:
            try:
                sig = p.get("signal_ts", p.get("created_ts"))
                k = (symbol, str(sig), float(p["entry"]))
            except Exception:
                k = (symbol, str(p.get("created_ts")), str(p.get("entry")))
            if k in st.session_state["tg_seen_pending"]:
                continue
            entry = float(p["entry"]); sl = float(p["sl"])
            tp1 = float(p.get("tp1", float("nan"))); tp2 = float(p.get("tp2", float("nan")))
            risk = abs(entry - sl) or float("nan")
            R1 = ((tp1 - entry) / risk) if (np.isfinite(risk) and risk > 0 and np.isfinite(tp1)) else float("nan")
            R2 = ((tp2 - entry) / risk) if (np.isfinite(risk) and risk > 0 and np.isfinite(tp2)) else float("nan")
            reason = p.get("reason", "")
            created = p.get("created_ts")
            signal  = p.get("signal_ts", None)
            lines = [
                f"🟨 <b>PREVIEW</b> — <b>{symbol}</b> ({_esc(tf_now)})",
                f"Side: <b>{_esc(p['side']).upper()}</b>  ·  {_esc(reason)}",
                f"Entry: <b>{entry:.4f}</b> | SL: <b>{sl:.4f}</b>",
                f"TP1: {'—' if not np.isfinite(tp1) else f'<b>{tp1:.4f}</b>'}{'' if not np.isfinite(R1) else f'  (≈ {R1:.2f}R)'}",
                f"TP2: {'—' if not np.isfinite(tp2) else f'<b>{tp2:.4f}</b>'}{'' if not np.isfinite(R2) else f'  (≈ {R2:.2f}R)'}",
                _val_line(comp),
                (f"Created: {created}" if created is not None else ""),
                (f"Signal: {signal}" if signal is not None else ""),
            ]
            _send_tg(tg_token, tg_chat_id, "\n".join([ln for ln in lines if ln]))
            st.session_state["tg_seen_pending"].add(k)

    if ("notify_active" in globals() and notify_active) and (active_open is not None):
        try:
            k2 = (symbol, str(active_open.get("entry_time")), float(active_open["entry"]))
        except Exception:
            k2 = (symbol, str(active_open.get("entry_time")), str(active_open.get("entry")))
        if k2 not in st.session_state["tg_seen_active"]:
            entry = float(active_open["entry"]); sl = float(active_open["sl"])
            tp1 = float(active_open.get("tp1", float("nan"))); tp2 = float(active_open.get("tp2", float("nan")))
            last_px = float(df["close"].iloc[-1])
            risk = abs(entry - sl) or float("nan")
            u_pnl = (last_px - entry) if active_open["side"] == "long" else (entry - last_px)
            u_R = (u_pnl / risk) if (np.isfinite(risk) and risk > 0) else float("nan")
            lines2 = [
                f"🟢 <b>ACTIVE</b> — <b>{symbol}</b> ({_esc(tf_now)})",
                f"Side: <b>{_esc(active_open['side']).upper()}</b>  ·  {_esc(active_open.get('reason',''))}",
                f"Entry: <b>{entry:.4f}</b> | SL: <b>{sl:.4f}</b>",
                f"TP1: {'—' if not np.isfinite(tp1) else f'<b>{tp1:.4f}</b>'}  |  "
                f"TP2: {'—' if not np.isfinite(tp2) else f'<b>{tp2:.4f}</b>'}",
                f"Last: <b>{last_px:.4f}</b>  ·  U.PnL: {'—' if not np.isfinite(u_pnl) else f'{u_pnl:.4f}'}"
                f"{'' if not np.isfinite(u_R) else f'  (≈ {u_R:.2f}R)'}",
                _val_line(comp),
            ]
            _send_tg(tg_token, tg_chat_id, "\n".join([ln for ln in lines2 if ln]))
            st.session_state["tg_seen_active"].add(k2)

        if ("notify_hits" in globals() and notify_hits):
            if "tg_hits_seen" not in st.session_state:
                st.session_state["tg_hits_seen"] = set()
            entry_time = active_open.get("entry_time")
            side = str(active_open.get("side", "")).lower()
            last_px = float(df["close"].iloc[-1])
            sl = float(active_open.get("sl", float("nan")))
            tp1 = float(active_open.get("tp1", float("nan")))
            tp2 = float(active_open.get("tp2", float("nan")))
            def _hit(mk: str) -> bool:
                if not np.isfinite(sl): return False
                if mk == "SL":  return (last_px <= sl) if side=="long" else (last_px >= sl)
                if mk == "TP1": return np.isfinite(tp1) and ((last_px >= tp1) if side=="long" else (last_px <= tp1))
                if mk == "TP2": return np.isfinite(tp2) and ((last_px >= tp2) if side=="long" else (last_px <= tp2))
                return False
            for mk, badge in (("SL","🔴"),("TP2","🟣"),("TP1","🟩")):
                key = (symbol, str(entry_time), mk)
                if key in st.session_state["tg_hits_seen"]:
                    continue
                if not _hit(mk):
                    continue
                entry = float(active_open["entry"])
                risk = abs(entry - sl) if np.isfinite(sl) else float("nan")
                u_pnl = (last_px - entry) if side == "long" else (entry - last_px)
                u_R = (u_pnl / risk) if (np.isfinite(risk) and risk > 0) else float("nan")
                _targets = []
                if np.isfinite(tp1): _targets.append(f"TP1: <b>{tp1:.4f}</b>")
                if np.isfinite(tp2): _targets.append(f"TP2: <b>{tp2:.4f}</b>")
                targets_line = "  |  ".join(_targets) if _targets else "No TPs set"
                lines_hit = [
                    f"{badge} <b>{mk} HIT</b> — <b>{symbol}</b> ({_esc(tf_now)})",
                    f"Side: <b>{side.upper()}</b>  ·  {_esc(active_open.get('reason',''))}",
                    f"Entry: <b>{entry:.4f}</b>  |  SL: <b>{sl if not np.isfinite(sl) else f'{sl:.4f}'}</b>",
                    targets_line,
                    f"Last: <b>{last_px:.4f}</b>  ·  U.PnL: {'—' if not np.isfinite(u_pnl) else f'{u_pnl:.4f}'}"
                    f"{'' if not np.isfinite(u_R) else f'  (≈ {u_R:.2f}R)'}",
                ]
                _send_tg(tg_token, tg_chat_id, "\n".join([ln for ln in lines_hit if ln]))
                st.session_state["tg_hits_seen"].add(key)



def compute_for_symbol(symbol: str) -> Dict[str, Any]:
    """Compute everything for a symbol and return a dict; also triggers Telegram alerts (same as before)."""

    ctx = prepare_symbol_context(symbol, tf)
    df = ctx["df"]
    comp = ctx["comp"]
    rng = ctx["rng"]
    sp_bands = ctx["sp_bands"]

    params = build_strategy_params(symbol, df, sp_bands)
    trades_df, summary, active_open, pending_list = backtest_xo_logic(df, comp, rng, params, force_eod_close=False)

    notify_closed_trades(symbol, trades_df, tf)
    notify_live_states(symbol, tf, comp, df, pending_list, active_open)

    if "all_trades" not in st.session_state:
        st.session_state["all_trades"] = []
    tdf = trades_df.copy()
    tdf["symbol"] = symbol
    st.session_state["all_trades"].append(tdf)

    return {"df": df, "comp": comp, "trades_df": trades_df, "summary": summary,
            "active_open": active_open, "pending_list": pending_list}

def render_symbol(symbol: str, res: Dict[str,Any]):
    """Render the SAME UI you had for run_for_symbol, using precomputed results (no recompute)."""
    levels = {"support": [], "resistance": []}
    df = res["df"]; comp = res["comp"]; trades_df = res["trades_df"]
    summary = res["summary"]; active_open = res["active_open"]; pending_list = res["pending_list"]

    st.header(symbol)

    st.subheader("Composite Levels")
    st.write(pd.DataFrame({"Level":["VAL","POC","VAH"], "Price":[comp["VAL"], comp["POC"], comp["VAH"]]}))

    st.subheader("Active (open) position now")
    if active_open is None:
        st.info("No currently-open position under the rules.")
    else:
        last = float(df["close"].iloc[-1])
        entry = float(active_open["entry"]); sl = float(active_open["sl"])
        tp1 = float(active_open.get("tp1", np.nan)); tp2 = float(active_open.get("tp2", np.nan))
        side = active_open["side"]
        risk = abs(entry - sl) or float("nan")
        u_pnl = (last - entry) if side == "long" else (entry - last)
        u_R = (u_pnl / risk) if (risk == risk and risk > 0) else float("nan")
        st.dataframe(pd.DataFrame([{
            "Side": side, "Reason": active_open.get("reason",""), "Entry Time": active_open.get("entry_time"),
            "Entry": entry, "SL": sl, "TP1": tp1, "TP2": tp2, "Last": last, "Unrealized PnL": u_pnl, "Unrealized R": u_R
        }]).style.format({
            "Entry":"{:,.2f}","SL":"{:,.2f}","TP1":"{:,.2f}","TP2":"{:,.2f}",
            "Last":"{:,.2f}","Unrealized PnL":"{:,.4f}","Unrealized R":"{:.2f}"
        }), use_container_width=True)

    st.subheader("Pending orders (awaiting fill)")
    if not pending_list:
        st.info("No pending orders right now.")
    else:
        rows = []
        last_ts = df.index[-1]
        for p in pending_list:
            entry = float(p["entry"]); sl=float(p["sl"]); tp1=float(p.get("tp1", np.nan)); tp2=float(p.get("tp2", np.nan))
            side=p["side"]; risk=abs(entry-sl) or float("nan")
            R1=((tp1-entry)/risk) if (risk==risk and risk>0 and tp1==tp1) else float("nan")
            R2=((tp2-entry)/risk) if (risk==risk and risk>0 and tp2==tp2) else float("nan")
            created=p.get("created_ts"); age_mins=None
            try:
                if created is not None:
                    age_mins=(pd.to_datetime(last_ts) - pd.to_datetime(created)).total_seconds()/60.0
            except Exception:
                pass
            rows.append({"Side":side,"Reason":p.get("reason",""),"Created":created,"Age":_fmt_ddhhmm(age_mins),
                "Entry":entry,"SL":sl,"TP1":tp1,"TP2":tp2,"Planned R→TP1":R1,"Planned R→TP2":R2})
        dfp = pd.DataFrame(rows)[["Side","Reason","Created","Age","Entry","SL","TP1","TP2","Planned R→TP1","Planned R→TP2"]]
        st.dataframe(
            dfp.style.format({
                "Entry":"{:,.2f}","SL":"{:,.2f}","TP1":"{:,.2f}","TP2":"{:,.2f}",
                "Planned R→TP1":"{:.2f}","Planned R→TP2":"{:.2f}"
            }),
            use_container_width=True
        )


    st.subheader("Backtest Summary")
    c = st.columns(5)
    c[0].metric("Trades", summary["trades"])
    c[1].metric("Hit Rate", "—" if math.isnan(summary["hit_rate"]) else f"{round(summary['hit_rate']*100,1)}%")
    c[2].metric("Total PnL (units)", round(summary["total_pnl"],2))
    c[3].metric("Avg R", "—" if math.isnan(summary["avg_R"]) else round(summary["avg_R"],3))
    c[4].metric("Total R", round(summary["total_R"],2))

    st.subheader("Trades")
    if trades_df.empty:
        st.info("No trades under current params/period.")
        return

    show_cols = [c for c in ["entry_time","exit_time","side","reason","exit_kind","entry","sl","tp1","tp2","exit","pnl","pnl_R","hold_minutes","hold_bars"] if c in trades_df.columns]
    st.dataframe(trades_df[show_cols], use_container_width=True)

    # ---------------- Support / Resistance Chart ----------------
    with st.expander("Price with Support/Resistance (selected timeframe)", expanded=True):
        st.caption("Swing-based S/R with de-duplication; composite VAL/POC/VAH included.")
        colA, colB, colC, colD = st.columns(4)
        lookback_bars = colA.number_input("Lookback bars", value=300, min_value=50, max_value=5000, step=50)
        swing         = colB.number_input("Swing window", value=3, min_value=1, max_value=20, step=1,
                                          help="Local extrema over a (2*swing+1)-bar window.")
        max_levels    = colC.number_input("Max levels/side", value=3, min_value=1, max_value=30, step=1)
        min_dist_bps  = colD.number_input("Min separation (bps)", value=12.0, min_value=1.0, max_value=200.0, step=1.0)

        # Build composite dict (VAL/POC/VAH) for context lines
        comp_levels = {"VAL": comp.get("VAL", np.nan), "POC": comp.get("POC", np.nan), "VAH": comp.get("VAH", np.nan)}

        try:
            levels = _find_sr_levels(
                df=df,
                lookback_bars=int(lookback_bars),
                swing=int(swing),
                max_levels=int(max_levels),
                min_distance_bps=float(min_dist_bps),
                include_composite=comp_levels,
            )
        except Exception as e:
            st.warning(f"S/R level calc error: {e}")
            levels = {"support": [], "resistance": []}

    # Plot close with S/R horizontals (date-aware axis)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig_sr, ax_sr = plt.subplots(figsize=(9, 4.5))
        df_plot = df.iloc[-int(lookback_bars):] if len(df) > int(lookback_bars) else df

        # Price (close) line
        ax_sr.plot(df_plot.index, df_plot["close"], linewidth=1.2)

        # S/R lines
        for y in levels["support"]:
            ax_sr.axhline(y, color="tab:green", linestyle="--", linewidth=1.0, alpha=0.8)

        for y in levels["resistance"]:
            ax_sr.axhline(y, color="tab:red", linestyle="-.", linewidth=1.0, alpha=0.8)

        # Composite lines (thicker)
        if np.isfinite(comp_levels.get("VAL", np.nan)):
            ax_sr.axhline(comp_levels["VAL"], color="black", linewidth=1.4, alpha=0.9)
        if np.isfinite(comp_levels.get("POC", np.nan)):
            ax_sr.axhline(comp_levels["POC"], color="black", linewidth=1.6, alpha=0.9)
        if np.isfinite(comp_levels.get("VAH", np.nan)):
            ax_sr.axhline(comp_levels["VAH"], color="black", linewidth=1.4, alpha=0.9)

        ax_sr.set_title(f"{symbol} — Price with Support / Resistance ({_esc(tf)})")
        ax_sr.set_ylabel("Price")
        ax_sr.grid(True, alpha=0.3)

        # Date axis: concise labels
        locator = AutoDateLocator(minticks=3, maxticks=8)
        ax_sr.xaxis.set_major_locator(locator)
        ax_sr.xaxis.set_major_formatter(ConciseDateFormatter(locator))
        fig_sr.autofmt_xdate()

        st.pyplot(fig_sr)

        # Legend note (text) to keep chart clean
        st.caption("Legend: price (line), support (dashed), resistance (dash-dot), POC/VAH/VAL (bold).")
    except Exception as _e:
        st.warning(f"S/R plot error: {_e}")


    # Per-symbol chart (same tidy x-axis)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.ticker import MaxNLocator

        st.markdown("#### Cumulative R (closed trades)")
        fig_cr, ax_cr = plt.subplots(figsize=(7,3.5))
        y_cr = trades_df.reset_index(drop=True)["pnl_R"].fillna(0).cumsum().values
        ax_cr.plot(range(len(y_cr)), y_cr, marker="o", markersize=3 if len(y_cr) > 80 else 5)
        ax_cr.set_xlabel("Trade #"); ax_cr.set_ylabel("Cumulative R"); ax_cr.set_title(f"{symbol} — Cumulative R")
        ax_cr.grid(True, which="both", axis="both", alpha=0.3); ax_cr.margins(x=0.01)
        ax_cr.xaxis.set_major_locator(MaxNLocator(nbins=9, integer=True, prune="both"))
        ax_cr.tick_params(axis="x", labelsize=8)
        st.pyplot(fig_cr)
    except Exception as _e:
        st.warning(f"Plot error: {_e}")

def _find_sr_levels(
    df: pd.DataFrame,
    lookback_bars: int = 300,
    swing: int = 3,
    max_levels: int = 8,
    min_distance_bps: float = 12.0,
    include_composite: dict | None = None,   # e.g., {"VAL":val, "POC":poc, "VAH":vah}
) -> dict:
    """
    Extract simple support/resistance levels:
    - Recent swing highs (resistance) and swing lows (support) within the last N bars
    - Optionally include composite levels (VAL/POC/VAH)
    - De-duplicate levels within 'min_distance_bps'
    Returns {"support":[...], "resistance":[...]}
    """
    if df.empty:
        return {"support": [], "resistance": []}

    sub = df.iloc[-lookback_bars:].copy() if len(df) > lookback_bars else df.copy()
    H = pd.to_numeric(sub["high"], errors="coerce")
    L = pd.to_numeric(sub["low"], errors="coerce")
    idx = sub.index

    # Swing highs/lows (centered rolling window)
    # A swing high: the high equals the local max within the window and is unique (to avoid flat regions)
    roll_max = H.rolling(window=2*swing+1, center=True).max()
    roll_min = L.rolling(window=2*swing+1, center=True).min()
    swing_highs = sub[H.notna() & (H == roll_max)].dropna(subset=["high"])
    swing_lows  = sub[L.notna() & (L == roll_min)].dropna(subset=["low"])

    cand_res = sorted(pd.to_numeric(swing_highs["high"], errors="coerce").dropna().unique().tolist(), reverse=True)
    cand_sup = sorted(pd.to_numeric(swing_lows["low"],  errors="coerce").dropna().unique().tolist())

    # Optionally include composite levels
    if include_composite:
        for k in ("VAL", "POC", "VAH"):
            v = include_composite.get(k, None)
            if v is None or not np.isfinite(v):
                continue
            # POC/VAH generally act as resistance, VAL as support (conservative default)
            if k == "VAL":
                cand_sup.append(float(v))
            else:
                cand_res.append(float(v))

    # De-dup using a bps threshold (distance measured relative to the level)
    def _dedup(levels: list[float], is_res: bool) -> list[float]:
        levels_sorted = sorted(levels, reverse=is_res)
        out: list[float] = []
        for lv in levels_sorted:
            if not np.isfinite(lv):
                continue
            keep = True
            for taken in out:
                bps = abs(lv - taken) / taken * 10000.0
                if bps <= min_distance_bps:
                    keep = False
                    break
            if keep:
                out.append(float(lv))
            if len(out) >= max_levels:
                break
        # restore natural order: supports ascending, resistances descending
        return sorted(out) if not is_res else sorted(out, reverse=True)

    sup = _dedup(cand_sup, is_res=False)
    res = _dedup(cand_res, is_res=True)
    return {"support": sup, "resistance": res}


# Driver: compute for ALL symbols once; store in session
try:
    run_btn
except NameError:
    run_btn = True
if run_btn:
    for k in ("tg_seen_pending","tg_seen_active","tg_seen_closed","tg_hits_seen"):
        st.session_state[k] = set()
    st.session_state["results"] = {}       # symbol -> result dict
    st.session_state["all_trades"] = []    # rebuild aggregation

    for sym in symbols_final:
        try:
            st.session_state["results"][sym] = compute_for_symbol(sym)
        except Exception as _e:
            st.warning(f"{sym}: compute error — {_e}")

# ---- Render section (no recompute required) ----
if st.session_state.get("results"):
    if selection == "ALL":
        # Combined charts across all symbols (same as your existing section)
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import pandas as _pd

        all_df = _pd.concat([d["trades_df"].assign(symbol=s) for s, d in st.session_state["results"].items()], ignore_index=True)
        if all_df.empty or ("pnl_R" not in all_df.columns):
            st.info("No closed trades across symbols for this run (nothing to plot yet).")
            st.stop()



        # Bar chart: total R per symbol
        st.subheader("All Symbols — Total R (closed trades)")
        try:
            totals = all_df.groupby("symbol")["pnl_R"].sum().sort_values(ascending=False)
            fig_bar, ax_bar = plt.subplots(figsize=(8,4))
            labels = totals.index.astype(str).tolist()
            values = totals.values
            ax_bar.bar(labels, values)
            ax_bar.set_ylabel("Total R")
            ax_bar.set_xlabel("Symbol")
            ax_bar.set_title("Total R per Symbol (all closed trades)")
            ax_bar.grid(axis="y", alpha=0.3)

            # De-clutter x labels
            ax_bar.tick_params(axis="x", labelrotation=45, labelsize=8)
            for lbl in ax_bar.get_xticklabels():
                lbl.set_ha("right")

            # If too many symbols, only show every n-th label
            if len(labels) > 16:
                step = max(1, len(labels) // 12)
                for i, lbl in enumerate(ax_bar.get_xticklabels()):
                    lbl.set_visible(i % step == 0)

            st.pyplot(fig_bar)

        except Exception as _e:
            st.warning(f"Combined totals plot error: {_e}")

        # Cumulative R across all trades (daily aggregation to reduce noise)
        try:
            st.markdown("#### All Symbols — Cumulative R over time")
            df_time = all_df.dropna(subset=["exit_time", "pnl_R"]).copy()
            df_time["exit_time"] = _pd.to_datetime(df_time["exit_time"], utc=True, errors="coerce")
            df_time["pnl_R"] = _pd.to_numeric(df_time["pnl_R"], errors="coerce")
            df_time = df_time.dropna(subset=["exit_time", "pnl_R"]).sort_values("exit_time")

            s_daily = df_time.set_index("exit_time").resample("1D")["pnl_R"].sum().fillna(0.0)
            cumR = s_daily.cumsum()

            fig_all_cr, ax_all_cr = plt.subplots(figsize=(8,4))
            ax_all_cr.plot(cumR.index, cumR.values)

            # Concise, auto-scaling date labels
            locator = AutoDateLocator(minticks=3, maxticks=8)
            ax_all_cr.xaxis.set_major_locator(locator)
            ax_all_cr.xaxis.set_major_formatter(ConciseDateFormatter(locator))

            ax_all_cr.set_xlabel("Date")
            ax_all_cr.set_ylabel("Cumulative R (Daily aggregated)")
            ax_all_cr.set_title("All Symbols — Cumulative R (by day)")
            ax_all_cr.grid(True, alpha=0.3)
            fig_all_cr.autofmt_xdate()  # final tidy-up
            st.pyplot(fig_all_cr)

        except Exception as _e:
            st.warning(f"Combined cumulative plot error: {_e}")

        # Cumulative R across all symbols as a function of number of trades
        try:
            st.markdown("#### All Symbols — Cumulative R vs # Trades")
            df_tr = all_df.dropna(subset=["exit_time", "pnl_R"]).copy()
            df_tr["exit_time"] = _pd.to_datetime(df_tr["exit_time"], utc=True, errors="coerce")
            df_tr["pnl_R"] = _pd.to_numeric(df_tr["pnl_R"], errors="coerce")
            df_tr = df_tr.dropna(subset=["exit_time", "pnl_R"]).sort_values("exit_time")

            # Build trade index (1..N) and cumulative R
            y = df_tr["pnl_R"].fillna(0.0).cumsum().to_numpy()
            x = np.arange(1, len(y) + 1, dtype=int)

            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig_nr, ax_nr = plt.subplots(figsize=(8, 4))
            ax_nr.plot(x, y, marker="o", markersize=3 if len(y) > 100 else 5)
            ax_nr.set_xlabel("Trade # (all symbols, time-ordered)")
            ax_nr.set_ylabel("Cumulative R")
            ax_nr.set_title("All Symbols — Cumulative R vs Number of Trades")
            ax_nr.grid(True, alpha=0.3)
            ax_nr.margins(x=0.01)

            # Keep x-axis uncluttered: show at most ~9 integer ticks, prune edges
            ax_nr.xaxis.set_major_locator(MaxNLocator(nbins=9, integer=True, prune="both"))
            ax_nr.tick_params(axis="x", labelsize=8)

            st.pyplot(fig_nr)
        except Exception as _e:
            st.warning(f"Cumulative R vs trades plot error: {_e}")


        # Optional: per-symbol cumulative R over time (daily)
        try:
            st.markdown("#### All Symbols — Per-Symbol Cumulative R (by day)")
            all_df2 = all_df.dropna(subset=["exit_time", "pnl_R"]).copy()
            all_df2["exit_time"] = _pd.to_datetime(all_df2["exit_time"], utc=True, errors="coerce")
            all_df2["pnl_R"] = _pd.to_numeric(all_df2["pnl_R"], errors="coerce")
            all_df2 = all_df2.dropna(subset=["exit_time", "pnl_R"])

            groups = []
            for sym, g in all_df2.groupby("symbol"):
                s = g.set_index("exit_time").resample("1D")["pnl_R"].sum().fillna(0.0).cumsum()
                groups.append((sym, s))

            if groups:
                fig_ps, ax_ps = plt.subplots(figsize=(8,4))
                for sym, s in groups:
                    ax_ps.plot(s.index, s.values, label=str(sym))

                # Concise, auto-scaling date labels
                locator2 = AutoDateLocator(minticks=3, maxticks=8)
                ax_ps.xaxis.set_major_locator(locator2)
                ax_ps.xaxis.set_major_formatter(ConciseDateFormatter(locator2))

                ax_ps.set_xlabel("Date")
                ax_ps.set_ylabel("Cumulative R")
                ax_ps.set_title("Per-Symbol Cumulative R (daily aggregated)")
                ax_ps.grid(True, alpha=0.3)

                # Keep legends readable when many lines
                n = len(groups)
                ax_ps.legend(
                    loc="upper left",
                    ncols=2 if n <= 10 else 3,
                    fontsize=7 if n <= 12 else 6,
                    frameon=False
                )

                fig_ps.autofmt_xdate()
                st.pyplot(fig_ps)

        except Exception as _e:
            st.warning(f"Per-symbol cumulative plot error: {_e}")

    elif selection == "Pending":
        # Cross-symbol Pending table
        rows = []
        for sym, res in st.session_state["results"].items():
            df = res["df"]
            last_ts = df.index[-1] if not df.empty else None

            for p in (res.get("pending_list") or []):
                # Pull fields
                side = str(p.get("side", "")).lower()
                reason = p.get("reason", "")
                created = p.get("created_ts")

                # Numbers (safe casts)
                try:
                    entry = float(p["entry"]); sl = float(p["sl"])
                    tp1 = float(p.get("tp1", float("nan"))); tp2 = float(p.get("tp2", float("nan")))
                except Exception:
                    entry = p.get("entry"); sl = p.get("sl"); tp1 = p.get("tp1"); tp2 = p.get("tp2")

                risk = abs(entry - sl) if isinstance(entry, (int, float)) and isinstance(sl, (int, float)) else float("nan")

                # Planned R to TP1/TP2 (by side)
                def _planned_R(tp):
                    if not (isinstance(tp, (int, float)) and isinstance(risk, (int, float)) and risk > 0 and np.isfinite(tp)):
                        return float("nan")
                    if side == "long":
                        return (tp - entry) / risk
                    else:
                        return (entry - tp) / risk

                R1 = _planned_R(tp1)
                R2 = _planned_R(tp2)

                # Age since created (min), using last bar time
                age_mins = None
                try:
                    if created is not None and last_ts is not None:
                        age_mins = (pd.to_datetime(last_ts) - pd.to_datetime(created)).total_seconds() / 60.0
                except Exception:
                    pass

                rows.append({
                    "Symbol": sym,
                    "Side": side.upper(),
                    "Reason": reason,
                    "Created": created,
                    "Age": _fmt_ddhhmm(age_mins),
                    "Entry": entry,
                    "SL": sl,
                    "TP1": tp1,
                    "TP2": tp2,
                    "Planned R→TP1": R1,
                    "Planned R→TP2": R2,
                })


        st.subheader("All Symbols — Pending Orders")
        if not rows:
            st.info("No pending orders across symbols.")
        else:
            dfp = pd.DataFrame(rows)[["Symbol","Side","Reason","Created","Age","Entry","SL","TP1","TP2","Planned R→TP1","Planned R→TP2"]]
            st.dataframe(
                dfp.style.format({
                    "Entry":"{:,.2f}","SL":"{:,.2f}","TP1":"{:,.2f}","TP2":"{:,.2f}",
                    "Planned R→TP1":"{:.2f}","Planned R→TP2":"{:.2f}"
                }),
                use_container_width=True
            )


    elif selection == "Active":
        # Cross-symbol Active positions table
        rows = []
        for sym, res in st.session_state["results"].items():
            a = res.get("active_open")
            if a is None:
                continue

            df = res["df"]
            last_ts = df.index[-1] if not df.empty else None
            last = float(df["close"].iloc[-1])

            # Pull numbers
            entry = float(a["entry"]); sl = float(a["sl"])
            tp1 = float(a.get("tp1", float("nan"))); tp2 = float(a.get("tp2", float("nan")))
            side = str(a.get("side","")).lower()

            risk = abs(entry - sl) if np.isfinite(entry) and np.isfinite(sl) else float("nan")
            u_pnl = (last - entry) if side == "long" else (entry - last)
            u_R = (u_pnl / risk) if (np.isfinite(risk) and risk > 0) else float("nan")

            # Time since entry (min), use last bar time vs entry_time
            age_mins = None
            try:
                et = pd.to_datetime(a.get("entry_time"))
                if last_ts is not None and not pd.isna(et):
                    age_mins = (pd.to_datetime(last_ts) - et).total_seconds() / 60.0
            except Exception:
                pass

            # Remaining R to TP1/TP2 from current price (positive = R to go)
            def _R_to(tp):
                if not (np.isfinite(tp) and np.isfinite(risk) and risk > 0):
                    return float("nan")
                if side == "long":
                    return (tp - last) / risk
                else:
                    return (last - tp) / risk

            R_to_TP1 = _R_to(tp1)
            R_to_TP2 = _R_to(tp2)

            rows.append({
                "Symbol": sym,
                "Side": side.upper(),
                "Reason": a.get("reason",""),
                "Entry Time": a.get("entry_time"),
                "Age": _fmt_ddhhmm(age_mins),
                "Entry": entry,
                "SL": sl,
                "TP1": tp1,
                "TP2": tp2,
                "Last": last,
                "Unrealized PnL": u_pnl,
                "Unrealized R": u_R,
                "R→TP1": R_to_TP1,
                "R→TP2": R_to_TP2,
            })



        st.subheader("All Symbols — Active Positions")
        if not rows:
            st.info("No active positions across symbols.")
        else:
            dfa = pd.DataFrame(rows)[[
                "Symbol","Side","Reason","Entry Time","Age",
                "Entry","SL","TP1","TP2","Last",
                "Unrealized PnL","Unrealized R","R→TP1","R→TP2"
            ]]
            st.dataframe(
                dfa.style.format({
                    "Entry":"{:,.2f}","SL":"{:,.2f}","TP1":"{:,.2f}","TP2":"{:,.2f}",
                    "Last":"{:,.2f}","Unrealized PnL":"{:,.4f}","Unrealized R":"{:.2f}",
                    "R→TP1":"{:.2f}","R→TP2":"{:.2f}",
                }),
                use_container_width=True
            )


    else:
        # A single symbol's full view (precomputed; no re-run needed)
        res = st.session_state["results"].get(selection)
        if res is None:
            st.info("Click **Run** to prepare results for the selected symbol.")
        else:
            render_symbol(selection, res)
else:
    st.info("Pick settings and click **Run** to compute all symbols.")
