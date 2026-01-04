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
    w = resample_ohlcv(df, "1W")
    d3 = df.resample("3D").agg({"open":"first","high":"max","low":"min","close":"last","volume":"sum"})
    def _state(x):
        if x.empty: return {"bull_ok": False, "bear_cross": True}
        e12, e25 = _ema(x["close"], 12), _ema(x["close"], 25)
        bull_ok = (x["close"].iloc[-1] > e12.iloc[-1]) and (x["close"].iloc[-1] > e25.iloc[-1])
        bear_cross = (e12.shift(1).iloc[-1] > e25.shift(1).iloc[-1]) and (e12.iloc[-1] <= e25.iloc[-1])
        return {"bull_ok": bull_ok, "bear_cross": bear_cross}
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

# --- HTF EMA 12/25 state (weekly & 3-day) ---
def _ema(series, span): 
    return series.ewm(span=span, adjust=False).mean()

def _resample_ohlc(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    o = df["open"].resample(rule, label="right", closed="right").first()
    h = df["high"].resample(rule, label="right", closed="right").max()
    l = df["low"].resample(rule, label="right", closed="right").min()
    c = df["close"].resample(rule, label="right", closed="right").last()
    v = df["volume"].resample(rule, label="right", closed="right").sum()
    return pd.DataFrame({"open":o,"high":h,"low":l,"close":c,"volume":v}).dropna()

def htf_ema_state(df: pd.DataFrame) -> dict:
    # Weekly
    w = _resample_ohlc(df, "7D")
    w["ema12"], w["ema25"] = _ema(w["close"],12), _ema(w["close"],25)
    w["bull_ok"] = (w["close"] > w["ema12"]) & (w["close"] > w["ema25"])
    w["bear_cross"] = (w["ema12"].shift(1) > w["ema25"].shift(1)) & (w["ema12"] <= w["ema25"])

    # 3-Day
    d3 = _resample_ohlc(df, "3D")
    d3["ema12"], d3["ema25"] = _ema(d3["close"],12), _ema(d3["close"],25)
    d3["bull_ok"] = (d3["close"] > d3["ema12"]) & (d3["close"] > d3["ema25"])
    d3["bear_cross"] = (d3["ema12"].shift(1) > d3["ema25"].shift(1)) & (d3["ema12"] <= d3["ema25"])

    return {
        "weekly":  {"bull_ok": bool(w["bull_ok"].iloc[-1]) if len(w) else False,
                    "bear_cross": bool(w["bear_cross"].iloc[-1]) if len(w) else True},
        "d3":      {"bull_ok": bool(d3["bull_ok"].iloc[-1]) if len(d3) else False,
                    "bear_cross": bool(d3["bear_cross"].iloc[-1]) if len(d3) else True},
    }


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

def composite_profile(df: pd.DataFrame, days: int = 35, n_bins: int = 240) -> Dict[str, Any]:
    window = df[df.index >= (df.index.max() - pd.Timedelta(days=days))].copy()
    if window.empty:
        return {"VAL": np.nan, "POC": np.nan, "VAH": np.nan, "hist": pd.DataFrame({"price":[],"vol":[]})}
    vp = volume_profile(window, n_bins=n_bins)
    val, poc, vah = value_area(vp, 0.70)
    return {"VAL": val, "POC": poc, "VAH": vah, "hist": vp}

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

class LevelTriggerCfg:
    def __init__(self, hold_bars: int = 2, retest_window: int = 6, tolerance: float = 0.0005):
        self.hold_bars = int(hold_bars)
        self.retest_window = int(retest_window)
        self.tolerance = float(tolerance)

def breakout_or_deviation(df: pd.DataFrame, level: float, side: str, retest: int, tol_bps: float):
    """Return 'continuation'|'deviation'|None; side='up' (range high) or 'down' (range low)."""
    try:
        closes = df["close"]
        eps = (level * tol_bps / 1e4)
        if side == "up":
            broke = closes > (level + eps)
            if not broke.any():
                return None
            i = int(broke.idxmax()==closes.index[0]) and 0 or list(broke).index(True)
            win = df.iloc[i+1:i+1+max(1, int(retest))]
            if not win.empty and (win["low"] >= (level - eps)).all():
                return "continuation"
            if (i+1 < len(df)) and (closes.iloc[i+1] < (level - eps)):
                return "deviation"
        else:
            broke = closes < (level - eps)
            if not broke.any():
                return None
            i = int(broke.idxmax()==closes.index[0]) and 0 or list(broke).index(True)
            win = df.iloc[i+1:i+1+max(1, int(retest))]
            if not win.empty and (win["high"] <= (level + eps)).all():
                return "continuation"
            if (i+1 < len(df)) and (closes.iloc[i+1] > (level + eps)):
                return "deviation"
        return None
    except Exception:
        pass
    return None


def crossed_and_held(df: pd.DataFrame, level: float, side: str, cfg: LevelTriggerCfg) -> List[pd.Timestamp]:
    """Return timestamps where close crossed & held for `hold_bars`, then retest within `retest_window`."""
    if not np.isfinite(level):
        return []
    closes = df["close"].values
    highs = df["high"].values
    lows  = df["low"].values
    eps = level * cfg.tolerance
    out: List[pd.Timestamp] = []

    def ok_retest(i0: int) -> bool:
        w = slice(i0, min(i0+cfg.retest_window, len(df)))
        if side=='up':
            return np.any(lows[w] >= level - eps)
        else:
            return np.any(highs[w] <= level + eps)

    run_len = 0
    cond = (closes > level) if side=='up' else (closes < level)
    for i, c in enumerate(cond):
        run_len = run_len+1 if c else 0
        if run_len == cfg.hold_bars and ok_retest(i+1):
            out.append(df.index[i])
    return out

def cvd_proxy(df: pd.DataFrame) -> pd.Series:
    sign = np.where(df["close"].diff().fillna(0)>=0, 1, -1)
    return pd.Series((sign * df["volume"].values).cumsum(), index=df.index, name="cvd_proxy")

def level_to_level_plan_safe(entry_px: float, side: str, targets: List[float],
                             atrv: float, atr_mult: float,
                             min_stop_bps: int, min_tp1_bps: int) -> Dict[str,float]:
    assert side in ("long","short")
    sign = 1 if side == "long" else -1

    sl_raw = entry_px - sign * (atr_mult * atrv)
    extra_atr = float(params.get("_extra_atr_buffer", 0.0)) if "params" in globals() else 0.0
    sl_raw = entry_px - sign * ((atr_mult + extra_atr) * atrv)
    min_stop_abs = entry_px * (min_stop_bps / 10000.0)
    sl = sl_raw if abs(entry_px - sl_raw) >= min_stop_abs else entry_px - sign * min_stop_abs

    min_tp1_abs = entry_px * (min_tp1_bps / 10000.0)

    def _first_far_enough(ts):
        for t in ts:
            if np.isfinite(t) and abs(t - entry_px) >= min_tp1_abs and (sign*(t-entry_px) > 0):
                return float(t)
        return None

    tp1 = _first_far_enough(targets)
    if tp1 is None:
        tp1 = entry_px + sign * max(min_tp1_abs, atr_mult * atrv)

    ts_same_side = [t for t in targets if np.isfinite(t) and sign*(t-entry_px) > 0]
    if ts_same_side:
        tp2 = max(ts_same_side) if side == "long" else min(ts_same_side)
        if sign*(tp2 - tp1) <= 0:
            tp2 = tp1 + sign * max(min_tp1_abs, 2 * atr_mult * atrv)
    else:
        tp2 = tp1 + sign * max(min_tp1_abs, 2 * atr_mult * atrv)

    return {"sl": float(sl), "tp1": float(tp1), "tp2": float(tp2)}

def _rr(entry: float, sl: float, tp: float) -> float:
    risk = abs(entry - sl)
    return (abs(tp - entry) / risk) if risk > 0 and np.isfinite(tp) else np.nan

# ----------------- Backtest & live calculation -----------------

def backtest_xo_logic(df: pd.DataFrame, comp: Dict[str,Any], params: Dict[str,Any], force_eod_close: bool = False):
    """
    Returns: trades_df, summary, open_pos (active if any), pending_list (unfilled orders)
    params expects:
      atr_period, ema_fast, ema_slow, hold_bars, retest_bars, tol,
      atr_mult, min_stop_bps, min_tp1_bps, min_rr,
      use_val_reclaim, use_vah_break, use_sp_cont, sp_bands, entry_style
    """
    df = df.copy()
    df["atr"] = atr(df, params["atr_period"])
    df["cvd_proxy"] = cvd_proxy(df)
    df = ema_regime(df, params["ema_fast"], params["ema_slow"])

    trades: List[Dict[str,Any]] = []
    open_pos: Optional[Dict[str,Any]] = None
    pending_order: Optional[Dict[str,Any]] = None

    VAL, POC, VAH = comp["VAL"], comp["POC"], comp["VAH"]
    trig_cfg = LevelTriggerCfg(params["hold_bars"], params["retest_bars"], params["tol"])
    val_reclaims = set(crossed_and_held(df, VAL, "up", trig_cfg)) if params["use_val_reclaim"] else set()
    vah_breaks   = set(crossed_and_held(df, VAH, "down", trig_cfg)) if params["use_vah_break"] else set()
    sp_bands = params.get("sp_bands", []) or []

    atr_mult     = float(params["atr_mult"])
    min_stop_bps = int(params["min_stop_bps"])
    min_tp1_bps  = int(params["min_tp1_bps"])
    min_rr       = float(params["min_rr"])
    entry_style  = params.get("entry_style", "Stop-through")
    filters_func = params.get("filters_func", None)

    # % to exit at TP1; remainder rides to TP2/SL
    tp1_exit_pct = int(params.get("tp1_exit_pct", 100))
    if tp1_exit_pct not in (0, 25, 50, 75, 100):
        tp1_exit_pct = 100
    tp1_exit_frac = tp1_exit_pct / 100.0


    for ts, row in df.iterrows():
        px = float(row["close"])
        atrv = float(row["atr"]) if np.isfinite(row["atr"]) else np.nan
        regime = row["regime"]

        # Try to fill pending on later bars
        if pending_order and open_pos is None and ts > pending_order["created_ts"]:
            lo = float(row["low"]); hi = float(row["high"])
            e  = float(pending_order["entry"])
            if lo <= e <= hi:
                open_pos = {
                    "side": "long" if pending_order["side"] == "long" else "short",
                    "entry_time": ts,
                    "entry": e,
                    "sl":   float(pending_order["sl"]),
                    "tp1":  float(pending_order["tp1"]),
                    "tp2":  float(pending_order.get("tp2", np.nan)),
                    "reason": pending_order["reason"],
                    "qty": 1.0,
                }
                pending_order = None

        # If flat & no pending, evaluate triggers
        if open_pos is None and pending_order is None and np.isfinite(atrv):
            # VAL reclaim → Long
            if params["use_val_reclaim"] and (ts in val_reclaims) and np.isfinite(VAL):
                if entry_style == "Stop-through":
                    entry_target = float(VAL * (1 + params["tol"]))  # stop-buy above
                    plan = level_to_level_plan_safe(entry_target, "long", [POC, VAH], atrv, atr_mult, min_stop_bps, min_tp1_bps)
                    plan["sl"] = min(plan["sl"], float(row["low"])) 
                    rr1 = _rr(entry_target, plan["sl"], plan["tp1"])
                    ok_gate = True
                    if filters_func is not None:
                        ok_gate = bool(filters_func(ts, "long", entry_target, "VAL_reclaim", df, comp, row))
                    if np.isfinite(rr1) and rr1 >= min_rr and ok_gate:
                        pending_order = {"side":"long","entry":float(entry_target),
                                        "sl":float(plan["sl"]),"tp1":float(plan["tp1"]),"tp2":float(plan["tp2"]),
                                        "reason":"VAL_reclaim","created_ts":ts}
                elif entry_style == "Limit-on-retest":
                    entry_target = float(VAL * (1 - params["tol"]))  # buy-limit into retest
                    plan = level_to_level_plan_safe(entry_target, "long", [POC, VAH], atrv, atr_mult, min_stop_bps, min_tp1_bps)
                    plan["sl"] = min(plan["sl"], float(row["low"])) 
                    rr1 = _rr(entry_target, plan["sl"], plan["tp1"])
                    ok_gate = True
                    if filters_func is not None:
                        ok_gate = bool(filters_func(ts, "long", entry_target, "VAL_reclaim_limit_retest", df, comp, row))
                    if np.isfinite(rr1) and rr1 >= min_rr and ok_gate:
                        pending_order = {"side":"long","entry":float(entry_target),
                                        "sl":float(plan["sl"]),"tp1":float(plan["tp1"]),"tp2":float(plan["tp2"]),
                                        "reason":"VAL_reclaim_limit_retest","created_ts":ts}
                else:  # Market-on-trigger
                    entry_target = float(px)
                    plan = level_to_level_plan_safe(entry_target, "long", [POC, VAH], atrv, atr_mult, min_stop_bps, min_tp1_bps)
                    plan["sl"] = min(plan["sl"], float(row["low"])) 
                    rr1 = _rr(entry_target, plan["sl"], plan["tp1"])
                    ok_gate = True
                    if filters_func is not None:
                        ok_gate = bool(filters_func(ts, "long", entry_target, "VAL_reclaim_market", df, comp, row))
                    if np.isfinite(rr1) and rr1 >= min_rr and ok_gate:
                        open_pos = {"side":"long","entry_time":ts,"entry":float(entry_target),
                                    "sl":float(plan["sl"]),"tp1":float(plan["tp1"]),"tp2":float(plan["tp2"]),
                                    "reason":"VAL_reclaim_market"}

            # VAH breakdown → Short
            elif params["use_vah_break"] and (ts in vah_breaks) and np.isfinite(VAH):
                if entry_style == "Stop-through":
                    entry_target = float(VAH * (1 - params["tol"]))  # stop-sell below
                    plan = level_to_level_plan_safe(entry_target, "short", [POC, VAL], atrv, atr_mult, min_stop_bps, min_tp1_bps)
                    rr1 = _rr(entry_target, plan["sl"], plan["tp1"])
                    ok_gate = True
                    if filters_func is not None:
                        ok_gate = bool(filters_func(ts, "short", entry_target, "VAH_breakdown", df, comp, row))
                    if np.isfinite(rr1) and rr1 >= min_rr and ok_gate:
                        pending_order = {"side":"short","entry":float(entry_target),
                                         "sl":float(plan["sl"]),"tp1":float(plan["tp1"]),"tp2":float(plan["tp2"]),
                                         "reason":"VAH_breakdown","created_ts":ts}
                elif entry_style == "Limit-on-retest":
                    entry_target = float(VAH * (1 + params["tol"]))  # sell-limit into retest
                    plan = level_to_level_plan_safe(entry_target, "short", [POC, VAL], atrv, atr_mult, min_stop_bps, min_tp1_bps)
                    rr1 = _rr(entry_target, plan["sl"], plan["tp1"])
                    ok_gate = True
                    if filters_func is not None:
                        ok_gate = bool(filters_func(ts, "short", entry_target, "VAH_breakdown_limit_retest", df, comp, row))
                    if np.isfinite(rr1) and rr1 >= min_rr and ok_gate:
                        pending_order = {"side":"short","entry":float(entry_target),
                                         "sl":float(plan["sl"]),"tp1":float(plan["tp1"]),"tp2":float(plan["tp2"]),
                                         "reason":"VAH_breakdown_limit_retest","created_ts":ts}
                else:  # Market-on-trigger
                    entry_target = float(px)
                    plan = level_to_level_plan_safe(entry_target, "short", [POC, VAL], atrv, atr_mult, min_stop_bps, min_tp1_bps)
                    rr1 = _rr(entry_target, plan["sl"], plan["tp1"])
                    ok_gate = True
                    if filters_func is not None:
                        ok_gate = bool(filters_func(ts, "short", entry_target, "VAH_breakdown_market", df, comp, row))
                    if np.isfinite(rr1) and rr1 >= min_rr and ok_gate:
                        open_pos = {"side":"short","entry_time":ts,"entry":float(entry_target),
                                    "sl":float(plan["sl"]),"tp1":float(plan["tp1"]),"tp2":float(plan["tp2"]),
                                    "reason":"VAH_breakdown_market"}

            # Single-print continuation
            elif params["use_sp_cont"] and len(sp_bands) > 0:
                touched = None
                for (lo_band, hi_band) in sp_bands:
                    if row["high"] >= lo_band and row["low"] <= hi_band:
                        if regime == "downtrend" and px < lo_band:
                            touched = ("short", (lo_band, hi_band)); break
                        if regime == "uptrend" and px > hi_band:
                            touched = ("long", (lo_band, hi_band)); break
                if touched:
                    side, (lo_band, hi_band) = touched
                    if side == "long":
                        if entry_style == "Limit-on-retest":
                            entry_target = float(hi_band * (1 - params["tol"]))  # buy-limit back into band
                        else:
                            entry_target = float(hi_band * (1 + params["tol"]))  # stop-buy above band
                        plan = level_to_level_plan_safe(entry_target, "long", [POC, VAH], atrv, atr_mult, min_stop_bps, min_tp1_bps)
                    else:
                        if entry_style == "Limit-on-retest":
                            entry_target = float(lo_band * (1 + params["tol"]))  # sell-limit back into band
                        else:
                            entry_target = float(lo_band * (1 - params["tol"]))  # stop-sell below band
                        plan = level_to_level_plan_safe(entry_target, "short", [POC, VAL], atrv, atr_mult, min_stop_bps, min_tp1_bps)
 
                    rr1 = _rr(entry_target, plan["sl"], plan["tp1"])
                    ok_gate = True
                    if filters_func is not None:
                        ok_gate = bool(filters_func(ts, side, entry_target, f"SP_reject_{side}", df, comp, row))
                    if np.isfinite(rr1) and rr1 >= min_rr and ok_gate:
                        if entry_style == "Market-on-trigger":
                            open_pos = {"side":side,"entry_time":ts,"entry":float(px),
                                        "sl":float(plan["sl"]),"tp1":float(plan["tp1"]),"tp2":float(plan["tp2"]),
                                        "reason":f"SP_reject_{side}_market"}
                        else:
                            pending_order = {"side":side,"entry":float(entry_target),
                                            "sl":float(plan["sl"]),"tp1":float(plan["tp1"]),"tp2":float(plan["tp2"]),
                                            "reason":f"SP_reject_{side}","created_ts":ts}

        # Manage an open position
        if open_pos is not None:
            hit = None
            if open_pos["side"] == "long":
                if row["low"] <= open_pos["sl"]:
                    hit = ("SL", float(open_pos["sl"]))
                elif np.isfinite(open_pos["tp1"]) and row["high"] >= open_pos["tp1"]:
                    # TP1 tagged; check if TP2 also tagged this bar
                    tp2_tagged_same_bar = np.isfinite(open_pos.get("tp2", np.nan)) and (row["high"] >= float(open_pos["tp2"]))
                    hit = ("TP1", float(open_pos["tp1"]), tp2_tagged_same_bar)
            else:  # short
                if row["high"] >= open_pos["sl"]:
                    hit = ("SL", float(open_pos["sl"]))
                elif np.isfinite(open_pos["tp1"]) and row["low"] <= open_pos["tp1"]:
                    tp2_tagged_same_bar = np.isfinite(open_pos.get("tp2", np.nan)) and (row["low"] <= float(open_pos["tp2"]))
                    hit = ("TP1", float(open_pos["tp1"]), tp2_tagged_same_bar)

            if hit:
                if hit[0] == "SL":
                    kind, px_exit = hit
                    entry = float(open_pos["entry"])
                    side  = open_pos["side"]
                    sl    = float(open_pos["sl"])
                    tp1   = float(open_pos["tp1"])
                    tp2   = float(open_pos.get("tp2", np.nan))
                    qty   = float(open_pos.get("qty", 1.0))

                    risk_abs = abs(entry - sl) or float("nan")
                    exit_price = float(px_exit)
                    pnl_px = ((exit_price - entry) if side == "long" else (entry - exit_price)) * qty
                    pnl_R  = (((exit_price - entry) / risk_abs) if side == "long" else ((entry - exit_price) / risk_abs)) * qty

                    trades.append({
                        "entry_time": open_pos["entry_time"], "exit_time": ts,
                        "side": side, "reason": open_pos["reason"], "exit_kind": kind,
                        "entry": entry, "sl": sl, "tp1": tp1, "tp2": tp2,
                        "exit": exit_price, "pnl": float(pnl_px), "pnl_R": float(pnl_R),
                        "qty": float(qty),
                    })
                    open_pos = None

                else:
                    # TP1 logic with optional partial exit and same-bar TP2 upgrade
                    _, tp1_px, tp2_same = hit
                    entry = float(open_pos["entry"])
                    side  = open_pos["side"]
                    sl    = float(open_pos["sl"])
                    tp1   = float(open_pos["tp1"])
                    tp2   = float(open_pos.get("tp2", np.nan))
                    qty0  = float(open_pos.get("qty", 1.0))
                    risk_abs = abs(entry - sl) or float("nan")

                    # Helper to append a realized leg
                    def _append_leg(kind: str, exit_px: float, leg_qty: float):
                        pnl_px = ((exit_px - entry) if side == "long" else (entry - exit_px)) * leg_qty
                        pnl_R  = (((exit_px - entry) / risk_abs) if side == "long" else ((entry - exit_px) / risk_abs)) * leg_qty
                        trades.append({
                            "entry_time": open_pos["entry_time"], "exit_time": ts,
                            "side": side, "reason": open_pos["reason"], "exit_kind": kind,
                            "entry": entry, "sl": sl, "tp1": tp1, "tp2": tp2,
                            "exit": float(exit_px), "pnl": float(pnl_px), "pnl_R": float(pnl_R),
                            "qty": float(leg_qty),
                        })

                    # (1) Nothing to take at TP1 → effectively skip TP1
                    if tp1_exit_frac <= 0.0:
                        if tp2_same and np.isfinite(tp2):
                            # All out at TP2 on the same bar
                            _append_leg("TP2", float(tp2), qty0)
                            open_pos = None
                        else:
                            # Keep the entire position open
                            pass

                    # (2) Full exit at TP1 (legacy behavior)
                    elif tp1_exit_frac >= 1.0:
                        # If TP2 also tagged, prefer TP2 (keeps original best-case upgrade on same bar)
                        if tp2_same and np.isfinite(tp2):
                            _append_leg("TP2", float(tp2), qty0)
                        else:
                            _append_leg("TP1", float(tp1_px), qty0)
                        open_pos = None

                    # (3) Partial exit at TP1; remainder continues to TP2/SL
                    else:
                        # Partial exit at TP1; remainder continues to TP2/SL
                        take_qty = qty0 * tp1_exit_frac
                        rem_qty  = qty0 - take_qty
                        _append_leg("TP1_part", float(tp1_px), take_qty)

                        if tp2_same and np.isfinite(tp2):
                            _append_leg("TP2", float(tp2), rem_qty)
                            open_pos = None
                        else:
                            # Keep remainder open but disable further TP1 hits
                            open_pos["qty"] = float(rem_qty)
                            open_pos["tp1"] = float("nan")   # <-- add this line


    # Optional mark-to-market close at end
    if open_pos is not None and force_eod_close:
        ts_end   = df.index[-1]
        entry    = float(open_pos["entry"])
        side     = open_pos["side"]
        sl       = float(open_pos["sl"])
        tp1      = float(open_pos["tp1"])
        tp2      = float(open_pos.get("tp2", np.nan))
        exit_price = float(df["close"].iloc[-1])
        risk_abs = abs(entry - sl) or float("nan")
        qty = float(open_pos.get("qty", 1.0))
        pnl_R = ( (exit_price - entry) / risk_abs if side == "long" else (entry - exit_price) / risk_abs ) * qty
        pnl_px = ( (exit_price - entry) if side == "long" else (entry - exit_price) ) * qty

        trades.append({
            "entry_time": open_pos["entry_time"], "exit_time": ts_end,
            "side": side, "reason": open_pos["reason"], "exit_kind": "EOD",
            "entry": entry, "sl": sl, "tp1": tp1, "tp2": tp2,
            "exit": exit_price, "pnl": float(pnl_px), "pnl_R": float(pnl_R), "qty": float(qty),
        })
        open_pos = None

    trades_df = pd.DataFrame(trades)

    # Summary
    if not trades_df.empty:
        delta = (pd.to_datetime(trades_df["exit_time"], utc=True) - pd.to_datetime(trades_df["entry_time"], utc=True))
        trades_df["hold_minutes"] = delta.dt.total_seconds() / 60.0
        period_mins = _period_minutes(params.get("tf_for_stats", "1h")) or 60
        trades_df["hold_bars"] = trades_df["hold_minutes"] / period_mins

        legacy = {
            "trades": int(len(trades_df)),
            "hit_rate": float((trades_df["pnl"] > 0).mean()),
            "total_pnl": float(trades_df["pnl"].sum()),
            "avg_pnl": float(trades_df["pnl"].mean()),
            "median_pnl": float(trades_df["pnl"].median()),
        }
        r_series = trades_df["pnl_R"]
        r_summary = {
            "total_R": float(r_series.fillna(0).sum()),
            "avg_R": float(r_series.dropna().mean()) if r_series.notna().any() else float("nan"),
            "median_R": float(r_series.dropna().median()) if r_series.notna().any() else float("nan"),
            "hit_rate_R": float((r_series.dropna() > 0).mean()) if r_series.notna().any() else float("nan"),
        }
        summary = {**legacy, **r_summary}
    else:
        summary = {"trades": 0, "hit_rate": float("nan"), "total_pnl": 0.0, "avg_pnl": float("nan"),
                   "median_pnl": float("nan"), "total_R": 0.0, "avg_R": float("nan"),
                   "median_R": float("nan"), "hit_rate_R": float("nan")}

    # Package any unfilled pending order
    pending_list: List[Dict[str,Any]] = []
    if pending_order is not None:
        pending_list.append({
            "side": pending_order["side"],
            "entry": float(pending_order["entry"]),
            "sl":   float(pending_order["sl"]),
            "tp1":  float(pending_order["tp1"]),
            "tp2":  float(pending_order.get("tp2", np.nan)),
            "reason": pending_order.get("reason",""),
            "created_ts": pending_order.get("created_ts")
        })

    return trades_df, summary, open_pos, pending_list

# ----------------- UI -----------------

st.set_page_config(page_title="Level Mapping Strategy — XO", layout="wide")
st.title("Level Mapping Strategy — XO")



# --- Ensure Telegram sender is defined before UI uses it ---

try:
    import requests  # may already be imported elsewhere
except Exception:
    requests = None

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

with st.sidebar:
    st.subheader("Inputs")
    predefined_symbols = [
    "BTCUSDT","ETHUSDT","SOLUSDT","XRPUSDT","AVAXUSDT",
    "LTCUSDT","ZECUSDT", "ASTERUSDT", "AAVEUSDT", "HYPEUSDT", "ENAUSDT", "XLMUSDT", "DOGEUSDT", "BNBUSDT"
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
    tf = st.selectbox("Timeframe", ["5m","15m","30m","1h","4h","12h","1d","1W"], index=3, help="Bar interval for resampling. Shorter TF = faster signals and smaller ATR; affects hold/retest windows and R/R.")
    lookback_days = st.slider("Composite days", 21, 49, 35, 1, help="Days of history to build the volume profile (VAL/POC/VAH). Longer = steadier levels, fewer triggers.")
    n_bins = st.slider("Profile bins", 120, 400, 240, 20, help="Number of histogram bins in the volume profile. Higher = finer granularity; can slightly shift VAL/POC/VAH.")

    # Risk/logic knobs
    with st.expander("Risk/logic knobs", expanded=False):

        atr_mult = st.number_input("ATR stop multiple", value=1.5, min_value=0.5, max_value=10.0, step=0.25, help="Stop distance = ATR × this multiple. Larger = wider stop → lower R/R; Smaller = tighter stop → higher R/R but easier stopouts.")
        min_stop_bps = st.number_input("Min stop distance (bps)", value=25, min_value=0, max_value=500, step=5, help="Hard minimum stop distance in basis points. Prevents unrealistically tight stops; can reduce R/R and block ideas.")
        min_tp1_bps  = st.number_input("Min TP1 distance (bps)",  value=50, min_value=0, max_value=5000, step=10, help="Minimum distance from entry to TP1 (bps). Ensures TP1 is not too close; affects R/R gating.")
        min_rr       = st.number_input("Min R/R (to TP1)",        value=1.2, min_value=0.5, max_value=5.0, step=0.1, help="Trades are only armed if R/R(entry→TP1 vs entry→SL) ≥ this. Raise for pickier ideas; lower to allow more.")

        ema_fast = st.number_input("EMA fast", 5, 50, 12, 1, help="Fast EMA for regime. Smaller = more reactive regimes → more SP continuation signals.")
        ema_slow = st.number_input("EMA slow", 10, 100, 21, 1, help="Slow EMA for regime. Larger = steadier regimes → fewer SP continuation signals.")
        atr_period = st.number_input("ATR period", 5, 50, 14, 1, help="ATR lookback. Shorter = more responsive stops/targets; Longer = smoother stops.")

        hold_bars = st.number_input("Hold bars past level", 1, 10, 2, 1, help="After crossing VAL/VAH, # of consecutive closes beyond the level required before a retest counts. Higher = stricter triggers.")
        retest_bars = st.number_input("Retest window (bars)", 1, 20, 6, 1, help="Bars allowed for the retest after the hold. If no retest within this window → no order is armed.")
        tol_bps = st.number_input("Level tolerance (bps)", 0, 50, 5, 1, help="Tolerance applied to entry relative to VAL/VAH/SP bands. Larger = entry further from level; can change R/R and fill probability.")

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
        st.caption("These gate entries. All are OFF by default.")
        col1, col2 = st.columns(2)
        use_htf_conf = col1.checkbox("Require HTF confluence (Monthly/Weekly)", value=False)
        htf_tol_bps  = col1.slider("HTF confluence tolerance (bps)", 2, 60, 15, 1)
        use_accept   = col1.checkbox("Require acceptance at level", value=False)
        accept_bars  = col1.slider("Bars of acceptance (proxy)", 1, 6, 2, 1)
        use_htf_ema_gate = col1.checkbox("Require HTF EMA(12/25) bull intact", value=False,
                                 help="Weekly & 3-Day: price > both EMAs, no fresh bearish cross.")
        avoid_mid    = col2.checkbox("Avoid mid-range entries", value=False)
        mid_min_bps  = col2.slider("Min distance from mid (bps)", 10, 200, 40, 5)
        no_fade      = col2.checkbox("No shorts into support / longs into resistance", value=False)
        opp_buf_bps  = col2.slider("Opposite-level buffer (bps)", 5, 60, 15, 1)

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

        use_htf_ema_gate = st.checkbox("Require HTF EMA(12/25) bull intact (1W & 3D)", value=False)
        use_mid_gate = st.checkbox("Mid-gate is crisper, avoid-mid is spacing", value=False)


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
    if "time" in df.columns:
        idx = pd.to_datetime(df["time"], utc=True)
        df = df.drop(columns=["time"]).set_index(idx)
    elif not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, utc=True)
    return df.sort_index().dropna(subset=["open","high","low","close"]).copy()

def run_for_symbol(symbol: str):
    st.header(symbol)
    raw = _load(symbol, tf)
    df = resample_ohlcv(raw, tf)
    htf_state = htf_ema_state(df)
    htf_bull_intact = (htf_state["weekly"]["bull_ok"] and not htf_state["weekly"]["bear_cross"]) \
                  and (htf_state["d3"]["bull_ok"] and not htf_state["d3"]["bear_cross"])

    if drop_last_incomplete:
        df = drop_incomplete_bar(df, tf)

    comp = composite_profile(df, days=lookback_days, n_bins=n_bins)
    rng = rolling_swing_range(df, lookback= max(150, n_bins//2))
    eq_highs = equal_extrema_mask(df["high"].rolling(5).max(), tol_bps)  # small window cluster
    eq_lows  = equal_extrema_mask(df["low"].rolling(5).min(), tol_bps)
    sp_bands = single_prints(comp["hist"])

    params = {
        "tf_for_stats": tf,
        "ema_fast": int(ema_fast), "ema_slow": int(ema_slow),
        "atr_period": int(atr_period),
        "hold_bars": int(hold_bars), "retest_bars": int(retest_bars),
        "tol": float(tol_bps)/10000.0,
        "use_val_reclaim": True,
        "use_vah_break": True,
        "use_sp_cont": True,
        "sp_bands": sp_bands,
        "atr_mult": float(atr_mult),
        "min_stop_bps": int(min_stop_bps),
        "min_tp1_bps": int(min_tp1_bps),
        "min_rr": float(min_rr),
        "entry_style": entry_style,
        "tp1_exit_pct": int(tp1_exit_pct),  # NEW
        "_extra_atr_buffer": float(atr_buf_x) if add_atr_buf else 0.0,
    }

    # --- Build HTF composites (used only if user enables HTF confluence) ---
    def _htf_composites(df_in: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        out = {}
        try:
            out["weekly"]  = composite_profile(df_in, days=84,  n_bins=n_bins)
            out["monthly"] = composite_profile(df_in, days=168, n_bins=n_bins)
        except Exception:
            out["weekly"]={"VAL":np.nan,"POC":np.nan,"VAH":np.nan}
            out["monthly"]={"VAL":np.nan,"POC":np.nan,"VAH":np.nan}
        return out

    htf = _htf_composites(df) if use_htf_conf else {"weekly":{}, "monthly":{}}

    def _bps(a,b): 
        try: return abs((a-b)/a)*1e4
        except Exception: return np.inf

    def _nearest_bps(px, levels: Dict[str,float]):
        vals = [v for v in [levels.get("VAL"), levels.get("POC"), levels.get("VAH")] if np.isfinite(v)]
        if not vals: return np.inf
        return min(_bps(px, v) for v in vals)

    def _opp_nearest_bps(side: str, px: float, comp_levels: Dict[str,float]):
        # Longs must not be too close to resistance (VAH/POC), Shorts must not be too close to support (VAL/POC)
        if side == "long":
            vals = [v for v in [comp_levels.get("POC"), comp_levels.get("VAH")] if np.isfinite(v)]
        else:
            vals = [v for v in [comp_levels.get("POC"), comp_levels.get("VAL")] if np.isfinite(v)]
        if not vals: return np.inf
        return min(_bps(px, v) for v in vals)

    def _accepted(side: str, i_ts: pd.Timestamp) -> bool:
        if not use_accept or accept_bars <= 0: 
            return True
        try:
            ix = df.index.get_loc(i_ts)
            w  = df.iloc[max(0, ix-accept_bars+1):ix+1]
            if w.empty: return False
            if side == "long":
                return (w["close"] > comp["VAL"]).all() if np.isfinite(comp["VAL"]) else True
            else:
                return (w["close"] < comp["VAH"]).all() if np.isfinite(comp["VAH"]) else True
        except Exception:
            return True

    def _relvol_ok(i_ts: pd.Timestamp) -> bool:
        if not use_relvol: return True
        try:
            vol = df["volume"]
            ma  = vol.rolling(50, min_periods=10).mean()
            v   = float(vol.loc[i_ts])
            m   = float(ma.loc[i_ts])
            thr = float(relvol_x)/100.0
            return (m > 0) and (v >= thr*m)
        except Exception:
            return True

    def _wick_ok(side: str, i_ts: pd.Timestamp, entry: float) -> bool:
        if not use_wick: return True
        try:
            r = df.loc[i_ts]
            hi, lo, op, cl = float(r["high"]), float(r["low"]), float(r["open"]), float(r["close"])
            up_w  = max(0.0, hi - max(op, cl))
            dn_w  = max(0.0, min(op, cl) - lo)
            w_bps = lambda x: (x/entry)*1e4 if entry>0 else 0.0
            if side == "long":
                return w_bps(dn_w) >= float(wick_min_bp)
            else:
                return w_bps(up_w) >= float(wick_min_bp)
        except Exception:
            return True

    # compose optional entry filters (all AND-ed)
    def _filters_func(i_ts: pd.Timestamp, side: str, entry_px: float, reason: str,
                    df_in: pd.DataFrame, comp_in: Dict[str,Any], rng_in: Dict[str, float],
                    row_now: pd.Series) -> bool:
        # HTF confluence
        if use_htf_conf:
            ok_w = _nearest_bps(entry_px, htf.get("weekly", {}))  <= float(htf_tol_bps)
            ok_m = _nearest_bps(entry_px, htf.get("monthly", {})) <= float(htf_tol_bps)
            if not (ok_w or ok_m): 
                return False
        # Acceptance proxy (relative to VAL/VAH band)
        if not _accepted(side, i_ts):
            return False
        # Avoid mid-range entries
        if avoid_mid and np.isfinite(comp_in["VAL"]) and np.isfinite(comp_in["VAH"]):
            mid = 0.5*(float(comp_in["VAL"])+float(comp_in["VAH"]))
            if _bps(entry_px, mid) < float(mid_min_bps):
                return False
        # No-fade guard (opposite level too close)
        if no_fade:
            if _opp_nearest_bps(side, entry_px, comp_in) < float(opp_buf_bps):
                return False
        # Flow proxies
        if not _relvol_ok(i_ts):
            return False
        if not _wick_ok(side, i_ts, entry_px):
            return False
        if use_htf_ema_gate and not htf_bull_intact and side == "long":
            return False
        # Mid as routing gate (binary)
        if np.isfinite(rng["mid"]):
            if entry_px >= rng["mid"] and side == "long":
                pass  # OK → favor range high targeting
            elif entry_px <= rng["mid"] and side == "short":
                pass  # OK → favor range low targeting
            else:
                # entering into the wrong half: reduce selectivity
                return False
        rng = rolling_swing_range(df, lookback=300)  # or reuse existing
        if use_mid_gate and np.isfinite(rng.get("mid", np.nan)):
            if side == "long" and entry_px < rng["mid"]:
                return False
            if side == "short" and entry_px > rng["mid"]:
                return False
        if use_breakout_logic and np.isfinite(rng.get("hi", np.nan)) and np.isfinite(rng.get("lo", np.nan)):
            dev_up = breakout_or_deviation(df, float(rng["hi"]), "up", retest=2, tol_bps=tol_bps*1e4)
            dev_dn = breakout_or_deviation(df, float(rng["lo"]), "down", retest=2, tol_bps=tol_bps*1e4)
            if side == "long" and dev_up == "deviation":
                return False
            if side == "short" and dev_dn == "deviation":
                return False
        if sfp_only and np.isfinite(rng_in.get("hi", np.nan)) and np.isfinite(rng_in.get("lo", np.nan)):
            if side == "short" and not is_sfp(df, float(rng_in["hi"]), "short"): return False
            if side == "long"  and not is_sfp(df, float(rng_in["lo"]), "long"):  return False
        if use_htf_ema_gate:
            stt = htf_ema_state(df)
            if side == "long":
                if not (stt["weekly"]["bull_ok"] and stt["d3"]["bull_ok"]) or (stt["weekly"]["bear_cross"] or stt["d3"]["bear_cross"]):
                    return False
            # (You can mirror shorts with a separate toggle if desired.)
        if use_comp_expansion and np.isfinite(comp.get("VAL", np.nan)) and np.isfinite(comp.get("VAH", np.nan)):
            key = comp["VAL"] if side == "long" else comp["VAH"]
            if compression_squeeze(df):
                # Require closes to accept beyond key for N bars
                acc = df["close"].tail(3) > key if side=="long" else df["close"].tail(3) < key
                if not bool(acc.all()):
                    return False
        if use_cvd_absorption:
            if side=="long" and rising_cvd_no_progress_near_resistance(df_in, comp): return False
            if side=="short" and rising_cvd_no_progress_near_support(df_in, comp):   return False

        if use_oi_spike_reduce and oi_spike(df_in.tail(200)):
            # Either block, or reduce size via a size factor you already expose in UI
            return False
        if naked_poc_guard and naked_pocs:
            for npoc in naked_pocs:
                if np.isfinite(npoc) and _bps(entry_px, float(npoc)) < float(naked_poc_bps):
                    return False
        if use_btc_guard and side=="long" and symbol.upper() in ("ETHUSDT","SOLUSDT","ETHUSD","SOLUSD"):
            if btc_guard.get("near_resistance", False):
                return False
        return True

    # attach filters + atr buffer into params (engine will call conditionally)
    params["filters_func"] = _filters_func if any([use_htf_conf, use_accept, avoid_mid, no_fade, use_relvol, use_wick]) else None
    params["_extra_atr_buffer"] = float(atr_buf_x) if add_atr_buf else 0.0

    trades_df, summary, active_open, pending_list = backtest_xo_logic(df, comp, params, force_eod_close=False)

    # --- Telegram notifications (UI-only; no trade-logic changes) ---
    if ("enable_tg" in globals() and enable_tg) and tg_token and tg_chat_id:
        tf_now = tf  # timeframe chosen above

        def _val_line(c: Dict[str, Any]) -> str:
            vals = []
            try:
                for k in ("VAL", "POC", "VAH"):
                    v = c.get(k, float("nan"))
                    if np.isfinite(v):
                        vals.append(f"{k}: <b>{v:.4f}</b>")
            except Exception:
                pass
            return " · ".join(vals) if vals else ""

        # PREVIEWED (pending) orders
        if ("notify_preview" in globals() and notify_preview) and pending_list:
            for p in pending_list:
                try:
                    k = (symbol, str(p.get("created_ts")), float(p["entry"]))
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

                lines = [
                    f"🟨 <b>PREVIEW</b> — <b>{symbol}</b> ({_esc(tf_now)})",
                    f"Side: <b>{_esc(p['side']).upper()}</b>  ·  {_esc(reason)}",
                    f"Entry: <b>{entry:.4f}</b> | SL: <b>{sl:.4f}</b>",
                    f"TP1: {'—' if not np.isfinite(tp1) else f'<b>{tp1:.4f}</b>'}{'' if not np.isfinite(R1) else f'  (≈ {R1:.2f}R)'}",
                    f"TP2: {'—' if not np.isfinite(tp2) else f'<b>{tp2:.4f}</b>'}{'' if not np.isfinite(R2) else f'  (≈ {R2:.2f}R)'}",
                    _val_line(comp),
                    (f"Created: {created}" if created is not None else "")
                ]
                _send_tg(tg_token, tg_chat_id, "\n".join([ln for ln in lines if ln]))
                st.session_state["tg_seen_pending"].add(k)

        # ACTIVE (open) position
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
                # --- ACTIVE milestone hits (SL/TP1/TP2) ---
                if ("enable_tg" in globals() and enable_tg) and tg_token and tg_chat_id \
                and ("notify_hits" in globals() and notify_hits) \
                and ("active_open" in locals() and active_open is not None):

                    entry_time = active_open.get("entry_time")
                    side = str(active_open.get("side", "")).lower()

                    # Current last price on chart
                    last_px = float(df["close"].iloc[-1])

                    # Targets
                    sl = float(active_open.get("sl", float("nan")))
                    tp1 = float(active_open.get("tp1", float("nan")))
                    tp2 = float(active_open.get("tp2", float("nan")))

                    def _hit(milestone: str) -> bool:
                        # Returns True if milestone is hit by last_px for given side
                        if not np.isfinite(sl):
                            return False
                        if milestone == "SL":
                            return (last_px <= sl) if side == "long" else (last_px >= sl)

                        if milestone == "TP1":
                            if not np.isfinite(tp1):
                                return False
                            return (last_px >= tp1) if side == "long" else (last_px <= tp1)

                        if milestone == "TP2":
                            if not np.isfinite(tp2):
                                return False
                            return (last_px >= tp2) if side == "long" else (last_px <= tp2)

                        return False

                    # Check each milestone in priority order (SL first, then TP2, then TP1)
                    for mk, badge in (("SL", "🔴"), ("TP2", "🟣"), ("TP1", "🟩")):
                        key = (symbol, str(entry_time), mk)
                        if key in st.session_state["tg_hits_seen"]:
                            continue
                        if not _hit(mk):
                            continue

                        # Compose a nice message
                        entry = float(active_open["entry"])
                        risk = abs(entry - sl) if np.isfinite(sl) else float("nan")
                        u_pnl = (last_px - entry) if side == "long" else (entry - last_px)
                        u_R = (u_pnl / risk) if (np.isfinite(risk) and risk > 0) else float("nan")

                        _targets = []
                        if np.isfinite(tp1): _targets.append(f"TP1: <b>{tp1:.4f}</b>")
                        if np.isfinite(tp2): _targets.append(f"TP2: <b>{tp2:.4f}</b>")
                        targets_line = "  |  ".join(_targets) if _targets else "No TPs set"

                        lines_hit = [
                            f"{badge} <b>{mk} HIT</b> — <b>{symbol}</b> ({_esc(tf)})",
                            f"Side: <b>{side.upper()}</b>  ·  {_esc(active_open.get('reason',''))}",
                            f"Entry: <b>{entry:.4f}</b>  |  SL: <b>{sl if not np.isfinite(sl) else f'{sl:.4f}'}</b>",
                            targets_line,
                            f"Last: <b>{last_px:.4f}</b>  ·  U.PnL: {'—' if not np.isfinite(u_pnl) else f'{u_pnl:.4f}'}"
                            f"{'' if not np.isfinite(u_R) else f'  (≈ {u_R:.2f}R)'}"
                        ]
                        _send_tg(tg_token, tg_chat_id, "\n".join([ln for ln in lines_hit if ln]))
                        st.session_state["tg_hits_seen"].add(key)

                st.session_state["tg_seen_active"].add(k2)
                # --- Telegram dedupe for milestone hits ---
                if "tg_hits_seen" not in st.session_state:
                    st.session_state["tg_hits_seen"] = set()  # keys: (symbol, str(entry_time), "SL"/"TP1"/"TP2")


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
            rows.append({"Side":side,"Reason":p.get("reason",""),"Created":created,"Age (min)":age_mins,
                         "Entry":entry,"SL":sl,"TP1":tp1,"TP2":tp2,"Planned R→TP1":R1,"Planned R→TP2":R2})
        dfp = pd.DataFrame(rows)[["Side","Reason","Created","Age (min)","Entry","SL","TP1","TP2","Planned R→TP1","Planned R→TP2"]]
        st.dataframe(dfp.style.format({
            "Age (min)":"{:.1f}",
            "Entry":"{:,.2f}","SL":"{:,.2f}","TP1":"{:,.2f}","TP2":"{:,.2f}",
            "Planned R→TP1":"{:.2f}","Planned R→TP2":"{:.2f}"
        }), use_container_width=True)

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
    else:
        show_cols = [c for c in ["entry_time","exit_time","side","reason","exit_kind","entry","sl","tp1","tp2","exit","pnl","pnl_R","hold_minutes","hold_bars"] if c in trades_df.columns]
        st.dataframe(trades_df[show_cols], use_container_width=True)

        # --- Per-symbol charts: Cumulative R and Distribution of R ---
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            # Cumulative R (trade order)
            st.markdown("#### Cumulative R (closed trades)")
            fig_cr, ax_cr = plt.subplots(figsize=(7,3.5))
            trades_df_sorted = trades_df.reset_index(drop=True)
            y_cr = trades_df_sorted["pnl_R"].fillna(0).cumsum().values
            ax_cr.plot(range(len(y_cr)), y_cr, marker="o", markersize=3 if len(y_cr) > 80 else 5)

            ax_cr.set_xlabel("Trade #")
            ax_cr.set_ylabel("Cumulative R")
            ax_cr.set_title(f"{symbol} — Cumulative R")
            ax_cr.grid(True, which="both", axis="both", alpha=0.3)
            ax_cr.margins(x=0.01)

            # show at most ~8–10 integer ticks on x; prune to avoid overlap at edges
            ax_cr.xaxis.set_major_locator(MaxNLocator(nbins=9, integer=True, prune="both"))
            ax_cr.tick_params(axis="x", labelsize=8)
            st.pyplot(fig_cr)

            
        except Exception as _e:
            st.warning(f"Plot error: {_e}")

        # Collect for all-symbol aggregation
        try:
            if "all_trades" not in st.session_state:
                st.session_state["all_trades"] = []
            tdf = trades_df.copy()
            tdf["symbol"] = symbol
            st.session_state["all_trades"].append(tdf)
        except Exception as _e:
            st.warning(f"Aggregation error: {_e}")

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


def compute_for_symbol(symbol: str) -> Dict[str, Any]:
    """Compute everything for a symbol and return a dict; also triggers Telegram alerts (same as before)."""
    raw = _load(symbol, tf)
    df = resample_ohlcv(raw, tf)
    if drop_last_incomplete:
        df = drop_incomplete_bar(df, tf)

    comp = composite_profile(df, days=lookback_days, n_bins=n_bins)
    rng = rolling_swing_range(df, lookback= max(150, n_bins//2))
    eq_highs = equal_extrema_mask(df["high"].rolling(5).max(), tol_bps)  # small window cluster
    eq_lows  = equal_extrema_mask(df["low"].rolling(5).min(), tol_bps)
    sp_bands = single_prints(comp["hist"])

    params = {
        "tf_for_stats": tf,
        "ema_fast": int(ema_fast), "ema_slow": int(ema_slow),
        "atr_period": int(atr_period),
        "hold_bars": int(hold_bars), "retest_bars": int(retest_bars),
        "tol": float(tol_bps)/10000.0,
        "use_val_reclaim": True,
        "use_vah_break": True,
        "use_sp_cont": True,
        "sp_bands": sp_bands,
        "atr_mult": float(atr_mult),
        "min_stop_bps": int(min_stop_bps),
        "min_tp1_bps": int(min_tp1_bps),
        "min_rr": float(min_rr),
        "entry_style": entry_style,
        "min_rr": float(min_rr),
        "entry_style": entry_style,
        "tp1_exit_pct": int(tp1_exit_pct),  # NEW
    }

    trades_df, summary, active_open, pending_list = backtest_xo_logic(df, comp, params, force_eod_close=False)

    # Telegram (same rules as your in-function block)
    if ("enable_tg" in globals() and enable_tg) and tg_token and tg_chat_id:
        tf_now = tf

        def _val_line(c: Dict[str, Any]) -> str:
            vals = []
            try:
                for k in ("VAL", "POC", "VAH"):
                    v = c.get(k, float("nan"))
                    if np.isfinite(v):
                        vals.append(f"{k}: <b>{v:.4f}</b>")
            except Exception:
                pass
            return " · ".join(vals) if vals else ""

        # PREVIEWED (pending)
        if ("notify_preview" in globals() and notify_preview) and pending_list:
            for p in pending_list:
                try:
                    k = (symbol, str(p.get("created_ts")), float(p["entry"]))
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
                lines = [
                    f"🟨 <b>PREVIEW</b> — <b>{symbol}</b> ({_esc(tf_now)})",
                    f"Side: <b>{_esc(p['side']).upper()}</b>  ·  {_esc(reason)}",
                    f"Entry: <b>{entry:.4f}</b> | SL: <b>{sl:.4f}</b>",
                    f"TP1: {'—' if not np.isfinite(tp1) else f'<b>{tp1:.4f}</b>'}{'' if not np.isfinite(R1) else f'  (≈ {R1:.2f}R)'}",
                    f"TP2: {'—' if not np.isfinite(tp2) else f'<b>{tp2:.4f}</b>'}{'' if not np.isfinite(R2) else f'  (≈ {R2:.2f}R)'}",
                    _val_line(comp),
                    (f"Created: {created}" if created is not None else "")
                ]
                _send_tg(tg_token, tg_chat_id, "\n".join([ln for ln in lines if ln]))
                st.session_state["tg_seen_pending"].add(k)

        # ACTIVE + milestone hits (SL/TP1/TP2)
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

            # Hits (your newer behavior)
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
                        f"{'' if not np.isfinite(u_R) else f'  (≈ {u_R:.2f}R)'}"
                    ]
                    _send_tg(tg_token, tg_chat_id, "\n".join([ln for ln in lines_hit if ln]))
                    st.session_state["tg_hits_seen"].add(key)

    # add to all_trades aggregation
    if "all_trades" not in st.session_state:
        st.session_state["all_trades"] = []
    tdf = trades_df.copy()
    tdf["symbol"] = symbol
    st.session_state["all_trades"].append(tdf)

    return {"df": df, "comp": comp, "trades_df": trades_df, "summary": summary,
            "active_open": active_open, "pending_list": pending_list}

def render_symbol(symbol: str, res: Dict[str,Any]):
    """Render the SAME UI you had for run_for_symbol, using precomputed results (no recompute)."""
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

        levels = _find_sr_levels(
            df=df,
            lookback_bars=int(lookback_bars),
            swing=int(swing),
            max_levels=int(max_levels),
            min_distance_bps=float(min_dist_bps),
            include_composite=comp_levels,
        )

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
    st.session_state["results"] = {}       # symbol -> result dict
    st.session_state["all_trades"] = []    # rebuild aggregation

    for sym in symbols_final:
        try:
            st.session_state["results"][sym] = compute_for_symbol(sym)
        except Exception as _e:
            st.warning(f"{sym}: compute error — {_e}")

# ---- Render section (no recompute required) ----
# ---- Render section (no recompute required) ----
if st.session_state.get("results"):
    if selection == "ALL":
        # Combined charts across all symbols (same as your existing section)
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import pandas as _pd

        all_df = _pd.concat([d["trades_df"].assign(symbol=s) for s, d in st.session_state["results"].items()], ignore_index=True)


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