# mtf_ob_report_app.py
# Offline Streamlit app: upload 1+ OHLCV CSVs (W/D/4H/1H/30m/15m),
# auto-detect high-quality bullish/bearish order blocks (OBs),
# compute TPO (Value Area/POC), CVD proxy/true (if present),
# produce Tables + Bulleted Reasons + downloadable PDFs,
# and render annotated candlestick charts with OB/BOS/FVG/sweep/profile overlays.

from __future__ import annotations

import io, re, math
from dataclasses import dataclass, replace
from typing import List, Tuple, Optional, Dict

import os
from bybit_fetch import fetch_all_dfs, SUPPORTED_TFS

import streamlit as st
import pandas as pd
import numpy as np

from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.dates as mdates

import mplfinance as mpf

from functools import lru_cache

try:
    import ccxt
except Exception as e:
    ccxt = None

@lru_cache(maxsize=1)
def _bybit():
    if ccxt is None:
        raise RuntimeError(
            "ccxt not installed. Run `pip install ccxt` in your environment."
        )
    return ccxt.bybit({"enableRateLimit": True})

def _to_ccxt_symbol(symbol: str, *, category: str = "linear") -> str:
    """
    Map 'BTCUSDT' → CCXT market id. For USDT-margined perps (default):
      'BTCUSDT' -> 'BTC/USDT:USDT'
    For spot, pass category='spot' to get 'BTC/USDT'.
    """
    s = symbol.upper().strip()
    if s.endswith("USDT"):
        base = s[:-4]
        return f"{base}/USDT" if category == "spot" else f"{base}/USDT:USDT"
    if s.endswith("USDC"):
        base = s[:-4]
        return f"{base}/USDC" if category == "spot" else f"{base}/USDC:USDC"
    # fallback: passthrough if caller already supplied CCXT id
    return s

def load_ohlcv(symbol: str, timeframe: str, *, limit: int = 1500, category: str = "linear") -> pd.DataFrame:
    """
    Return OHLCV as a DataFrame with columns: open, high, low, close, volume
    Index is UTC DatetimeIndex. Uses Bybit public endpoints via CCXT.
    """
    ex = _bybit()
    market = _to_ccxt_symbol(symbol, category=category)
    ohlcv = ex.fetch_ohlcv(market, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=["time","open","high","low","close","volume"])
    df["time"] = pd.to_datetime(df["time"], unit="ms", utc=True)
    df = df.set_index("time").sort_index()
    # ensure numeric
    for c in ["open","high","low","close","volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


st.set_page_config(page_title="MTF OB Trade Plan", layout="centered")

# -------- Config defaults & session init (place near the top) --------
DEFAULTS = {
    "pair": "SOLUSD",
    "stop_buffer": 0.02,

    # detection
    "preset": "Balanced (recommended)",
    "min_score": 3.0,
    "displacement_k": 1.5,
    "adaptive_disp": False,
    "displacement_mult": 1.0,

    # pivots / fvg / overlap
    "pivot_left": 3,
    "pivot_right": 3,
    "fvg_span": 3,
    "iou_thresh": 0.5,

    # rendering / UX
    "use_wicks": True,
    "label_style": "Numeric tags",
    "focus_mode": "Auto-zoom: latest OB",
    "bars_before": 120,
    "bars_after": 60,
    "show_fvg": True,
    "show_sweeps": True,
    "dark_charts": True,

    # ✅ add these two (they’re referenced later)
    "render_charts": True,
    "include_charts_pdf": True,

    # trade plan
    "target_policy": "Opposing OB mids",
}

# --- ATR14-per-timeframe config (define BEFORE any UI that uses it) ---
SUPPORTED_TFS = ["Weekly", "Daily", "4H", "1H", "30m", "15m"]

# Recommended ATR14 multiples per timeframe (used when stop mode = ATR14 × multiple)
DEFAULT_ATR14_MULTS = {
    "Weekly": 0.60,
    "Daily":  0.80,
    "4H":     1.00,
    "1H":     1.20,
    "30m":    1.40,
    "15m":    1.60,
}

# Ensure session defaults exist before any UI reads them
if "atr14_mults" not in st.session_state:
    st.session_state["atr14_mults"] = DEFAULT_ATR14_MULTS.copy()

if "stop_buffer_atr_fallback" not in st.session_state:
    st.session_state["stop_buffer_atr_fallback"] = 1.00

# Mutable dict you edit with number_inputs
ATR14_MULTS = st.session_state["atr14_mults"]


# --- place this right after DEFAULTS (before any st.* widget uses help=HELP["..."]) ---

class _Help(dict):
    def __missing__(self, key):
        return ""   # silently return empty help if a key is missing

HELP = _Help({
    # Inputs
    "pair": "Used only in chart titles, captions, and output file names.",
    "stop_buffer": "Absolute price distance beyond the OB extreme to place the stop. Affects SL, R/R, and 2R/3R fallbacks.",

    # Theme & palette
    "dark_charts": "Switches mplfinance style to a dark scheme for the charts.",

    # Overlays
    "show_fvg": "Draw Fair Value Gap rectangles near the OB (detection always computes FVG regardless of this toggle).",
    "show_sweeps": "Draw a horizontal reference line at the recent buy/sell-side sweep preceding the OB.",

    # Advanced detection
    "pivot_left": "Bars to the left used to confirm a swing. Higher values = stricter swing highs/lows.",
    "pivot_right": "Bars to the right used to confirm a swing. Higher values = stricter swing highs/lows.",
    "fvg_span": "Span used for FVG detection. 3 = classic (i-1,i+1); 5 = wider (i-2,i+2). Impacts scoring and overlays.",
    "iou_thresh": "Interval IoU threshold for deduping overlapping OBs. Lower = more aggressive merging.",
    "adaptive_disp": "Use a rolling 60-bar 60th percentile of (range/ATR14) as a displacement floor that adapts to volatility.",
    "displacement_mult": "Scale the adaptive displacement floor. >1 tightens; <1 loosens.",

    # Targeting
    "target_policy": "How TP1/TP2 are chosen: opposing OB mid(s), fixed 2R/3R, or next swing highs/lows (Structure).",

    # Detection parameters (presets/custom)
    "preset": "Quick sets for min_score and displacement_k (Balanced/Conservative/Aggressive).",
    "min_score": "Min total quality score required for a candidate OB to be accepted.",
    "displacement_k": "Required BOS bar impulse: (bar_range / ATR14) > k (subject to adaptive floor).",

    # Chart rendering & UX
    "label_style": "How much labeling to draw on the overlays (minimal tags vs numeric vs verbose).",
    "focus_mode": "Auto-zoom displays a window around the latest OB→BOS; Full range shows all bars.",
    "bars_before": "How many bars before the OB origin to include when Auto-zoom is enabled.",
    "bars_after": "How many bars after the BOS to include when Auto-zoom is enabled.",
    "use_wicks": "OB vertical bounds include wick on the far side (recommended) instead of body-only.",
    "render_charts": "Toggle generation of annotated/clean charts (turn off for faster runs).",
    "include_charts_pdf": "If charts were rendered, also offer a single PDF bundling them.",

    
    # Time handling
    "dayfirst_ui": "Treat string dates as DD/MM/YYYY (instead of MM/DD/YYYY).",
    "epoch_unit_ui": "For numeric timestamps, choose how to parse (e.g., seconds, ms, Excel serial days). Auto-detect tries to guess.",
    "tz_naive_ui": "Convert timestamps to UTC then drop timezone info (mplfinance prefers tz-naive).",

    # Run
    "generate": "Parse uploads, detect OBs, build trade plans, and render outputs."
})


# 1) seed session once
if "ui_settings" not in st.session_state:
    st.session_state.ui_settings = DEFAULTS.copy()

# 2) always work off a merged config (defaults < session)
cfg = {**DEFAULTS, **st.session_state.ui_settings}

# 3) materialize named variables so downstream code can use them safely
pair              = cfg["pair"]
stop_buffer       = cfg["stop_buffer"]
preset            = cfg["preset"]
min_score         = cfg["min_score"]
displacement_k    = cfg["displacement_k"]
adaptive_disp     = cfg["adaptive_disp"]
displacement_mult = cfg["displacement_mult"]
pivot_left        = cfg["pivot_left"]
pivot_right       = cfg["pivot_right"]
fvg_span          = cfg["fvg_span"]
iou_thresh        = cfg["iou_thresh"]
use_wicks         = cfg["use_wicks"]
label_style       = cfg["label_style"]
focus_mode        = cfg["focus_mode"]
bars_before       = cfg["bars_before"]
bars_after        = cfg["bars_after"]
show_fvg          = cfg["show_fvg"]
show_sweeps       = cfg["show_sweeps"]
dark_charts       = cfg["dark_charts"]
target_policy     = cfg["target_policy"]


# ---------- Consistent canvas for ALL charts ----------
CANVAS_W_PX = 1000
CANVAS_H_PX = 560
CANVAS_DPI  = 120
IDEA_CANVAS_H_PX = 720

# === New config constants (put near your CANVAS_* constants) ===
DEFAULT_TF_ORDER = ["Weekly", "Daily", "4H", "1H", "30m", "15m"]


# Color palettes
PALETTE_DEFAULT = {
    "bull_face": (0, 0.60, 0, 0.45),
    "bull_edge": (0, 0.45, 0, 1.00),
    "bear_face": (0.85, 0, 0, 0.45),
    "bear_edge": (0.65, 0, 0, 1.00),
    "fvg_bull_face": (0.20, 0.70, 1.0, 0.40),
    "fvg_bull_edge": (0.20, 0.55, 1.0, 1.00),
    "fvg_bear_face": (1.00, 0.60, 0.20, 0.40),
    "fvg_bear_edge": (1.00, 0.50, 0.20, 1.00),
    "bos": "black",
    "sweep_sell": "magenta",
    "sweep_buy": "orange",
    "bos_anchor": (0.25, 0.25, 0.25, 0.65)
}

plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "legend.fontsize": 9,
})

# =========================
# ------- Data types ------
# =========================

@dataclass
class OB:
    side: str
    index: int
    start: float
    end: float
    open: float
    high: float
    low: float
    close: float
    bos_index: int
    score: float
    reasons: List[str]
    # mitigation
    touches: int = 0
    first_touch_index: Optional[int] = None
    last_touch_index: Optional[int] = None
    fill_ratio: float = 0.0
    mitigated: bool = False
    consumed: bool = False
    broken_level: Optional[float] = None    # swing level broken at BOS (anchor)
    consumed_index: Optional[int] = None    # first bar index that consumed it

@dataclass
class TradePlan:
    entry: float
    sl: float
    tp1: Optional[float]
    tp2: Optional[float]
    rr1: Optional[float]
    rr2: Optional[float]
    invalidations: List[str]
    reasons: List[str]

# =========================
# -------- Helpers --------
# =========================

def interval_iou(a: Tuple[float,float], b: Tuple[float,float]) -> float:
    lo1, hi1 = min(a), max(a)
    lo2, hi2 = min(b), max(b)
    inter = max(0.0, min(hi1, hi2) - max(lo1, lo2))
    union = (hi1 - lo1) + (hi2 - lo2) - inter
    return float(inter/union) if union > 0 else 0.0

def last_two_swings_after(piv_mask: np.ndarray, start_idx: int) -> List[int]:
    idxs = np.where(piv_mask)[0]
    idxs = idxs[idxs > start_idx]
    return list(idxs[:2])


def _parse_time_column(raw, *, dayfirst: Optional[bool] = None,
                       epoch_unit: Optional[str] = None,
                       force_utc: bool = True,
                       tz_naive: bool = True) -> pd.Series:
    """
    Parse a 'time' column that could be:
    - ISO strings (with/without timezone)
    - DD/MM or MM/DD strings
    - Epoch (s/ms/us/ns)
    - Excel serial days (since 1899-12-30)

    Auto-detects most cases; allows overrides via args.
    """
    s = pd.Series(raw)
    out = None

    # Try numeric detection (epoch / excel)
    sn = pd.to_numeric(s, errors="coerce")
    if sn.notna().mean() > 0.95:  # mostly numeric
        vals = sn.dropna()
        med = float(vals.median()) if len(vals) else 0.0

        # Decide unit
        unit = None
        if epoch_unit:
            # map friendly labels to pandas units
            m = {
                "s": "s", "seconds": "s",
                "ms": "ms", "milliseconds": "ms",
                "us": "us", "microseconds": "us",
                "ns": "ns", "nanoseconds": "ns",
                "excel": "excel",
            }
            unit = m.get(epoch_unit.lower(), None)

        if unit is None:
            # Auto-detect
            if 20000 < med < 80000:
                unit = "excel"          # Excel serial days
            elif med > 1e18:
                unit = "ns"
            elif med > 1e15:
                unit = "us"
            elif med > 1e10:
                unit = "ms"
            elif med > 1e9:
                unit = "s"
            else:
                # if it's small but > 1e8, still likely seconds
                unit = "s" if med > 1e8 else "s"

        if unit == "excel":
            dt = pd.to_datetime(vals, unit="D", origin="1899-12-30",
                                utc=force_utc, errors="coerce", infer_datetime_format=True)
        else:
            dt = pd.to_datetime(vals, unit=unit, utc=force_utc, errors="coerce", infer_datetime_format=True)

        out = pd.Series(index=s.index, dtype="datetime64[ns, UTC]" if force_utc else "datetime64[ns]")
        out.loc[vals.index] = dt

    else:
        # String-like timestamps
        # First pass: respect dayfirst if provided, otherwise try False then True
        def try_parse(dfirst: bool):
            return pd.to_datetime(s, errors="coerce", utc=force_utc,
                                  dayfirst=dfirst, infer_datetime_format=True)

        if dayfirst is None:
            dt0 = try_parse(False)
            # If too many NaT OR many slashed dates, try dayfirst=True and pick the better hit-rate
            better = dt0
            if (dt0.isna().mean() > 0.2) or s.astype(str).str.contains(r"/").mean() > 0.2:
                dt1 = try_parse(True)
                if dt1.isna().mean() < dt0.isna().mean():
                    better = dt1
            dt = better
        else:
            dt = try_parse(dayfirst)
        
        

        # Fallback through a few common explicit formats
        if dt.isna().any():
            for fmt in (
                "%Y-%m-%d %H:%M:%S",
                "%Y-%m-%d %H:%M",
                "%m/%d/%Y %H:%M:%S",
                "%d/%m/%Y %H:%M:%S",
                "%m/%d/%Y %H:%M",
                "%d/%m/%Y %H:%M",
                "%Y-%m-%dT%H:%M:%S.%fZ",
                "%Y-%m-%dT%H:%M:%S%z",
            ):
                try:
                    dt2 = pd.to_datetime(s, format=fmt, errors="coerce", utc=force_utc)
                    dt = dt.fillna(dt2)
                except Exception:
                    pass
        out = dt

    # Normalize TZ for plotting backends (mplfinance prefers tz-naive)
    from pandas.api.types import is_datetime64tz_dtype

    if tz_naive and is_datetime64tz_dtype(out.dtype):
        # drop timezone after normalizing to UTC
        out = out.dt.tz_convert("UTC").dt.tz_localize(None)

    return out

def load_csv(file_like_or_bytes,
             *, dayfirst: Optional[bool] = None,
             epoch_unit: Optional[str] = None,
             tz_naive: bool = True) -> pd.DataFrame:
    """Robust CSV loader that parses many time formats and normalizes OHLCV."""
    # Normalize to bytes -> BytesIO (avoids 'already read' issues)
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

    # Flexible column picking
    cols = {c.lower().strip(): c for c in df.columns}
    def pick(*names):
        for n in names:
            for k, v in cols.items():
                if k == n.lower():
                    return v
        return None

    ts_col = pick("timestamp", "time", "date", "datetime", "open time", "time (utc)")
    o_col  = pick("open", "o")
    h_col  = pick("high", "h")
    l_col  = pick("low", "l")
    c_col  = pick("close", "c")
    v_col  = pick("volume", "v", "vol")

    if ts_col is None or o_col is None or h_col is None or l_col is None or c_col is None:
        raise ValueError("CSV must include time, open, high, low, close (volume optional).")

    # Parse time robustly (auto-detects epochs/Excel/day-first; keeps UTC clock, drops tz info by default)
    t = _parse_time_column(df[ts_col], dayfirst=dayfirst, epoch_unit=epoch_unit,
                           force_utc=True, tz_naive=tz_naive)

    out = pd.DataFrame({
        "time":  t,
        "open":  pd.to_numeric(df[o_col], errors="coerce"),
        "high":  pd.to_numeric(df[h_col], errors="coerce"),
        "low":   pd.to_numeric(df[l_col], errors="coerce"),
        "close": pd.to_numeric(df[c_col], errors="coerce"),
        "volume": pd.to_numeric(df[v_col], errors="coerce") if v_col else np.nan,
    })

    # Clean: drop rows with bad time/price, sort, dedupe times
    out = out.dropna(subset=["time", "open", "high", "low", "close"])
    out = out.drop_duplicates(subset="time").sort_values("time").reset_index(drop=True)

    return out

import re
def infer_tf_from_name(name: str) -> Optional[str]:
    """
    Return canonical TF key from a filename.
    Handles 1W / W / weekly / week, 1D / D / daily, 4H / H4 / 240,
    1H / H1 / 60, 30m / m30, 15m / m15, etc.
    """
    s = name.lower()
    # quick normalize
    s = s.replace(" ", "_")

    # exact-ish signals first (use regex word boundaries where sensible)
    patterns = [
        (r"(?:^|_|-)(1w|w|weekly|week)(?:_|-|\.|$)", "Weekly"),
        (r"(?:^|_|-)(1d|d|daily|day)(?:_|-|\.|$)",   "Daily"),

        # 4H variants: 4h, h4, 240, 240m
        (r"(?:^|_|-)(4h|h4)(?:_|-|\.|$)",            "4H"),
        (r"(?:^|_|-)(240m?|240min)(?:_|-|\.|$)",     "4H"),

        # 1H variants: 1h, h1, 60, 60m
        (r"(?:^|_|-)(1h|h1)(?:_|-|\.|$)",            "1H"),
        (r"(?:^|_|-)(60m?|60min)(?:_|-|\.|$)",       "1H"),

        # 30m variants: 30m, m30
        (r"(?:^|_|-)(30m?|m30)(?:_|-|\.|$)",         "30m"),

        # 15m variants: 15m, m15
        (r"(?:^|_|-)(15m?|m15)(?:_|-|\.|$)",         "15m"),
    ]
    for pat, tf in patterns:
        if re.search(pat, s):
            return tf
    return None

def infer_tf_from_df(df: pd.DataFrame) -> Optional[str]:
    if len(df) < 5: return None
    dt = df["time"].diff().dt.total_seconds().median()
    if pd.isna(dt): return None
    # map within tolerance
    def near(x, tgt, tol): return abs(x - tgt) <= tol
    if near(dt, 7*24*3600, 2*24*3600): return "Weekly"
    if near(dt, 24*3600, 4*3600): return "Daily"
    if near(dt, 4*3600, 15*60): return "4H"
    if near(dt, 3600, 5*60): return "1H"
    if near(dt, 1800, 3*60): return "30m"
    if near(dt, 900, 2*60): return "15m"
    return None

def apply_mtf_confluence(detected: Dict[str, Tuple[List[OB], List[OB]]], tf_order: List[str]) -> None:
    """
    If a lower-TF OB sits fully inside a higher-TF OB, add +0.5 score and reason.
    Mutates in place.
    """
    # build accumulative higher-TF zones per side
    higher_map = {"bullish": [], "bearish": []}
    for tf in tf_order:
        bulls, bears = detected.get(tf, ([], []))
        # apply against currently known higher-TF pools
        for z in bulls:
            for hi in higher_map["bullish"]:
                if z.start >= hi.start and z.end <= hi.end:
                    z.score += 0.5
                    z.reasons.append("Within higher-TF OB")
                    break
        for z in bears:
            for hi in higher_map["bearish"]:
                if z.start >= hi.start and z.end <= hi.end:
                    z.score += 0.5
                    z.reasons.append("Within higher-TF OB")
                    break
        # add current TF to higher pools for lower TFs to use later
        higher_map["bullish"].extend(bulls)
        higher_map["bearish"].extend(bears)


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

# ========== OB Backtest (self-contained) ==========

@dataclass(frozen=True)
class OBBacktestIdea:
    side: str             # "bullish" or "bearish"
    entry: float
    sl: float
    tp1: float
    tp2: float
    created_idx: int      # bar index when BOS confirms (z.bos_index)
    note: str = ""

def _ob_bt_bar_hits(hi: float, lo: float, level: float) -> bool:
    return (lo <= level <= hi)

def _ob_bt_simulate(df: pd.DataFrame,
                    idea: OBBacktestIdea,
                    expiry_bars: int = 30,
                    max_hold_bars: int = 200) -> dict:
    """
    Two-stage simulation:
      1) Wait for limit fill at entry (within expiry_bars) after created_idx.
      2) After fill:
         - Stage A: TP1 vs SL (worst-case if both in same bar -> SL).
         - If TP1 hit: move SL to entry (breakeven). Stage B: TP2 vs Entry (worst-case -> BE).
    Returns a dict with status in {"NoFill","SL","TP2","BE","TimedOut"} and R.
    """
    side   = idea.side.lower()
    entry  = float(idea.entry)
    sl     = float(idea.sl)
    tp1    = float(idea.tp1)
    tp2    = float(idea.tp2)
    i0     = int(idea.created_idx) + 1
    n      = len(df)

    # 1) Wait for fill
    fill_idx = None
    for i in range(i0, min(n, i0 + expiry_bars + 1)):
        hi = float(df["high"].iloc[i]); lo = float(df["low"].iloc[i])
        if _ob_bt_bar_hits(hi, lo, entry):
            fill_idx = i
            break

    if fill_idx is None:
        return {"status":"NoFill","filled":False,"fill_idx":None,"exit_idx":None,"exit_price":None,"R":0.0,"bars_held":0}

    # Guard
    risk = abs(entry - sl)
    if risk <= 0:
        return {"status":"BadRisk","filled":False,"fill_idx":fill_idx,"exit_idx":None,"exit_price":None,"R":0.0,"bars_held":0}

    # 2A) After fill: TP1 vs SL
    def r_mult(px_exit: float) -> float:
        if side == "bullish":
            return (px_exit - entry) / risk
        return (entry - px_exit) / risk

    exit_idx = None
    exit_px  = None
    # Stage-A scan includes the fill bar (limit order can hit + continue)
    for j in range(fill_idx, min(n, fill_idx + max_hold_bars + 1)):
        hi = float(df["high"].iloc[j]); lo = float(df["low"].iloc[j])
        hit_sl  = _ob_bt_bar_hits(hi, lo, sl)
        hit_tp1 = _ob_bt_bar_hits(hi, lo, tp1)
        if hit_sl and hit_tp1:
            # Worst-case resolution: SL first
            exit_idx, exit_px = j, sl
            return {"status":"SL","filled":True,"fill_idx":fill_idx,"exit_idx":exit_idx,"exit_price":exit_px,"R":r_mult(exit_px),"bars_held":exit_idx-fill_idx}
        if hit_sl:
            exit_idx, exit_px = j, sl
            return {"status":"SL","filled":True,"fill_idx":fill_idx,"exit_idx":exit_idx,"exit_price":exit_px,"R":r_mult(exit_px),"bars_held":exit_idx-fill_idx}
        if hit_tp1:
            # 2B) Move SL to entry (breakeven), then TP2 vs Entry
            be_level = entry
            # Resolve same-bar first (worst-case → BE if both)
            hit_be_same = _ob_bt_bar_hits(hi, lo, be_level)
            hit_tp2_same= _ob_bt_bar_hits(hi, lo, tp2)
            if hit_tp2_same and hit_be_same:
                exit_idx, exit_px = j, be_level
                return {"status":"BE","filled":True,"fill_idx":fill_idx,"exit_idx":exit_idx,"exit_price":exit_px,"R":0.0,"bars_held":exit_idx-fill_idx}
            if hit_tp2_same:
                exit_idx, exit_px = j, tp2
                return {"status":"TP2","filled":True,"fill_idx":fill_idx,"exit_idx":exit_idx,"exit_price":exit_px,"R":r_mult(exit_px),"bars_held":exit_idx-fill_idx}
            if hit_be_same:
                exit_idx, exit_px = j, be_level
                return {"status":"BE","filled":True,"fill_idx":fill_idx,"exit_idx":exit_idx,"exit_price":exit_px,"R":0.0,"bars_held":exit_idx-fill_idx}

            # Continue to next bars
            for k in range(j+1, min(n, fill_idx + max_hold_bars + 1)):
                hi2 = float(df["high"].iloc[k]); lo2 = float(df["low"].iloc[k])
                hit_be = _ob_bt_bar_hits(hi2, lo2, be_level)
                hit_tp2= _ob_bt_bar_hits(hi2, lo2, tp2)
                if hit_tp2 and hit_be:
                    exit_idx, exit_px = k, be_level
                    return {"status":"BE","filled":True,"fill_idx":fill_idx,"exit_idx":exit_idx,"exit_price":exit_px,"R":0.0,"bars_held":exit_idx-fill_idx}
                if hit_tp2:
                    exit_idx, exit_px = k, tp2
                    return {"status":"TP2","filled":True,"fill_idx":fill_idx,"exit_idx":exit_idx,"exit_price":exit_px,"R":r_mult(exit_px),"bars_held":exit_idx-fill_idx}
                if hit_be:
                    exit_idx, exit_px = k, be_level
                    return {"status":"BE","filled":True,"fill_idx":fill_idx,"exit_idx":exit_idx,"exit_price":exit_px,"R":0.0,"bars_held":exit_idx-fill_idx}
            # Timed out after TP1 (SL at entry) → BE
            return {"status":"BE","filled":True,"fill_idx":fill_idx,"exit_idx":None,"exit_price":None,"R":0.0,"bars_held":(min(n, fill_idx + max_hold_bars + 1)-fill_idx)}

    # Timed out without reaching TP1 or SL
    return {"status":"TimedOut","filled":True,"fill_idx":fill_idx,"exit_idx":None,"exit_price":None,"R":0.0,"bars_held":(min(n, fill_idx + max_hold_bars + 1)-fill_idx)}

def _ob_bt_build_ideas_from_zones(df: pd.DataFrame,
                                  zones: List[OB],
                                  side: str,
                                  *,
                                  stop_buffer_mode: str,
                                  stop_buffer_abs: float,
                                  stop_buffer_pct: float,
                                  stop_buffer_atr: float,
                                  rr_min: float = 1.0) -> List[OBBacktestIdea]:
    """
    Build limit ideas at OB mid with SL beyond far side + buffer determined *at the time of BOS*.
    Targets: **2R/3R fallback** to guarantee TP1 & TP2 exist.
    """
    ideas: List[OBBacktestIdea] = []
    for z in zones:
        created_idx = int(z.bos_index)
        if created_idx <= 0 or created_idx >= len(df)-1:
            continue

        # Use data up to creation to compute ATR-based buffers
        df_sub = df.iloc[:created_idx+1]
        entry = float((z.start + z.end) / 2.0)
        buf = calc_stop_buffer(
            df=df_sub, entry=entry, side=side,
            mode=stop_buffer_mode,
            abs_buf=stop_buffer_abs,
            pct_buf=stop_buffer_pct,
            atr_mult=stop_buffer_atr
        )
        sl = (min(z.start, z.end) - buf) if side == "bullish" else (max(z.start, z.end) + buf)

        risk = abs(entry - sl)
        if risk <= 0:
            continue

        # Always provide TP1/TP2 via 2R/3R
        if side == "bullish":
            tp1 = entry + 1.0 * risk
            tp2 = entry + 2.0 * risk
        else:
            tp1 = entry - 1.0 * risk
            tp2 = entry - 2.0 * risk

        rr1 = abs(tp1 - entry) / risk
        rr2 = abs(tp2 - entry) / risk
        if rr2 < float(rr_min):
            continue

        ideas.append(OBBacktestIdea(
            side=side, entry=round(entry,6), sl=round(sl,6),
            tp1=round(float(tp1),6), tp2=round(float(tp2),6),
            created_idx=created_idx,
            note=f"{side.title()} OB mid; SL beyond far side + buffer; targets 1R/2R."
        ))
    return ideas

def ob_backtest_tf(df: pd.DataFrame,
                   bulls: List[OB],
                   bears: List[OB],
                   *,
                   rr_min: float = 2.0,
                   expiry_bars: int = 30,
                   max_hold_bars: int = 200,
                   stop_buffer_mode: str = "ATR14 × multiple",
                   stop_buffer_abs: float = 0.02,
                   stop_buffer_pct: float = 0.0,
                   stop_buffer_atr: float = 0.0) -> pd.DataFrame:
    """
    Build ideas from zones (both sides) and simulate.

    Validity rule (pre-fill):
      - While waiting for a fill, if the invalidation level (the SL you computed from the zone + buffer)
        is tagged BEFORE entry within `expiry_bars`, the idea is cancelled as NoFill with `invalidated=True`.

    BE rule (post-fill):
      - Once TP1 is hit after fill, move SL to entry (breakeven). From there, only TP2 or BE can close the trade.

    Returns a trades DataFrame with statuses in: TP2 / SL / BE / NoFill / TimedOut
    and extra columns: invalidated [bool], invalidated_idx, invalidated_time.
    """
    # 1) Build trade ideas (builder should already enforce TP floors: TP1>=1R, TP2>=2R)
    ideas_long  = _ob_bt_build_ideas_from_zones(
        df, bulls, "bullish",
        stop_buffer_mode=stop_buffer_mode,
        stop_buffer_abs=stop_buffer_abs,
        stop_buffer_pct=stop_buffer_pct,
        stop_buffer_atr=stop_buffer_atr,
        rr_min=rr_min
    )
    ideas_short = _ob_bt_build_ideas_from_zones(
        df, bears, "bearish",
        stop_buffer_mode=stop_buffer_mode,
        stop_buffer_abs=stop_buffer_abs,
        stop_buffer_pct=stop_buffer_pct,
        stop_buffer_atr=stop_buffer_atr,
        rr_min=rr_min
    )
    ideas = ideas_long + ideas_short
    ideas.sort(key=lambda x: x.created_idx)

    rows = []
    H, L, C = df["high"].values, df["low"].values, df["close"].values
    T = df["time"].values if "time" in df.columns else None
    N = len(df)

    for idea in ideas:
        created_idx = int(idea.created_idx)
        side  = idea.side
        entry = float(idea.entry)
        sl    = float(idea.sl)   # this is also the pre-fill invalidation level
        tp1   = float(idea.tp1)
        tp2   = float(idea.tp2)
        note  = idea.note

        risk = abs(entry - sl)
        if risk <= 0 or created_idx >= N - 1:
            rows.append({
                "created_time": (T[created_idx] if T is not None else created_idx),
                "created_idx": created_idx,
                "side": side, "entry": entry, "sl": sl, "tp1": tp1, "tp2": tp2, "note": note,
                "status": "NoFill", "fill_idx": None, "exit_idx": None, "exit_price": None, "R": 0.0, "bars_held": 0,
                "exit_time": None,
                "invalidated": False, "invalidated_idx": None, "invalidated_time": None,
            })
            continue

        # 2) Wait for fill within expiry window, while checking *validity* (invalidation before entry)
        last_i_for_fill = min(N - 1, created_idx + int(expiry_bars))
        fill_idx = None
        invalidated = False
        invalid_i = None

        if side == "bullish":
            for i in range(created_idx + 1, last_i_for_fill + 1):
                entry_hit = (L[i] <= entry <= H[i])
                inval_hit = (L[i] <= sl)
                # Tie-breaking if both touched same bar: prefer entry-first (gives the trade a chance),
                # post-fill logic can still stop it on the same bar if price snaps through SL.
                if entry_hit and not inval_hit:
                    fill_idx = i; break
                if inval_hit and not entry_hit:
                    invalidated = True; invalid_i = i; break
                if entry_hit and inval_hit:
                    fill_idx = i; break  # entry first
        else:  # bearish
            for i in range(created_idx + 1, last_i_for_fill + 1):
                entry_hit = (L[i] <= entry <= H[i])
                inval_hit = (H[i] >= sl)
                if entry_hit and not inval_hit:
                    fill_idx = i; break
                if inval_hit and not entry_hit:
                    invalidated = True; invalid_i = i; break
                if entry_hit and inval_hit:
                    fill_idx = i; break

        if invalidated:
            rows.append({
                "created_time": (T[created_idx] if T is not None else created_idx),
                "created_idx": created_idx,
                "side": side, "entry": entry, "sl": sl, "tp1": tp1, "tp2": tp2, "note": note,
                "status": "NoFill",  # cancelled before fill due to invalidation
                "fill_idx": None,
                "exit_idx": None,
                "exit_price": None,
                "R": 0.0,
                "bars_held": 0,
                "exit_time": None,
                "invalidated": True,
                "invalidated_idx": invalid_i,
                "invalidated_time": (T[invalid_i] if (T is not None and invalid_i is not None) else None),
            })
            continue

        if fill_idx is None:
            # No fill in time (but not invalidated)
            rows.append({
                "created_time": (T[created_idx] if T is not None else created_idx),
                "created_idx": created_idx,
                "side": side, "entry": entry, "sl": sl, "tp1": tp1, "tp2": tp2, "note": note,
                "status": "NoFill", "fill_idx": None, "exit_idx": None, "exit_price": None, "R": 0.0, "bars_held": 0,
                "exit_time": None,
                "invalidated": False, "invalidated_idx": None, "invalidated_time": None,
            })
            continue

        # 3) Manage after fill (BE after TP1)
        moved_to_be = False
        current_sl  = sl
        status = None; exit_idx = None; exit_price = None; R = 0.0
        last_i_for_exit = min(N - 1, fill_idx + int(max_hold_bars))

        for i in range(fill_idx, last_i_for_exit + 1):
            hi, lo = H[i], L[i]

            if side == "bullish":
                if moved_to_be:
                    if hi >= tp2:
                        status, exit_idx, exit_price = "TP2", i, tp2
                        R = (exit_price - entry) / risk
                        break
                    if lo <= current_sl:
                        status, exit_idx, exit_price = "BE", i, entry
                        R = 0.0
                        break
                else:
                    hit_tp1 = (hi >= tp1)
                    hit_sl  = (lo <= current_sl)
                    if hit_tp1 and hit_sl:
                        # Prefer TP1 first, then same-bar check for TP2 or BE
                        moved_to_be = True
                        current_sl = entry
                        if hi >= tp2:
                            status, exit_idx, exit_price = "TP2", i, tp2
                            R = (exit_price - entry) / risk
                        elif lo <= current_sl:
                            status, exit_idx, exit_price = "BE", i, entry
                            R = 0.0
                        if status: break
                    elif hit_tp1:
                        moved_to_be = True
                        current_sl = entry
                        if hi >= tp2:
                            status, exit_idx, exit_price = "TP2", i, tp2
                            R = (exit_price - entry) / risk
                            break
                        if lo <= current_sl:
                            status, exit_idx, exit_price = "BE", i, entry
                            R = 0.0
                            break
                    elif hit_sl:
                        status, exit_idx, exit_price = "SL", i, current_sl
                        R = (exit_price - entry) / risk
                        break

            else:  # bearish
                if moved_to_be:
                    if lo <= tp2:
                        status, exit_idx, exit_price = "TP2", i, tp2
                        R = (entry - exit_price) / risk
                        break
                    if hi >= current_sl:
                        status, exit_idx, exit_price = "BE", i, entry
                        R = 0.0
                        break
                else:
                    hit_tp1 = (lo <= tp1)
                    hit_sl  = (hi >= current_sl)
                    if hit_tp1 and hit_sl:
                        moved_to_be = True
                        current_sl = entry
                        if lo <= tp2:
                            status, exit_idx, exit_price = "TP2", i, tp2
                            R = (entry - exit_price) / risk
                        elif hi >= current_sl:
                            status, exit_idx, exit_price = "BE", i, entry
                            R = 0.0
                        if status: break
                    elif hit_tp1:
                        moved_to_be = True
                        current_sl = entry
                        if lo <= tp2:
                            status, exit_idx, exit_price = "TP2", i, tp2
                            R = (entry - exit_price) / risk
                            break
                        if hi >= current_sl:
                            status, exit_idx, exit_price = "BE", i, entry
                            R = 0.0
                            break
                    elif hit_sl:
                        status, exit_idx, exit_price = "SL", i, current_sl
                        R = (entry - exit_price) / risk
                        break

        if status is None:
            status = "TimedOut"
            exit_idx = last_i_for_exit
            exit_price = float(C[exit_idx])
            if side == "bullish":
                R = (exit_price - entry) / risk
            else:
                R = (entry - exit_price) / risk

        bars_held = int(exit_idx - fill_idx) if (fill_idx is not None and exit_idx is not None) else 0

        rows.append({
            "created_time": (T[created_idx] if T is not None else created_idx),
            "created_idx": created_idx,
            "side": side,
            "entry": entry,
            "sl": sl,
            "tp1": tp1,
            "tp2": tp2,
            "note": note,
            "status": status,
            "fill_idx": fill_idx,
            "exit_idx": exit_idx,
            "exit_price": exit_price,
            "R": float(R),
            "bars_held": bars_held,
            "exit_time": (T[exit_idx] if (T is not None and exit_idx is not None) else None),
            "invalidated": False,
            "invalidated_idx": None,
            "invalidated_time": None,
        })

    return pd.DataFrame(rows)

def ob_backtest_stats(trades: pd.DataFrame, risk_cash: float = 1000.0) -> dict:
    """
    Stats with BE tracking and $ metrics (assumes fixed cash risk per trade).
    Win = TP2, Loss = SL, BE = 0R.
    """
    if trades.empty:
        return {"trades":0,"wins":0,"losses":0,"breakevens":0,"win_rate_pct":0.0,"win_over_wl_pct":0.0,
                "avg_R":0.0,"expectancy_R":0.0,"profit_factor":0.0,"median_R":0.0,"p25_R":0.0,"p75_R":0.0,
                "max_drawdown_R":0.0,"sharpe_R":0.0,"gross_R":0.0,"gross_$":0.0,"avg_$":0.0,"exp_$":0.0}

    closed = trades[trades["status"].isin(["TP2","SL","BE"])].copy()
    wins   = (closed["status"] == "TP2").sum()
    losses = (closed["status"] == "SL").sum()
    be     = (closed["status"] == "BE").sum()

    wr_all = wins / max(1, len(closed)) * 100.0
    wr_wl  = wins / max(1, (wins + losses)) * 100.0

    R = closed["R"].fillna(0.0).values
    avg_R = float(np.mean(R)) if len(R) else 0.0
    expectancy = avg_R

    gross_win  = float(closed.loc[closed["R"] > 0, "R"].sum())
    gross_loss = float(-closed.loc[closed["R"] < 0, "R"].sum())
    pf = (gross_win / gross_loss) if gross_loss > 0 else np.nan

    median_R = float(closed["R"].median()) if len(closed) else 0.0
    p25 = float(closed["R"].quantile(0.25)) if len(closed) else 0.0
    p75 = float(closed["R"].quantile(0.75)) if len(closed) else 0.0

    # Equity curve stats
    eq = closed.sort_values(["exit_idx","fill_idx"])["R"].cumsum().reset_index(drop=True)
    roll_max = eq.cummax()
    drawdown = eq - roll_max
    max_dd = float(drawdown.min()) if len(drawdown) else 0.0
    sharpe = (np.mean(R) / (np.std(R, ddof=1) + 1e-12) * np.sqrt(12)) if len(R) >= 2 else 0.0

    gross_R   = float(closed["R"].sum())
    avg_cash  = avg_R * float(risk_cash)
    exp_cash  = expectancy * float(risk_cash)
    gross_cash= gross_R * float(risk_cash)

    return {
        "trades": int(len(closed)),
        "wins": int(wins),
        "losses": int(losses),
        "breakevens": int(be),
        "win_rate_pct": round(wr_all, 1),
        "win_over_wl_pct": round(wr_wl, 1),
        "avg_R": round(avg_R, 3),
        "expectancy_R": round(expectancy, 3),
        "profit_factor": float(round(pf, 3)) if np.isfinite(pf) else float("nan"),
        "median_R": round(median_R, 3),
        "p25_R": round(p25, 3),
        "p75_R": round(p75, 3),
        "max_drawdown_R": round(max_dd, 3),
        "sharpe_R": round(sharpe, 3),
        "gross_R": round(gross_R, 3),
        "gross_$": round(gross_cash, 2),  # keys can include '$' since they are strings
        "avg_$": round(avg_cash, 2),
        "exp_$": round(exp_cash, 2),
    }

def _ob_bt_plot_equity(trades: pd.DataFrame, title: str = "Equity (R) Curve"):
    closed = trades[trades["status"].isin(["TP2","SL","BE"])]
    if closed.empty:
        st.info("No closed trades to plot.")
        return
    closed = closed.sort_values(["exit_idx","fill_idx"], na_position="last")
    eq = closed["R"].cumsum()
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    ax.plot(eq.values)
    ax.set_title(title); ax.set_xlabel("Closed trades (chronological)"); ax.set_ylabel("Cumulative R")
    st.pyplot(fig)


# ---- Trade plan helper ----
def compute_tradeplan_for_side(
    side: str,
    zones: List[OB],
    opposing_zones: List[OB],
    stop_buffer: float,
    tf_name: str,
    tf_bias: str,
    target_policy: str = "Opposing OB mids",
    df: Optional[pd.DataFrame] = None,
    piv_hi: Optional[np.ndarray] = None,
    piv_lo: Optional[np.ndarray] = None,
    # NEW:
    stop_buffer_mode: str = "Absolute",
    stop_buffer_abs: float = 0.02,
    stop_buffer_pct: float = 0.0,
    stop_buffer_atr: float = 0.0,
) -> List[TradePlan]:

    """
    target_policy ∈ {"Opposing OB mids", "2R/3R", "Structure (swing highs/lows)"}
    If target_policy == Structure, df and pivot masks are used to fetch swing targets after BOS.
    """
    plans: List[TradePlan] = []
    opp_mids = [ (z.start + z.end) / 2 for z in opposing_zones ]

    for z in zones:
        if getattr(z, "consumed", False):
            continue

        lo, hi = z.start, z.end
        entry = (lo + hi) / 2
        buf = calc_stop_buffer(
            df=df, entry=entry, side=side,
            mode=stop_buffer_mode,
            abs_buf=stop_buffer_abs if stop_buffer_mode=="Absolute" else stop_buffer,
            pct_buf=stop_buffer_pct,
            atr_mult=stop_buffer_atr
        )

        sl = (lo - buf) if side == "bullish" else (hi + buf)

        # ---------- TARGETS (safe for all policies) ----------
        tp1, tp2 = None, None
        risk = abs(entry - sl)

        # Primary selection per policy
        if target_policy == "Opposing OB mids":
            if opp_mids:
                tp1 = opp_mids[0]
                if len(opp_mids) > 1:
                    tp2 = opp_mids[1]

        elif target_policy == "2R/3R":
            # EXACT 2R / 3R targets
            if risk > 0:
                if side == "bullish":
                    tp1 = entry + 2.0 * risk
                    tp2 = entry + 3.0 * risk
                else:
                    tp1 = entry - 2.0 * risk
                    tp2 = entry - 3.0 * risk

        elif target_policy == "Structure (swing highs/lows)" and df is not None and piv_hi is not None and piv_lo is not None:
            if side == "bullish":
                swings = last_two_swings_after(piv_hi, z.bos_index)
                levels = [float(df["high"].iloc[s]) for s in swings]
            else:
                swings = last_two_swings_after(piv_lo, z.bos_index)
                levels = [float(df["low"].iloc[s]) for s in swings]
            if levels:
                tp1 = levels[0]
                if len(levels) > 1:
                    tp2 = levels[1]

        # Minimum RR floors (None-safe) — always guarantee TP1 ≥ 1R and TP2 ≥ 2R
        if risk > 0:
            if side == "bullish":
                min_tp1 = entry + 1.0 * risk
                min_tp2 = entry + 2.0 * risk
                tp1 = min_tp1 if tp1 is None else max(tp1, min_tp1)
                # tp2 must be ≥ tp1 and ≥ 2R
                tp2_floor = max(min_tp2, tp1)
                tp2 = tp2_floor if tp2 is None else max(tp2, tp2_floor)
            else:
                min_tp1 = entry - 1.0 * risk
                min_tp2 = entry - 2.0 * risk
                tp1 = min_tp1 if tp1 is None else min(tp1, min_tp1)
                # tp2 must be ≤ tp1 and ≤ −2R
                tp2_floor = min(min_tp2, tp1)
                tp2 = tp2_floor if tp2 is None else min(tp2, tp2_floor)

        # Final fallbacks if still None (shouldn’t happen after floors, but stay defensive)
        if tp1 is None:
            tp1 = entry + (1.0 * (entry - sl)) if side == "bullish" else entry - (1.0 * (sl - entry))
        if tp2 is None:
            tp2 = entry + (2.0 * (entry - sl)) if side == "bullish" else entry - (2.0 * (sl - entry))

        # Keep near→far ordering ONLY if both exist (avoid None comparisons)
        if tp1 is not None and tp2 is not None:
            if abs(tp1 - entry) > abs(tp2 - entry):
                tp1, tp2 = tp2, tp1

        # Ensure targets are on the correct side of entry
        if side == "bullish":
            tp1 = max(tp1, entry); tp2 = max(tp2, entry)
        else:
            tp1 = min(tp1, entry); tp2 = min(tp2, entry)

        def rr(e, s, t):
            r = abs(e - s)
            return round(abs(t - e) / r, 2) if r > 0 else None



        rr1 = rr(entry, sl, tp1)
        rr2 = rr(entry, sl, tp2)

        reasons = z.reasons.copy()
        if getattr(z, "mitigated", False):
            reasons.append(f"Already mitigated {z.touches}x (~{int(z.fill_ratio*100)}% fill)")
        else:
            reasons.append("Fresh/untouched since BOS")
        if tf_bias != "neutral":
            reasons.append(f"MTF bias on {tf_name} appears {tf_bias}")
        reasons.append(f"Target policy: {target_policy}")

        invalids = [
            f"{tf_name} close beyond far side of OB (for {side})",
            "Opposite-direction CHoCH/BOS soon after entry",
            "Multiple mitigations with no displacement follow-through",
        ]

        plans.append(TradePlan(
            entry=round(entry, 6),
            sl=round(sl, 6),
            tp1=round(tp1, 6),
            tp2=round(tp2, 6),
            rr1=rr1, rr2=rr2,
            invalidations=invalids,
            reasons=reasons
        ))
    return plans



def fvg_mask(df: pd.DataFrame, span: int = 3) -> Tuple[np.ndarray, np.ndarray]:
    """
    FVG detection: span = 3 (classic) or 5 (wider). Returns (bull_mask, bear_mask).
    """
    n = len(df)
    highs = df["high"].values
    lows  = df["low"].values
    bull = np.zeros(n, dtype=bool)
    bear = np.zeros(n, dtype=bool)

    if span == 3:
        for i in range(1, n - 1):
            if lows[i + 1] > highs[i - 1]:
                bull[i] = True
            if highs[i + 1] < lows[i - 1]:
                bear[i] = True
    else:  # span == 5
        for i in range(2, n - 2):
            if lows[i + 2] > highs[i - 2]:
                bull[i] = True
            if highs[i + 2] < lows[i - 2]:
                bear[i] = True
    return bull, bear

def bias_label(df: pd.DataFrame, ma=50) -> str:
    sma = df["close"].rolling(ma).mean()
    if len(sma.dropna()) == 0:
        return "neutral"
    last = df["close"].iloc[-1]; last_sma = sma.iloc[-1]
    if pd.isna(last_sma): return "neutral"
    if last > last_sma * 1.005: return "bullish"
    if last < last_sma * 0.995: return "bearish"
    return "neutral"

def equal_levels(a: np.ndarray, tol_ratio=0.0015) -> List[List[int]]:
    if len(a) < 2: return []
    idx = np.argsort(a)
    groups, group = [], [idx[0]]
    for i in range(1, len(idx)):
        prev = a[idx[i - 1]]; cur = a[idx[i]]
        tol  = max(abs(prev), 1.0) * tol_ratio
        if abs(cur - prev) <= tol:
            group.append(idx[i])
        else:
            if len(group) >= 2: groups.append(group)
            group = [idx[i]]
    if len(group) >= 2: groups.append(group)
    return groups

# ---------- TPO profile (Value Area/POC) ----------
def _value_area_from_hist(bins: np.ndarray, counts: np.ndarray, coverage: float=0.70) -> Tuple[float,float,float]:
    counts = counts.astype(float)
    total = counts.sum()
    if total <= 0: 
        return float(bins[np.argmax(counts)]), float(bins[np.argmax(counts)]), float(bins[np.argmax(counts)])
    poc_idx = int(np.argmax(counts))
    cum = counts[poc_idx]
    left = right = poc_idx
    # expand to cover target
    while cum / total < coverage:
        left_val  = counts[left-1]  if left-1  >= 0 else -1
        right_val = counts[right+1] if right+1 < len(counts) else -1
        if right_val >= left_val and right+1 < len(counts):
            right += 1; cum += counts[right]
        elif left-1 >= 0:
            left -= 1; cum += counts[left]
        else:
            break
    poc = float(bins[poc_idx])
    val = float(bins[left])
    vah = float(bins[right])
    return val, vah, poc

def tpo_profile(df: pd.DataFrame, coverage: float=0.70) -> Dict[str, float]:
    if len(df) == 0:
        return {"VAL": np.nan, "VAH": np.nan, "POC": np.nan}
    # bin size ~ fine but stable across instruments, scale by ATR
    a = atr(df, 100)
    approx_tick = max((a.iloc[-1] / 200.0) if not np.isnan(a.iloc[-1]) else (df["close"].iloc[-1]*0.0005), 1e-6)
    lo = float(df["low"].min()); hi = float(df["high"].max())
    # create bins on a fixed grid
    nb = int(max(5, min(400, math.ceil((hi - lo) / approx_tick))))
    bins = np.linspace(lo, hi, nb)
    counts = np.zeros_like(bins, dtype=int)
    # count price "touch" per candle (time-at-price approximation)
    lows = df["low"].values; highs = df["high"].values
    for L,H in zip(lows, highs):
        if not np.isfinite(L) or not np.isfinite(H): continue
        # which bins are inside [L,H]?
        mask = (bins >= L) & (bins <= H)
        counts[mask] += 1
    VAL, VAH, POC = _value_area_from_hist(bins, counts, coverage=coverage)
    return {"VAL": VAL, "VAH": VAH, "POC": POC}

# ---------- CVD (true if present, else proxy) ----------
def cvd_series(df: pd.DataFrame) -> pd.Series:
    if "ask_volume" in df and "bid_volume" in df and df["ask_volume"].notna().any() and df["bid_volume"].notna().any():
        delta = (df["ask_volume"].fillna(0) - df["bid_volume"].fillna(0))
    else:
        # keep this as a Series so we can use .replace/.ffill
        sign = pd.Series(np.sign(df["close"] - df["open"]), index=df.index)
        sign = sign.replace(0, np.nan).ffill().fillna(0)
        delta = sign * df["volume"].fillna(0)
    return delta.cumsum()

def detect_cvd_divergence(price: pd.Series, cvd: pd.Series, lookback: int=120) -> Optional[str]:
    """Simple last swing divergence. Returns 'bullish', 'bearish' or None."""
    if len(price) < 10: return None
    p = price.tail(lookback).reset_index(drop=True)
    c = cvd.tail(lookback).reset_index(drop=True)
    # find last two swing highs and lows
    def last_two_swings(arr, up=True, L=3, R=3):
        n = len(arr); idx=[]
        for i in range(L, n-R):
            if up:
                if all(arr[i] > arr[i-j] for j in range(1,L+1)) and all(arr[i] > arr[i+j] for j in range(1,R+1)):
                    idx.append(i)
            else:
                if all(arr[i] < arr[i-j] for j in range(1,L+1)) and all(arr[i] < arr[i+j] for j in range(1,R+1)):
                    idx.append(i)
        return idx[-2:] if len(idx) >= 2 else []
    ph = last_two_swings(p.values, up=True)
    pl = last_two_swings(p.values, up=False)
    div = None
    # bearish: price HH, CVD LH
    if len(ph)==2:
        i1,i2 = ph
        if p[i2] > p[i1] and c[i2] < c[i1]:
            div = "bearish"
    # bullish: price LL, CVD HL
    if len(pl)==2:
        i1,i2 = pl
        if p[i2] < p[i1] and c[i2] > c[i1]:
            div = "bullish" if div is None else div
    return div

# ---------- Display windowing ----------
def crop_for_display(df: pd.DataFrame,
                     bulls: List[OB],
                     bears: List[OB],
                     mode: str,
                     before: int,
                     after: int) -> tuple[pd.DataFrame, List[OB], List[OB]]:
    n = len(df)
    all_z = bulls + bears
    if mode.startswith("Auto-zoom") and all_z:
        target = max(all_z, key=lambda z: z.bos_index)
        left = max(0, target.index - before)
        right = min(n - 1, target.bos_index + after)
    else:
        right = n - 1
        left = max(0, right - 300)
    dfc = df.iloc[left:right+1].reset_index(drop=True)

    def shift(z: OB) -> Optional[OB]:
        if not (left <= z.index <= right or left <= z.bos_index <= right):
            return None
        zi = min(max(z.index, left), right) - left
        zb = min(max(z.bos_index, left), right) - left
        return replace(z, index=zi, bos_index=zb)

    bulls2 = [s for z in bulls if (s := shift(z)) is not None]
    bears2 = [s for z in bears if (s := shift(z)) is not None]
    return dfc, bulls2, bears2

# ---------- Core detection ----------
def detect_ob_zones(
    df: pd.DataFrame,
    use_wicks: bool = False,
    pivot_left: int = 2,
    pivot_right: int = 2,
    displacement_k: float = 1.2,          # acts as a baseline
    adaptive_disp: bool = False,
    adaptive_q: float = 0.60,
    adaptive_window: int = 60,
    displacement_mult: float = 0.8,       # user multiplier applied to adaptive floor
    min_score: float = 2.5,
    fvg_span: int = 3,
    iou_thresh: float = 0.80,
    return_debug: bool = False
) -> Tuple[List[OB], List[OB], str, Optional[dict]]:
    """
    Enhanced OB finder:
      - adaptive displacement threshold via rolling quantile of (range/ATR14)
      - FVG span 3 or 5
      - IoU-based dedupe
      - rich debug counts / reasons
    """
    n = len(df)
    dbg = {"notes": [], "steps": []}

    if n < 50:
        warn = "Insufficient candles (<50)."
        return ([], [], warn, dbg if return_debug else None)

    # Core series
    atr14 = atr(df, 14)
    sh, sl = pivots(df, pivot_left, pivot_right)
    bull_fvg, bear_fvg = fvg_mask(df, span=fvg_span)

    highs = df["high"].values
    lows  = df["low"].values
    opens = df["open"].values
    closes = df["close"].values

    # Adaptive floor: rolling quantile of R = (range/ATR)
    R = (highs - lows) / atr14.replace(0, np.nan)
    if adaptive_disp:
        floor_series = R.rolling(adaptive_window, min_periods=max(20, adaptive_window//2)).quantile(adaptive_q)
        floor_series = floor_series.bfill().ffill()
    else:
        floor_series = pd.Series(np.nan, index=df.index)

    swing_high_idx = np.where(sh)[0]
    swing_low_idx  = np.where(sl)[0]

    def last_swing_before(i, swings):
        j = swings[swings < i]
        return int(j[-1]) if len(j) else None

    bulls: List[OB] = []
    bears: List[OB] = []
    dropped = []

    for i in range(max(pivot_left + 1, 20), n):
        if pd.isna(atr14.iloc[i]) or atr14.iloc[i] == 0:
            continue

        # Effective displacement threshold this bar
        r_i = (highs[i] - lows[i]) / atr14.iloc[i]
        base = displacement_k  # baseline from preset/slider
        if adaptive_disp and not pd.isna(floor_series.iloc[i]):
            base = max(base, float(floor_series.iloc[i]) * displacement_mult)  # floor × user factor

        is_displacement = r_i > base

        # BOS up against last swing high
        lsh = last_swing_before(i, swing_high_idx)
        if lsh is not None and closes[i] > highs[lsh] and is_displacement:
            # find last bearish candle within last 30 bars
            j = i - 1; ob_j = None
            while j > 0 and j > i - 30:
                if closes[j] < opens[j]:
                    ob_j = j; break
                j -= 1
            if ob_j is not None:
                if use_wicks:
                    ob_low  = lows[ob_j]
                    ob_high = max(opens[ob_j], closes[ob_j])
                else:
                    ob_low  = min(opens[ob_j], closes[ob_j])
                    ob_high = max(opens[ob_j], closes[ob_j])

                score, reasons = 0.0, []
                score += 1.0; reasons.append("Displacement + BOS up confirmed")
                if np.any(bull_fvg[ob_j:min(n, ob_j + 5)]):
                    score += 1.0; reasons.append("Fair Value Gap (bullish) in wake of move")
                if (highs[ob_j] - lows[ob_j]) < 1.2 * atr14.iloc[ob_j]:
                    score += 1.0; reasons.append("Clean origin (compact OB candle)")
                window = df.iloc[max(0, ob_j - 15):ob_j]
                if len(window) > 5:
                    groups = equal_levels(window["low"].values)
                    if len(groups) >= 1 or (window["low"].min() < window["low"].quantile(0.15)):
                        score += 1.0; reasons.append("Liquidity sweep/inducement before OB")
                touched = np.any((lows[ob_j + 1:i] <= ob_high) & (highs[ob_j + 1:i] >= ob_low))
                if not touched:
                    score += 1.0; reasons.append("Unmitigated at creation")

                if score >= min_score:
                    bulls.append(OB(
                        side="bullish", index=ob_j, start=min(ob_low, ob_high), end=max(ob_low, ob_high),
                        open=opens[ob_j], high=highs[ob_j], low=lows[ob_j], close=closes[ob_j],
                        bos_index=i, score=score, reasons=reasons,
                        broken_level=float(highs[lsh])
                    ))
                else:
                    dropped.append(("bull", ob_j, f"score<{min_score}"))
            else:
                dropped.append(("bull", i, "no bearish OB candle within 30 bars"))

        # BOS down against last swing low
        lsl = last_swing_before(i, swing_low_idx)
        if lsl is not None and closes[i] < lows[lsl] and is_displacement:
            j = i - 1; ob_j = None
            while j > 0 and j > i - 30:
                if closes[j] > opens[j]:
                    ob_j = j; break
                j -= 1
            if ob_j is not None:
                if use_wicks:
                    ob_low  = min(opens[ob_j], closes[ob_j])
                    ob_high = highs[ob_j]
                else:
                    ob_low  = min(opens[ob_j], closes[ob_j])
                    ob_high = max(opens[ob_j], closes[ob_j])

                score, reasons = 0.0, []
                score += 1.0; reasons.append("Displacement + BOS down confirmed")
                if np.any(bear_fvg[ob_j:min(n, ob_j + 5)]):
                    score += 1.0; reasons.append("Fair Value Gap (bearish) in wake of move")
                if (highs[ob_j] - lows[ob_j]) < 1.2 * atr14.iloc[ob_j]:
                    score += 1.0; reasons.append("Clean origin (compact OB candle)")
                window = df.iloc[max(0, ob_j - 15):ob_j]
                if len(window) > 5:
                    groups = equal_levels(window["high"].values)
                    if len(groups) >= 1 or (window["high"].max() > window["high"].quantile(0.85)):
                        score += 1.0; reasons.append("Buy-side sweep/inducement before OB")
                touched = np.any((lows[ob_j + 1:i] <= ob_high) & (highs[ob_j + 1:i] >= ob_low))
                if not touched:
                    score += 1.0; reasons.append("Unmitigated at creation")

                if score >= min_score:
                    bears.append(OB(
                        side="bearish", index=ob_j, start=min(ob_low, ob_high), end=max(ob_low, ob_high),
                        open=opens[ob_j], high=highs[ob_j], low=lows[ob_j], close=closes[ob_j],
                        bos_index=i, score=score, reasons=reasons,
                        broken_level=float(lows[lsl])
                    ))
                else:
                    dropped.append(("bear", ob_j, f"score<{min_score}"))
            else:
                dropped.append(("bear", i, "no bullish OB candle within 30 bars"))

    # IoU dedupe
    def dedupe_iou(zs: List[OB]) -> List[OB]:
        zs = sorted(zs, key=lambda z: (-z.score, z.index))
        kept: List[OB] = []
        for z in zs:
            if any(interval_iou((z.start, z.end), (k.start, k.end)) > iou_thresh for k in kept):
                continue
            kept.append(z)
        return kept

    bulls = dedupe_iou(bulls)
    bears = dedupe_iou(bears)

    dbg["steps"].append({"bulls_final": len(bulls), "bears_final": len(bears), "dropped": dropped[:50]})
    return (bulls, bears, "", dbg if return_debug else None)

def evaluate_mitigation(df: pd.DataFrame, zones: List[OB], side: str) -> List[OB]:
    """Update zones with touches/fill/consumed after BOS → end of data and add consumed_index & reason."""
    if not zones:
        return zones
    highs = df["high"].values
    lows  = df["low"].values
    closes = df["close"].values

    for z in zones:
        lo, hi = z.start, z.end
        k0 = max(0, z.bos_index + 1)
        touches, first, last = 0, None, None
        deepest_low, highest_high = np.inf, -np.inf
        consumed_at = None

        for k in range(k0, len(df)):
            # overlap touch?
            if (lows[k] <= hi) and (highs[k] >= lo):
                touches += 1
                if first is None: first = k
                last = k
                deepest_low = min(deepest_low, lows[k])
                highest_high = max(highest_high, highs[k])

            # consumption (close beyond far side)
            if side == "bullish" and closes[k] < lo and consumed_at is None:
                consumed_at = k
            if side == "bearish" and closes[k] > hi and consumed_at is None:
                consumed_at = k

        z.touches = touches
        z.first_touch_index = first
        z.last_touch_index  = last
        z.mitigated = touches > 0
        if consumed_at is not None:
            z.consumed = True
            z.consumed_index = consumed_at
            z.reasons.append(f"Consumed at bar #{consumed_at}")

        # approx fill ratio
        if touches > 0:
            rng = max(hi - lo, 1e-12)
            if side == "bullish":
                deepest = min(deepest_low, hi)
                z.fill_ratio = float(np.clip((hi - deepest) / rng, 0.0, 1.0))
            else:
                highest = max(highest_high, lo)
                z.fill_ratio = float(np.clip((highest - lo) / rng, 0.0, 1.0))
        else:
            z.fill_ratio = 0.0

    return zones

def is_actionable_zone(
    z: OB,
    df: pd.DataFrame,
    *,
    only_fresh: bool,
    max_fill: float,
    max_bars_since_bos: int,
    max_entry_dist_atr: float
) -> bool:
    # never trade consumed
    if getattr(z, "consumed", False):
        return False

    # optionally require untouched since BOS
    if only_fresh and getattr(z, "mitigated", False):
        return False

    # too much fill already
    if getattr(z, "fill_ratio", 0.0) > float(max_fill):
        return False

    # too old since BOS
    age = (len(df) - 1) - int(z.bos_index)
    if age > int(max_bars_since_bos):
        return False

    # too far from current price (entry at mid of OB) in ATR units
    entry = float((z.start + z.end) / 2.0)
    last = float(df["close"].iloc[-1])
    a = float(atr(df, 14).iloc[-1])
    if np.isfinite(a) and a > 0:
        dist_atr = abs(entry - last) / a
        if dist_atr > float(max_entry_dist_atr):
            return False

    return True


# ---------- Extra context + rescoring ----------
def enrich_ob_context(df: pd.DataFrame,
                      bulls: List[OB],
                      bears: List[OB],
                      profile: Dict[str,float],
                      bull_fvg: np.ndarray,
                      bear_fvg: np.ndarray) -> Tuple[List[OB], List[OB], Dict[str,int]]:
    VAL, VAH, POC = profile["VAL"], profile["VAH"], profile["POC"]
    cvd = cvd_series(df)
    div = detect_cvd_divergence(df["close"], cvd, lookback=120)  # overall TF signal

    def acceptance_ratio(z: OB, after: int=30) -> float:
        lo, hi = z.start, z.end
        i0 = max(0, z.index); i1 = min(len(df)-1, z.bos_index+after)
        sliceL = df.iloc[i0:i1+1]
        if len(sliceL)==0: return 0.0
        overlaps = ((sliceL["low"] <= hi) & (sliceL["high"] >= lo)).sum()
        return overlaps / float(len(sliceL))

    def stacked_fvg_count(z: OB, side: str) -> int:
        i0 = max(0, z.index); i1 = min(len(df)-1, z.bos_index)
        mask = bull_fvg if side=="bullish" else bear_fvg
        seg = mask[i0:i1+1]
        # count clusters of consecutive True
        cnt = 0; run = 0
        for b in seg:
            if b: run += 1
            else:
                if run >= 1: cnt += 1
                run = 0
        if run >= 1: cnt += 1
        return cnt

    def apply(z: OB) -> OB:
        # POC/VA confluence
        if np.isfinite(POC) and z.start <= POC <= z.end:
            z.score += 1.0; z.reasons.append("POC confluence at OB")
        elif np.isfinite(VAH) and abs(z.end - VAH) <= (z.end - z.start)*0.15:
            z.score += 0.5; z.reasons.append("Near VAH boundary")
        elif np.isfinite(VAL) and abs(z.start - VAL) <= (z.end - z.start)*0.15:
            z.score += 0.5; z.reasons.append("Near VAL boundary")
        # acceptance vs rejection at OB
        ar = acceptance_ratio(z)
        if ar <= 0.15:
            z.score += 0.5; z.reasons.append("Rejection at OB (low time-at-price)")
        elif ar >= 0.35:
            z.score -= 0.3; z.reasons.append("Acceptance at OB (high time-at-price)")
        # FVG stacks in wake of BOS
        stacks = stacked_fvg_count(z, z.side)
        if stacks >= 2:
            z.score += 0.5; z.reasons.append(f"Stacked FVGs ({stacks}) in wake of BOS")
        # CVD divergence (TF-wide)
        if div == "bullish" and z.side=="bullish":
            z.score += 0.5; z.reasons.append("CVD divergence (bullish exhaustion) supports long")
        if div == "bearish" and z.side=="bearish":
            z.score += 0.5; z.reasons.append("CVD divergence (bearish exhaustion) supports short")
        return z

    bulls2 = [apply(replace(z)) for z in bulls]
    bears2 = [apply(replace(z)) for z in bears]

    counts = {
        "bull_fvg_stacks_total": sum(stacked_fvg_count(z,"bullish") for z in bulls),
        "bear_fvg_stacks_total": sum(stacked_fvg_count(z,"bearish") for z in bears),
    }
    return bulls2, bears2, counts

# ---------- PDF builders ----------
def build_tables_pdf(pair: str, results: Dict[str, Dict[str, List[TradePlan]]], ordered_tfs: List[str]) -> bytes:
    styles = getSampleStyleSheet()
    title = ParagraphStyle(name="Title", parent=styles["Title"], alignment=1, spaceAfter=8)
    section = ParagraphStyle(name="Section", parent=styles["Heading2"], textColor=colors.darkblue, spaceBefore=10, spaceAfter=6)
    sm = ParagraphStyle(name="Small", parent=styles["Normal"], fontSize=9.5, leading=13)

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, rightMargin=18, leftMargin=18, topMargin=18, bottomMargin=18,
                            title=f"{pair} — MTF OB Trade Plan")

    def make_table(rows):
        t = Table(rows, colWidths=[55, 70, 70, 70, 45, 70, 45])
        t.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,0), colors.whitesmoke),
            ("BOX", (0,0), (-1,-1), 0.6, colors.gray),
            ("INNERGRID", (0,0), (-1,-1), 0.3, colors.gray),
            ("VALIGN", (0,0), (-1,-1), "TOP"),
            ("FONTSIZE", (0,0), (-1,-1), 9),
        ]))
        return t

    story = []
    story.append(Paragraph(f"{pair} — Multi-Timeframe Order-Block Trade Plan (Tables + Bulleted Reasons)", title))
    story.append(Paragraph("Entries=mid-OB; SL beyond far side + buffer. TP1/TP2 prefer opposing OB mids; fallback to 2R/3R when none. R/R = |TP−Entry| / |Entry−SL|.", sm))

    


    for tf in ordered_tfs:
        story.append(Paragraph(tf, section))
        rows = [["Side", "Entry", "Stop", "TP1", "R/R1", "TP2", "R/R2"]]
        if results[tf]["longs"]:
            for p in results[tf]["longs"]:
                rows.append(["Long", f"{p.entry:.6f}", f"{p.sl:.6f}", f"{p.tp1:.6f}",
                             f"{p.rr1 if p.rr1 is not None else '—'}",
                             f"{p.tp2:.6f}", f"{p.rr2 if p.rr2 is not None else '—'}"])
        else:
            rows.append(["Long", "—", "—", "—", "—", "—", "—"])
        if results[tf]["shorts"]:
            for p in results[tf]["shorts"]:
                rows.append(["Short", f"{p.entry:.6f}", f"{p.sl:.6f}", f"{p.tp1:.6f}",
                             f"{p.rr1 if p.rr1 is not None else '—'}",
                             f"{p.tp2:.6f}", f"{p.rr2 if p.rr2 is not None else '—'}"])
        else:
            rows.append(["Short", "—", "—", "—", "—", "—", "—"])
        story.append(make_table(rows))
        story.append(Spacer(1, 4))

        # Bulleted reasons + invalidations
        if results[tf]["longs"]:
            for i, p in enumerate(results[tf]["longs"], 1):
                story.append(Paragraph(f"• Long — Why (#{i}): " + "; ".join(p.reasons), styles["Normal"]))
                for inv in p.invalidations:
                    story.append(Paragraph(f"  • Long — Invalidation: {inv}", styles["Normal"]))
        else:
            story.append(Paragraph("• Long — No high-quality bullish OB found.", styles["Normal"]))
        if results[tf]["shorts"]:
            for i, p in enumerate(results[tf]["shorts"], 1):
                story.append(Paragraph(f"• Short — Why (#{i}): " + "; ".join(p.reasons), styles["Normal"]))
                for inv in p.invalidations:
                    story.append(Paragraph(f"  • Short — Invalidation: {inv}", styles["Normal"]))
        else:
            story.append(Paragraph("• Short — No high-quality bearish OB found.", styles["Normal"]))
        story.append(Spacer(1, 8))

    doc.build(story)
    buf.seek(0)
    return buf.read()

def build_charts_pdf(pair: str, charts: Dict[str, bytes], ordered_tfs: List[str]) -> bytes:
    styles = getSampleStyleSheet()
    title = ParagraphStyle(name="Title", parent=styles["Title"], alignment=1, spaceAfter=8)
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, rightMargin=18, leftMargin=18, topMargin=18, bottomMargin=18,
                            title=f"{pair} — Annotated Charts")
    story = [Paragraph(f"{pair} — Annotated Charts (OB zone, BOS, FVG, sweep, profile)", title)]
    for tf in ordered_tfs:
        if tf in charts and charts[tf]:
            story.append(Paragraph(tf, styles["Heading2"]))
            img_stream = io.BytesIO(charts[tf]); img_stream.seek(0)
            story.append(RLImage(img_stream, width=540, height=540 * 0.56))
            story.append(Spacer(1, 12))
    doc.build(story)
    buf.seek(0)
    return buf.read()

# ---------- Annotated chart rendering ----------
def annotated_chart_png(df: pd.DataFrame,
                        bulls: List[OB],
                        bears: List[OB],
                        title: str,
                        label_style: str = "Numeric tags",
                        focus_mode: str = "Auto-zoom: latest OB",
                        bars_before: int = 120,
                        bars_after: int = 60,
                        show_fvg: bool = True,
                        show_sweeps: bool = True,
                        dark_charts: bool = False,
                        width_px: int = CANVAS_W_PX,
                        height_px: int = CANVAS_H_PX,
                        dpi: int = CANVAS_DPI) -> bytes:

    df, bulls, bears = crop_for_display(df, bulls, bears, focus_mode, bars_before, bars_after)
    dfp = df.copy().set_index("time")[["open","high","low","close"]].rename(
        columns={"open":"Open","high":"High","low":"Low","close":"Close"}
    )

    xdates = mdates.date2num(dfp.index.to_pydatetime())
    dx = np.median(np.diff(xdates)) if len(xdates) > 1 else 1.0
    def x_at(i): 
        return float(xdates[int(i)])


    # choose a proper mplfinance style (and consistent colors) for dark vs light
    if dark_charts:
        mc = mpf.make_marketcolors(up="#3CCB7F", down="#FF6B6B", edge="inherit", wick="inherit", volume="inherit")
        mpf_style = mpf.make_mpf_style(
            base_mpf_style="nightclouds",  # a dark base
            marketcolors=mc,
            facecolor="#0E1117",           # Streamlit dark bg
            edgecolor="#0E1117",
            gridcolor="#2A2A2A",
            rc={
                "axes.labelcolor": "#EAECEE",
                "xtick.color": "#EAECEE",
                "ytick.color": "#EAECEE",
            },
        )
    else:
        mc = mpf.make_marketcolors(up="#00A86B", down="#D62728", edge="inherit", wick="inherit", volume="inherit")
        mpf_style = mpf.make_mpf_style(base_mpf_style="yahoo", marketcolors=mc)

    fig, axlist = mpf.plot(
        dfp,
        type="candle",
        style=mpf_style,
        volume=False,
        returnfig=True,
        figsize=(width_px / dpi, height_px / dpi)   # ✅ control geometry here
    )
    ax = axlist[0]
    # DO NOT force white here; the mpf style already set the face colors.


    
    highs, lows = df["high"].values, df["low"].values
    atr14 = atr(df, 14)

    pal = PALETTE_DEFAULT

    
    want_verbose = (label_style == "Verbose")
    want_numeric = (label_style == "Numeric tags")

    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    handles = []

    # Show only sides that are actually present
    if bulls:
        handles.append(Patch(facecolor=pal["bull_face"], edgecolor=pal["bull_edge"], label="Bullish OB"))
    if bears:
        handles.append(Patch(facecolor=pal["bear_face"], edgecolor=pal["bear_edge"], label="Bearish OB"))

    # Always show BOS (you always draw it for each zone)
    handles.append(Line2D([0],[0], linestyle="--", color=pal["bos"], lw=1.8, label="BOS"))

    # Only include FVG if overlays are enabled
    if show_fvg:
        handles.append(Patch(facecolor=pal["fvg_bull_face"], edgecolor=pal["fvg_bull_edge"], label="FVG (near OB)"))

    # Only include sweep items if overlays are enabled
    if show_sweeps:
        # Both sell- and buy-side sweeps can appear across zones
        handles.append(Line2D([0],[0], linestyle=":", color=pal["sweep_sell"], lw=1.6, label="Sell-side sweep"))
        handles.append(Line2D([0],[0], linestyle=":", color=pal["sweep_buy"],  lw=1.6, label="Buy-side sweep"))

    leg = ax.legend(handles=handles, loc="upper left", framealpha=0.97, fontsize=9, ncol=2)
    if dark_charts:
        leg.get_frame().set_facecolor("#0E1117")
        leg.get_frame().set_edgecolor("#2A2A2A")

    text_y_offsets = {}
    label_fc = "#0E1117" if dark_charts else "white"  # match theme
    label_ec = "#D0D4DA" if dark_charts else None     # subtle outline in dark

    def tag(x, y, t, edge):
        ax.text(
            x, y, t, fontsize=9, va="bottom", ha="left", zorder=30,
            bbox=dict(boxstyle="round,pad=0.2",
                    fc=label_fc, ec=(label_ec or edge), alpha=0.98, lw=1.2)
        )

    def vlabel(x, y, t, edge, above=True):
        b = int(x // 1)
        off = text_y_offsets.get(b, 0); dy = 12 + off
        text_y_offsets[b] = off + 12
        ax.text(
            x, y + (dy if above else -dy), t, fontsize=9,
            va="bottom" if above else "top", ha="left", zorder=30,
            bbox=dict(boxstyle="round,pad=0.2",
                    fc=label_fc, ec=(label_ec or edge), alpha=0.98, lw=1.2)
        )

    def min_height(lo, hi, mult=0.20):
        pad = float(atr14.iloc[-1]) * mult if not np.isnan(atr14.iloc[-1]) else 0.0
        h = hi - lo
        if h < pad and pad > 0:
            mid = (lo + hi) / 2
            lo, hi = mid - pad/2.0, mid + pad/2.0
        return lo, hi

    def draw_fvg(z: OB):
        if not show_fvg: return
        n = len(dfp)
        for k in range(z.index, min(z.index + 6, n - 1)):
            # bullish FVG
            if k - 1 >= 0 and k + 1 < n and (lows[k + 1] > highs[k - 1]):
                y0, y1 = highs[k - 1], lows[k + 1]
                x0 = x_at(k - 1) - 0.1*dx
                width = (x_at(k + 1) - x_at(k - 1)) + 0.2*dx
                rect = Rectangle((x0, y0), width, (y1 - y0),
                                facecolor=pal["fvg_bull_face"],
                                edgecolor=pal["fvg_bull_edge"], linewidth=1.6, zorder=28)
                ax.add_patch(rect)
                if want_verbose: vlabel(x0, y1, "FVG", pal["fvg_bull_edge"], above=True)
                return
            # bearish FVG
            if k - 1 >= 0 and k + 1 < n and (highs[k + 1] < lows[k - 1]):
                y0, y1 = highs[k + 1], lows[k - 1]
                x0 = x_at(k - 1) - 0.1*dx
                width = (x_at(k + 1) - x_at(k - 1)) + 0.2*dx
                rect = Rectangle((x0, y0), width, (y1 - y0),
                                facecolor=pal["fvg_bear_face"],
                                edgecolor=pal["fvg_bear_edge"], linewidth=1.6, zorder=28)
                ax.add_patch(rect)
                if want_verbose: vlabel(x0, y1, "FVG", pal["fvg_bear_edge"], above=True)
                return

    def draw_sweep(z: OB):
        if not show_sweeps: return
        win = df.iloc[max(0, z.index - 15):z.index]
        if len(win) == 0: return
        if z.side == "bullish":
            li = win["low"].idxmin(); y = df.loc[li, "low"]
            ax.hlines(y, x_at(max(0, z.index-15)), x_at(z.index),
                      colors=pal["sweep_sell"], linestyles=":", lw=1.6, zorder=29)
            if want_verbose: vlabel(x_at(max(0, z.index-15)), y, "sweep (sell-side)", pal["sweep_sell"], above=False)
        else:
            hi = win["high"].idxmax(); y = df.loc[hi, "high"]
            ax.hlines(y, x_at(max(0, z.index-15)), x_at(z.index),
                      colors=pal["sweep_buy"], linestyles=":", lw=1.6, zorder=29)
            if want_verbose: vlabel(x_at(max(0, z.index-15)), y, "sweep (buy-side)", pal["sweep_buy"], above=True)

    ymax = float(df["high"].max()); bos_y = ymax * 1.001

    def draw_group(zs: List[OB], side: str):
        base_edge = pal["bull_edge"] if side == "bullish" else pal["bear_edge"]
        base_face = pal["bull_face"] if side == "bullish" else pal["bear_face"]
        tagp = "B" if side == "bullish" else "S"
        right_end = x_at(len(dfp) - 1) + 0.50*dx

        for idx, z in enumerate(zs, start=1):
            edge = base_edge; face = base_face
            x0 = x_at(max(0, z.index)) - 0.32*dx
            x1_core = x_at(min(len(dfp) - 1, z.bos_index)) + 0.32*dx
            x1_proj = right_end
            lo, hi = min_height(z.start, z.end, mult=0.20)

            is_mitigated = getattr(z, "mitigated", False)
            is_consumed  = getattr(z, "consumed", False)
            local_hatch = None
            if is_consumed:
                edge = (0.35, 0.35, 0.35, 1.0)
                face = (0.65, 0.65, 0.65, 0.40)
                local_hatch = "xx"
            elif is_mitigated:
                local_hatch = "////" if side == "bullish" else "\\\\"

            rect_core = Rectangle((x0, lo), (x1_core - x0), (hi - lo),
                                  facecolor=face, edgecolor=edge, linewidth=2.2, zorder=27)
            ax.add_patch(rect_core)

            rect_proj = Rectangle((x1_core, lo), (x1_proj - x1_core), (hi - lo),
                                  facecolor=(face[0], face[1], face[2], 0.18),
                                  edgecolor=edge, linewidth=1.2, hatch=local_hatch, zorder=26)
            ax.add_patch(rect_proj)

            # BOS vertical
            ax.axvline(x_at(z.bos_index), linestyle="--", color=pal["bos"], lw=1.8, zorder=31)
            if want_verbose:
                ax.text(x_at(z.bos_index), bos_y, "BOS", fontsize=9, rotation=90,
                        va="bottom", ha="center",
                        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec=pal["bos"],
                                  alpha=0.98, lw=1.2), zorder=32)

            # BOS anchor horizontal @ broken_level
            if z.broken_level is not None:
                ax.hlines(z.broken_level, x_at(0) - 0.50, x1_proj,
                          colors=pal["bos_anchor"], linestyles="--", lw=1.0, zorder=25)

            # FVG & sweeps
            draw_fvg(z)
            draw_sweep(z)

            # Labels
            if label_style == "Minimal (recommended)":
                pass
            elif want_numeric:
                tag(x0, hi, f"{tagp}{idx}", edge)
            elif want_verbose:
                vlabel(x0, hi, f"{side.title()} OB #{idx}", edge, above=True)

    draw_group(bulls, "bullish")
    draw_group(bears, "bearish")

    ax.set_title(title)
    ax.set_xlabel("Time"); ax.set_ylabel("Price")
    fig.subplots_adjust(left=0.065, right=0.995, top=0.92, bottom=0.12)

    out = io.BytesIO()
    fig.savefig(out, format="png", dpi=dpi, bbox_inches=None, pad_inches=0)
    plt.close(fig)
    out.seek(0)
    return out.getvalue()

def inject_accessibility_styles(*, enabled: bool, font_scale: float = 1.0,
                                high_contrast: bool = False, large_targets: bool = True):
    if not enabled:
        return
    base_fg = "#EAECEE" if high_contrast else "inherit"
    base_bg = "#0E1117" if high_contrast else "transparent"
    border = "#FFFFFF" if high_contrast else "#CCCCCC"

    st.markdown(f"""
    <style>
      /* Skip link for screen readers / keyboard users */
      .skip-link {{
        position: absolute; left: -1000px; top: -1000px;
        background: #000; color: #FFF; padding: 8px 12px; border-radius: 8px;
      }}
      .skip-link:focus {{ left: 12px; top: 12px; z-index: 9999; }}

      /* Global font scaling */
      html, body, .stApp, [class^="css"] {{
        font-size: {font_scale}rem !important;
      }}

      /* High-contrast backgrounds and text */
      .stApp {{
        background-color: {base_bg};
        color: {base_fg};
      }}

      /* Visible focus ring everywhere */
      *:focus {{
        outline: 3px solid #FFB300 !important;
        outline-offset: 2px !important;
      }}

      /* Larger buttons & sliders (when requested) */
      {''
        if not large_targets else
        """
        .stButton>button, .stDownloadButton>button, .stLinkButton>a {
            padding: 0.75rem 1.0rem !important;
            font-size: 1.0rem !important;
            border-radius: 10px !important;
            border: 2px solid """ + border + """ !important;
        }
        [data-baseweb="slider"]>div { height: 28px !important; }
        [role="slider"] { width: 28px !important; height: 28px !important; }
        .stSelectbox div[role="combobox"] { min-height: 46px !important; }
        """
      }

      /* Tables: larger row height for readability */
      .stDataFrame tbody tr td {{
        padding-top: 10px !important; padding-bottom: 10px !important;
      }}

      /* Expander headers & alerts: stronger contrast */
      .stExpander, .stAlert {{
        color: {base_fg} !important;
      }}
    </style>
    """, unsafe_allow_html=True)



def sr_zone_summary(z: OB, tf_name: str) -> str:
    mid = (float(z.start) + float(z.end)) / 2.0
    return (
        f"- Side: {z.side}. "
        f"Entry (mid-OB): {mid:.6f}. "
        f"Bounds: [{float(z.start):.6f}, {float(z.end):.6f}]. "
        f"BOS at bar index {int(z.bos_index)}. "
        f"Score: {float(z.score):.1f}. "
        f"{'Mitigated' if z.mitigated else 'Fresh'}; "
        f"{'Consumed' if z.consumed else 'Active'}. "
        f"Notes: " + "; ".join(z.reasons[:4])
    )


def plain_chart_png(df, title, focus_mode="Auto-zoom: latest OB", bars_before=120, bars_after=60, *, dark_charts=False):
    dfc, _, _ = crop_for_display(df, [], [], focus_mode, bars_before, bars_after)

    dfp = dfc.copy().set_index("time")[["open", "high", "low", "close"]]
    dfp = dfp.rename(columns={"open":"Open","high":"High","low":"Low","close":"Close"})
    if dark_charts:
        mc = mpf.make_marketcolors(up="#3CCB7F", down="#FF6B6B", edge="inherit", wick="inherit", volume="inherit")
        mpf_style = mpf.make_mpf_style(
            base_mpf_style="nightclouds",
            marketcolors=mc,
            facecolor="#0E1117", edgecolor="#0E1117", gridcolor="#2A2A2A",
            rc={"axes.labelcolor":"#EAECEE","xtick.color":"#EAECEE","ytick.color":"#EAECEE"},
        )
    else:
        mc = mpf.make_marketcolors(up="#00A86B", down="#D62728", edge="inherit", wick="inherit", volume="inherit")
        mpf_style = mpf.make_mpf_style(base_mpf_style="yahoo", marketcolors=mc)

    fig, axlist = mpf.plot(
        dfp, type="candle", style=mpf_style,
        volume=False, returnfig=True, figratio=(16,9), figscale=1.3
    )
    ax = axlist[0]
    # no forced white backgrounds

    ax.set_title(title); ax.set_xlabel("Time"); ax.set_ylabel("Price")
    fig.set_size_inches(CANVAS_W_PX / CANVAS_DPI, CANVAS_H_PX / CANVAS_DPI)
    fig.subplots_adjust(left=0.065, right=0.995, top=0.92, bottom=0.12)

    out = io.BytesIO()
    fig.savefig(out, format="png", dpi=CANVAS_DPI, bbox_inches=None, pad_inches=0)
    plt.close(fig)
    out.seek(0)
    return out.getvalue()

def annotated_chart_for_zone_png(
    df: pd.DataFrame,
    zone: OB,
    title: str,
    label_style: str = "Numeric tags",
    focus_mode: str = "Auto-zoom: latest OB",
    bars_before: int = 120,
    bars_after: int = 60,
    show_fvg: bool = True,
    show_sweeps: bool = True,
    dark_charts: bool = False
) -> bytes:
    # Route the single zone to the existing multi-zone renderer
    bulls = [zone] if zone.side == "bullish" else []
    bears = [zone] if zone.side == "bearish" else []
    return annotated_chart_png(
        df, bulls, bears, title,
        label_style=label_style, focus_mode=focus_mode,
        bars_before=bars_before, bars_after=bars_after,
        show_fvg=show_fvg, show_sweeps=show_sweeps,
        dark_charts=dark_charts,
        width_px=CANVAS_W_PX,
        height_px=IDEA_CANVAS_H_PX,    # ✅ taller per-idea charts (no squish)
        dpi=CANVAS_DPI
    )







st.title("📄 Order-Block Trade Plan (CSV-driven)")
st.caption("Tables + Bulleted Reasons + PDFs, annotated charts with OB/BOS/FVG/sweeps + Value Area/POC + CVD proxy/true.")

# --- MODE SWITCH (Basic / Advanced) ---
mode = st.radio(
    "Mode",
    ["Basic", "Advanced"],
    index=0,
    horizontal=True,
    help="Basic = core OB+BOS workflow from the first 2 videos. Advanced = adds FVG/sweeps, TPO/POC, CVD divergence, adaptive displacement, and full controls."
)
is_basic = (mode == "Basic")

with st.expander("How to export CSVs + formats"):
    st.markdown("""
- You may upload **one, some, or many** CSVs. Timeframes supported: **Weekly, Daily, 4H, 1H, 30m, 15m**.
- CSV needs **time, open, high, low, close** (volume optional).  
- If available, add **ask_volume, bid_volume** (or synonyms) to enable **true CVD**.
- Recommended: ~300+ candles per TF.
- Bulk upload works: use the **Multi-CSV** uploader; names like `BTCUSD_4H.csv`, `eth-15m.csv`, `weekly.csv` help auto-detect TF.
""")

pair         = st.text_input("Pair symbol (used in titles):", value=cfg["pair"], help=HELP["pair"])
stop_buffer  = st.number_input("Stop buffer (absolute, beyond OB extreme)", value=0.02, step=0.01, format="%.6f", help=HELP["stop_buffer"])

# --- Stops / buffer ---
with st.expander("Stops", expanded=False):
    stop_buffer_mode = st.radio(
        "Stop buffer mode",
        ["Absolute", "% of entry", "ATR14 × multiple"],
        index=2,
        help="How to place the stop beyond the OB extreme."
    )
    if stop_buffer_mode == "Absolute":
        stop_buffer_abs = st.number_input(
            "Absolute buffer (price units)",
            value=float(st.session_state.ui_settings.get("stop_buffer", 0.02)),
            step=0.01, format="%.6f",
            help="Fixed price amount beyond OB extreme (current behavior)."
        )
        stop_buffer_pct = 0.0
        stop_buffer_atr = 0.0
    elif stop_buffer_mode == "% of entry":
        stop_buffer_pct = st.number_input(
            "Percent of entry (%)",
            value=0.25, step=0.05, format="%.2f",
            help="E.g., 0.25 means 0.25% beyond the OB extreme."
        )
        stop_buffer_abs = 0.0
        stop_buffer_atr = 0.0
    else:
        # Per-TF ATR multiples editor (used only when mode = ATR14 × multiple)
        with st.expander("ATR14 stop buffer per timeframe", expanded=False):
            st.caption("These override the single ‘ATR14 multiple’ value per timeframe.")
            c1, c2, c3 = st.columns(3)
            with c1:
                ATR14_MULTS["Weekly"] = st.number_input("Weekly", value=float(ATR14_MULTS["Weekly"]), step=0.05, format="%.2f")
                ATR14_MULTS["4H"]     = st.number_input("4H",     value=float(ATR14_MULTS["4H"]),     step=0.05, format="%.2f")
                ATR14_MULTS["30m"]    = st.number_input("30m",    value=float(ATR14_MULTS["30m"]),    step=0.05, format="%.2f")
            with c2:
                ATR14_MULTS["Daily"]  = st.number_input("Daily",  value=float(ATR14_MULTS["Daily"]),  step=0.05, format="%.2f")
                ATR14_MULTS["1H"]     = st.number_input("1H",     value=float(ATR14_MULTS["1H"]),     step=0.05, format="%.2f")
                ATR14_MULTS["15m"]    = st.number_input("15m",    value=float(ATR14_MULTS["15m"]),    step=0.05, format="%.2f")
            with c3:
                stop_buffer_atr = st.number_input(
                    "Default ATR14 multiple (fallback)",
                    value=float(st.session_state["stop_buffer_atr_fallback"]),
                    step=0.05, format="%.2f"
                )
                st.session_state["stop_buffer_atr_fallback"] = float(stop_buffer_atr)

                if st.button("Reset ATR multiples"):
                    st.session_state["atr14_mults"] = DEFAULT_ATR14_MULTS.copy()
                    st.rerun()

        stop_buffer_abs = 0.0
        stop_buffer_pct = 0.0

# --- Bybit OHLCV loader (USDT Perps) ---
import requests
from datetime import datetime, timezone

# Map your UI/strategy TF names to Bybit intervals
_BYBIT_INTERVAL = {
    "15m": "15", "30m": "30",
    "1H": "60", "4H": "240",
    "Daily": "D", "Weekly": "W",
    # Optional extras if you use them elsewhere:
    "1h": "60", "4h": "240", "1D": "D", "1W": "W"
}

# Case-insensitive, alias-friendly mapping to Bybit v5 intervals
def _bybit_interval_from(tf: str) -> str:
    tf = (tf or "").strip().lower()
    mapping = {
        # minutes
        "1m": "1", "1min": "1", "1minute": "1",
        "3m": "3", "3min": "3",
        "5m": "5", "5min": "5",
        "15m": "15", "15min": "15",
        "30m": "30", "30min": "30",
        # hours
        "1h": "60", "1hr": "60", "1hour": "60", "60": "60",
        "2h": "120", "2hr": "120", "2hour": "120", "120": "120",
        "4h": "240", "4hr": "240", "4hour": "240", "240": "240",
        "6h": "360", "6hr": "360", "6hour": "360", "360": "360",
        "12h":"720", "12hr":"720", "12hour":"720", "720":"720",
        # days / weeks / months
        "1d": "D", "d": "D", "day": "D", "daily": "D",
        "1w": "W", "w": "W", "week": "W", "weekly": "W",
        "1mo": "M", "1mth": "M", "month": "M", "monthly": "M",  # Bybit supports 'M'
    }
    return mapping.get(tf, tf)  # allow raw "1","3","5","60","D","W","M"



def load_ohlcv(symbol: str,
               timeframe: str,
               *,
               limit: int = 1000,
               category: str = "linear") -> pd.DataFrame:
    """
    Fetch candles from Bybit v5 market/kline for USDT perpetuals.
    Example: load_ohlcv("BTCUSDT", "1H")  -> DataFrame[time, open, high, low, close, volume]
    """
    interval = _bybit_interval_from(timeframe)
    allowed = {"1","3","5","15","30","60","120","240","360","720","D","W","M"}
    if interval not in allowed:
        raise ValueError(f"Unsupported timeframe for Bybit: {timeframe} -> {interval}")

    url = "https://api.bybit.com/v5/market/kline"
    params = {
        "category": category,        # "linear" = USDT perpetuals
        "symbol":   symbol.upper(),  # e.g., "BTCUSDT", "ETHUSDT"
        "interval": interval,
        "limit":    int(limit),
    }
    r = requests.get(url, params=params, timeout=15)
    r.raise_for_status()
    data = r.json()
    if data.get("retCode") != 0:
        raise RuntimeError(f"Bybit error: {data.get('retCode')} {data.get('retMsg')}")

    rows = data["result"].get("list", [])
    if not rows:
        return pd.DataFrame(columns=["time", "open", "high", "low", "close", "volume"])

    # Bybit returns newest-first; reverse to oldest-first
    rows = list(reversed(rows))

    # Each row = [startTime(ms), open, high, low, close, volume, turnover]
    ts = pd.to_datetime([int(r[0]) for r in rows], unit="ms", utc=True)
    # make tz-naive UTC to match the rest of your app and avoid tz-localize errors
    time_utc_naive = ts.tz_convert("UTC").tz_localize(None)

    df = pd.DataFrame({
        "time":   time_utc_naive,
        "open":   pd.to_numeric([r[1] for r in rows], errors="coerce"),
        "high":   pd.to_numeric([r[2] for r in rows], errors="coerce"),
        "low":    pd.to_numeric([r[3] for r in rows], errors="coerce"),
        "close":  pd.to_numeric([r[4] for r in rows], errors="coerce"),
        "volume": pd.to_numeric([r[5] for r in rows], errors="coerce"),
    })
    df = df.dropna(subset=["time","open","high","low","close"]).reset_index(drop=True)
    return df

# ---------- UPLOADS (Bulk is default) ----------
st.subheader("Live data (Bybit)")

pair = st.text_input("Symbol (Bybit)", value=cfg["pair"])  # e.g., BTCUSDT / ETHUSDT / SOLUSDT
selected_tfs = st.multiselect("Timeframes", SUPPORTED_TFS, default=SUPPORTED_TFS)
bars_per_tf = st.number_input("Bars per timeframe (limit)", value=1000, min_value=50, max_value=1000, step=50)
category = st.selectbox("Bybit category", ["linear", "spot"], index=0)

generate_clicked = st.button("Fetch & Generate")


# Single source of truth: {TF: bytes}
stored_uploads: Dict[str, bytes] = {}

def calc_stop_buffer(df: pd.DataFrame,
                     entry: float,
                     side: str,
                     mode: str,
                     abs_buf: float,
                     pct_buf: float,
                     atr_mult: float) -> float:
    """
    Returns a price delta to add/subtract from the OB extreme.
    mode ∈ {"Absolute", "% of entry", "ATR14 × multiple"}
    """
    if mode == "Absolute":
        return float(abs_buf)

    if mode in ("% of entry", "Percent"):
        return float(entry) * (float(pct_buf) / 100.0)


    # ATR14 × multiple
    a = float(atr(df, 14).iloc[-1]) if len(df) else float("nan")
    if np.isfinite(a) and a > 0 and float(atr_mult) > 0:
        return float(atr_mult) * a

    # Graceful fallbacks if ATR not available or multiplier <= 0
    if abs_buf and abs_buf > 0:
        return float(abs_buf)
    if pct_buf and pct_buf > 0:
        return float(entry) * (float(pct_buf) / 100.0)
    return float(entry) * 0.002  # ~0.20% as last resort

# Helpers
def _bytes(uf):
    if not uf: 
        return None
    try:
        return uf.getvalue()
    except Exception:
        try:
            return uf.read()
        except Exception:
            return None

def _key_safe(name: str, i: int) -> str:
    return "map_" + re.sub(r"[^a-zA-Z0-9_]", "_", name) + f"_{i}"

# Uses your existing infer_tf_from_name(name: str) -> Optional[str]
# If you removed it earlier, re-add the version you prefer.

# Tabs (Bulk is first so it’s the default)
tab_bulk, tab_single = st.tabs(["Bulk upload", "Single TFs"])

with tab_bulk:
    st.caption(
        "Drag in one or more CSVs. TF is auto-detected from names like "
        "`BTCUSD_4H.csv`, `eth-15m.csv`, `weekly.csv`. "
        "You can manually map any files we can’t detect."
    )

    bulk_files = st.file_uploader(
        "Drop one or more CSVs",
        type=["csv"], accept_multiple_files=True, key="csv_bulk", help=HELP["bulk_files"]
    )

    unmatched_files = []
    if bulk_files:
        for i, uf in enumerate(bulk_files):
            tf = infer_tf_from_name(uf.name)  # your helper
            b = _bytes(uf)
            if not b:
                continue
            if tf:
                # Last one wins per TF
                stored_uploads[tf] = b
            else:
                unmatched_files.append((i, uf))

        if unmatched_files:
            st.info("Timeframe not detected for these files — please choose:")
            for i, uf in unmatched_files:
                sel = st.selectbox(
                    f"{uf.name}",
                    SUPPORTED_TFS,
                    index=None,
                    placeholder="Select timeframe…",
                    key=_key_safe(uf.name, i),
                )
                if sel:
                    b = _bytes(uf)
                    if b:
                        stored_uploads[sel] = b





# =========================
# --------- UI ------------
# =========================

# ---- Theme & palette ----
dark_charts = st.checkbox("Dark charts theme", value=cfg["dark_charts"], help=HELP["dark_charts"])

# --- Accessibility ---
with st.expander("Accessibility", expanded=False):
    a11y_mode   = st.checkbox("Accessibility mode (add screen-reader descriptions under charts)", value=False)
    font_scale = st.slider("UI font scale", 1.0, 1.8, 1.25, 0.05,
                           help="Scales most UI text and table fonts.")
    high_contrast_ui = st.checkbox("High-contrast UI", value=True,
                                   help="Boosts contrast of text, borders, and controls.")
    high_contrast_charts = st.checkbox("High-contrast chart styling", value=True,
                                       help="Thicker lines, stronger colors, and larger labels on charts.")
    texture_overlays = st.checkbox("Use textures on OB zones", value=True,
                                   help="Adds hatches so bullish/bearish areas are distinguishable without color.")
    verbose_labels_always = st.checkbox("Always use verbose labels on charts", value=False,
                                        help="Forces descriptive chart labels instead of minimal/numeric tags.")
    larger_touch_targets = st.checkbox("Larger buttons & sliders", value=True,
                                       help="Increases hit-area for controls.")

# Apply accessibility CSS now that the toggles exist
st.markdown('<a class="skip-link" href="#main-content">Skip to main content</a>', unsafe_allow_html=True)

inject_accessibility_styles(
    enabled=a11y_mode,
    font_scale=font_scale,
    high_contrast=high_contrast_ui,      # <-- correct kw name
    large_targets=larger_touch_targets,
)

st.markdown('<div id="main-content"></div>', unsafe_allow_html=True)


# ---- Overlays ----
if is_basic:
    # Keep charts clean in Basic mode
    show_fvg = False
    show_sweeps = False
    st.caption("Overlays are simplified in Basic mode (no FVG/sweep drawings). Switch to Advanced to enable them.")
else:
    show_fvg    = st.checkbox("Show FVG overlays", value=True, help=HELP["show_fvg"])
    show_sweeps = st.checkbox("Show sweep lines", value=True, help=HELP["show_sweeps"])

# ---- Advanced detection ----
if is_basic:
    # Safe, simple defaults for Basic
    pivot_left        = 3
    pivot_right       = 3
    fvg_span          = 3
    iou_thresh        = 0.50
    adaptive_disp     = False
    displacement_mult = 1.0
    st.caption("Detection uses simple defaults in Basic mode. Switch to Advanced to tune pivots, IoU, and adaptive displacement.")
else:
    with st.expander("Advanced detection", expanded=False):
        pivot_left        = st.slider("Pivot left", 2, 6, 3, help=HELP["pivot_left"])
        pivot_right       = st.slider("Pivot right", 2, 6, 3, help=HELP["pivot_right"])
        fvg_span          = st.selectbox("FVG window", [3, 5], index=0, help=HELP["fvg_span"])
        iou_thresh        = st.slider("Dedupe IoU threshold", 0.30, 0.90, 0.50, 0.05, help=HELP["iou_thresh"])
        adaptive_disp     = st.checkbox("Adaptive displacement (rolling 60-bar 0.60-quantile of range/ATR as floor)", value=True, help=HELP["adaptive_disp"])
        displacement_mult = st.slider("Adaptive floor multiplier", 0.6, 1.8, 1.0, 0.1, help=HELP["displacement_mult"])

# ---- Target policy ----
if is_basic:
    target_policy = "Opposing OB mids"
    st.caption("Basic mode uses opposing OB mids for targets. Switch to Advanced for 2R/3R or Structure.")
else:
    target_policy = st.selectbox(
        "Target policy",
        ["Opposing OB mids", "2R/3R", "Structure (swing highs/lows)"],
        index=0,
        help=HELP["target_policy"]
    )

# ---- Settings save/load ----
with st.expander("Save / Load settings", expanded=False):
    import json

    # Build from the single source of truth
    settings = {
        "pair": pair,
        "stop_buffer": float(stop_buffer),
        "preset": preset,
        "min_score": float(min_score),
        "displacement_k": float(displacement_k),
        "adaptive_disp": bool(adaptive_disp),
        "displacement_mult": float(displacement_mult),
        "pivot_left": int(pivot_left),
        "pivot_right": int(pivot_right),
        "fvg_span": int(fvg_span),
        "iou_thresh": float(iou_thresh),
        "use_wicks": bool(use_wicks),
        "label_style": label_style,
        "focus_mode": focus_mode,
        "bars_before": int(bars_before),
        "bars_after": int(bars_after),
        "show_fvg": bool(show_fvg),
        "show_sweeps": bool(show_sweeps),
        "dark_charts": bool(dark_charts),
        "target_policy": target_policy,
    }


    # Save (download as JSON)
    st.download_button(
        "⬇️ Save settings (JSON)",
        data=json.dumps(settings, indent=2),
        file_name="mtf_ob_settings.json",
        mime="application/json",
    )

    # Load (merge -> session -> refresh)
    cfg_file = st.file_uploader("Load settings JSON", type=["json"])
    if cfg_file is not None:
        try:
            loaded = json.load(cfg_file)
            # merge with precedence to loaded values
            st.session_state.ui_settings = {**DEFAULTS, **st.session_state.ui_settings, **loaded}
            st.success("Settings loaded.")
            st.rerun()
        except Exception as e:
            st.error(f"Failed to load settings: {e}")

#


with st.expander("Detection parameters", expanded=False):
    preset = st.selectbox(
        "Sensitivity preset",
        ["Balanced (recommended)", "Conservative", "Aggressive", "Custom"],
        index=0 if is_basic else 0,
        help=HELP["preset"],
    )
    if preset == "Balanced (recommended)":
        min_score = 3.0
        displacement_k = 1.5
        st.caption(f"Using recommended: min_score={min_score:.1f}, displacement_k={displacement_k:.1f}")
    elif preset == "Conservative":
        min_score = 3.5
        displacement_k = 1.8
        st.caption(f"Conservative: min_score={min_score:.1f}, displacement_k={displacement_k:.1f}")
    elif preset == "Aggressive":
        min_score = 2.5
        displacement_k = 1.2
        # Keep adaptive bits OFF in Basic; ON in Advanced
        if not is_basic:
            adaptive_disp=True; displacement_mult=0.8; pivot_left=2; pivot_right=2; iou_thresh=0.85; use_wicks=False
        st.caption(f"Aggressive: min_score={min_score:.1f}, displacement_k={displacement_k:.1f}")
    else:
        # Custom only makes sense in Advanced
        if is_basic:
            st.info("Custom tuning is available in Advanced mode.")
            min_score = 3.0; displacement_k = 1.5
        else:
            min_score = st.slider("Minimum OB quality score", 2.0, 4.5, 3.0, 0.5, help=HELP["min_score"])
            displacement_k = st.slider("Displacement threshold (k × ATR14)", 1.0, 2.5, 1.5, 0.1, help=HELP["displacement_k"])

if is_basic:
    label_style = st.selectbox("Chart label style", ["Minimal (recommended)", "Numeric tags"], index=0, help=HELP["label_style"])
else:
    label_style = st.selectbox("Chart label style", ["Minimal (recommended)", "Numeric tags", "Verbose"], index=1, help=HELP["label_style"])

focus_mode  = st.selectbox("Chart focus", ["Auto-zoom: latest OB", "Full range"], index=0, help=HELP["focus_mode"])
bars_before = st.slider("Bars before OB", 30, 200, 120, 10, help=HELP["bars_before"])
bars_after  = st.slider("Bars after BOS", 20, 120, 60, 10, help=HELP["bars_after"])
use_wicks   = st.checkbox("Use wicks for OB bounds (recommended)", value=True, help=HELP["use_wicks"])
render_charts = st.checkbox("Render annotated charts (OB/BOS/FVG/sweep/profile)", value=cfg["render_charts"], help=HELP["render_charts"])
include_charts_pdf = st.checkbox("Offer a separate Annotated Charts PDF", value=cfg["include_charts_pdf"], help=HELP["include_charts_pdf"])

with st.expander("Time handling (if your timestamps look off)", expanded=False):
    dayfirst_ui = st.checkbox("Dates are day-first (DD/MM/YYYY)", value=False, key="dayfirst_ui", help=HELP["dayfirst_ui"])
    epoch_ui = st.selectbox(
        "If your time column is numeric, interpret as:",
        ["Auto-detect", "seconds", "milliseconds", "microseconds", "nanoseconds", "Excel serial days"],
        index=0, key="epoch_unit_ui", help=HELP["epoch_unit_ui"]
    )
    tz_naive_ui = st.checkbox("Make timestamps timezone-naive in UTC (recommended for charts)", value=True, key="tz_naive_ui", help=HELP["tz_naive_ui"])

    # Map UI string → load_csv token
    epoch_map = {
        "Auto-detect": None,
        "seconds": "s",
        "milliseconds": "ms",
        "microseconds": "us",
        "nanoseconds": "ns",
        "Excel serial days": "excel",
    }
    epoch_kw = epoch_map[epoch_ui]

with st.expander("Idea freshness filters", expanded=True):
    only_fresh = st.checkbox("Only show *fresh* OBs (untouched since BOS)", value=False)
    max_fill = st.slider("Max zone fill allowed (0=fresh, 1=fully filled)", 0.0, 1.0, 0.85, 0.01)
    max_bars_since_bos = st.slider("Max bars since BOS (age filter)", 10, 800, 360, 10)
    max_entry_dist_atr = st.slider("Max entry distance from last price (ATR14 multiples)", 0.1, 10.0, 3.5, 0.1)

apply_actionable_filters = st.checkbox("Apply freshness filters", value=not is_basic)

generate_clicked = st.button("Generate", help=HELP["generate"])

# === COMPUTE on Generate (save to session) ===
# === COMPUTE on Generate (save to session) ===
if generate_clicked:
    try:
        dfs = {
            tf: load_csv(b, dayfirst=dayfirst_ui, epoch_unit=epoch_kw, tz_naive=tz_naive_ui)
            for tf, b in stored_uploads.items()
        }
    except Exception as e:
        st.error(f"CSV parse error: {e}")
        st.stop()

    if not dfs:
        st.error("Please upload at least one CSV (Weekly, Daily, 4H, 1H, 30m, or 15m).")
        st.stop()

    ordered_tfs = [tf for tf in ["Weekly", "Daily", "4H", "1H", "30m", "15m"] if tf in dfs]
    st.caption("Detected timeframes: " + ", ".join(ordered_tfs))
    if stop_buffer_mode == "ATR14 × multiple":
        stop_buffer_atr = float(st.session_state.get("stop_buffer_atr_fallback", 1.00))
        used = ", ".join(f"{tf}={ATR14_MULTS.get(tf, stop_buffer_atr):.2f}" for tf in ordered_tfs)
        st.caption(f"ATR14×multiple in use → {used}")


    detected_obs = {}
    debug_info = {}
    charts_png = {}
    results = {}

    for tf_name in ordered_tfs:
        df = dfs[tf_name]
        bulls, bears, warn, dbg = detect_ob_zones(
            df,
            use_wicks=use_wicks,
            pivot_left=pivot_left, pivot_right=pivot_right,
            displacement_k=displacement_k,
            adaptive_disp=adaptive_disp,
            adaptive_q=0.60,
            adaptive_window=60,
            displacement_mult=displacement_mult,
            min_score=min_score,
            fvg_span=fvg_span,
            iou_thresh=iou_thresh,
            return_debug=True
        )
        bulls = evaluate_mitigation(df, bulls, "bullish")
        bears = evaluate_mitigation(df, bears, "bearish")

        # Advanced-only: rescore with profile/FVG/CVD confluence
        if not is_basic:
            profile = tpo_profile(df)  # VAL/VAH/POC
            bull_mask, bear_mask = fvg_mask(df, span=fvg_span)
            bulls, bears, _ = enrich_ob_context(df, bulls, bears, profile, bull_mask, bear_mask)

        # after evaluate_mitigation(...)
        pre_bulls, pre_bears = len(bulls), len(bears)

        bulls = [z for z in bulls if is_actionable_zone(
            z, df,
            only_fresh=only_fresh,
            max_fill=max_fill,
            max_bars_since_bos=max_bars_since_bos,
            max_entry_dist_atr=max_entry_dist_atr
        )]
        bears = [z for z in bears if is_actionable_zone(
            z, df,
            only_fresh=only_fresh,
            max_fill=max_fill,
            max_bars_since_bos=max_bars_since_bos,
            max_entry_dist_atr=max_entry_dist_atr
        )]

        # (optional) quick visibility on what got filtered
        if pre_bulls != len(bulls) or pre_bears != len(bears):
            st.caption(
                f"{tf_name}: kept {len(bulls)}/{pre_bulls} bullish and {len(bears)}/{pre_bears} bearish after filters."
            )

        if warn:
            st.warning(f"{tf_name}: {warn}")

        detected_obs[tf_name] = (bulls, bears)
        debug_info[tf_name] = dbg or {}

    # 3) MTF confluence bump (mutates in place)
    apply_mtf_confluence(detected_obs, ordered_tfs)

    # 4) Build trade plans (with target policy)
    for tf_name in ordered_tfs:
        df = dfs[tf_name]
        bulls, bears = detected_obs[tf_name]
        tf_bias = bias_label(df, ma=50)
        # pivots for structure targets
        piv_hi, piv_lo = pivots(df, left=pivot_left, right=pivot_right)
        # pick a per-TF ATR14 multiple (fallback to the single UI value)
        atr_mult_tf = ATR14_MULTS.get(tf_name, stop_buffer_atr)

        long_plans = compute_tradeplan_for_side(
            "bullish", bulls, bears, stop_buffer, tf_name, tf_bias,
            target_policy=target_policy, df=df, piv_hi=piv_hi, piv_lo=piv_lo,
            stop_buffer_mode=stop_buffer_mode,
            stop_buffer_abs=stop_buffer_abs,
            stop_buffer_pct=stop_buffer_pct,
            stop_buffer_atr=atr_mult_tf,
        )
        short_plans = compute_tradeplan_for_side(
            "bearish", bears, bulls, stop_buffer, tf_name, tf_bias,
            target_policy=target_policy, df=df, piv_hi=piv_hi, piv_lo=piv_lo,
            stop_buffer_mode=stop_buffer_mode,
            stop_buffer_abs=stop_buffer_abs,
            stop_buffer_pct=stop_buffer_pct,
            stop_buffer_atr=atr_mult_tf,
        )
        results[tf_name] = {"longs": long_plans, "shorts": short_plans}

    # Safety: make sure every TF in ordered_tfs has a results bucket
    for tf in ordered_tfs:
        results.setdefault(tf, {"longs": [], "shorts": []})

    # Banner if nothing at all
    if all((len(b) == 0 and len(s) == 0) for (b, s) in detected_obs.values()):
        st.info("No order blocks found with the current settings. Try **Aggressive** or lower min_score / displacement thresholds slightly.")

    st.session_state["_dfs"] = dfs
    st.session_state["_ordered_tfs"] = ordered_tfs
    st.session_state["_detected_obs"] = detected_obs
    st.session_state["_debug_info"] = debug_info
    st.session_state["_results"] = results
    st.session_state["_charts_png"] = charts_png
    st.session_state["_has_report"] = True


# === RENDER EXISTING REPORT (works after any rerun/backtest) ===
if st.session_state.get("_has_report"):
    dfs          = st.session_state.get("_dfs", {})
    ordered_tfs  = st.session_state.get("_ordered_tfs", [])
    detected_obs = st.session_state.get("_detected_obs", {})
    debug_info   = st.session_state.get("_debug_info", {})
    results      = st.session_state.get("_results", {})
    charts_png   = {}  # rebuild during render

    if not dfs or not ordered_tfs:
        st.info("No generated data in session. Please upload CSVs and click Generate.")
    else:
        # 5) Per-TF rendering
        export_rows = []  # for CSV export

        for tf in ordered_tfs:
            st.subheader(tf)

            # mini sanity panel
            with st.expander("Recent candles preview (last 10)", expanded=False):
                st.dataframe(dfs[tf].tail(10).reset_index(drop=True))

            df = dfs[tf]
            bulls, bears = detected_obs.get(tf, ([], []))

            if len(bulls) == 0 and len(bears) == 0:
                st.info(
                    f"{tf}: no OBs detected under current settings. "
                    "Try **Aggressive** or lower thresholds a bit."
                )

            # Charts
            if render_charts:
                clean_png = plain_chart_png(
                    df, f"{pair} — {tf} (Clean)",
                    focus_mode=focus_mode, bars_before=bars_before, bars_after=bars_after,
                    dark_charts=dark_charts
                )
                ann_png = annotated_chart_png(
                    df, bulls, bears, f"{pair} — {tf}",
                    label_style=label_style, focus_mode=focus_mode,
                    bars_before=bars_before, bars_after=bars_after,
                    show_fvg=show_fvg, show_sweeps=show_sweeps,
                    dark_charts=dark_charts
                )

                c1, c2 = st.columns([1, 1], gap="large")
                with c1:
                    st.image(clean_png, use_container_width=True, caption="Clean chart (focus window)")
                    st.download_button(f"⬇️ Download {tf} clean chart (PNG)",
                                       data=clean_png, file_name=f"{pair}_{tf}_clean.png")
                with c2:
                    st.image(ann_png, use_container_width=True, caption="Annotated chart (OB/BOS/FVG/Sweeps)")
                    st.download_button(f"⬇️ Download {tf} annotated chart (PNG)",
                                       data=ann_png, file_name=f"{pair}_{tf}_annotated.png")

                # Screen-reader description if enabled
                if a11y_mode and (bulls or bears):
                    z0 = bulls[0] if bulls else bears[0]
                    st.markdown(
                        f"<details><summary><b>Screen-reader description</b></summary>"
                        f"<div><p>{sr_zone_summary(z0, tf)}</p></div></details>",
                        unsafe_allow_html=True
                    )

                charts_png[tf] = ann_png

            plans_tf = results.get(tf, {"longs": [], "shorts": []})
            longs_tf = plans_tf.get("longs", [])
            shorts_tf = plans_tf.get("shorts", [])

            # === Per-idea LONG charts ===
            if longs_tf:
                st.markdown("**Per-idea LONG charts**")
                for i, _ in enumerate(longs_tf, 1):
                    if i <= len(bulls):
                        z_long = bulls[i - 1]
                        img_long = annotated_chart_for_zone_png(
                            df, z_long, title=f"{pair} — {tf} — Long idea #{i}",
                            label_style=label_style, focus_mode=focus_mode,
                            bars_before=bars_before, bars_after=bars_after,
                            show_fvg=show_fvg, show_sweeps=show_sweeps,
                            dark_charts=dark_charts
                        )
                        st.image(img_long, use_container_width=True, caption=f"Long idea #{i}")
                        st.download_button(
                            f"⬇️ Download {tf} long idea #{i} (PNG)",
                            data=img_long, file_name=f"{pair}_{tf}_long_{i}.png"
                        )
                        if a11y_mode:
                            st.markdown(
                                f"<details><summary><b>Screen-reader description</b></summary>"
                                f"<div><p>{sr_zone_summary(z_long, tf)}</p></div></details>",
                                unsafe_allow_html=True
                            )

            # === Per-idea SHORT charts ===
            if shorts_tf:
                st.markdown("**Per-idea SHORT charts**")
                for i, _ in enumerate(shorts_tf, 1):
                    if i <= len(bears):
                        z_short = bears[i - 1]
                        img_short = annotated_chart_for_zone_png(
                            df, z_short, title=f"{pair} — {tf} — Short idea #{i}",
                            label_style=label_style, focus_mode=focus_mode,
                            bars_before=bars_before, bars_after=bars_after,
                            show_fvg=show_fvg, show_sweeps=show_sweeps,
                            dark_charts=dark_charts
                        )
                        st.image(img_short, use_container_width=True, caption=f"Short idea #{i}")
                        st.download_button(
                            f"⬇️ Download {tf} short idea #{i} (PNG)",
                            data=img_short, file_name=f"{pair}_{tf}_short_{i}.png"
                        )

                        if a11y_mode:
                            st.markdown(
                                f"<details><summary><b>Screen-reader description</b></summary>"
                                f"<div><p>{sr_zone_summary(z_short, tf)}</p></div></details>",
                                unsafe_allow_html=True
                            )

            # Tables
            def rows(side):
                plans = longs_tf if side == "longs" else shorts_tf
                if not plans:
                    return pd.DataFrame([{
                        "Side": "Long" if side == "longs" else "Short",
                        "Entry": "—", "Stop": "—", "TP1": "—", "R/R1": "—", "TP2": "—", "R/R2": "—"
                    }])
                recs = []
                for p in plans:
                    recs.append({
                        "Side": "Long" if side == "longs" else "Short",
                        "Entry": p.entry, "Stop": p.sl,
                        "TP1": p.tp1, "R/R1": p.rr1,
                        "TP2": p.tp2, "R/R2": p.rr2
                    })
                    export_rows.append({
                        "TF": tf, "Side": "Long" if side == "longs" else "Short",
                        "Entry": p.entry, "SL": p.sl,
                        "TP1": p.tp1, "TP2": p.tp2, "RR1": p.rr1, "RR2": p.rr2,
                        "Reasons": "; ".join(p.reasons),
                        "Invalidations": "; ".join(p.invalidations)
                    })
                return pd.DataFrame(recs)

            st.dataframe(pd.concat([rows("longs"), rows("shorts")], ignore_index=True))

            # Bulleted reasons
            def reason_md(rs: List[str], side: str, idx: int) -> str:
                items = []
                for r in rs:
                    items.append(f'<li><span title="{r}">{r}</span></li>')
                return f'<ul>{"".join(items)}</ul>'

            if longs_tf:
                for i, p in enumerate(longs_tf, 1):
                    st.markdown(f"**Long — Why (#{i}):**", unsafe_allow_html=True)
                    st.markdown(reason_md(p.reasons, "Long", i), unsafe_allow_html=True)
                    for inv in p.invalidations:
                        st.markdown(f"- Long — Invalidation: {inv}")
            else:
                st.markdown("- **Long — No high-quality bullish OB found.**")

            if shorts_tf:
                for i, p in enumerate(shorts_tf, 1):
                    st.markdown(f"**Short — Why (#{i}):**", unsafe_allow_html=True)
                    st.markdown(reason_md(p.reasons, "Short", i), unsafe_allow_html=True)
                    for inv in p.invalidations:
                        st.markdown(f"- Short — Invalidation: {inv}")
            else:
                st.markdown("- **Short — No high-quality bearish OB found.**")

            # ---- Backtest this timeframe (OB ideas) ----
            exp_key = f"bt_{tf}_expanded"
            with st.expander(f"Backtest this timeframe ({tf})",
                             expanded=st.session_state.get(exp_key, False)):

                rr_min_bt       = st.number_input(f"{tf} · Min R/R (TP1)", value=1.0, step=0.5, min_value=1.0, key=f"bt_rr_{tf}")
                expiry_bars_bt  = st.number_input(f"{tf} · Idea expiry (bars)", value=200, step=5, min_value=1, key=f"bt_exp_{tf}")
                max_hold_bt     = st.number_input(f"{tf} · Max hold (bars)", value=200, step=10, min_value=10, key=f"bt_hold_{tf}")
                risk_cash_bt    = st.number_input(f"{tf} · Risk per trade ($)", value=1000.0, step=100.0, min_value=0.0, key=f"bt_risk_{tf}")

                sb_mode = stop_buffer_mode
                sb_abs  = stop_buffer_abs
                sb_pct  = stop_buffer_pct
                sb_atr  = ATR14_MULTS.get(tf, stop_buffer_atr) if sb_mode == "ATR14 × multiple" else 0.0

                bt_key = f"_bt_trades_{tf}"

                if st.button(f"Run backtest — {tf}", use_container_width=True, key=f"bt_run_{tf}"):
                    with st.spinner("Running OB backtest…"):
                        df_tf = st.session_state["_dfs"][tf]
                        bulls_tf, bears_tf = st.session_state["_detected_obs"][tf]
                        bt_trades = ob_backtest_tf(
                            df_tf, bulls_tf, bears_tf,
                            rr_min=rr_min_bt,
                            expiry_bars=expiry_bars_bt,
                            max_hold_bars=max_hold_bt,
                            stop_buffer_mode=sb_mode,
                            stop_buffer_abs=sb_abs,
                            stop_buffer_pct=sb_pct,
                            stop_buffer_atr=sb_atr,
                        )
                    st.session_state[exp_key] = True
                    st.session_state[bt_key]  = bt_trades

                # Render existing backtest results if present
                bt_trades = st.session_state.get(bt_key)
                if bt_trades is not None:
                    if bt_trades.empty:
                        st.warning("No trades met the filters / simulation on this timeframe.")
                    else:
                        stats = ob_backtest_stats(bt_trades, risk_cash=risk_cash_bt)
                        st.write({
                            "Trades (closed)": stats["trades"],
                            "Wins (TP2)": stats["wins"],
                            "Losses (SL)": stats["losses"],
                            "Breakeven": stats["breakevens"],
                            "Win rate % (incl. BE)": stats["win_rate_pct"],
                            "Win / (Win + Loss) %": stats["win_over_wl_pct"],
                            "Avg R": stats["avg_R"],
                            "Expectancy (R)": stats["expectancy_R"],
                            "Profit factor": stats["profit_factor"],
                            "Median R": stats["median_R"],
                            "R p25 / p75": f'{stats["p25_R"]} / {stats["p75_R"]}',
                            "Max drawdown (R)": stats["max_drawdown_R"],
                            "Sharpe (per 12 trades)": stats["sharpe_R"],
                            "Gross R": stats["gross_R"],
                            "Gross $": stats["gross_$"],
                            "Avg $/trade": stats["avg_$"],
                            "Expectancy $": stats["exp_$"],
                        })

                        _ob_bt_plot_equity(bt_trades, title=f"{tf} Equity (R) Curve")

                        show_cols = ["created_time","side","entry","sl","tp1","tp2","status","R","bars_held","exit_time","note"]
                        st.markdown("#### Trades")
                        st.dataframe(bt_trades[show_cols].reset_index(drop=True), use_container_width=True, key=f"bt_df_{tf}")

                        st.download_button(
                            f"⬇️ Download {tf} backtest trades (CSV)",
                            data=bt_trades.to_csv(index=False).encode("utf-8"),
                            file_name=f"{pair}_{tf}_OB_backtest_trades.csv",
                            mime="text/csv",
                            key=f"bt_dl_{tf}"
                        )

            # Debug panel per TF
            with st.expander("🛠 Debug (why candidates were dropped & counts)", expanded=False):
                st.json(debug_info.get(tf, {}))

            st.divider()

        # ===== 6) Exports =====
        tables_pdf = build_tables_pdf(pair, results, ordered_tfs)
        st.download_button("⬇️ Download Tables + Bulleted Reasons PDF",
                           data=tables_pdf, file_name=f"{pair}_OB_TradePlan.pdf", mime="application/pdf")

        if include_charts_pdf and charts_png:
            charts_pdf = build_charts_pdf(pair, charts_png, ordered_tfs)
            st.download_button("⬇️ Download Annotated Charts PDF",
                               data=charts_pdf, file_name=f"{pair}_Annotated_Charts.pdf", mime="application/pdf")

        if export_rows:
            out_df = pd.DataFrame(export_rows)
            st.download_button("⬇️ Export all trade plans (CSV)",
                               data=out_df.to_csv(index=False).encode("utf-8"),
                               file_name=f"{pair}_trade_plans.csv", mime="text/csv")

        # persist charts for later PDF availability
        st.session_state["_charts_png"] = charts_png
