# bybit_fetch.py
from __future__ import annotations
import os, time, typing as _t
import pandas as pd, numpy as np, requests

try:
    import streamlit as st
    def _safe_secret(key: str, default: str|None=None):
        try:
            return st.secrets[key]
        except Exception:
            return os.getenv(key, default)
except Exception:
    def _safe_secret(key: str, default: str|None=None):
        return os.getenv(key, default)

BYBIT_BASE = (_safe_secret("BYBIT_BASE", "https://api.bybit.com") or "https://api.bybit.com").rstrip("/")

SUPPORTED_TFS = ["Weekly", "Daily", "4H", "1H", "30m", "15m"]
TF_TO_INTERVAL = {"Weekly":"W","Daily":"D","4H":"240","1H":"60","30m":"30","15m":"15"}

def _raise_for_bybit_error(j: dict):
    if not isinstance(j, dict):
        raise RuntimeError("Unexpected response from Bybit.")
    if j.get("retCode") != 0:
        msg = j.get("retMsg") or "Unknown error from Bybit"
        raise RuntimeError(f"Bybit error: {j.get('retCode')} {msg}")

def fetch_bybit_ohlcv(symbol: str, *, interval: str, limit: int = 1000, category: str = "linear") -> pd.DataFrame:
    url = f"{BYBIT_BASE}/v5/market/kline"
    params = {"category": category, "symbol": symbol, "interval": str(interval), "limit": int(limit)}
    try:
        r = requests.get(url, params=params, timeout=20)
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Network error to Bybit endpoint: {e}")
    if r.status_code == 403:
        raise RuntimeError("403 Forbidden from Bybit endpoint. If you're on Streamlit Cloud, set BYBIT_BASE to your Fly.io proxy URL.")
    if r.status_code >= 400:
        raise RuntimeError(f"HTTP error from Bybit endpoint: {r.status_code} {r.text[:200]}")
    try:
        j = r.json()
    except Exception:
        raise RuntimeError(f"Non-JSON response from Bybit: {r.text[:200]}")
    _raise_for_bybit_error(j)
    result = j.get("result") or {}
    rows = result.get("list") or []
    if not rows:
        return pd.DataFrame(columns=["time","open","high","low","close","volume"])
    rows = list(reversed(rows))
    out = []
    for row in rows:
        try:
            ts_ms = int(row[0]); o = float(row[1]); h = float(row[2]); l = float(row[3]); c = float(row[4])
            v = float(row[5]) if row[5] is not None else np.nan
        except Exception:
            continue
        t = pd.to_datetime(ts_ms, unit="ms", utc=True).tz_convert(None)
        out.append((t, o, h, l, c, v))
    df = pd.DataFrame(out, columns=["time","open","high","low","close","volume"]).dropna().reset_index(drop=True)
    return df

def fetch_all_dfs(symbol: str, tfs: _t.Sequence[str], *, limit: int = 1000, category: str = "linear", sleep_sec: float = 0.0) -> dict[str, pd.DataFrame]:
    dfs: dict[str, pd.DataFrame] = {}
    for tf in tfs:
        if tf not in TF_TO_INTERVAL: 
            continue
        interval = TF_TO_INTERVAL[tf]
        df = fetch_bybit_ohlcv(symbol, interval=interval, limit=limit, category=category)
        if len(df): dfs[tf] = df
        if sleep_sec > 0: time.sleep(sleep_sec)
    return dfs
