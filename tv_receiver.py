# tv_receiver.py
from fastapi import FastAPI, Request
from pydantic import BaseModel
from pathlib import Path
import csv, datetime, os

app = FastAPI()

# Where to store CSVs (point your Streamlit “Upload CSVs” to this folder later)
BASE_DIR = Path("./data/from_tradingview").resolve()
BASE_DIR.mkdir(parents=True, exist_ok=True)

def _safe(s: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_", ".", "@") else "_" for ch in str(s))[:80]

def _iso_utc(ms: int) -> str:
    return datetime.datetime.utcfromtimestamp(ms / 1000.0).replace(microsecond=0).isoformat() + "Z"

def write_csv_row(symbol: str, tf: str, row: dict):
    sym_dir = BASE_DIR / _safe(symbol)
    sym_dir.mkdir(parents=True, exist_ok=True)
    fpath = sym_dir / f"{_safe(symbol)}_{_safe(tf)}.csv"
    new = not fpath.exists()
    with fpath.open("a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["time", "open", "high", "low", "close", "volume"])
        w.writerow([
            _iso_utc(int(row["t"])),
            f'{float(row["o"]):.10f}',
            f'{float(row["h"]):.10f}',
            f'{float(row["l"]):.10f}',
            f'{float(row["c"]):.10f}',
            row.get("v", "")
        ])

@app.post("/tv")
async def tv_webhook(req: Request):
    try:
        data = await req.json()
    except Exception:
        # Some brokers post as raw text; try again
        text = await req.body()
        raise ValueError(f"Non-JSON payload: {text[:200]!r}")

    symbol   = data.get("symbol") or data.get("tickerid") or "UNKNOWN"
    events   = data.get("events", [])
    if not isinstance(events, list) or not events:
        return {"ok": True, "note": "no events"}

    for ev in events:
        # Expect keys: tf, t, o, h, l, c, (v)
        if all(k in ev for k in ("tf", "t", "o", "h", "l", "c")):
            write_csv_row(symbol, ev["tf"], ev)

    return {"ok": True, "saved": len(events), "symbol": symbol, "dir": str(BASE_DIR)}
