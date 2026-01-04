import pandas as pd
import numpy as np
from unittest import mock

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import pages.level_mapping_strategy as strat


def _base_params():
    return {
        "atr_period": 1,
        "ema_fast": 2,
        "ema_slow": 3,
        "hold_bars": 1,
        "retest_bars": 1,
        "tol": 0.0001,
        "atr_mult": 1.0,
        "min_stop_bps": 10,
        "min_tp1_bps": 10,
        "min_rr": 0.1,
        "use_val_reclaim": True,
        "use_vah_break": False,
        "use_sp_cont": False,
        "sp_bands": [],
        "entry_style": "Market-on-trigger",
        "tp1_exit_pct": 100,
    }


def _make_df(prices):
    idx = pd.date_range("2023-01-02", periods=len(prices), freq="1H", tz="UTC")
    data = {
        "open": prices,
        "high": prices,
        "low": prices,
        "close": prices,
        "volume": np.ones(len(prices)),
    }
    return pd.DataFrame(data, index=idx)


@mock.patch.object(strat, "composite_profile", autospec=True)
def test_market_entry_uses_next_bar_open(fake_comp):
    fake_comp.return_value = {"VAL": 100.0, "POC": 105.0, "VAH": 110.0, "hist": None}
    df = _make_df([101, 102, 103])
    params = _base_params()
    params.update({"hold_bars": 1, "retest_bars": 1})
    trades, summary, open_pos, pending = strat.backtest_xo_logic(df, {}, params, force_eod_close=False)
    assert open_pos is not None
    assert open_pos["entry_time"] == df.index[2]
    assert open_pos["entry"] == df.iloc[2]["open"]


@mock.patch.object(strat, "composite_profile", autospec=True)
def test_hold_and_retest_respects_window(fake_comp):
    fake_comp.return_value = {"VAL": 100.0, "POC": 105.0, "VAH": 110.0, "hist": None}
    prices = [101, 101, 102, 102]
    df = _make_df(prices)
    params = _base_params()
    params.update({"hold_bars": 2, "retest_bars": 1})
    trades, summary, open_pos, pending = strat.backtest_xo_logic(df, {}, params, force_eod_close=False)
    assert open_pos is not None
    assert open_pos["entry_time"] == df.index[3]


@mock.patch.object(strat, "composite_profile", autospec=True)
def test_pending_timeout_and_cancel(fake_comp):
    fake_comp.return_value = {"VAL": 100.0, "POC": 105.0, "VAH": 110.0, "hist": None}
    df = _make_df([101, 101, 101, 101, 101])
    params = _base_params()
    params.update({"entry_style": "Stop-through", "max_pending_bars": 1, "tol": 0.02})
    trades, summary, open_pos, pending = strat.backtest_xo_logic(df, {}, params, force_eod_close=False)
    assert pending and len(pending) == 1
    assert open_pos is None


@mock.patch.object(strat, "composite_profile", autospec=True)
def test_same_bar_fill_blocked(fake_comp):
    fake_comp.return_value = {"VAL": 100.0, "POC": 105.0, "VAH": 110.0, "hist": None}
    df = _make_df([100, 101, 102])
    params = _base_params()
    params.update({"entry_style": "Stop-through", "allow_same_bar_fills": False})
    trades, summary, open_pos, pending = strat.backtest_xo_logic(df, {}, params, force_eod_close=False)
    # Pending should be armed on bar1 but not filled because fill would be same bar
    assert pending and len(pending) == 1
    assert open_pos is None


def test_monday_monthly_levels_history_only():
    dates = pd.date_range("2023-01-02", periods=5, freq="1D", tz="UTC")
    df = pd.DataFrame({
        "open": [10,11,12,13,14],
        "high": [11,12,13,14,15],
        "low": [9,10,11,12,13],
        "close": [10,11,12,13,14],
        "volume": [1,1,1,1,1],
    }, index=dates)
    monday = strat.monday_range_levels(df)
    assert np.isnan(monday.iloc[0]["monday_high"])
    # After Monday completes, later bars carry Monday range
    assert monday.iloc[2]["monday_high"] == 11
    mo = strat.monthly_open_levels(df)
    assert mo.iloc[0] == df.iloc[0]["open"]
    assert mo.iloc[-1] == df.iloc[0]["open"]


def test_structural_stop_uses_history_only():
    df_hist = pd.DataFrame({
        "open": [100, 99],
        "high": [101, 100],
        "low": [98, 95],
        "close": [100, 99],
        "volume": [1,1],
    }, index=pd.date_range("2023-01-01", periods=2, freq="1H", tz="UTC"))
    sl = strat.structural_stop(df_hist, level=101, side="long", lookback=2, atrv=1.0, atr_mult=1.0, min_stop_abs=0.5)
    assert sl < 99.5
