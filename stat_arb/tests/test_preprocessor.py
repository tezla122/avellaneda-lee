from __future__ import annotations
import numpy as np
import pandas as pd
import pandas_market_calendars as mcal
from src.data.preprocessor import _listed_before_window_mask, _trading_index, build_phase1_outputs

def _nyse_days(start: str, end: str) -> pd.DatetimeIndex:
    cal = mcal.get_calendar('NYSE')
    sched = cal.schedule(start_date=start, end_date=end)
    return pd.DatetimeIndex(sched.index.normalize()).sort_values()

def _make_ticker(index: pd.DatetimeIndex, seed: int, vol_scale: float=10000000.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = len(index)
    price = 100 * np.exp(rng.standard_normal(n).cumsum() * 0.01)
    volume = rng.uniform(1000000.0, 5000000.0, n) * vol_scale / 10000000.0
    return pd.DataFrame({'adjclose': price, 'volume': volume}, index=index)

def test_universe_mask_is_one_day_shift_of_raw_eligibility():
    idx = _nyse_days('2024-01-02', '2024-12-31')[:140]
    tickers = {'AAA': _make_ticker(idx, 1), 'BBB': _make_ticker(idx, 2), 'CCC': _make_ticker(idx, 3)}
    returns, universe_mask, _meta = build_phase1_outputs(tickers, adv_window=63, adv_threshold=1.0, min_constituents_per_day=1, calendar_name='NYSE', winsorize_lower=None, winsorize_upper=None)
    start = min((df.index.min() for df in tickers.values()))
    end = max((df.index.max() for df in tickers.values()))
    cal_idx = _trading_index(start, end, 'NYSE')
    prices = pd.DataFrame({k: v['adjclose'].reindex(cal_idx) for k, v in tickers.items()}, index=cal_idx)
    vol = pd.DataFrame({k: v['volume'].reindex(cal_idx) for k, v in tickers.items()}, index=cal_idx)
    dollar = vol * prices
    adv = dollar.rolling(63, min_periods=63).median()
    liquidity_ok = (adv >= 1.0) & prices.notna()
    listed_ok = _listed_before_window_mask(prices, 63)
    raw = liquidity_ok & listed_ok
    expected = raw.shift(1).fillna(False).astype(bool).loc[returns.index]
    pd.testing.assert_frame_equal(universe_mask, expected)

def test_no_lookahead_returns_used_for_same_day_mask():
    idx = _nyse_days('2024-01-02', '2024-12-31')[:120]
    base = {'AAA': _make_ticker(idx, 10), 'BBB': _make_ticker(idx, 11)}
    r1, m1, _ = build_phase1_outputs(base, adv_threshold=1.0, min_constituents_per_day=1, winsorize_lower=None, winsorize_upper=None)
    alt = {k: v.copy() for k, v in base.items()}
    cut = len(idx) // 2
    for k in alt:
        alt[k] = alt[k].copy()
        alt[k].iloc[cut:, alt[k].columns.get_loc('adjclose')] *= 1.5
    r2, m2, _ = build_phase1_outputs(alt, adv_threshold=1.0, min_constituents_per_day=1, winsorize_lower=None, winsorize_upper=None)
    common = m1.index.intersection(m2.index)
    assert m1.loc[common].iloc[:cut].equals(m2.loc[common].iloc[:cut])

def test_metadata_has_ticker_fields():
    idx = _nyse_days('2024-06-01', '2024-12-31')[:90]
    df = _make_ticker(idx, 0)
    df['sector'] = 'Test'
    df['exchange'] = 'TEST'
    _, _, meta = build_phase1_outputs({'ZZZ': df}, adv_threshold=1.0, min_constituents_per_day=1, winsorize_lower=None, winsorize_upper=None)
    t = meta['tickers']['ZZZ']
    assert t['sector'] == 'Test'
    assert t['exchange'] == 'TEST'
    assert 'yearly_tickers_with_price_data' in meta
