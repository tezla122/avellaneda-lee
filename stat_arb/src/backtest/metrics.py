from __future__ import annotations
from typing import Any
import numpy as np
import pandas as pd

def equity_curve_from_net(net: pd.Series, initial_aum: float) -> pd.Series:
    return float(initial_aum) + net.cumsum()

def drawdown(aum: pd.Series) -> pd.Series:
    peak = aum.cummax()
    return aum / peak.replace(0.0, np.nan) - 1.0

def compute_metrics(net_pnl: pd.Series, aum: pd.Series, *, delta_weight_l1: pd.Series | None=None, benchmark: pd.Series | None=None) -> dict[str, Any]:
    net = net_pnl.astype(float)
    mu = net.mean()
    sig = net.std(ddof=1)
    sharpe = float(np.sqrt(252) * mu / sig) if sig and sig > 0 else float('nan')
    neg = net[net < 0]
    downs = neg.std(ddof=1)
    sortino = float(np.sqrt(252) * mu / downs) if downs and downs > 0 else float('nan')
    dd = drawdown(aum)
    max_dd = float(dd.min()) if len(dd) else float('nan')
    years = len(net) / 252.0
    a0 = float(aum.iloc[0]) if len(aum) else float('nan')
    a1 = float(aum.iloc[-1]) if len(aum) else float('nan')
    total_ret = a1 / a0 - 1.0 if a0 and np.isfinite(a0) and (a0 != 0) else float('nan')
    ann_ret = (1 + total_ret) ** (1 / years) - 1 if years > 0 and np.isfinite(total_ret) else float('nan')
    calmar = float(ann_ret / abs(max_dd)) if max_dd and max_dd < 0 and np.isfinite(ann_ret) else float('nan')
    out: dict[str, Any] = {'sharpe': sharpe, 'sortino': sortino, 'max_drawdown': max_dd, 'calmar': calmar, 'hit_rate': float((net > 0).mean()) if len(net) else float('nan'), 'mean_daily_net_pnl': float(mu), 'vol_daily_net_pnl': float(sig)}
    if delta_weight_l1 is not None and len(delta_weight_l1):
        out['avg_daily_turnover_l1'] = float(delta_weight_l1.mean())
        out['annualized_turnover_l1'] = float(delta_weight_l1.mean() * 252.0)
    if benchmark is not None:
        b = benchmark.reindex(net.index).astype(float)
        ex = (net - b).dropna()
        es = ex.std(ddof=1)
        out['information_ratio'] = float(np.sqrt(252) * ex.mean() / es) if es and es > 0 and len(ex) else float('nan')
    return out

def attribution_long_short(positions: pd.DataFrame, simple_returns: pd.DataFrame) -> pd.DataFrame:
    w = positions.shift(1)
    long_w = w.clip(lower=0.0)
    short_w = w.clip(upper=0.0)
    long_pnl = (long_w * simple_returns).sum(axis=1)
    short_pnl = (short_w * simple_returns).sum(axis=1)
    return pd.DataFrame({'long_book_pnl': long_pnl, 'short_book_pnl': short_pnl, 'combined_gross': long_pnl + short_pnl}, index=positions.index)

def rolling_sharpe(net: pd.Series, window: int=63) -> pd.Series:
    x = net.astype(float)
    return np.sqrt(252) * x.rolling(window, min_periods=max(5, window // 3)).mean() / x.rolling(window, min_periods=max(5, window // 3)).std(ddof=1)
