from __future__ import annotations
import numpy as np
import pandas as pd

def log_to_simple_returns(log_r: pd.DataFrame) -> pd.DataFrame:
    return np.expm1(log_r.astype(float))

def assert_no_missing_returns_with_positions(positions: pd.DataFrame, returns: pd.DataFrame, simple: pd.DataFrame) -> None:
    w = positions.shift(1)
    bad = w.notna() & (w != 0) & simple.isna()
    if bad.any().any():
        raise ValueError(f'Non-zero lagged position with missing simple return (would corrupt P&L). Bad count: {int(bad.sum().sum())}')

def gross_pnl(positions: pd.DataFrame, simple_returns: pd.DataFrame) -> pd.Series:
    return (positions.shift(1) * simple_returns).sum(axis=1)

def transaction_costs(positions: pd.DataFrame, *, cost_bps: float, use_drift_adjusted: bool=True, simple_returns: pd.DataFrame | None=None) -> tuple[pd.Series, pd.Series]:
    if use_drift_adjusted and simple_returns is not None:
        prev = positions.shift(1)
        drift = prev * (1 + simple_returns)
        delta = positions - drift
    else:
        delta = positions.diff()
    delta = delta.fillna(0.0)
    tc = float(cost_bps) * 0.0001 * delta.abs().sum(axis=1)
    return (tc, delta.abs().sum(axis=1))

def short_borrow_cost(positions: pd.DataFrame, *, borrow_bps_annual: float) -> pd.Series:
    if borrow_bps_annual <= 0:
        return pd.Series(0.0, index=positions.index)
    short_exp = positions.clip(upper=0.0).abs().sum(axis=1)
    daily_rate = float(borrow_bps_annual) * 0.0001 / 252.0
    return daily_rate * short_exp

def run_vectorized_backtest(positions: pd.DataFrame, log_returns: pd.DataFrame, *, initial_aum: float=1.0, transaction_cost_bps: float=5.0, short_borrow_bps_annual: float=0.0, drift_adjusted_tc: bool=True) -> dict[str, pd.Series | pd.DataFrame]:
    simple = log_to_simple_returns(log_returns)
    simple = simple.reindex_like(positions)
    positions = positions.reindex_like(simple).fillna(0.0)
    assert_no_missing_returns_with_positions(positions, log_returns, simple)
    gross = gross_pnl(positions, simple)
    tc, delta_sum = transaction_costs(positions, cost_bps=transaction_cost_bps, use_drift_adjusted=drift_adjusted_tc, simple_returns=simple if drift_adjusted_tc else None)
    borrow = short_borrow_cost(positions, borrow_bps_annual=short_borrow_bps_annual)
    net = gross - tc - borrow
    aum = float(initial_aum) + net.cumsum()
    return {'gross_pnl': gross, 'transaction_costs': tc, 'short_borrow_cost': borrow, 'net_pnl': net, 'aum': aum, 'delta_weight_l1': delta_sum}
