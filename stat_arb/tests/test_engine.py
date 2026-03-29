from __future__ import annotations
import numpy as np
import pandas as pd
from src.backtest.engine import gross_pnl, log_to_simple_returns, run_vectorized_backtest
from src.backtest.metrics import compute_metrics

def test_shift_one_positions_for_pnl():
    idx = pd.bdate_range('2020-01-02', periods=5)
    pos = pd.DataFrame({'A': [1.0, 0.0, 0.0, 0.0, 0.0]}, index=idx)
    log_r = pd.DataFrame(np.nan, index=idx, columns=['A'])
    log_r.iloc[1, 0] = 0.01
    simple = log_to_simple_returns(log_r)
    g = gross_pnl(pos, simple)
    assert np.isclose(g.iloc[1], np.expm1(0.01), rtol=1e-06)

def test_run_vectorized_backtest_metrics():
    idx = pd.bdate_range('2021-03-01', periods=100)
    rng = np.random.default_rng(0)
    cols = ['x', 'y']
    pos = pd.DataFrame(rng.uniform(-0.5, 0.5, (100, 2)), index=idx, columns=cols)
    lr = pd.DataFrame(rng.normal(0, 0.01, (100, 2)), index=idx, columns=cols)
    out = run_vectorized_backtest(pos, lr, initial_aum=1.0, transaction_cost_bps=0.0)
    m = compute_metrics(out['net_pnl'], out['aum'], delta_weight_l1=out['delta_weight_l1'])
    assert 'sharpe' in m
    assert 'max_drawdown' in m
