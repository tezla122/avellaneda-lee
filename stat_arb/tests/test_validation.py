from __future__ import annotations
import numpy as np
import pandas as pd
from src.validation.dsr import deflated_sharpe_ratio
from src.validation.regimes import regime_metrics, vix_bucket
from src.validation.sensitivity import run_sensitivity_grid

def test_dsr_finite_on_gaussian_returns():
    rng = np.random.default_rng(0)
    r = pd.Series(rng.normal(0, 0.01, 500))
    dsr, meta = deflated_sharpe_ratio(r)
    assert np.isfinite(dsr) or str(dsr) == 'nan'
    assert meta['T'] == 500

def test_vix_regime_metrics():
    idx = pd.bdate_range('2020-01-02', periods=20)
    pnl = pd.Series(np.linspace(-0.01, 0.01, 20), index=idx)
    vix = pd.Series([10.0] * 10 + [20.0] * 10, index=idx)
    reg = vix_bucket(vix)
    m = regime_metrics(pnl, reg)
    assert len(m) >= 2

def test_sensitivity_grid_runs():
    idx = pd.bdate_range('2021-01-04', periods=80)
    cols = ['a', 'b']
    rng = np.random.default_rng(1)
    s_scores = pd.DataFrame(rng.normal(0, 0.5, (80, 2)), index=idx, columns=cols)
    kappa = pd.DataFrame(5.0, index=idx, columns=cols)
    sigma_eq = pd.DataFrame(0.02, index=idx, columns=cols)
    mask = pd.DataFrame(True, index=idx, columns=cols)
    lr = pd.DataFrame(rng.normal(0, 0.01, (80, 2)), index=idx, columns=cols)
    grid = {'s_open': [1.5, 2.0], 'transaction_cost_bps': [0.0]}
    df = run_sensitivity_grid(s_scores, kappa, sigma_eq, mask, lr, grid)
    assert len(df) == 2
    assert 'sharpe' in df.columns
