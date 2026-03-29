from __future__ import annotations
import numpy as np
import pandas as pd
from src.factors.residuals import build_phase2_outputs

def _panel(n_days: int=320, n_assets: int=8, seed: int=0) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range('2020-01-02', periods=n_days)
    fac = rng.standard_normal((n_days, 2))
    load = rng.standard_normal((n_assets, 2))
    eps = 0.5 * rng.standard_normal((n_days, n_assets))
    r = fac @ load.T + eps
    returns = pd.DataFrame(r, index=idx, columns=[f'A{i}' for i in range(n_assets)])
    mask = pd.DataFrame(True, index=idx, columns=returns.columns)
    return (returns, mask)

def test_estimation_window_excludes_rebalance_day():
    returns, mask = _panel(n_days=400, n_assets=8, seed=1)
    W, K = (60, 3)
    cum1, betas1, _ = build_phase2_outputs(returns, mask, W=W, K=K, rebal_freq='BMS', cond_warn_threshold=1000000000000000.0)
    rebal = pd.Timestamp('2020-06-01')
    if rebal not in returns.index:
        rebal = returns.index[returns.index >= '2020-06-01'][0]
    pos = returns.index.get_loc(rebal)
    returns2 = returns.copy()
    returns2.iloc[pos:, :] *= 1.25
    cum2, betas2, _ = build_phase2_outputs(returns2, mask, W=W, K=K, rebal_freq='BMS', cond_warn_threshold=1000000000000000.0)
    key = rebal.strftime('%Y-%m-%d')
    if key in betas1 and key in betas2:
        pd.testing.assert_frame_equal(betas1[key], betas2[key], rtol=1e-10, atol=1e-10)
    past = returns.index < rebal
    if past.any():
        c1 = cum1.loc[past]
        c2 = cum2.loc[past]
        pd.testing.assert_frame_equal(c1, c2, rtol=1e-10, atol=1e-10)

def test_outputs_align_index_and_columns():
    returns, mask = _panel(n_days=350, n_assets=6, seed=2)
    cum, betas, diag = build_phase2_outputs(returns, mask, W=50, K=2, rebal_freq='BMS', cond_warn_threshold=1000000000000000.0)
    assert cum.shape == returns.shape
    assert cum.index.equals(returns.index)
    assert list(cum.columns) == list(returns.columns)
    if len(diag):
        assert 'condition_XtX' in diag.columns or 'explained_variance_top_k' in diag.columns

def test_pca_explained_variance_bounded():
    returns, mask = _panel(n_days=300, n_assets=10, seed=3)
    _, _, diag = build_phase2_outputs(returns, mask, W=80, K=4, rebal_freq='BMS', cond_warn_threshold=1000000000000000.0)
    if len(diag) and 'explained_variance_top_k' in diag.columns:
        ev = diag['explained_variance_top_k'].astype(float)
        assert ((ev >= 0) & (ev <= 1.01)).all()
