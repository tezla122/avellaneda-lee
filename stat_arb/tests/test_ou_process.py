from __future__ import annotations
import numpy as np
import pandas as pd
from src.models.ou_process import build_phase3_outputs

def _panel(n: int=200, n_assets: int=4, seed: int=0) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range('2022-01-03', periods=n)
    x = pd.DataFrame(rng.standard_normal((n, n_assets)).cumsum(axis=0) * 0.1, index=idx, columns=[f'S{i}' for i in range(n_assets)])
    mask = pd.DataFrame(True, index=idx, columns=x.columns)
    return (x, mask)

def test_outputs_shape_and_mask():
    X, mask = _panel(n=250, n_assets=5, seed=1)
    kappa, m_bar, sigma_eq, s_scores, diag = build_phase3_outputs(X, mask, W_ou=40, sigma_eq_floor=1e-06)
    assert kappa.shape == X.shape
    assert s_scores.shape == X.shape
    assert diag['W_ou'] == 40

def test_inactive_assets_nan_scores():
    X, mask = _panel(n=180, n_assets=3, seed=2)
    mask.loc[:, 'S0'] = False
    _, _, _, s_scores, _ = build_phase3_outputs(X, mask, W_ou=30)
    assert s_scores['S0'].isna().all()

def test_s_score_uses_shifted_m_bar():
    rng = np.random.default_rng(42)
    n, W = (120, 40)
    idx = pd.bdate_range('2023-01-02', periods=n)
    x = pd.Series(rng.standard_normal(n).cumsum() * 0.05, index=idx, name='A')
    X = x.to_frame()
    mask = pd.DataFrame(True, index=idx, columns=['A'])
    _, m_bar, sigma_eq, s_scores, _ = build_phase3_outputs(X, mask, W_ou=W, sigma_eq_floor=1e-08)
    m_eff = m_bar.shift(1)
    se_eff = sigma_eq.shift(1)
    manual = (X['A'] - m_eff['A']) / se_eff['A']
    pd.testing.assert_series_equal(s_scores['A'], manual, rtol=1e-09, atol=1e-09, check_names=False)
