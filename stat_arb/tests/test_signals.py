from __future__ import annotations
import numpy as np
import pandas as pd
from src.signals.phase4 import build_phase4_outputs, raw_signals_state_machine

def test_hysteresis_long():
    idx = pd.bdate_range('2020-01-02', periods=10)
    s = pd.DataFrame({'A': [-2.0, -1.0, -0.5, 0.0, 0.5, 0.0, -2.0, -0.5, 0.0, 2.0]}, index=idx)
    u = pd.DataFrame(True, index=idx, columns=['A'])
    el = pd.DataFrame(True, index=idx, columns=['A'])
    raw = raw_signals_state_machine(s, u, el, s_open=1.25, s_close=0.75)
    assert raw.iloc[0, 0] == 1
    assert raw.iloc[1, 0] == 1
    assert raw.iloc[2, 0] == 0
    assert raw.iloc[4, 0] == 0

def test_universe_kills_position():
    idx = pd.bdate_range('2020-01-02', periods=5)
    s = pd.DataFrame({'A': [-2.0, -2.0, -2.0, -2.0, -2.0]}, index=idx)
    u = pd.DataFrame([True, True, False, True, True], index=idx, columns=['A'])
    el = pd.DataFrame(True, index=idx, columns=['A'])
    raw = raw_signals_state_machine(s, u, el, s_open=1.25, s_close=0.75)
    assert raw.iloc[2, 0] == 0

def test_build_phase4_positions_sum_gross():
    rng = np.random.default_rng(0)
    idx = pd.bdate_range('2021-01-04', periods=50)
    cols = ['a', 'b']
    s_scores = pd.DataFrame(rng.standard_normal((50, 2)) * 0.5, index=idx, columns=cols)
    kappa = pd.DataFrame(5.0, index=idx, columns=cols)
    sigma_eq = pd.DataFrame(0.02, index=idx, columns=cols)
    mask = pd.DataFrame(True, index=idx, columns=cols)
    raw, pos, diag = build_phase4_outputs(s_scores, kappa, sigma_eq, mask, s_open=1.25, s_close=0.75, gross_notional=1.0, max_half_life_days=30.0, sigma_entry_floor=1e-06)
    g = diag['gross_exposure']
    assert (g <= 1.01).all()
    assert (g >= 0).all()
