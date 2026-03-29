from __future__ import annotations
from typing import Any
import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew

def deflated_sharpe_ratio(daily_returns: pd.Series, *, annualization: float=252.0) -> tuple[float, dict[str, Any]]:
    r = daily_returns.dropna().astype(float)
    T = int(len(r))
    if T < 3:
        return (float('nan'), {'T': T, 'reason': 'too_few_observations'})
    mu = float(r.mean())
    sig = float(r.std(ddof=1))
    if sig == 0 or not np.isfinite(sig):
        return (float('nan'), {'T': T, 'reason': 'zero_volatility'})
    sr_daily = mu / sig
    g3 = float(skew(r, bias=False))
    g4 = float(kurtosis(r, fisher=True, bias=False))
    inner = 1.0 - g3 * sr_daily + g4 / 4.0 * sr_daily ** 2
    if inner <= 0 or not np.isfinite(inner):
        return (float('nan'), {'T': T, 'sr_daily': sr_daily, 'skew': g3, 'kurtosis_excess': g4, 'inner': inner, 'reason': 'non_positive_denominator'})
    dsr = sr_daily * np.sqrt(T) / np.sqrt(inner)
    sr_ann = float(np.sqrt(annualization) * sr_daily)
    return (float(dsr), {'T': T, 'sr_daily': sr_daily, 'sharpe_annualized': sr_ann, 'skew': g3, 'kurtosis_excess': g4, 'deflated_sharpe_ratio': float(dsr)})
