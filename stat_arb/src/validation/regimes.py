from __future__ import annotations
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd

def vix_bucket(vix: pd.Series) -> pd.Series:
    x = vix.astype(float)
    out = pd.Series('unknown', index=x.index, dtype=object)
    m = x.notna()
    out.loc[m & (x < 15)] = 'low'
    out.loc[m & (x >= 15) & (x <= 25)] = 'medium'
    out.loc[m & (x > 25)] = 'high'
    return out

def regime_metrics(daily_pnl: pd.Series, regime: pd.Series, *, annualization: float=252.0) -> pd.DataFrame:
    df = pd.DataFrame({'pnl': daily_pnl.astype(float), 'regime': regime})
    df = df.dropna(subset=['pnl', 'regime'])
    rows: list[dict[str, Any]] = []
    for name in sorted(df['regime'].dropna().unique()):
        sub = df.loc[df['regime'] == name, 'pnl']
        if len(sub) < 2:
            rows.append({'regime': name, 'n_days': len(sub), 'sharpe': np.nan})
            continue
        mu = float(sub.mean())
        sig = float(sub.std(ddof=1))
        sh = float(np.sqrt(annualization) * mu / sig) if sig > 0 else float('nan')
        rows.append({'regime': name, 'n_days': int(len(sub)), 'mean_daily_pnl': mu, 'vol_daily': sig, 'sharpe': sh})
    return pd.DataFrame(rows)

def load_vix_series(path: str | Path) -> pd.Series:
    p = Path(path)
    if p.suffix.lower() in ('.parquet', '.pq'):
        v = pd.read_parquet(p)
    else:
        v = pd.read_csv(p)
    if isinstance(v.index, pd.DatetimeIndex):
        idx = pd.DatetimeIndex(v.index).normalize()
    elif 'date' in v.columns:
        idx = pd.to_datetime(v['date']).dt.normalize()
        v = v.drop(columns=['date'])
    else:
        raise ValueError('VIX file needs DatetimeIndex or a date column.')
    num = v.select_dtypes(include=[np.number])
    if num.shape[1] < 1:
        raise ValueError('No numeric VIX column found.')
    s = num.iloc[:, 0].copy()
    s.index = idx
    s.name = 'vix'
    return s.sort_index()
