from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

def write_sample_raw(raw_dir: Path, *, n_tickers: int=3, n_days: int=400, seed: int=42) -> list[Path]:
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range('2018-01-02', periods=n_days)
    out: list[Path] = []
    for i in range(n_tickers):
        sym = f'DEMO{i}'
        price = 50.0 + np.cumsum(rng.standard_normal(n_days) * 0.25)
        vol = rng.uniform(2000000.0, 8000000.0, n_days)
        df = pd.DataFrame({'date': idx.strftime('%Y-%m-%d'), 'adjclose': price, 'volume': vol})
        path = raw_dir / f'{sym}.csv'
        df.to_csv(path, index=False)
        out.append(path)
    return out
