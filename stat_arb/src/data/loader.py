from __future__ import annotations
from pathlib import Path
from typing import Iterable
import pandas as pd
_REQUIRED = ('adjclose', 'volume')

def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip().lower().replace(' ', '_') for c in out.columns]
    aliases = {'adj_close': 'adjclose', 'adjusted_close': 'adjclose', 'adj_close_price': 'adjclose'}
    out = out.rename(columns=aliases)
    return out

def _parse_date_index(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.index, pd.DatetimeIndex):
        idx = pd.to_datetime(df.index, utc=False)
    else:
        for cand in ('date', 'datetime', 'time', 'timestamp'):
            if cand in df.columns:
                df = df.set_index(cand)
                break
        else:
            raise ValueError('Expected a DateTime index or a column named date/datetime/timestamp.')
        idx = pd.to_datetime(df.index, utc=False)
    if getattr(idx, 'tz', None) is not None:
        idx = idx.tz_convert('America/New_York').normalize()
    else:
        idx = idx.normalize()
    out = df.copy()
    out.index = idx
    out = out.sort_index()
    out = out[~out.index.duplicated(keep='last')]
    return out

def load_ticker_file(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() in ('.parquet', '.pq'):
        df = pd.read_parquet(path)
    elif path.suffix.lower() in ('.csv', '.txt'):
        df = pd.read_csv(path)
    else:
        raise ValueError(f'Unsupported file type: {path}')
    df = _normalize_columns(df)
    df = _parse_date_index(df)
    missing = [c for c in _REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f'{path}: missing required columns {missing}')
    keep = [c for c in df.columns if c in _REQUIRED or c in ('open', 'high', 'low', 'close', 'sector', 'exchange')]
    df = df[keep]
    for c in _REQUIRED:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    return df

def discover_raw_files(raw_dir: str | Path, glob: str='*') -> list[Path]:
    raw_dir = Path(raw_dir)
    if not raw_dir.is_dir():
        return []
    paths: list[Path] = []
    for pat in (f'{glob}.csv', f'{glob}.parquet', f'{glob}.pq'):
        paths.extend(raw_dir.glob(pat))
    out = sorted({p.resolve() for p in paths})
    return out

def ticker_from_filename(path: Path) -> str:
    return path.stem.upper()
