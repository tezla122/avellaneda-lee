from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping
import numpy as np
import pandas as pd
import yaml
from src.data.loader import discover_raw_files, load_ticker_file, ticker_from_filename
try:
    import pandas_market_calendars as mcal
except ImportError:
    mcal = None

@dataclass
class Phase1Config:
    raw_dir: Path
    processed_dir: Path
    returns_file: Path
    universe_mask_file: Path
    metadata_file: Path
    adv_window: int = 63
    adv_threshold: float = 5000000.0
    min_constituents_per_day: int = 100
    calendar_name: str = 'NYSE'
    raw_glob: str = '*'
    winsorize_lower: float | None = 0.001
    winsorize_upper: float | None = 0.999
    universe_documentation: str = "Document your universe source (e.g. point-in-time constituents). Backfilling today's index introduces survivorship bias."
    root: Path | None = None

    def resolve_paths(self) -> None:
        base = self.root or Path.cwd()
        self.raw_dir = (base / self.raw_dir).resolve()
        self.processed_dir = (base / self.processed_dir).resolve()
        self.returns_file = (base / self.returns_file).resolve()
        self.universe_mask_file = (base / self.universe_mask_file).resolve()
        self.metadata_file = (base / self.metadata_file).resolve()

def load_config(path: str | Path, root: Path | None=None) -> Phase1Config:
    path = Path(path)
    with open(path, encoding='utf-8') as f:
        raw = yaml.safe_load(f)
    paths = raw.get('paths', {})
    wins = raw.get('winsorize') or {}
    cfg = Phase1Config(raw_dir=Path(paths.get('raw_dir', 'data/raw')), processed_dir=Path(paths.get('processed_dir', 'data/processed')), returns_file=Path(paths.get('returns_file', 'data/processed/returns.parquet')), universe_mask_file=Path(paths.get('universe_mask_file', 'data/processed/universe_mask.parquet')), metadata_file=Path(paths.get('metadata_file', 'data/processed/metadata.json')), adv_window=int(raw.get('adv_window', 63)), adv_threshold=float(raw.get('adv_threshold', 5000000)), min_constituents_per_day=int(raw.get('min_constituents_per_day', 100)), calendar_name=str(raw.get('calendar_name', 'NYSE')), raw_glob=str(raw.get('raw_glob', '*')), winsorize_lower=wins.get('lower_quantile', 0.001), winsorize_upper=wins.get('upper_quantile', 0.999), universe_documentation=str(raw.get('universe_documentation', Phase1Config.universe_documentation)), root=root)
    cfg.resolve_paths()
    return cfg

def _trading_index(start: pd.Timestamp, end: pd.Timestamp, calendar_name: str) -> pd.DatetimeIndex:
    if mcal is None:
        raise ImportError('pandas_market_calendars is required for calendar alignment.')
    cal = mcal.get_calendar(calendar_name)
    sched = cal.schedule(start_date=start, end_date=end)
    idx = pd.DatetimeIndex(sched.index.normalize())
    return idx.sort_values()

def _first_valid_position(s: pd.Series) -> int | None:
    m = s.notna()
    if not m.any():
        return None
    return int(m.argmax())

def _listed_before_window_mask(prices: pd.DataFrame, window: int) -> pd.DataFrame:
    n = len(prices.index)
    pos = np.arange(n, dtype=np.int64)
    out = pd.DataFrame(False, index=prices.index, columns=prices.columns)
    w = window
    for col in prices.columns:
        fp = _first_valid_position(prices[col])
        if fp is None:
            continue
        start_need = pos - (w - 1)
        out[col] = fp <= start_need
    return out

def _winsorize_returns(returns: pd.DataFrame, lower_q: float | None, upper_q: float | None) -> tuple[pd.DataFrame, dict[str, Any]]:
    if lower_q is None or upper_q is None:
        return (returns, {'enabled': False})
    r = returns.to_numpy(dtype=float)
    flat = r[np.isfinite(r)]
    if flat.size == 0:
        return (returns, {'enabled': True, 'n_clipped_lower': 0, 'n_clipped_upper': 0})
    lo = float(np.quantile(flat, lower_q))
    hi = float(np.quantile(flat, upper_q))
    clipped = (r < lo) | (r > hi)
    n_lo = int((r < lo).sum())
    n_hi = int((r > hi).sum())
    r2 = np.clip(r, lo, hi)
    out = pd.DataFrame(r2, index=returns.index, columns=returns.columns)
    return (out, {'enabled': True, 'lower_bound': lo, 'upper_bound': hi, 'n_clipped_lower': n_lo, 'n_clipped_upper': n_hi, 'clipped_fraction': float(np.mean(clipped))})

def build_phase1_outputs(ticker_frames: Mapping[str, pd.DataFrame], *, adv_window: int=63, adv_threshold: float=5000000.0, min_constituents_per_day: int=100, calendar_name: str='NYSE', winsorize_lower: float | None=0.001, winsorize_upper: float | None=0.999, universe_documentation: str=Phase1Config.universe_documentation) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    if not ticker_frames:
        raise ValueError('No ticker data provided.')
    all_idx: list[pd.DatetimeIndex] = []
    for _sym, df in ticker_frames.items():
        all_idx.append(pd.DatetimeIndex(df.index))
    start = min((idx.min() for idx in all_idx))
    end = max((idx.max() for idx in all_idx))
    cal_idx = _trading_index(start, end, calendar_name)
    prices = pd.DataFrame(index=cal_idx, columns=sorted(ticker_frames.keys()), dtype=float)
    volume = pd.DataFrame(index=cal_idx, columns=prices.columns, dtype=float)
    sector_last: dict[str, str | None] = {c: None for c in prices.columns}
    exch_last: dict[str, str | None] = {c: None for c in prices.columns}
    for sym, df in ticker_frames.items():
        d = df.reindex(cal_idx)
        prices[sym] = d['adjclose']
        volume[sym] = d['volume']
        if 'sector' in df.columns:
            ss = d['sector'].dropna()
            if len(ss):
                sector_last[sym] = str(ss.iloc[-1])
        if 'exchange' in df.columns:
            ee = d['exchange'].dropna()
            if len(ee):
                exch_last[sym] = str(ee.iloc[-1])
    dollar_vol = volume * prices
    adv = dollar_vol.rolling(adv_window, min_periods=adv_window).median()
    liquidity_ok = (adv >= adv_threshold) & prices.notna()
    listed_ok = _listed_before_window_mask(prices, adv_window)
    raw_universe = liquidity_ok & listed_ok
    universe_mask = raw_universe.shift(1).fillna(False).astype(bool)
    returns = np.log(prices).diff()
    returns, win_info = _winsorize_returns(returns, winsorize_lower, winsorize_upper)
    universe_mask = universe_mask.reindex(returns.index).fillna(False)
    counts = universe_mask.sum(axis=1)
    keep_days = counts >= min_constituents_per_day
    n_dropped_low = int((~keep_days).sum())
    if not keep_days.all():
        returns = returns.loc[keep_days]
        universe_mask = universe_mask.loc[keep_days]
    prices_out = prices.reindex(returns.index)
    per_ticker: dict[str, Any] = {}
    for sym in prices_out.columns:
        s = prices_out[sym]
        valid = s.notna()
        if valid.any():
            fi = s[valid].index[0]
            la = s[valid].index[-1]
        else:
            fi = la = None
        per_ticker[sym] = {'first_date': fi.isoformat() if fi is not None else None, 'last_date': la.isoformat() if la is not None else None, 'sector': sector_last.get(sym), 'exchange': exch_last.get(sym)}
    yearly: dict[str, list[str]] = {}
    for year in sorted(set(prices.index.year)):
        sl = prices.loc[prices.index.year == year]
        active = [c for c in prices.columns if sl[c].notna().any()]
        yearly[str(year)] = sorted(active)
    meta: dict[str, Any] = {'tickers': per_ticker, 'yearly_tickers_with_price_data': yearly, 'universe_documentation': universe_documentation, 'data_quality': {'winsorize': win_info, 'adv_window': adv_window, 'adv_threshold': adv_threshold, 'min_constituents_per_day': min_constituents_per_day, 'calendar': calendar_name, 'dropped_dates_low_constituents': n_dropped_low}}
    return (returns, universe_mask, meta)

def run_phase1(cfg: Phase1Config | None=None, *, config_path: Path | None=None) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    if cfg is None:
        if config_path is None:
            raise ValueError('Provide cfg or config_path.')
        cfg = load_config(config_path)
    paths = discover_raw_files(cfg.raw_dir, cfg.raw_glob)
    if not paths:
        raise FileNotFoundError(f'No CSV/Parquet files in {cfg.raw_dir!s} (glob={cfg.raw_glob!r}). Add per-ticker files, or run: python run_phase1.py --demo')
    ticker_frames: dict[str, pd.DataFrame] = {}
    for p in paths:
        sym = ticker_from_filename(p)
        ticker_frames[sym] = load_ticker_file(p)
    returns, universe_mask, meta = build_phase1_outputs(ticker_frames, adv_window=cfg.adv_window, adv_threshold=cfg.adv_threshold, min_constituents_per_day=cfg.min_constituents_per_day, calendar_name=cfg.calendar_name, winsorize_lower=cfg.winsorize_lower, winsorize_upper=cfg.winsorize_upper, universe_documentation=cfg.universe_documentation)
    cfg.processed_dir.mkdir(parents=True, exist_ok=True)
    returns.to_parquet(cfg.returns_file, engine='pyarrow')
    universe_mask.to_parquet(cfg.universe_mask_file, engine='pyarrow')
    with open(cfg.metadata_file, 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2)
    return (returns, universe_mask, meta)
