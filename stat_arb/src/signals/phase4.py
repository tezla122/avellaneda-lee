from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
import yaml

def kappa_min_for_half_life(max_half_life_trading_days: float) -> float:
    return float(np.log(2.0) * 252.0 / max_half_life_trading_days)

def _state_step_update(s_row: np.ndarray, state: np.ndarray, universe_row: np.ndarray, eligible_row: np.ndarray, s_open: float, s_close: float) -> None:
    u = universe_row.astype(bool)
    state[~u] = 0
    row = s_row
    nanm = np.isnan(row)
    state[nanm] = 0
    in_long = (state == 1) & u & ~nanm
    in_short = (state == -1) & u & ~nanm
    exit_long = in_long & (row > -s_close)
    exit_short = in_short & (row < s_close)
    state[exit_long] = 0
    state[exit_short] = 0
    flat = (state == 0) & u & ~np.isnan(row)
    enter_long = flat & eligible_row & (row < -s_open)
    enter_short = flat & eligible_row & (row > s_open)
    state[enter_long] = 1
    state[enter_short] = -1

def raw_signals_state_machine(s_scores: pd.DataFrame, universe_mask: pd.DataFrame, eligible_entry: pd.DataFrame, *, s_open: float, s_close: float) -> pd.DataFrame:
    if not (s_scores.shape == universe_mask.shape == eligible_entry.shape and s_scores.index.equals(universe_mask.index)):
        raise ValueError('s_scores, universe_mask, eligible_entry must align.')
    s = s_scores.to_numpy(dtype=float)
    u = universe_mask.to_numpy(dtype=bool)
    el = eligible_entry.to_numpy(dtype=bool)
    T, N = s.shape
    state = np.zeros(N, dtype=np.int8)
    out = np.zeros((T, N), dtype=np.int8)
    for t in range(T):
        _state_step_update(s[t], state, u[t], el[t], s_open, s_close)
        out[t] = state
    return pd.DataFrame(out, index=s_scores.index, columns=s_scores.columns, dtype=np.int8)

@dataclass
class Phase4Config:
    s_scores_file: Path
    kappa_file: Path
    sigma_eq_file: Path
    universe_mask_file: Path
    processed_dir: Path
    raw_signals_file: Path
    positions_file: Path
    signal_diagnostics_file: Path
    s_open: float = 1.25
    s_close: float = 0.75
    gross_notional: float = 1.0
    max_half_life_days: float = 30.0
    sigma_entry_floor: float = 0.0001
    root: Path | None = None

    def resolve_paths(self) -> None:
        base = self.root or Path.cwd()
        for name in ('s_scores_file', 'kappa_file', 'sigma_eq_file', 'universe_mask_file', 'processed_dir', 'raw_signals_file', 'positions_file', 'signal_diagnostics_file'):
            p = getattr(self, name)
            if isinstance(p, Path) and (not p.is_absolute()):
                setattr(self, name, (base / p).resolve())

def load_phase4_config(path: str | Path, root: Path | None=None) -> Phase4Config:
    path = Path(path)
    with open(path, encoding='utf-8') as f:
        raw = yaml.safe_load(f)
    paths = raw.get('paths') or {}
    p3 = raw.get('phase3') or {}
    p4 = raw.get('phase4') or {}
    cfg = Phase4Config(s_scores_file=Path(p4.get('s_scores_file', p3.get('s_scores_file', 'data/processed/s_scores.parquet'))), kappa_file=Path(p4.get('kappa_file', p3.get('kappa_file', 'data/processed/kappa.parquet'))), sigma_eq_file=Path(p4.get('sigma_eq_file', p3.get('sigma_eq_file', 'data/processed/sigma_eq.parquet'))), universe_mask_file=Path(paths.get('universe_mask_file', 'data/processed/universe_mask.parquet')), processed_dir=Path(paths.get('processed_dir', 'data/processed')), raw_signals_file=Path(p4.get('raw_signals_file', 'data/processed/raw_signals.parquet')), positions_file=Path(p4.get('positions_file', 'data/processed/positions.parquet')), signal_diagnostics_file=Path(p4.get('signal_diagnostics_file', 'data/processed/signal_diagnostics.parquet')), s_open=float(p4.get('s_open', 1.25)), s_close=float(p4.get('s_close', 0.75)), gross_notional=float(p4.get('gross_notional', 1.0)), max_half_life_days=float(p4.get('max_half_life_days', 30.0)), sigma_entry_floor=float(p4.get('sigma_entry_floor', 0.0001)), root=root)
    cfg.resolve_paths()
    return cfg

def build_phase4_outputs(s_scores: pd.DataFrame, kappa: pd.DataFrame, sigma_eq: pd.DataFrame, universe_mask: pd.DataFrame, *, s_open: float=1.25, s_close: float=0.75, gross_notional: float=1.0, max_half_life_days: float=30.0, sigma_entry_floor: float=0.0001) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    idx = s_scores.index
    cols = list(s_scores.columns)
    kappa = kappa.reindex(index=idx, columns=cols)
    sigma_eq = sigma_eq.reindex(index=idx, columns=cols)
    universe_mask = universe_mask.reindex(index=idx, columns=cols).fillna(False)
    kappa_min = kappa_min_for_half_life(max_half_life_days)
    eligible = universe_mask.astype(bool) & (kappa > kappa_min) & (sigma_eq > sigma_entry_floor) & kappa.notna() & sigma_eq.notna()
    raw = raw_signals_state_machine(s_scores, universe_mask, eligible, s_open=s_open, s_close=s_close)
    raw = raw.astype(np.int8)
    raw_masked = raw.where(universe_mask, 0)
    n_sig = (raw_masked != 0).sum(axis=1).replace(0, np.nan)
    scale = gross_notional / n_sig
    positions = raw_masked.astype(float).mul(scale, axis=0).fillna(0.0)
    n_long = (raw_masked > 0).sum(axis=1).astype(np.int64)
    n_short = (raw_masked < 0).sum(axis=1).astype(np.int64)
    gross = positions.abs().sum(axis=1)
    net = positions.sum(axis=1)
    diagnostics = pd.DataFrame({'n_long': n_long, 'n_short': n_short, 'n_signals': (raw_masked != 0).sum(axis=1), 'gross_exposure': gross, 'net_exposure': net}, index=s_scores.index)
    return (raw_masked, positions, diagnostics)

def run_phase4(cfg: Phase4Config | None=None, *, config_path: Path | None=None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if cfg is None:
        if config_path is None:
            raise ValueError('Provide cfg or config_path.')
        cfg = load_phase4_config(config_path)
    for label, p in (('s_scores', cfg.s_scores_file), ('kappa', cfg.kappa_file), ('sigma_eq', cfg.sigma_eq_file), ('universe_mask', cfg.universe_mask_file)):
        if not p.is_file():
            raise FileNotFoundError(f'Missing {label} input: {p}\nRun prior phases so Phase 3 outputs exist, or use config/smoke_config.yaml after run_phase1.py --demo.')
    s_scores = pd.read_parquet(cfg.s_scores_file)
    kappa = pd.read_parquet(cfg.kappa_file)
    sigma_eq = pd.read_parquet(cfg.sigma_eq_file)
    universe_mask = pd.read_parquet(cfg.universe_mask_file)
    for df in (s_scores, kappa, sigma_eq, universe_mask):
        df.index = pd.DatetimeIndex(df.index).normalize()
    s_scores = s_scores.reindex(columns=sorted(s_scores.columns))
    kappa = kappa.reindex(columns=s_scores.columns)
    sigma_eq = sigma_eq.reindex(columns=s_scores.columns)
    universe_mask = universe_mask.reindex(columns=s_scores.columns)
    raw, positions, diag = build_phase4_outputs(s_scores, kappa, sigma_eq, universe_mask, s_open=cfg.s_open, s_close=cfg.s_close, gross_notional=cfg.gross_notional, max_half_life_days=cfg.max_half_life_days, sigma_entry_floor=cfg.sigma_entry_floor)
    cfg.processed_dir.mkdir(parents=True, exist_ok=True)
    raw.to_parquet(cfg.raw_signals_file, engine='pyarrow')
    positions.to_parquet(cfg.positions_file, engine='pyarrow')
    diag.to_parquet(cfg.signal_diagnostics_file, engine='pyarrow')
    return (raw, positions, diag)
