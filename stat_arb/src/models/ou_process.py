from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
import yaml
DT_ANNUAL = 1.0 / 252.0

@dataclass
class Phase3Config:
    cumulative_residuals_file: Path
    universe_mask_file: Path
    processed_dir: Path
    kappa_file: Path
    m_bar_file: Path
    sigma_eq_file: Path
    s_scores_file: Path
    ou_diagnostics_file: Path
    W_ou: int = 60
    sigma_eq_floor: float = 0.0001
    b_clip: tuple[float, float] = (-0.999, 0.999)
    b_ou_bounds: tuple[float, float] = (1e-10, 0.999)
    near_unit_root_threshold: float = 0.99
    root: Path | None = None

    def resolve_paths(self) -> None:
        base = self.root or Path.cwd()
        for name in ('cumulative_residuals_file', 'universe_mask_file', 'processed_dir', 'kappa_file', 'm_bar_file', 'sigma_eq_file', 's_scores_file', 'ou_diagnostics_file'):
            p = getattr(self, name)
            if isinstance(p, Path) and (not p.is_absolute()):
                setattr(self, name, (base / p).resolve())

def load_phase3_config(path: str | Path, root: Path | None=None) -> Phase3Config:
    path = Path(path)
    with open(path, encoding='utf-8') as f:
        raw = yaml.safe_load(f)
    paths = raw.get('paths') or {}
    p2 = raw.get('phase2') or {}
    p3 = raw.get('phase3') or {}
    cum_default = p2.get('cumulative_residuals_file', 'data/processed/cumulative_residuals.parquet')
    cfg = Phase3Config(cumulative_residuals_file=Path(p3.get('cumulative_residuals_file', paths.get('cumulative_residuals_file', cum_default))), universe_mask_file=Path(paths.get('universe_mask_file', 'data/processed/universe_mask.parquet')), processed_dir=Path(paths.get('processed_dir', 'data/processed')), kappa_file=Path(p3.get('kappa_file', 'data/processed/kappa.parquet')), m_bar_file=Path(p3.get('m_bar_file', 'data/processed/m_bar.parquet')), sigma_eq_file=Path(p3.get('sigma_eq_file', 'data/processed/sigma_eq.parquet')), s_scores_file=Path(p3.get('s_scores_file', 'data/processed/s_scores.parquet')), ou_diagnostics_file=Path(p3.get('ou_diagnostics_file', 'data/processed/ou_diagnostics.json')), W_ou=int(p3.get('W_ou', 60)), sigma_eq_floor=float(p3.get('sigma_eq_floor', 0.0001)), near_unit_root_threshold=float(p3.get('near_unit_root_threshold', 0.99)), root=root)
    cfg.resolve_paths()
    return cfg

def build_phase3_outputs(X: pd.DataFrame, universe_mask: pd.DataFrame, *, W_ou: int=60, sigma_eq_floor: float=0.0001, b_clip: tuple[float, float]=(-0.999, 0.999), b_ou_bounds: tuple[float, float]=(1e-10, 0.999), near_unit_root_threshold: float=0.99) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    if not X.index.equals(universe_mask.index):
        raise ValueError('X and universe_mask must share the same index.')
    if not X.columns.equals(universe_mask.columns):
        raise ValueError('X and universe_mask must share the same columns.')
    X = X.astype(float)
    x_lag = X.shift(1)
    mx = x_lag.rolling(W_ou, min_periods=W_ou).mean()
    my = X.rolling(W_ou, min_periods=W_ou).mean()
    ex = x_lag - mx
    ey = X - my
    cov_xy = (ex * ey).rolling(W_ou, min_periods=W_ou).sum() / max(W_ou - 1, 1)
    var_x = x_lag.rolling(W_ou, min_periods=W_ou).var(ddof=1)
    b = cov_xy / var_x.replace(0.0, np.nan)
    a = my - b * mx
    near_unit_root = b > near_unit_root_threshold
    b_c = b.clip(lower=b_clip[0], upper=b_clip[1])
    lo, hi = b_ou_bounds
    b_ou = b_c.clip(lower=lo, upper=hi)
    xi = X - a.shift(1) - b.shift(1) * x_lag
    sigma_xi = xi.rolling(W_ou, min_periods=W_ou).std()
    dt = DT_ANNUAL
    log_b = np.log(b_ou)
    kappa = -log_b / dt
    m_bar = a / (1.0 - b_ou).replace(0.0, np.nan)
    inner = -2.0 * log_b / (dt * (1.0 - b_ou ** 2))
    inner = inner.clip(lower=0.0)
    sigma = sigma_xi * np.sqrt(inner)
    denom_ou = np.sqrt(2.0 * kappa)
    sigma_eq = sigma / denom_ou.replace(0.0, np.nan)
    sigma_eq = sigma_eq.clip(lower=sigma_eq_floor)
    m_eff = m_bar.shift(1)
    se_eff = sigma_eq.shift(1)
    s_scores = (X - m_eff) / se_eff.replace(0.0, np.nan)
    active = universe_mask.astype(bool)
    kappa = kappa.where(active)
    m_bar = m_bar.where(active)
    sigma_eq = sigma_eq.where(active)
    s_scores = s_scores.where(active)
    n_active = int(active.sum().sum())
    n_flag = int((near_unit_root & active).sum().sum())
    diag: dict[str, Any] = {'W_ou': W_ou, 'sigma_eq_floor': sigma_eq_floor, 'near_unit_root_fraction': n_flag / n_active if n_active else 0.0, 'near_unit_root_threshold': near_unit_root_threshold}
    return (kappa, m_bar, sigma_eq, s_scores, diag)

def run_phase3(cfg: Phase3Config | None=None, *, config_path: Path | None=None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    if cfg is None:
        if config_path is None:
            raise ValueError('Provide cfg or config_path.')
        cfg = load_phase3_config(config_path)
    if not cfg.cumulative_residuals_file.is_file():
        raise FileNotFoundError(f'Missing Phase 2 output: {cfg.cumulative_residuals_file}\nRun Phase 2 after Phase 1 has produced returns and universe_mask, e.g.\n  cd stat_arb && python run_phase2.py --config config/base_config.yaml\nOr from the repo root:\n  python run_phase2.py --config stat_arb/config/base_config.yaml')
    if not cfg.universe_mask_file.is_file():
        raise FileNotFoundError(f'Missing universe mask: {cfg.universe_mask_file}\nRun Phase 1 first if this file does not exist.')
    X = pd.read_parquet(cfg.cumulative_residuals_file)
    universe_mask = pd.read_parquet(cfg.universe_mask_file)
    X.index = pd.DatetimeIndex(X.index).normalize()
    universe_mask.index = pd.DatetimeIndex(universe_mask.index).normalize()
    kappa, m_bar, sigma_eq, s_scores, diag = build_phase3_outputs(X, universe_mask, W_ou=cfg.W_ou, sigma_eq_floor=cfg.sigma_eq_floor, b_clip=cfg.b_clip, b_ou_bounds=cfg.b_ou_bounds, near_unit_root_threshold=cfg.near_unit_root_threshold)
    cfg.processed_dir.mkdir(parents=True, exist_ok=True)
    kappa.to_parquet(cfg.kappa_file, engine='pyarrow')
    m_bar.to_parquet(cfg.m_bar_file, engine='pyarrow')
    sigma_eq.to_parquet(cfg.sigma_eq_file, engine='pyarrow')
    s_scores.to_parquet(cfg.s_scores_file, engine='pyarrow')
    with open(cfg.ou_diagnostics_file, 'w', encoding='utf-8') as f:
        json.dump(diag, f, indent=2)
    return (kappa, m_bar, sigma_eq, s_scores, diag)
