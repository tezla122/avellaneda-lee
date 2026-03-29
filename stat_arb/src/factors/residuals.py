from __future__ import annotations
import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
import yaml
from src.factors.pca import maybe_ledoit_wolf_cov, pca_loadings_from_cov, sample_covariance

@dataclass
class Phase2Config:
    returns_file: Path
    universe_mask_file: Path
    processed_dir: Path
    cumulative_residuals_file: Path
    pca_diagnostics_file: Path
    betas_dir: Path
    W: int = 252
    K: int = 15
    rebal_freq: str = 'BMS'
    cond_warn_threshold: float = 1000.0
    root: Path | None = None

    def resolve_paths(self) -> None:
        base = self.root or Path.cwd()
        self.returns_file = (base / self.returns_file).resolve()
        self.universe_mask_file = (base / self.universe_mask_file).resolve()
        self.processed_dir = (base / self.processed_dir).resolve()
        self.cumulative_residuals_file = (base / self.cumulative_residuals_file).resolve()
        self.pca_diagnostics_file = (base / self.pca_diagnostics_file).resolve()
        self.betas_dir = (base / self.betas_dir).resolve()

def load_phase2_config(path: str | Path, root: Path | None=None) -> Phase2Config:
    path = Path(path)
    with open(path, encoding='utf-8') as f:
        raw = yaml.safe_load(f)
    p2 = raw.get('phase2') or {}
    paths = raw.get('paths') or {}
    cfg = Phase2Config(returns_file=Path(paths.get('returns_file', 'data/processed/returns.parquet')), universe_mask_file=Path(paths.get('universe_mask_file', 'data/processed/universe_mask.parquet')), processed_dir=Path(paths.get('processed_dir', 'data/processed')), cumulative_residuals_file=Path(p2.get('cumulative_residuals_file', 'data/processed/cumulative_residuals.parquet')), pca_diagnostics_file=Path(p2.get('pca_diagnostics_file', 'data/processed/pca_diagnostics.parquet')), betas_dir=Path(p2.get('betas_dir', 'data/processed/betas')), W=int(p2.get('W', 252)), K=int(p2.get('K', 15)), rebal_freq=str(p2.get('rebal_freq', 'BMS')), cond_warn_threshold=float(p2.get('cond_warn_threshold', 1000.0)), root=root)
    cfg.resolve_paths()
    return cfg

def _rebalancing_dates(idx: pd.DatetimeIndex, freq: str) -> pd.DatetimeIndex:
    if len(idx) == 0:
        return pd.DatetimeIndex([])
    start, end = (idx.min(), idx.max())
    if freq.upper() == 'BMS':
        dr = pd.date_range(start=start, end=end, freq='BMS')
    elif freq.upper() in ('MS', 'M'):
        dr = pd.date_range(start=start, end=end, freq='MS')
    else:
        dr = pd.date_range(start=start, end=end, freq=freq)
    dr = pd.DatetimeIndex(dr).normalize()
    ix = pd.DatetimeIndex(idx).normalize()
    return dr.intersection(ix).sort_values()

def _pos_of_date(idx: pd.DatetimeIndex, d: pd.Timestamp) -> int:
    pos = idx.get_indexer([pd.Timestamp(d)], method=None)[0]
    if pos < 0:
        raise KeyError(d)
    return int(pos)

def build_phase2_outputs(returns: pd.DataFrame, universe_mask: pd.DataFrame, *, W: int=252, K: int=15, rebal_freq: str='BMS', cond_warn_threshold: float=1000.0) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], pd.DataFrame]:
    if not returns.index.equals(universe_mask.index):
        raise ValueError('returns and universe_mask must share the same index.')
    if not returns.columns.equals(universe_mask.columns):
        raise ValueError('returns and universe_mask must share the same columns.')
    idx = returns.index
    cols = list(returns.columns)
    rebal_dates = _rebalancing_dates(idx, rebal_freq)
    cumulative = pd.DataFrame(np.nan, index=idx, columns=cols, dtype=float)
    betas_dict: dict[str, pd.DataFrame] = {}
    diag_rows: list[dict[str, Any]] = []
    b_prev: np.ndarray | None = None
    for j, rebal in enumerate(rebal_dates):
        pos = _pos_of_date(idx, rebal)
        if pos < W:
            continue
        win_start = pos - W
        win_end = pos - 1
        win_idx = idx[win_start:win_end + 1]
        t_prev = idx[pos - 1]
        active = universe_mask.loc[t_prev]
        active_tickers = active[active].index.tolist()
        if len(active_tickers) <= K:
            continue
        R_win = returns.loc[win_idx, active_tickers]
        valid_cols = R_win.columns[R_win.notna().all(axis=0)]
        R_win = R_win[valid_cols]
        if R_win.shape[0] != W or R_win.shape[1] <= K:
            continue
        tickers = list(R_win.columns)
        R_w = np.asarray(R_win.values, dtype=float)
        mu = R_w.mean(axis=0)
        R_centered = R_w - mu
        cov = sample_covariance(R_centered, ddof=1)
        cov = maybe_ledoit_wolf_cov(R_centered, cov, K)
        rank_cov = int(np.linalg.matrix_rank(cov))
        B, eigvals, evr_top = pca_loadings_from_cov(cov, K, b_prev=b_prev)
        b_prev = B.copy()
        f_in = R_centered @ B
        ones = np.ones((W, 1))
        X = np.hstack([ones, f_in])
        w_mat = X.T @ X
        cond_ftf = float(np.linalg.cond(w_mat))
        if cond_ftf > cond_warn_threshold:
            warnings.warn(f"Rebalance {rebal.date()}: cond(X'X)={cond_ftf:.2e} exceeds {cond_warn_threshold}.", stacklevel=2)
        theta, *_ = np.linalg.lstsq(X, R_w, rcond=None)
        eps_in = R_w - X @ theta
        _ = eps_in
        if j + 1 < len(rebal_dates):
            next_rebal = rebal_dates[j + 1]
            p_end = _pos_of_date(idx, next_rebal)
            period_idx = idx[pos:p_end]
        else:
            period_idx = idx[pos:]
        if len(period_idx) == 0:
            continue
        R_os = returns.loc[period_idx, tickers].astype(float)
        u_m = universe_mask.loc[period_idx, tickers].astype(bool)
        mu_b = mu[np.newaxis, :]
        f_os = (R_os.values - mu_b) @ B
        ones_os = np.ones((len(period_idx), 1))
        x_os = np.hstack([ones_os, f_os])
        pred = x_os @ theta
        eps = R_os.values - pred
        eps_df = pd.DataFrame(eps, index=period_idx, columns=tickers)
        eps_df = eps_df.where(u_m.values)
        cum_period = eps_df.fillna(0.0).cumsum()
        cum_period = cum_period.where(u_m)
        cumulative.loc[period_idx, tickers] = cum_period.values
        beta_df = pd.DataFrame(theta.T, index=tickers, columns=['alpha'] + [f'f{k + 1}' for k in range(theta.shape[0] - 1)])
        key = rebal.strftime('%Y-%m-%d')
        betas_dict[key] = beta_df
        row: dict[str, Any] = {'rebalance_date': key, 'n_assets': len(tickers), 'rank_cov': rank_cov, 'condition_XtX': cond_ftf, 'explained_variance_top_k': evr_top}
        for kk in range(K):
            row[f'eigenvalue_{kk + 1}'] = float(eigvals[kk]) if kk < len(eigvals) else np.nan
        diag_rows.append(row)
    diagnostics = pd.DataFrame(diag_rows)
    if len(diagnostics) and 'rebalance_date' in diagnostics.columns:
        diagnostics = diagnostics.set_index('rebalance_date')
    return (cumulative, betas_dict, diagnostics)

def run_phase2(cfg: Phase2Config | None=None, *, config_path: Path | None=None) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], pd.DataFrame]:
    if cfg is None:
        if config_path is None:
            raise ValueError('Provide cfg or config_path.')
        cfg = load_phase2_config(config_path)
    if not cfg.returns_file.is_file():
        raise FileNotFoundError(f'Missing Phase 1 output (returns): {cfg.returns_file}\nRun Phase 1 first to create returns.parquet and universe_mask.parquet, e.g.\n  cd stat_arb && python run_phase1.py --config config/base_config.yaml\nOr from the repo root:\n  python run_phase1.py --config stat_arb/config/base_config.yaml')
    if not cfg.universe_mask_file.is_file():
        raise FileNotFoundError(f'Missing Phase 1 output (universe mask): {cfg.universe_mask_file}\nRun Phase 1 first (same outputs as returns.parquet).')
    returns = pd.read_parquet(cfg.returns_file)
    universe_mask = pd.read_parquet(cfg.universe_mask_file, engine='pyarrow')
    returns.index = pd.DatetimeIndex(returns.index).normalize()
    universe_mask.index = pd.DatetimeIndex(universe_mask.index).normalize()
    cumulative, betas_dict, diagnostics = build_phase2_outputs(returns, universe_mask, W=cfg.W, K=cfg.K, rebal_freq=cfg.rebal_freq, cond_warn_threshold=cfg.cond_warn_threshold)
    cfg.processed_dir.mkdir(parents=True, exist_ok=True)
    cumulative.to_parquet(cfg.cumulative_residuals_file, engine='pyarrow')
    diagnostics.to_parquet(cfg.pca_diagnostics_file, engine='pyarrow')
    cfg.betas_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {'rebalances': []}
    for name, bdf in sorted(betas_dict.items()):
        safe = str(name).replace(':', '-')[:19]
        p = cfg.betas_dir / f'betas_{safe}.parquet'
        bdf.to_parquet(p, engine='pyarrow')
        manifest['rebalances'].append({'date': name, 'file': str(p.name)})
    with open(cfg.betas_dir / 'manifest.json', 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2)
    return (cumulative, betas_dict, diagnostics)
