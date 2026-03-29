from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
import yaml
from src.validation.dsr import deflated_sharpe_ratio
from src.validation.regimes import load_vix_series, regime_metrics, vix_bucket
from src.validation.sensitivity import run_sensitivity_grid

@dataclass
class Phase6Config:
    s_scores_file: Path
    kappa_file: Path
    sigma_eq_file: Path
    universe_mask_file: Path
    returns_file: Path
    pnl_file: Path
    processed_dir: Path
    dsr_file: Path
    regime_metrics_file: Path
    sensitivity_file: Path
    vix_file: Path | None
    sensitivity_grid: dict[str, list[Any]]
    phase5_defaults: dict[str, Any]
    root: Path | None = None

    def resolve_paths(self) -> None:
        base = self.root or Path.cwd()
        for name in ('s_scores_file', 'kappa_file', 'sigma_eq_file', 'universe_mask_file', 'returns_file', 'pnl_file', 'processed_dir', 'dsr_file', 'regime_metrics_file', 'sensitivity_file'):
            p = getattr(self, name)
            if isinstance(p, Path) and (not p.is_absolute()):
                setattr(self, name, (base / p).resolve())
        if self.vix_file is not None and isinstance(self.vix_file, Path) and (not self.vix_file.is_absolute()):
            self.vix_file = (base / self.vix_file).resolve()

def load_phase6_config(path: str | Path, root: Path | None=None) -> Phase6Config:
    path = Path(path)
    with open(path, encoding='utf-8') as f:
        raw = yaml.safe_load(f)
    paths = raw.get('paths') or {}
    p5 = raw.get('phase5') or {}
    p6 = raw.get('phase6') or {}
    vix = p6.get('vix_file')
    cfg = Phase6Config(s_scores_file=Path(p6.get('s_scores_file', 'data/processed/s_scores.parquet')), kappa_file=Path(p6.get('kappa_file', 'data/processed/kappa.parquet')), sigma_eq_file=Path(p6.get('sigma_eq_file', 'data/processed/sigma_eq.parquet')), universe_mask_file=Path(paths.get('universe_mask_file', 'data/processed/universe_mask.parquet')), returns_file=Path(paths.get('returns_file', 'data/processed/returns.parquet')), pnl_file=Path(p6.get('pnl_file', p5.get('pnl_file', 'data/processed/pnl.parquet'))), processed_dir=Path(paths.get('processed_dir', 'data/processed')), dsr_file=Path(p6.get('dsr_file', 'data/processed/validation_dsr.json')), regime_metrics_file=Path(p6.get('regime_metrics_file', 'data/processed/regime_metrics.parquet')), sensitivity_file=Path(p6.get('sensitivity_file', 'data/processed/sensitivity_results.parquet')), vix_file=Path(vix) if vix else None, sensitivity_grid=dict(p6.get('sensitivity_grid') or {}), phase5_defaults=dict(p6.get('phase5_defaults') or {}), root=root)
    cfg.resolve_paths()
    return cfg

def run_phase6(cfg: Phase6Config | None=None, *, config_path: Path | None=None) -> dict[str, Any]:
    if cfg is None:
        if config_path is None:
            raise ValueError('Provide cfg or config_path.')
        cfg = load_phase6_config(config_path)
    for label, p in (('s_scores', cfg.s_scores_file), ('kappa', cfg.kappa_file), ('sigma_eq', cfg.sigma_eq_file), ('universe_mask', cfg.universe_mask_file), ('returns', cfg.returns_file)):
        if not p.is_file():
            raise FileNotFoundError(f'Missing {label}: {p}')
    s_scores = pd.read_parquet(cfg.s_scores_file)
    kappa = pd.read_parquet(cfg.kappa_file)
    sigma_eq = pd.read_parquet(cfg.sigma_eq_file)
    universe_mask = pd.read_parquet(cfg.universe_mask_file)
    log_returns = pd.read_parquet(cfg.returns_file)
    for df in (s_scores, kappa, sigma_eq, universe_mask, log_returns):
        df.index = pd.DatetimeIndex(df.index).normalize()
    out: dict[str, Any] = {}
    if cfg.pnl_file.is_file():
        pnl = pd.read_parquet(cfg.pnl_file)
        pnl.index = pd.DatetimeIndex(pnl.index).normalize()
        net = pnl['net_pnl']
    else:
        net = pd.Series(dtype=float)
    if len(net):
        dsr, dsr_meta = deflated_sharpe_ratio(net)
        out['deflated_sharpe_ratio'] = dsr
        out['dsr_meta'] = dsr_meta
    else:
        out['deflated_sharpe_ratio'] = float('nan')
        out['dsr_meta'] = {'reason': 'no pnl_file'}
    regime_df = pd.DataFrame()
    if cfg.vix_file is not None and cfg.vix_file.is_file() and len(net):
        vix = load_vix_series(cfg.vix_file)
        vix = vix.reindex(net.index).ffill()
        reg = vix_bucket(vix)
        regime_df = regime_metrics(net, reg)
        out['regime_metrics_rows'] = int(len(regime_df))
    sens_df = pd.DataFrame()
    if cfg.sensitivity_grid:
        sens_df = run_sensitivity_grid(s_scores, kappa, sigma_eq, universe_mask, log_returns, cfg.sensitivity_grid, phase5_kwargs_base=cfg.phase5_defaults)
        out['sensitivity_rows'] = int(len(sens_df))
    cfg.processed_dir.mkdir(parents=True, exist_ok=True)

    def _json_safe(x: Any) -> Any:
        if isinstance(x, dict):
            return {k: _json_safe(v) for k, v in x.items()}
        if isinstance(x, float):
            return x if np.isfinite(x) else None
        if isinstance(x, (np.integer, np.floating)):
            return float(x)
        return x
    with open(cfg.dsr_file, 'w', encoding='utf-8') as f:
        json.dump(_json_safe(out), f, indent=2)
    if len(regime_df):
        regime_df.to_parquet(cfg.regime_metrics_file, engine='pyarrow')
    if len(sens_df):
        sens_df.to_parquet(cfg.sensitivity_file, engine='pyarrow')
    return out
