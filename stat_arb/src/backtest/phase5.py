from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
import yaml
from src.backtest.engine import log_to_simple_returns, run_vectorized_backtest
from src.backtest.metrics import attribution_long_short, compute_metrics, rolling_sharpe

@dataclass
class Phase5Config:
    positions_file: Path
    returns_file: Path
    processed_dir: Path
    pnl_file: Path
    aum_file: Path
    metrics_file: Path
    attribution_file: Path
    initial_aum: float = 1.0
    transaction_cost_bps: float = 5.0
    short_borrow_bps_annual: float = 0.0
    drift_adjusted_tc: bool = True
    benchmark_file: Path | None = None
    root: Path | None = None

    def resolve_paths(self) -> None:
        base = self.root or Path.cwd()
        for name in ('positions_file', 'returns_file', 'processed_dir', 'pnl_file', 'aum_file', 'metrics_file', 'attribution_file'):
            p = getattr(self, name)
            if isinstance(p, Path) and (not p.is_absolute()):
                setattr(self, name, (base / p).resolve())
        if self.benchmark_file is not None and isinstance(self.benchmark_file, Path):
            if not self.benchmark_file.is_absolute():
                self.benchmark_file = (base / self.benchmark_file).resolve()

def load_phase5_config(path: str | Path, root: Path | None=None) -> Phase5Config:
    path = Path(path)
    with open(path, encoding='utf-8') as f:
        raw = yaml.safe_load(f)
    paths = raw.get('paths') or {}
    p4 = raw.get('phase4') or {}
    p5 = raw.get('phase5') or {}
    bench = p5.get('benchmark_file')
    cfg = Phase5Config(positions_file=Path(p5.get('positions_file', p4.get('positions_file', 'data/processed/positions.parquet'))), returns_file=Path(paths.get('returns_file', 'data/processed/returns.parquet')), processed_dir=Path(paths.get('processed_dir', 'data/processed')), pnl_file=Path(p5.get('pnl_file', 'data/processed/pnl.parquet')), aum_file=Path(p5.get('aum_file', 'data/processed/aum.parquet')), metrics_file=Path(p5.get('metrics_file', 'data/processed/metrics.json')), attribution_file=Path(p5.get('attribution_file', 'data/processed/attribution.parquet')), initial_aum=float(p5.get('initial_aum', 1.0)), transaction_cost_bps=float(p5.get('transaction_cost_bps', 5.0)), short_borrow_bps_annual=float(p5.get('short_borrow_bps_annual', 0.0)), drift_adjusted_tc=bool(p5.get('drift_adjusted_tc', True)), benchmark_file=Path(bench) if bench else None, root=root)
    cfg.resolve_paths()
    return cfg

def build_phase5_outputs(positions: pd.DataFrame, log_returns: pd.DataFrame, *, initial_aum: float=1.0, transaction_cost_bps: float=5.0, short_borrow_bps_annual: float=0.0, drift_adjusted_tc: bool=True, benchmark: pd.Series | None=None) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame]:
    idx = positions.index
    cols = sorted(set(positions.columns).intersection(log_returns.columns))
    positions = positions.reindex(index=idx, columns=cols).fillna(0.0)
    log_returns = log_returns.reindex(index=idx, columns=cols)
    out = run_vectorized_backtest(positions, log_returns, initial_aum=initial_aum, transaction_cost_bps=transaction_cost_bps, short_borrow_bps_annual=short_borrow_bps_annual, drift_adjusted_tc=drift_adjusted_tc)
    simple = log_to_simple_returns(log_returns.reindex_like(positions))
    att = attribution_long_short(positions, simple)
    pnl = pd.DataFrame({'gross_pnl': out['gross_pnl'], 'transaction_costs': out['transaction_costs'], 'short_borrow_cost': out['short_borrow_cost'], 'net_pnl': out['net_pnl']}, index=positions.index)
    pnl['aum'] = out['aum']
    pnl['rolling_sharpe_63'] = rolling_sharpe(out['net_pnl'], 63)
    metrics = compute_metrics(out['net_pnl'], out['aum'], delta_weight_l1=out['delta_weight_l1'], benchmark=benchmark)
    metrics['execution_model'] = 'Positions at t-1 times simple return on (t-1,t]; TC on weight changes vs drift-adjusted prior (optional).'
    return (pnl, metrics, att)

def run_phase5(cfg: Phase5Config | None=None, *, config_path: Path | None=None) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame]:
    if cfg is None:
        if config_path is None:
            raise ValueError('Provide cfg or config_path.')
        cfg = load_phase5_config(config_path)
    for label, p in (('positions', cfg.positions_file), ('returns', cfg.returns_file)):
        if not p.is_file():
            raise FileNotFoundError(f'Missing {label}: {p}\nRun Phase 1 (returns) and Phase 4 (positions), or the smoke pipeline.')
    positions = pd.read_parquet(cfg.positions_file)
    log_returns = pd.read_parquet(cfg.returns_file)
    for df in (positions, log_returns):
        df.index = pd.DatetimeIndex(df.index).normalize()
    bench = None
    if cfg.benchmark_file is not None and cfg.benchmark_file.is_file():
        b = pd.read_parquet(cfg.benchmark_file)
        if isinstance(b, pd.DataFrame) and b.shape[1] == 1:
            bench = b.iloc[:, 0]
        else:
            bench = b.squeeze()
    if bench is not None:
        bench = bench.reindex(positions.index)
    pnl, metrics, att = build_phase5_outputs(positions, log_returns, initial_aum=cfg.initial_aum, transaction_cost_bps=cfg.transaction_cost_bps, short_borrow_bps_annual=cfg.short_borrow_bps_annual, drift_adjusted_tc=cfg.drift_adjusted_tc, benchmark=bench)
    cfg.processed_dir.mkdir(parents=True, exist_ok=True)
    pnl.to_parquet(cfg.pnl_file, engine='pyarrow')
    pnl[['aum']].to_parquet(cfg.aum_file, engine='pyarrow')

    def _json_safe(x: Any) -> Any:
        if isinstance(x, dict):
            return {k: _json_safe(v) for k, v in x.items()}
        if isinstance(x, float):
            return x if np.isfinite(x) else None
        if hasattr(x, 'item'):
            try:
                return x.item()
            except Exception:
                return str(x)
        return x
    with open(cfg.metrics_file, 'w', encoding='utf-8') as f:
        json.dump(_json_safe(metrics), f, indent=2)
    att.to_parquet(cfg.attribution_file, engine='pyarrow')
    return (pnl, metrics, att)
