from __future__ import annotations
from itertools import product
from typing import Any
import pandas as pd
from src.backtest.metrics import compute_metrics
from src.backtest.phase5 import build_phase5_outputs
from src.signals.phase4 import build_phase4_outputs

def run_sensitivity_grid(s_scores: pd.DataFrame, kappa: pd.DataFrame, sigma_eq: pd.DataFrame, universe_mask: pd.DataFrame, log_returns: pd.DataFrame, grid: dict[str, list[Any]], *, phase5_kwargs_base: dict[str, Any] | None=None) -> pd.DataFrame:
    p4_keys = {'s_open', 's_close', 'gross_notional', 'max_half_life_days', 'sigma_entry_floor'}
    p5_keys = {'initial_aum', 'transaction_cost_bps', 'short_borrow_bps_annual', 'drift_adjusted_tc'}
    if not grid:
        return pd.DataFrame()
    keys = list(grid.keys())
    phase5_base = dict(phase5_kwargs_base or {})
    rows: list[dict[str, Any]] = []
    for combo in product(*[grid[k] for k in keys]):
        kw = dict(zip(keys, combo))
        kw4 = {k: v for k, v in kw.items() if k in p4_keys}
        kw5 = {k: v for k, v in kw.items() if k in p5_keys}
        merged5 = {**phase5_base, **kw5}
        raw, positions, _diag = build_phase4_outputs(s_scores, kappa, sigma_eq, universe_mask, **kw4)
        pnl, metrics, _att = build_phase5_outputs(positions, log_returns, **merged5)
        row = {**kw, **metrics}
        row['net_pnl_mean'] = float(pnl['net_pnl'].mean())
        rows.append(row)
    return pd.DataFrame(rows)
