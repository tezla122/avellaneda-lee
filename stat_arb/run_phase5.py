#!/usr/bin/env python3
from __future__ import annotations
import argparse
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.cli_paths import resolve_config_path
from src.backtest.phase5 import load_phase5_config, run_phase5

def main() -> None:
    p = argparse.ArgumentParser(description='Phase 5 backtest and analytics.')
    p.add_argument('--config', type=Path, default=ROOT / 'config' / 'base_config.yaml', help='YAML config (paths + phase5 block)')
    args = p.parse_args()
    cfg = load_phase5_config(resolve_config_path(args.config, ROOT), root=ROOT)
    pnl, metrics, _att = run_phase5(cfg)
    print(f'Wrote {cfg.pnl_file} rows={len(pnl)}')
    print(f"Wrote {cfg.metrics_file} sharpe={metrics.get('sharpe')}")
if __name__ == '__main__':
    main()
