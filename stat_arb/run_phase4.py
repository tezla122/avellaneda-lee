#!/usr/bin/env python3
from __future__ import annotations
import argparse
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.cli_paths import resolve_config_path
from src.signals.phase4 import load_phase4_config, run_phase4

def main() -> None:
    p = argparse.ArgumentParser(description='Phase 4 signals and position sizing.')
    p.add_argument('--config', type=Path, default=ROOT / 'config' / 'base_config.yaml', help='YAML config (paths + phase4 block)')
    args = p.parse_args()
    cfg = load_phase4_config(resolve_config_path(args.config, ROOT), root=ROOT)
    raw, positions, diag = run_phase4(cfg)
    print(f'Wrote {cfg.raw_signals_file} shape={raw.shape}')
    print(f'Wrote {cfg.positions_file} shape={positions.shape}')
    print(f'Wrote {cfg.signal_diagnostics_file} rows={len(diag)}')
if __name__ == '__main__':
    main()
