#!/usr/bin/env python3
from __future__ import annotations
import argparse
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.cli_paths import resolve_config_path
from src.validation.phase6 import load_phase6_config, run_phase6

def main() -> None:
    p = argparse.ArgumentParser(description='Phase 6 validation and sensitivity.')
    p.add_argument('--config', type=Path, default=ROOT / 'config' / 'base_config.yaml', help='YAML config (paths + phase6 block)')
    args = p.parse_args()
    cfg = load_phase6_config(resolve_config_path(args.config, ROOT), root=ROOT)
    out = run_phase6(cfg)
    print(f"Wrote {cfg.dsr_file} DSR={out.get('deflated_sharpe_ratio')}")
    if cfg.sensitivity_grid:
        print(f'Wrote {cfg.sensitivity_file}')
if __name__ == '__main__':
    main()
