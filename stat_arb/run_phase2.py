#!/usr/bin/env python3
from __future__ import annotations
import argparse
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.cli_paths import resolve_config_path
from src.factors.residuals import load_phase2_config, run_phase2

def main() -> None:
    p = argparse.ArgumentParser(description='Phase 2 PCA factors and residuals.')
    p.add_argument('--config', type=Path, default=ROOT / 'config' / 'base_config.yaml', help='YAML config (expects paths + phase2 block)')
    args = p.parse_args()
    cfg = load_phase2_config(resolve_config_path(args.config, ROOT), root=ROOT)
    cumulative, betas_dict, diagnostics = run_phase2(cfg)
    print(f'Wrote {cfg.cumulative_residuals_file} shape={cumulative.shape}')
    print(f'Wrote {cfg.pca_diagnostics_file} rows={len(diagnostics)}')
    print(f'Betas rebalances={len(betas_dict)} -> {cfg.betas_dir}')
if __name__ == '__main__':
    main()
