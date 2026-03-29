#!/usr/bin/env python3
from __future__ import annotations
import argparse
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.cli_paths import resolve_config_path
from src.models.ou_process import load_phase3_config, run_phase3

def main() -> None:
    p = argparse.ArgumentParser(description='Phase 3 OU process and s-scores.')
    p.add_argument('--config', type=Path, default=ROOT / 'config' / 'base_config.yaml', help='YAML config (paths + phase3 block)')
    args = p.parse_args()
    cfg = load_phase3_config(resolve_config_path(args.config, ROOT), root=ROOT)
    kappa, m_bar, sigma_eq, s_scores, diag = run_phase3(cfg)
    print(f'Wrote {cfg.kappa_file} shape={kappa.shape}')
    print(f'Wrote {cfg.s_scores_file} shape={s_scores.shape}')
    print(f'Diagnostics: {diag}')
if __name__ == '__main__':
    main()
