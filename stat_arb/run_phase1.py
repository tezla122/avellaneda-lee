#!/usr/bin/env python3
from __future__ import annotations
import argparse
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.cli_paths import resolve_config_path
from src.data.preprocessor import load_config, run_phase1

def main() -> None:
    p = argparse.ArgumentParser(description='Phase 1 data ingestion and preprocessing.')
    p.add_argument('--config', type=Path, default=ROOT / 'config' / 'base_config.yaml', help='Path to YAML config (default: config/base_config.yaml)')
    p.add_argument('--demo', action='store_true', help='Write sample DEMO*.csv under data/raw and run config/smoke_config.yaml')
    args = p.parse_args()
    if args.demo:
        from src.sample_data import write_sample_raw
        write_sample_raw(ROOT / 'data' / 'raw')
        cfg_path = resolve_config_path(ROOT / 'config' / 'smoke_config.yaml', ROOT)
    else:
        cfg_path = resolve_config_path(args.config, ROOT)
    cfg = load_config(cfg_path, root=ROOT)
    returns, universe_mask, meta = run_phase1(cfg)
    print(f'Wrote {cfg.returns_file} shape={returns.shape}')
    print(f'Wrote {cfg.universe_mask_file} shape={universe_mask.shape}')
    print(f"Wrote {cfg.metadata_file} tickers={len(meta.get('tickers', {}))}")
    if args.demo:
        print('Next (same smoke settings): python run_phase2.py --config config/smoke_config.yaml && python run_phase3.py --config config/smoke_config.yaml')
if __name__ == '__main__':
    main()
