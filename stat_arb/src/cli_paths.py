from __future__ import annotations
from pathlib import Path

def resolve_config_path(config_arg: Path, package_root: Path) -> Path:
    p = Path(config_arg)
    if p.is_file():
        return p.resolve()
    rel = package_root / p
    if rel.is_file():
        return rel.resolve()
    if p.parts and p.parts[0] == 'stat_arb':
        alt = package_root / Path(*p.parts[1:])
        if alt.is_file():
            return alt.resolve()
    raise FileNotFoundError(f'Config file not found: {config_arg!s}\n  cwd: {Path.cwd()}\n  package root: {package_root}\n  tried: {rel}\nHint: from stat_arb use --config config/base_config.yaml (not stat_arb/config/...), or pass an absolute path.')
