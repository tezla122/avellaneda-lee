from __future__ import annotations
from pathlib import Path
import pytest
from src.cli_paths import resolve_config_path

def test_resolve_from_package_root():
    root = Path(__file__).resolve().parents[1]
    p = resolve_config_path(Path('config/base_config.yaml'), root)
    assert p.is_file()
    p2 = resolve_config_path(Path('stat_arb/config/base_config.yaml'), root)
    assert p2 == p

def test_resolve_missing_raises():
    root = Path(__file__).resolve().parents[1]
    with pytest.raises(FileNotFoundError):
        resolve_config_path(Path('config/does_not_exist.yaml'), root)
