from __future__ import annotations
import subprocess
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent
SCRIPT = ROOT / 'stat_arb' / 'run_phase5.py'

def main() -> None:
    if not SCRIPT.is_file():
        print(f'Missing {SCRIPT}', file=sys.stderr)
        sys.exit(1)
    raise SystemExit(subprocess.call([sys.executable, str(SCRIPT)] + sys.argv[1:]))
if __name__ == '__main__':
    main()
