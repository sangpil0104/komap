import sys
from pathlib import Path
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT), str(ROOT / 'vendor')]
from evoacclab.cli import main
if __name__ == '__main__':
    main()
