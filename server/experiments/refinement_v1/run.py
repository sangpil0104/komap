#!/usr/bin/env python
import os
import sys
from pathlib import Path

sys.dont_write_bytecode = True
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT), str(ROOT.parent / "accuracy_v1/baseline/source/src")]

if __name__ == "__main__":
    from refinelab.cli import main
    main()
