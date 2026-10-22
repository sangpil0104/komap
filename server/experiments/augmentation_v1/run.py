#!/usr/bin/env python
"""Terminal-only experiments using the preserved model and training engine."""
import os
import sys
from pathlib import Path

sys.dont_write_bytecode = True
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "accuracy_v1/baseline/source/src"))
sys.path.insert(0, str(ROOT))

if __name__ == "__main__":
    from auglab.cli import main
    main()
