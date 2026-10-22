#!/usr/bin/env python
"""Isolated entry point. The existing editable alphase installation is not changed."""
import os
import sys
from pathlib import Path

sys.dont_write_bytecode = True
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
WORKSPACE = Path(__file__).resolve().parent
sys.path.insert(0, str(WORKSPACE / "baseline/source/src"))
sys.path.insert(0, str(WORKSPACE))

if __name__ == "__main__":
    from lab.cli import main
    main()
