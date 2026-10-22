import os
import sys
from pathlib import Path

sys.dont_write_bytecode = True
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
workspace = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(workspace/"baseline/source/src"))
sys.path.insert(0, str(workspace))
