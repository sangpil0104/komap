import os
from pathlib import Path
import sys

os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
