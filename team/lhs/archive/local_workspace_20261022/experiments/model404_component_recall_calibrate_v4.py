#!/usr/bin/env python3
"""Run v4 two-class loss calibration through the unchanged Model404 calibrator."""

from __future__ import annotations

import importlib
import sys


loss_v4 = importlib.import_module("model404_component_recall_loss_v4")
sys.modules["model404_component_recall_loss"] = loss_v4
calibrator = importlib.import_module("model404_component_recall_calibrate")


if __name__ == "__main__":
    calibrator.main()
