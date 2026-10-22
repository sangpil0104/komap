#!/usr/bin/env python3
"""Reuse the matched Model404 trainer with Model407's joint component loss."""

from __future__ import annotations

import importlib
import sys


loss_v1 = importlib.import_module("model407_component_fp_loss")
sys.modules["model404_component_recall_loss"] = loss_v1
trainer = importlib.import_module("model404_component_recall_train")


if __name__ == "__main__":
    trainer.main()
