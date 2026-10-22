#!/usr/bin/env python3
"""Reuse the matched continuation trainer with Model408's confusion hinge."""

from __future__ import annotations

import importlib
import sys


loss_v1 = importlib.import_module("model408_tiny_confusion_loss")
sys.modules["model404_component_recall_loss"] = loss_v1
trainer = importlib.import_module("model404_component_recall_train")


if __name__ == "__main__":
    trainer.main()
