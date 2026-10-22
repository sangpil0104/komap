#!/usr/bin/env python3
"""Run the fixed Model404 continuation trainer with Model411's loss module."""

from __future__ import annotations

import importlib
import sys


loss_module = importlib.import_module("model411_bicc_loss")
sys.modules["model404_component_recall_loss"] = loss_module
trainer = importlib.import_module("model404_component_recall_train")


if __name__ == "__main__":
    trainer.main()
