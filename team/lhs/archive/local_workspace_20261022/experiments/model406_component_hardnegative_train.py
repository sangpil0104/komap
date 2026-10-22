#!/usr/bin/env python3
"""Use the unchanged Model404 paired trainer with Model406's joint loss."""

from __future__ import annotations

import importlib
import sys


loss_v1 = importlib.import_module("model406_component_hardnegative_loss")
sys.modules["model404_component_recall_loss"] = loss_v1
trainer = importlib.import_module("model404_component_recall_train")


if __name__ == "__main__":
    trainer.main()
