#!/usr/bin/env python3
"""Run the locked paired trainer with the v4 two-class component objective."""

from __future__ import annotations

import importlib
import sys


loss_v4 = importlib.import_module("model404_component_recall_loss_v4")
# The shared, unchanged trainer imports this stable module name. Inject the
# versioned implementation before importing it so its optimizer/data recipe is
# reused verbatim while the v3 single-class source remains immutable.
sys.modules["model404_component_recall_loss"] = loss_v4
trainer = importlib.import_module("model404_component_recall_train")


if __name__ == "__main__":
    trainer.main()
