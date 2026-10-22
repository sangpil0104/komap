#!/usr/bin/env python3
"""Synthetic-only test for Model388's nested-key aggregation recovery."""

from model388_component_sampler_audit_recovery import add_legacy_nested_access_shape


def main() -> None:
    raw = {
        "baseline_miou": 0.5,
        "cells": {
            "1:64_to_lt256": {
                "component_count": 3,
                "fn_pixels": 7,
                "oracle_miou_gain": 0.125,
            }
        },
    }
    fixed = add_legacy_nested_access_shape(raw)
    cell = "1:64_to_lt256"
    cell_rows = [fixed["cells"][cell], fixed["cells"][cell]]
    assert sum(row["cells"][cell]["component_count"] for row in cell_rows) == 6
    assert sum(row["cells"][cell]["fn_pixels"] for row in cell_rows) == 14
    assert sum(row["oracle_miou_gain"] for row in cell_rows) / 2 == 0.125
    assert "cells" not in cell_rows[0]["cells"][cell]
    print("Model388 aggregation-recovery synthetic smoke: PASS")


if __name__ == "__main__":
    main()
