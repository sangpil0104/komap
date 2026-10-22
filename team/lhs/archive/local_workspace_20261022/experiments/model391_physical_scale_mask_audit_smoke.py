"""Synthetic-only regression smoke for the Model391 train-mask audit."""

from __future__ import annotations

import numpy as np

import model391_physical_scale_mask_audit as audit


def test_boundary_runs() -> None:
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[:, 20:24] = 1
    mask[:, 34:38] = 2
    mask[:, 48:52] = 3
    widths, counts = audit.collect_boundary_widths(mask)
    for class_id in range(4):
        assert counts[class_id] > 0
        assert widths[class_id].size == counts[class_id]
    assert float(np.median(widths[1])) == 4.0
    assert float(np.median(widths[2])) == 4.0
    assert float(np.median(widths[3])) == 4.0


def test_step_adjusted_group_bootstrap_slope() -> None:
    rows = []
    for group_index in range(12):
        for width in (600, 1200, 1800):
            step = (group_index + width // 600) % 4 + 1
            rows.append({
                "group": f"A{group_index}",
                "step": step,
                "width": width,
                "median_run_px": [width * 0.12, width * 0.08,
                                  width * 0.04, width * 0.02],
            })
    audit.BOOTSTRAP_DRAWS = 400
    for class_id in range(4):
        result = audit.bootstrap_slope(rows, class_id)
        assert np.isclose(result["slope"], 1.0, atol=1e-10)
        assert np.allclose(result["group_bootstrap_95ci"], [1.0, 1.0], atol=1e-10)
        assert result["valid_draws"] == 400


def main() -> None:
    test_boundary_runs()
    test_step_adjusted_group_bootstrap_slope()
    print("PASS: class-side boundary run lengths match synthetic stripe geometry.")
    print("PASS: step-adjusted slope and specimen-group bootstrap recover a known linear scale relation.")
    print("Scope: synthetic arrays only; no KoMaP archive or prediction artifacts read.")


if __name__ == "__main__":
    main()
