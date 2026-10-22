#!/usr/bin/env python3
"""Read-only locator for Model374/376/377 grouped-OOF archives on Colab."""

from __future__ import annotations

import json
from pathlib import Path


NAMES = {"valid_probabilities.npz", "valid_summary.json", "complete.json"}
ROOTS = [
    Path("/content/komap_model1/outputs/model374_hrnet_group_oof"),
    Path("/content/komap_model1/outputs/model376_boundary_ce_group_oof"),
    Path("/content/komap_model1/outputs/model377_directional_group_oof"),
    Path("/content/drive/MyDrive/KoMaP/model374_hrnet_group_oof"),
    Path("/content/drive/MyDrive/KoMaP/model376_boundary_ce_group_oof"),
    Path("/content/drive/MyDrive/KoMaP/model377_directional_group_oof"),
]


def main() -> None:
    report = {}
    for root in ROOTS:
        files = []
        if root.is_dir():
            for path in root.rglob("*"):
                if path.is_file() and path.name in NAMES:
                    files.append({"path": str(path), "bytes": path.stat().st_size})
        report[str(root)] = {"exists": root.is_dir(), "files": sorted(files, key=lambda x: x["path"])}
    code_dir = Path("/content/komap_model1/model377_directional/experiments")
    report["model379_preflight"] = {
        "evaluator_exists": (code_dir / "model379_uniform_oof_ensemble.py").exists(),
        "output_exists": Path(
            "/content/komap_model1/outputs/model379_uniform_oof_ensemble/results.json"
        ).exists(),
        "all_required_archives_exist": all(
            (Path("/content/komap_model1/outputs") / folder / f"fold{fold}_oof/valid_probabilities.npz").is_file()
            for folder in (
                "model374_hrnet_group_oof",
                "model376_boundary_ce_group_oof",
                "model377_directional_group_oof",
            )
            for fold in range(1, 5)
        ),
    }
    report["model381_preflight"] = {
        "evaluator_exists": (code_dir / "model381_boundary_error_diagnostic.py").exists(),
        "output_exists": Path(
            "/content/komap_model1/outputs/model381_boundary_error_diagnostic/results.json"
        ).exists(),
        "all_required_archives_exist": all(
            (Path("/content/komap_model1/outputs") / folder / f"fold{fold}_oof/valid_probabilities.npz").is_file()
            for folder in (
                "model374_hrnet_group_oof",
                "model376_boundary_ce_group_oof",
                "model377_directional_group_oof",
            )
            for fold in range(1, 5)
        ),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
