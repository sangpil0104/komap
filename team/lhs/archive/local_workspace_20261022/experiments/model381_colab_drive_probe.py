#!/usr/bin/env python3
"""Read-only check of whether the authorized KoMaP Drive path is mounted."""

from __future__ import annotations

import os
from pathlib import Path


def main() -> None:
    drive = Path("/content/drive")
    mydrive = drive / "MyDrive"
    komap = mydrive / "KoMaP"
    model374 = Path("/content/komap_model1/outputs/model374_hrnet_group_oof")
    checks = {
        "drive_path_exists": drive.exists(),
        "drive_is_mountpoint": os.path.ismount(drive),
        "mydrive_exists": mydrive.is_dir(),
        "komap_project_dir_exists": komap.is_dir(),
        "model374_output_tree_exists": model374.is_dir(),
        "model374_fold34_oof_archives_exist": all(
            (model374 / f"fold{fold}_oof" / "valid_probabilities.npz").is_file()
            for fold in (3, 4)
        ),
    }
    print(checks)


if __name__ == "__main__":
    main()
