#!/usr/bin/env python3
"""Reassemble exact Model374 fold-1/2 OOF archives transferred in chunks."""

from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path


ROOT = Path("/content/komap_model1")
SPECS = {
    1: (85_308_082, "2176c5c2c2f5e230a911e246cdf894e74755f2ee81cdab8bb547b600adee457d"),
    2: (115_154_110, "14362486e73c8e7af53f8eecae45147573b21c2c897eb8b217ea1e4caeb33614"),
}


def chunk_paths(fold: int) -> list[Path]:
    if fold == 1:
        return [Path("/content/komap_model377_f1_000")] + [
            Path(f"/content/komap_f1_{index:03d}") for index in range(1, 11)
        ]
    return [Path(f"/content/komap_f2_{index:03d}") for index in range(14)]


def digest(path: Path) -> tuple[int, str]:
    value = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            size += len(block)
            value.update(block)
    return size, value.hexdigest()


def main() -> None:
    reports = []
    for fold, (expected_size, expected_sha256) in SPECS.items():
        target = ROOT / f"outputs/model374_hrnet_group_oof/fold{fold}_oof/valid_probabilities.npz"
        if not (ROOT / "outputs/model374_hrnet_group_oof").is_dir():
            raise FileNotFoundError("Model374 OOF output root is missing")
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            observed = digest(target)
            if observed != (expected_size, expected_sha256):
                raise FileExistsError(f"refusing to overwrite nonmatching existing target: {target} {observed}")
            reports.append({"fold": fold, "status": "already_present_verified", "bytes": observed[0], "sha256": observed[1]})
            continue

        chunks = chunk_paths(fold)
        if any(not path.is_file() for path in chunks):
            missing = [str(path) for path in chunks if not path.is_file()]
            raise FileNotFoundError(f"missing uploaded chunks for fold {fold}: {missing}")

        fd, temporary_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
        temporary = Path(temporary_name)
        value = hashlib.sha256()
        size = 0
        try:
            with os.fdopen(fd, "wb") as output:
                for chunk in chunks:
                    with chunk.open("rb") as source:
                        while block := source.read(8 * 1024 * 1024):
                            output.write(block)
                            size += len(block)
                            value.update(block)
                output.flush()
                os.fsync(output.fileno())
            sha256 = value.hexdigest()
            if (size, sha256) != (expected_size, expected_sha256):
                raise ValueError(f"fold {fold} transfer checksum mismatch: {size} {sha256}")
            os.link(temporary, target)
            reports.append({"fold": fold, "status": "reassembled_verified", "bytes": size, "sha256": sha256})
        finally:
            temporary.unlink(missing_ok=True)

    print(reports, flush=True)


if __name__ == "__main__":
    main()
