#!/usr/bin/env python3
"""Convert Model183 cache to minimal Model349 pre-auxiliary entries.

The converted cache verifies downstream score parity without storing the
auxiliary representation model's probability map. Future raw-image inference
must compute the pre-auxiliary mixed probability directly, before the 0.40
Primary-Si blend, to avoid running that model in the first place.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
for path in (HERE / "historical_scripts", HERE):
    if path.is_dir() and str(path) not in sys.path:
        sys.path.insert(0, str(path))

from model187_schedule_branch_blend import undo_model181_primary_blend  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    with args.source.open("rb") as stream:
        old = pickle.load(stream)
    if len(old) != 20:
        raise ValueError("expected 20 cached validation entries")
    fields = ("z", "q", "prior_z_stats", "prior_q_stats", "params", "target")
    compact = []
    for index, item in enumerate(old):
        if not all(field in item for field in fields) or not all(
                field in item for field in ("mixed", "extra")):
            raise ValueError(f"entry {index} missing required fields")
        preaux = undo_model181_primary_blend(item)
        if not np.allclose(preaux.sum(axis=0), 1.0, atol=1e-5):
            raise ValueError(f"entry {index} preaux is not normalized")
        compact.append({field: item[field] for field in fields} | {"preaux": preaux})
        print(f"converted {index+1}/{len(old)}", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as stream:
        pickle.dump(compact, stream, protocol=pickle.HIGHEST_PROTOCOL)
    summary = {
        "source": str(args.source),
        "output": str(args.output),
        "entries": len(compact),
        "removed_fields": ["extra", "mixed", "branches", "pair", "base_prior",
                           "delta", "log_mixed", "smooth_components"],
        "note": "Cache conversion alone is not raw-image inference; direct preaux construction is needed to avoid auxiliary network execution",
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
