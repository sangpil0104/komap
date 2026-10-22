"""Print valid scores of finished runs, grouped by config, with seed mean and spread."""
import argparse
import glob
import json
import os
import re
from collections import defaultdict

import numpy as np

p = argparse.ArgumentParser()
p.add_argument("--runs", nargs="+", required=True, help="run directories (each with args.json + valid.json)")
args = p.parse_args()

rows = []
for d in args.runs:
    for f in sorted(glob.glob(os.path.join(d, "valid.json"))):
        run = os.path.basename(os.path.dirname(f))
        v = json.load(open(f))
        a = json.load(open(os.path.join(os.path.dirname(f), "args.json")))
        key = re.sub(r"_?seed\d+$", "", run) or run
        rows.append((key, a.get("seed", 0), v["miou_tta"], v["per_class_iou"]))

if not rows:
    raise SystemExit("no finished runs found (valid.json missing)")

print(f"{'config':<24}{'seed':>5}{'mIoU':>8}   per-class [Al, Al3Ni, PrimSi, EutSi]")
groups = defaultdict(list)
for key, seed, miou, per in sorted(rows):
    print(f"{key:<24}{seed:>5}{miou:>8.2f}   {per}")
    groups[key].append(miou)

print(f"\n{'config':<24}{'n':>3}{'mean':>8}{'sd':>7}{'min':>8}{'max':>8}")
for key, vals in sorted(groups.items()):
    v = np.array(vals)
    sd = v.std(ddof=1) if len(v) > 1 else float("nan")
    print(f"{key:<24}{len(v):>3}{v.mean():>8.2f}{sd:>7.2f}{v.min():>8.2f}{v.max():>8.2f}")

if len(groups) > 1:
    print("\nA config only beats another when the gap survives this spread -- with 2-3 seeds and "
          "sd near 1.0, differences below ~1 point are not decidable.")
