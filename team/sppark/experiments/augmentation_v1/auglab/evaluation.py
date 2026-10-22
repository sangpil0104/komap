from __future__ import annotations

import json
import numpy as np
import pandas as pd

from alphase.cli.common import load_trained
from alphase.data.audit import verify_manifest
from alphase.evaluation.evaluate import evaluate
from alphase.utils import code_hash, json_hash, sha256, write_json
from auglab.workspace import BASELINE, OUTPUTS, output_path, read_json, run_dir


def evaluate_run(name, seed=2026, split="official", smoke=False, d4=False):
    checkpoint = run_dir(name, seed, split, smoke) / "best.pt"
    model, cfg, device, state, selected = load_trained(checkpoint, [f"inference.d4={str(d4).lower()}"])
    current_data = verify_manifest(cfg["data"]["manifest"])
    if current_data != state["dataset_hash"]:
        raise ValueError("Dataset changed since training")
    ids = state["validation_ids"]
    if not ids or set(ids) & set(state["train_ids"]):
        raise ValueError("Independent validation IDs are required; all90 has no held-out validation")
    table = pd.read_csv(cfg["data"]["manifest"]).fillna("")
    rows = table[table.image_id.isin(ids)].to_dict("records")
    if len(rows) != len(ids):
        raise ValueError("Missing validation images")
    out = output_path("smoke_evaluations" if smoke else "evaluations", name, f"seed{seed}", split, "d4" if d4 else "plain")
    provenance = dict(experiment=name, seed=seed, split=split, smoke_test=smoke,
                      checkpoint_sha256=sha256(checkpoint), selected_weights=selected,
                      dataset_hash=current_data, frozen_code_sha256=code_hash(),
                      inference=cfg["inference"], validation_ids=ids, train_ids=state["train_ids"])
    marker = out / "provenance.json"
    if out.exists() and any(out.iterdir()):
        if marker.is_file() and read_json(marker) == provenance and (out / "metrics_summary.json").is_file():
            print(f"Reusing evaluation: {out}")
            return read_json(out / "metrics_summary.json")
        raise FileExistsError(f"Incomplete or incompatible evaluation exists: {out}. Preserve/move it before retrying.")
    summary = evaluate(model, rows, cfg, device, output=out,
                       label=f"{name} seed={seed} {'D4' if d4 else 'plain'}")
    write_json(marker, provenance)
    print(f"{name} seed={seed} {'D4' if d4 else 'plain'} mIoU={summary['mIoU_exclude']:.4f}%", flush=True)
    return summary


def compare():
    candidates = [("baseline", 2026, "plain", BASELINE / "run/evaluation/per_image_metrics.csv"),
                  ("baseline", 2026, "d4", BASELINE / "run/evaluation_d4_review/per_image_metrics.csv")]
    for path in sorted(output_path("evaluations").glob("*/seed*/official/*/per_image_metrics.csv")):
        if not (path.parent / "provenance.json").is_file():
            continue
        p = read_json(path.parent / "provenance.json")
        if p["smoke_test"]:
            continue
        candidates.append((p["experiment"], p["seed"], path.parent.name, path))
    refs = {mode: pd.read_csv(path).set_index("image_id") for _, _, mode, path in candidates[:2]}
    result = []
    for name, seed, mode, path in candidates:
        f = pd.read_csv(path).set_index("image_id")
        ref = refs[mode]
        if set(f.index) != set(ref.index) or len(f) != 20:
            raise ValueError("Comparison requires exactly the same 20 validation images")
        f = f.loc[ref.index]
        present = np.array([np.array(json.loads(v)).sum(axis=1) for v in f.confusion_matrix]) > 0
        common = present.all(axis=1)
        delta = f.mIoU_exclude - ref.mIoU_exclude
        row = dict(experiment=name, seed=seed, inference=mode, miou=float(f.mIoU_exclude.mean()),
                   delta_matched_baseline_pp=float(delta.mean()), improved_images=int((delta > 1e-8).sum()),
                   miou_empty_one=float(f.mIoU_one.mean()), miou_empty_zero=float(f.mIoU_zero.mean()),
                   all_classes_present_images=int(common.sum()),
                   delta_common_images_pp=float(delta.to_numpy()[common].mean()), source=str(path))
        for c in range(4):
            row[f"iou_{c}"] = float(f[f"IoU_{c}"].mean())
            row[f"iou_{c}_image_count"] = int(f[f"IoU_{c}"].notna().sum())
        result.append(row)
    frame = pd.DataFrame(result).sort_values("miou", ascending=False)
    folder = output_path("reports")
    folder.mkdir(parents=True, exist_ok=True)
    frame.to_csv(folder / "comparison.csv", index=False)
    columns = ["experiment", "seed", "inference", "miou", "delta_matched_baseline_pp", "improved_images", "delta_common_images_pp"]
    print(frame[columns].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    (folder / "comparison.txt").write_text(
        "공식 검증 20장. 변화량은 동일 추론(plain/D4)의 기존 seed2026 모델 대비.\n"
        "common_images는 정답에 모든 클래스가 있는 이미지의 민감도 분석이며 공식 지표를 대체하지 않음.\n"
        "다른 seed 후보는 seed 효과가 섞임; 동일 seed 대조 실험/group CV로 확인.\n\n" + frame[columns].to_string(index=False))
    return frame
