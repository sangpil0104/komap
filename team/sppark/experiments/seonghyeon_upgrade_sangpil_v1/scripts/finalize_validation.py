"""Verify preservation and summarize measured results without training."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import json
from tqdm import tqdm
from alphase.io import ROOT, sha256, json_write, jsonl_read, digest
from alphase.config import load_config, inference_hash
from alphase.evaluation import write_csv

audit = json.loads((ROOT / "reports/audit.json").read_text())
items = [(ROOT / p, h) for p, h in audit["input_hashes"].items()]
items += [(Path(audit["source"]) / p, h) for p, h in audit["source_hashes"].items()]
checkpoints = jsonl_read(ROOT / "manifests/checkpoints.jsonl")
items += [(Path(r["checkpoint"]), r["checkpoint_sha256"]) for r in checkpoints]
for r in jsonl_read(ROOT / "manifests/dataset.jsonl"):
    for kind in ["image", "mask"]:
        if r.get(kind + "_path"):
            items.append((Path(r[kind + "_path"]), r[kind + "_sha256"]))
changed = [
    str(path)
    for path, h in tqdm(items, desc="verify preserved inputs/source/data/checkpoints")
    if sha256(path) != h
]
if changed:
    raise RuntimeError(f"Preservation check failed: {changed}")
json_write(
    ROOT / "reports/preservation.json",
    {
        "checked_files": len(items),
        "input_files": len(audit["input_hashes"]),
        "legacy_source_files": len(audit["source_hashes"]),
        "historical_checkpoints": len(checkpoints),
        "changed_files": changed,
    },
)
repro = json.loads((ROOT / "reports/seed32_reproduction.json").read_text())
smoke = json.loads((ROOT / "reports/smoke_candidate_unvalidated_full.json").read_text())
summary = json.loads((ROOT / "reports/seed32_baseline/summary.json").read_text())
a71 = next(r for r in repro["per_image"] if r["image_id"] == "A71_step2_sample1")
pytest = (ROOT / "reports/pytest.log").read_text().strip().splitlines()[-1]
assert "failed" not in pytest and "passed" in pytest
assert abs(repro["historical_score_difference"]) < 1e-12
assert summary["val_image_miou"] == repro["val_image_miou"]
assert a71["max_abs_probability_difference"] == 0 and a71["argmax_mismatch_pixels"] == 0
cfg = load_config("configs/resolved_legacy.yaml")
checkpoint = checkpoints[32]
write_csv(
    ROOT / "reports/seed32_baseline/experiments.csv",
    [
        {
            "experiment_id": "E00_reproduced",
            "seed": 32,
            "parent_checkpoint": checkpoint["checkpoint"],
            "training_recipe": "historical_fixed_last_16000; no new training",
            "train_ids_hash": checkpoint["training_ids_hash"],
            "valid_ids_hash": audit["valid_ids_hash"],
            "geometry_profile": "legacy_scaled_crop",
            "branch_flags": cfg["model"]["new_branches"],
            "sampler_config_hash": digest(cfg["sampling"]),
            "loss_config_hash": digest(cfg["loss"]),
            "optimizer_updates": 16000,
            "new_optimizer_updates": 0,
            "selection_metric": "image_miou",
            "selection_policy": "fixed_last_not_validation_selected",
            "selection_protocol_hash": digest(cfg["evaluation"]),
            "selected_step": 16000,
            "checkpoint_hash": checkpoint["checkpoint_sha256"],
            "inference_config_hash": inference_hash(cfg),
            "val_image_miou": summary["val_image_miou"],
            "val_global_miou": summary["val_global_miou"],
            "mean_iou_Al3Ni": summary["mean_class_iou"][1],
            "mean_iou_EutecticSi": summary["mean_class_iou"][3],
            "A71_image_miou_diagnostic": a71["image_miou"],
            "status": "RUN_COMPLETED",
        }
    ],
)
(ROOT / "VALIDATION_REPORT.md").write_text(
    f"""# 검증 보고서

실제 데이터 학습은 시작하지 않았습니다. 새 모델의 성능 향상을 주장하지 않습니다.

| 검사 | 실측 결과 |
|---|---|
| 데이터/팔레트/짝 검사 | train70 / valid20 / test10 통과 |
| 기존100 checkpoint | SHA-256 기록, 최종 보존 검사 통과 |
| 기존 seed32 strict load | 통과 |
| valid20 재추론 image mIoU | {repro['val_image_miou']*100:.10f}% |
| 기존 점수와 차이 | {repro['historical_score_difference']:.3e} (0–1 단위) |
| A71 기존/새 어댑터 확률 최대 차이 | {a71['max_abs_probability_difference']} |
| A71 기존/새 어댑터 argmax 불일치 | {a71['argmax_mismatch_pixels']} pixels |
| A71 image mIoU | {a71['image_miou']*100:.10f}% |
| CPU 수학·통합 검사 | {pytest} |
| 실제 B4 합성 입력 크기 | batch{smoke['batch_size']}, detail{smoke['detail_size']}, context{smoke['context_size']} |
| 새 head0 초기 출력 차이 | {smoke['zero_residual_max_difference']} |
| GPU forward/backward 최대 메모리 | {smoke['peak_memory_GiB']:.4f} GiB |
| 합성 GPU smoke optimizer updates | 0 |
| 실제 seed32 확률 캐시/진단 | valid20 완료 |
| baseline test PNG | test10, 원본 H/W·4색 검사 통과 |
| 원본 보존 최종 검사 | {len(items)} files, 변경0 |

CPU 검사에는 A71 fixture, 네 색상/unknown/ignore, empty union, image/global 평균 차이, 중복 이미지 ID 처리, D4, native 좌표 ramp/gradient, tile coverage, cache 손상/재개, K1/동일100-member 평균, train-only sampler·online hard refresh·resume, 실제 B4 분기 on/off·zero-init·BN freeze·native gradient, refiner zero-init, 합성 training loop의 중단/재개 일치, 합성 OOF teacher registry→cache→deployment→refiner 입력 연결과 오염 차단을 포함합니다.

GPU smoke는 새 texture/context/native loss를 모두 켠 실제 B4 구조에서 실행했습니다. 첫 backward에서 마지막 head에 gradient가 생기고, 테스트용 head를 메모리에서 미세하게 바꾼 다음 backward에서 texture/context에도 finite gradient가 전달됐습니다. 이 과정은 합성 입력이며 optimizer 업데이트나 가중치 저장을 하지 않았습니다. 학습 optimizer 상태와 실제 데이터 증강을 포함한 장시간 최고 메모리 측정은 아닙니다.

합성 resume 테스트는 작은 모델/가짜 PNG만 사용했습니다. 실제 프로젝트 데이터 학습을 실행한 것으로 보고하지 않습니다. 실제 train 패치 overfit 명령은 README에 준비했고 사용자가 시작하도록 남겨뒀습니다.

아직 실행하지 않은 항목: 새 모델4000/16000 update 학습, 실제 train-patch overfit, 전체100-model 앙상블 sweep, 실제 OOF teacher/refiner 학습, outer-fold 평가. 실제 OOF/refiner 가중치는 아직 없으며 학습 후 생성됩니다. 기본 test PNG는 기존 seed32 기준선입니다.

기록: `reports/pytest.log`, `reports/seed32_reproduction.json`, `reports/smoke_candidate_unvalidated_full.json`, `reports/seed32_baseline/`, `reports/preservation.json`, `outputs/submission/validation.json`.
"""
)
progress = (ROOT / "reports/progress.md").read_text()
progress = progress.replace(
    "Current status: IMPLEMENTED; contract/integration verification in progress.",
    "Current status: TESTED; implementation and requested pre-training preparation complete.",
)
progress = progress.replace(
    "| P6 | IMPLEMENTED | strict registry+ancestry validator,5-fold teacher orchestration,8-channel refiner; synthetic workflow verification in progress; real OOF/refiner training unrun |",
    "| P6 | TESTED | strict registry+ancestry validator,5-fold orchestration,8-channel refiner; synthetic full workflow passed; real OOF/refiner training unrun |",
)
progress = progress.replace(
    "Next action: finish final verification/documentation, then user may run `./run.sh launch --execute`.",
    "Next action: user may start the12 paired experiments on GPUs0–5 with `./run.sh launch --execute`. No training process is running.\n\nFinal test result: "
    + pytest
    + ". Final preservation check: "
    + str(len(items))
    + " files unchanged. See ../VALIDATION_REPORT.md.",
)
(ROOT / "reports/progress.md").write_text(progress)
p = ROOT / "reports/repo_audit.md"
p.write_text(
    p.read_text().replace(
        "Reproduction still to run.",
        "Full valid20 reproduction passed; see seed32_reproduction.json.",
    )
)
print(
    pytest, "; preservation:", len(items), "unchanged; seed32:", repro["val_image_miou"]
)
