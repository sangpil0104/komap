#!/usr/bin/env python3
"""Build an exploratory 14-checkpoint Model349 D3 inference directory.

Input is the verified original Model349 release directory. Weight files are
hard-linked when possible (copy fallback), so no original files are changed.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path


B4 = ("model326_hybrid_fullstats_group_oof", "model329_fullstats_width4000_uniform_group_oof")
HIGH = ("restart51_aspp_768_finetune", "model109_aspp_native",
        "restart59_aspp768_seed456", "restart61_pairwise_margin")
REQUIRED = tuple(
    [f"weights/{folder}/fold{fold}/last.pt" for folder in B4 for fold in range(1, 5)]
    + ["weights/model336_hrnet_best/last.pt", "weights/model269_fullstats_1x_extended/best.pt"]
    + [f"weights/{folder}/best.pt" for folder in HIGH]
)


def link_or_copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--script", type=Path, required=True,
                        help="patched infer_lite.py from experiments/model349_compression")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    release, output = args.release.resolve(), args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    if not (release / "src").is_dir():
        raise FileNotFoundError(release / "src")
    manifest = json.loads((release / "provenance/weights_manifest.json").read_text())
    original = {entry["path"]: entry for entry in manifest}
    if len(REQUIRED) != 14 or any(path not in original for path in REQUIRED):
        raise ValueError("Expected exactly 14 known checkpoints from the original manifest")
    missing = [str(release / path) for path in REQUIRED if not (release / path).is_file()]
    if missing:
        raise FileNotFoundError(missing)
    output.mkdir(parents=True)
    shutil.copytree(release / "src", output / "src", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for name in ("environment.yaml", "requirements.lock.txt"):
        if (release / name).is_file():
            shutil.copy2(release / name, output / name)
    for name in ("model349_lite.yaml", "calibration.json"):
        target = output / "configs" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(release / "configs" / name, target)
    shutil.copy2(args.script, output / "infer_lite.py")
    for path in REQUIRED:
        link_or_copy(release / path, output / path)
    # Each B4 checkpoint is interpreted using its fold-local architecture,
    # input-normalization, and target-width settings.
    for folder in B4:
        for fold in range(1, 5):
            relative = Path("weights") / folder / f"fold{fold}" / "args.json"
            if not (release / relative).is_file():
                raise FileNotFoundError(release / relative)
            shutil.copy2(release / relative, output / relative)
    provenance = output / "provenance"
    provenance.mkdir()
    (provenance / "weights_manifest.json").write_text(
        json.dumps([original[path] for path in REQUIRED], indent=2, ensure_ascii=False) + "\n")
    (output / "README.md").write_text(
        "# Model349 D3 14가중치 탐색 추론\n\n"
        "322·347·기초 보조 분기·표현모델181을 생략하고, 고해상도 EMA 경로만 "
        "restart51로 대체합니다. 체크포인트는 14개입니다. 반복 사용한 valid에서 선택된 "
        "탐색 후보이며 독립 일반화 성능은 확인되지 않았습니다.\n\n"
        "실행: `python -B infer_lite.py --input /path/to/images --output /path/to/new_run "
        "--gpus 0`\n\n"
        "정답 마스크가 있으면 `--gt /path/to/masks`를 추가합니다. "
        "점수는 이미지별 4클래스 mIoU 평균이며 결측 클래스 IoU=0입니다.\n\n"
        "검증 20장 재추론 점수와 비교 결과는 생성 실험 기록을 참조하세요. "
        "시험 정답 점수는 확인되지 않았습니다.\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(output), "checkpoint_count": len(REQUIRED),
                      "required_weights": REQUIRED}, indent=2))


if __name__ == "__main__":
    main()
