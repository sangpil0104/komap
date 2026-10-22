# Model349 13가중치 추가 축소 탐색판

이 ZIP은 학습된 가중치와 대회 데이터를 **포함하지 않는 코드 재현 패키지**입니다. 기록 점수를 재현하려면 `provenance/weights_manifest.json`의 13개 체크포인트가 필요합니다. B4 fold별 `args.json` 8개, 추론 소스, 설정, 검증 기록은 포함돼 있습니다.

기존 15가중치 경량판에서 고해상도 기초 경로의 `model123_ema`와 `restart59_aspp768_seed456` 두 체크포인트를 빼고, 해당 밝기 구간을 `restart51`로 대체했습니다. 나머지 모델·TENT·후처리는 유지합니다.

P100 원본 영상 valid 20장 공식 image-wise 4-class mIoU(empty=0)는 **80.0572455%**입니다. 15가중치판 80.0643596% 대비 **−0.0071141%p**로 평균 손실은 작지만 A74 한 장은 **−0.1639561%p** 악화했습니다. test 10장 마스크는 15개판과 9,074,964픽셀 중 13픽셀만 다르지만, test에는 제거한 두 밝기 구간 영상이 없고 정답도 없습니다. 따라서 이 수치로 그 구간의 새 자료 성능을 보증할 수 없습니다. **기본 배포판으로 권장하지 않는 탐색판**입니다.

원본 23가중치 실행 릴리스가 있으면:

```bash
python build_lite13_bundle.py \
  --release /path/to/original_model349_release \
  --script ./infer_lite_13.py \
  --output /path/to/new_model349_lite13_bundle
python verify_rep_free_bundle.py /path/to/new_model349_lite13_bundle --expected-count 13
```

또는 13개 가중치를 manifest의 상대 경로대로 이 ZIP의 `weights/`에 직접 배치하고 `python -B infer_lite_13.py --input /path/to/images --output /path/to/new_run --gpus 0`으로 실행할 수 있습니다. valid 정답이 있으면 `--gt /path/to/masks`를 추가합니다. 의존성은 `environment.yaml`과 `requirements.lock.txt`를 참고하세요. 새로 학습한 가중치는 기록된 SHA-256과 다를 수 있습니다.
