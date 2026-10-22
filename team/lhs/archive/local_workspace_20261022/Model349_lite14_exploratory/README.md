# Model349 14가중치 탐색판 재현 코드

이 ZIP은 **코드·설정·체크포인트 manifest·fold 설정**만 포함합니다. 학습된 가중치와 대회 데이터는 포함하지 않으므로 ZIP만으로 즉시 추론할 수 없습니다. 기존 Model349 실행 릴리스에서 가중치를 가져오거나 별도 학습으로 생성해야 합니다. 기록된 점수를 재현하려면 `provenance/weights_manifest.json`의 원본 체크포인트 14개와 동일한 파일이 필요합니다.

## 15가중치판과 차이

기존 15가중치 경량판의 고해상도 경로에서, 평균 밝기가 `161.3–162.0` 또는 `155.2–155.7`인 영상에 선택되던 `model123_ema`를 `restart51`로 대체합니다. 다른 B4 4-fold 두 세트, HRNet336, native269, 고해상도 TENT, 고정 라우팅과 후처리는 동일합니다. 따라서 학습 체크포인트가 **15개에서 14개**로 줄어듭니다.

원본 영상에서 재추론한 valid 20장의 image-wise 4-class mIoU(empty=0)는 **80.0650399%**였습니다. 같은 실행 환경의 15개판은 80.0643596%, 원형 Model349 23개판은 80.0683465%입니다. 15개판보다 `+0.0006802%p`이고, 최악의 개별 영상 변화는 `−0.01608%p`였습니다. test 10장 마스크는 15개판과 9,074,964픽셀 중 11픽셀만 달랐습니다. **다만 제거 후보를 반복 사용한 valid에서 골랐고, test 10장에는 EMA 밝기 구간 영상이 없어 독립 일반화 개선은 입증하지 못했습니다.** test에는 정답이 없어 test mIoU를 계산할 수 없습니다.

## 실행

원본 23가중치 실행 릴리스가 있으면 다음 명령으로 14가중치 독립 묶음을 만듭니다. 이 패키지의 `weights/`에는 B4 fold별 `args.json` 8개만 들어 있고, `.pt`는 들어 있지 않습니다.

```bash
python build_lite14_bundle.py \
  --release /path/to/original_model349_release \
  --script ./infer_lite_14.py \
  --output /path/to/new_model349_lite14_bundle
python verify_rep_free_bundle.py /path/to/new_model349_lite14_bundle --expected-count 14
```

이미 14개 체크포인트를 따로 갖고 있다면 manifest의 상대 경로대로 이 ZIP의 `weights/`에 배치하고, `python -B infer_lite_14.py --input /path/to/images --output /path/to/new_run --gpus 0`으로 직접 실행해도 됩니다. 정답 마스크가 있으면 `--gt /path/to/masks`를 추가합니다. CUDA 환경 의존성은 `environment.yaml`과 `requirements.lock.txt`를 참고하세요. 새로운 학습 체크포인트는 manifest의 SHA-256이 다르므로 검증 도구가 실패하는 것이 정상입니다.

`lite14_valid20_metrics.json`과 `lite14_valid20_completion.json`은 기록된 단회 검증 결과입니다. 14개판은 별도 탐색 후보로 보관하며, 독립 자료가 생기기 전에는 15개판을 보수적인 기본 선택으로 둡니다.
