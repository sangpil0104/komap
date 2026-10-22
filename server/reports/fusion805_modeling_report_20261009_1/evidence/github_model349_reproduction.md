# Model349 학습·route 재현 안내

이 문서는 **코드만 포함하는** 재현 묶음 안내다. 사용자는 원본 라벨 데이터 `train.zip`(영상/마스크 70쌍)과 `valid.zip`(20쌍)을 별도로 준비해야 한다. 파이프라인은 자체 체크포인트를 학습하고 validation 확률을 생성한 뒤 Model349까지의 클래스별 route를 다시 구성한다. 기존 학습 체크포인트, Model183 cache, validation 확률 파일은 필요하지 않다.

## 실행 방법

Linux/CUDA 환경에서 ZIP을 풀고 두 데이터 archive를 한 디렉터리에 둔 뒤, 압축 해제한 저장소 루트에서 실행한다.

```bash
bash experiments/setup_model349_environment.sh
KOMAP_DATA_DIR=/absolute/path/to/archives bash experiments/run_model349_pipeline.sh preflight
KOMAP_DATA_DIR=/absolute/path/to/archives bash experiments/run_model349_pipeline.sh train
```

`KOMAP_DATA_DIR`에는 `train.zip`과 `valid.zip`이 있어야 한다(기본값 `./datasets`). 기존 호환 가상환경을 쓰려면 `KOMAP_PYTHON`을 지정한다. setup 스크립트는 P100에서 사용한 Python 패키지 버전을 고정한다. `preflight`는 GPU 학습 없이 데이터 개수와 4개 grouped-CV 분할을 검사한다.

ImageNet encoder 초기 가중치가 캐시에 없으면 해당 라이브러리가 내려받으므로, 실행 호스트는 공개 가중치에 접근할 수 있어야 한다. 이는 공개된 **초기 가중치**이며 대회 데이터로 학습한 체크포인트가 아니다.

매 실행은 새로운 `runs/model349_train.*` 디렉터리를 만들며 이전 run을 덮어쓰지 않는다. 해당 디렉터리의 `outputs/`에는 새 체크포인트·예측 archive·route cache·결과 JSON을, `logs/`에는 단계별 로그를 저장한다. 최종 결과는 `outputs/model349_model347_into_346_route/results.json`에 기록되며 empty union을 0으로 둔 validation image-wise 4-class mIoU를 출력한다. `KOMAP_RUN_ROOT`로 새 출력 경로를 지정할 수 있고 기존 경로면 덮어쓰지 않도록 거부한다. `KOMAP_WORKERS`는 CPU route 탐색 worker 수를 조절한다.

## 학습되는 구성

1. ResNet34/U-Net++ 기반과 ASPP 계열: Model6 → 62 → 97 → 98 → 109.
2. Model183 base-route 확률원: restart51/59/61, Model123 EMA, Model181 representation. Route47 cache와 Model183 validation 항목은 새 체크포인트에서 다시 만든다.
3. Native/full-stat 계열: Model109 → 194 → 195 → 239 → 242 → 257 → 269. Model220 TENT는 추론 때 Model195를 적응시킨다.
4. EfficientNet-B4 grouped-CV branch Model322/326/329(각 4 folds, fold당 16,000 updates), HRNet-W48/U-Net++ Model336 및 weighted-CE fine-tune Model347.
5. CPU 클래스별 route 탐색: Model335 → 338 → 339 → 341 → 346 → 349.

포함된 `model303_progress.json`은 과거 grouped-CV 분할 manifest일 뿐, 체크포인트나 validation 예측은 담고 있지 않다. 파이프라인은 checkpoint와 route를 고르는 데 `valid.zip`을 사용하므로 보고 점수는 **validation 점수**이며, 손대지 않은 test 일반화 추정치가 아니다.

기록된 Model349 validation 점수는 공식 image-wise 4-class, empty=0 기준 `79.9095189%`다. 재학습은 stochastic하고 checkpoint/route가 달라질 수 있으므로 이 묶음은 **절차 재현**을 목적으로 하며 bit 단위 체크포인트나 정확히 같은 점수를 보장하지 않는다. 기존 Model194의 1024px 3단계 시도는 P100 메모리를 초과했으므로 이 runner는 실패한 단계를 건너뛰고 1단계 최고 체크포인트부터 이어 간다.

학습 비용이 크며 특히 B4 12개 fold가 많은 연산을 요구한다. 여유 있는 GPU에서 로그를 확인하며 실행하고, 학과 GPU 서버에서는 서버 사용 규정을 따른다. 이 ZIP/저장소에는 test mask, 데이터 archive, 기존 학습 checkpoint, cached validation prediction을 업로드하지 않는다.
