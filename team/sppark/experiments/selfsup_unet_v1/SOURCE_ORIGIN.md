# 기존 코드 재사용 기록

이 실험은 이전 학습 소스를 변경하지 않습니다. 다음 파일은 새 경로로 복사한 뒤 이 실험 안에서만 사용합니다.

- `sslab/parallel.py`: `experiments/single_unet_v3/singlelab/parallel.py`의 고정 유효 배치·dummy slot 구현 사본.
- `sslab/inference.py`: 같은 이전 실험의 `inference.py` 사본. 원해상도 halo 타일 추론·확률 결합 재사용.
- `sslab/metrics.py`: 같은 이전 실험의 `metrics.py` 사본. 이미지 평균·경계·외곽 지표 재사용.
- `sslab/panels.py`: `experiments/adaptive_threshold_v1/reporting.py`의 `font`, `error_rgb`, `comparison` 함수 사본. palette import만 이 실험의 common으로 연결.

실행 중 기존 실험 디렉터리에서 코드를 import하지 않습니다. 모델·데이터·학습 엔진·실험 orchestration은 이 실험의 별도 소스입니다. 실행 시 각 suite의 `source/`와 `source_hashes.json`에 실제 사용 소스를 보존합니다.
