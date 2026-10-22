# lhs 설정 검증 — 2026-09-29

- 원본 commit: `aa970bef8d16b267c8d7bfbffe3ca1b4c4a4813f`.
- 원본 파일 862개 SHA-256 일치. 기존 실험·학습 모델·이전 결과를 수정하지 않음.
- train 70 / valid 20 / test 10 이미지 점검 완료. 마스크 크기, 4색 palette, 불투명도 검사 통과.
- 시편 단위 4-fold 분할 검사 통과. fold 간 OOF 중복 없음, 전체 train 70장 정확히 한 번 포함.
- 의존성 설치 및 `pip check` 통과. 원본 Model338 함수 import 통과. Python 구문 및 shell 구문 검사 통과.
- `train326` dry run 통과. **실제 학습·optimizer step은 실행하지 않음.**
- ImageNet 초기 가중치 로드 완료. 학습된 segmentation 체크포인트는 저장소에 없음.

GPU 0(RTX 4090)에서 무작위 텐서로 확인한 결과:

| 항목 | 결과 |
|---|---|
| 입력 / 출력 | `[2,3,1536,1536]` / `[2,4,1536,1536]` |
| FP16 forward + FP32 CE/Dice + backward | 성공, loss와 gradient 유한 |
| peak allocated memory | 12.344 GiB (optimizer 상태를 만들지 않은 검사) |
| 가장 큰 valid 영상의 3배 확대, D4 없는 추론 | 원본 1024×1440 → 확률 4×1024×1440, 성공 |
| 전체 영상 추론 peak allocated memory | 4.069 GiB |

이 검사는 학습 안정성·최종 성능을 증명하지 않습니다. 가중치 업데이트 없이 입출력과 메모리를 확인했습니다.
기록은 `reports/model326_tensor_check.json`과 `.log`입니다.

`tests/test_integration.py`의 8개 검사 통과:

1. 없는 클래스=0인 image mIoU 예제의 60% 계산.
2. 원본 학습 Dataset, worker seed, Dice, augmentation의 AST 동일성.
3. 실제 학습 영상에서 1536 crop 생성과 전체 영상 통계 사용.
4. GPU 0–5 범위와 클래스 번호 변환.
5. 확률 파일 키 중복·누락·잘못된 확률 거부.
6. 2×2 패널 위치, 오답의 정답 색, 정답의 흰색 표시.
7. 합성 20장으로 Model338 최종 경로 실행, 순서를 뒤집은 cache 대응, 원본 함수와 동일 예측, 보고서·마스크 생성과 기존 결과 보존.
8. 가짜 프로세스로 fold GPU 배치·완료 후 평가 연결·코드 snapshot·중복 실행 잠금 검사. 학습 프로세스는 생성하지 않음.

Model338의 실제 파일 4개가 없어 **실제 예측 및 원본 기록의 79.339963% 점수는 검증하지 못했습니다**.
합성 테스트 결과는 임시 디렉터리에서만 생성하고 제거했으며 프로젝트 `results/`에는 실험 결과로 기록하지 않았습니다.
누락 목록과 입력 형식: [MISSING_ARTIFACTS.md](MISSING_ARTIFACTS.md).
