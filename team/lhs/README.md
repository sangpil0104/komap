# KoMaP 4상 미세조직 분할: 최종 제출 모델 (lite13)

고탄성 알루미늄 합금 광학 현미경 이미지에서 Al₃Ni, Primary Si, Al, Eutectic Si 네 상을 픽셀 단위로 나누는 모델입니다.
대회 예선에는 통과하지 못했습니다([원인 분석](docs/예선_탈락_분석_20261022.md)). 최종 제출본은 가중치 13개(R 7개 + E 5개 + F 1개)를 결합한 **lite13**이며, 검증용(valid) 20장 이미지별 4상 mIoU는 **80.779378%**입니다.

- 지표: 이미지마다 네 상의 IoU를 평균한 뒤 20장 평균, 정답과 예측 모두 없는 상은 IoU 0
- 결합 비율과 후처리 설정은 검증용 20장에서 골랐습니다. 평가용(test)은 정답이 없어 점수를 계산하지 않았습니다.

## 저장소 구성

| 경로 | 내용 |
|---|---|
| [`final_lite13/`](final_lite13/) | 최종 제출 코드: 예측·학습 코드, 실행 파일, 개발 환경, README, 모델링 결과서 PDF |
| [릴리스 `submission-final-20261015`](https://github.com/CNUKoMaP/lhs/releases/tag/submission-final-20261015) | 실제 제출 파일: `A616277.zip`(코드 + valid·test 데이터 + 평가용 예측 마스크), `A616277_가중치.zip`(가중치 13개) |
| [`docs/예선_탈락_분석_20261022.md`](docs/예선_탈락_분석_20261022.md) | 예선 탈락 원인 분석과 교훈 |
| [`archive/`](archive/) | 최종판 이전의 실험 기록 전부: 실험 번들, 실험 코드, 결과, 문서, 로컬 작업 폴더 백업. 구성은 [`archive/README.md`](archive/README.md)에 있습니다. |

저장소에는 대회 데이터, 학습한 가중치, 예측 마스크를 커밋하지 않습니다. 데이터(valid·test)와 가중치는 위 릴리스 첨부 파일에 있습니다.

## 실행 방법

```bash
# 릴리스에서 A616277.zip, A616277_가중치.zip 을 내려받아 같은 폴더에 푼 뒤
cd A616277/1_소스코드_실행파일_개발환경/komap_model
conda env create -f environment.yml && conda activate komap
bash run_predict.sh ../../../A616277_가중치/weights data/test/images out/test 0     # 평가용 마스크 생성
bash run_predict.sh ../../../A616277_가중치/weights data/valid/images out/valid 0   # 검증용 예측
python tools/score_masks.py --labels out/valid/fusion/predicted_labels --gt data/valid/masks --output out/valid/metrics.json
```

- 재학습(`run_retrain.sh`)에는 대회 train 70장이 필요합니다. 제출 용량 제한으로 빠져 있으므로 `data/train/images`, `data/train/masks`에 넣은 뒤 실행합니다.
- 가중치 구성, 모델별 학습 코드·시드, 재현 범위는 [`final_lite13/1_소스코드_실행파일_개발환경/komap_model/README.md`](final_lite13/1_소스코드_실행파일_개발환경/komap_model/README.md)에 있습니다.

## 확인한 결과

- 제출 가중치 13개로 원본 이미지부터 예측: 검증용 mIoU 80.779378%, 평가용 마스크는 제출 마스크와 픽셀 단위로 같음(9,074,964픽셀 중 차이 0)
- 학습용 이미지 4장으로 재학습 흐름(SMOKE) 점검: 13개 학습 → F 학습 → 채점 → 평가용 마스크 생성까지 오류 없음
- 재학습하면 시드·GPU 연산 편차로 가중치가 같아지지 않아 검증용 점수가 80.779378%와 다를 수 있습니다.
