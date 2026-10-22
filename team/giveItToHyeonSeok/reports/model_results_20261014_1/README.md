# Model349 lite13 retuned — 2026-10-14 학습 모델

전체 앙상블 valid20 image mIoU **80.471931%**. R7 + E5 + F1 = **13개 가중치**입니다.
선행/OOF를 포함한 33개 학습 작업에서 생성했으며 이 패키지는 최종 추론에 필요한 13개를 포함합니다.

| 데이터 | 전체 추론시간 | 장당 평균 |
|---|---:|---:|
| valid 20장 | 407.561초 | 20.378초 |
| test 10장 | 223.297초 | 22.330초 |

RTX 4090 GPU 3~6에서 R 구성요소를 분배하고 E는 GPU 3, F는 CPU에서 실행했습니다.
전체 시간은 프로세스 준비·모델 로딩·전처리·R(TTA/TENT)/E/F·후처리·출력 저장·임시 정리를 포함합니다.
학습, 정답 채점과 보고서 생성은 제외했습니다. 별도 warm-up 없이 split마다 1회 측정했습니다.
장당 값은 전체 시간/장수이며 단일 이미지 지연시간이 아닙니다.

## 실행

기존 `komap` conda 환경을 활성화하세요. 검증 환경은 Python 3.11.16,
PyTorch 2.7.1+cu128 / torchvision 0.22.1+cu128 / SMP 0.5.0 / timm 1.0.29입니다.
`results/environment.freeze.txt`는 이번 실험 환경입니다. `code/environment.yml`은 upstream 환경(Pytorch 2.8) 기록입니다.

```bash
conda activate komap
python predict_portable.py --verify-only
python predict_portable.py --images /path/to/images --output /path/to/new_output --gpus 0
# 4개 GPU 사용 예:
python predict_portable.py --images /path/to/images --output /path/to/another_output --gpus 3,4,5,6
```

`fusion/predicted_labels/이름.png`와 `fusion/predicted_masks/이름_mask.png`에 예측을 저장합니다.
`predict_portable.py`는 이번 가중치와 코드를 원본 manifest로 검증한 후 실행합니다.
보존된 원래 `predict.sh`에는 학습 서버의 conda 경로가 있으므로 다른 서버에서는 위 portable 실행기를 사용하세요.

## 구성과 지표

R: restart61, model269, B4-326 fold2, B4-329 fold1·3·4, HRNet347.
E: B4 warmstart fold1·3, HRNet414 N3, HRNet392 seed1 fold4, HRNet392 train70 seed5. F: 새 OOF로 학습한 보정기.
R `al_gamma=1`, `al3ni_gamma=1`, `eut_edge_blend=0.125`; E 비율 `0.4 / 0.42 / 0.18`입니다.
각 모델의 seed/fold/해시는 `model_manifest.json`에 있습니다.

image mIoU는 이미지별 네 클래스 IoU 평균을 20장에 대해 평균하며 빈 클래스 IoU는 0입니다.
개별 구성 모델의 단독 점수는 측정하지 않았습니다. test10은 정답이 없어 점수가 없습니다.
일부 R checkpoint 선택과 원본 결합 비율 조정에 valid20이 사용되므로 독립 test 성능으로 해석하지 않습니다.
공개 80.779378%는 원래 가중치의 참조 점수이며 이번 실측값은 위 80.471931%입니다.

원본 가중치/코드는 파일 단위 그대로 보존했습니다. 새 portable 실행기와 문서/결과만 추가했습니다.
`files_sha256.json`은 원래 모델/코드 해시, `bundle_sha256.json`은 이 전달 패키지의 파일 해시입니다.
학습 33개 중 중간/OOF 전용 가중치는 이 패키지에 포함되지 않습니다. 데이터 원본도 포함되지 않습니다.
