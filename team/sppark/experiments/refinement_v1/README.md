# 동료 결과와 현재 결과를 반영한 CNN refinement 실험 10개

`/Data3/sppark/komap/spec.md`, `AGENTS.md`를 따릅니다. 기존 모델·소스·결과를 수정하지 않고 새 구현과 결과를 이 폴더에만 둡니다. 진행률·경과 시간·ETA는 터미널 tqdm으로 표시하며 웹 서버나 대시보드를 실행하지 않습니다.

## 바로 실행

```bash
cd /Data3/sppark/komap
conda activate komap
bash experiments/refinement_v1/run_suite.sh 8
```

위 명령 하나로 **학습 70장의 오분류 위치 추출 → 실험 10개 순차 추가 학습 → 각각 plain/D4 검증 → 비교표 갱신**을 수행합니다. GPU 1/2/4/8장을 지원합니다. 실험 자체는 하나씩 순서대로 실행하고, 각 실험 안에서 지정한 GPU들을 사용합니다. 새로운 패키지나 DINO 사전학습 가중치는 필요 없습니다.

중단되면 같은 명령으로 다시 실행하세요. 설정·소스·데이터가 일치하는 완료 학습과 평가는 재사용하고, 미완료 학습은 같은 GPU 수로 `last.pt`에서 재개합니다. `이미 학습 완료 … 재사용합니다`는 정상 안내입니다. 평가 중 끊긴 부분 결과는 별도 `.interrupted-*` 폴더로 보관하고 다시 평가합니다. 기존 모델을 덮어쓰거나 새 모델을 자동으로 기존 모델 대신 배치하지 않습니다.

## 왜 이 실험인가

동료의 실제 파일명은 `model_results_0919.md`입니다. 사용자가 이미지가 동일하다고 확인했지만, 동료의 클래스 매핑·원본 평가 CSV·체크포인트는 제공되지 않았습니다.

| 결과 | 이미지별 평균 mIoU | Global mIoU |
|---|---:|---:|
| 현재 기준 CNN + D4 | 78.6865% | 81.4989% |
| 현재 seed2028 + D4 | 78.7858% | 약 81.5082% |
| 동료 residual ASPP 모델 42 (문서 기록) | 75.83% | 79.48% |
| 동료 최종 앙상블 모델 50 (문서 기록) | 75.73% | 79.48% |

**79.48과 78.79는 서로 다른 지표라 직접 비교하면 안 됩니다.** 동료 문서의 79.48%를 이 코드의 목표치로 그대로 복사하지 않습니다. 클래스별 수치 대응도 다릅니다. 현재 로컬 기준으로 공정 Si/Al3Ni가 약하므로 이 두 클래스를 중심으로 실험합니다. 동료의 Al logit bias나 클래스별 앙상블 가중치는 가져오지 않았습니다.

동료가 개선을 관찰한 native patch, hard-negative 중심 추출, residual ASPP를 후보로 삼습니다. 현재 모델은 이미 native patch·stride-1 skip을 사용합니다. 따라서 새로운 ResNet34/U-Net++를 그대로 재현하기보다 **현재 검증된 CNN의 가중치를 보존해 가져오고, 문맥 블록과 샘플링/손실을 추가**합니다. 이는 동료 모델의 동일 재현이 아니라 해당 아이디어를 현재 코드에 적용한 실험입니다. 지난 실험에서 나빠졌던 노이즈/배율 확대는 넣지 않습니다.

## 10개 조건

| 이름 | 기존 가중치 추가 학습 대조군 대비 변경 |
|---|---|
| `e00_ft_control` | 구조·샘플링·손실은 기존 그대로, 추가 학습 대조군 |
| `e01_hard` | 학습 이미지의 오분류 위치 중심 패치 추출 |
| `e02_weighted` | CE 클래스 가중치 `[1.0, 1.15, 1.5, 1.0]` |
| `e03_overlap_loss` | Dice/Lovasz 계수 각각 0.5 → 0.75 |
| `e04_hard_weighted` | e01 + e02 |
| `e05_aspp` | bottleneck에 residual ASPP 추가 |
| `e06_aspp_hard` | ASPP + 오분류 중심 추출 |
| `e07_aspp_weighted` | ASPP + 클래스 가중 CE |
| `e08_aspp_hard_weighted` | ASPP + 오분류 중심 추출 + 클래스 가중 CE |
| `e09_aspp_long_tail` | e08 + 전체 영상 단계 확대 |

클래스 순서는 **초정 Si / Al3Ni / 공정 Si / Al**입니다. 설정 목록은 `configs/plan.yaml`, 실제 공통 설정은 `refinelab/workspace.py`에서 확인할 수 있습니다. 조합 실험의 이득을 단일 요소의 이득으로 해석하지 않도록 대조군과 단독 변경 후보들을 함께 제공합니다.

## 공통 학습 조건

- 시작 가중치: `../accuracy_v1/baseline/run/best.pt`의 선택된 가중치. 파일은 읽기만 합니다.
- 기존 CNN 기준 모델을 사용하는 이유: seed2028의 근소한 최고점에는 A71 빈 클래스 제외 영향이 있었으므로, 넓은 이미지 집합의 기준 모델을 공통 출발점으로 삼습니다.
- 새 optimizer로 **2,000 updates 추가 학습**합니다. 기존 4,500 updates는 이미 수행된 초기화 비용이며, 이 실험은 처음부터 학습하는 실험과 같지 않습니다.
- 기본 일정: 512 패치 800 / 768 패치 600 / 전체 영상 600 updates.
- e09 일정: 512 패치 600 / 768 패치 400 / 전체 영상 1,000 updates. 총량은 동일합니다.
- 단계별 학습률: `3e-5 / 1e-5 / 5e-6`, warmup `50 / 30 / 25`.
- EMA decay `0.99`, effective batch 8, micro batch 1, seed 2026, 200 updates마다 검증. 원본의 EMA decay 0.999와 다르지만 모든 새 후보/대조군에 동일하게 적용합니다.
- 밝기·대비·감마 범위는 원래 약한 설정 유지. 마지막 전체 영상 단계에서 광도 변환을 끄고 회전/반전은 유지합니다.
- 단계 전환 시 직전 단계의 최고 가중치, 최종 평가 시 해당 실험의 최고 검증 체크포인트를 사용합니다. 원본 학습 엔진의 동작을 유지합니다.

기존 CNN 속도에서 보면 전체는 GPU 8장 기준 대략 **45~70분 규모의 계획 예산**입니다. 10개 전체 실측 시간은 아직 아니며 ASPP 비용·데이터 로딩 등에 따라 달라집니다. 각 학습의 단계별 ETA와 전체 완료 실험 수 기반 ETA를 터미널에서 보세요. 첫 실험 완료 전 전체 ETA는 미정입니다.

## 구현상 보장과 한계

### Residual ASPP

기존 bottleneck(384채널)에 64채널 축소와 dilation 1/2/4/8의 depthwise separable branch를 병렬로 적용합니다. 마지막 projection을 0으로 초기화해 처음에는 기존 CNN과 정확히 같은 출력을 냅니다. 기존 skip/decoder/head의 가중치 이름과 값을 그대로 옮기며 새 context 가중치만 추가됩니다. 전체 네트워크를 낮은 학습률로 미세조정합니다.

ASPP의 multi-scale context 개념은 [DeepLabv3 원 논문](https://arxiv.org/abs/1706.05587)을 참고했습니다. 이 구현은 현재 CNN에 맞춘 가벼운 residual 변형이며 논문/동료 모델의 구조와 완전히 같지는 않습니다.

### 실제 오분류 중심 추출

기준 체크포인트로 **공식 train 70장만** 추론합니다. 정답 클래스 1/2(Al3Ni/공정 Si)에서 틀린 픽셀을 우선 저장하고, 없으면 다른 클래스의 오분류를 사용합니다. 해당 이미지에 오류가 전혀 없으면 기존 경계 중심 추출로 fallback합니다.

hard 후보의 패치 중심 비중은 uniform 45% / 공정 Si 25% / hard 30%입니다. 기존 대조군의 uniform 60% / 공정 Si 25% / boundary 15%와 다릅니다. hard map도 이미지·마스크와 같은 회전/반전을 적용합니다. 전체 영상 단계에서는 별도 crop 중심을 사용하지 않습니다.

오류 지도는 `outputs/mining/`에 저장하며 SHA-256과 학습 ID를 검사합니다. validation/test 이미지는 이 지도에 들어가지 않습니다. 이 방식은 학습한 이미지에 대한 오류 추출이며 **OOF hard mining이 아닙니다**. 동료 문서에서도 OOF 변형이 항상 개선되지는 않았으므로 별도 효과 검증이 필요합니다.

### 실제 가중 CE

원본 손실 코드는 `class_weights` 필드를 CE에 전달하지 않았습니다. 이 실험은 원본 파일을 수정하지 않고 새 손실 함수에서 [PyTorch weighted cross-entropy](https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html)를 실제 적용합니다. 원본과 같은 이미지별 평균·ignore=255 처리를 유지합니다. Dice/Lovasz 정의는 기존과 같습니다.

### 평가와 과적합 확인

주 지표는 기존과 같은 공식 검증 20장의 이미지별 평균 mIoU입니다. 비교표에는 다음을 함께 저장합니다.

- Global mIoU와 클래스별 이미지 평균 IoU.
- 동일 추론(plain 또는 D4)의 기존 모델 대비 변화.
- 같은 seed의 추가 학습 대조군 e00 대비 변화.
- 클래스별 평균에 기여한 이미지 수와 빈 클래스 처리 one/zero 결과.
- 정답에 네 클래스가 모두 있는 동일 19장의 변화(A71 효과 민감도 분석).

10개를 같은 검증셋에서 선택하므로 최댓값은 개발 검증 점수입니다. 독립 시험 성능을 보장하지 않습니다. 이 실험들은 official 70장으로 학습한 체크포인트를 출발점으로 사용하기 때문에 다른 fold에서 그대로 시작하면 누출될 수 있습니다. 따라서 **이 runner는 official split만 지원**합니다. group CV에서는 fold별로 출발 모델부터 새로 학습해야 합니다.

## 개별 실행·평가·테스트 마스크

```bash
python -B experiments/refinement_v1/run.py list

# 필요한 학습 데이터 오류 지도만 미리 만들기 (suite에서 자동 수행)
python -B experiments/refinement_v1/run.py prepare

python -B experiments/refinement_v1/run.py train e08_aspp_hard_weighted --gpus 8
python -B experiments/refinement_v1/run.py train e08_aspp_hard_weighted --gpus 8 --resume

python -B experiments/refinement_v1/run.py evaluate e08_aspp_hard_weighted
python -B experiments/refinement_v1/run.py evaluate e08_aspp_hard_weighted --d4
python -B experiments/refinement_v1/run.py compare

# 비교 후 선택한 모델로 test 10장 PNG 출력
python -B experiments/refinement_v1/run.py predict e08_aspp_hard_weighted --d4
```

e08은 예시 이름이며 학습 전 최고 모델이라고 정한 것은 아닙니다. 테스트는 정답이 없어 정확도를 계산하지 않고 ID/색상/크기/개수만 검사합니다. 새 ASPP 체크포인트는 이 폴더의 runner로 불러오세요. 원본 CLI는 새 모델 이름을 알지 못합니다.

추가 seed에서 대조군과 후보를 함께 반복하려면:

```bash
python -B experiments/refinement_v1/run.py suite --gpus 8 \
  --experiments e00_ft_control e08_aspp_hard_weighted --seeds 2027 2028
```

여기서 seed 변경은 같은 기준 체크포인트 이후의 미세조정 난수만 바꾸며, 출발 모델을 새 seed로 다시 학습한다는 뜻은 아닙니다.

## 저장 경로와 재현

```text
outputs/
  mining/                          # train 70장 전용 오류 지도
  runs/<실험>/seed2026/official/     # best.pt, last.pt, config, metrics, status
  evaluations/<실험>/seed2026/{plain,d4}/
  reports/comparison.csv           # 모든 완료된 본 실험 + 기준 모델
  reports/comparison.txt
  reports/selection.json           # 관측 최고점 정보; 기존 모델 자동 교체 안 함
  predictions/<실험>/seed2026/d4/masks/
  smoke*/                          # 짧은 기능 점검용, 본 비교표에서 제외
```

체크포인트 설정에 출발 가중치 SHA-256과 새 구현 SHA-256을 넣어 다른 코드/설정으로 잘못 resume하는 것을 거부합니다. 원본 엔진의 `code_sha256`은 frozen alphase 코드 해시이고, 새 모델/손실/데이터 코드 해시는 별도 `refinement.source_sha256`에 기록합니다. 프로젝트 기존 파일 2,361개의 작업 전 해시는 `preservation.json`에 저장했습니다.

```bash
python -B experiments/refinement_v1/run.py verify
python -B -m pytest -c experiments/refinement_v1/pytest.ini -q experiments/refinement_v1/tests
python -B experiments/refinement_v1/run.py suite --gpus 8 --smoke
```

`--smoke`는 각 단계 1 update만 실행하며 결과는 별도 경로에 저장합니다. 기존 학습 가중치로 시작하므로 smoke도 점수가 높을 수 있지만 이것을 본 실험의 개선 증거로 해석하지 않습니다. 구현 확인 결과는 `BUILD_REPORT.md`에 기록합니다.
