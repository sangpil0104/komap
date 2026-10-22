# AlPhaseNet 구현 명세서

> **실행 안내:** 데이터 준비와 CNN 기준 모델의 학습·검증·추론 구현을 완료했습니다. 기존 `komap` conda 환경에서 실행할 명령, GPU별 예상 시간, 실시간 진행 화면은 **[QUICKSTART.md](QUICKSTART.md)** 를 참고하세요. 아래는 원본 설계 명세이며, DINOv3 공식 가중치와 후속 앙상블/OOF 연구 항목의 구현·검증 범위는 실행 안내에 구분했습니다.

## 알루미늄 합금 OM 이미지의 4상 분할 — mIoU 우선 설계

- 문서 버전: 1.0 / 2026-09-20
- 대상: 모델을 직접 구현할 개발자 및 코딩 에이전트
- 입력 자료: `dataset.zip`, `phase_analysis_final.md`, 이전에 제안한 하이브리드·앙상블 구조
- 목표: 추론 속도나 모델 크기보다 **독립 검증에서의 이미지별 평균 mIoU**를 우선한다.
- 문서 성격: 구현 설계서와 핵심 참조 코드. 완성된 학습 프로그램이나 학습 완료 가중치는 아니다.

> **핵심 설계**
>
> `DINOv3 문맥 인코더 + 원해상도 CNN + 다중 해상도 특징 결합 + 원해상도 U-Net 디코더`를 주력 단일 모델로 구현한다. 이후 검증에서 이득이 확인된 CNN/HRNet 모델, 확률 앙상블, OOF 기반 잔차 보정기를 추가한다.
>
> 여기서 제시하는 채널 수, 학습률, 학습 길이는 **실험의 시작값**이다. 이 조합의 최고 성능을 이미 검증했다는 뜻은 아니다. 모듈이 복잡해져도 검증 mIoU가 낮아지면 해당 모듈을 최종 구성에서 제외한다.

---

## 목차

1. 데이터와 문제의 고정 정의
2. 최종 시스템과 구현 순서
3. 저장소와 실행 환경
4. 데이터 로더·마스크·manifest
5. 좌표계와 전체 영상/패치 정렬
6. 주력 모델의 상세 구조
7. 손실함수
8. 샘플링과 데이터 증강
9. 단계별 학습 설정
10. 평가와 모델 선택
11. 이종 모델과 앙상블
12. OOF 및 중첩 검증
13. 원해상도 잔차 보정기
14. 추론·TTA·클래스 보정
15. 추가 성능 실험
16. 전체 90장 재학습과 제출
17. 구현 완료 기준과 실험표
18. CLI와 산출물 계약
19. 설정 파일 예시
20. 핵심 참조 코드
21. 구현 시 피해야 할 오류
22. 근거 자료

---

## 1. 데이터와 문제의 고정 정의

### 1.1 실제 ZIP 구조

현재 ZIP에는 다음 경로가 들어 있다. 입력과 마스크는 **파일의 정렬 순서가 아니라 공통 ID**로 매칭한다.[^data]

```text
dataset/
├── train/
│   ├── images/  # 70개, 예: A101_step3_sample3_image.png
│   └── masks/   # 70개, 예: A101_step3_sample3_mask.png
├── valid/
│   ├── images/  # 20개
│   └── masks/   # 20개
└── test/
    └── images/  # 10개, 정답 없음
```

데이터 ZIP SHA-256:

```text
ebbc9e5e7e758152f006b4e515abf5d61309d432b8ae028330bc86e0372fbc55
```

초기 모델 비교는 공식 분할인 `train 70 → valid 20`을 사용한다. 설정을 확정한 뒤에는 사용자 전제에 따라 정답이 있는 90장을 전체 학습에 사용한다. 외부 사전학습과 정답 없는 테스트 영상 활용도 실험 가능한 것으로 둔다. 이것은 사용자가 정한 개발 전제이지, 공개 페이지에서 추가 허용 규칙을 확인했다는 뜻은 아니다.

### 1.2 클래스 ID와 색상

아래 ID 순서를 **데이터 로더, 손실, 체크포인트, OOF, 앙상블, 제출에서 동일하게 유지**한다. ID는 이 프로젝트 내부 정의이고, RGB는 실제 마스크에 들어 있는 값이다.[^data]

| ID | 이름 | 기호 | RGB | 정답 90장 전체 픽셀 비율 |
|---:|---|---|---|---:|
| 0 | 초정 Si | Primary Si | `(76, 178, 76)` | 9.5673% |
| 1 | Al₃Ni | Al3Ni | `(25, 76, 153)` | 25.3461% |
| 2 | 공정 Si | Eutectic Si | `(204, 204, 204)` | 7.1044% |
| 3 | Al | Al | `(153, 127, 76)` | 57.9822% |

`ignore_index=255`는 학습 중 생기는 패딩·무효 영역용이다. 원본 정답에 존재하는 다섯 번째 상이 아니다. Al도 평가 클래스이며, 배경으로 제외하지 않는다.

입력과 마스크 파일은 RGBA이다. 입력은 `R=G=B`, 알파는 255이므로 정보는 단일 그레이스케일 채널에 있다. 입력 크기는 다음 다섯 종류이며 표는 **폭 × 높이**다.[^data]

| 크기 | train | valid | test |
|---|---:|---:|---:|
| 675 × 480 | 17 | 9 | 3 |
| 1096 × 780 | 0 | 1 | 0 |
| 1230 × 875 | 20 | 3 | 4 |
| 1278 × 909 | 24 | 5 | 2 |
| 1440 × 1024 | 9 | 2 | 1 |

### 1.3 설계 근거와 검증 범위

이전 데이터 분석에서 공정 Si 픽셀의 약 77.44%는 가장 가까운 다른 상 픽셀 중심과의 거리가 2픽셀 이하였다. 따라서 가는 조직을 보존하는 원해상도 경로를 우선 구현한다. 이는 거리 변환으로 얻은 픽셀 중심 간 거리이며, 기하학적 경계선까지의 정확한 거리와는 다르다.[^data]

이전 진단 실험의 검증 mIoU는 밝기 lookup 48.22%, 밝기·전체 통계 Extra Trees 57.59%, 국소 특징을 추가한 Extra Trees 72.24%였다. 이 수치는 분석 기록의 실측값을 인용한 것이며, 이번 문서 작성 중 신경망을 학습해 얻은 결과가 아니다. DINOv3 하이브리드나 보정기의 성능 상한을 뜻하지도 않는다.[^data]

**최적화할 과제는 4클래스 semantic segmentation이다.** 입자별 instance ID 생성, 탄성계수 회귀, 조성 수치 예측은 본 구현의 목표가 아니다. 파일명의 `step`이나 `sample`을 물리적으로 보정된 공정 변수로 간주하지 않는다.

---

## 2. 최종 시스템과 구현 순서

### 2.1 추론 파이프라인

```text
OM 원본
  │
  ├─ A: DINOv3 + 원해상도 CNN + U-Net ───┐
  ├─ B: ConvNeXt + 원해상도 U-Net ───────┤
  └─ C: HRNet + 원해상도 출력 경로 ──────┤
                                        │
             모델별 TTA 복원 및 확률 평균
                                        │
                 구조·seed별 확률 앙상블
                                        │
                  P_base: [4, H, W]
                                        │
              원본 + P_base + 불확실성 + 기울기
                                        │
              OOF로 학습한 잔차 보정기 [선택]
                                        │
                   클래스별 logit 편향 [선택]
                                        │
                      마지막에 argmax
                                        │
                    원본 크기의 색상 PNG
```

TTA는 입력의 회전·반전·배율 변화를 적용해 얻은 예측을 원본 좌표로 되돌려 결합하는 방법이다. OOF는 해당 이미지의 정답을 학습하지 않은 모델에서 얻은 예측이다.

### 2.2 단계별 개발 우선순위

| 순서 | 구현물 | 다음 단계로 넘어가는 조건 |
|---:|---|---|
| 1 | 데이터 검사, 마스크 변환, 정확한 평가 함수 | 색상 왕복 변환·무효 픽셀·없는 클래스 테스트 통과 |
| 2 | 원해상도 CNN/U-Net 기준 모델 | 작은 학습 패치 과적합 가능, 전체 검증 및 저장·복원 가능 |
| 3 | DINOv3 다중 층 특징 + 단순 다중 스케일 결합 | 좌표 정렬 검증, CNN 단독과 동일 조건 비교 |
| 4 | H+/16 및 선택적 상호작용 Adapter | 추가 모듈의 독립 검증 이득 확인 |
| 5 | 다른 인코더, seed, fold 학습 | 모든 예측의 클래스·좌표·학습 ID 추적 가능 |
| 6 | 확률 앙상블과 TTA | 기본 모델보다 외부 검증 mIoU 개선 |
| 7 | OOF 잔차 보정기 | 보정기까지 포함한 중첩 검증에서 개선 |
| 8 | 의사라벨·도메인 적응·7B 실험 | 앞 단계와 분리한 대조 실험에서 개선 |
| 9 | 90장 재학습, 제출 파이프라인 고정 | 테스트 10장 출력 규격 및 재현성 검사 통과 |

**앙상블과 보정기를 한꺼번에 먼저 만들지 않는다.** 처음부터 복잡한 시스템을 만들면 mIoU 저하가 데이터·정렬·학습·결합 중 어디서 생겼는지 구별하기 어렵다.

---

## 3. 저장소와 실행 환경

### 3.1 권장 저장소 구조

아래는 새로 구현할 프로젝트의 구조다. 이 문서가 이 파일들을 모두 제공하는 것은 아니다.

```text
alphase/
├── configs/
│   ├── hybrid_hplus.yaml
│   ├── hybrid_large.yaml
│   ├── convnext_unet.yaml
│   ├── hrnet_detail.yaml
│   ├── refiner.yaml
│   └── final_inference.yaml
├── src/alphase/
│   ├── data/
│   │   ├── audit.py               # 파일, 색상, 크기, 중복 검사
│   │   ├── manifest.py            # ID 기반 입력/마스크 매칭
│   │   ├── palette.py             # RGB ↔ ID
│   │   ├── dataset.py             # 전체 영상 + 로컬 영역
│   │   ├── geometry.py            # 패딩, crop, 좌표 변환
│   │   ├── transforms.py          # 기하/광학 증강의 분리
│   │   ├── sampler.py             # 이미지 균등·어려운 패치
│   │   └── splits.py              # 고정 분할·group fold
│   ├── models/
│   │   ├── detail_encoder.py
│   │   ├── dino_encoder.py
│   │   ├── fusion.py
│   │   ├── decoder.py
│   │   ├── hybrid.py
│   │   ├── convnext_unet.py
│   │   ├── hrnet_detail.py
│   │   └── refiner.py
│   ├── losses/                    # CE, Dice, Lovasz, 경계 보조
│   ├── engine/                    # trainer, EMA, checkpoint
│   ├── evaluation/                # 이미지별 mIoU와 진단
│   ├── inference/                 # 타일, D4, multi-scale, ensemble
│   └── cli/                       # 18절에 정의한 명령
├── tests/
├── third_party/dinov3/
├── data/raw/
├── data/processed/
├── weights/pretrained/
├── runs/
├── oof/
├── artifacts/
├── submission/
├── pyproject.toml
├── requirements.lock.txt
└── environment.yaml
```

### 3.2 환경 원칙

Python 3.11을 시작 환경으로 사용한다. 이는 확인한 DINOv3의 공식 conda 설정과도 맞는다. PyTorch와 torchvision은 GPU 드라이버/CUDA와 호환되는 조합을 설치한 뒤 실제 smoke test를 통과한 버전을 고정한다. 설치하지 않은 버전 조합을 검증 완료 조합처럼 기록하지 않는다.[^env]

기본 구현은 PyTorch 중심으로 만든다. V1에는 MMCV나 사용자 정의 deformable attention CUDA 연산을 필수로 넣지 않는다. NumPy, Pillow, SciPy, pandas, scikit-learn, OmegaConf, PyYAML, pytest를 사용하고, timm은 추가 인코더를 선택할 때 별도로 고정한다.

```bash
# 호환되는 PyTorch/torchvision을 먼저 설치한 환경에서 실행
python -m pip install numpy pillow scipy pandas scikit-learn omegaconf pyyaml tqdm pytest
mkdir -p third_party artifacts weights/pretrained
git clone https://github.com/facebookresearch/dinov3.git third_party/dinov3

# 이 커밋을 smoke test한 뒤 고정한다. 이후 실험 중에는 임의로 pull하지 않는다.
git -C third_party/dinov3 rev-parse HEAD > artifacts/dinov3_commit.txt
python -m pip freeze > requirements.lock.txt
```

공식 DINOv3 가중치는 공식 배포 경로에서 접근 권한을 받아 로컬 파일로 준비한다. 접근 실패를 무작위 초기화로 조용히 대체하지 않는다. 체크포인트 경로가 없으면 학습을 시작하지 않는다.[^dino]

특히 확인한 ViT-L 로더는 로컬 가중치 파일명의 해시 접미사를 검사하는 로직이 있다. **공식 `.pth` 파일명을 임의로 `backbone.pth`로 바꾸지 않는다.** 로더가 바뀌면 고정한 소스에서 동작을 재확인한다.[^backbones]

최종 기록에는 Python/PyTorch/torchvision/CUDA/cuDNN 버전, GPU 이름, 코드 commit, DINOv3 commit, 사전학습 가중치 SHA-256, 데이터 SHA-256, fold 파일 hash, seed, 실제 적용된 설정 전체를 포함한다. 인증 토큰은 저장하지 않는다.

---

## 4. 데이터 로더·마스크·manifest

### 4.1 입력 및 정답 검사

`audit.py`는 최소한 다음을 검사하고 JSON으로 저장한다.

- 입력 100장, 정답 90장, 입력/정답의 ID 매칭 및 같은 공간 크기.
- RGBA/그레이스케일 채널 조건, alpha, 네 정답 색상, 예상 밖 파일.
- 파일 바이트 hash와 픽셀 배열 hash를 구분한 중복 검사.
- 클래스별 픽셀 수, 이미지별 상 비율, 클래스 부재 여부.
- 그룹 접두사, 파일별 폭·높이·평균 밝기·표준편차.

마스크에 알 수 없는 RGB가 있으면 **가장 가까운 색으로 자동 교정하지 말고 오류를 낸다.** 현재 데이터는 정확한 네 색상이므로 보간·입출력 버그를 조기에 찾아야 한다. 상세 참조 코드는 20절에 있다.

### 4.2 manifest 계약

`manifest.csv`는 원본 이미지당 한 행이며 다음 필드를 가진다.

```text
image_id,source_split,image_path,mask_path,width,height,
group_id,step_token,sample_token,image_sha256,
phase0_fraction,phase1_fraction,phase2_fraction,phase3_fraction
```

- `image_id`: `_image.png` 또는 `_mask.png`를 제거한 공통 문자열.
- `group_id`: 우선 파일명 `A숫자` 접두사를 사용한다. 실제 동일 시편임을 확정한 것은 아니다.
- `source_split`: 원래 train/valid/test를 영구 보존한다. 재학습 split과 구분한다.
- `mask_path`와 상 비율: test에서는 비어 있어야 한다. 테스트 예측으로 정답 통계 필드를 채우지 않는다.

정규화나 표본 가중치 등 **학습되는 데이터셋 전체 통계는 해당 fold의 학습 부분에서만 추정**한다. 각 입력 한 장에서 직접 계산하는 평균·표준편차·CDF는 추론 시에도 같은 방식으로 얻을 수 있으므로 사용 가능하다.

### 4.3 두 경로의 입력

그레이스케일을 `g ∈ [0,1]`로 변환한다. 시작 설정의 CNN 입력은 세 채널이다.

```text
detail[0] = g
detail[1] = clip((g - mean_image) / max(std_image, 1/255), -5, 5) / 5
detail[2] = image_CDF(g)
```

여기서 CDF는 영상 한 장의 256-bin 밝기 누적분포다. 통계와 CDF는 **패딩 전의 실제 전체 영상**에서 계산한다. 로컬 패치별로 계산하면 같은 위치가 패치마다 다른 값이 되므로 기본 설정에서는 사용하지 않는다.

첫 번째 대조 실험은 `g` 한 채널만 사용하는 모델이다. 정규화/CDF 채널이 항상 이득이라고 가정하지 않는다.

DINOv3 입력은 `g`를 세 채널로 복제한 RGB 텐서다. LVD 사전학습 가중치의 시작 정규화는 평균 `(0.485, 0.456, 0.406)`, 표준편차 `(0.229, 0.224, 0.225)`다. 분류 예제의 정사각형 강제 resize는 가져오지 않고, 본 과제에서는 원래 비율과 좌표를 유지한다.[^dino]

증강된 영상을 사용하는 경우, 전체 영상에 같은 광학 변환을 적용한 뒤 두 경로를 만든다. **CNN은 밝기 변화가 적용된 영상인데 DINO는 변화 전 영상인 상태**가 기본 파이프라인에 섞이지 않게 한다.

---

## 5. 좌표계와 전체 영상/패치 정렬

### 5.1 기본 좌표 규약

코드의 공간 크기는 모두 `(H, W)`, 텐서는 `[B, C, H, W]`다. 원본 ID 마스크는 `[H, W]`, 학습 target은 `[B, H, W]`의 정수형이다.

영상 경계 좌표는 `[0,W] × [0,H]`로 정의하고 픽셀 중심은 `(x+0.5,y+0.5)`다. crop box는 `(x0,y0,x1,y1)`이며 오른쪽/아래는 exclusive다. 모든 스케일·반전·회전·패딩의 역변환을 metadata로 보존한다.

입력은 오른쪽과 아래를 32의 배수까지 패딩한다. 입력 영상은 reflect를 기본으로 하고, 정답은 255, 유효 영역은 boolean으로 관리한다. reflect가 성립하지 않는 극단적으로 작은 입력은 replicate로 명시적으로 전환한다.

예를 들어 폭 1278, 높이 909인 영상은 `[H_p,W_p]=[928,1280]`이다. 최종 출력은 `[..., :909, :1278]`로 잘라 원래 크기로 복원한다.

### 5.2 두 가지 forward 모드

**모드 A: full-frame.** CNN과 DINO가 같은 전체 영상을 본다. 최초 참조 구현과 최종 전체 영상 미세조정의 기준 모드다. GPU 메모리가 허용되면 추론에서도 우선 비교한다.

**모드 B: global-context/local-detail.** DINO는 전체 영상을 보고, CNN은 원해상도 로컬 패치를 본다. 전체 DINO 특징에서 로컬 패치에 해당하는 위치를 샘플링해 결합한다. 패치만 보고 얻은 DINO 특징으로 바꾸면 문맥 범위가 달라지므로 별도 모델 변형으로 취급한다.

원해상도 보존은 항상 배율 1만 써야 한다는 뜻이 아니다. **원본 세부 경로를 강제 축소하지 않는 것**이 기본이며, 확대·축소는 별도 증강/TTA 실험이다. 문맥 경로만 낮은 배율로 사용하는 변형도 허용하되 기본 설정과 분리한다.

### 5.3 ROI 특징의 정렬 수식

DINO 전체 특징 `F`가 `[B,D,H_p/16,W_p/16]`이고, 원본/패딩 영상 좌표에서 로컬 영역이 `(x0,y0,x1,y1)`이라고 하자. 출력 특징 크기가 `(h,w)`이면 샘플 중심은 다음과 같다.

```text
x(j) = x0 + (j + 0.5) * (x1 - x0) / w
y(i) = y0 + (i + 0.5) * (y1 - y0) / h
gx(j) = 2 * x(j) / W_p - 1
gy(i) = 2 * y(i) / H_p - 1
```

이를 `grid_sample(mode="bilinear", align_corners=False, padding_mode="border")`에 전달한다. 전체 영상/패치 resize에도 같은 `align_corners=False` 규약을 사용한다. PyTorch의 두 연산은 같은 좌표 해석을 맞춰 사용해야 한다.[^grid]

**grid 좌표 생성과 ROI 샘플링은 float32로 수행한다.** 큰 영상에서 정규화 좌표를 낮은 정밀도로 저장해 위치 정밀도가 떨어지는 것을 피하기 위한 설계다. 그 뒤 합성곱에는 AMP를 사용해도 된다.

CNN의 stride 2 다운샘플은 참조 구현에서 `kernel=2,stride=2,padding=0` 합성곱을 사용한다. 이렇게 각 cell 중심을 정의하면 위 ROI 좌표 규약과 일관되게 맞추기 쉽다. 다른 다운샘플로 바꿀 때는 중심 위치도 다시 점검한다.

### 5.4 halo

로컬 모드의 시작 설정은 중앙 학습 영역 `512×512`와 주위 halo 64픽셀이다. 실제 CNN 입력은 `640×640`이고, 손실은 중앙 영역과 실제 영상이 겹치는 유효 픽셀에서만 계산한다. 다음 단계는 중앙 `768×768`, 입력 `896×896`이다.

작은 영상에서는 영상 밖을 패딩하되 정답을 만들어내지 않는다. 전체 영상 미세조정에서는 halo를 쓰지 않고 유효한 원본 전체에 손실을 적용한다.

### 5.5 특징 캐시의 제한

처음에는 캐시 없이 online forward로 구현한다. 캐시는 정합성 검증 뒤에 추가한다.

DINO가 고정되어 있고, 입력 영상·배율·변환·패딩·정규화·가중치·평가 모드가 모두 같을 때만 캐시를 재사용한다. DINO 파라미터를 업데이트하면 이전 캐시는 무효다. 반전/회전된 입력의 특징을 원래 특징의 단순 반전/회전으로 대체하지 않는다. 두 연산의 동등성이 보장되지 않기 때문이다.

DINOv3에는 학습 모드에서 동작하는 위치 좌표 증강 코드가 있다. 본 프로젝트의 시작 설정은 backbone을 `eval()` 모드로 유지해 이 동작을 끄고, **fine-tuning 여부는 `requires_grad`와 autograd 문맥으로 별도 제어**한다. `eval()` 자체가 gradient를 차단하지는 않는다.[^rope]

---

## 6. 주력 모델의 상세 구조

### 6.1 모델 이름과 버전 구분

- `AlPhaseNet-Hybrid-v1`: DINO 다중 층 특징 → projection → ROI/resize → CNN 특징과 결합. **먼저 구현할 버전**이다.
- `AlPhaseNet-Hybrid-v2`: v1의 성공 후 비교할, Transformer와 공간 특징 사이의 상호작용 Adapter 버전이다.

v1을 원 논문의 ViT-Adapter를 그대로 재현했다고 부르지 않는다. ViT-Adapter는 조밀 예측을 위한 공간적 사전정보 및 특징 상호작용 설계의 참고 근거이며, 단순 피라미드 결합과는 다른 구조다.[^adapter]

### 6.2 CNN detail encoder

기본 block은 다음과 같다.

```text
ResidualBlock(in, out):
  Conv3×3 → GroupNorm(8 groups) → GELU
  Conv3×3 → GroupNorm(8 groups)
  + identity 또는 Conv1×1 skip
  → GELU
```

BatchNorm 대신 GroupNorm을 시작 설정으로 사용해 작은 micro-batch에서의 동작을 일정하게 한다. 그룹 수는 채널 수의 약수여야 한다.

| 이름 | 해상도 | 채널 | 구성 |
|---|---:|---:|---|
| C1 | 1/1 | 32 | stem + residual block 2개 |
| C2 | 1/2 | 64 | stride 2 downsample + residual block 2개 |
| C4 | 1/4 | 128 | 동일 |
| C8 | 1/8 | 256 | 동일 |
| C16 | 1/16 | 384 | 동일 |

**C1을 마지막 분류 단계까지 직접 전달한다.** C1 없이 1/4 특징만 늘리는 모델을 원해상도 세부 모델로 취급하지 않는다.

### 6.3 DINOv3 context encoder

사용할 공식 Hub 이름과 특징 층은 아래와 같다. 층 index는 0-based다. 공식 코드의 depth와 embedding dimension을 기준으로 하며, 실제 로드 후에도 assert한다.[^backbones][^vitcode]

| 용도 | Hub 이름 | blocks | 특징 차원 D | 사용할 block index |
|---|---|---:|---:|---|
| 연결/학습 기준 후보 | `dinov3_vitl16` | 24 | 1024 | `[5,11,17,23]` |
| 주력 성능 후보 | `dinov3_vith16plus` | 32 | 1280 | `[7,15,23,31]` |
| 대형 추가 실험 | `dinov3_vit7b16` | 40 | 4096 | `[9,19,29,39]` |

검증용 smoke test만 작은 모델을 쓸 수 있지만, 최종 선택은 실제 비교 결과로 한다. H+ 또는 7B가 L보다 반드시 좋다고 가정하지 않는다.

특징은 공식 API의 `get_intermediate_layers(..., n=layer_indices, reshape=True)`에서 얻는다. `n=4`는 임의로 고른 네 층이 아니라 마지막 네 층을 뜻하므로 명시적 index 목록을 넘긴다. 클래스 토큰과 storage token을 이미지 격자에 섞지 않는다.[^vitcode]

중요하게, **네 층의 출력은 모두 stride 16**이다. 서로 다른 층을 읽었다고 자동으로 stride 4/8/16/32 특징이 생기는 것이 아니다. 다중 해상도 특징은 다음 fusion 모듈에서 생성한다.

### 6.4 Projection과 fusion

각 DINO 특징 `T_k`를 독립적인 `1×1 Conv(D→256) + GroupNorm + GELU`로 투영해 `P_k`를 만든다.

로컬 CNN의 stride `s ∈ {4,8,16}`마다 다음을 수행한다.

```text
A_s = concat(
        sample(P_0, local_box, target_shape=C_s.shape),
        sample(P_1, local_box, target_shape=C_s.shape),
        sample(P_2, local_box, target_shape=C_s.shape),
        sample(P_3, local_box, target_shape=C_s.shape)
      )                         # 1024 channels

G_s = Conv1×1(1024→256) + GN + GELU
F_s = ResidualBlock(concat(C_s, G_s) → channels(C_s))
```

초기 버전에서는 attention을 추가하지 않는다. 4개 층을 한 번에 사용하는 것과 마지막 층만 사용하는 것을 비교해 각 추가 특징의 이득을 확인한다.

### 6.5 Decoder

```text
D16 = F16                                      # 384ch, 1/16
D8  = ResBlock(concat(upsample(D16), F8), 256)   # 1/8
D4  = ResBlock(concat(upsample(D8),  F4), 128)   # 1/4
D2  = ResBlock(concat(upsample(D4),  C2),  64)   # 1/2
D1  = ResBlock(concat(upsample(D2),  C1),  32)   # 1/1
Z   = Conv1×1(D1, 4)                           # logits
```

upsample은 목표 feature의 정확한 `(H,W)`를 명시해 bilinear로 수행한다. `scale_factor=2`를 연쇄 사용해 반올림 오차가 누적되는 구현을 피한다. U-Net의 skip 연결 원리를 사용하되 본 설계의 채널·fusion은 별도 제안이다.[^unet]

`Z`에는 모델 내부에서 softmax를 적용하지 않는다. CE는 logits를 받고, Dice/Lovasz/추론에서 필요한 시점에 softmax를 계산한다.

출력 계약:

```text
{
  "logits":          [B,4,H_local,W_local],
  "aux_logits":      [B,4,H_local/4,W_local/4],
  "boundary_logits": [B,1,H_local,W_local]
}
```

보조 출력은 선택적이다. 최종 제출에 사용되는 것은 네 클래스의 원해상도 출력 하나다.

### 6.6 실수하기 쉬운 실제 shape 예시

전체 영상 `H=909,W=1278`을 패딩한 뒤 H+ 인코더를 사용하면 다음과 같다.

```text
image_detail: [B,3,928,1280]
image_dino:   [B,3,928,1280]
T_k:          [B,1280,58,80]      # 4개 모두 같은 공간 크기
P_k:          [B,256,58,80]
C1:           [B,32,928,1280]
C2:           [B,64,464,640]
C4:           [B,128,232,320]
C8:           [B,256,116,160]
C16:          [B,384,58,80]
logits:       [B,4,928,1280]
저장 전:      [B,4,909,1278]
```

### 6.7 v2: 선택적 상호작용 Adapter

v1을 기준으로만 비교한다. 논문 구현을 고정 commit으로 포팅하고, CNN 공간 특징과 Transformer patch token 사이에 여러 단계의 정보 교환을 추가한다. 기본 후보는 4개 block 구간에 대응하는 interaction이며, residual scale을 작게 초기화한다.[^adapter]

DINOv3의 RoPE와 storage token 처리를 유지해야 한다. 다른 ViT용 코드를 그대로 연결해 class token 수, positional embedding, 정규화가 맞을 것으로 가정하지 않는다. v2는 v1 체크포인트와 state dict 구조가 다르므로 별도 모델명과 설정을 사용한다.

---

## 7. 손실함수

### 7.1 기본 설정

```text
L_main = 1.0 * CE_image
       + 0.5 * Dice_image_present
       + 0.5 * Lovasz_image_present

L_total = L_main
        + lambda_aux * L_aux_softCE
        + lambda_boundary * L_boundary
```

최초 연결 및 기준 실험에서는 `lambda_aux=0`, `lambda_boundary=0`으로 둔다. 그다음 `lambda_aux=0.1`, `lambda_boundary=0.05`를 각각 분리해서 시험한다. 이전 제안의 경계 손실 0.05는 **기본 확정값이 아니라 이 비교 단계의 시작값**으로 구체화한다.

### 7.2 이미지별 평균

유효 픽셀 집합을 `V_i`라 할 때 CE는 다음처럼 계산한다.

```text
CE_i = mean_{p in V_i}( -log P_i[y_i(p),p] )
CE_image = mean_i(CE_i)
```

전체 batch의 유효 픽셀을 한꺼번에 평균하면 큰 이미지의 영향력이 더 커질 수 있으므로 이미지별 정규화 후 평균한다. 유효 픽셀이 없는 crop은 손실 0으로 조용히 처리하지 말고 sampler 단계에서 다시 뽑는다.

Dice는 이미지별·정답에 존재하는 클래스별로 계산하고 평균한다. 정답에 없는 클래스의 거짓 양성은 CE가 계속 벌점으로 다룬다. 없는 클래스에 대한 별도 벌점은 추가 실험으로 분리한다.

### 7.3 Lovász-Softmax

공식 구현의 `per_image=True`, `classes="present"`, `ignore=255`를 시작 설정으로 사용한다. Lovász-Softmax는 IoU를 최적화하기 위한 대리 손실이며, hard argmax mIoU와 정확히 같은 함수는 아니다.[^lovasz][^lovaszcode]

패치를 넣었을 때의 `per_image=True`는 원본 전체 이미지가 아니라 **각 패치별** 계산이다. 마지막 full-frame 미세조정이 평가 단위와 더 가까워지는 이유도 여기에 있다.

CE·Dice·Lovasz의 확률 및 합산·정렬 계산은 float32에서 수행한다. 마스크를 softmax에 넣거나 logits에 중복 softmax를 적용하지 않는다.

### 7.4 저해상도 보조 정답

stride 4 보조 head의 정답을 hard mask nearest resize로만 만들면 가는 상이 사라질 수 있다. 보조 손실을 사용할 때는 다음과 같이 만든다.

1. 원해상도 target을 4채널 one-hot으로 변환한다. ignore 영역은 모두 0이다.
2. one-hot과 유효 영역을 각각 같은 영역 평균으로 축소한다.
3. 각 cell에서 `soft_target = pooled_one_hot / max(pooled_valid, eps)`를 계산한다.
4. 유효 비율로 가중한 soft cross-entropy를 계산한다.

이 보조 손실이 원해상도 main loss를 대체해서는 안 된다.

### 7.5 경계 보조 head

서로 다른 클래스의 유효한 4-neighbor 쌍을 경계로 정의하고, 쌍의 양쪽 픽셀을 표시한다. 영상 바깥·ignore와 맞닿은 부분을 가짜 상 경계로 만들지 않는다.

시작 손실은 boundary BCE와 binary Dice의 평균이다. 경계 head는 학습 보조용이며, 추론 때 이진 경계를 최종 마스크에 강제로 덮어쓰지 않는다. 특정 상을 일정 두께로 만들도록 강제하지도 않는다.

---

## 8. 샘플링과 데이터 증강

### 8.1 이미지 균등 선택

먼저 학습 이미지 ID를 균등하게 선택한 뒤 그 이미지 안의 patch를 선택한다. 큰 이미지가 패치 수 때문에 학습을 지배하지 않도록 한다. 초기 패치 선택 비율의 예시는 다음과 같다.

| 패치 유형 | 비율 | 선택 방법 |
|---|---:|---|
| 일반 위치 | 60% | 영상 안의 중심을 균일하게 선택 |
| 공정 Si 중심 | 25% | class 2 픽셀을 중심 후보로 선택 |
| 오류/경계 중심 | 15% | 해당 학습 이미지의 오류 지도 또는 경계 지도에서 선택 |

초기에는 오류 지도가 없으므로 마지막 항목을 경계 또는 일반 패치로 대체한다. 대상 클래스가 없는 영상에서도 무한 재추출 루프가 생기지 않게 일반 패치로 fallback한다.

오류 지도는 해당 fold의 학습 영상에서만 만든다. 검증 정답의 어려운 영역을 이용해 학습 patch 위치를 선택하지 않는다. 전체 검증에서 얻은 실패 유형으로 일반적인 실험 방향을 정하는 것과, 검증 픽셀을 직접 학습 표본으로 쓰는 것은 구분한다.

### 8.2 기하 증강

기본은 D4의 8가지 변환 중 균일 선택이다. `rot90(k), k∈{0,1,2,3}`와 수평 반전 여부로 정의하면 중복 없이 구성할 수 있다.

전체 영상과 정답에 같은 기하 변환을 먼저 적용하고, 그 좌표계에서 로컬 crop을 뽑는다. 확대 실험은 `scale∈{1.0,1.25,1.5}`로 시작한다. `0.75` 같은 축소는 별도 비교 대상으로 둔다.

영상 resize는 bilinear/antialias 설정을 고정하고, **정수 클래스 ID** resize는 `nearest-exact` 또는 검증한 동일 좌표 규약을 사용한다. RGB 색상 마스크에 bilinear 보간을 하지 않는다. PyTorch는 `nearest`와 `nearest-exact`를 별도 모드로 제공한다.[^resize]

### 8.3 광학 증강

아래는 `[0,1]` 밝기 기준의 탐색 시작 범위다.

```text
적용 확률 0.5:
  contrast gain: [0.90,1.10]
  brightness offset: [-0.04,0.04]
  gamma: [0.90,1.10]

선택적 노이즈:
  Gaussian sigma: [0,0.01]
```

그레이스케일 데이터에 임의의 hue/saturation 변화는 기본으로 넣지 않는다. 강한 블러·JPEG 손상·탄성 변형·MixUp·CutMix도 기본값에서 제외하고 독립 실험으로만 추가한다. 특히 CutMix는 원본 전체 문맥과 로컬 내용이 불일치하지 않도록 두 경로를 함께 구성해야 한다.

---

## 9. 단계별 학습 설정

### 9.1 optimizer와 수치 설정

| 항목 | 시작값 |
|---|---|
| optimizer | AdamW |
| decoder/detail/fusion weight decay | 0.01 |
| fine-tuning backbone weight decay | 0.05 |
| norm·bias weight decay | 0 |
| effective batch | 이미지/crop 8개 |
| micro-batch | 우선 GPU당 1개 |
| gradient accumulation | `8 / (micro_batch × world_size)`가 정수가 되도록 설정 |
| gradient clipping | global norm 1.0 |
| AMP | bfloat16 지원 장치에서 bf16, 손실은 fp32 |
| EMA | optimizer update 기준 decay 0.999 |
| scheduler | 각 stage별 warmup 후 cosine, 마지막 LR는 최대의 1% |
| full validation | 500 optimizer updates마다, stage 전환 시에도 실행 |

bf16 미지원이면 fp32 또는 fp16+GradScaler를 별도 설정으로 쓴다. gradient accumulation의 손실 배율, scaler의 unscale 이후 clipping, EMA update 시점을 일관되게 구현한다. PyTorch AMP 예제는 이 연산 순서를 확인하는 기준이다.[^amp]

### 9.2 주력 H+ 모델의 시작 학습 계획

아래 update 수는 최대 탐색 예산의 예시이며 전부 수행해야 한다는 뜻은 아니다. 모든 stage에서 최고 검증 체크포인트를 보존한다.

| stage | 입력 | backbone 학습 범위 | 신규 모듈 LR | backbone 최대 LR | update 예산 |
|---|---|---|---:|---:|---:|
| S0 | core 512 + halo 64 | 전체 고정 | 3e-4 | 0 | 2,000 |
| S1 | core 512/768 혼합 + halo 64 | 마지막 8 blocks와 출력 norm | 1e-4 | 5e-6 | 12,000 |
| S2 | core 768 + halo 64 | 전체 또는 마지막 절반 비교 | 5e-5 | 2e-6 | 8,000 |
| S3 | 원해상도 full-frame | S2에서 선택한 범위 | 1e-5 | 5e-7 | 3,000 |

S0 warmup은 200 updates, 나머지는 각 stage의 처음 5%를 시작값으로 둔다. S2의 전체 fine-tuning에는 layer-wise LR decay 0.9를 적용해 입력에 가까운 block의 LR를 더 작게 둔다. S3에서는 광학 증강을 약하게 하거나 끄는 설정을 비교한다.

L 모델은 backbone LR `1e-5`부터도 비교한다. 7B는 우선 고정 특징 또는 마지막 block 일부 학습부터 시작한다. 7B 전체 fine-tuning은 별도 실험이며, 단순히 가능하다는 이유로 필수 단계로 만들지 않는다.

### 9.3 stage 전환과 체크포인트

각 stage는 **직전 stage의 최고 검증 모델**에서 시작한다. optimizer/scheduler를 초기화할지 이어갈지 명시하며, 기본은 stage별로 새로 구성한다. 새로 unfrozen된 파라미터가 optimizer에 실제로 포함되었는지 검사한다.

12번의 검증 동안 개선이 없으면 해당 stage 종료를 검토한다. 아주 작은 개선을 무시하는 threshold를 고정적인 성능 규칙으로 두지 말고, raw 점수·seed 편차·이미지별 변화를 함께 기록한다.

다음 항목을 함께 저장한다.

```text
model, ema_model, optimizer, scheduler, scaler,
optimizer_step, stage_step, trainable_parameter_groups,
random_states, resolved_config, dataset_hash,
train_ids, validation_ids, fold_id, best_metric,
backbone_commit, pretrained_sha256, palette
```

검증 때 raw 가중치와 EMA 가중치를 모두 비교할 수 있다. 다만 최종 OOF 생성 및 추론에서는 선택된 방식 하나를 명시적으로 고정한다.

### 9.4 backbone 모드와 메모리

고정 backbone은 `torch.no_grad()`로 특징을 얻고, 그 뒤 projection/fusion은 일반 autograd에서 학습한다. backbone 특징을 만드는 부분 전체를 `inference_mode()`로 감싼 채 trainable layer에 연결하면 저장 tensor 관련 문제가 생길 수 있으므로 학습 코드에서는 no_grad를 기본으로 사용한다.

fine-tuning backbone에는 no_grad나 detach를 적용하지 않는다. 여러 crop에 같은 global 특징 graph를 공유한다면 손실들을 합쳐 한 번 backward하거나 전체 graph의 수명을 정확히 관리한다. 단순히 메모리를 줄이려고 context 특징을 detach하면 의도한 fine-tuning이 사라진다.

메모리가 부족하면 micro-batch 축소·gradient accumulation·activation checkpointing·분산 shard를 우선 적용한다. 그 뒤에도 부족할 때만 context 해상도를 줄이는 변형을 별도로 평가한다. 메모리 사정으로 원본 전체를 조용히 512 정사각형으로 바꾸지 않는다.

---

## 10. 평가와 모델 선택

### 10.1 목표 지표

공식 설명은 각 평가 이미지의 네 상 mIoU를 계산한 다음 이미지들에 대해 평균하는 방식이다.[^competition]

```text
IoU_i,c = TP_i,c / (TP_i,c + FP_i,c + FN_i,c)
score   = 100 * mean_over_images( mean_over_classes(IoU_i,c) )
```

전체 데이터셋의 confusion matrix를 합친 뒤 한 번 mIoU를 계산한 값을 주지표로 사용하지 않는다. 그 수치는 별도 진단 지표로만 저장한다. 검증은 모든 원본 픽셀에서 실시하고, 패치 표본 mIoU로 대체하지 않는다.

### 10.2 union=0

현재 공개 설명에는 정답과 예측 모두에 없는 클래스의 처리 방식까지 명시되어 있지 않다. 이를 임의로 공식 규칙이라고 단정하지 않는다.

프로젝트 evaluator는 아래 세 정책을 모두 지원하고 기록한다.

| 정책 | 정답·예측 모두에 없는 클래스 |
|---|---|
| `exclude` | 그 이미지의 클래스 평균에서 제외 |
| `one` | IoU 1로 계산 |
| `zero` | IoU 0으로 계산 |

개발 기본값은 이전 분석 코드와의 비교를 위해 `exclude`로 두되, 세 점수를 함께 저장한다. 정답에는 없지만 예측이 있으면 union은 양수이므로 **어느 정책에서도 IoU는 0**이다. 정답에 존재하는데 예측하지 않은 경우도 0이다.

정답에 초정 Si가 없는 영상이 실제로 존재한다. 따라서 이 분기 테스트를 생략하지 않는다.[^data] 정책별 순위가 바뀌는 개선은 주의해서 해석하고, 공식 evaluator가 제공되면 해당 설정에 맞춘다. 이 불확실성이 나머지 구현을 막도록 하지는 않는다.

### 10.3 반드시 저장할 평가 결과

```text
image_id, H, W, fold,
IoU_primary_Si, IoU_Al3Ni, IoU_eutectic_Si, IoU_Al,
mIoU_exclude, mIoU_one, mIoU_zero,
GT_phase_fractions, predicted_phase_fractions,
confusion_matrix, inference_config_hash
```

추가 진단으로 해상도별 점수, 공정 Si 비율별 점수, 경계 근처/내부 오류, 클래스별 FP/FN, 이미지별 최악 사례를 저장한다. 이것들은 최종 주지표를 대체하지 않는다.

### 10.4 선택 기준

동일 원본 이미지에 대한 점수 차이를 비교한다. 가능하면 이미지 또는 실제 독립 그룹 단위의 paired bootstrap과 여러 seed 결과도 확인한다. 픽셀을 독립 표본으로 간주한 신뢰구간은 사용하지 않는다.

검증 한 장만 크게 좋아지고 나머지가 나빠진 경우, 평균 개선과 실패 위험을 함께 기록한다. 주관적으로 더 매끈해 보이는 마스크를 mIoU가 낮은데도 선택하지 않는다.

---

## 11. 이종 모델과 앙상블

### 11.1 B: ConvNeXt 기반 모델

공식 DINOv3의 ConvNeXt-L 사전학습 인코더를 후보로 사용한다. 이 모델의 native feature는 계층별 해상도가 다르므로, ViT의 동일 stride 16 다중 층 출력 처리와 구분한다. 실제 feature shape를 로드 후 검사한다.[^dino][^convnext]

ConvNeXt의 깊은 특징에 U-Net 디코더를 붙이고, 별도 raw-image stem에서 stride 1·2 특징을 생성해 마지막 복원에 연결한다. 사전학습된 3채널 stem을 임의로 정규화/CDF 채널용으로 바꾸지 않고, 복제 그레이스케일을 입력으로 사용한다.

### 11.2 C: HRNet 기반 모델

HRNet-W48 계열을 후보로 사용하고, 병렬 특징을 결합한 뒤 stride 2·1의 원본 특징 경로를 붙인다. HRNet의 “고해상도” 특징이 원본과 같은 1/1 해상도라고 가정하지 않는다. 실제 backbone 출력 stride를 확인한다.[^hrnet]

배치가 작은 경우 사전학습 BatchNorm을 고정할지, 충분한 동기화 batch를 사용할지 비교한다. 모든 BN을 임의로 GN으로 교체한 모델은 같은 사전학습 모델이 아니므로 별도 변형으로 기록한다.

### 11.3 확률 앙상블

각 모델의 logits에 softmax를 적용하고 원본 좌표로 복원한 뒤 결합한다. 시작값은 **구조별 동일 가중치**다. 특정 구조의 seed 수가 많다는 이유로 영향력이 자동 증가하지 않도록 먼저 seed 평균을 만들고 그다음 구조를 결합한다.

```text
P_family = mean_over_seeds(P_seed)
P_base   = sum_m(w_m * P_family_m)
w_m >= 0, sum_m(w_m) = 1
```

가중치는 OOF 예측에서 조정하며, 외부 검증에서 이득을 확인한다. 단독 점수가 높더라도 추가 앙상블 이득이 없으면 가중치 0을 허용한다. 클래스별·이미지별 복잡한 gating은 최초 설정에서 제외한다.

기존 Extra Trees도 동일 조건의 OOF를 새로 생성해 후보로 넣을 수 있지만, 예전 70장 학습 모델의 예측을 90장 전체에 대한 정직한 OOF처럼 재사용해서는 안 된다.

---

## 12. OOF 및 중첩 검증

### 12.1 base 모델의 cross-fitting

정답이 있는 이미지 ID를 기준으로 fold를 나눈다. 같은 원본의 patch는 무조건 같은 fold다. A-prefix 그룹 분할을 5-fold 시작 후보로 사용하되, prefix가 실제 독립 시편 단위라는 보장은 없음을 기록한다. 공식 70/20 성능과 group CV 성능을 함께 본다.[^data]

클래스 부재, 해상도, 상 비율이 지나치게 한 fold로 몰리지 않도록 fold 전체 분포를 검사한다. 초정 Si 부재 영상 수가 적으므로 모든 fold에 이를 균등 배분할 수 있다고 가정하지 않는다. fold는 생성 후 JSON으로 고정한다.

```text
for heldout_fold k:
    model_k = train(images not in fold k)
    P_i = predict(model_k, image_i), for image_i in fold k
```

OOF 파일은 `[4,H,W]` float32 확률을 저장한다. 같은 파일에 image ID, 원본 크기, 클래스 순서, 모델/weights hash, 학습 ID 집합 hash, fold, TTA 설정을 연결한다. 학습 대상 이미지를 직접 학습한 가중치의 예측은 OOF 디렉터리에 저장하지 못하도록 검사한다.

홀드아웃 정답을 보고 epoch를 고르거나 normalization·pseudo-label·calibration을 조정하는 것도 선택 정보의 사용이다. 엄격한 외부 평가에서는 그러한 선택을 모두 외부 학습 영역 내부에서 수행한다. OOF 생성에는 내부에서 정한 고정 학습 길이 또는 더 안쪽 검증으로 선택한 체크포인트를 사용한다.

### 12.2 보정기의 간접 누수

**전체 데이터에서 한 번 만든 base OOF를 나눠 보정기만 교차검증하는 것으로는 전체 시스템의 독립성이 보장되지 않는다.**

예를 들어 외부 검증 E의 정답으로 학습된 base 모델이 보정기 학습용 T 이미지의 OOF를 만들었다면, T의 예측 특징을 통해 E 정보가 간접적으로 들어갈 수 있다. 보정기 자체가 E를 직접 학습하지 않았다고 끝나는 문제가 아니다.

### 12.3 외부 fold 하나를 평가하는 올바른 절차

```text
전체 labeled set D
외부 검증 E = outer fold k
외부 학습 T = D \ E

1. T 안에서만 inner folds를 만든다.
2. 각 inner fold를 제외한 T로 base 모델을 학습한다.
3. inner holdout 예측을 모아 T에 대한 base OOF를 만든다.
4. T의 OOF와 정답만 사용해 ensemble/refiner/calibration을 개발한다.
5. base 모델을 T 전체로 학습한다.
6. E의 입력을 예측한다.
7. 4에서 고정한 ensemble/refiner/calibration을 적용한다.
8. 마지막에만 E 정답으로 전체 시스템 점수를 계산한다.
```

가중치 선택, refiner 길이, 최종 편향값을 비교하는 내부 성능이 필요하면 T 안에서 추가 cross-fitting을 수행한다. **외부 E는 모든 선택 과정에서 제외**한다. 미리 전체 데이터로 만든 OOF/teacher/통계/SSL 체크포인트를 외부 fold에 재사용하지 않는다.

보정기 학습 입력을 만드는 ensemble 가중치는 처음에는 uniform으로 두면 의존관계가 단순하다. 학습된 ensemble 가중치를 입력 생성에 사용하려면 그 가중치가 어느 정답으로 학습됐는지도 기록한다.

### 12.4 최종 학습용 OOF

전체 시스템 선택을 끝낸 뒤 90장에 대해 base OOF를 다시 생성하고 최종 refiner를 학습한다. 이 refiner의 자기 학습 데이터 점수를 독립 검증 점수로 보고하지 않는다.

OOF 생성과 test 추론은 TTA·배율·seed 평균 방식·확률 정규화·보정 순서가 같아야 한다. 예를 들어 OOF는 단일 seed인데 test는 20개 모델 평균이면 예측 불확실성 분포가 달라질 수 있다. 모델 수와 학습 데이터 수의 차이에 따른 refiner 입력 분포 변화도 외부 검증에서 점검한다.

---

## 13. 원해상도 잔차 보정기

### 13.1 입력 8채널

```text
R_input = concat(
  raw_gray,               # 1
  P_base,                 # 4
  normalized_entropy,     # 1
  sobel_x, sobel_y         # 2
)                         # 합계 8
```

`entropy = -sum(P*log(P+eps))/log(4)`로 정의한다. Sobel은 원본 `[0,1]` 밝기에서 고정된 정규화 kernel로 계산하고, 두 signed 기울기를 모두 사용한다. 학습과 추론의 kernel·padding·scale을 일치시킨다.

### 13.2 구조와 출력

stride 1/2/4/8, 채널 `[32,64,128,256]`인 작은 residual U-Net을 사용한다. 마지막 네 채널 convolution의 weight와 bias는 0으로 초기화한다.

```text
delta = Refiner(R_input)
delta = delta - mean_over_classes(delta)
Z_ref = log(clamp(P_base, min=1e-7)) + delta
P_ref = softmax(Z_ref)
```

초기 상태는 clamp/정규화의 작은 수치 차이를 제외하면 base 예측과 같아야 한다. `P_base`는 저장한 OOF에서 읽고 detach한다. 보정기 학습으로 base 모델까지 다시 학습하지 않는다.

### 13.3 학습

시작 손실은 `CE + 0.5 Dice + 0.5 Lovasz + 1e-4 mean(delta²)`다. 잔차 벌점은 작은 수정부터 시작하기 위한 실험 항목이며, 0과 비교한다.

AdamW LR `1e-4`, weight decay `0.01`, 원해상도 patch 512/768, 최종 full-frame LR `1e-5`를 시작값으로 둔다. 최대 6,000 updates에서 검증하고, 앞선 base 학습량과 별개로 최고 외부 검증 설정을 선택한다.

보정기는 전체 영역을 처리한다. 불확실한 픽셀만 처리하면 **확신하면서 틀린 픽셀**은 수정 기회를 잃을 수 있다. 불확실성은 입력 특징으로 쓰되 hard gating으로 쓰지 않는다.

OOF가 완벽한 마스크에 가까운 일부 영상에만 편중되지 않도록 원본별 균등 표본을 사용한다. 보정기가 작은 공정 Si를 삭제하거나 경계를 두껍게 만들어 mIoU를 낮추면 최종 구성에서 빼야 한다.

---

## 14. 추론·TTA·클래스 보정

### 14.1 순서 고정

```text
모델별:
  영상 변환 → 예측 → 패딩 제거 → 역변환 → 원본 크기 복원
  → 확률 정규화 → 변환별 확률 평균

구조/seed 결합:
  seed 평균 → 구조별 가중 평균 → P_base

후단:
  원본 좌표에서 refiner 한 번 적용 [선택]
  → 클래스 편향 적용 [선택]
  → argmax 한 번 → 색상 인코딩
```

refiner를 TTA별로 적용한 뒤 평균하는 방식은 다른 파이프라인이다. 시험할 수는 있지만, 보정기 학습 입력과 평가 순서를 그 방식에 맞춰 새로 구성해야 한다.

### 14.2 TTA 후보

시작 후보는 D4 8개와 배율 `[1.0,1.25,1.5]`다. 모든 변환을 항상 쓰는 것이 아니라 단일 배율, D4, multi-scale의 추가 이득을 분리한다. 축소 배율 0.75도 후보일 뿐 기본 포함하지 않는다.

역변환은 적용 순서의 역순으로 수행한다. 회전으로 H/W가 바뀐 경우, 패딩 제거 좌표도 변환된 영상 기준이어야 한다. resize의 실제 정수 크기를 저장하고 복원에는 `size=(H_original,W_original)`를 명시한다.

평균은 float32 확률에서 수행한다. 복원·보간 후 클래스 확률 합을 다시 1로 정규화한다. 클래스 번호·RGB 마스크·argmax 결과를 평균하지 않는다.

### 14.3 타일 추론

full-frame 추론과 비교해 선택한다. 타일 모드에서는 DINO 전체 문맥을 해당 변환에 대해 한 번 계산하고, CNN tile마다 ROI 특징을 샘플링한다.

중앙 tile 크기 768, halo 64, 중앙 tile 간 overlap 50%를 시작값으로 둔다. 타일의 유효 중앙부 확률을 Gaussian 또는 양수 바닥값을 둔 Hann 가중치로 누적한다.

```text
P(x) = sum_t(w_t(x) * P_t(x)) / sum_t(w_t(x))
```

모든 원본 픽셀에서 분모가 양수여야 한다. 마지막 행/열이 누락되지 않도록 끝 타일 좌표를 명시적으로 추가한다. 타일별 평균 확률 합도 다시 정규화한다.

### 14.4 클래스별 편향

최종 후보는 다음처럼 클래스별 작은 편향을 더하는 방식이다.

```text
prediction = argmax_c(log(P_final[c] + eps) + b[c])
```

softmax/argmax는 모든 클래스에 같은 상수를 더해도 변하지 않으므로 `b[3]=0` 또는 `sum(b)=0`으로 자유도를 3개로 제한한다. 시작 탐색 범위는 각 자유 변수 `[-0.3,0.3]`이다. 편향은 내부 cross-fit 예측에서 조정하고 외부 fold에서 비교한다.

공통 온도를 마지막 단일 확률에 적용하는 것만으로는 argmax가 달라지지 않는다. 온도 조정의 효과를 실험하려면 모델 평균 전 또는 클래스 편향과의 상호작용처럼 실제 결정에 영향을 주는 위치를 명확히 한다.

작은 연결 성분 제거·opening/closing·구멍 채우기·CRF는 기본값에서 모두 끈다. 실제 작은 상을 노이즈로 지울 수 있기 때문이다. 적용하려면 클래스별 효과를 독립적으로 측정한다.

---

## 15. 추가 성능 실험

### 15.1 자기지도 도메인 적응

외부 pretrained backbone을 사용한 상태와, 허용된 OM 입력으로 추가 자기지도 학습한 상태를 비교한다. 90~100장만으로 대형 backbone을 추가 학습하면 항상 개선된다고 가정하지 않는다.

엄격한 비전이 검증에서는 외부 검증 이미지도 자기지도 학습에 넣지 않는다. 최종 테스트 입력까지 사용하는 실험은 **transductive 변형**으로 별도 표시한다. 평가 시에도 보류 영상을 정답 없는 입력으로만 제공하는 동일 프로토콜을 재현한다.

### 15.2 의사라벨 학습

teacher는 선택된 base 또는 전체 파이프라인으로 고정한다. 테스트 10장의 입력으로 확률 예측을 만들고, 모델 간·변환 간 일치도를 이용해 사용할 영역을 고른다.

```text
L = L_supervised_90 + lambda_u * L_consistency
```

초기 `lambda_u=0`에서 0.3까지 늘리는 설정을 후보로 둔다. 클래스별 신뢰도 threshold는 내부 모의 실험에서 정한다. 하나의 높은 전역 threshold가 Al만 남기고 공정 Si를 거의 제거하지 않는지 확인한다.

약한 변환 teacher와 강한 변환 student를 사용하는 일관성 학습은 UniMatch V2 등 반지도 분할 연구를 참고할 수 있지만, 그 논문의 설정이 본 데이터에서 최적인 것은 아니다.[^unimatch]

의사라벨을 원본 정답 폴더에 섞지 않는다. 별도 provenance와 teacher hash를 저장하고, supervised 정답 90장의 손실이 중심이 되게 한다. 생성형으로 이미지를 만들었다는 이유만으로 그 이미지에 정확한 상 정답이 있다고 간주하지 않는다.

### 15.3 구조·손실 추가 후보

원본 세부 채널, 마지막 DINO 층만/다중 층, detail 폭, 더 넓은 receptive field, local texture auxiliary head, class-balanced loss, hard pixel mining, v2 Adapter, 7B를 각각 분리한다. 모든 후보를 한 번에 넣지 않는다.

면적이 적다는 이유로 inverse-frequency weight를 바로 강하게 주지 않는다. 공정 Si의 recall 증가와 거짓 양성 증가를 함께 봐야 한다. mIoU 손실과 균형 표본을 적용한 기본형에 추가 이득이 있는지 검증한다.

---

## 16. 전체 90장 재학습과 제출

### 16.1 최종 재학습

외부 검증에서 선택한 구조·손실·증강·TTA·보정 순서를 고정한다. 그 뒤 선택된 base 모델을 90장 전체로 재학습한다.

학습 길이는 CV에서 정한 값을 사용한다. fold당 이미지 수가 다르므로 고정 update 수와 이미지당 노출량 기준을 구분한다. 예를 들어 `exposure = updates × effective_batch / n_train_images`를 기록하고, 어느 기준을 유지했는지 명시한다. 90장 재학습 중 자기 학습 점수를 보고 epoch를 새로 최적화하지 않는다.

최종 refiner는 90장 base OOF와 정답으로 학습한다. test 입력에는 90장 전체로 학습한 base seed ensemble을 적용한다. OOF보다 더 많은 학습 데이터로 인해 확률 분포가 달라지는 효과는 외부 fold의 같은 상황에서 미리 비교한다. full-data ensemble과 fold ensemble 중 어느 쪽을 사용할지도 그 비교로 선택한다.

### 16.2 출력 규격

최종 출력은 각 test 입력과 같은 H/W의 lossless PNG다. class ID로 결정한 다음 1.2절 palette로 변환한다. 현재 정답 파일과 형식을 맞추는 프로젝트 기본값은 alpha=255인 RGBA이며, evaluator에 맞춰 RGB 출력도 지원한다.

`*_image.png → *_mask.png`를 프로젝트 기본 파일명 규칙으로 정한다. 이는 실제 입력/학습 정답 매칭에 따른 내부 규칙이며, 공개 페이지에서 제출 파일명까지 그렇게 명시했다는 뜻은 아니다. 최종 exporter의 naming 옵션으로 변경 가능하게 한다.

```text
submission/
├── predictions/                    # 10개 PNG
├── inference_manifest.json         # 파일명, 크기, hash, 모델 설정
├── final_inference.yaml
├── environment.yaml
├── requirements.lock.txt
└── README_inference.md
```

공식 제출 안내는 프로그램 소스·실행 관련 파일·개발 환경 YAML과 테스트 masking 이미지 10개 등을 요구한다. 최종 공모안은 별도 제출물이며 이 구현 문서 자체가 공모안의 정해진 분량을 대체하는 것은 아니다.[^submission]

### 16.3 최종 자동 검사

테스트 입력 ID 집합과 출력 ID 집합이 정확히 같고, 10장 모두 크기·색상·알파 조건을 만족해야 한다. 예측에 255 ignore ID가 남아 있거나 확률에 NaN이 있으면 export를 중단한다.

새 프로세스에서 체크포인트를 다시 읽고 추론해 결과를 확인한다. 네트워크 없이도 로컬 가중치만으로 실행되게 한다. 제출용 모드에서는 임의 증강 RNG, dropout, train-time 좌표 증강을 끈다. 결정성 설정과 허용 오차도 기록한다.

---

## 17. 구현 완료 기준과 실험표

### 17.1 최소 테스트

| 테스트 | 통과 조건 |
|---|---|
| palette 왕복 | 원본 RGB → ID → RGB가 모든 픽셀에서 동일 |
| 알 수 없는 색 | 오류 발생, 자동 최근접 변환 금지 |
| 파일 매칭 | 70/20쌍과 test 10개가 중복 없이 확인됨 |
| 출력 shape | 모든 실제 해상도에서 원본 H/W로 복원됨 |
| 전체/ROI 좌표 | full box ROI와 같은 규약의 resize가 허용 오차 내 일치 |
| 좌표 ramp | 합성 좌표 특징에서 예상 위치의 값 반환 |
| D4 왕복 | 8개 변환 모두 inverse(transform(x)) = x |
| 타일 coverage | 모든 원본 픽셀의 누적 가중치 > 0 |
| constant predictor | 타일/TTA를 거쳐도 동일한 확률 유지 |
| mIoU perfect | 네 클래스 모두 존재하면 세 정책 모두 1 |
| class absent | 없는 클래스 및 거짓 양성의 정책별 기대값과 일치 |
| 이미지별 평균 | 다른 크기 두 이미지에서 수기 계산과 일치 |
| ignore | 패딩 값을 바꿔도 유효 픽셀 손실과 점수 불변 |
| backbone freeze | S0에서 backbone gradient 없음, head에는 있음 |
| backbone unfreeze | 지정 block에 유한한 gradient가 실제 존재 |
| refiner 초기화 | delta=0일 때 base 예측과 일치 |
| OOF provenance | 해당 이미지가 base 학습 ID에 없음을 확인 |
| 외부 fold 격리 | refiner용 내부 OOF를 포함해 외부 정답 미사용 |
| checkpoint 왕복 | 저장·재로드 후 eval 예측 일치 |
| 제출 roundtrip | PNG를 다시 decode하면 원래 예측 ID와 일치 |

작은 원해상도 patch 1~2개에서 증강 없이 정답을 거의 외우는지도 확인한다. 이것이 전체 데이터 성능의 증거는 아니지만, loss/label/gradient 배선 오류를 찾는 중요한 진단이다.

### 17.2 실험 장부

| ID | 실험 | 고정할 비교 조건 |
|---|---|---|
| E00 | CNN-only, raw 입력, 원해상도 decoder | fold·seed·학습 예산 |
| E01 | raw + z-score + CDF | E00에서 입력 채널만 변경 |
| E02 | DINO-L 마지막 층 추가 | 선택한 CNN 입력 유지 |
| E03 | DINO-L 4개 층 | E02에서 context 층 구성만 변경 |
| E04 | DINO-H+ | 공통 decoder·표본·검증 사용 |
| E05 | 마지막 block만 vs 전체 fine-tuning | 동일 initialization과 나머지 설정 |
| E06 | full-frame 미세조정 | 미세조정 유무와 길이 비교 |
| E07 | aux soft target | main loss 고정 |
| E08 | boundary auxiliary | E07 여부 고정하고 분리 |
| E09 | ConvNeXt/HRNet | 같은 평가기와 fold 사용 |
| E10 | D4 및 multi-scale | 동일 체크포인트의 추론만 변경 |
| E11 | 확률 앙상블 | 외부 holdout에서 추가 기여 확인 |
| E12 | OOF refiner | 전체 stack을 중첩 평가 |
| E13 | 클래스 편향 | calibration 외부 검증 |
| E14 | v2 Adapter/7B/의사라벨 | 각각 독립 비교 |

각 행에 이미지별 mIoU, 클래스별 IoU, 세 empty 정책 점수, 변경점, GPU 메모리, 업데이트 수, 파라미터 수, 모델 hash를 기록한다. 예산 제한이 없더라도 비교 가능한 실험 기록은 유지한다.

---

## 18. CLI와 산출물 계약

아래 명령은 **개발할 CLI의 인터페이스 명세**다. 20절 참조 코드만으로 이 명령들이 이미 존재하는 것은 아니다. 명령 구현 후 이 계약을 smoke test에 사용한다.

```bash
# 1. 데이터 검사 및 원본 ID manifest
python -m alphase.cli.audit --root data/raw --out data/processed/audit.json
python -m alphase.cli.prepare --root data/raw --out data/processed

# 2. 고정 분할 생성
python -m alphase.cli.make_folds \
  --manifest data/processed/manifest.csv --scheme group --folds 5 --seed 2026

# 3. 공식 70/20 분할에서 기준 학습
python -m alphase.cli.train --config configs/hybrid_hplus.yaml \
  --split official --set model.pretrained_checkpoint=/absolute/path/to/original_checkpoint_filename.pth

# 4. 독립적인 전체 영상 평가
python -m alphase.cli.validate \
  --checkpoint runs/hybrid_hplus/official/best.pt --split valid --save-probabilities

# 5. 중첩 평가: 내부 OOF를 외부 fold마다 새로 생성
python -m alphase.cli.run_outer_cv \
  --config configs/final_inference.yaml --outer-folds 5 --inner-folds 4

# 6. 설정 선택 완료 후 최종 base OOF와 refiner
python -m alphase.cli.generate_oof --config configs/final_inference.yaml --labeled all90
python -m alphase.cli.train_refiner --config configs/refiner.yaml --oof oof/final90

# 7. base 전체 재학습과 최종 추론
python -m alphase.cli.train_final --config configs/final_inference.yaml --labeled all90
python -m alphase.cli.predict --config configs/final_inference.yaml \
  --input data/raw/test/images --output submission/predictions

# 8. 결과 검증과 패키징
python -m alphase.cli.check_submission \
  --inputs data/raw/test/images --predictions submission/predictions
```

`--set key=value`는 OmegaConf dotlist override로 구현한다. 모든 CLI는 적용 완료된 설정을 출력 경로에 저장하고, 기존 run을 덮어쓰려면 별도 `--overwrite`를 요구한다. 하이퍼파라미터가 다르거나 데이터 hash가 달라지면 같은 run ID로 resume하지 못하게 한다.

각 단계의 필수 산출물:

```text
audit:        audit.json, manifest.csv, palette.json
split:        folds.json, split_summary.csv
train:        best.pt, last.pt, resolved_config.yaml, metrics.jsonl
validate:     per_image_metrics.csv, metrics_summary.json, probabilities/
oof:          probabilities/, provenance.jsonl
ensemble:     weights.json, inference_config.yaml
refiner:      best_refiner.pt, input_schema.json
final:        모델 가중치 목록, config, test PNG 10개, inference_manifest.json
```

---

## 19. 설정 파일 예시

아래는 H+ 주력 후보의 시작 설정이다. `pretrained_checkpoint`가 null이면 실행 시 오류를 내도록 구현한다. 이 값들은 최고 성능이 검증된 조합이 아니다.

```yaml
project: alphase
experiment: hybrid_hplus_v1
seed: 2026

data:
  root: data/raw
  manifest: data/processed/manifest.csv
  folds: data/processed/folds.json
  classes: [primary_si, al3ni, eutectic_si, al]
  palette: [[76, 178, 76], [25, 76, 153], [204, 204, 204], [153, 127, 76]]
  ignore_index: 255
  image_channels: [gray, image_zscore, image_cdf]
  pad_multiple: 32
  image_pad_mode: reflect
  mask_pad_value: 255

model:
  name: alphase_hybrid_v1
  dino_repo: third_party/dinov3
  backbone_name: dinov3_vith16plus
  pretrained_checkpoint: null
  dino_embed_dim: 1280
  dino_layers: [7, 15, 23, 31]
  dino_eval_mode: true
  context_scale: 1.0
  context_channels: 256
  detail_channels: [32, 64, 128, 256, 384]
  residual_blocks_per_stage: 2
  fusion_strides: [4, 8, 16]
  norm: groupnorm
  norm_groups: 8
  upsample_mode: bilinear
  align_corners: false
  aux_head: true
  boundary_head: true

sampling:
  sample_image_uniformly: true
  core_sizes: [512, 768]
  halo: 64
  mix:
    uniform: 0.60
    eutectic_center: 0.25
    error_or_boundary: 0.15

augmentation:
  d4: true
  scales: [1.0]
  photometric_probability: 0.5
  brightness_offset: [-0.04, 0.04]
  contrast_gain: [0.90, 1.10]
  gamma: [0.90, 1.10]
  gaussian_noise_std: [0.0, 0.0]
  strong_blur: false
  elastic: false
  mixup: false
  cutmix: false

loss:
  ce: 1.0
  dice: 0.5
  lovasz: 0.5
  per_image: true
  dice_classes: present
  lovasz_classes: present
  class_weights: null
  auxiliary: 0.0
  auxiliary_target: area_fraction
  boundary: 0.0
  float32_reductions: true

train:
  optimizer: adamw
  new_weight_decay: 0.01
  backbone_weight_decay: 0.05
  norm_bias_weight_decay: 0.0
  effective_batch_size: 8
  micro_batch_size: 1
  precision: bf16
  gradient_clip_norm: 1.0
  ema_decay: 0.999
  eval_every_updates: 500
  no_improvement_checks: 12
  stages:
    - name: s0_frozen
      max_updates: 2000
      mode: local_detail_global_context
      core_sizes: [512]
      backbone_last_n: 0
      new_lr: 0.0003
      backbone_lr: 0.0
      warmup_updates: 200
    - name: s1_last_quarter
      max_updates: 12000
      mode: local_detail_global_context
      core_sizes: [512, 768]
      backbone_last_n: 8
      new_lr: 0.0001
      backbone_lr: 0.000005
      warmup_updates: 600
    - name: s2_finetune
      max_updates: 8000
      mode: local_detail_global_context
      core_sizes: [768]
      backbone_last_n: -1
      layer_lr_decay: 0.9
      new_lr: 0.00005
      backbone_lr: 0.000002
      warmup_updates: 400
    - name: s3_fullframe
      max_updates: 3000
      mode: full_frame
      backbone_last_n: -1
      new_lr: 0.00001
      backbone_lr: 0.0000005
      warmup_updates: 150
      photometric_probability: 0.0

evaluation:
  primary_metric: image_mean_miou
  empty_union: exclude
  report_empty_policies: [exclude, one, zero]
  restore_original_resolution: true
  compute_on_all_pixels: true
  validation_tta: false

inference:
  mode: full_frame
  local_core_size: 768
  halo: 64
  overlap_fraction: 0.5
  tile_weight: gaussian
  d4: false
  scales: [1.0]
  probability_dtype: float32
  refiner_enabled: false
  class_logit_bias: [0.0, 0.0, 0.0, 0.0]
  output_mode: RGBA
```

모델 탐색 단계의 기본 추론은 TTA 없이 둔다. TTA/ensemble/refiner는 별도의 `final_inference.yaml`에서 검증 후 켠다. 이를 통해 모델 자체의 개선과 추론 비용 증가의 효과를 구분한다.

---

## 20. 핵심 참조 코드

이 절의 코드는 전체 프로젝트 중 색상 변환, 평가, 하이브리드 핵심 연산의 참조 구현이다. Dataset/Trainer/EMA/Lovasz/분산 학습/최종 앙상블 CLI는 앞의 계약에 따라 추가 구현해야 한다. pretrained 가중치를 불러온 실제 대형 모델의 학습 성능은 이 코드만으로 검증된 것이 아니다.

### 20.1 Palette codec

```python
from pathlib import Path
import numpy as np
from PIL import Image

PALETTE = np.asarray(
    [(76, 178, 76), (25, 76, 153), (204, 204, 204), (153, 127, 76)],
    dtype=np.uint8,
)


def decode_mask(path: str | Path) -> np.ndarray:
    """RGB/RGBA palette PNG -> int64 [H,W]. Unknown colors raise."""
    with Image.open(path) as image:
        if "A" in image.getbands():
            alpha = np.asarray(image.getchannel("A"))
            if np.any(alpha != 255):
                raise ValueError(f"Non-opaque annotation: {path}")
        rgb = np.asarray(image.convert("RGB"))
    labels = np.full(rgb.shape[:2], 255, dtype=np.int64)
    for class_id, color in enumerate(PALETTE):
        labels[np.all(rgb == color, axis=-1)] = class_id
    unknown = labels == 255
    if unknown.any():
        examples = np.unique(rgb[unknown], axis=0)[:8].tolist()
        raise ValueError(f"Unknown mask colors in {path}: {examples}")
    return labels


def save_mask(labels: np.ndarray, path: str | Path, rgba: bool = True) -> None:
    """Integer [H,W] -> exact-palette lossless PNG."""
    if labels.ndim != 2 or not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("Expected a two-dimensional integer label array")
    if labels.size == 0 or labels.min() < 0 or labels.max() >= len(PALETTE):
        raise ValueError("Label values must be in [0,3]; ignore is not exportable")
    rgb = PALETTE[labels]
    output = rgb
    if rgba:
        alpha = np.full((*labels.shape, 1), 255, dtype=np.uint8)
        output = np.concatenate([rgb, alpha], axis=-1)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(output).save(path, format="PNG")
```

### 20.2 이미지별 mIoU evaluator

각 이미지를 별도로 이 함수에 전달하고 마지막에 `np.mean(image_scores)`를 사용한다. 반환값은 0~1이며 표시 단계에서만 100을 곱한다.

```python
from typing import Literal
import numpy as np


def image_miou(
    target: np.ndarray,
    prediction: np.ndarray,
    empty_union: Literal["exclude", "one", "zero"] = "exclude",
    ignore_index: int = 255,
) -> tuple[float, np.ndarray, np.ndarray]:
    if empty_union not in {"exclude", "one", "zero"}:
        raise ValueError(f"Invalid empty-union policy: {empty_union}")
    if target.ndim != 2 or target.shape != prediction.shape:
        raise ValueError("Target and prediction must have the same [H,W] shape")
    if not np.issubdtype(target.dtype, np.integer):
        raise TypeError("Target must contain integer class IDs")
    if not np.issubdtype(prediction.dtype, np.integer):
        raise TypeError("Prediction must contain integer class IDs")

    valid = target != ignore_index
    if not valid.any():
        raise ValueError("The image has no valid ground-truth pixels")
    t = target[valid].astype(np.int64, copy=False)
    p = prediction[valid].astype(np.int64, copy=False)
    if t.min() < 0 or t.max() > 3 or p.min() < 0 or p.max() > 3:
        raise ValueError("Valid target/prediction IDs must be in [0,3]")

    cm = np.bincount(4 * t + p, minlength=16).reshape(4, 4)
    intersection = np.diag(cm).astype(np.float64)
    union = cm.sum(axis=0) + cm.sum(axis=1) - intersection
    iou = np.full(4, np.nan, dtype=np.float64)
    np.divide(intersection, union, out=iou, where=union > 0)
    if empty_union == "one":
        iou[union == 0] = 1.0
    elif empty_union == "zero":
        iou[union == 0] = 0.0
    score = float(np.nanmean(iou)) if empty_union == "exclude" else float(iou.mean())
    return score, iou, cm
```

검증 예: target과 prediction이 모두 Al만 있으면 `exclude=1`, `one=1`, `zero=0.25`다. 실제로 없는 초정 Si를 한 픽셀이라도 예측하면 그 클래스는 union이 양수가 되어 `exclude` 정책에서도 평가에 포함된다.

### 20.3 하이브리드 핵심 구현

다음 코드는 PyTorch backbone 객체를 받아 두 경로와 ROI 정렬, decoder를 연결한다. 공식 backbone을 넘기면 그 구조를 사용하고, 작은 가짜 backbone을 넘겨 연결·shape·gradient 테스트를 할 수도 있다. 데이터 정규화와 패딩은 호출 전에 수행한다.

`last_n=0`은 고정, 양수는 마지막 N blocks와 출력 norm 학습, `-1`은 전체 학습이다. 전체 영상 모드에서는 box를 생략한다. 로컬 모드에서는 **halo를 포함한 로컬 입력 전체의 box**를 넘기고, 손실 계산 때 중앙 core만 사용한다.


```python
from __future__ import annotations

from contextlib import nullcontext
from collections.abc import Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        if out_channels % 8:
            raise ValueError("out_channels must be divisible by 8")
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(8, out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(8, out_channels),
        )
        self.skip = (
            nn.Identity() if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, 1, bias=False)
        )
        self.activation = nn.GELU()

    def forward(self, x: Tensor) -> Tensor:
        return self.activation(self.body(x) + self.skip(x))


class DetailEncoder(nn.Module):
    def __init__(self, in_channels: int = 3) -> None:
        super().__init__()
        widths = (32, 64, 128, 256, 384)
        self.stem = nn.Sequential(
            ResidualBlock(in_channels, widths[0]),
            ResidualBlock(widths[0], widths[0]),
        )
        self.down = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(cin, cout, kernel_size=2, stride=2),
                ResidualBlock(cout, cout),
                ResidualBlock(cout, cout),
            )
            for cin, cout in zip(widths[:-1], widths[1:])
        ])

    def forward(self, x: Tensor) -> list[Tensor]:
        result = [self.stem(x)]
        for block in self.down:
            result.append(block(result[-1]))
        return result  # strides 1,2,4,8,16


def sample_context(
    feature: Tensor,
    boxes_xyxy: Tensor,
    global_hw: tuple[int, int],
    output_hw: tuple[int, int],
) -> Tensor:
    """Sample per-item boxes in padded global-image edge coordinates."""
    batch = feature.shape[0]
    if feature.ndim != 4 or boxes_xyxy.shape != (batch, 4):
        raise ValueError("Expected feature [B,C,Hf,Wf] and boxes [B,4]")
    gh, gw = global_hw
    oh, ow = output_hw
    if min(gh, gw, oh, ow) <= 0:
        raise ValueError("Spatial dimensions must be positive")

    device = feature.device
    box = boxes_xyxy.to(device=device, dtype=torch.float32)
    if not torch.isfinite(box).all() or (box[:, 2:] <= box[:, :2]).any():
        raise ValueError("Boxes must be finite and have positive width/height")

    # Keep coordinates and interpolation in float32, even with AMP enabled.
    with torch.autocast(device_type=device.type, enabled=False):
        ux = (torch.arange(ow, device=device, dtype=torch.float32) + 0.5) / ow
        uy = (torch.arange(oh, device=device, dtype=torch.float32) + 0.5) / oh
        x = box[:, 0, None] + (box[:, 2] - box[:, 0])[:, None] * ux[None]
        y = box[:, 1, None] + (box[:, 3] - box[:, 1])[:, None] * uy[None]
        gx = (2.0 * x / gw - 1.0)[:, None, :].expand(batch, oh, ow)
        gy = (2.0 * y / gh - 1.0)[:, :, None].expand(batch, oh, ow)
        grid = torch.stack((gx, gy), dim=-1)
        return F.grid_sample(
            feature.float(), grid, mode="bilinear",
            padding_mode="border", align_corners=False,
        )


class PhaseHybrid(nn.Module):
    """V1 core. Inputs must already be normalized, aligned and padded."""
    def __init__(
        self,
        backbone: nn.Module,
        layer_indices: Sequence[int],
        detail_in_channels: int = 3,
        context_channels: int = 256,
    ) -> None:
        super().__init__()
        indices = tuple(layer_indices)
        if len(indices) != 4 or tuple(sorted(set(indices))) != indices:
            raise ValueError("Provide four distinct ascending block indices")
        if indices[0] < 0 or indices[-1] >= len(backbone.blocks):
            raise ValueError("Block index out of bounds")
        if context_channels % 8:
            raise ValueError("context_channels must be divisible by 8")
        if int(backbone.patch_size) != 16:
            raise ValueError("This reference implementation assumes patch_size=16")

        self.backbone = backbone
        self.indices = indices
        self.detail = DetailEncoder(detail_in_channels)
        dim = int(backbone.embed_dim)
        self.projections = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(dim, context_channels, 1, bias=False),
                nn.GroupNorm(8, context_channels),
                nn.GELU(),
            ) for _ in indices
        ])
        self.context_mix = nn.ModuleDict({
            str(stride): nn.Sequential(
                nn.Conv2d(4 * context_channels, context_channels, 1, bias=False),
                nn.GroupNorm(8, context_channels),
                nn.GELU(),
            ) for stride in (4, 8, 16)
        })
        self.fusion = nn.ModuleDict({
            str(stride): ResidualBlock(width + context_channels, width)
            for stride, width in ((4, 128), (8, 256), (16, 384))
        })
        self.up8 = ResidualBlock(384 + 256, 256)
        self.up4 = ResidualBlock(256 + 128, 128)
        self.up2 = ResidualBlock(128 + 64, 64)
        self.up1 = ResidualBlock(64 + 32, 32)
        self.classifier = nn.Conv2d(32, 4, 1)
        self.aux_head = nn.Conv2d(128, 4, 1)
        self.boundary_head = nn.Conv2d(32, 1, 1)
        self.set_backbone_trainable(last_n=0)

    def train(self, mode: bool = True) -> PhaseHybrid:
        super().train(mode)
        # Deterministic backbone mode; parameter gradients are controlled below.
        self.backbone.eval()
        return self

    def set_backbone_trainable(self, last_n: int) -> None:
        depth = len(self.backbone.blocks)
        if last_n < -1 or last_n > depth:
            raise ValueError("last_n must be -1 or within [0, depth]")
        self.backbone.requires_grad_(False)
        if last_n == -1:
            self.backbone.requires_grad_(True)
        elif last_n > 0:
            for block in self.backbone.blocks[-last_n:]:
                block.requires_grad_(True)
            for name in ("norm", "cls_norm"):
                norm = getattr(self.backbone, name, None)
                if isinstance(norm, nn.Module):
                    norm.requires_grad_(True)
        self.backbone.eval()

    def encode_context(self, image_dino: Tensor) -> list[Tensor]:
        if image_dino.ndim != 4 or image_dino.shape[1] != 3:
            raise ValueError("DINO input must be [B,3,H,W]")
        h, w = image_dino.shape[-2:]
        if h % 32 or w % 32:
            raise ValueError("Pad global input dimensions to multiples of 32")
        needs_grad = any(p.requires_grad for p in self.backbone.parameters())
        guard = nullcontext() if needs_grad else torch.no_grad()
        with guard:
            features = self.backbone.get_intermediate_layers(
                image_dino, n=list(self.indices), reshape=True, norm=True,
                return_class_token=False, return_extra_tokens=False,
            )
        if len(features) != 4:
            raise RuntimeError("Backbone returned an unexpected number of features")
        for f in features:
            if f.shape[-2:] != (h // 16, w // 16):
                raise RuntimeError("Unexpected DINO feature grid")
        # Deliberately outside no_grad: projection parameters must learn in S0.
        return [project(f) for project, f in zip(self.projections, features)]

    @staticmethod
    def _up(x: Tensor, skip: Tensor) -> Tensor:
        return F.interpolate(
            x, size=skip.shape[-2:], mode="bilinear", align_corners=False,
        )

    def decode_local(
        self,
        image_detail: Tensor,
        projected_context: list[Tensor],
        boxes_xyxy: Tensor,
        global_hw: tuple[int, int],
    ) -> dict[str, Tensor]:
        if image_detail.ndim != 4:
            raise ValueError("Detail input must be [B,C,H,W]")
        if any(size % 32 for size in image_detail.shape[-2:]):
            raise ValueError("Pad local input dimensions to multiples of 32")
        if len(projected_context) != 4 or any(
            f.shape[0] != image_detail.shape[0] for f in projected_context
        ):
            raise ValueError("Context and detail batches must match")
        c1, c2, c4, c8, c16 = self.detail(image_detail)
        fused = {}
        for stride, local_feature in ((4, c4), (8, c8), (16, c16)):
            aligned = [
                sample_context(p, boxes_xyxy, global_hw, local_feature.shape[-2:])
                for p in projected_context
            ]
            context = self.context_mix[str(stride)](torch.cat(aligned, dim=1))
            fused[stride] = self.fusion[str(stride)](
                torch.cat((local_feature, context), dim=1)
            )
        d8 = self.up8(torch.cat((self._up(fused[16], fused[8]), fused[8]), dim=1))
        d4 = self.up4(torch.cat((self._up(d8, fused[4]), fused[4]), dim=1))
        d2 = self.up2(torch.cat((self._up(d4, c2), c2), dim=1))
        d1 = self.up1(torch.cat((self._up(d2, c1), c1), dim=1))
        return {
            "logits": self.classifier(d1),
            "aux_logits": self.aux_head(d4),
            "boundary_logits": self.boundary_head(d1),
        }

    def forward(
        self,
        image_detail: Tensor,
        image_dino: Tensor,
        boxes_xyxy: Tensor | None = None,
    ) -> dict[str, Tensor]:
        if image_detail.shape[0] != image_dino.shape[0]:
            raise ValueError("Context and detail batches must match")
        gh, gw = image_dino.shape[-2:]
        if boxes_xyxy is None:
            if image_detail.shape[-2:] != (gh, gw):
                raise ValueError("Local mode requires explicit global-coordinate boxes")
            boxes_xyxy = image_dino.new_tensor([0, 0, gw, gh], dtype=torch.float32)
            boxes_xyxy = boxes_xyxy.unsqueeze(0).repeat(image_dino.shape[0], 1)
        projected = self.encode_context(image_dino)
        return self.decode_local(image_detail, projected, boxes_xyxy, (gh, gw))

```

공식 backbone 연결 예시는 다음과 같다. 입력 `image_detail`과 `image_dino`를 이미 준비했다는 가정의 연결 코드이며, 가중치 경로는 실제 공식 파일로 지정해야 한다.[^dino]

```python
from pathlib import Path
import torch

repo = Path("third_party/dinov3").resolve()
# 공식 배포 파일명은 유지한다. 실행 환경에 맞는 실제 경로로 설정한다.
checkpoint = Path("weights/pretrained/dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth")
if not (repo / "hubconf.py").is_file() or not checkpoint.is_file():
    raise FileNotFoundError("DINOv3 source and official pretrained checkpoint are required")

backbone = torch.hub.load(
    str(repo), "dinov3_vith16plus", source="local",
    weights=str(checkpoint.resolve()),
)
model = PhaseHybrid(backbone, layer_indices=[7, 15, 23, 31])
model.set_backbone_trainable(last_n=0)   # S0
# S1 전환 시 model.set_backbone_trainable(last_n=8) 후 optimizer도 다시 구성한다.
```

**참조 코드의 범위:** 고정된 단일 context scale=1.0 좌표계를 기준으로 한다. context 영상만 별도로 resize하거나 letterbox하는 기능을 추가한다면, `global_hw`만 바꾸지 말고 원본→context affine 변환까지 ROI에 반영해야 한다. 학습 stage 관리, 입력 채널 생성, valid/core 마스킹, TTA와 제출 기능은 별도 구현이다.

### 20.4 loss 연결 의사코드

아래는 연산 순서 계약이며 `dice_per_image`, `lovasz_softmax`, `aux_loss`, `boundary_loss`는 별도 구현/도입할 함수다.

```text
outputs = model(detail_tensor, dino_tensor, global_box)
logits = outputs["logits"].float()
valid = (target != 255) AND core_valid_mask

main_target = target.clone()
main_target[NOT valid] = 255

CE = 이미지마다 유효 픽셀의 CE 평균을 구한 뒤 이미지 평균
probabilities = softmax(logits, class_dimension)
Dice = 이미지마다 정답에 존재하는 클래스의 Dice loss 평균
Lovasz = lovasz_softmax(probabilities, main_target,
                       classes="present", per_image=True, ignore=255)

loss = CE + 0.5 * Dice + 0.5 * Lovasz
if lambda_aux > 0:
    loss += lambda_aux * aux_softCE(outputs["aux_logits"], main_target)
if lambda_boundary > 0:
    loss += lambda_boundary * boundary_loss(outputs["boundary_logits"], main_target)

(loss / accumulation_steps).backward()
optimizer update 시점:
    gradient clip → optimizer.step → scheduler.step → EMA update → zero_grad
```

### 20.5 문서 작성 중 점검한 범위

이 문서의 codec/evaluator와 하이브리드 참조 코드를 CPU 환경에서 점검했다. 실제 점검 범위는 아래와 같다. 공식 H+ 가중치 로드, GPU mixed precision, 전체 학습, 실제 mIoU 향상은 이 단위 점검과 별개다.

| 점검 항목 | 결과 |
|---|---|
| Python 코드 블록 4개 | 문법 검사 통과 |
| YAML 설정 | 파싱 및 핵심 필드 확인 통과 |
| Markdown | 렌더링 및 각주 정의 검사 통과 |
| 실제 정답 90장 | RGB → ID → RGB 픽셀 일치 확인 |
| PNG 저장/재로드 | 실제 마스크 1장에서 RGBA 왕복 일치 |
| 예상 밖 마스크 색상 | 오류 발생 확인 |
| 평가 함수 | 완전 일치·클래스 부재·거짓 양성·ignore·이미지별 평균 테스트 통과 |
| ROI 샘플링 | full-box resize 동등성 및 합성 좌표 ramp 테스트 통과 |
| 하이브리드 연결 | 가짜 backbone으로 full-frame 64×96, local 32×32 출력 shape 확인 |
| gradient | backbone 고정 시 head 학습, 마지막 block 동결 해제 시 gradient 확인 |
| 체크포인트 | 가짜 backbone 모델의 저장/재로드 예측 일치 |
| 점검 환경 | PyTorch 2.10.0+cpu |
| 실제 DINOv3 pretrained 모델/GPU 학습 | 수행하지 않음 |
| 제안 시스템의 검증 mIoU | 아직 측정하지 않음 |

이 결과를 17절의 모든 통합 테스트를 완료했다는 뜻으로 해석하지 않는다. 특히 실제 전체 크기의 GPU forward, AMP, D4/타일의 최종 파이프라인, 외부 fold 학습과 제출 프로그램은 구현 후 추가 점검해야 한다.

---

## 21. 구현 시 피해야 할 오류

1. **색상·클래스 순서 오류:** 공정 Si의 `(204,204,204)`를 `(255,255,255)`로 바꾸거나 Al을 무시 클래스에 넣지 않는다.
2. **전체 영상 강제 축소:** 원본의 가는 조직을 잃은 뒤 큰 모델로 복구할 수 있다고 가정하지 않는다.
3. **가짜 다중 해상도:** ViT의 네 block 출력을 읽었다고 native stride가 네 종류가 되는 것은 아니다.
4. **경계의 반 픽셀 이동:** crop 좌표, padding, resize, `align_corners` 규약을 모듈마다 다르게 사용하지 않는다.
5. **캐시·증강 불일치:** 밝기/회전/배율/가중치가 바뀌었는데 context cache를 재사용하지 않는다.
6. **동결 해제 실패:** `requires_grad=True`여도 no_grad·detach·optimizer 누락으로 학습이 막힐 수 있다.
7. **잘못된 평가 평균:** 전체 픽셀/전체 confusion matrix 기준 mIoU와 이미지별 평균 mIoU를 혼동하지 않는다.
8. **OOF 간접 누수:** base OOF까지 외부 fold로부터 독립적인지 확인하지 않고 refiner CV 점수를 믿지 않는다.
9. **불일치한 보정기 입력:** 학습에는 단일 모델 확률, 추론에는 다른 TTA·다른 ensemble을 넣고 같은 분포라고 가정하지 않는다.
10. **노이즈 제거라는 이름의 상 삭제:** 작은 성분 제거·closing을 기본으로 적용하지 않는다.
11. **학습 점수로 최종 모델 선정:** 90장 재학습 이후에는 그 90장 성능을 독립 검증 성능으로 보고하지 않는다.
12. **규모와 성능을 동일시:** 7B, 추가 head, 더 많은 ensemble이 실제로 도움이 되는지 검증하지 않고 무조건 포함하지 않는다.

**최종 완료 상태는 “가장 복잡한 구조를 모두 켠 상태”가 아니라, 선택 과정에서 분리된 데이터로 이득을 확인한 구성만 남기고, 원본 픽셀 좌표·클래스·실행 환경까지 재현되는 상태다.**

---

## 22. 근거 자료

자료 확인일은 2026-09-20이다. 코드 저장소의 `main`은 변경될 수 있으므로 실제 구현에는 검증한 commit을 고정한다. 본 문서의 fusion 채널, ROI 결합 방식, 학습 스케줄, refiner·ensemble 조합은 이 과제를 위한 제안이며 아래 논문의 실험 결과를 본 데이터의 성능으로 옮겨 주장하지 않는다.

[^data]: 사용자 제공 `dataset.zip`과 이전 대화에서 생성된 `phase_analysis_final.md`의 데이터 통계·진단 실험. ZIP 구조, 파일 개수, RGBA 모드, 해상도, SHA-256은 이번 문서 작성 시 로컬 파일에서 다시 확인했다. 픽셀 비율·경계 거리·진단 분류기 점수는 분석 기록에서 인용했다. 두 파일의 이름과 ZIP SHA-256을 1절에 기록했다.

[^competition]: KoMaP 대회 데이터 및 평가 지표 설명. 주소: `https://komap.ai/front/competition2026/process1`.

[^submission]: KoMaP 공모작 제출 안내. 주소: `https://komap.ai/front/competition2026/apply_info`.

[^dino]: Meta/Facebook Research, DINOv3 공식 저장소. 모델군, 공식 Hub 로딩, LVD 입력 정규화와 가중치 접근 안내. 주소: `https://github.com/facebookresearch/dinov3`.

[^env]: DINOv3 공식 conda 및 의존성 설정. 주소: `https://raw.githubusercontent.com/facebookresearch/dinov3/main/conda.yaml` 및 `https://raw.githubusercontent.com/facebookresearch/dinov3/main/requirements.txt`.

[^backbones]: DINOv3 공식 backbone 생성 코드. H+/L/7B의 depth·차원·모델명, 가중치 처리. 주소: `https://raw.githubusercontent.com/facebookresearch/dinov3/main/dinov3/hub/backbones.py`.

[^vitcode]: DINOv3 공식 Vision Transformer 코드. `get_intermediate_layers`의 인자와 토큰/reshape 처리. 주소: `https://raw.githubusercontent.com/facebookresearch/dinov3/main/dinov3/models/vision_transformer.py`.

[^rope]: DINOv3 RoPE 구현과 train-mode 좌표 증강. 주소: `https://raw.githubusercontent.com/facebookresearch/dinov3/main/dinov3/layers/rope_position_encoding.py`.

[^convnext]: DINOv3 공식 ConvNeXt 구현. 주소: `https://raw.githubusercontent.com/facebookresearch/dinov3/main/dinov3/models/convnext.py`.

[^adapter]: Chen et al., *Vision Transformer Adapter for Dense Predictions*, ICLR 2023. 주소: `https://arxiv.org/abs/2205.08534`.

[^unet]: Ronneberger et al., *U-Net: Convolutional Networks for Biomedical Image Segmentation*, 2015. 주소: `https://arxiv.org/abs/1505.04597`.

[^hrnet]: Wang et al., *Deep High-Resolution Representation Learning for Visual Recognition*, TPAMI. 주소: `https://arxiv.org/abs/1908.07919`. 공식 분할 구현: `https://github.com/HRNet/HRNet-Semantic-Segmentation`.

[^lovasz]: Berman et al., *The Lovász-Softmax loss*, CVPR 2018. 주소: `https://arxiv.org/abs/1705.08790`.

[^lovaszcode]: Lovász-Softmax 저자 공식 구현. 주소: `https://github.com/bermanmaxim/LovaszSoftmax`.

[^unimatch]: Yang et al., *UniMatch V2: Pushing the Limit of Semi-Supervised Semantic Segmentation*, 2024. 주소: `https://arxiv.org/abs/2410.10777`.

[^grid]: PyTorch `grid_sample` 공식 API 문서. 좌표 정규화와 `align_corners`의 의미를 참조했다. 주소: `https://docs.pytorch.org/docs/2.8/generated/torch.nn.functional.grid_sample.html`.

[^resize]: PyTorch `interpolate` 공식 API 문서. `nearest-exact`, `size`, `align_corners`의 의미를 참조했다. 주소: `https://docs.pytorch.org/docs/2.8/generated/torch.nn.functional.interpolate.html`.

[^amp]: PyTorch Automatic Mixed Precision 공식 예제. 주소: `https://docs.pytorch.org/docs/2.8/notes/amp_examples.html`.
