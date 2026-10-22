# Model402 이후 다음 성능 개선 방향 검토 (2026-10-05)

**상태 (2026-10-05 09:29 UTC):** Model403 train-only grouped OOF 진행 중이며 fold1 완료, fold2는 epoch53 checkpoint에서 재개했다. 09:24 UTC에 원격 trainer/runner 부재와 fold-2 OOF 미생성을 확인했고, 09:28에 동일 recipe를 `nohup`으로 재개해 09:29에 실제 trainer 프로세스를 확인했다. 중단 원인은 아직 모른다([현재 절차](model403_local_supcon_preregistration.md)). valid/test는 사용하지 않았다. Claude Code CLI는 로그인 상태지만, 이번 read-only 검토 요청은 주간 한도에 막혔다. Model403의 SupCon은 component-aware가 아니며, 뒤이은 component-normalized full-resolution loss는 현 실험의 사전 gate가 끝난 경우에만 별도 CPU preflight/사전등록 대상으로 고려한다([Model403 독립 검토](model403_independent_review.md)). DINOv3 검토 결과와 P100·가중치 사전조건은 아래 독립 기술 검토에 기록한다. 현재 GPU 실험은 Model403 하나뿐이며, DINOv3 가중치는 local/remote cache에서 찾지 못했고 학습은 시작하지 않았다.

## 확인된 현재 문제

- 공식 valid 20장 기준 관측 최고는 Model364 `79.9429727%`이고, 최근 Model401 고정 B4/HRNet blend는 `79.4181709%`로 최고보다 `−0.5248018%p`다. Model397 `79.9858620%`는 70장/59 specimen group의 grouped train-OOF 값이지 valid 점수가 아니다.
- Model402 train-OOF 구조 진단에서 Eutectic Si class mean IoU는 `59.0029%`; 1–64 native-pixel GT components의 `89.83%`, 64–256px의 `99.19%`에서 FN이 하나 이상 있었다. Al3Ni와 Primary Si의 GT 경계 0–2px FN rate는 각각 `27.35%`, `27.38%`다. Oracle correction 수치는 실현 성능이 아닌 unattainable upper bounds다. 상세 값은 [Model402 결과](results/model402_model397_component_boundary_audit.json)에 있다.
- 압축 입력 분할을 확인했다: `train.zip`은 70 images + 70 masks, `valid.zip`은 20 images + 20 masks, `test.zip`은 10 images뿐이다. 즉 train에는 별도의 unlabeled image pool이 없다. 이 문서는 valid/test image를 semi-supervised training에 가져오지 않는다.

## 중복 실험 방지

Al–Si를 직접 다룬 2026 RUCS-BE-MT는 500개 문헌 micrograph와 unlabeled pool을 이용하고, 회전별 teacher 예측의 uncertainty-weighted consistency와 Boundary Enhancement Module(BEM)을 결합한다. KoMaP에선 유사 요소를 이미 각각 시험했다: Model300 rotation-uncertainty Mean-Teacher의 grouped OOF 변화 `−0.0142%p` (95% CI `[-0.0756,+0.0447]`); Model302 BEM `+0.2296%p` (CI `[-0.3177,+0.7440]`); Model303의 추가 continuation `+0.0533%p`. BEM/consistency를 새 아이디어인 것처럼 반복하지 않는다. 논문의 Jaccard/ARI/VI와 KoMaP image-wise four-class mIoU도 다른 estimand다. [RUCS-BE-MT paper](https://www.techscience.com/cmc/v87n1/66100/html), [official code/data](https://github.com/yx1539946445/RUCS-BE-MT).

또한 Eutectic용 clDice(Model249), OOF hard-crop mining(Model311), generic boundary-CE(Model376), directional phase-confusion loss(Model377), hierarchical phase head(Model291), train-fitted region/decoder correctors(Models293–299), DINOv2 residual feature branch(Models308–310)는 개선 근거가 없거나 사전 gate에 미달했다. 기존 결과는 [model_results.md](../model_results.md)에서 확인한다.

## 가장 구체적인 미검증 후보: 팀 DINOv3 전역문맥 + 로컬세부 hybrid

팀 비공개 [`sppark` commit `e0d71b1`](https://github.com/CNUKoMaP/sppark/tree/e0d71b1/experiments/accuracy_v1)의 `accuracy_v1`에는 DINOv3-L/16 intermediate features(5/11/17/23 blocks)를 원본 전역 문맥으로 쓰고 stride 1/2/4/8/16 CNN detail encoder와 결합하는 hybrid 구현이 있다. DINOv2 Model308의 patch-feature residual 덧셈과 다른 내부 multiscale global/local fusion 구조이므로, Eutectic Si의 작은·조밀한 형상에 적용할 미검증 후보로는 이 구현이 가장 구체적이다. DINOv3 일반 dense-feature 성능은 metallography transfer를 보장하지 않는다. [팀 hybrid 코드](https://github.com/CNUKoMaP/sppark/blob/e0d71b1/experiments/accuracy_v1/baseline/source/src/alphase/models/hybrid.py), [DINOv3 model card](https://github.com/facebookresearch/dinov3/blob/main/MODEL_CARD.md).

**성능 증거는 아직 없다.** 팀 저장소의 `dino_wiring_test.json`은 ViT-S의 untrained shape/wiring 검사이며 `pretrained:false`; `dino_large.yaml`은 `pretrained_checkpoint:null`이다. 저장된 `78.4729%` 기본 및 `78.6865%` D4 평가는 CNN U-Net baseline이지 DINOv3 점수가 아니다. 팀 실행 환경은 8×RTX4090, PyTorch 2.7.1+cu128이고, 현재 P100 `komap`은 PyTorch 2.6.0+cu118이다. 팀 DINO config의 512/768 global-context patch와 BF16 설정을 P100에서 그대로 실행할 수 있다고 가정하지 않는다. 먼저 isolated venv, official weights/license/access, pretrained load, P100 forward/backward·메모리/시간 smoke gate가 필요하며 기존 `/workspace/venv-komap`은 바꾸지 않는다. 가중치 접근 또는 P100 비용 gate가 실패하면 no-go다.

## Codex 독립 기술 검토 (2026-10-05)

팀 `CNUKoMaP/sppark`의 읽기 전용 최신 main snapshot (`007bda794ed6fc7b27b235bdbb3c52868e379bf0`)을 확인했다. `PhaseHybrid`는 DINOv3 ViT-L/16의 5·11·17·23번째 block feature를 추출해 stride 4/8/16 CNN detail pyramid와 결합한다. 설계 자체는 기존 DINOv2 residual feature branch와 다르지만, 지금 저장소에는 실제 DINOv3 사전학습 실행 증거가 없다: `dino_large.yaml`의 `pretrained_checkpoint`가 null이고, wiring artifact도 `pretrained:false`인 shape 검사뿐이다. 기록된 78.47/78.69%는 CNN baseline이지 DINOv3 점수가 아니다.

P100에 그대로 실행하는 것은 **NO-GO**다. 팀 설정은 `precision: bf16`인데 현재 P100은 compute capability 6.0이고, CUDA의 BF16 native 요구사항은 8.0 이상이다. 원격 환경도 Python 3.10.12 / torch 2.6.0+cu118이며 시스템에 Python 3.11이 없다. Meta 공식 DINOv3 저장소 `setup.py`는 Python ≥3.11을 요구한다. Hugging Face 경로는 Transformers 4.56.0부터 공식 지원되며 현재 `komap` 환경의 Transformers 5.17.0에서 버전 조건은 충족하지만, 해당 ViT-L 모델 페이지는 연락처 정보 공유에 동의해야 접근 가능하고 DINOv3 별도 라이선스가 적용된다. 사용자 계정의 접근 동의는 내가 대신하지 않는다. ([Meta 저장소](https://github.com/facebookresearch/dinov3), [공식 ViT-L 모델 카드](https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m), [NVIDIA BF16 요구사항](https://docs.nvidia.com/cuda/archive/13.2.0/cuda-programming-guide/05-appendices/mathematical-functions.html))

로컬 Hugging Face 캐시와 P100 서버의 캐시/workspace를 read-only로 확인했으며 DINOv3 가중치 또는 모델 캐시는 발견되지 않았다. 아직 가중치를 내려받거나 gated access/연락처 동의를 시도하지 않았다.

추가로 저장소의 AMP 실제 경로를 읽었다. `amp_context`는 `bf16`/`fp32`만 허용하며, P100에서 `bf16`을 선택하면 예외를 발생시킨다. 학습 loop는 일반 `backward()`/`optimizer.step()`만 사용하고 FP16 autocast/GradScaler 경로가 없다. 따라서 FP16은 YAML 값만 바꿔 적용할 수 없고 isolated fork에서 AMP와 GradScaler를 구현·검증해야 한다. ([검토한 trainer](https://github.com/CNUKoMaP/sppark/blob/007bda794ed6fc7b27b235bdbb3c52868e379bf0/experiments/accuracy_v1/baseline/source/src/alphase/engine/trainer.py), [AMP helper](https://github.com/CNUKoMaP/sppark/blob/007bda794ed6fc7b27b235bdbb3c52868e379bf0/experiments/accuracy_v1/baseline/source/src/alphase/utils.py))

**조건부 GO는 사전 smoke만**이다. Model403가 끝난 뒤 공식 가중치 접근/라이선스를 확인하고 격리 branch에서 FP16 autocast + GradScaler를 구현한 뒤, frozen ViT-L forward와 최소 1회 학습 update의 finite loss/gradient·peak memory·step time을 P100으로 검증한다. 해당 코드 변경이나 weight gate가 어렵다면 DINOv3는 no-go로 둔다. 팀의 4,500-update staged-unfreeze BF16 recipe를 P100에서 그대로 돌리지 않는다. smoke가 resource gate를 넘으면 그때 동일한 4-fold specimen-group train OOF와 고정 학습 길이로 full candidate를 준비한다. 팀 config의 기본 `empty_union: exclude`를 채택하지 않고, 모든 held-out 예측을 우리 공식 image-wise 4-class mIoU (`empty=0`)로 채점한다. DINOv3 논문·모델카드의 일반 dense-task 성적은 금속조직 도메인의 성능 증거가 아니며, 예상 점수 향상은 주장하지 않는다. ([DINOv3 논문](https://arxiv.org/abs/2508.10104), [Transformers 지원 안내](https://github.com/facebookresearch/dinov3#pretrained-backbones-via-hugging-face-transformers))

## Claude Opus에 요청할 판단

아래 프롬프트는 다음 Claude 사용량 초기화 후 단 한 번 제출한다. 요청은 (1) 팀 DINOv3 global-context/local-detail hybrid를 P100에 맞춰 재현할 가치가 있는지, (2) prior DINOv2 branch와 실질적으로 다른지, (3) resource/license 조건까지 포함한 GO/NO-GO 및 단 하나의 대체 구조가 필요한지 판단하는 것이다. sweep, routing/calibration 조합 탐색, valid/test 튜닝은 제안하지 말고, GO 시 4-fold specimen-group paired OOF의 비교 대상·비용·중단 gate를 구체화하게 한다.

Claude 호출은 2026-10-06 23:00 KST의 사용량 초기화 전까지 우회하지 않는다. 그 전에는 검토를 받은 것처럼 말하거나 새 GPU 학습을 시작하지 않는다.

### 제출용 Claude prompt

```text
Act as an independent segmentation researcher. We need to improve KoMaP's
official image-wise mean mIoU (four class IoUs averaged within each image,
then across images; absent-class IoU is 0), currently below 80%. Do not assume
that a promising oracle upper bound is achievable.

Current evidence:
- Official valid has 20 labeled images and has been reused during development.
  Raw recorded leader Model364 is 79.9429727%; recent fixed Model401 blend was
  79.4181709%. These are selection-conditional, not independent test scores.
- Model397 has 70 exactly-once grouped train-OOF images / 59 specimen groups,
  with fixed blend C at 79.9858620%; this is OOF, not official valid.
- Model402 on the same fixed OOF maps: Eutectic-Si class image-mean IoU
  59.0029%; GT component miss rates 89.83% for 1–64 px and 99.19% for
  64–256 px. Al3Ni / Primary-Si FN rates within 2 px of their GT boundaries
  were 27.35% / 27.38%. Perfect-correction deltas are unattainable ceilings,
  not expected gains.
- train.zip contains only 70 image+mask pairs; valid.zip has 20 pairs and
  test.zip has 10 unlabeled images. Do not propose using valid/test images or
  labels for training, model selection, pseudo-labels, or adaptation.
- Already attempted without adoptable gain: hierarchical phase heads and
  component/region correctors; DINOv2 ViT residual dense branch; Eutectic
  clDice and OOF hard-crop mining; rotation-uncertainty mean teacher; the
  Al-Si RUCS-BE-MT-inspired boundary-enhancement module (Model302/303); generic
  boundary-weighted CE (Model376); directional phase-confusion loss (Model377);
  HRNet width normalization and several global ensemble/routes. Do not repackage
  these as new ideas or recommend another sweep.
- The team's private sppark/accuracy_v1 includes a DINOv3-L global-context plus
  multiscale CNN local-detail hybrid. Its DINOv3 test is wiring-only and
  explicitly unpretrained; the saved 78.4729/78.6865 scores are CNN baseline,
  not DINOv3 results. The team environment was 8x RTX4090 / torch 2.7.1+cu128;
  current P100 komap is torch 2.6.0+cu118. Weight license/access and P100
  compatibility/cost are open. Do not infer the hybrid's score from the CNN.

Return exactly one decision: GO to evaluate/reproduce a P100-feasible version of
the team's DINOv3 global-context/local-detail hybrid, NO-GO, or replace it with
one more promising untried structural/representation intervention. If GO,
specify whether ViT-L is feasible or a smaller DINOv3 backbone changes the
method too much; define isolated-environment, pretrained-weight/license,
forward/backward, and runtime prerequisites before full training. Then give an
exactly paired 4-fold specimen-group OOF protocol against a locked baseline,
with a preregistered pass/futility gate using image-wise mIoU, fold deltas,
paired specimen-group bootstrap, class and Step4 safeguards, and cost limits.
Do not recommend tuning on the reused valid set. Test remains sealed. Do not
claim an expected score or that the 80% target is likely. Cite primary sources
for any literature-based claim; otherwise label it as your own hypothesis.
```
