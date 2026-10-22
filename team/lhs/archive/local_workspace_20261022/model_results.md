# KoMaP 모델 결과 기록

## Model410: 작은 성분 균형 crop sampler (NO-GO; 2026-10-06)

- 고정 4-fold specimen-group train-OOF (70장/59 group) 한 번의 채점 결과, candidate `79.0771%`, matched control `79.0114%`, delta `+0.0657%p`; paired group-bootstrap 95% CI `[-0.0633,+0.2158]%p`. 양수 fold는 `2/4`, candidate가 이긴 이미지는 `42/70`, 가장 큰 group의 양수 기여 비중은 `87.1%`로 효과가 작고 한 group에 집중됐다.
- 사전 gate는 **NO-GO**: 최소 `+0.20%p`, CI 하한 `>0`, 양수 fold `3/4` 기준을 모두 충족하지 못했다. Al3Ni tiny/small/medium GT-component hit-rate 안전 조건과 두 target class의 unmatched tiny prediction 비증가 조건도 실패했다. Overall class delta는 Al `+0.0502`, Al3Ni `+0.1250`, Primary Si `+0.0220`, Eutectic Si `+0.0657%p`; Step4 delta `−0.0406%p`다.
- 폭 정규화 component 진단에서는 Al3Ni hit-rate가 modal-width-equivalent `[1,4)`, `[4,16)`, `[16,64)` bins에서 각각 `−0.536`, `−0.554`, `−0.655%p` 하락했고 unmatched island는 `−1.40`, `−0.80`, `−2.26/image` 감소했다. Eutectic Si hit-rate는 각각 `+0.252`, `+0.585`, `+0.464%p`였으나 unmatched predictions는 `−0.014`, `+0.443`, `+1.657/image` 변했다. 따라서 component sampling은 target class별로 상반된 recall/precision 효과를 냈으며 전체 개선으로 채택하지 않는다.
- 결과는 개발용 train-OOF일 뿐 공식 valid/test를 읽지 않았다(`valid_read=false`, `test_read=false`, `official_valid_stage_allowed=false`). source/OOF lock과 scorer claim 체인 검증 통과. [사전등록](experiments/model410_small_component_sampler_preregistration.md), [결과 JSON](experiments/results/model410_component_sampler_oof.json); result SHA-256 `7d1349e1a6861f020c587abb431a0829db94b503ff9bc0f27d1cb966b5637f5a`.
- NO-GO 분기에 따라 조건부 실행된 Model411도 최종 train-OOF에서 NO-GO였다. 상세 효과·안전 gate와 잠금 검증은 아래 Model411 기록을 참조한다.

## Model411: 작은 성분 bidirectional loss (최종 NO-GO; 2026-10-06)

- **고정 train-only OOF 완료:** 70장/59 specimen group/4-fold, 공식 image-wise mean mIoU(empty=0). Candidate 대 matched control은 `+0.0156%p`, paired group-bootstrap 95% CI `[+0.0066,+0.0249]%p`; 4/4 fold 및 46/70 이미지에서 양수, 최대 group 기여 비중 `19.8%`였다. class delta(Al/Al3Ni/Primary Si/Eutectic Si)는 `+0.0155/+0.0285/+0.0086/+0.0096%p`, Step4 `+0.0243%p`다.
- **사전등록 NO-GO:** 채택에 필요한 matched-control `+0.20%p` 효과 크기에 미달했다. CI 하한, 3/4 fold, worst-fold, Step4, class-IoU 기준은 통과했지만 두 표적 클래스의 FP-pixel 비증가 및 unmatched tiny-prediction 비증가 안전 gate가 실패했다. Al3Ni/Eutectic Si unmatched tiny prediction은 각각 `+23.24/+16.59` components/image, target FP pixels는 `+201.16/+30.47` pixels/image 증가했다. Tiny GT hit rate는 Al3Ni `+1.33%p`, Eutectic Si `+1.00%p` 개선됐고 Al3Ni 64–255px bin은 `+0.42%p`였으나 Eutectic Si 64–255px bin은 `−0.14%p`였다. Recall 증가는 실제 mIoU 개선으로 충분히 전환되지 않았고 FP 안전 기준도 깨져 이 처치는 채택하지 않는다.
- Model392 seed-1 초기화 parent와의 OOF 차이는 `+0.0493%p`, 95% CI `[-0.1509,+0.2624]%p`; 3/4 fold 양수였지만 최대 group 기여 비중 `90.8%`, Step4 delta `−0.1615%p`로 불확실하고 group 집중도가 높다. 이를 독립 개선 근거로 보지 않는다.
- 결과 생성 당시 `valid_read=false`, `test_read=false`, `valid_stage_allowed=false`. Source/calibration/OOF lock과 claim의 해시 연결이 일치했고 잠금된 OOF 확률·manifest 8개도 모두 SHA-256 검증을 통과했다. 결과 SHA-256 `bb65b9d4c74f7d79777b297de1ae2e0d64f460ed0cee2c81f51521f059d7bb24`; source/calibration/OOF lock SHA-256은 각각 `92b84a663b962876cbc3bdc046a56a867f8f5ed0de1a096ff8ecb099aa068058`, `a1d0066f76a7b1b861d1c23d79d78f0c7df4286f62da63e04f6c3b05f6400b9f`, `b2e1189a4c980056945543b6ae4fc3a0b0203461bad6ee807d2ca02621683e08`다. [결과 JSON](experiments/results/model411_bicc_oof.json), [claim](experiments/results/model411_bicc_oof.json.claim), [OOF lock](experiments/results/model411_bicc_oof_lock.json).
- Claude Code 재검토 요청은 도구 접근을 끈 상태로 보냈으나 8분 이상 응답 없이 종료했다. 독립 검토상 Model410은 component-balanced sampling 전체를 반증하지는 않지만, Model411까지 작은 성분 hit-rate와 FP 부담의 상충이 확인되어 이 loss/sampler 계열의 추가 반복은 중단하고 다른 원인 축을 검토한다.

- Claude Code 사용 불가 상태에서 독립 검토했다. 라우팅·앙상블 대신 작은 성분 오분류 자체를 다루는 단일 HRNet-W48/U-Net++ 학습 후보를 준비했다. 기존 train-OOF에서 Model404 `+0.032%p`, Model406 `+0.020%p`, Model408 `+0.029%p`로 작은 GT 성분 hit-rate는 오르지만 unmatched tiny 예측도 늘었고, Model407은 tiny FP를 줄이는 대신 mIoU `−0.087%p`, tiny hit-rate `−2.05~−3.86%p`였다. 따라서 원인은 단순 미탐이나 FP만이 아니라 component loss의 집계 강도/정밀도-재현율 균형으로 본다.
- 신규 가설은 Model407처럼 한 crop에 생긴 모든 작은 false-island penalty를 합산하지 않고, 8-connected GT/예측 성분을 crop×target-class×size-bin별로 각각 평균하는 것이다. 고정 mixture는 GT recall `0.75` + unmatched 예측 성분 억제 `0.25`; classes Al3Ni/Eutectic Si, augmented area `[4,16)`, `[16,64)`, `[64,256)`만 쓴다. 추론/모델 구조/앙상블은 바꾸지 않는다. BiCC (2026 preprint)의 양방향 component objective에서 착안했으나 원 논문은 3D medical lesion과 lesion-wise 지표이므로 KoMaP 개선 근거로 간주하지 않는다 ([논문](https://arxiv.org/abs/2609.30223)).
- [Model411 사전등록](experiments/model411_bicc_preregistration.md), [loss 구현](experiments/model411_bicc_loss.py), 기존 trainer를 재사용하는 [wrapper](experiments/model411_bicc_train.py), outer-train-only [zero-step calibrator](experiments/model411_bicc_calibrate.py), image-only [OOF predictor](experiments/model411_predict_oof_probabilities.py), [source/calibration/OOF lock](experiments/model411_bicc_lock.py), [one-shot scorer](experiments/score_model411_bicc_oof.py), [runner](experiments/run_model411_bicc_oof.sh), [Model410 gate 뒤 대기기](experiments/queue_model411_after410.sh)를 준비했다. P100 `komap`에서 py_compile/bash 구문 및 CPU 392-component stress smoke 통과.
- 2026-10-06 첫 조건부 queue는 Model410 NO-GO 결과 뒤 lock preflight를 시작했고, Model411 loss smoke 및 split 검증(outer-train `52/52/53/53`, held-out `18/18/17/17`)까지 통과했다. 이어 source-lock 생성 중 `model406_component_hardnegative_v3_lock`에 존재하지 않는 `CALIBRATION_LOCK` attribute를 참조해 종료됐다. calibration, optimizer step, checkpoint, OOF prediction, Model411 source/calibration/OOF lock은 생성되지 않았다. [오류/수정 및 후속 검증 기록](experiments/model411_bicc_preregistration.md).
- 이 provenance 도구 오류를 pre-lock 상태에서 수정했다: Model406 v3가 상속한 `model406_component_hardnegative_v2_lock.CALIBRATION_LOCK`를 참조하고 해당 v2 lock 모듈을 source inventory에 추가했다. 수정본은 Python compile 및 runner/queue bash syntax 검사를 통과했으며, P100에서 read-only reference preflight를 다시 통과한 뒤 새 source lock을 만들고 학습을 시작해야 한다. loss, 고정 `0.75/0.25` weights, recipe, folds, paired control 및 adoption gate는 바꾸지 않았다.
- Model410 gate가 NO-GO여서 조건부 실행을 재개했다. P100 read-only reference preflight 및 수정된 source-lock 검증이 통과했고, Model411 source lock은 2026-10-06 12:26:37 UTC에 생성됐다(SHA-256 `92b84a663b962876cbc3bdc046a56a867f8f5ed0de1a096ff8ecb099aa068058`). Outer-train-only 32-crop/fold zero-step calibration은 12:34:26 UTC에 끝났고 calibration lock은 12:36:53 UTC에 생성됐다. 12:52 UTC 점검에서 fold1 candidate trainer PID `1874035`가 학습 중이었고 `last.pt`/state는 각각 12:51:58/12:52:02 UTC에 갱신됐다. Fold1 OOF prediction과 최종 점수는 아직 없다. 첫 queue 진단 로그는 보존하고 재시도는 별도 로그 파일로 분리했다.
- 13:12 UTC 점검에서 fold1 held-out-train OOF 확률이 저장됐고 runner는 candidate fold2 학습 중이었다(PID `1877215`; `last.pt`/state mtime `13:11:27/13:11:29 UTC`). Fold2–4 OOF와 최종 score는 아직 없다.
- 13:31 UTC 점검에서 fold2 OOF 확률 저장이 확인됐고 candidate fold3 학습이 시작됐다(PID `1879945`). Fold1–2 candidate OOF는 완료, fold3–4 및 one-shot scoring은 미완료다.
- 13:40 UTC 점검에서 candidate fold3 학습은 계속 활성이고 `last.pt`/state가 13:38:25/13:38:28 UTC에 갱신됐다. Fold3 OOF predictor는 아직 시작되지 않았다.

## Model409: 작은 GT 성분별 국소 특징 분리 (진행 중, 2026-10-06)

- 독립 재검토에서 Model403의 전역 same-class contrast, Model404/406의 GT component recall, Model407의 예측 island 억제, Model408의 tiny-confuser logit hinge가 이미 시험된 것을 확인했다. 그래서 단일 새 처치로 같은 8-connected tiny GT component의 decoder embedding만 positive로, 주변 2–6 augmented-pixel ring에서 Al3Ni에는 Al, Eutectic Si에는 Al/Al3Ni를 local negative로 쓰는 training-only instance contrastive loss를 고정했다. 추론 시 projection head를 버리며, backbone/decoder, CE+Dice+boundary, crop, inference, routing/ensemble은 바꾸지 않는다. [v2 사전등록](experiments/model409_instance_metric_preregistration_v2.md), [v2 loss](experiments/model409_instance_metric_loss_v2.py), [trainer](experiments/model409_instance_metric_train.py), [runner](experiments/run_model409_instance_metric_oof.sh).
- v1 보정 OOM은 외부 프로세스 점유로 단정할 근거가 없었다. 코드 재검토에서 각 component마다 feature map 전체를 float 변환해 `grid_sample`하는 반복 autograd context를 확인했고, v2는 성분 좌표를 모아 이미지당 한 번만 sampling하도록 고쳤다. 392개 작은 성분이 한 crop에 든 dense stress test가 CPU/P100 `komap`에서 모두 통과했다. v1 source lock `bea52de2…`과 실패 기록은 보존하고 optimizer step 전에 revision 2로 분리했다.
- 고정 split(outer-train `52/52/53/53`, held-out `18/18/17/17`) 및 Model406 reference lock을 P100에서 재검증했다. v2 outer-train-only 32-crop/fold 보정은 4/4 fold 모두 active 32/32였고 optimizer step 0으로 완료했다. 고정 auxiliary weights는 `0.0083830791 / 0.0063468995 / 0.0089394828 / 0.0079580497`이며 각각 decoder-gradient 비율 10%를 목표로 정했다. v2 source lock SHA-256 `f9a58cf694862d2d2e1927ebe2e281ef10392fa6965d936dfd7777983c1eaee4`, calibration lock `803916f512a3939d4138a14138854251ebec279e57ce8f46a9f75a126ae8cd78`.
- 최초 SSH 실행은 fold1 epoch1 checkpoint 뒤 종료됐고, 저장된 optimizer/RNG state로 detached runner에서 재개했다. fold1·2의 8/8 epoch 및 held-out train-OOF 확률 생성이 끝났다. 2026-10-06 07:13 UTC 확인 시 runner PID `1788575/1788576`, fold4 trainer PID `1826189`은 활성이고 fold4 checkpoint/state metadata는 `07:07:29/32 UTC`로 갱신됐다. fold3·4 OOF와 최종 채점은 미완료이며 official valid/test는 읽지 않았다. OOF 로그 노출 addendum에 따라 이후 로그/OOF 값을 열지 않고, 결과는 채택 근거가 아닌 탐색 결과로만 취급한다.
- 투명성 기록: 2026-10-06 04:17 UTC의 상태 확인에서 fold1 evaluator가 포함된 runner 로그 끝부분을 읽어 한 fold의 클래스 점수 줄이 도구 출력에 노출됐다. 해당 수치는 저장·복사하거나 의사결정에 쓰지 않는다. 이후 runner 로그 대신 process/checkpoint/output metadata만 확인한다. 고정 4-fold scorer는 처치 변경 없이 한 번 완결하되, 결과는 이 부분 출력 노출을 명시한 탐색적·selection-conditional OOF로만 보고하며 채택 gate나 다음 하이퍼파라미터 선택에 쓰지 않는다. [노출 기록](experiments/model409_oof_exposure_addendum.md).
- **다음 실험 중복 방지 문헌 screen:** 2026년 EISC 금상 분할 논문은 U-Net++와 다른 U-Net decoder의 다중 스케일 side predictions에 supervised/internal consistency 및 teacher–student cross-scale consistency를 결합한다. 하지만 다중 decoder supervision은 Model121/122, teacher consistency는 Model300에서 이미 시험됐고 KoMaP train에는 별도 unlabeled image pool도 없다. 따라서 EISC를 이름만 바꿔 반복하지 않는다([EISC 원 논문](https://www.mdpi.com/2079-9292/15/15/3425)). ExpertSegmentation은 현미경 재료상 분율·연결성 prior를 tree classifier의 학습 손실로 적용했지만, KoMaP에 정당화된 외부 phase-fraction prior나 eutectic connectivity 제약은 아직 확인되지 않아 현 후보로 채택하지 않는다([NLR/Acta Materialia 연구 기록](https://research-hub.nlr.gov/en/publications/expertsegmentation-segmentation-for-microscopy-with-domain-inform/)).
- 라우팅·앙상블은 후속 탐색에서 제외한다. Model409의 일부 fold 점수 노출은 채택 및 후속 설정 선택에 사용하지 않는다. 독립 검토에서 weighted-CE 초안은 전역 class prior만 바꾸고 작은 연결요소 노출을 직접 다루지 않으며, CE+Dice+boundary 조합에서 matching probability correction의 이론적 정당성도 약해 실행하지 않기로 했다. 이를 대체하는 Model410은 50% positive crop 중 Al3Ni/Eutectic-Si 성분을 fold modal-width-equivalent area bin `[1,4)`, `[4,16)`, `[16,64)`별로 균형 선택하는 단일 데이터 샘플링 비교다. backbone/loss/optimizer/inference는 고정하고 control은 기존 pixel sampler를 그대로 쓴다. [Model410 사전등록](experiments/model410_small_component_sampler_preregistration.md), [sampler](experiments/model410_component_uniform_sampler.py), [trainer wrapper](experiments/model410_component_sampler_train.py), [prediction-only OOF](experiments/model410_predict_oof_probabilities.py), [runner](experiments/run_model410_component_sampler_oof.sh).
- 합성 smoke에서 같은 seed 재현, class·size-bin 균형, width-normalized-equivalent area bins 및 control sampler의 원본 동작 일치를 확인했다. prediction-only 경로는 held-out 이미지 확률만 저장하고, 전체 70장·59-group 산출물 잠금 뒤 one-shot scorer가 처음 점수를 계산한다. 2026-10-06 07:13 UTC 현재 Model410 queue PID `1821541`이 대기 중이며 source lock은 아직 없어 학습은 시작하지 않았다. 공식 valid/test는 사용하지 않는다.
- 독립 scorer 감사에서 상속 scorer의 모든 클래스 퇴행 허용치가 사전등록보다 느슨한 것을 발견해 Model410 one-shot scorer에 `all-class delta >= −0.30%p` 검사를 추가했다. 원격 P100에서 변경된 sampler/scorer/lock 코드를 다시 py_compile했고 합성 smoke도 통과했다. source lock 생성 전 변경이므로 고정 recipe 전체가 실행 전에 잠긴다.
- 2026-10-06 독립 재검토에서 기존 Model406 mask-only audit의 width-normalization 주의사항과 일치하지 않게 Model410 초안이 raw native-area bin을 사용한 점을 발견했다. train 이미지 폭이 675–1440px이고 width-normalized scale이 fold에 따라 달라지므로 source lock/학습 전 사전등록을 revision 1.1로 수정했다. binning area는 `raw_area*(fold_modal_width/image_width)^2`로 정규화한다. outer-train mask-only 감사에서 모든 fold/class의 세 bin이 지원됨을 확인했다(Al3Ni bin별 fold count 범위 `2,070–2,388 / 20,186–21,582 / 23,937–25,540`; Eutectic Si `529–642 / 6,253–7,660 / 21,289–23,328`). 모든 split 이미지가 crop보다 커 reflect padding은 0장이었다. cross-width synthetic smoke, P100 py_compile/bash -n 및 scale-aware audit가 통과했다. 수정 파일의 local/P100 SHA-256도 일치하며 Model410 source lock은 아직 없어 revision 1.1이 그대로 잠길 수 있다. 07:13 UTC 확인 시 Model409 fold4는 계속 학습 중이고 Model410·411은 대기 상태였다. score/GPU 조회 및 official valid/test 접근은 하지 않았다.
- Claude Code 재확인: CLI는 설치·인증 상태지만 읽기 전용 검토 요청이 weekly limit로 거절됐고, 재설정 안내는 23:00 KST였다. 따라서 이번 Model410 revision 1.1 판단에는 Claude 응답을 사용하지 않았다. 한도 복구 후에는 이미 잠긴 recipe를 바꾸지 않고 OOF 결과 해석과 다음 가설 검토에 Claude 의견을 보탠다.

## Model407: 작은 정답 성분 회수 + 작은 오검출 island 억제 (NO-GO, 2026-10-06)

- 2026-10-06 확인: 이전 Model406에서는 작은 Al3Ni/Eutectic-Si 정답 성분 recall이 올랐지만 고립된 작은 예측 성분도 증가했다. Model407은 정답 component-recall 보조손실은 유지하고, pixel hard-negative 항만 예측 연결성분 단위 FP penalty로 교체한 단일 후보다. 성분 분류는 argmax 기준 8-connectivity이며 같은 클래스 GT 1px halo와 crop 경계 성분은 penalty에서 제외한다.
- 외부 train 70장/59 specimen group의 고정 4-fold OOF만 사용한다. Model392 seed-1 고정 체크포인트, 8 epochs, 같은 split·증강·학습 설정을 따르고 Model406 v3의 잠금된 matched-control OOF와 paired 비교한다. official valid/test는 읽지 않는다. outer-train-only gradient calibration은 4/4 fold, 각 32/32 crop 및 optimizer step 0으로 끝났다.
- 원격 calibration JSON 재확인: fold별 joint-aux coefficient는 `0.0001683965/0.0001893933/0.0001694625/0.0001822461`; 보정 전 auxiliary/main decoder-gradient norm 중앙값은 `593.84/528.00/590.10/548.71`로 coefficient 적용 후 목표 비율 10%다. 4개 폴드 모두 32/32 유효 crop으로 계산됐다.
- 해석상 한계: 예측 FP 손실은 unmatched component마다 `-log(1-mean p_target)`를 더하는 count-sensitive 항이며 면적 bin별 가중은 하지 않는다. 그래서 여러 작은 island는 각각 벌점을 받지만 단일 큰 FP component도 같은 component-level 항 하나를 받는다. 직접적인 size-specific 교정이라고 단정하지 않고, OOF 후 native 면적 및 GT-overlap별 FP 픽셀 분해로 검증한다.
- 네 fold 학습과 train-held-out 추론이 완료됐고, 고정 OOF는 `18+18+17+17=70`장/59 specimen group으로 잠갔다. fold별 단독 점수는 표본 하나의 fold 결과라 모델 선택 근거로 사용하지 않는다. evaluator의 `--split valid`는 fold별 train-held-out이고 대회 `valid.zip`은 열지 않았다. 공통 학습 엔진 배너는 Model404지만 Model407 wrapper가 loss 모듈을 교체하며, 해당 source는 사전 잠금됐다.
- 기존 진단은 zero-overlap 1–63px 성분 수와 FP 픽셀 총량만 보고해, 면적 및 GT overlap별 분해를 추가 사전등록했다. [별도 면적 감사](experiments/results/model407_predicted_component_error_audit.json)는 작은 예측 성분의 고립/GT중첩 FP 픽셀을 나눠 설명하는 용도이며 score/gate/threshold를 바꾸지 않는다. 모든 train 이미지에 Al3Ni와 Eutectic Si가 존재해 image-level absent-class 층은 support 0이다.
- 별도 감사 인자 경로도 확인했다: `--data-root /workspace/komap_model1/data`, `--candidate-root /workspace/komap_model1/outputs/model407_component_fp/candidate`, `--control-root /workspace/komap_model1/outputs/model406_component_hardnegative_v3/control`, `--baseline-root /workspace/komap_model1/outputs/model392_widthnorm`, `--out /workspace/komap_model1/experiments/results/model407_predicted_component_error_audit.json`. 감사기가 `--baseline-root` 아래에 `seed1`을 자동 결합하므로 해당 인자에 이미 `/seed1`을 포함하지 않는다.
- paired OOF (70장/59 groups, empty=0 image mIoU): 후보 `78.9244%`, 매칭 대조군 `79.0112%`, Δ`−0.0868%p` (paired group-bootstrap 95% CI `[−0.1173,−0.0586]`), 양수 fold `0/4`; NO-GO. Al3Ni/Eutectic Si class IoU 변화는 `−0.1002/−0.1886%p`; Step4 `−0.1277%p`. Model392 고정 parent 대비도 `−0.0531%p` (CI `[−0.2600,+0.1636]`). 공식 valid/test 미사용.
- 억제 효과는 있었지만 진짜 작은 정답 성분도 함께 약화됐다. 후보−대조군 1–63px 예측-component FP 픽셀은 Al3Ni `−430.7/image` (95% CI `[−513.3,−354.5]`), Eutectic Si `−306.3/image` (`[−374.8,−239.7]`) 감소했다. 동시에 tiny GT-component hit-rate는 Al3Ni `−3.86%p`, Eutectic Si `−2.05%p`; 세부 1–3px hit-rate는 각각 `−5.54/−3.19%p`였고, 상간 혼동이 더 Al 쪽으로 이동했다. 4개 fold 모두 전체 mIoU가 낮아져 해당 손실은 채택하지 않는다.
- 면적 분해에서도 후보는 작은 zero-overlap FP뿐 아니라 GT와 일부 겹치는 작은 성분의 FP 픽셀까지 감소시켰다. 특히 Al3Ni의 1–63px GT-overlap FP `−185/image`와 zero-overlap FP `−245/image`, Eutectic Si는 각각 `−87/−220/image`였다. 따라서 작은 island 숫자만 줄이는 것이 목표가 아니라, true tiny GT recall을 보존하는 것이 병목이다. image-level target-absent 억제는 이 데이터에서 불가능(두 표적 클래스 모두 70/70 이미지에 존재)하므로, crop-level 부재 여부는 별도 사전 감사 없이는 가설로만 둔다.
- scorer 첫 시도는 기준 경로에 `seed1`을 중복 결합해 GT mask/mIoU를 읽기 전에 실패했다. 실패 claim을 보존하고 같은 잠금 OOF를 올바른 parent 경로로 한 번 채점해 recovery 결과를 별도 파일로 남겼다. 결과·lock·claim·두 감사 JSON은 원격과 로컬 SHA-256이 일치한다. [recovered OOF](experiments/results/model407_component_fp_oof_recovered.json), [tiny-component audit](experiments/results/model407_tiny_component_confusion_audit.json), [FP 면적 audit](experiments/results/model407_predicted_component_error_audit.json).
- SHA-256: source lock `ff785e151fbc5480ec25e9819f56d6b544473c30946d1da7179f874459f0ece6`, calibration lock `5f0938b7123f46254a6ebe1bdcd7bab7dec3a2d7b64bf03fe4bdff6632ac23b6`, OOF lock `dbf6b417e611bed01061dcfdea3a58b086ace052f42928baf37a5b34ee28356f`, recovered result `3a75a5f64ec1ffab86e30c40ab3c64af92ac650d59b0a77f3fb6a15f472dcbc3`, tiny audit `342ed5b0ff020d706a4b142b327789eaa5edc7fcd537ccb70ce8b3a0de1c1a12`, area audit `6c990700f655900ef80d9a8ee1e93c6aaf2904f2132d2f34deaf839820e1e068`. All copied result/lock/claim/audit SHA-256 values matched P100.
- Claude Code read-only 검토 요청 [프롬프트](experiments/model407_claude_review_prompt.md)를 준비했다. 전체 잠금·train-only 진단 요약만 검토하게 하며, official valid/test 접근·편집·학습·GPU 문의를 금지한다. 현재 알려진 CLI quota 재개 시각 전에는 호출하지 않는다.
- **후속 가설(미실행):** tiny GT 오분류가 Al3Ni→Al, Eutectic Si→Al/Al3Ni로 치우치고 Model407 억제 loss가 이를 악화시켰다. 일반적인 전역 SupCon/기존 generic margin은 반복하지 않는다. 다음 후보는 tiny-GT component 내부의 표적 혼동쌍만 다루는 local feature/prototype 또는 confusion-aware positive loss다. 동일 4-fold grouped OOF의 사전등록 gate를 먼저 정하고, Model407의 FP 감소를 근거로 component-FP loss를 다시 튜닝하지 않는다. Claude Code 사용 가능 시 독립 검토를 거친다.
- 프로토콜: [Model407 사전등록](experiments/model407_component_fp_preregistration.md). 학습/loss 구현은 [trainer wrapper](experiments/model407_component_fp_train.py), [loss](experiments/model407_component_fp_loss.py), 실행은 [OOF runner](experiments/run_model407_component_fp_oof.sh).
- **독립 브레인스토밍 갱신:** 이미 시도한 RGB+grayscale/local-contrast/Sobel/LoG 8채널 입력(Model118, 79.07%, 미채택), rare-phase hard crop/component-anchor sampling(Model311/395 계열), global SupCon(Model403, 음의 OOF delta), 단순 pairwise margin(Model117–120/376–377)은 반복하지 않는다. MicroNet encoder도 초기 설정에서 Model91 ResNet34 `72.15%`, Model92 EfficientNet-B4 `69.76%`로 시험되어 당시 채택되지 않았으므로 같은 조건을 반복하지 않는다. Model407 이후에도 작은 성분 상간 혼동이 남는다면 우선 후보는 제한된 tiny-GT 주변의 중간층 표현을 구분하는 local feature/prototype hard-example 보조항이며, Model407 전체 grouped-OOF/component-error audit 후 Claude Code 검토와 별도 사전등록이 필요하다. 새 문헌 가설인 MatSSL은 ImageNet 초기화 ResNet50에 금속조직 unlabeled 이미지의 SSL과 Gated Feature Fusion을 적용한 뒤 U-Net++로 fine-tune해 MetalDAM에서 ImageNet 대비 `+3.22%p`를 보고했다. 다만 이는 UHCS/MetalDAM/Aachen의 약 29.9k–46.7k SSL patch와 steel metallographic 데이터 조건이며, KoMaP의 70장 Al-Si 이미지·HRNet·공식 image-wise metric에서 효과가 검증된 것이 아니다. 따라서 바로 적용하지 않고, 현행 OOF 결과 및 domain-data 접근성을 확인한 뒤 Claude가 local-feature 후보와 함께 검토하도록 한다([MatSSL preprint](https://arxiv.org/abs/2507.18184), [TopoDICE 비교](https://www.nature.com/articles/s41524-025-01801-4)).

- **MatSSL 데이터 접근성 screen:** 공개 메타데이터상 UHCS는 NIST handle `11256/940`, Aachen–Heerlen은 Figshare, MetalDAM은 GitHub 저장소에서 조회 가능하다. MetalDAM 저장소는 grayscale SEM 42 labeled + 164 unlabeled를 명시하지만 MatSSL preprint는 178 unlabeled라고 기술해 버전/카운트 차이를 사용 전에 확인해야 한다. Aachen은 steel SEM이고, MatSSL은 이들 source에서 256×256 patch 약 29.9k–46.7k개로 사전학습했다. KoMaP은 광학 RGB Al-Si라 도메인 차와 외부 데이터 사용 조건을 별도로 점검해야 한다. 다운로드·학습은 아직 하지 않았고, Model407 뒤 Claude 검토에서 성분 audit와 함께 후보 우선순위를 정한다([UHCS/NIST 경로](https://github.com/bdecost/uhcsdb), [Aachen Figshare](https://springernature.figshare.com/collections/Aachen-Heerlen_Annotated_Steel_Microstructure_Dataset/5185004), [MetalDAM data repo](https://github.com/ari-dasci/OD-MetalDAM), [MatSSL paper](https://openaccess.thecvf.com/content/ICCV2025W/CV4MS/papers/Nguyen_MatSSL_Robust_Self-Supervised_Representation_Learning_for_Metallographic_Image_Segmentation_ICCVW_2025_paper.pdf)).

## Model408: tiny-GT 혼동쌍 hinge (NO-GO, 2026-10-06)

- Model407에서 FP-island 억제가 tiny-GT 회수율까지 낮춘 뒤, Model408은 negative loss를 빼고 post-augmentation tiny GT component의 오분류만 겨냥했다. Al3Ni→Al, Eutectic Si→Al/Al3Ni에 zero-margin hinge를 적용하고 CE+Dice+boundary, 구조, crop, optimizer, 8 epochs는 고정했다. component bins는 augmented pixel `[4,16)`, `[16,64)`, `[64,256)`이며 component loss gradient는 outer-train crop으로 fold별 main gradient의 10%에 보정했다.
- 외부 train 70장/59 specimen group, 고정 4-fold paired OOF에서 후보 `79.0401128%`, matched Model406 v3 대조군 `79.0112214%`; Δ`+0.0288914%p` (group-bootstrap 95% CI `[+0.0087219,+0.0490260]`). 양수 fold `3/4` (fold별 `+0.04237,+0.06236,−0.00346,+0.01154`%p). 후보−Model392 parent는 `+0.06262%p`이나 CI `[-0.13320,+0.27543]`, Step4 변화 `−0.12632%p`다. Class IoU 변화(Al/Al3Ni/Primary Si/Eutectic Si)는 `+0.01878/+0.02651/−0.00237/+0.07265%p`; Step4 후보−matched-control은 `+0.05946%p`다.
- 사전등록 gate 10개 중 8개 통과, 2개 실패: 최소 mIoU gain `+0.20%p` 미달 및 unmatched tiny predicted-component 수의 두 표적 클래스 비증가 조건 실패. group CI 하한 양수와 3/4 fold 개선은 확인됐지만 실효 크기는 작아 **NO-GO**, seed-2 확인이나 official valid/test 단계로 승격하지 않는다.
- 작은 GT component의 회수율은 개선됐다. native 8-connected `[1,64)`px component 중 ≥50% 정답 예측 hit-rate는 Al3Ni `60.45→61.82%` (`+1.370%p`, CI `[+1.106,+1.611]`), Eutectic Si `67.58→69.83%` (`+2.248%p`, `[+1.979,+2.519]`). `[64,256)`px 변화는 각각 `+0.420/+0.233%p`. 세부 native bin의 평균 component recall도 1–3px에서 `+2.161/+5.780%p`, 4–15px에서 `+1.928/+3.586%p`, 16–63px에서 `+0.735/+1.241%p`(Al3Ni/Eutectic Si) 개선됐다.
- 동시에 GT와 겹치지 않는 native 1–63px 예측 성분은 이미지당 Al3Ni `317.87→341.59` (`+23.71`, paired 95% CI `[+19.42,+28.22]`), Eutectic Si `207.44→234.91` (`+27.47`, `[+21.51,+33.69]`) 늘었다. 해당 class FP 픽셀도 각각 `+172.5`, `+449.3/image` 증가했다. 1–3px GT 혼동 감사에서 Al3Ni는 정답 픽셀 비율 `+2.01%p`와 함께 Eutectic Si로 간 비율 `+1.25%p`; Eutectic Si는 Al 및 Al3Ni 오분류를 각각 `−2.90/−2.88%p` 줄이고 정답 비율을 `+5.78%p` 올렸다. 따라서 small-GT recall은 좋아졌지만 precision/island가 악화되어 mIoU 개선을 거의 상쇄한 것으로 해석한다.
- 별도 면적 분해는 공유 Model404 analyzer를 Model408 candidate / Model406 control / Model392 parent 경로로 실행한 **train-OOF 설명용 진단**이며 adoption gate가 아니다(JSON 내부의 generic experiment label은 Model404로 남아 있다). tiny `[1,64)` 예측 성분 총수는 이미지당 Al3Ni `+34.94`, Eutectic Si `+41.61`; 그중 zero-overlap은 `+23.71/+27.47`이다. tiny-bin FP 픽셀은 `+200.41/+288.87` 늘었고, zero-overlap 성분 안의 FP 면적은 `+97.90/+200.77` 증가했다. 나머지 tiny-bin 일부-overlap FP도 point estimate상 `+102.51/+88.10` 늘어 완전히 고립된 island만의 문제는 아니다. [공유 analyzer 출력](experiments/results/model408_predicted_component_fp_audit.json), SHA-256 `ebfa47034f931c0717c9c06b9458924f840c5169988df63c2e39a68aa87ed873`.
- Step4와 non-Step4 분해에서도 target FP는 양쪽 strata에서 늘었다. Al3Ni false-positive pixel 증가는 각각 `+276.7/+107.0 image`, Eutectic Si는 `+432.6/+459.7 image`; tiny unmatched component 증가는 각각 `+28.15/+20.93`, `+25.07/+28.98 image` (Step4/non-Step4 순)다. 방향이 모든 strata에서 같으므로 한 Step 조건에만 라우팅하면 해결된다는 근거는 없다.
- 고정 32 crop/fold/epoch×8 train-only exposure audit에서 Model408 loss bin `[4,16)` augmented px(대략 원본 1–3px에 대응) crop hit 비율은 Al3Ni `63.3–76.2%`, Eutectic Si `30.1–36.7%`였다. Eutectic Si `[16,64)` bin은 `87.5–94.9%`, `[64,256)`은 `97.7–98.8%`였다. 즉 극소 Eutectic GT가 학습 crop에서 상대적으로 드물게 노출되는 정황은 있지만, 이 탐색 audit 자체는 sampler/loss 변경의 성능 근거가 아니다.
- 다음 실험은 아직 시작하지 않는다. 새 증거는 단순히 tiny GT 회수를 늘리는 것보다 드문 작은 Eutectic 노출과 증가한 배경 island를 동시에 다뤄야 함을 보여준다. 기존 global SupCon, generic pairwise margin, hard-crop/anchor sampler, Model407식 일반 predicted-component 억제를 그대로 반복하지 않고, Claude Code 독립 검토 후 한 가지 후보만 사전등록한다. quota 재개 예정 2026-10-06 23:00 KST 전에는 Claude CLI를 재호출하거나 우회하지 않는다.
- 공식 valid/test는 scorer·GT audit·exposure audit·FP-area audit 모두 읽지 않았다. 결과/lock/claim/audit를 P100에서 내려받아 SHA-256 대조했다: source lock `73478db6d28f30dd578be58b77d00aed806ffc75e77630bc8251fc3f76283c6c`, calibration lock `ba3cfadd36649ff2b4c10d75762e108ef783a93fed464ef8c1dcb390861b3fbc`, OOF lock `950ab9de267e976d2d7bcc9226b6536d198b410b5b4d5fc581d2f6611fd030e7`, OOF result `3a7dfdfaf5be12651fd756bd8a488bb0ee4fdfdda9faee194c3b8abab94b8899`, GT audit `7aa09380e15495124953dcc89ca32e5bd10aae16539a7c49d6c1b577bd4b4863`, crop-exposure audit `ec8c10dfcaaa51272ca1571238bbb819706f3c206a0853042f3f36da77e89f34`, FP-area audit `ebfa47034f931c0717c9c06b9458924f840c5169988df63c2e39a68aa87ed873`. [OOF 결과](experiments/results/model408_tiny_confusion_oof.json), [GT 성분 감사](experiments/results/model408_tiny_confusion_audit.json), [crop 노출 감사](experiments/results/model408_tiny_component_exposure_audit.json), [FP 면적 감사](experiments/results/model408_predicted_component_fp_audit.json), [사전등록](experiments/model408_tiny_confusion_preregistration.md), [Claude 검토 prompt](experiments/model408_claude_review_prompt.md).
- HRNet-W48은 stride-2 stem feature를 최고 해상도로 사용하므로 scale 2 입력에서 원본 1–3px 조각은 초기에 약 1–3 feature cell 크기다. 다만 이번 실패를 stride-1 skip 부족으로 단정하는 OOF 시각 근거는 아직 없다. 얕은 stride-1 detail skip은 Claude가 crop exposure/false-island tradeoff와 함께 우선순위 후보로 제안할 경우에만 검토한다.

## Model405: 작은 성분 분기 교체 고정 OOF 평가 (2026-10-06)

- Model404 component-recall 분기를 기존 Model397 고정 앙상블의 Model392 seed1 자리에 한 번만 대체했다. 기준은 0.50*B4 + 0.25*seed1 + 0.25*seed2, 후보는 0.50*B4 + 0.25*Model404 + 0.25*seed2였다. 가중치·threshold·routing 탐색은 없었다.
- 70 train images/59 specimen groups, 4-fold train-OOF, official image-wise 4-class mIoU(empty=0), CPU-only. 기준 점수 79.9858620%를 재현했고 후보는 79.9700230%, 변화 −0.0158390%p (paired group-bootstrap 95% CI [−0.0749588,+0.0358405], 양수 fold 2/4)였다. Al/Al3Ni/Primary Si/Eutectic Si IoU 변화는 +0.0107/+0.0004/+0.0058/−0.0802%p였다.
- 1–63 native-pixel GT 성분 hit-rate 변화는 Al3Ni +0.5140%p (95% CI [+0.2349,+0.8650]), Eutectic Si +0.1676%p (CI [−0.4218,+0.6255])였다. 반면 same-class GT와 겹치지 않는 tiny 예측 성분은 이미지당 각각 +4.89 (CI [+2.09,+7.83]), +4.03 (CI [−1.80,+8.45]) 증가했다. target FP pixel은 이미지당 +45/+164, FN은 −6/−105 변화했다. 재현율을 소폭 얻는 대신 작은 오검출을 늘렸고 특히 Eutectic Si 순 IoU는 낮아졌다.
- 사전등록 mIoU 최소 이득/CI/fold 조건, Eutectic IoU 안전 조건, 두 target의 고립 tiny-FP 비증가 조건을 통과하지 못해 NO-GO; 조합은 채택하지 않는다. Valid/test 미사용. [OOF JSON](experiments/results/model405_small_component_fixed_blend_oof.json), [입력 lock](experiments/results/model405_small_component_fixed_blend_lock.json), [one-shot claim](experiments/results/model405_small_component_fixed_blend_oof.json.claim), [사전등록](experiments/model405_small_component_fixed_blend_preregistration.md), [CPU scorer](experiments/model405_small_component_fixed_blend_oof.py).
- SHA-256: lock 74d5a8ee5087d877620f5d929f790c89ee0e9fe92aa030c568b03c7d39e172b3; result e83a71b28630ce4e9e32e40169efc9a8ce33813e559a8319e6abb43cab443cf1; claim 2251fa7db8c07856c3ffc799a82f4142116f5454b2e674e9f1ba89a231eceea7.

## Model406 v3: 작은 성분 recall + hard-negative island 억제 (paired OOF NO-GO)

- Model404 v6가 작은 GT 성분 hit-rate를 올리는 대신 작은 고립 FP island를 늘렸고, Model405의 분기 교체는 OOF mIoU 개선에 실패했다. Model406는 같은 GT component recall 보조손실에 Al3Ni/Eutectic-Si의 상위 0.1% 비-GT 확률 hard-negative 억제를 결합한다. 같은 클래스 GT의 1px halo는 보호한다. 고정 후보 하나이며 threshold/weight 탐색은 없다.
- 첫 batch-4 gradient calibration은 메모리 부족으로 중단되어 optimizer step과 체크포인트가 없었다. Model406 v2는 outer-train만 사용해 fold당 32 crop을 batch 1로 gradient scale 보정했고 4/4 fold, 각 32/32 crop 완료했다. 후보 auxiliary weight는 `0.00209941 / 0.00211753 / 0.00211942 / 0.00206013`; 이 값은 성능 측정치가 아니다. calibration 잠금 SHA-256 `17156f9216375d709fd459c1e561fec595dffc08ef885b1158bab93b4997ee63`.
- 2026-10-05 19:26 UTC에 8개 arm 학습 및 70장/59-group grouped train-OOF 추론이 끝났고, 42개 입력의 OOF lock SHA-256 `21b3152353958adade3a3d1690b08c2e7567adea9bdd03fb1d41d9e76dec43c3`을 생성·검증했다. one-shot official metric(empty=0, per-image 4-class mean) scorer에서 candidate `79.03106797%`, matched control `79.01122142%`, 차이 `+0.01984654%p` (paired specimen-group bootstrap 95% CI `[+0.00751303,+0.03266302]`, positive images 40/70, positive folds 4/4; fold delta `+0.04371/+0.01386/+0.00803/+0.01273%p`)였다. class delta(Al/Al3Ni/Primary Si/Eutectic Si)는 `+0.01260/+0.01325/+0.01874/+0.03480%p`, Step4 `+0.03707%p`다. 통계적 방향은 양수지만 gate에 사전 고정한 최소 `+0.20%p` 개선폭에 크게 미달했고, tiny predicted-component guard도 실패해 최종 **NO-GO**다. 공식 valid/test 미사용.
- paired tiny-island 진단에서 same-class GT와 겹치지 않는 1–63 native-pixel 예측 성분은 Al3Ni `+18.43/image` (95% CI `[+15.36,+21.63]`), Eutectic Si `+17.91/image` (`[+14.07,+21.89]`) 늘었다. target FP pixels는 각각 `+15.67` 및 `+148.69/image` 변했다. 별도 descriptive audit는 1–3/4–15/16–63 native-pixel GT component mean recall이 Al3Ni `+1.664/+1.274/+0.401%p`, Eutectic Si `+3.587/+1.918/+0.457%p` 개선됐고 여섯 group-bootstrap 95% CI가 모두 0 위임을 보였다. 즉 작은 정답 성분 recall은 전 크기 구간에서 올랐지만, pixel top-0.1% hard-negative는 고립 FP component 수를 억제하지 못했다. 이 결과로 다음 후보는 positive component recall 항은 유지하고 pixel hard-negative를 prediction-connected-component FP penalty로 대체하는 단일 변경으로 좁힌다. Model406 결과 [JSON](experiments/results/model406_component_hardnegative_oof_v3.json), [one-shot claim](experiments/results/model406_component_hardnegative_oof_v3.json.claim), [OOF lock](experiments/results/model406_component_hardnegative_oof_lock_v3.json), [secondary size/confusion audit](experiments/results/model406_tiny_component_confusion_audit_v3.json).
- 원격 P100에서 받은 결과·claim·source/OOF lock·v2 calibration lock chain과 secondary audit를 로컬에 복사한 뒤 원격 SHA-256과 일치함을 확인했다. 주요 해시: result `9cf5f30011d60cce645653f38dd4e48b0d964f2cdebb00d6950bd2605bd5b33e`, claim `dc1bf1ea9ab5080b185325670633b936d3d132d5147b09889d478d19225d36d6`, secondary audit `ca54e5ace678de5b4a4477a0d1711cc5b05da321035eee2f3dc7f691d6f41c0e`. 이미지, GT mask 및 전체 확률 archive는 복사하지 않았다.
- 2026-10-05 17:21 UTC에 원격 fold2 `args.json` 전체를 비교했다. control/candidate의 차이는 의도된 `arm`, `component_weight`(0 vs 0.0021175321474467), 별도 저장을 위한 `output_dir`뿐이며 나머지 모든 CLI 설정은 동일하다.
- **Model406 v3 metric/scorer 독립 확인:** final scorer가 각 이미지에서 4개 class IoU를 계산하고 empty union을 0으로 둔 뒤 이미지별 평균을 내므로 대회 image-wise mIoU와 일치한다. 70 train images/59 specimen groups를 4-fold OOF로 한 번씩 채점하고 20,000 paired group-bootstrap draws 및 별도 tiny-island FP 진단을 수행한다. 점수 경로는 공식 valid/test를 열지 않으며 전체 OOF artifact hash lock 이후에만 one-shot 실행된다. lock chain을 추적한 결과 Model392 parent checkpoint·parent OOF maps뿐 아니라 Model406 loss/trainer와 scorer 의존 파일도 v1 source lock에 들어 있고 v2/v3 lock에서 transitively 검증된다. 확인 코드: [`scorer`](experiments/score_model406_component_hardnegative_oof.py), [`metric`](model397_reproduction_bundle/scripts/score_model397_oof.py), [`OOF lock`](experiments/model406_component_hardnegative_v3_lock.py), [`v1 source lock`](experiments/model406_component_hardnegative_lock.py).
- Claude Code read-only 자문은 CLI 주간 한도(`resets 11pm Asia/Seoul`)로 아직 실행하지 못했다. quota를 우회하지 않으며, result/claim/source·OOF lock과 secondary audit의 로컬 사본 및 SHA-256 대조가 완료되어 [Model406 검토 프롬프트](experiments/model406_claude_review_prompt.md)를 이제 실행 가능한 상태다. full probability archive, train image/mask 및 official valid/test는 복사하지 않았다.
- **Model406 작은 GT 성분 혼동 보조 진단 완료:** [`model406_tiny_component_confusion_audit.py`](experiments/model406_tiny_component_confusion_audit.py)가 전체 OOF lock 검증 뒤 train-OOF 70장/59 groups에서 1–3/4–15/16–63 native-pixel GT components의 predicted-class pixel 분포, component recall 및 paired specimen-group bootstrap CI를 집계했다. 이는 설명용 진단이며 mIoU gate/threshold 선택에 사용하지 않고 official valid/test도 읽지 않았다. 결과 [JSON](experiments/results/model406_tiny_component_confusion_audit_v3.json), SHA-256 `ca54e5ace678de5b4a4477a0d1711cc5b05da321035eee2f3dc7f691d6f41c0e`; 원격 사본과 일치 확인했다.
- **학습 중 확인한 size-bin 해석 주의:** train.zip의 70개 원본 이미지 width는 `675/1230/1278/1440px`, 고유 이미지 수 `17/20/24/9`다. dataset width-normalization 공식상 nominal scale 2에서 실제 scale은 `3.644/2.000/1.925/1.708`이다. 따라서 fixed augmented-grid bin 전체 `[4,256)`은 변환 전 native 면적으로 대략 `0.30–19.3 / 1–64 / 1.08–69.1 / 1.37–87.8px`에 대응한다. CPU mask-only 감사에서 native 1–63px component 중 고정 bin에 들어오는 비율은 width675 이미지에서 Al3Ni `16,203/19,277 (84.1%)`, Eutectic Si `9,165/13,433 (68.2%)`; width1230은 둘 다 100%; width1278은 `99.0/99.5%`; width1440은 100%였다. 반대로 native 64–255px 성분 중 고정 bin에 들어오는 것은 width1278에서 `535/528`, width1440에서 `697/657`개였다. 이는 crop 절단·border 제외와 geometric augmentation 전의 명목 면적 대응이지 실제 optimizer 노출량이 아니다. 현 진행 중 Model406 recipe는 사전 잠금되어 바꾸지 않는다. 다음 component 실험은 image-specific scale/native-area 기준의 mask-only 사전감사부터 해야 한다.
- **Model406 fold2 scale 재감사 (2026-10-05 16:55 UTC, train.zip만):** 현재 fold2 trainer가 출력한 `target_width=1278`에 맞춰 size-bin audit를 재실행했다. 고정 augmented bin `[4,256)`은 width675에서 native 약 `[0.279,17.85)`px에 해당해 native 1–63px 성분 중 Al3Ni 81.6%, Eutectic Si 64.3%만 포착한다. width1230에서는 native 약 `[0.93,59.28)`px로 각각 98.5%, 96.7%; width1440에서는 `[1.27,81.25)`px로 native 64–255px 성분 Al3Ni 569개, Eutectic Si 498개도 bin에 포함된다. 이 감사는 전체 원본 GT 기준의 기하학적 근사라 crop/augmentation 후 실제 optimizer 노출과 같지 않다. valid/test/모델 예측을 사용하지 않았고 현재 v3 잠금 recipe는 변경하지 않는다.
- **후속 scale-aware component loss 설계 주의:** 실제 dataset은 원본 crop→resize 후 `Affine(scale=0.8–1.2)` 및 elastic/grid distortion을 적용하고, 현재 component loss는 변환된 mask를 다시 연결요소 분석한다. 그러므로 width별 원본 면적 대응만으로 실제 bin 노출을 정확히 알 수 없다. 다음 실험에서는 원본 train mask에서 target-class connected-component ID와 native area를 먼저 만들고, mask/image와 같은 crop·resize·기하변환에 ID map을 nearest-neighbor로 전달하는 방안을 검토한다. 변환 후 같은 원본 ID가 여러 조각으로 보이면 원래 native area bin을 유지하되, crop 부분 관측의 평균 확률·border 정책 및 ID 보존을 train-only smoke/노출 감사로 먼저 검증한다. 현 진행 중 v3 잠금 recipe에는 반영하지 않는다.
- **native component 크기 분포 보완:** train.zip 전체 mask의 8-connected Al3Ni 89,366개 중 1px는 2.1%, 2–3px는 6.2%이고, 4–15px 33.3%, 16–63px 32.6%다. Eutectic-Si 63,393개 중 1px는 1.0%, 2–3px는 2.4%, 4–15px 20.4%, 16–63px 44.9%다. 따라서 small-component 목표를 단순 raster speckle만으로 치부할 수는 없지만, 1–3px 라벨은 별도 품질/시각성 audit 없이 크게 가중하지 않는다. train GT만 사용했고 공식 valid/test 및 모델 예측은 읽지 않았다.
- **Model406 결과 기반 다음 실험 설계:** GT 작은 성분 recall은 여섯 native-size×class 셀 모두 개선됐지만 unmatched 1–63px prediction components는 두 target 모두 증가했다. 따라서 후보는 positive component-recall term을 유지하고 top-0.1% pixel hard-negative를 8-connected argmax false-component penalty로 교체하는 단일 mechanism 변경이다. 예측 partition은 `argmax(logits.detach())`로 만들고, same-class GT의 1px halo와 겹치거나 crop border에 닿는 component는 boundary/truncation 보호 차원에서 제외한다. 남은 target-class component마다 내부 mean class probability를 `-log(1-p)`로 벌하고, crop×class cell별 합산해 오검출 component가 많은 crop에 더 큰 억제력을 준다; 보조 gradient norm은 outer-train 32 crop/fold, optimizer step 0으로 10% main-loss decoder gradient에 맞춘다. 시작점·4 group folds·optimizer·8 epochs·inference는 Model406 v3와 고정하고, 동일 Model406 v3 control OOF를 paired baseline으로 재사용하되 source/OOF lock에서 그 archive hashes를 고정한다. adoption에는 기존 `+0.20%p` mIoU/CI/fold/class/Step4/component gates와 두 target unmatched tiny-component 비증가 조건을 모두 요구한다. 이 설계는 현재 미실행이며, implementation smoke와 preregistration lock 이전에는 학습하지 않는다. P100 code는 Claude quota reset 전 독립 검토하고, Claude Code review는 한도 초기화 후 보완한다.
- **Al3Ni–Eutectic 혼동 후속 후보 재검토:** 단순 pairwise logit-margin은 새 실험으로 반복하지 않는다. Model117–120에서 standalone margin 모델이 기준 ensemble보다 낮았고, pairwise source 추가는 image-wise `+0.0034%p`로 사실상 동률이었다; 이후 Model376/377 방향성 감사도 swap 감소 gate를 통과하지 못했다. 따라서 Model406 secondary audit에서 이 두 상 사이의 혼동이 계속되더라도 다음 후보는 동일 margin 재현이 아니라, 분리된 local-texture/feature-hard-example 기전이어야 하며 Claude 검토와 새 grouped-OOF 사전등록 전에는 미실행한다.
- **추가 문헌 screen:** Mun et al.의 size-balanced CE는 작은 객체 개선 아이디어지만 image-level weak supervision/pseudo-label 설정에서의 결과라 현재 fully-supervised dense-mask 과제에 직접 이전할 수 없다 ([WACV 2024](https://openaccess.thecvf.com/content/WACV2024/html/Mun_Small_Objects_Matters_in_Weakly-Supervised_Semantic_Segmentation_WACV_2024_paper.html)). Model404/406은 이미 GT component recall을 class×size-bin별로 균형화하므로 같은 방향의 일반적 size-reweighting보다, v3의 paired OOF 결과에서 실제 잔존 FP island가 확인될 경우 prediction-derived FP 억제의 독립 효과를 검증하는 편이 더 구분력 있다.
- **금속조직 shape prior 주의:** Al-Si-Ni 방향응고 연구는 Al3Ni 형태가 응고 조건에 따라 plate-like에서 fishbone으로 바뀔 수 있다고 보고하고, Al-Si 연구에서도 공정 Si가 거친 판상에서 섬유상으로 개질될 수 있다 ([Al-Si-Ni 방향응고 연구](https://www.sciencedirect.com/science/article/abs/pii/S2589152919301541), [Al-Si eutectic morphology 연구](https://www.sciencedirect.com/science/article/abs/pii/S0921509305011147)). 그러므로 전 데이터에 고정 aspect-ratio/shape filter를 적용하는 것은 위험하다. morphology/context를 추가로 쓰려면 train-only 분포 감사로 class·크기 조건부 신호를 확인하고, 학습된 표현을 grouped OOF로 검증한다.
- **새 금속조직 hard-example 문헌 screen (후속 가설, 미실행):** Xie et al.의 HEL-NC는 24장의 645×484 회색조 steel micrograph(4개 phase)에서 ETF prototype classifier와 prototype 유사도 기반 hard-pixel 가중을 결합해 class imbalance를 다뤘다 ([Applied Soft Computing, 2025](https://doi.org/10.1016/j.asoc.2025.112755)). 이는 작은 GT connected component를 직접 다루는 방법은 아니며 KoMaP의 RGB/4-class image-wise mIoU 개선 증거도 아니다. Model403 global SupCon은 NO-GO였으므로, 이 방법은 현재 Model406 변경 근거가 아니라 OOF에서 Al3Ni/Eutectic의 클래스 혼동이 계속 확인될 때만 Claude가 비교 검토할 별도 representation/hard-example 가설이다.

## 최신 결과 요약 (2026-10-06)

- **Model406 v3 small-component recall + hard-negative OOF:** 70장/59-group image-wise mIoU(empty=0) candidate `79.0311%` vs matched control `79.0112%`, `+0.01985%p` (paired group CI `[+0.00751,+0.03266]`, folds 4/4 양수). 그러나 최소 `+0.20%p` 및 unmatched tiny-FP nonincrease gates에 실패해 **NO-GO**. 작은 GT component recall은 1–63px 모든 크기 구간에서 두 target 모두 개선했지만 unmatched tiny prediction CCs/image는 Al3Ni `+18.43`, Eutectic Si `+17.91` 늘었다. valid/test 미사용. [OOF](experiments/results/model406_component_hardnegative_oof_v3.json), [크기별 진단](experiments/results/model406_tiny_component_confusion_audit_v3.json).
- **Model405 작은 성분 분기 교체:** image-wise train-OOF mIoU 79.9700%, 기준 Model397보다 −0.0158%p (CI에 0 포함, 2/4 folds 양수), 사전등록 NO-GO. GT 작은 성분 hit-rate는 조금 올랐지만 고립 tiny 오검출도 늘었다. valid/test 미사용. [상세](experiments/results/model405_small_component_fixed_blend_oof.json).
- **Model404 v6 작은 성분 grouped-OOF 완료 (2026-10-05):** 70장/59 specimen-group, official empty=0 기준 candidate `79.0433%`, compute-matched control `79.0114%`, Model392 seed1 parent `78.9775%`. Candidate−control `+0.0319%p` (paired group-bootstrap 95% CI `[+0.0182,+0.0458]`, 4/4 folds 양수)이지만 사전등록 최소 개선폭 `+0.20%p`에 미달해 **gate NO-GO; 채택하지 않음**. 나머지 control/parent/class/Step4/component-hit safety checks는 통과했으나 parent 비교 CI `[-0.1321,+0.2781]`는 0을 포함한다. 공식 validation/test 미사용. [전체 OOF JSON](experiments/results/model404_component_recall_oof_v6.json).
- **Model404 v6 component FP 사후 감사:** 작은 GT component hit-rate는 Al3Ni `+1.822%p` (95% CI `[+1.591,+2.033]`), Eutectic-Si `+1.715%p` (`[+1.483,+1.954]`) 개선됐다. 동시에 tiny predicted components는 이미지당 Al3Ni `+47.1`, Eutectic-Si `+41.9`; 같은 상 GT와 전혀 겹치지 않는 tiny predicted components는 각각 `+30.9`, `+24.6` 증가했다. Tiny component FP pixels는 이미지당 `+283/+231`; 전체 target-class FP pixels는 `+266/+171`, FN pixels는 `−273/−142` 변화했다. 즉 미검출 감소와 함께 작은 가짜 조각이 늘어 mIoU 순이득을 상쇄한 것으로 해석된다. Train-OOF 70장/59 groups의 진단 전용 결과이며 새 gate/threshold 선택에 쓰지 않는다. [FP audit JSON](experiments/results/model404_predicted_component_fp_audit_v6.json).
- **Model403 train-only grouped OOF — gate 불통과로 조기 종료 (2026-10-05 11:07 UTC):** 완료 fold1·fold2의 paired deltas가 각각 `−0.2847/−0.5675%p`라서, 채택에 필요한 `3/4 fold 양수` gate는 나머지 두 fold가 모두 양수여도 도달 불가능했다. 임의 futility 기준을 새로 만든 것이 아니라 잠긴 채택 gate의 수학적 불가능성을 확인해 종료했다. fold1·2만 70 epoch 및 held-out OOF가 완료(36/70장); 부분 평균 Model403 `77.1141%`, 같은 이미지의 Model392 seed1 `77.5402%`(`−0.4261%p`). Eutectic-Si image-mean IoU 변화는 fold별 `−1.0134/−1.1041%p`. **이는 불완전한 train-OOF 기술값이지 최종 OOF 또는 valid 점수가 아니다.** fold3은 epoch22/70에서 종료했고 `last.pt`·`train_state.pt`를 보존, fold4 미실행; 전체 gate/aggregate는 미산출. valid/test 미사용. 독립 검토에서는 global SupCon의 morphology 억제 위험을 지적했다. [방법론 검토](experiments/model403_independent_review.md), [사전등록·실행 기록](experiments/model403_local_supcon_preregistration.md).
- **Model404 v6 작은 island 교정 (2026-10-05 11:44:55 UTC, candidate fold1 학습):** Model397 고정 train-OOF 70장/59-group에서 native 1–<64px Al3Ni pixels의 26.26%가 Al·12.86%가 Eutectic Si로, 작은 Eutectic Si 픽셀의 21.91%가 Al3Ni·11.46%가 Al로 예측됐다. 과반 recall 미달 성분은 Al3Ni 40.83%, Eutectic 31.90%이며 완전 미검출은 25.18%, 21.51%다. v6는 두 클래스를 대상으로 component loss를 crop×class×size-bin 균등 평균하고 CE+Dice/boundary loss를 유지한다. outer-train-only 8-batch/fold 보정 완료(optimizer step 0); fold1–4 고정 weight `0.00242777/0.00257845/0.00292623/0.00275332`. control/fold1 학습 8/8 epoch 및 held-out train OOF 확률맵 저장이 끝났고, candidate/fold1은 epoch 4/8 checkpoint 저장(11:44:04 UTC), epoch 5 step 50/156 진행 중이다. 전체 paired OOF scorer 전까지 fold별 중간값으로 성능을 판단하지 않는다. Model404 최종 OOF 미산출, 공식 valid/test 미사용. Model403은 3/4-fold 양수 gate가 수학적으로 불가능해 fold3 epoch22 후 종료했고 checkpoint를 보존했다. [혼동 audit](experiments/results/model404_tiny_component_confusion_audit.json), [v6 protocol](experiments/model404_component_recall_preregistration_v6.md), [v6 lock](experiments/results/model404_preregistration_lock_v6.json), [FP component 진단기](experiments/model404_predicted_component_fp_audit.py), [Claude read-only review prompt](experiments/model404_claude_review_prompt.md). Claude quota reset: 2026-10-06 23:00 KST.
- **후속 가설(미실행): 예측 성분에도 명시적 FP 비용 부여.** 현재 Model404 component loss는 GT 성분에서 정답 class 확률을 키우지만, prediction-derived false-positive component 자체를 별도 항으로 벌하지는 않는다. BiCC preprint는 GT·prediction 성분 partition을 양방향으로 쓰고 예측 성분의 GT-empty 영역에 mass-based penalty를 주지만, 의학 영상 5개 3D 데이터셋 결과이지 KoMaP 전이 증거는 아니다. 공개 저자 구현도 binary 3D 전용 API(`[B,2,D,H,W]` logits, `[B,1,D,H,W]` target)이므로 KoMaP 2D 4-class에 직접 꽂을 수 없다. 현재 실험에서 tiny hit-rate는 오르나 pixel FP/class-IoU 또는 official mIoU가 악화될 때만, class-wise 2D prediction-component penalty를 별도 matched grouped-OOF 실험으로 검토한다; 현재 학습에는 반영하지 않는다. [BiCC preprint](https://arxiv.org/abs/2609.30223), [저자 구현](https://github.com/TIO-IKIM/BiCC-Loss).
- **작은 성분 loss 후속 문헌 검토(미실행):** 2026년 다중클래스 3D 의료영상 연구는 one-vs-rest로 클래스를 분해한 뒤 각 클래스 안에서 CC/Blob connected component loss를 균등 집계한다. CC loss는 voxel Dice 쪽, Blob loss는 instance detection 쪽에서 상대적으로 강했고, 성분 내부 inverse-size 가중은 rare-class Dice를 높이는 대신 검출 품질을 낮췄다. 현재 Model404의 GT component probability loss와는 달리 local CC loss는 성분 주변 Voronoi 문맥의 FP도 함께 보므로, v6에서 recall은 오르고 pixel FP가 악화되는 경우에만 다음 가설로 검토한다. MRI/3D 결과라 금속조직 2D·image-mIoU 개선의 증거는 아니다. [논문](https://arxiv.org/abs/2604.24276).
- **다음 구조 후보—DINOv3 independent review:** 팀 PhaseHybrid은 ViT-L/16 중간층+multi-scale CNN이지만 checkpoint null, artifact는 `pretrained:false` wiring test뿐이다. 과거 Model308 DINOv2 residual OOF gain `+0.1626%p`(group CI 하한 `+0.0003`), valid `+0.2034%p`(CI 0 포함), Model309 held-out route gain 0이었다. 팀 trainer의 AMP helper는 BF16/FP32만 허용하고 P100 BF16은 예외, FP16/GradScaler 구현도 없다. DINOv3는 Model403 후 gated weights/license 확보 및 isolated FP16+GradScaler P100 smoke를 통과할 때만 검토한다. `empty=0`으로 평가. [독립 검토](experiments/next_direction_review_20261005.md).
- **Model402 Model397 grouped-OOF 오류 구조 진단 완료:** train OOF 70장/59 specimen group에서 Model397 고정 blend는 `79.9859%`(공식 valid 점수가 아님)이고, Eutectic Si class image-mean IoU가 `59.0029%`로 최저다. Eutectic Si의 1–64px component miss rate는 `89.83%`, 64–256px는 `99.19%`; Al3Ni와 Primary Si는 GT 경계 2px 이내 FN rate가 각각 `27.35%`, `27.38%`다. 서로 다른 오류 구조이므로 boundary 개선과 Eutectic 성분·내부구조 개선 가설을 분리해 검증해야 한다. oracle mIoU delta는 이상적 상한일 뿐 실현 이득이 아니다. valid/test 미사용, 추가 학습 없음. [진단 결과](experiments/results/model402_model397_component_boundary_audit.json), [프로토콜](experiments/model402_model397_component_boundary_audit.md).
- **Model401 Model397 고정 blend 단발 valid 평가 완료:** Model329 B4 4-fold 0.50 + Model392 HRNet-W48 seed1/seed2 각 4-fold 0.25의 공식 valid 20장 image-wise mIoU는 `79.4181709%`다. Model364 `79.9429727%`보다 `−0.5248018%p` 낮아 최고를 갱신하지 못했다. Class means(Al/Al3Ni/Primary Si/Eutectic Si)는 `94.3032/75.0061/86.9067/61.4567%`; test 미사용, 재사용 valid의 selection-conditional 개발 결과다. [결과](experiments/results/model401_model397_blend_valid_once.json), [절차](experiments/model401_model397_blend_valid_once.md), [recovery 절차](experiments/model401_model397_blend_recovery.md).
- **Model400 one-shot valid 평가 완료:** 고정 50:50 Model329 B4 4-fold + Model382 seed2 HRNet 4-fold 확률 평균은 공식 valid 20장 image-wise mIoU `79.3227251%`로, Model364의 `79.9429727%`보다 `−0.6202476%p` 낮았다. test는 사용하지 않았고, valid 재사용·OOF/valid inference estimand 차이 때문에 selection-conditional 결과로만 해석한다. 최고점은 Model364 그대로다. [결과](experiments/results/model400_model396_d2_valid_once.json), [절차·복구 기록](experiments/model400_model396_d2_valid_ensemble_preregistration.md).
- **Model392 auxiliary boundary-head localization 진단 완료(2026-10-04):** boundary head의 sigmoid 출력을 train-only grouped OOF 70장/59 specimen group에서 2 seeds × 4 folds로 재추론했다(140 image-seed records, 8 held-out checkpoints). 전체 원본 해상도 커버리지 100%, 25% tile overlap 및 flip4 Gaussian stitching, threshold-free image AP를 사용했다. Group-macro AP 92.0032% (95% specimen-group bootstrap CI [91.1800,92.7993]%), boundary prevalence 30.2709%, paired AP−prevalence +61.7323%p (CI [+59.0827,+64.2441]%p); seed AP 92.0292/91.9772%, fold AP 90.8125–93.2972%. 사전고정 판정 LOCALIZATION_SIGNAL. 이는 boundary 위치를 예측하는 신호이지 클래스 방향 정확도나 semantic mIoU 이득은 아니다. 사용한 25% overlap은 Model392 semantic OOF의 더 조밀한 overlap과 다르므로 Model392 공식 추론 점수로 해석하지 않는다. 선택조건부 진단이며 valid/test 미사용, mIoU 미계산, 추가 학습 미승인. [프로토콜](experiments/model392_boundary_head_localization_preregistration.md), [코드](experiments/model392_boundary_head_localization.py), [잠금](experiments/results/model392_boundary_head_localization_lock.json), [결과](experiments/results/model392_boundary_head_localization_oof.json). Claude Code 검토 요청은 [프롬프트](experiments/model392_boundary_head_claude_review_prompt.md)에 준비했다.
- **Model392 boundary-to-interior 방향 타깃 지원도 감사 완료(2026-10-04):** 고정 4-fold의 outer-train 마스크만으로 후보 Al3Ni/Primary Si 경계 pixel에서 같은 class의 interior(clearance ≥4px)까지 16px 안에 타깃이 있는 비율을 계산했다. Al3Ni 지원율은 fold별 `71.75–75.31%`, Primary Si는 `94.11–95.49%`; 8개 compass 방향 모두 각 fold에서 관찰됐고 방향별 group coverage는 Al3Ni `44/44–45/45`, Primary Si `43/44–44/44`였다. Sensitivity grid에서 offset 4px 제한 시 지원율은 각각 `37–41%`, `65–68%`로 줄었다. 이는 여러 group/fold에 학습 타깃 기하가 있다는 feasibility 근거일 뿐 방향 예측 정확도나 mIoU 향상 근거가 아니다. CPU 전용 outer-train-mask 감사로 valid/test·OOF prediction·mIoU·GPU 학습은 사용하지 않았다. 방향 head 학습은 아직 승인하지 않았으며 Claude 검토와 별도 OOF 사전등록 gate가 필요하다. [사전등록](experiments/model392_direction_target_support_preregistration.md), [코드](experiments/model392_direction_target_support.py), [잠금](experiments/results/model392_direction_target_support_lock.json), [결과](experiments/results/model392_direction_target_support.json).
- **목표 기준 보정:** 75.55%는 Model66/67의 과거 raw-valid 값이다. 저장된 공식 empty=0 결과 기준 raw validation 최고는 Model364 `79.9429727%`; P100 원본 [결과 JSON](experiments/results/model364_model361_entropy_into_360.json)에서 metric definition, 점수, 반복 valid 선택 경고를 확인했고 SHA-256은 `d93e47139046567909d738014a95ebb1aae33781a1f0061c8a9ffc38afbe1ae1`이다. Per-image 벡터 재평균과 JSON 요약이 일치하며 80%까지 `0.0570273%p` 남았다. Model360 `79.9416620%` 대비 `+0.0013108%p`지만 group-held-out 변화는 `−0.0043870%p`여서 일반화 개선으로 채택하지 않았다. Model349는 `79.9095189%`로 더 낮다. 이 valid 수치는 반복 선택 편향을 포함하며 독립 test 성능이 아니다. Model382 2-seed train-OOF 평균 `79.1622%`는 별도 split의 OOF 점수고 사전등록 채택 gate를 통과하지 못했다.
- **Model392 최종 OOF 결과 (2026-10-04 02:46 UTC):** 2 seeds × 4 folds, train-only 70 images/59 specimen groups, 공식 empty=0 image-wise mIoU. Model382 기준 평균 `78.9395%`, Model392 `79.0437%`, paired Δ `+0.1042%p` (95% group-bootstrap CI `[-0.4175,+0.6007]%p`). Gate 실패: 요구 개선폭 `+0.50%p` 미달, CI 하한이 0 초과 아님, pooled folds 개선은 2/4, 최악 fold `−0.9984%p`. Al/Al3Ni/Primary Si/Eutectic Si 변화는 `−0.091/+0.258/+0.317/−0.067%p`, Step4 `+0.827%p`. `valid_evaluation_authorized=false`; full-train과 official valid 미실행, test 미사용. [결과](experiments/results/model392_widthnorm_two_seed_confirmatory.json).
- **팀 weighted-CE 근거 분리:** 팀이 공유한 HRNet-W48 81.55% 설명은 보고된 레시피 결과다. 별도 `KoMaP_code.zip` B4 `train.py`는 unweighted CE+Dice와 area-proportional image sampling/random crops를 쓰며 inverse-frequency CE 옵션이 없다. ZIP은 weighted-HRNet 레시피 재현 근거가 아니고, 그 crop 분포도 Model392의 50% rare-centered sampler와 다르다. [Model399 초안](experiments/model399_class_weighted_hrnet_finetune_draft.md).
- **Model399 fold-local crop prior audit 완료(2026-10-04):** 4 folds × 2 Model392 coordinate-cap seeds, 70-epoch native crop Monte Carlo와 실제 증강 1-epoch probe를 outer-train mask만으로 계산했다. Rare-centered arm의 증강 후 crop prior/source pooled prior 평균 비율(Al/Al3Ni/Primary Si/Eutectic Si)은 `1.0168/0.9553/0.9892/1.0444`; fold-seed 범위는 Al3Ni `0.9301–0.9768`, Eutectic Si `1.0076–1.0812`다. Uniform-crop 비교 대비 rare-centered 평균 픽셀 비율 변화는 `+0.287/+0.008/−0.548/+0.253%p`로, 50% positive-crop 설정이 class-pixel prior를 50%로 바꾸는 것은 아니며 전체 shift도 크지 않았다. 증강은 resize 직후 대비 평균적으로 작게 prior를 바꿨다(클래스별 비율 평균 `0.9988–1.0017`; 한 fold-seed probe의 최대 편차는 약 `0.85%`). q/p는 fold·seed 평균이며 단회 증강 probe의 Monte Carlo 추정이지 성능 결과가 아니다. mIoU가 image-wise 평균이므로 pooled-pixel prior와 mean-per-image prior를 함께 보존했다. GPU 학습, OOF prediction, valid/test는 사용하지 않았다. [감사 코드](experiments/model399_crop_prior_audit.py), [결과](experiments/results/model399_crop_prior_audit.json).
- **팀 B4 scale 기록의 해석:** ZIP `EXPERIMENTS.md`는 B4 3×가 2×보다 높은 valid image-mean 수치를 보고하지만, 같은 20장 valid를 반복 사용했고 그 문서의 주 계산은 결측 클래스를 제외해 공식 `empty=0`와 다르다. 따라서 scale 효과는 transfer 가설이지 Model392 성능 예측이 아니다. 고정 matched-scale OOF 후보 Model398만 Model392/397 뒤 gate에 따라 고려한다.
- **금속조직 문헌 대조(미검증):** Al-Si 합금 synchrotron-CT 분할 연구는 Si 상과 Si–aluminide 계면 픽셀을 강조하는 pixel-wise loss weighting으로 해당 데이터의 분할·연결성 지표를 개선했다. KoMaP의 2D RGB·4-class image-wise mIoU와 영상 모달리티 및 목표 지표가 달라 직접 성능 근거는 아니다. Model387 OOF에서 Eutectic Si 오류가 주로 경계 안쪽 deep interior에 있었으므로 계면 가중치를 그대로 복사하지 말고, Model399의 class-weighted CE가 Eutectic 내부 오류에도 이득인지 OOF로 분리 확인할 후보로만 둔다. [Strohmann et al., Scientific Reports (2019)](https://www.nature.com/articles/s41598-019-56008-7).
- **추가 후속 가설(미잠금):** Model392는 class-agnostic 2px GT-boundary band를 학습하는 auxiliary head를 포함하지만 현재 추론 함수는 semantic logits만 사용한다. head 자체는 어느 클래스가 경계의 어느 쪽인지 알려주지 않는다. Model387 train-OOF 거리 분석에서는 Al3Ni와 Primary Si의 FN이 `<4px` 경계대에 크게 집중했으나 Eutectic Si는 오류 대부분이 deep interior였다. 또한 folds3/4 image-mean mIoU가 seed1 `83.94/77.74%`, seed2 `83.95/77.71%`로 약 `6.2%p` 차이가 반복돼 specimen-group 구성에 따른 큰 변동을 경계로 둬야 한다. SegFix 원 논문은 경계 위치와 boundary-to-interior 방향을 별도 예측해 경계 레이블을 내부 예측으로 대체하며, 8방향 구분을 사용했다. Cityscapes HRNet 실험의 +0.8–1.0 mIoU는 다른 dataset/metric 결과라 KoMaP 향상량으로 전이할 수 없다. [SegFix 논문](https://arxiv.org/abs/2007.04269). Model392/397 뒤 OOF에서 현재 boundary head의 localization 성능을 먼저 측정하고, 신호가 확인될 때만 GT distance transform에서 same-class interior 방향을 만든 8-way head의 matched continuation을 Claude Opus와 검토한다. 그 후보를 잠글 경우 official image-wise mIoU(empty=0), paired specimen-group bootstrap, class/Step4 및 per-image gain concentration 안전 분석을 사전등록한다. 학습 시작 전 outer-train masks에서 per-class/direction support를 audit하고, same-class interior가 없는 boundary pixels는 direction loss에서 제외하며 inference에서 raw semantic prediction으로 fallback하는 기준을 고정한다. Eutectic의 deep-interior 오류는 이 경계 개입으로 해결되지 않는 위험이다. 현재 학습·score·official valid 튜닝에는 반영하지 않는다.
- **미잠금 multi-scale consistency 가설(낮은 우선순위):** Model392 native-width OOF에서 이질적 효과와 두 scale 간 실제 예측 불안정이 함께 확인되고, 잠긴 Model397/조건부 Model398가 80%에 못 미칠 때만 우선 민감도 진단 후 검토한다. Model300 D4 Mean-Teacher consistency는 OOF `−0.0142%p` (group-bootstrap 95% CI `[-0.0756,+0.0447]%p`)였고, Model253 고정 3× fine-tune도 parent보다 낮았다. 그러므로 consistency나 고해상도 자체의 일반 이득은 가정하지 않는다. Cross-scale 학습은 focused search에서 구현을 찾지 못한 별도 가설일 뿐이며, compute-matched supervised control, group OOF, 얇은 Eutectic Si의 보간·aliasing 안전 gate가 필요하다. Opus review 및 별도 protocol 전에는 미승인·미실행이며 향상량은 예측하지 않는다.
- **PSCL-inspired local-contrastive 후보(미승인·원 논문 직접 재현 아님):** metallography용 PSCL은 image/patch contrastive pretraining, decoder multi-scale local features, feature-similarity 기반 patch sampling을 사용한다. 논문 dataset은 198장(376²)이고, 주요 low-label 실험은 159장 unlabeled pretraining+39장 holdout, annotated image 한 장만으로 Dice `0.6296`(무작위 초기화 baseline `0.5134`)을 보고한다. 별도의 159-labeled-image 설정에서는 PSCL-MoCo/SimCLR가 baseline보다 Dice `+0.0221/+0.0238`을 보고했지만, taxonomy가 다르고 논문은 foreground 평균 Dice를 쓰는 반면 KoMaP 공식 점수는 empty union=0인 per-image 4-class mIoU이므로 gain을 전이하지 않는다 ([논문](https://www.nature.com/articles/s41598-025-32855-5), [공식 코드](https://github.com/neulmc/PSCL)). 검토 후보는 이와 별개로 현재 decoder feature에 mask-aware local contrastive 보조손실을 추가하는 adaptation이다. 즉 outer-train masks로 positives/negatives를 정의하는 건 PSCL 원 구현을 그대로 옮기는 게 아니며, 별도 falsifiable hypothesis다. 원 논문의 supervised-assisted sampler도 labeled class-anchor와 feature-similarity filtering으로 의미상 모호한 negatives를 거른다. 공식 코드 설명에 따르면 MoCo queue는 파일명에 담긴 aging-time/sample metadata도 이용한다. KoMaP filenames의 `Axx_stepN_sampleM` 필드에 어떤 metadata 의미가 안정적으로 대응하는지 별도 확인하고, 같은 클래스 cross-step positive는 specimen-group split을 지키는 범위에서만 검토해야 한다. `forward_with_boundary`가 이미 decoder feature를 반환하므로 encoder/decoder 재추론 없이 projection head를 붙일 수 있지만, contrastive loss의 추가 비용·false negative·specimen memorization은 측정되지 않았다. 기존 Model236은 마스크 없는 global-pooled NT-Xent이고 Model237 supervised fine-tune valid `76.2376%`는 시작 Model195 `76.1109%`보다 `+0.1267%p`였지만 Model234 continuation `76.4220%`보다 `−0.1844%p`였으며 grouped OOF가 없어 대비가 약하다. Model308의 frozen DINOv2 multi-layer patch-feature branch도 512px에서 OOF `+0.1626%p`였으나 Model225 route를 보완하지 못했다. 이는 contrastive 학습과 다른 처치지만, patch feature 추가만으로 충분하지 않다는 선행 결과다. Model121/122 auxiliary-logit deep supervision 및 Model119 probability-space pairwise margin과도 다른 intervention이다. 단순 metric-aligned overlap loss는 새 아이디어가 아니다: Model333 B4 CE+Dice 대비 per-image Lovasz가 `+0.0040%p` (95% CI `[-0.0084,+0.0171]%p`)로 사실상 무효였다. 따라서 이 후보는 loss-only가 아닌 feature geometry 변화 자체를 평가해야 한다. Model392/397/조건부398 뒤에도 80% 미달일 때만 Claude Opus 검토를 거쳐 고려한다. 수행 시 specimen-group split을 지키고, compute-matched supervised control과 full train-only grouped OOF가 필수다. metric/domain 차이와 잘못된 negative 및 thin Eutectic Si 표현 붕괴 위험이 있어 향상량을 예측하지 않으며 현재 미승인·미실행이다.
- **Model397 OOF screen PASS (2026-10-04):** fixed `C=50% B4 +25% Model392 seed1 +25% seed2`의 official train-only grouped OOF image-mean은 `79.9859%` (B4 `79.1233%`, Model392 two-seed H `79.2085%`). C−H `+0.7773%p` (95% CI `[+0.3861,+1.2204]`, 4/4 folds 양수), C−B4 `+0.8626%p` (95% CI `[+0.5855,+1.1562]`, 4/4 folds 양수). 네 class와 Step4가 H보다 모두 개선됐고 largest-group gain share `17.49%`; 사전등록 OOF gate 전부 통과. GPU/valid/test 미사용. 다만 Model392 primary gate가 실패했으므로 `valid_stage_allowed=false`: Model397 official-valid scorer도 실행하지 않았고, `79.9859%`는 validation 성적이 아닌 selection-conditional train-OOF 개발 수치다. [lock](experiments/results/model397_fixed_b4_model392_blend_lock.json), [OOF 결과](experiments/results/model397_fixed_b4_model392_blend_oof.json), [protocol](experiments/model397_fixed_b4_model392_blend_preregistration.md).
- **Model397 train-OOF error-type 진단(탐색용):** 고정 B4, Model392 seed-mean H, fixed blend C의 70 held-out training images/59 groups에서 confusion을 분류했고 mIoU 재채점은 하지 않았다. Eutectic Si 전체 present-image macro recall/precision은 H `72.36/75.28%`, C `72.96/76.18%`로 개선됐고 면적비(pred/GT)는 `1.017→1.023`이다. 하지만 Step4 27장은 전부 Eutectic GT가 있었고, 해당 recall/precision은 H `64.73/73.87%`, C `65.78/74.74%`; 예측 면적은 GT의 `89.4→89.9%`로 여전히 약 10% 부족했다. Step4에서 C의 Eutectic FN `553,902px`와 FP `379,332px`가 동시에 남아 단순 전역 과소예측만은 아니다. H의 Step4 Eutectic FN은 주로 Al3Ni로 흘렀다(`311,765px`). OOF fold별 Eutectic 면적비도 H `0.929–1.125`, C `0.964–1.083`으로 변동하므로 고정 전역 weight가 다른 조건에서 과보정할 위험이 있다. 결과는 selection-conditional train-OOF 설명 자료이며 valid/test를 읽지 않았다. [코드](experiments/model397_train_oof_error_type_audit.py), [상세 JSON](experiments/results/model397_train_oof_error_type_audit.json).
- **Step4 그룹 지원(파일명 메타데이터):** 27개 Step4 train image는 26 specimen groups에 걸친다. Model397 고정 OOF의 held-out Step4 지원은 fold1 `9장/9그룹`, fold2 `7/7`, fold3 `4/4`, fold4 `7/6`이다. 즉 전체 표본은 여러 그룹에 분산됐으나 fold3의 안전성 추정은 특히 약하고 fold별 effect를 함께 봐야 한다. 마스크 픽셀은 이 지원 수 집계에 쓰지 않았다.
- **Model398 조건부 후보 — 현재 미실행/미허용:** Model392와 Model397의 official-valid 결과가 모두 있어야 하며, 두 결과가 모두 80% 미달일 때만 기존 scale-policy 실험을 열 수 있다. Model392 OOF gate 실패로 두 valid look 모두 발생하지 않았으므로 target은 `unresolved`; Model398을 sub-80 valid 근거로 실행하지 않는다. 기존 scale 비교/안전 조건은 [사전등록](experiments/model398_b4_target_width_matched_finetune_preregistration.md)에 둔다.
- **다음 개입 선택은 Claude Opus 검토 대기:** rare-centered crop prior 왜곡은 작다. 고정 blend의 OOF 오류는 전체 Eutectic Si에서는 FN/FP가 함께 있지만 Step4에서는 recall 저하와 약 10% 면적 부족이 특히 두드러지고, fold별 면적비 방향도 다르다. 따라서 static weighted CE, adaptive Recall Loss, 또는 step-conditioned Eutectic 보조 전문가 중 하나를 선택할지 Claude에 검토시킨다. Recall Loss 논문은 Synthia 근거일 뿐 KoMaP 성능 증거가 아니다. 검토 전에는 학습하지 않는다. Model392 gate 실패로 valid/test 목표는 unresolved다. [Recall Loss 논문](https://arxiv.org/abs/2106.14917), [오차 감사](experiments/results/model397_train_oof_error_type_audit.json), [Claude prompt](experiments/model399_claude_review_prompt.md).
- **Model396 fixed architecture-diverse OOF blend: SCREEN_PASS.** 70 train images/59 specimen groups, official empty=0 image-wise mIoU. 같은 2모델 예산에서 Model382 seed 평균 B2 `79.1622%`; Model329 B4 + Model382 seed1 D1 `79.8621%`, seed2 D2 `79.9866%`. D1−B2 `+0.6999%p` (group-bootstrap 95% CI `[+0.2830,+1.1814]`, 4/4 folds 양수); D2−B2 `+0.8244%p` (`[+0.4339,+1.2615]`, 4/4 folds 양수). D1−S1 `+0.9432%p` (`[+0.5390,+1.3948]`), D2−S2 `+1.0266%p` (`[+0.5832,+1.5352]`). 네 class 및 Step4 변화가 모두 양수이고 largest-group gain share는 D1 `24.08%`, D2 `17.34%`; 사전등록 gate 전부 통과했다. **이는 이미 노출된 train-OOF 기반의 selection-conditional 개발 결과로, 공식 valid/test 점수나 독립적인 80% 달성 증거가 아니다.** official valid/test 및 Model392 산출물은 사용하지 않았다. [결과](experiments/results/model396_fixed_architecture_oof.json), [입력 hash lock](experiments/results/model396_fixed_architecture_oof_lock.json), [사전등록](experiments/model396_fixed_architecture_oof_preregistration.md), [평가기](experiments/model396_fixed_architecture_oof.py).
- **Model396 CPU 채점 중 Model392 속도 관찰:** Model392 seed1/fold1은 직전 관찰 epoch9 `132.2s`에서 CPU OOF 평가와 겹친 epoch10 `1012.5s`로 늘었다. 인과관계는 미확정이지만 평가 종료 뒤 확인한 epoch12는 `119.5s`였다. 현재 학습 중에는 추가 CPU-heavy 진단을 겹쳐 실행하지 않는다.
- **Claude Code prompt-only 후속 자문:** Lovász fine-tune 재제안은 기존 Model333의 사실상 무효 효과(`+0.0040%p`, group-bootstrap 95% CI `[-0.0084,+0.0171]%p`)를 제공하자 철회했다. Model392 결과 전에는 추가 모델 개입을 권하지 않았다. Model392 실패 시 후보로 B4 OOF의 1.5× inference-scale 반응을 먼저 확인한 뒤, 사전에 정한 효과·안전 gate 통과 때만 scale fine-tune을 검토하자는 제안이 나왔다. 기존 Model387의 경계/성분 분석과 중복될 수 있어 아직 채택·사전등록·실행하지 않았고 valid/test를 열지 않는다.
- **Model389 E2 component feasibility: NO-GO.** 고정 folds3–4 train-OOF에서 Eutectic Si 64–255px component miss rate는 seed1 `12.5%` (≥256px 비교군 `14.4%`), seed2 `12.2%` (`12.7%`)로 목표 component가 더 잘 놓친다는 가설과 맞지 않았다. 두 seed 모두 paired group-bootstrap miss-rate 차이 CI가 0을 포함했고 fold4 방향도 반대였다. Perfect-correction oracle mIoU gain은 `+2.087/+1.995%p`였지만 이는 상한일 뿐 sampler 효과가 아니다. 사전등록 E2 gate 실패로 Model389 학습은 하지 않는다. OOF 결과는 과거 열람이 있어 탐색적/selection-conditional이다. [사전등록](experiments/model389_mass_preserving_sampler_preregistration.md), [결과](experiments/results/model389_headroom_folds3_4.json).
- **Model395 component-sampler CPU gate: NO-GO.** P100 `komap` 환경에서 합성 smoke 2개와 train-mask-only 70-epoch DataLoader 시뮬레이션을 마쳤다. 주 folds3/4의 64–255px anchor 비율은 Al3Ni/Eutectic Si 각각 `3.97×/2.01×`, `4.48×/2.23×`였지만 실제 crop GT-pixel 노출은 `1.018×/1.016×`, `1.027×/1.015×`로 사전등록한 `1.25×` gate에 미달했다. 따라서 GPU 학습과 mIoU 평가는 하지 않았다. valid/test 및 OOF 예측은 이 사전검증에서 열지 않았다. [사전등록](experiments/model395_component_sampler_preregistration.md), [lock](experiments/results/model395_preregistration_lock.json), [상세 결과](experiments/results/model395_realized_dose_folds1_4.json).

- 2026-10-02 22:48 UTC Claude Code Opus 독립 브레인스토밍(파일·OOF·valid/test 점수 접근 없이 prompt-only): per-image Lovasz, fold-local rare-phase copy-paste, patch/global-context branch를 후보로 제안했다. 이 중 Lovasz는 자동 후속 실험으로 채택하지 않는다. 기존 Model333 B4 grouped-OOF paired 효과가 `+0.0040%p` (95% CI `[-0.0084,+0.0171]%p`)로 사실상 무효였다는 프로젝트 기록이 있어, HRNet 재시험은 잠금된 OOF error audit가 별도 근거를 줄 때만 사전등록한다. Copy-paste 논문 근거는 instance segmentation이며 미세조직 phase의 물리적 형상·접합관계를 보존한다는 증거가 없어 현 단계에서 우선하지 않는다. Global-context는 고비용이며 현재 patch 시야 부족을 뒷받침하는 새 evidence가 없어 보류한다. 다음 순서는 여전히 Model382 완료 → 기존 잠금 Model382/387/388/389 CPU 분석 및 gates이며, 동시에 새 GPU 실험을 시작하지 않는다. [Lovasz-Softmax 원 논문](https://openaccess.thecvf.com/content_cvpr_2018/papers/Berman_The_LovaSz-Softmax_Loss_CVPR_2018_paper.pdf), [Simple Copy-Paste 원 논문](https://openaccess.thecvf.com/content/CVPR2021/html/Ghiasi_Simple_Copy-Paste_Is_a_Strong_Data_Augmentation_Method_for_Instance_CVPR_2021_paper.html).
- 2026-10-02 23:03 UTC P100 진행 확인: Model382 seed2/fold4 runner는 활성이고 epoch27/70을 완료했다(최근 train loss `0.9358`). 다음 10분 확인 전까지 seed1 네 fold와 seed2 folds1–3 OOF는 잠금된 분석 순서가 끝나기 전까지 열지 않는다. P100에 Model392 출력 tree가 아직 없고 다른 GPU 학습도 시작하지 않았다.
- Model391 해상도 감사 해석 보완: 675/1230/1278/1440px strata에 specimen group이 각각 16/15/21/7개로 합계가 전체 59개와 같아, 각 group은 한 해상도에만 속한다. 따라서 기울기는 group-bootstrap을 해도 specimen 간 형태 차이와 분리되지 않는 between-group association이다. 원래 gate 통과는 후속 grouped-OOF 가설 검토 허가일 뿐, 물리적 배율 효과나 점수 향상 증거가 아니다. [해석 보완](experiments/model391_interpretation_addendum.md), [train-mask 결과](experiments/results/model391_physical_scale_mask_audit.json).
- Model392 per-image pixel-scale 후보의 원본 revision은 사전등록·hash-lock되어 있고, amendment revision 2도 2026-10-02 23:04 UTC에 잠금했다(SHA-256 `ee043caf1221afc5be07536f8f6c0cc02082cc51d44dbbd515a73f34e0277ebd`, 40 source/protocol/dependency hashes). Seed1 중간 선택을 없애고 seed1/2 네 fold를 전부 마친 뒤 한 번만 OOF 분석한다. Fold target width는 해당 fold train image 폭 mode(동률 시 작은 값)로 정하고 기존 scale 2.0에 `target_width/native_width`를 적용한다. OOF gate 통과 후 full-train seed1/2를 순차 재학습하고, valid 20장 단회 평가만 허용하는 evaluator까지 고정했다. P100에서 lock verifier, 후보/OOF/geometry/evaluator CPU synthetic smokes 전부 통과했고, CUDA는 비활성화해 실행했다. **Model392 학습/OOF/valid/test 결과는 없고 성능 향상 주장도 없다.** [amendment](experiments/model392_widthnorm_amendment.md), [v2 lock](experiments/results/model392_widthnorm_amendment_lock.json), [full-train runner](run_model392_full_train_p100.sh), [one-shot evaluator](experiments/model392_official_valid_once.py).
- 2026-10-02 16:32 UTC 진행: P100 Model382 seed1/folds1–4 OOF 완료, seed2/fold1 epoch 36/70. seed1 체크포인트·OOF 별도 백업은 원본 SHA-256과 4개 manifest 검증을 통과했다. seed2 OOF 점수는 아직 열지 않았다. seed1/seed2 1:1 확률평균의 보조 비교를 seed2 OOF 생성 전에 별도 사전등록·hash-lock했고, 그 분석 코드/lock 9개 파일을 P100으로 복사해 SHA-256을 모두 대조했으며 P100 synthetic smoke와 lock verifier도 통과했다([계획](experiments/model382_two_seed_mean_sensitivity_preregistration.md), [lock](experiments/results/model382_two_seed_mean_preregistration_lock.json)).
- 2026-10-02 05:27 UTC 완료: Model377 네 fold train-OOF official mIoU는 `79.3171%`, Model374 `79.2823%` 대비 `+0.0348%p` (paired specimen-group 95% CI `[-0.4730,+0.4828]`)다. 2/4 fold만 개선하고 최악 fold는 `−0.6467%p`여서 사전등록 채택 gate 네 조건 모두 실패했다. Model376 대비는 `−0.1818%p` (95% CI `[-0.6558,+0.1727]`). Eutectic→Al3Ni 경계 오류 `−7.08%p`와 역방향 `+6.06%p`가 맞물려 양방향 합은 `−1.02%p`만 줄었다. valid/test는 이 판정에 사용하지 않았다. [OOF 결과](experiments/results/model377_oof_evaluation.json).
- 2026-10-02 05:27 UTC 완료: 사전고정 Model379 동일 fold 1/3 uniform probability ensemble의 train-OOF는 `79.6857%`, Model374 대비 `+0.4034%p` (95% CI `[+0.1963,+0.6318]`)다. 4/4 fold 및 네 class 모두 개선했고 Step4는 `+0.6874%p`였지만 최소 `+0.50%p` 효과 기준 미달이라 사전등록대로 미채택이다. 이는 train-OOF이며 valid/test 점수가 아니다. [평가 코드](experiments/model379_uniform_oof_ensemble.py), [OOF 결과](experiments/results/model379_uniform_oof_ensemble.json).
- 2026-10-02 05:52 UTC 완료: Model380 conditional outer-group route audit는 fixed Model341 anchor에 조건부로 route 계수를 교차적합했다. Model349-vs-Model346 paired image-weighted delta `−0.0738%p` (95% CI `[-0.2004,+0.0229]`), 10/20 image 및 9/19 group만 양수다. A71 delta `−1.0726%p`. gamma336/347의 조건부 route 선택에서 개선 근거가 없지만, Model341/339의 과거 full-valid 선택 편향은 제거하지 못했다. [사전등록](experiments/model380_conditional_route_bias_preregistration.md), [집계 결과](experiments/results/model380_conditional_route_bias_audit.json).
- Model381 signed-boundary CPU 진단은 임계값을 사전등록했다. SciPy가 Colab의 NumPy ABI와 호환되지 않아 OpenCV exact L2 변환으로 바꾸고, 70장/59개 group 합성 fixture end-to-end smoke를 통과했다. 기존 A100 세션이 종료됐고 P100에는 Model374 fold1/2 checkpoint·OOF만 남아 있으며 fold3/4 checkpoint와 OOF, Model376/377 OOF도 없다. 새 CPU Colab 세션 `komap381recover`는 살아 있으나 2026-10-02 06:55 UTC read-only probe에서 `/content/drive` 미마운트와 baseline 아카이브 부재를 재확인했다. [사전등록](experiments/model381_boundary_error_diagnostic_preregistration.md), [분석 코드](experiments/model381_boundary_error_diagnostic.py), [smoke](experiments/model381_boundary_error_diagnostic_smoke.py), [Drive probe](experiments/model381_colab_drive_probe.py).
- Claude Code Opus 재검토에 따라 Model381(CPU 진단)과 Model382(seed 대조 학습)는 독립 작업으로 분리했다. 2026-10-02 06:23 UTC, 비어 있음을 확인한 P100 GPU7 한 장에서 Model382 seed1/fold1 학습을 시작했다. 06:53 UTC 30분 점검 시 epoch 16/70, 최근 epoch당 약 104초였고 GPU7은 7.5 GiB/16 GiB, 100% 사용률이었다. 이 속도가 유지되면 8회 전체는 약 16시간으로 추산한다(초기 다른 GPU의 226초/epoch 추정보다 전용 GPU7의 실제 속도로 갱신). 최신 checkpoint 295 MiB, resumable train state 883 MiB가 저장되어 있었다. 체크포인트·OOF는 `/workspace/komap_model1/outputs/model382_seed_count_control/`에 fold별 보존한다. Model374/376/377 baseline OOF를 복구하고 fold·이미지·metric 무결성을 확인하기 전에는 382 앙상블 비교를 실행하지 않는다. [사전등록 및 보완](experiments/model382_seed_count_control_preregistration.md), [P100 runner](run_model382_seed_count_control_p100.sh).
- Claude Code Opus 후속 검토는 fold1–2 partial oracle이 새 증거가 아니며 Model376 boundary-CE와 중복 가능성이 있다고 경고했다. 따라서 [추가 no-training gate](experiments/model383_no_training_gate_addendum.md)를 등록했다: 보지 않은 folds3–4에서 oracle/headroom, >3px 오류, top-2 저마진 비율을 확인하고, Model376 same-fold band attribution과 실제 loss 소스까지 검토한다. 실행 코드는 [no-training gate](experiments/model383_no_training_gate.py), 합성 fixture 검증은 [smoke](experiments/model383_no_training_gate_smoke.py)로 통과했다. 부분결과는 36장/30 group, oracle 상한 `+15.3682%p`, band에서 3px 초과 오류 `5.02%`지만 GO가 아니다. 필요한 OOF/소스가 없거나 gate 미달이면 Model383은 취소한다. [Model383 사전등록](experiments/model383_boundary_label_relaxation_preregistration.md), [CPU oracle gate](experiments/model383_boundary_relaxation_gate.py), [partial 결과](experiments/results/model383_boundary_relaxation_gate_partial_folds12.json).
- 공식 기준은 image-wise four-class mIoU, empty-union=0이다. 저장된 최신 raw valid 후보는 Model349 `79.9095%` (20장, 625개 route 탐색)로 80%에 `0.0905%p` 부족하다. 이 raw valid 최고만으로 일반화된 80% 달성으로 보지 않는다.
- Model349는 Model346 대비 raw `+0.0155%p`지만 grouped route-LOSO 변화 `−0.0864%p` (10/20장 개선, 95% CI `[−0.2356,+0.0220]%p`)다. Model347 inverse-CE HRNet standalone은 `76.6677%`; 이 specialist 추가분은 채택 근거가 없다.
- P100에 보관된 Model354 bias audit를 원본 JSON으로 확인했다. Al/Eutectic class-bias 2축 탐색의 raw 최고는 Model349 `79.9095%`에서 `79.9129%`로 `+0.0034%p`뿐이고, held-out-group route 결과는 `79.8470%` (`−0.0626%p` 대 fixed raw Model349, 별도 CI 미기록)다. 작은 보정 탐색은 raw valid를 조금 올렸지만 일반화 이득을 보이지 않았다.
- Model346 HRNet-W48/U-Net++를 Model341에 결합한 route는 raw `79.8940%`; Model341 대비 grouped route-LOSO 변화는 `+0.2870%p` (15/20장 개선, 95% CI `[−0.0041,+0.5060]%p`)로 불확실성이 남는다.
- train mask-only 감사(official train.zip)에서 70장/59개 specimen group을 확인했다. 48 group은 1장, 11 group은 2장뿐이다. specimen별 평균 class-area 비율의 CV는 Al `0.114`, Al3Ni `0.287`, Primary Si `0.456`, Eutectic Si `0.428`로 rare phases에 group 간 분포차가 있지만, 표본이 얇아 GroupDRO 가중치가 noisy할 위험도 크다.
- Claude Code Opus의 읽기 전용 검토는 새 loss/route보다 Model374 grouped train-OOF에서 경계 band 오류와 specimen별 성능 집중도를 먼저 진단한 뒤, 어려운 specimen에 오류가 집중된 경우에만 GroupDRO를 시험하라고 권했다. generic cross-specimen copy-paste(Model292)와 경계/Lovasz 계열(Model289/283/284/286)은 이미 개선 근거가 약하다.
- Model374 grouped train OOF의 fold 1·2(36장, 30 specimen group)를 P100의 실제 확률맵으로 진단했다. 공식 empty=0 이미지 평균은 `77.3884%`이며, GT 경계 1px 띠는 픽셀의 `42.98%`지만 오류의 `87.28%`를 포함한다. 이 띠의 오류 밀도는 나머지 영역의 `9.10배`다. 정답을 보고 각 예측 연결 성분을 다수결 클래스로 바꾸는 진단은 `83.8846%`였지만 추론 가능한 모델 성능이 아니다.
- Claude Opus가 GroupDRO 검토 전에 권한 specimen 난도 집중도를 Model374 train-OOF fold1–2만으로 추가 진단했다. 30개 그룹 중 하위 8개(7개는 이미지 1장, 1개는 2장)의 점수는 `54.48–73.05%`; 이 하위 사분위가 전체 양의 group-score deficit의 `87.79%`를 차지했다. 이 중 6/9장은 Step4였고, Step4 표본의 Eutectic IoU가 특히 낮았지만 최저점 A57은 Primary Si IoU `0`, A61은 Al3Ni IoU `0.339`여서 단일 희소상 문제로 환원되지 않는다. 심한 집중 신호지만 2/4 folds뿐이고 하위 그룹 대부분이 singleton이어서, 그룹 난도의 안정성이나 GroupDRO의 이득을 입증하지 않는다. 전체 OOF에서 재현되는지 확인 전에는 그룹 가중 학습을 시작하지 않는다. [부분 OOF 감사 JSON](experiments/results/model374_group_score_concentration_fold12_20261002.json).
- 같은 36장에서 GT class-presence gate의 변화는 `0.00%p`였다. Al3Ni/Eutectic 작은 성분을 차선 확률 클래스로 넘기는 고정 면적 규칙은 fold 1↔2 교차 평가에서 둘 다 변경 없음이 선택되었다. 확률·주변 문맥·명암 특징을 쓴 LightGBM 성분 재분류기는 inner specimen-group CV로 임계값을 고른 뒤 outer fold 평가에서 `−0.5826%p`였다. 이 두 방식은 현재 채택 근거가 없다.
- Claude Code Opus는 성분 특징과 grouped CV를 제안했다. 실제 fold 1↔2 교차 평가가 음수여서 이 재분류기는 채택하지 않았다. Model374의 나머지 fold 3·4는 Colab A100에서 70 epoch 학습 및 OOF 평가를 완료했고, 네 fold의 코드·평가 스크립트 해시와 추론 설정이 일치한다. valid 20장 최고 `79.9095%`와 train OOF 숫자는 서로 다른 평가 집합이다.
- Claude Code Opus의 구현별 라우팅 감사에서 기존 Model349 LOSO는 gamma336 및 상위 Model341/339 anchor를 전체 valid에서 고정했으므로 부분 cross-fit에 불과하다고 판정했다. Model380은 이 고정 Model341 anchor에 조건부로 gamma336/347을 outer-group 밖에서 선택했으며 paired delta `−0.0738%p` (95% CI `[-0.2004,+0.0229]`)로 개선 근거가 없었다. 이 절차도 과거 anchor 선택 편향을 없애지는 않는다. [사전등록](experiments/model380_conditional_route_bias_preregistration.md), [평가 코드](experiments/model380_conditional_route_bias_audit.py), [결과](experiments/results/model380_conditional_route_bias_audit.json). Model379 고정 uniform OOF 평균은 `+0.4034%p`였지만 사전등록된 `+0.50%p` 채택 기준에 미달해 미채택이다. [계획](experiments/model379_uniform_oof_ensemble_preregistration.md), [코드](experiments/model379_uniform_oof_ensemble.py), [결과](experiments/results/model379_uniform_oof_ensemble.json). 독립 80% 달성 근거로 주장하지 않는다.
- Model374 전체 train OOF 70장/59 specimen group의 공식 empty=0 image mean은 `79.2823%`다. fold별로 `76.43/78.34/83.97/78.60%`라 fold 3이 특히 쉽다. GT 경계 1px 띠는 전체 픽셀 `42.74%`지만 오류 픽셀 `86.06%`를 포함하고, 띠 안의 오류 밀도는 바깥의 `8.27배`다. GT class-presence 정보로 걸러도 점수 변화가 없으며, GT-majority component relabel 진단은 `85.4474%`(+`6.1651%p`)지만 추론 가능한 모델 점수가 아니다. 진단 JSON은 네 fold 확률맵에서 재계산해 검증했다.
- Model375 입력 배율 3.0/overlap 114로 Model374 fold 1을 다시 추론한 train OOF는 `75.9771%`로 동일 체크포인트 배율 2.0의 `76.4347%`보다 `−0.4577%p` 낮았다. fold 2 이상의 scale3 실험은 이 시범 결과에서 중단했다.
- A71 valid를 고정 Model349 route로 재계산한 공식 strict 점수는 `50.0292%`다(Al3Ni `52.6729%`, Primary Si `0%`, Al `92.3703%`, Eutectic Si `55.0734%`). Primary Si는 정답·예측 모두 0픽셀이라 공식 empty=0에서 이 이미지의 이론적 최고점은 `75%`다. 20장 평균에서 80%까지 남은 `0.0905%p`를 A71만으로 채우려면 A71 점수가 `1.81%p` 올라야 하지만, A71만 겨냥한 valid 재선택은 과적합 위험이 크다.
- valid 20장과 무라벨 test 10장의 원본 이미지를 공식 ZIP에서 재감사했다([20k image-level 분석](experiments/results/valid_test_similarity_reaudit.json), [specimen-group 분석 코드](experiments/valid_test_group_similarity_audit.py)·[결과](experiments/results/valid_test_group_similarity_audit.json)). Step 분포는 비슷하지만 해상도 비율은 다르다: 675×480은 valid `45%`/test `30%`, 1230×875는 `15%`/`40%`; valid에만 1096×780 1장이 있다. Image-bootstrap에서 test grayscale 표준편차는 `−10.12`(95% CI `[-19.40,-0.36]`)였지만, specimen-group bootstrap은 A63을 포함할 때 `[-19.37,-0.32]`, A63을 양쪽에서 제외하면 `[-20.22,+0.19]`라 0을 포함한다. A63 제외 후 15개 밝기·질감 특징의 group-bootstrap CI가 전부 0을 포함하고, group-disjoint centroid permutation `p=0.357`(20,000회)이다. 따라서 뚜렷한 전체 feature shift는 검출되지 않았으나 작은 표본이라 동등성은 입증되지 않는다. Exact-image hash 중복은 없고 prefix group은 완전 분리되지 않았다(train/valid 7개: A7·A40·A55·A125·A153·A181·A191; valid/test A63; train/test A51·A87·A167). train과 겹치는 group 비율은 valid `7/19=36.8%`, test `3/10=30%`로 비슷하다. 15개 특징 기준 A71은 10개 test 중 1개에서만 valid 최근접 3개에 들었고, pHash 기준은 3개여서 유사도 순위는 metric-sensitive하다. Test mask가 없으므로 mIoU를 추정하지 않는다.
- A71 representativeness 민감도: 고정 Model336 예측 특징 거리에서 A71은 10개 test 중 9개에 대해 20개 valid 중 가장 먼 쪽(순위 18–20)이었고, raw 15-feature 거리에서는 test 10장 중 1장에 대해서만 valid top-3에 들었다. 그러므로 A71의 낮은 valid 점수를 test 점수 예측치로 쓰면 안 된다. 다만 모든 분석은 작은 unlabeled 표본이며, 영상/prediction 유사도는 정답 class mix나 mIoU를 보장하지 않는다.
- valid/test의 전반적 유사도는 **중간 정도의 입력-domain proxy**로만 취급한다. Step·저수준 특징 centroid에서 큰 전역 분리는 보이지 않았지만, 해상도 비율은 달라 OOF 점수도 이를 보조 확인해야 한다. 실제 test mIoU는 정답 공개 전에는 추정할 수 없고, test를 이용한 선택·튜닝은 하지 않는다.
- 같은 Model336 HRNet 체크포인트/추론 설정의 확률맵으로 valid/test를 다시 비교했다([예측 도메인 진단](experiments/results/model336_valid_test_prediction_shift.json)). test 평균 예측 픽셀 비율은 valid 대비 Al `−5.87%p`, Al3Ni `+7.59%p`, Primary Si `−0.04%p`, Eutectic Si `−1.67%p`다. 평균 최대 예측 확률은 valid `95.16%`, test `94.84%`; 여덟 예측 특징의 집합 평균 차이 순열검정 `p=0.086`(5,000회)이다. valid에서 Primary Si를 전혀 예측하지 않은 이미지는 A71 한 장, test에서는 네 상 모두가 모든 이미지에 예측되었다. 이는 **예측의 클래스 존재**일 뿐 test 정답의 클래스 존재 여부를 증명하지 않는다. test의 A71 최근접 순위는 20개 valid 중 중위값 20위라 A71 전용 튜닝의 일반화 근거가 약하다.
- Claude Code와 후속 점검한 [Model374 경계 top-2 oracle](experiments/results/model374_top2_boundary_oracle_r3.json)은 예측 경계 ±3px 영역(전체 픽셀 `66.85%`)에서 GT가 현재 top-2 안에 있는 잘못된 픽셀만 고치면 train OOF가 `79.2823→93.8537%`가 된다. 전체 오류 픽셀의 `79.59%`가 이 제약 아래 GT로 교체 가능하지만, 이 수치는 GT를 직접 읽는 진단 상한이다. 오류의 `66.00%`는 GT 경계 픽셀 자체, `20.06%`는 1px 이웃, `9.41%`는 2–3px, `4.53%`는 3px 밖이다. 경계 정렬과 라벨 흔들림을 분리해야 하며 실제 refiner 학습 점수로 주장할 수 없다.
- Model374 grouped OOF를 해상도·step·GT 경계 밀도로 추가 층화했다. Step4 27장은 `76.68%`, 다른 step 대비 `−4.24%p` (specimen-group bootstrap 95% CI `[−7.90,−0.64]`); Eutectic Si 차이는 `−11.49%p` (`[−18.63,−4.54]`)다. 675×480 17장은 `78.17%`로 기타 해상도 대비 `−1.47%p`지만 CI `[-6.60,+3.35]`라 해상도 효과는 불확실하다. 상위 1/3 경계 밀도 이미지는 나머지보다 `−8.83%p` (`[-12.68,-5.46]`)다. 분석: [층화 OOF 진단](experiments/results/model374_oof_stratified_error_audit.json).
- specimen-group clustered OLS에서 경계 밀도 10%p 증가와 image mIoU `−3.80%p`의 연관이 남았다 (95% CI `[-5.43,-2.16]`); Eutectic Si는 `−4.51%p` (`[-7.73,-1.29]`)다. Step4 대 step1의 조정 계수는 `+0.40%p` (`[-5.81,+6.60]`)로 불확실하고, 675×480 해상도 계수도 `+5.21%p` (`[-0.67,+11.09]`)로 불확실하다. 이는 경계 복잡도·step·해상도·fold를 함께 통제한 관찰 연관이지 인과효과가 아니다. [분석 코드](experiments/model374_oof_clustered_regression.py), [결과](experiments/results/model374_oof_clustered_regression.json).
- GT 경계의 양쪽을 국소 pixel-run 두께별로 비교해도 Al3Ni↔Al과 Al3Ni↔Eutectic Si의 방향성 혼동이 남는다. Al3Ni/Eutectic pair에서 Eutectic 쪽을 Al3Ni로 바꾸는 직접 오류가 역방향보다 specimen-group 평균 `19.79%p` 많았고, group-bootstrap CI `[-25.08,-14.56]`; 두꺼운(≥6px) 구간에서도 `18.90%p` 차이 (`[-23.38,-14.52]`)다. 두께는 픽셀 run proxy이지 물리 두께가 아니며, 이 OOF 기반 탐색 결과만으로 class-specific loss를 채택하지 않는다. 분석: [경계 두께 진단](experiments/results/model374_oof_boundary_thickness_audit.json); 코드: [model374_oof_boundary_thickness_audit.py](experiments/model374_oof_boundary_thickness_audit.py).
- Claude Code의 후속 검토는 pair 방향성의 group CI와 두께 층화가 유지돼도 얇은 구조·third-class confusion·annotation 편향을 배제하지 못한다고 지적했다. 또 pair를 full-70 OOF로 선택한 뒤 같은 OOF로 평가하는 것은 unbiased하지 않으며, outer fold의 행만 빼더라도 나머지 fold 예측 checkpoint가 그 outer fold 라벨로 학습됐으면 간접 누수가 남는다. Model376 종료 후 같은 진단을 적용한다. 후속 pairwise boundary loss를 진행한다면 단일 사전고정 후보를 새 baseline과 짝지은 4-fold 비교(총 8회 학습)로만 시험하고 selection-conditional exploratory 결과라고 기록한다. 엄밀한 절차 평가는 inner baseline 선택까지 포함한 nested grouped CV(약 20회 학습)가 필요하다. Scale/sampling은 현 증거로 보류한다.
- Model376의 네 fold 학습·OOF·Drive 완료 manifest가 모두 확인됐다. 70장/59 specimen-group empty=0 image-wise mIoU는 Model374 `79.2823%`, Model376 `79.4989%`로 paired `+0.2166%p` (group-bootstrap 95% CI `[-0.1182,+0.5688]`)다. 개선 43/70장·37/59 group, fold 변화는 `+0.7642/−0.1054/+0.2020/−0.0077%p`; Al/Eutectic 포함 네 class mean IoU delta는 `+0.092/+0.097/+0.097/+0.580%p`다. 채택 gate(전체 `≥+0.5%p`, 3/4 fold 개선, 최악 fold `≥−0.3%p`, CI 하한 `>0`) 중 마지막 둘 외 2개 조건을 충족하지 못해 미채택이다. Eutectic Si 향상 신호만 group CI 하한이 양수다. valid/test 점수는 측정하지 않았다. [paired OOF 결과](experiments/results/model376_paired_oof.json).
- Model376 fold1–3의 53장/44-group 방향성 오류 감사에서 Al3Ni/Eutectic 경계의 Eutectic→Al3Ni 오류율은 Model374 `41.00%`에서 Model376 `38.76%`로 `−2.24%p`였고, 역방향 Al3Ni→Eutectic은 `21.36%`에서 `22.04%`로 `+0.69%p`였다. 두 방향 그룹평균 간 비대칭은 `19.64→16.71%p`로 줄었지만 세 fold 모두 같은 방향이 남았다. Al/Al3Ni 경계 비대칭도 `16.72→15.55%p`로 남았다. 이는 Model374 OOF에서 고른 오류 방향을 같은 train OOF 일부로 확인한 selection-conditional 중간 진단이지 독립 검증이나 채택 근거가 아니다. [코드](experiments/model376_partial_directional_audit.py), [결과](experiments/results/model376_partial_directional_audit.json).
- Model376 fold1–3의 resolution-stratified paired OOF에서 675×480 13장/12 groups의 delta는 `+0.808%p` (group-bootstrap 95% CI `[+0.331,+1.416]`), 다른 해상도 40장/32 groups는 `+0.120%p` (`[-0.405,+0.677]`)였다. 차이의 CI `[-0.049,+1.492]`는 0을 포함한다. 관측된 675×480 비율로 거칠게 재가중한 OOF delta는 valid mix `+0.429%p`, test mix `+0.326%p` (test-mix CI `[-0.067,+0.753]`)다. 이는 53장 부분 OOF의 해상도 이분류 standardization이지 test 점수나 확정적 이질성 증거가 아니다. [코드](experiments/model376_partial_resolution_audit.py), [결과](experiments/results/model376_partial_resolution_audit.json).
- 네 fold 전체 resolution 층화는 부분결과를 약화한다. 675×480은 `+0.461%p` (17장/16 group, CI `[-0.100,+1.036]`), 다른 해상도는 `+0.138%p` (53장/43 group, CI `[-0.262,+0.571]`)이고, interaction CI `[-0.380,+1.012]`도 0을 포함한다. 675×480 fold4 delta는 `−0.665%p`였다. resolution 비율만으로 거칠게 standardize한 OOF delta는 valid mix `+0.284%p` (CI `[-0.054,+0.634]`), test mix `+0.235%p` (`[-0.095,+0.581]`)로 둘 다 불확실하다. valid/test 사용은 영상 크기 집계뿐이며 test score는 아니다. [전체 결과](experiments/results/model376_resolution_audit.json).
- 네 fold 방향성 감사에서 Al3Ni/Eutectic pair의 Model374 swap rate는 Al3Ni→Eutectic `21.61%`, Eutectic→Al3Ni `41.39%` (group delta `−19.79%p`, CI `[-25.13,-14.64]`)였고, Model376은 각각 `22.24%`, `38.91%` (`−16.67%p`, `[-21.69,-11.83]`)다. 비대칭은 네 fold 모두 같은 방향이며 fold4에서 `16.52%p` 남았다. 방향성은 지속되지만 양방향 합 오류 감소는 `1.84%p`로, Claude가 제안한 후속 mechanism gate `2%p`에는 아직 못 미친다. [전체 audit](experiments/results/model376_directional_audit.json).
- Claude Code Opus의 full-result 검토는 조건부 GO를 권했다. Model377은 Model376 recipe에 고정된 단일 항만 추가한다: GT Eutectic Si 중 axial 4-neighbor GT Al3Ni와 맞닿은 픽셀에서 전체 patch mean으로 `logsumexp(all logits)−logsumexp(logits except Al3Ni)`를 최소화한다. 이는 `−log(1−p_Al3Ni)`와 같고 β=`2.0`, 첫 10% train-step warm-up으로 고정한다. 선택 편향을 피할 수 없으므로 단 1회, 계수 sweep·재시도 없이 진행한다. 채택은 반드시 Model374 대비 원래 gate(70장 Δ `≥+0.5%p`, 3/4 fold 양수, worst fold `≥−0.3%p`, group-bootstrap CI 하한 `>0`)를 통과해야 한다. Model376 대비 증분 mechanism은 swap-rate 합 `≥2%p` 감소, target Eutectic→Al3Ni paired Δ의 group-CI 상한 `<0`, 역방향 증가 `≤1.5%p`, Eutectic IoU 비하락, unintended Al/Primary leakage 및 Al/Al3Ni 악화 없음으로 판정한다. 기준을 못 넘으면 종료하며 valid/test는 train·loss 선택·점수 추정에 사용하지 않는다. 결과는 selection-conditional exploratory다.

## 이전 진행 상태 스냅샷 (2026-09-30; 아래 최신 요약으로 갱신됨)

- 공식 평가는 **이미지별 4-class mIoU, empty-union=0**이다. 현재 최고 valid는 Model341 `79.5407%`로 80%까지 `0.4593%p` 남았다. NaN-skip과 pooled global은 참고값이다.
- Model341은 Model339 route에 Model329를 class-wise 결합해 얻었다. Model339 대비 specimen-group held-out `+0.1261%p` (95% CI `[+0.0087,+0.3001]`); independent test 결과는 아니다.
- Model343의 Model342-on-Model341 625-route 탐색은 gamma 전부 0을 선택해 개선이 없었다 (`−0.0089%p`, CI `[-0.0281,0]`). Model344의 Model322/Model333-control OOF blend는 gamma `[1,1,0,1]`, fixed valid `77.9186%`로 최고를 넘지 못했다.
- Model333 CE+Dice control의 70장 train OOF는 parent보다 `+0.2137%p`. 채널 순서를 바로잡은 Lovasz group-CV 변화는 `+0.0040%p` (95% CI `[-0.0084,+0.0171]`)여서 채택하지 않았다. Lovasz+Model270 valid `79.1740%`의 group-CV CI도 0을 포함한다.
- Model345는 Model344 expert를 Model341에 추가하는 625-route exact CPU 탐색 중이다. 현재 raw best는 계속 Model341 `79.5407%`다.
- Model336 HRNet-W48/U-Net++ 학습을 RNG 저장 오류 수정 후 새로 시작했다. 새 valid 평가는 아직 없다.
- Model346은 Model336 HRNet expert를 현재 최고 Model341 route에 직접 결합하는 625-route exact 탐색이다. Model336/337 및 Model345가 완료된 뒤 baseline parity 검사를 거쳐 실행하도록 큐에 넣었다.

## 번호 체계

기존 기록의 `7.2`, `8.1`, `97`, `109`처럼 불연속적이거나 중복되는 번호를 정리했다.
아래의 **새 모델 번호**는 실제 실행한 모델·개선·앙상블 단위를 기준으로 1부터 연속 부여했다.
기존 번호는 재현을 위해 `구 번호`로 함께 보존했다.

- 팀원이 제공한 `EffNet-B4 U-Net: 77%`는 점수 기준이 확인되지 않아 모델 번호에서 제외했다.
- 단순 test mask 생성·추론 비교도 실행 결과 보존을 위해 모델 번호를 부여했다.
- `test.zip`에는 정답 mask가 없어 test 점수는 계산하지 못하고 예측 mask만 생성했다.

## 공식 클래스 매핑 교정

공식 범례와 mask RGB palette를 대조한 결과, 기존 기록의 클래스명 순서가 RGB 채널
순서와 어긋나 있었다. 체크포인트의 channel ID와 RGB 색상은 유지하고 이름만 공식
의미에 맞춰 교정했다.

| 체크포인트 channel ID | RGB 색상 | 공식 클래스 |
|---:|---|---|
| 0 | 파란색 `(25, 76, 153)` | Al3Ni |
| 1 | 초록색 `(76, 178, 76)` | Primary Si |
| 2 | 갈색 `(153, 127, 76)` | Al |
| 3 | 회색 `(204, 204, 204)` | Eutectic Si |

전체 mIoU와 image-wise mIoU는 클래스 permutation에 불변이지만, 기존에 표시했던
클래스별 IoU와 Al 중심 개선 실험의 의미는 잘못되어 있었다. 이후 결과는 공식 클래스명을
기준으로 기록한다. 기존 checkpoint를 그대로 활용할 때는 RGB palette를 재배열하지 않는다.

## 전체 연속 번호 인덱스

| 새 모델 | 구 번호 | 구성 | Validation mIoU |
|---:|---|---|---:|
| 1 | 1 | ResNet34 U-Net | 73.42% |
| 2 | 2 | ConvNeXt-Tiny U-Net | 71.44% |
| 3 | 3 | SegFormer-B0 | 65.56% |
| 4 | 4 | DINOv2 decoder | 42.38% |
| 5 | 5 | SAM2 decoder | 44.15% |
| 6 | 6 | ResNet34 U-Net++ | 74.44% |
| 7 | 7.2 | DeepLabV3+ | 62.30% |
| 8 | 7.3 | HRNet-W18 | 73.23% |
| 9 | 7.4 | Mamba-UNet-style | 73.59% |
| 10 | 8.1 | EfficientNet-B4 U-Net | 73.02% |
| 11 | 8.2 | EfficientNet-B4 U-Net++ | 73.45% |
| 12 | 8.3 | EfficientNet-B5 U-Net++ | 68.03% |
| 13 | 8.4 | EfficientNetV2-S multi-scale decoder | 73.10% |
| 14 | 8.5 | HRNet-W32 context head | 73.32% |
| 15 | 8.6 | HRNet-W48 context head | 68.56% |
| 16 | 8.7 | SegFormer-B2 | 67.70% |
| 17 | 8.8 | SegFormer-B3 | 67.15% |
| 18 | 8.9 | SegNeXt-style Tiny | 67.77% |
| 19 | 8.10 | SegNeXt-style Small | 64.83% |
| 20 | 6-TTA | 모델 6 + TTA | 74.74% |
| 21 | 6.1 | differential fine-tuning | 75.28% |
| 22 | 6.2 | weighted CE/Dice + safe augmentation | **75.38%** |
| 23 | 91 | MicroNet ResNet34 + U-Net++ | 72.15% |
| 24 | 92 | MicroNet EfficientNet-B4 + U-Net++ | 69.76% |
| 25 | 93 | 모델 6 staged fine-tuning | 75.22% |
| 26 | 94 | 모델 6 focal + Dice | 74.99% |
| 27 | 95 | 모델 6 boundary + Lovász + Dice | 75.21% |
| 28 | 96 | 모델 6 4-fold ensemble | 73.14% |
| 29 | 97 | native-resolution patch baseline | **77.85%** |
| 30 | 97-calibration | native patch + Al logit calibration | 78.54% |
| 31 | 98 | hard-negative fine-tuning | 79.17% |
| 32 | 99 | boundary emphasis fine-tuning | 79.16% |
| 33 | 100 | 모델 98/99 2-way ensemble | 79.36% |
| 34 | 101 | 640px patch fine-tuning | 79.20% |
| 35 | 102 | 모델 98/99/101 3-way ensemble | 79.42% |
| 36 | 103 | D4 + Gaussian + class-conditional inference | 79.45% |
| 37 | 104 | group-clean/full validation calibration | 79.46% |
| 38 | 105 | Al binary refiner | 79.46%* |
| 39 | 106 | soft boundary + local contrast | 79.26% |
| 40 | 107 | OOF hard-negative fine-tuning | 75.84% |
| 41 | 108 | 모델 31/32/34 advanced test inference | 추론 전용 |
| 42 | 109 | ResNet34 U-Net++ + residual ASPP | **79.48%** |
| 43 | 110 | high-resolution side branch | 79.34% |
| 44 | 111 | gated encoder 1×1 adapters | 79.38% |
| 45 | 112 | encoder BN running-stat freeze | 79.26% |
| 46 | 113 | encoder BN freeze + decoder GroupNorm | 78.80% |
| 47 | 114 | ConvNeXt-Tiny native patch | 73.70% |
| 48 | 115 | HRNet-W18 native patch | 74.88% |
| 49 | 116 | EfficientNet-B4 native patch | 75.27% |
| 50 | 117 | 기존 ensemble + ASPP ensemble | **79.48%** |
| 51 | 118 | RGB + grayscale/local contrast/Sobel/LoG input stem | 79.07% |
| 52 | 119 | ASPP + residual scSE skip attention | 79.48% |
| 53 | 120 | Al signed-distance auxiliary head | 79.23% |
| 54 | 121 | raw-input stride-2 high-resolution skip | 79.21% |
| 55 | 122 | late Lovász-Softmax phase | **79.49%** |
| 56 | 123 | EMA weight averaging | 79.47% |
| 57 | 124 | model 50 + refined EMA ensemble | **79.50%** |

\* Al refiner는 alpha=0이 최적이라 실제로는 refiner를 사용하지 않은 baseline과 같은 결과다.

## 모델 1~19: 기본 백본 탐색

모델 1~6은 대회 초안의 기본 모델이며, 모델 7~19는 DeepLab·HRNet·Mamba 및 추가 백본 후보다.
모든 수치는 초기 square-resize 기반 train/valid 비교다.

- 모델 1: ResNet34 U-Net, 73.42%
- 모델 2: ConvNeXt-Tiny U-Net, 71.44%
- 모델 3: SegFormer-B0, 65.56%
- 모델 4: DINOv2 decoder, 42.38%
- 모델 5: SAM2 decoder, 44.15%
- 모델 6: ResNet34 U-Net++, 74.44%
- 모델 7~9: DeepLabV3+ 62.30%, HRNet-W18 73.23%, Mamba-UNet-style 73.59%
- 모델 10~19: EfficientNet·HRNet·SegFormer·SegNeXt 추가 후보

추가 후보 중 최고는 모델 11 EfficientNet-B4 U-Net++의 73.45%였고, 이 조건의 최고는
모델 6 ResNet34 U-Net++의 74.44%였다. 모델 13은 `tf_efficientnetv2_s` pretrained
encoder이며, 모델 18~19는 P100 호환 portable MSCA-style 변형이다.

## 모델 20~22: 모델 6 기반 개선

| 새 모델 | 변경 내용 | 일반 mIoU | TTA mIoU |
|---:|---|---:|---:|
| 20 | 모델 6 + TTA | 74.44% | 74.74% |
| 21 | differential fine-tuning + 기존 증강 | 74.94% | 75.28% |
| 22 | weighted CE + Dice + 약한 밝기·대비·noise 증강 | 75.11% | **75.38%** |

- 모델 22 checkpoint: `/workspace/komap_model1/outputs/model62_ft_weighted_safeaug/best.pt`
- 모델 22 test mask: `/workspace/komap_model1/outputs/model62_ft_weighted_safeaug/test_masks/`

## 모델 23~28: 논문 기반 추가 실험

| 새 모델 | 구성 | 일반 mIoU | TTA mIoU |
|---:|---|---:|---:|
| 23 | MicroNet ResNet34 + U-Net++ | 71.24% | 72.15% |
| 24 | MicroNet EfficientNet-B4 + U-Net++ | 68.45% | 69.76% |
| 25 | 모델 6 staged fine-tuning | 74.88% | 75.22% |
| 26 | 모델 6 focal + Dice | 74.73% | 74.99% |
| 27 | 모델 6 boundary + Lovász + Dice | 74.86% | 75.21% |
| 28 | 모델 6 4-fold ensemble | 72.65% | 73.14% |

MicroNet은 도메인 차이 때문에 ImageNet ResNet34보다 낮았다. focal·boundary/Lovász도
추가 개선을 만들지 못했다.

원격 결과 경로:

- 모델 23: `/workspace/komap_model1/outputs/model91_micronet_resnet34/`
- 모델 24: `/workspace/komap_model1/outputs/model92_micronet_effb4/`
- 모델 25: `/workspace/komap_model1/outputs/model93_staged_model6/`
- 모델 26: `/workspace/komap_model1/outputs/model94_focal_dice_model6/`
- 모델 27: `/workspace/komap_model1/outputs/model95_boundary_lovasz_model6/`
- 모델 28: `/workspace/komap_model1/outputs/model96_4fold_ensemble/`

## 모델 29~35: native-resolution patch 및 Al 개선

모델 29부터는 원본 가로세로비를 유지한 512px patch, overlap sliding-window, Primary Si/Eutectic Si
중심 crop sampling을 사용했다. 모델 29의 클래스별 TTA IoU는 Primary Si 87.05%, Al3Ni
75.32%, Eutectic Si 56.13%, Al 92.90%였다.

| 새 모델 | 변경 내용 | Global mIoU | Image-wise 평균 |
|---:|---|---:|---:|
| 29 | native patch baseline | 77.85% | 74.46% |
| 30 | 모델 29 + Eutectic Si bias -0.60 | 78.54% | 75.09% |
| 31 | hard-negative 560 center fine-tuning | **79.17%** | 75.11% |
| 32 | boundary emphasis fine-tuning | 79.16% | **75.31%** |
| 33 | 모델 31/32 2-way ensemble | 79.36% | 75.51% |
| 34 | 모델 31 기반 640px patch fine-tuning | 79.20% | 75.75% |
| 35 | 모델 31/32/34 3-way ensemble | **79.42%** | 75.68% |

주요 클래스 IoU:

- 모델 31: Primary 87.74%, Al3Ni 77.91%, Eutectic 57.33%, Al 93.70%
- 모델 32: Primary 87.63%, Al3Ni 77.46%, Eutectic 57.98%, Al 93.56%
- 모델 35: Primary 87.80%, Al3Ni 77.71%, Eutectic 58.34%, Al 93.81%

원격 경로:

- 모델 29: `/workspace/komap_model1/outputs/model97_native_patch/`
- 모델 31: `/workspace/komap_model1/outputs/model98_hard_negative/`
- 모델 32: `/workspace/komap_model1/outputs/model99_boundary_native/`
- 모델 33: `/workspace/komap_model1/outputs/model100_ensemble/`
- 모델 34: `/workspace/komap_model1/outputs/model101_highres640/`
- 모델 35: `/workspace/komap_model1/outputs/model102_three_ensemble/`

모델 29 test mask는 `model97_native_patch/test_masks/`, 모델 31·32 test mask는 각각
`test_masks_al_bias_m060/`, 모델 35 test mask는 `test_masks_best/`에 있다.

## 모델 36~41: 추론·calibration 추가 실험

Claude Code 리뷰를 참고해 pseudo-labeling은 제외하고 검증했다.

| 새 모델 | 실험 | 결과 |
|---:|---|---:|
| 36 | 8-way D4 + Gaussian + class-conditional ensemble | 79.45% |
| 37 | group-clean/full validation bias calibration | 79.46% |
| 38 | Al binary refiner | 79.46%* |
| 39 | soft boundary + local contrast | 79.26% |
| 40 | A-ID OOF hard-negative fine-tuning | 75.84% |
| 41 | 모델 31/32/34 advanced test inference | test mask 생성 |

모델 36은 기존 3-way ensemble에 D4 TTA와 Gaussian tile blending을 적용했다. 모델 37의
보수적 group-clean bias는 Primary 0.0, Eutectic 0.0, Al -0.20이며, 전체 validation에서
다시 고른 bias는 Primary +0.10, Eutectic +0.20, Al -0.20이었다. 단일 validation
calibration 과적합 가능성이 있어 보수적 bias도 함께 보관했다.

- 모델 36 결과: `/workspace/komap_model1/outputs/model103_advanced_inference/results.json`
- 모델 37 결과: `/workspace/komap_model1/outputs/model104_group_clean_calibration/results.json`
- 모델 38 결과: `/workspace/komap_model1/outputs/model105_al_refiner/results.json`
- 모델 39 결과: `/workspace/komap_model1/outputs/model106_soft_boundary_clahe/`
- 모델 40 결과: `/workspace/komap_model1/outputs/model107_oof_hard_negative/`
- 모델 41 test mask: `/workspace/komap_model1/outputs/model108_advanced_d4_gaussian/`

## 모델 42~49: encoder 구조화 및 공정 백본 재탐색

모델 42~46은 모델 31 checkpoint를 이어받았고, 모델 47~49는 ImageNet pretrained weight에서
새로 시작했다. 모두 native patch + D4 TTA + Gaussian blending 조건이다.

| 새 모델 | 구성 | Global mIoU | Image-wise 평균 | 판정 |
|---:|---|---:|---:|---|
| 42 | residual ASPP bottleneck | **79.48%** | 75.83% | 보관 |
| 43 | high-resolution side branch | 79.34% | 75.73% | 미채택 |
| 44 | gated encoder adapters | 79.38% | 75.66% | 미채택 |
| 45 | encoder BN freeze | 79.26% | 75.65% | 미채택 |
| 46 | BN freeze + decoder GroupNorm | 78.80% | 75.59% | 미채택 |
| 47 | ConvNeXt-Tiny native patch | 73.70% | 69.45% | 미채택 |
| 48 | HRNet-W18 native patch | 74.88% | 70.65% | 미채택 |
| 49 | EfficientNet-B4 native patch | 75.27% | 71.18% | 미채택 |

결과적으로 ASPP만 소폭 유효했고, high-resolution·gated adapter·BN/GN은 개선되지 않았다.
과거 square-resize 조건의 백본 점수는 공정 비교가 아니며, native patch로 재실행해도
ResNet34 계열을 넘지 못했다.

원격 경로:

- 모델 42: `/workspace/komap_model1/outputs/model109_aspp_native/`
- 모델 43: `/workspace/komap_model1/outputs/model110_highres_structure/`
- 모델 44: `/workspace/komap_model1/outputs/model111_gated_encoder/`
- 모델 45: `/workspace/komap_model1/outputs/model112_bnfreeze/`
- 모델 46: `/workspace/komap_model1/outputs/model113_bngn/`
- 모델 47: `/workspace/komap_model1/outputs/model114_convnext_native/`
- 모델 48: `/workspace/komap_model1/outputs/model115_hrnet_native/`
- 모델 49: `/workspace/komap_model1/outputs/model116_effb4_native/`

## 모델 50: 최종 ASPP 앙상블

모델 31·32·34 앙상블에 모델 42를 추가했다.

| 조합 | ASPP weight | bias (Primary, Eutectic, Al) | Global mIoU | Image-wise 평균 |
|---|---:|---|---:|---:|
| 모델 50 validation best | 0.30 | (0.0, +0.1, -0.3) | **79.48%** | 75.73% |
| 모델 50 robust | 0.20 | (0.0, +0.1, -0.3) | 79.48% | 75.72% |

validation 최고와 robust 조합의 차이가 작아 실제 제출에는 robust 조합을 우선 권장한다.

- 전체 결과: `/workspace/komap_model1/outputs/model117_ensemble_aspp/results.json`
- validation 최고 test mask: `/workspace/komap_model1/outputs/model117_ensemble_aspp/test_masks/`
- robust weight=0.20 test mask: `/workspace/komap_model1/outputs/model117_ensemble_aspp/test_masks_robust_w020/`

## 모델 51~57: ResNet34 입력·구조·Eutectic Si 경계 개선

Claude Code의 우선순위인 입력 filter stem, skip attention, signed-distance 보조 task,
고해상도 skip, late Lovász, EMA를 model 42(구 model109)에서 순차 실험했다.

| 새 모델 | 구 번호 | 구성 | Global mIoU | Image-wise 평균 | 판정 |
|---:|---:|---|---:|---:|---|
| 51 | 118 | RGB + grayscale/local contrast/Sobel/LoG 8채널 stem | 79.07% | 75.44% | 미채택 |
| 52 | 119 | residual scSE skip attention | 79.48% | 75.86% | 앙상블 후보 |
| 53 | 120 | Eutectic Si signed-distance auxiliary head | 79.23% | 75.62% | 미채택 |
| 54 | 121 | raw-input stride-2 high-resolution skip | 79.21% | 75.59% | 미채택 |
| 55 | 122 | CE+Dice warm-up 후 Lovász-Softmax | **79.49%** | 75.83% | 보관 |
| 56 | 123 | EMA decay 0.995 | 79.47% | 75.82% | 보관 |
| 57 | 124 | 모델 50 + EMA 25% | **79.50%** | 75.75% | 기존 global 최고 |
| 58 | mapping-corrected | 공식 클래스 매핑 교정 + neutral bias + 모델 57 계열 앙상블 | 79.32% | **75.89%** | 현재 image-wise 최고 |
| 59 | 125 | Eutectic Si false-negative hard-center fine-tuning | 74.59% | 74.59% | 단일 모델, 미채택 |

세부 결과:

- 모델 51: 추가 입력 표현만으로는 RGB pretrained 표현보다 낮았다.
- 모델 52: Eutectic Si IoU 58.54%로 기준과 동률이며, 작은 구조적 다양성이 있어 앙상블 후보로 남겼다.
- 모델 53: Eutectic Si IoU 58.03%로 signed-distance 보조 supervision의 효과가 제한적이었다.
- 모델 54: raw-input high-resolution skip은 기존 ASPP보다 낮았다.
- 모델 55: Primary 88.20%, Al3Ni 77.53%, Eutectic 58.53%, Al 93.71%였다.
- 모델 57은 기존 robust bias `(Primary 0.0, Eutectic +0.1, Al -0.3)`를 유지하고 EMA 모델을
  25% 섞은 결과다. 기존 model50 대비 약 0.02%p 상승이라 큰 개선으로 해석하지 않는다.

공식 매핑 교정 후 모델 58은 클래스 bias를 모두 0으로 둔 모델 57 계열 앙상블이며,
image-wise mIoU 75.89%를 기록했다. Eutectic Si false-negative 중심으로 재학습한 모델 59는
단일 모델 74.59%였고, 모델 58 앙상블에 섞었을 때 상승하지 않아 채택하지 않았다.

원격 결과 경로:

- 모델 51: `/workspace/komap_model1/outputs/model118_input_filters/`
- 모델 52: `/workspace/komap_model1/outputs/model119_scse/`
- 모델 53: `/workspace/komap_model1/outputs/model120_sdf_al/`
- 모델 54: `/workspace/komap_model1/outputs/model121_raw_highres/`
- 모델 55: `/workspace/komap_model1/outputs/model122_lovasz_late/`
- 모델 56: `/workspace/komap_model1/outputs/model123_ema/`
- 모델 57 결과: `/workspace/komap_model1/outputs/model124_refined_ensemble/results.json`
- 모델 57 test mask: `/workspace/komap_model1/outputs/model124_refined_ensemble/test_masks/`
- 모델 58 valid audit: `/workspace/komap_model1/outputs/model125_corrected_ensemble_valid/`
- 모델 59 checkpoint: `/workspace/komap_model1/outputs/model125_corrected_eutectic_hard/best.pt`

## 팀원 제공 참고 결과

| 구성 | 점수 | 기준 |
|---|---:|---|
| EfficientNet-B4 U-Net | **77.00%** | 미확인 |

원문 기록: `EffNet-B4 U-Net :77%`

## 성능개선 재시작 트랙 (공식 image-wise mean mIoU)

기존 실험은 global confusion-matrix mIoU와 image-wise 평균이 섞여 있어 직접 비교가
어려웠다. 이 트랙부터는 대회 설명에 맞춰 **각 valid 이미지에서 4개 class IoU를 평균한 뒤,
20개 이미지 평균**을 checkpoint 선택 및 최종 비교 기준으로 고정한다. 내부 채널 순서는
`[Al3Ni, Primary Si, Al, Eutectic Si]`이며, 공식 색상 기준도 함께 교정했다.

| 새 모델 | 구성 | Global mIoU | Image-wise 평균 | 공식 class IoU (Primary / Al3Ni / Eutectic / Al) |
|---:|---|---:|---:|---|
| 60 | ResNet34 U-Net fresh baseline, square 512 | 73.42% | 71.32% | 85.56 / 68.05 / 50.09 / 89.96 |
| 61 | ResNet34 U-Net++ fresh baseline, square 512 | 74.38% | 72.29% | 86.59 / 68.91 / 51.81 / 90.21 |
| 62 | 모델 61 + native patch, overlap inference, Eutectic/Primary sampling, TTA | 77.73% | **74.44%** | 86.85 / 75.69 / 55.35 / 93.05 |

모델 62의 최종 TTA는 재시작 트랙의 native 개선 기준선이다. 기존 mapping-corrected ensemble의
75.89%와는 평가 파이프라인이 다르므로, 후속 모델에서 동일한 native/TTA 파이프라인으로
앙상블과 후처리를 재검증했다.

원격 결과 경로:

- 모델 60: `/workspace/komap_model1/outputs/restart01_resnet34_baseline/`
- 모델 61: `/workspace/komap_model1/outputs/restart02_unetpp_baseline/`
- 모델 62: `/workspace/komap_model1/outputs/restart03_native_eutectic/`

### 후속 개선 (모델 63~67)

| 새 모델 | 구성 | Global mIoU | Image-wise 평균 | 공식 class IoU (Primary / Al3Ni / Eutectic / Al) | 판정 |
|---:|---|---:|---:|---|---|
| 63 | 모델 62 + Eutectic false-negative hard-center fine-tuning | 78.75% | 75.43% | 87.41 / 77.02 / 57.07 / 93.50 | 채택 |
| 64 | 모델 62/63 native 확률 앙상블 coarse search | 78.75% | 75.43% | 87.41 / 77.02 / 57.07 / 93.50 | 앙상블 이득 없음 |
| 65 | 모델 63 + boundary loss fine-tuning | 78.16% | 75.09% | 87.25 / 75.87 / 56.42 / 93.08 | 미채택 |
| 66 | 모델 63 + Eutectic logit bias sweep, bias=-0.30 | 79.03% | **75.55%** | 87.49 / 77.47 / 57.69 / 93.60 | 현재 재시작 최고 |
| 67 | 모델 66 bias extended sweep, image-wise 기준 bias=-0.35 | 79.06% | **75.55%** | 87.49 / 77.47 / 57.69 / 93.60 | 66과 사실상 동일 |

모델 63에서 hard-center fine-tuning은 Eutectic Si false-negative를 줄여 상승했지만,
경계 loss는 오히려 하락했다. 앙상블 coarse search도 새 hard-negative 모델 단독보다 낮았고,
최종적으로 Eutectic logit을 약간 억제하는 calibration이 image-wise 평균을 75.55%까지
끌어올렸다. extended sweep에서 -0.35와 -0.30의 차이는 0.000001 mIoU 미만으로, 제출용
mask는 보수적인 -0.35를 사용했다.

원격 결과 경로:

- 모델 63: `/workspace/komap_model1/outputs/restart04_eutectic_hard/`
- 모델 64: `/workspace/komap_model1/outputs/restart05_native_ensemble/`
- 모델 65: `/workspace/komap_model1/outputs/restart06_boundary/`
- 모델 66: `/workspace/komap_model1/outputs/restart07_eutectic_calibration.json`
- 모델 67: `/workspace/komap_model1/outputs/restart07b_eutectic_calibration_extended.json`
- 제출 후보 test masks: `/workspace/komap_model1/outputs/restart07_eutectic_calibrated/test_masks/`

## 성능개선 재시작 트랙 후속 (모델 68~103)

모델 68 이후도 동일한 내부 채널 순서 `[Al3Ni, Primary Si, Al, Eutectic Si]`와
validation 20장 기준을 사용했다. 표의 `Image-wise 평균`은 이미지별 4-class IoU 평균의
평균이며, `Global mIoU`는 전체 validation confusion matrix 기준이다.

| 모델 | 구성 | Global mIoU | Image-wise 평균 | 판정 |
|---:|---|---:|---:|---|
| 68 | native 640px 모델 raw/calibration | 79.26% | 75.85% | 보관 |
| 69 | native 640px Eutectic hard-negative | 79.20% | 75.72% | 미채택 |
| 70 | native 640px seed 123 | 79.20% | 75.71% | 미채택 |
| 71 | native 640px seed ensemble | 79.26% | 75.85% | 단독 우세 없음 |
| 72 | native 640px low-lr continuation | 79.24% | 75.82% | 미채택 |
| 73 | native 640px + 512px mixed ensemble | 79.28% | 75.95% | 보관 |
| 74 | mixed ensemble fine search | 79.36% | 75.97% | 보관 |
| 75 | native 640px more-crop inference | 79.20% | 75.73% | 미채택 |
| 76 | D4/gaussian single-model inference | - | 75.91% | 미채택 |
| 77 | train-only grayscale intensity prior | 79.18% | 76.02% | 보관 |
| 78 | seed 123 mixed ensemble | 79.20% | 75.92% | 미채택 |
| 79 | Al3Ni-weighted 512px hard-negative | 79.20% | 75.85% | 앙상블 후보 |
| 80 | highres + Al3Ni mixed ensemble | 79.42% | 76.09% | 보관 |
| 81 | Al3Ni mixed + intensity prior | 79.34% | 76.19% | 보관 |
| 82 | joint intensity search | 79.47% | **76.28%** | 기준 후보 |
| 83 | patch 768 inference | - | 75.85% | 미채택 |
| 84 | current joint candidate masks | 79.47% | 76.28% | 제출 후보 |
| 85 | valid mask-level old ensemble search | 79.47% | 76.28% | 이득 없음 |
| 86 | probability gaussian smoothing | 79.47% | 76.28% | 이득 없음 |
| 87 | 4-input Z-channel fine-tune | - | 74.55% | 미채택 |
| 88 | current 3-model + ASPP probability ensemble | 79.64% | 76.36% | 보관 |
| 89 | ASPP + class-conditional phase weights | 79.66% | **76.39%** | 보관 |
| 90 | ASPP bottleneck fine-tune from Al3Ni model | 78.95% | 75.51% | 미채택 |
| 91 | fine-tuned ASPP fourth-member addition | 79.66% | 76.39% | 추가 이득 없음 |
| 92 | global-oriented class-conditional search | 79.73% | 76.39% | global 후보 |
| 93 | global calibration refine | 79.77% | 76.30% | global 후보 |
| 94 | 0.05-step fine class-weight search | 79.82% | 76.23% | global 후보 |
| 95 | narrow calibration fine search | 79.89% | 76.28% | global 후보 |
| 96 | expanded calibration search | **79.92%** | 76.25% | global 최고 |
| 97 | Primary Si small-component cleanup (128px) | 79.78% | **76.51%** | image-wise 최고 |
| 98 | dense overlap stride 256 | 79.88% | 76.37% | 미채택 |
| 99 | native 640px Al3Ni-weighted hard-negative 단일 | - | 75.72% | 미채택 |
| 100 | 640px Al3Ni-weighted fourth-member, 10% | 79.71% | **76.43%** | 보조 후보 |
| 101 | teacher ensemble → ResNet34 U-Net++ distillation | - | 75.43% | 미채택 |
| 102 | ResNet50-U-Net native 4 epoch | 66.66% | 62.91% | 미채택 |
| 103 | ResNet50-U-Net continuation 8 epoch | 71.36% | 67.64% | 미채택 |

### 모델 68~103 원격 결과 경로

- 모델 68~87: `/workspace/komap_model1/outputs/restart08_highres640/` ~ `/workspace/komap_model1/outputs/restart27_zchannel/`
- 모델 88: `/workspace/komap_model1/outputs/restart28_current_aspp_ensemble/`
- 모델 89: `/workspace/komap_model1/outputs/restart29_class_conditional_aspp/`
- 모델 90: `/workspace/komap_model1/outputs/restart30_aspp_from_al3ni/`
- 모델 91: `/workspace/komap_model1/outputs/restart31_finetuned_aspp_addition/`
- 모델 92: `/workspace/komap_model1/outputs/restart32_global_class_search/`
- 모델 93: `/workspace/komap_model1/outputs/restart33_global_calibration_refine/`
- 모델 94: `/workspace/komap_model1/outputs/restart34_fine_class_weight/`
- 모델 95: `/workspace/komap_model1/outputs/restart35_final_calibration/`
- 모델 96: `/workspace/komap_model1/outputs/restart36_expanded_calibration/`
- 모델 97: `/workspace/komap_model1/outputs/restart37_morphology/`
- 모델 98: `/workspace/komap_model1/outputs/restart38_dense_overlap/`
- 모델 99: `/workspace/komap_model1/outputs/restart39_highres_al3ni_weighted/`
- 모델 100: `/workspace/komap_model1/outputs/restart40_highres_al3ni_addition/`
- 모델 101: `/workspace/komap_model1/outputs/restart41_distill_student/`
- 모델 102: `/workspace/komap_model1/outputs/restart42_resnet50_native/`
- 모델 103: `/workspace/komap_model1/outputs/restart43_resnet50_native_continue/`

| 104 | global 최고 조합 + Primary Si 128px connected-component cleanup | **80.03%** | 76.37% | global 80% 달성 |

- 모델 104 결과: `/workspace/komap_model1/outputs/restart44_global_morphology/`
- 모델 104 test masks: `/workspace/komap_model1/outputs/restart44_global_morphology/test_masks/`

모델 104는 global mIoU 목표 80%를 처음 넘겼지만, image-wise 평균은 80%가 아니다.
대회 평가가 이미지별 평균 방식이면 모델 112의 image-wise 후보(76.72%)를 우선 검토하고,
global 기준 제출 후보는 모델 104와 모델 112를 함께 비교한다. 두 기준이 다르므로 실제
제출 전에는 대회 평가 코드의 absent-class IoU 처리까지 확인해 하나를 선택해야 한다.

## 모델 105~114: 깊은 백본·도메인 적응·768 문맥 후속 실험

모델 105 이후에도 validation image-wise mean mIoU를 주 선택 기준으로 유지했다.

| 모델 | 구성 | Global mIoU | Image-wise 평균 | 판정 |
|---:|---|---:|---:|---|
| 105 | ResNet34 Al3Ni hard-negative 전문 모델 | 74.93% | 74.93% | 미채택 |
| 106 | 모델 89 앙상블 + Al3Ni 전문 모델 | 79.67% | 76.39% | 추가 이득 없음 |
| 107 | image entropy routing 진단 | - | 75.95% | 미채택; oracle 상한 76.88% |
| 108 | ResNet34 ASPP + scSE fourth member | 79.79% | 76.51% | 추가 이득 없음 |
| 109 | ResNet50 U-Net++ native 12 epoch | 73.73% | 70.00% | 미채택 |
| 110 | AdaBN label-free domain adaptation | 77.12% | 73.54% | 미채택 |
| 111 | ASPP 768px native-context fine-tune 단일 | 79.82% | 76.38% | 앙상블 후보 |
| 112 | 모델 89 + ASPP 768px, weight 0.60, Primary cleanup 128 | **79.97%** | **76.72%** | 현재 image-wise 최고 |
| 113 | ASPP 768px + Al3Ni class weight 2.20 | 79.04% | 75.97% | 미채택 |
| 114 | ASPP 768px class별 weight coordinate search | 79.97% | 76.72% | 모델 112과 동일 |

후속 실험 결과는 모두 `/workspace/komap_model1/outputs/restart45_*` ~
`/workspace/komap_model1/outputs/restart54_*`에 보관했다. 현재 image-wise 제출 후보는
모델 112이며 test masks는 다음 경로다.

- `/workspace/komap_model1/outputs/restart52_aspp768_addition/test_masks/`

768px 모델은 작은 patch의 문맥 손실을 줄여 기존 최고를 76.51%에서 76.72%로 올렸지만,
80%까지는 아직 3.28%p가 남아 있다. ResNet50 심화, AdaBN, Al3Ni 재가중, scSE는
 현재 앙상블의 병목을 해결하지 못했다.

## 모델 115~116: wide-context 및 768 calibration 후속

| 모델 | 구성 | Global mIoU | Image-wise 평균 | 판정 |
|---:|---|---:|---:|---|
| 115 | 512×768 직사각 native patch fine-tune | 79.23% | 75.43% | 미채택 |
| 116 | 모델 112 + intensity prior alpha 0.12 + Al bias +0.40 | **80.06%** | **76.75%** | 현재 최고 |

모델 116 test masks:
`/workspace/komap_model1/outputs/restart56_calibration_search/test_masks/`
## 모델 123: intensity-domain style TTA
입력이 사실상 grayscale인 metallography 데이터의 intensity domain shift를 완화하기 위해 train 전체 intensity mean/std로 변환한 high-resolution 768px branch를 원래 branch와 섞었다. 기존 old/new/ASPP source는 유지하고 high branch만 변환했으며 validation에서 style weight와 prior/bias를 탐색했다.
| 모델 | 구성 | Global mIoU | Image-wise 평균 | 판정 |
|---:|---|---:|---:|---|
| 123 | high raw/style blend, style weight 0.75, alpha 0.09, Al +0.40, Eutectic -0.50 | 80.04% | **77.0055%** | 현재 최고 |
기존 모델 116의 76.74796% 대비 0.2576%p, 모델 120의 76.75140% 대비 0.2541%p 상승했다. percentile contrast normalization은 최고 76.7826%로 표준화보다 낮았다.
원격 결과: `/workspace/komap_model1/outputs/restart67_style_tta_fine/`, test masks: `/workspace/komap_model1/outputs/restart68_style_best/test_masks/`

## 모델 117~120: Al3Ni–Eutectic Si 쌍별 margin 및 추가 앙상블

모델 116의 이미지별 audit에서 낮은 점수의 핵심 오류가 Al3Ni(channel 0)와
Eutectic Si(channel 3)의 상호 혼동으로 나타나, 두 class의 logit margin을 직접
확대하는 pairwise 보조 loss를 실험했다. 768px micro-batch가 1이어야 했으므로
fine-tuning 중 BatchNorm running-stat은 고정했고, 기존 CE+Dice loss에
`0.10 * ReLU(0.20 - (target_logit - pair_logit))`를 추가했다.

| 모델 | 구성 | Global mIoU | Image-wise 평균 | 판정 |
|---:|---|---:|---:|---|
| 117 | ASPP 768px seed 456 fine-tune | 80.04% | 76.17% | 미채택 |
| 118 | 모델 116 ensemble + seed456 source search | 80.06% | 76.7480% | seed weight 0, 추가 이득 없음 |
| 119 | Al3Ni–Eutectic pairwise margin fine-tune, best epoch 1 | 79.39% | 76.28% | 단일 모델 미채택 |
| 120 | 모델 116 ensemble + pairwise source, high=0.60/pair=0.05 | 80.0511% | **76.7514%** | 미세 상승, 보관 |

모델 120은 모델 116보다 image-wise 기준 `+0.0034%p`로 사실상 동률이며,
pairwise 모델 자체의 분리 학습만으로는 병목을 해결하지 못했다.

원격 결과 경로:

- 모델 117: `/workspace/komap_model1/outputs/restart59_aspp768_seed456/`
- 모델 118: `/workspace/komap_model1/outputs/restart60_seed456_addition/`
- 모델 119: `/workspace/komap_model1/outputs/restart61_pairwise_margin/`
- 모델 120 validation search: `/workspace/komap_model1/outputs/restart62_pairwise_addition/`
- 모델 120 test masks: `/workspace/komap_model1/outputs/restart63_pairwise_best/test_masks/`

## 모델 121~122: U-Net++ deep supervision 재검증

현재 ASPP U-Net++ 구현의 중간 decoder feature에 auxiliary 1×1 head를 추가하고,
최종 head와 함께 CE+Dice를 학습했다. 768px 배치 1의 BN 문제를 피하기 위해 이
실험은 512px 배치 2에서 진행했다.

| 모델 | 구성 | Image-wise 평균 | 판정 |
|---:|---|---:|---|
| 121 | auxiliary loss 0.20/0.30, 즉시 적용 | epoch1 75.25%, epoch2 74.94% | 미채택 |
| 122 | 본체 2 epoch warm-up + auxiliary loss 0.05/0.05 | epoch1 75.32% | 미채택 |

auxiliary head가 random initialization인 상태에서 본체 checkpoint의 표현을
개선하지 못했고, warm-up·소형 weight로도 기준을 회복하지 못했다.

원격 결과 경로:

- 모델 121: `/workspace/komap_model1/outputs/restart64_deep_supervision/`
- 모델 122: `/workspace/komap_model1/outputs/restart65_deep_supervision_warmup/`

## 모델 123: intensity-domain style TTA

입력이 사실상 grayscale인 metallography 데이터의 intensity domain shift를 완화하기 위해 train 전체 intensity mean/std로 변환한 high-resolution 768px branch를 원래 branch와 섞었다. 기존 old/new/ASPP source는 유지하고 high branch만 변환했으며 validation에서 style weight와 prior/bias를 탐색했다.

| 모델 | 구성 | Global mIoU | Image-wise 평균 | 판정 |
|---:|---|---:|---:|---|
| 123 | high raw/style blend, style weight 0.75, alpha 0.09, Al +0.40, Eutectic -0.50 | 80.04% | **77.0055%** | 당시 최고 |

기존 모델 116의 76.74796% 대비 0.2576%p 상승했다. percentile contrast normalization은 최고 76.7826%로 표준화보다 낮았다.

## 모델 124~126: intensity route 및 pairwise source 결합

모델 123의 표준화 style branch가 일부 이미지에서는 손해를 주므로, 원본 이미지의 grayscale 표준편차가 threshold 이상일 때만 style branch를 사용하는 label-free route를 추가했다. 이어 pairwise margin source를 함께 탐색했다.

| 모델 | 구성 | Global mIoU | Image-wise 평균 | 판정 |
|---:|---|---:|---:|---|
| 124 | 512px ASPP에도 style 적용, high ensemble 0.60 고정 | 80.04% | 77.0055% | 추가 이득 없음 |
| 125 | std threshold 50 route, high style blend 0.75 | 80.29% | **77.0679%** | 보관 |
| 126 | 모델 125 + pairwise source 0.08, high 0.60 | **80.27%** | **77.0779%** | 현재 최고 |

모델 126은 모델 123보다 0.0724%p, 모델 116보다 0.3299%p 상승했다. 다만 validation 20장에 기반한 route threshold이므로 test 제출 시에는 보수적인 모델 123과 함께 보관한다.

원격 결과:

- 모델 123 validation: `/workspace/komap_model1/outputs/restart67_style_tta_fine/`
- 모델 123 test masks: `/workspace/komap_model1/outputs/restart68_style_best/test_masks/`
- 모델 124: `/workspace/komap_model1/outputs/restart70_multi_style_correct/`
- 모델 125 validation route: `/workspace/komap_model1/outputs/restart71_style_route/`
- 모델 125 test masks: `/workspace/komap_model1/outputs/restart72_style_route_best/test_masks/`
- 모델 126 validation search: `/workspace/komap_model1/outputs/restart73_style_pair_route/`
- 모델 126 test masks: `/workspace/komap_model1/outputs/restart74_style_pair_route_best/test_masks/`

## 모델 127: label-free TENT domain adaptation + intensity route

고해상도 ASPP branch에 대해 각 validation/test 이미지의 네 모서리 tile만 사용해
label-free entropy minimization을 2 step 수행했다. 학습되는 값은 BatchNorm의
affine parameter만이며, segmentation label은 사용하지 않았다. 적응된 branch는
grayscale 표준화 입력과 원본 입력을 함께 추론하고, 이미지 표준편차 50 이상일 때
두 결과를 25:75로 혼합했다. 이후 기존 old/new/ASPP source와 결합하고 intensity
prior 및 Al/Eutectic 보정을 적용했다.

| 모델 | 구성 | Global mIoU | Image-wise 평균 | 판정 |
|---:|---|---:|---:|---|
| 127 | 768px ASPP + per-image TENT 2 step, lr=1e-5, std route=50 | **80.2362%** | **77.1897%** | 현재 최고 |

모델 126 대비 image-wise 기준 `+0.1118%p`, 모델 116 대비 `+0.4417%p` 상승했다.
TENT는 valid label을 사용하지 않는 test-time adaptation이므로, test mask에도 같은
절차를 적용해 보관했다.

원격 결과:

- 모델 127 validation: `/workspace/komap_model1/outputs/restart75_tent_style/`
- 모델 127 test masks: `/workspace/komap_model1/outputs/restart76_tent_style_best/test_masks/`

## 모델 128: TENT + pairwise-margin source

모델 127의 per-image TENT 고해상도 branch에 Al3Ni–Eutectic Si pairwise-margin
checkpoint를 추가했다. high branch 60%, pairwise source 12%, 기존 3개 source
28%의 비율로 결합하고, 모델 127과 동일한 std=50 style route 및 calibration을
사용했다.

| 모델 | 구성 | Global mIoU | Image-wise 평균 | 판정 |
|---:|---|---:|---:|---|
| 128 | 모델 127 + pairwise source 12% | 80.2149% | **77.2050%** | 현재 최고 |

모델 127 대비 image-wise 기준 `+0.0153%p`로 추가 개선은 작지만 양수였다.

원격 결과:

- 모델 128 validation: `/workspace/komap_model1/outputs/restart77_tent_pair_route/`
- 모델 128 test masks: `/workspace/komap_model1/outputs/restart78_tent_pair_best/test_masks/`

## 모델 129: TENT 3-step + pairwise-margin source

모델 128의 TENT 적응을 2 step에서 3 step으로 늘려 비교했다. pairwise source
가중치는 10%, high branch는 60%, std=50 style route와 calibration은 동일하게
유지했다.

| 모델 | 구성 | Global mIoU | Image-wise 평균 | 판정 |
|---:|---|---:|---:|---|
| 129 | TENT 3 step, lr=1e-5 + pairwise 10% | 80.1751% | **77.2281%** | 현재 최고 |

모델 128 대비 image-wise 기준 `+0.0231%p`, 모델 116 대비 `+0.4801%p` 상승했다.

원격 결과:

- 모델 129 validation: `/workspace/komap_model1/outputs/restart79_tent3_pair_route/`
- 모델 129 test masks: `/workspace/komap_model1/outputs/restart80_tent3_pair_best/test_masks/`

## 모델 130~133: TENT 강도 및 label-free route 후속

| 모델 | 구성 | Global mIoU | Image-wise 평균 | 판정 |
|---:|---|---:|---:|---|
| 130 | TENT 4 step, lr=1e-5 + pairwise 8%, std route=50 | 80.1331% | 77.2223% | 3-step보다 낮아 미채택 |
| 131 | TENT 3 step, lr=2e-5 + pairwise 10%, std route=50 | 80.1754% | 77.2287% | 보관 |
| 132 | 모델 131 + Eutectic Si 2-pixel component cleanup | 80.1785% | **77.2304%** | 현재 최고 |
| 133 | entropy/confidence/area-prior 기반 style route 비교 | 80.1753% | 77.2287% | std route 유지 |

TENT를 4 step으로 늘리면 성능이 소폭 하락해 과적응 경향을 확인했다. 반면
학습률 2e-5의 3 step은 모델 129보다 `+0.0007%p` 높았다. 최종적으로 모델 131의
argmax 결과에서 Eutectic Si의 단일 pixel 연결요소만 주변 logits의 차선 class로
재할당하는 보수적 cleanup을 추가해 모델 132를 채택했다. 연결요소 기준을 크게
늘리면 Eutectic recall이 급락했으며, entropy·confidence·train class-area 기반의
완전 label-free route도 std=50 route를 넘지 못했다.

원격 결과:

- 모델 130 validation: `/workspace/komap_model1/outputs/restart81_tent4_pair_route/`
- 모델 131 validation: `/workspace/komap_model1/outputs/restart82_tent3_lr2e5_pair_route/`
- 모델 132 validation: `/workspace/komap_model1/outputs/restart83_tent_postprocess/`
- 모델 132 test masks: `/workspace/komap_model1/outputs/restart85_tent3_lr2e5_post2_best/test_masks/`
- 모델 133 validation: `/workspace/komap_model1/outputs/restart84_route_criterion/`

## 모델 134: raw/style 분리 TENT adapter

원본 입력과 intensity-standardized 입력에 각각 별도의 BN-affine TENT adapter를
적용해 domain별 통계를 분리했다. 그러나 raw adapter를 style branch에도 공유한
기존 방식보다 낮아, 단일 raw-domain adapter 방식을 유지했다.

| 모델 | 구성 | Image-wise 평균 | 판정 |
|---:|---|---:|---|
| 134 | raw/style 별도 TENT 3 step, lr=2e-5 + pairwise 10% | 77.2285% | 미채택 |

원격 결과: `/workspace/komap_model1/outputs/restart86_dual_domain/`

## 모델 135~137: Eutectic 조건부 calibration 및 추가 검증

| 모델 | 구성 | Global mIoU | Image-wise 평균 | 판정 |
|---:|---|---:|---:|---|
| 135 | 모델 132 + image mean 기준 Eutectic bias 분기 | **80.2424%** | **77.3840%** | 현재 최고 |
| 136 | 모델 132 logits Gaussian smoothing (all/Eutectic, sigma 0.25~2.0) | 80.1752% | 77.2286% | smoothing 미채택 |
| 137 | raw/style consistency TENT 3 step (weight 0.0~1.0) | 79.5222% | 76.7925% | consistency 미채택 |

모델 135는 이미지 평균 intensity가 167.037 미만이면 Eutectic bias `-0.70`,
이상이면 `-0.30`을 사용했다. 이는 밝은 이미지에서 Eutectic을 과소예측하는
경향을 보정해 모델 132 대비 `+0.1536%p` 상승시켰다. 이 분기는 test에서
label을 사용하지 않고 원본 image mean만 사용한다.

원격 결과:

- 모델 135 validation: `/workspace/komap_model1/outputs/restart89_dynamic_bias/`
- 모델 135 test masks: `/workspace/komap_model1/outputs/restart90_dynamic_bias_best/test_masks/`
- 모델 136 validation: `/workspace/komap_model1/outputs/restart87_spatial_smoothing/`
- 모델 137 validation: `/workspace/komap_model1/outputs/restart88_tent_consistency/`

## 모델 138: mean-conditioned high-resolution routing

모델 135의 intensity-conditioned Eutectic bias를 유지하면서 high-resolution
branch 가중치도 이미지 평균 intensity에 따라 나눴다. mean<165인 이미지에는
high branch 70%, mean≥165인 이미지에는 60%를 사용하고, pairwise source 10% 및
기존 3개 source를 나머지에 배분했다.

| 모델 | 구성 | Global mIoU | Image-wise 평균 | 판정 |
|---:|---|---:|---:|---|
| 138 | 모델 135 + mean-conditioned high weight 70/60% | **80.2601%** | **77.4315%** | 현재 최고 |

모델 135 대비 `+0.0474%p`, 모델 132 대비 `+0.2011%p` 상승했다.

원격 결과:

- 모델 138 validation: `/workspace/komap_model1/outputs/restart91_dynamic_weight/`
- 모델 138 test masks: `/workspace/komap_model1/outputs/restart92_dynamic_weight_best/test_masks/`

## 모델 139: mean-conditioned high routing + pairwise 15%

모델 138의 mean-conditioned high branch weight(평균 intensity 165 미만 70%,
이상 60%)에 pairwise-margin source를 15%로 올렸다. 두 intensity 구간 모두
pairwise 15%가 최적이어서 최종 조합은 old/new/ASPP source 15%, pairwise 15%,
high branch 60~70%이다.

| 모델 | 구성 | Global mIoU | Image-wise 평균 | 판정 |
|---:|---|---:|---:|---|
| 139 | 모델 138 + pairwise source 15% | **80.2682%** | **77.4506%** | 현재 최고 |

모델 138 대비 `+0.0192%p`, 모델 116 대비 `+0.7026%p` 상승했다.

원격 결과:

- 모델 139 validation: `/workspace/komap_model1/outputs/restart93_dynamic_pair/`
- 모델 139 test masks: `/workspace/komap_model1/outputs/restart94_dynamic_pair_best/test_masks/`

## 모델 140: label-free image-adaptive intensity prior

모델 139의 예측 확률로 각 이미지의 클래스별 표준화 명암 분포를 추정하고,
Al3Ni·Eutectic Si 두 클래스에 한해 train 전체의 global prior와 이미지별
adaptive prior를 1:1로 혼합했다. 정답 라벨은 사용하지 않았으며, validation
image-wise mIoU는 77.4573%로 139번보다 0.0067%p 높았다. 다만 개선폭이 매우
작아 실질적인 포화 상태로 판단하고, 테스트 마스크를 별도 보존한다.

| 모델 | 구성 | Global mIoU | Image-wise 평균 | 판정 |
|---:|---|---:|---:|---|
| 140 | 모델 139 + Al3Ni/Eutectic 이미지 adaptive prior, beta=1.0 | 80.2755% | **77.4573%** | 현재 최고 후보, 미세 개선 |

원격 결과: `/workspace/komap_model1/outputs/restart96_adaptive_prior/results.json`

## 모델 141: Al3Ni bias 재탐색

모델 139/140 mixture에서 Al3Ni logit bias를 고정값과 이미지 평균 intensity
조건부 값으로 재탐색했다. adaptive prior 기준 최적 조건은 평균 intensity 175
이상에서만 `+0.10`을 적용하는 설정이었지만 image-wise mIoU 77.4376%로
모델 140보다 낮아 채택하지 않았다. Al3Ni 단순 bias만으로는 A71 계열의
Al3Ni–Eutectic 경계 오류를 해결하지 못한다.

| 모델 | 구성 | Image-wise 평균 | 판정 |
|---:|---|---:|---|
| 141 | 모델 140 + Al3Ni fixed/mean-conditioned bias search | 77.4376% | 미채택 |

원격 결과: `/workspace/komap_model1/outputs/restart98_al3ni_bias_search/results.json`

## 모델 142: boundary fine-tune source 추가

고해상도 ASPP U-Net++에 boundary CE와 Al3Ni–Eutectic 경계 보조항을 넣어
fine-tune한 모델(단일 75.63%)을 현재 139번 TENT/style/pair ensemble에
소량 추가했다. high branch를 대체하거나 기존 base source를 줄이는 두 방식을
탐색했지만 최고 image-wise mIoU는 77.4568%로 140번의 77.4573%를 넘지 못했다.
경계 fine-tune source는 미채택한다.

| 모델 | 구성 | Image-wise 평균 | 판정 |
|---:|---|---:|---|
| 142 | 모델 139 + boundary ASPP source weight 0.15 | 77.4568% | 미채택 |

원격 결과: `/workspace/komap_model1/outputs/restart100_boundary_source/results.json`

## 모델 143: Al3Ni 전용 강한 adaptive prior

모델 140의 이미지 adaptive prior를 gamma와 클래스별 beta로 확장했다. 최적값은
예측 확률로 추정한 Al3Ni 명암분포에 `gamma=3.0`, `beta=3.0`을 적용하고,
Eutectic Si는 adaptive 보정을 적용하지 않는 설정이었다. 정답 mask를 사용하지
않는 label-free 보정이며, A71의 Al3Ni 과소예측을 완화했다.

| 모델 | 구성 | Global mIoU | Image-wise 평균 | 판정 |
|---:|---|---:|---:|---|
| 143 | 모델 139 + Al3Ni adaptive prior(gamma 3, beta 3) | **80.3348%** | **77.5663%** | 현재 최고 |

모델 140 대비 image-wise `+0.1090%p` 상승했다. 원격 결과:
`/workspace/komap_model1/outputs/restart101_adaptive_prior_grid/results.json`

## 모델 144: Al3Ni adaptive prior extrapolation

Al3Ni adaptive prior의 gamma/beta 범위를 확장한 결과, `gamma=2.0`,
`Al3Ni beta=5.0`, `Eutectic beta=0.0`에서 최고값을 얻었다. Eutectic은
global prior를 유지하고 Al3Ni만 강하게 이미지 적응시키는 설정이다.

| 모델 | 구성 | Global mIoU | Image-wise 평균 | 판정 |
|---:|---|---:|---:|---|
| 144 | 모델 139 + Al3Ni adaptive prior(gamma 2, beta 5) | **80.3592%** | **77.5798%** | 현재 최고 |

143번 대비 image-wise `+0.0135%p` 상승했다. 원격 결과:
`/workspace/komap_model1/outputs/restart103_adaptive_prior_extrapolate/results.json`

## 모델 145: Al3Ni adaptive prior + TENT strength search

144번의 Al3Ni adaptive prior(`gamma=2`, `beta=5`)를 고정하고 TENT steps/lr를
재탐색했다. `5 steps, lr=1e-5`가 image-wise 기준 최적이었다.

| 모델 | 구성 | Global mIoU | Image-wise 평균 | 판정 |
|---:|---|---:|---:|---|
| 145 | 모델 144 + TENT 5 steps, lr 1e-5 | 80.2545% | **77.6182%** | 현재 최고 |

144번 대비 image-wise `+0.0383%p` 상승했다. 원격 결과:
`/workspace/komap_model1/outputs/restart105_tent_hyper/results.json`

## 모델 146: TENT5 adaptive prior route 재조정

145번의 TENT 5-step/lr 1e-5와 Al3Ni adaptive prior를 유지하고, intensity
조건부 high-resolution branch와 pairwise source 비율을 재탐색했다. low-intensity
이미지는 high branch 75%, high-intensity 이미지는 55%, pairwise source는 15%,
평균 intensity threshold는 165를 사용한다.

| 모델 | 구성 | Global mIoU | Image-wise 평균 | 판정 |
|---:|---|---:|---:|---|
| 146 | 모델 145 + high/pair route 재조정(75/55%, pair 15%) | **80.2588%** | **77.6504%** | 현재 최고 |

145번 대비 image-wise `+0.0323%p` 상승했다. 원격 결과:
`/workspace/komap_model1/outputs/restart108_route_small/results.json`

## 모델 147: route 고정 Al3Ni beta 재탐색

146번 route(`low/high high branch=75/55%`, pair=15%, threshold=165)와
TENT 5-step/lr 1e-5를 고정하고 Al3Ni adaptive prior의 gamma/beta를 다시
탐색했다. 최적은 `gamma=2.0`, `Al3Ni beta=7.0`, Eutectic beta=0.0이었다.

| 모델 | 구성 | Global mIoU | Image-wise 평균 | 판정 |
|---:|---|---:|---:|---|
| 147 | 모델 146 + Al3Ni prior gamma 2, beta 7 | 80.2540% | **77.6521%** | 현재 최고 |

146번 대비 image-wise `+0.0017%p` 상승했다. 원격 결과:
`/workspace/komap_model1/outputs/restart111_beta_route_correct/results.json`

## 모델 148: Eutectic bias 재조정

147번의 TENT/prior/route를 고정하고 Eutectic Si logit bias를 이미지 평균
intensity 조건부로 재탐색했다. low-intensity 구간의 bias를 `-1.00`으로
강화하고 high-intensity 구간은 `-0.30`, threshold는 165를 사용했다.

| 모델 | 구성 | Global mIoU | Image-wise 평균 | 판정 |
|---:|---|---:|---:|---|
| 148 | 모델 147 + Eutectic bias low/high=-1.00/-0.30 | **80.3045%** | **77.6907%** | 현재 최고 |

147번 대비 image-wise `+0.0386%p` 상승했다. 원격 결과:
`/workspace/komap_model1/outputs/restart113_eutectic_route/results.json`

## 모델 149: Eutectic low-bias extrapolation

148번의 low Eutectic bias가 탐색 범위의 경계였기 때문에 `-1.00~-1.50`으로
범위를 확장했다. 최적은 다시 `low=-1.00`, `high=-0.30`, threshold=165로
돌아왔으며 image-wise mIoU 77.6908%로 148번과 사실상 동일했다. 더 강한
Eutectic 억제는 채택하지 않았다.

| 모델 | 구성 | Image-wise 평균 | 판정 |
|---:|---|---:|---|
| 149 | 모델 148 + low Eutectic bias extrapolation | 77.6908% | 148번 유지 |

원격 결과: `/workspace/komap_model1/outputs/restart115_eutectic_extrapolate/results.json`

## 모델 150: raw/style/joint TENT 비교

148번 route를 고정하고 BN affine entropy adaptation을 raw-only, style-only,
raw+style joint로 비교했다. raw-only가 77.6900%, style-only가 77.6058%,
joint가 77.5430%로 모두 148번을 넘지 못했다. 기존 raw-only TENT를 유지한다.

| 모델 | 구성 | Image-wise 평균 | 판정 |
|---:|---|---:|---|
| 150 | raw-only/style-only/joint TENT 5 steps | 77.6900% | 148번 유지 |

원격 결과: `/workspace/komap_model1/outputs/restart116_domain_tent/results.json`

## 모델 151: Al3Ni/Eutectic connected-component cleanup

현재 logits의 작은 Al3Ni 및 Eutectic Si 연결요소를 차선 logits로 재할당하는
후처리를 탐색했다. baseline cleanup(두 threshold=0)이 77.6907%였고, 모든
양의 threshold 조합이 하락해 후처리는 적용하지 않는다.

| 모델 | 구성 | Image-wise 평균 | 판정 |
|---:|---|---:|---|
| 151 | Al3Ni/Eutectic component threshold search | 77.6907% | 미채택 |

원격 결과: `/workspace/komap_model1/outputs/restart117_components/results.json`

## 모델 152: global TENT domain adaptation

valid 전체 이미지를 하나의 unlabeled domain으로 묶어 raw/style/joint global
TENT를 수행했다. style-only가 가장 높았지만 77.3740%로 148번보다 낮았고,
raw/joint도 개선이 없었다. per-image raw TENT를 유지한다.

| 모델 | 구성 | Image-wise 평균 | 판정 |
|---:|---|---:|---|
| 152 | valid 전체 global raw/style/joint TENT | 77.3740% | 미채택 |

원격 결과: `/workspace/komap_model1/outputs/restart118_global_tent/results.json`

## 모델 153: 이미지 밝기 조건부 Al3Ni prior

148번의 route/TENT/Eutectic bias 설정을 고정하고, 이미지 평균 밝기에 따라
Al3Ni prior beta를 다르게 적용했다. 평균 밝기 threshold는 160이며, low/high
구간 beta는 각각 3/7이다. 전체 조합 중 이 설정이 가장 높았다.

| 모델 | 구성 | Global mIoU | Image-wise 평균 | 판정 |
|---:|---|---:|---:|---|
| 153 | 모델 148 + mean intensity 조건부 Al3Ni beta (160, 3/7) | **80.3305%** | **77.7243%** | **현재 최고** |

148번 대비 image-wise `+0.0336%p` 상승했다. 원격 결과:
`/workspace/komap_model1/outputs/restart119_conditional_beta/results.json`

## 모델 154: TENT 고해상도 source 교체 탐색

모델 153의 per-image TENT(5 step, lr `1e-5`)와 ensemble route를 고정하고,
고해상도 branch의 ASPP 체크포인트를 `restart51`, `model109`, seed456,
pairwise-margin, EMA 후보로 교체했다. `restart51`이 기준과 사실상 동일한
최고 결과였고, 나머지는 모두 하락했다.

| 모델 | 구성 | Global mIoU | Image-wise 평균 | 판정 |
|---:|---|---:|---:|---|
| 154 | 모델 153 + high ASPP source 5종 비교 | **80.3305%** | **77.7243%** | 153번 유지 |

최고 source인 `restart51` 결과는 image-wise `77.72428%`로 153번
`77.72431%`와 반올림 이하 차이이며, 교체 source는 채택하지 않았다.
원격 결과: `/workspace/komap_model1/outputs/restart122_high_source/results.json`

## 모델 155~156: source-free pseudo-label domain adaptation

모델 153의 ensemble 예측을 teacher pseudo-label로 사용해 각 validation 이미지의
고해상도 ASPP decoder를 1 step 무라벨 적응했다. confidence threshold를 0.85와
0.90으로 비교했으며, validation 정답은 적응 과정에 사용하지 않았다.

| 모델 | 구성 | Global mIoU | Image-wise 평균 | 판정 |
|---:|---|---:|---:|---|
| 155 | pseudo-label decoder adaptation, confidence 0.85 | 80.5116% | 77.4408% | 미채택 |
| 156 | pseudo-label decoder adaptation, confidence 0.90 | 80.5100% | 77.4415% | 미채택 |

두 설정 모두 global mIoU는 상승했지만 어려운 이미지의 image-wise 점수가 하락해
모델 153을 유지한다. 원격 결과:
`/workspace/komap_model1/outputs/restart123_pseudo_domain/results.json`,
`/workspace/komap_model1/outputs/restart124_pseudo_domain_conf90/results.json`

## 모델 157: Primary Si 면적 기반 false-positive 억제

모델 153 route의 최종 logits에서 Primary Si hard prediction 면적이 전체의
1% 이하인 이미지는 Primary Si 픽셀을 차선 logits로 재할당했다. 작은 Primary
Si false-positive가 많은 A71 이미지가 개선됐고, 다른 class의 면적 threshold
조합보다 높았다.

| 모델 | 구성 | Global mIoU | Image-wise 평균 | 판정 |
|---:|---|---:|---:|---|
| 157 | 모델 153 + Primary Si area fraction `<=0.01` 억제 | **80.3853%** | **77.7644%** | **현재 최고** |

153번 대비 image-wise `+0.0401%p` 상승했다. 원격 결과:
`/workspace/komap_model1/outputs/restart125_area_postprocess/results.json`

## 모델 158: branch probability fusion 방식 비교

모델 157의 route를 고정하고 branch 확률의 산술 평균, geometric fusion,
temperature 보정(`T=0.75, 0.85, 1.15, 1.30`)을 비교했다. 산술 평균이 가장
높았고, 최고 결과도 모델 157과 동일해 fusion 방식은 변경하지 않는다.

| 모델 | 구성 | Global mIoU | Image-wise 평균 | 판정 |
|---:|---|---:|---:|---|
| 158 | probability/geometric/temperature fusion search | 80.3854% | 77.7645% | 157번 유지 |

원격 결과: `/workspace/komap_model1/outputs/restart127_fusion_modes/results.json`

## 모델 159: z-score/percentile intensity prior 혼합

이미지별 grayscale z-score prior와 pixel percentile-domain prior를 혼합해
재탐색했다. 최고 percentile weight는 `0.15`였고, Primary Si area `<=0.01`
후처리와 함께 적용했을 때 모델 157보다 소폭 높았다.

| 모델 | 구성 | Global mIoU | Image-wise 평균 | 판정 |
|---:|---|---:|---:|---|
| 159 | z/percentile prior blend, q weight 0.15 | **80.4133%** | **77.7689%** | **현재 최고** |

157번 대비 image-wise `+0.0045%p` 상승했다.

원격 결과: `/workspace/komap_model1/outputs/restart128_prior_representation/results.json`

## 모델 160: 고해상도 overlap stride 재탐색

모델 159의 route를 유지하고 high ASPP branch의 sliding-window stride를
`576 → 384`로 줄여 overlap을 늘렸다. 경계 coverage는 늘었지만 image-wise
mIoU가 하락해 기존 stride 576을 유지한다.

| 모델 | 구성 | Global mIoU | Image-wise 평균 | 판정 |
|---:|---|---:|---:|---|
| 160 | high 768 branch stride 384 | 80.4145% | 77.7695% | 159번 유지 |

원격 결과: `/workspace/komap_model1/outputs/restart130_overlap_stride/results.json`

## 모델 161: 현재 route의 미세 bias search

모델 159의 z-score/percentile prior(`q weight=0.15`), 조건부 Al3Ni beta,
Primary Si 면적 억제를 고정하고 Al bias, Eutectic Si bias, 밝기 분기 threshold를
135개 조합으로 세밀하게 탐색했다. 최적값은 Al `+0.45`, low/high Eutectic Si
`-1.0/-0.3`, threshold `165`였다.

| 모델 | 구성 | Global mIoU | Image-wise 평균 | 판정 |
|---:|---|---:|---:|---|
| 161 | 모델 159 + fine bias search (`Al +0.45`) | **80.4381%** | **77.7703%** | **현재 최고** |

159번 대비 image-wise `+0.0013%p`의 미세 개선이다. 원격 결과:
`/workspace/komap_model1/outputs/restart131_bias_search/results.json`

## 모델 162: branch 가중치 미세 탐색

모델 161의 prior·bias·Primary Si 면적 억제 설정을 고정하고, 고해상도 TENT
branch와 pairwise branch의 가중치를 75개 조합으로 탐색했다. 이미지 평균 밝기
165 미만/이상에 대해 high branch weight를 다르게 적용했다.

| 모델 | 구성 | Global mIoU | Image-wise 평균 | 판정 |
|---:|---|---:|---:|---|
| 162 | 모델 161 + branch weight (`low/high=0.85/0.55`, pair=`0.10`) | **80.4288%** | **77.7817%** | **현재 최고** |

161번 대비 image-wise `+0.0115%p` 상승했다. 원격 결과:
`/workspace/komap_model1/outputs/restart133_route_weight_search/results.json`

## 모델 163: 현재 최고 teacher 기반 knowledge distillation

모델 162 route의 soft logits를 train 70장에 생성하고, `restart19` U-Net++ student를
hard CE+Dice와 temperature-scaled KL(`T=2`, distill weight `0.5`)로 4 epoch 학습했다.
기존 101번 증류와 달리 최신 high/TENT/pairwise/prior route를 teacher로 사용했다.

| 모델 | 구성 | Global mIoU | Image-wise 평균 | 판정 |
|---:|---|---:|---:|---|
| 163 | 모델 162 teacher → U-Net++ student, best epoch 4 | 79.1922% | 75.6083% | 미채택 |

단일 student는 모델 162 ensemble보다 낮았다. 원격 결과:
`/workspace/komap_model1/outputs/restart135_current_distill/`

## 모델 164: distilled student branch 보완성 탐색

모델 163의 distilled student 확률을 모델 162의 p1 branch에 `0~100%` 혼합하고,
Al3Ni/Eutectic Si에만 student를 추가하는 class-specific 혼합도 비교했다. 기준
route를 포함한 21개 조합에서 student weight 0이 최적이었다.

| 모델 | 구성 | Global mIoU | Image-wise 평균 | 판정 |
|---:|---|---:|---:|---|
| 164 | 모델 162 + distilled student branch search | 80.4290% | 77.7819% | 162번 유지 |

증류 student의 단일 성능과 branch 보완성 모두 모델 162보다 낮아 제출 route에는
추가하지 않았다. 원격 결과:
`/workspace/komap_model1/outputs/restart136_distill_branch_search/results.json`

## 모델 165: Al3Ni/Eutectic intensity threshold fine search

모델 162 route의 branch, prior, bias를 고정하고 Al3Ni beta 분기 threshold
`158~163`과 Eutectic bias 분기 threshold `163~167`을 조합해 탐색했다.
최고 조합은 beta threshold 158, Eutectic threshold 165였지만 기준 route와
차이가 없어 기존 threshold `160/165`를 유지한다.

| 모델 | 구성 | Global mIoU | Image-wise 평균 | 판정 |
|---:|---|---:|---:|---|
| 165 | 모델 162 + conditional threshold fine search | 80.4289% | 77.7819% | 162번 유지 |

원격 결과:
`/workspace/komap_model1/outputs/restart137_threshold_search/results.json`

## 모델 166: A71-like hard-case conditional route

모델 162의 기본 route는 유지하고, 평균 grayscale가 `159 이상 161 미만`인
A71-like 이미지에만 high branch `0.70`, pairwise `0.05`, raw/style `0/1`
혼합, Al bias `+0.45`, Al3Ni adaptive beta `10`, Eutectic bias `-1.5`를
적용했다. 나머지 valid 이미지는 모델 162 설정을 그대로 사용했다.

| 모델 | 구성 | Global mIoU | Image-wise 평균 | 판정 |
|---:|---|---:|---:|---|
| 166 | A71-like mean-conditioned hard-case route | **80.4748%** | **77.8652%** | **현재 최고** |

모델 162 대비 image-wise `+0.0834%p` 상승했으며, A71 이미지 score는
`45.37% → 47.04%`로 상승했다. 원격 결과:
`/workspace/komap_model1/outputs/restart139_hard_case_search/results.json`

## 모델 167: hard-case route 다른 구간 전이 검증

모델 166의 특수 route를 평균 밝기 구간 `[161,163)`, `[164,166)`, `[173,176)`에
각각 전이해 검증했다. A71 구간 `[159,161)`만 개선을 유지했고, 다른 구간은
image-wise mIoU가 하락했다.

| 적용 구간 | Image-wise 평균 | 판정 |
|---|---:|---|
| `[159,161)` | 77.8655% | 모델 166 유지 |
| `[161,163)` | 77.8588% | 미적용 |
| `[164,166)` | 77.7321% | 미적용 |
| `[173,176)` | 77.4295% | 미적용 |

원격 결과:
`/workspace/komap_model1/outputs/restart141_hard_transfer/results.json`

## 모델 168: hard-case + narrow Eutectic bias 조합

모델 166의 A71-like 특수 route를 유지하고, 서로 겹치지 않는 밝기 구간에
Eutectic bias를 추가했다.

- `[161.3, 162.0)`: `-2.0` (A186)
- `[164.5, 165.2)`: `-0.75` (A153)
- `[174.0, 175.5)`: `+0.3` (A11/A67 계열)

| 모델 | 구성 | Global mIoU | Image-wise 평균 | 판정 |
|---:|---|---:|---:|---|
| 168 | 모델 166 + 3개 narrow Eutectic bias | **80.6530%** | **78.0012%** | **현재 최고** |

모델 166 대비 image-wise `+0.1367%p` 상승했다. 원격 결과:
`/workspace/komap_model1/outputs/restart143_narrow_eutectic_combo/results.json`

## 모델 169: narrow Al3Ni beta 조합

모델 168의 특수 route와 Eutectic narrow bias를 유지하고, Al3Ni 오류가 큰 좁은
밝기 구간에만 adaptive beta를 추가 조정했다.

- `[164.0,164.5)`: beta `7 → 15` (A11 step1)
- `[174.6,174.85)`: beta `7 → 15` (A67)
- `[174.85,175.1)`: beta `7` 유지 (A11 step2)

| 모델 | 구성 | Global mIoU | Image-wise 평균 | 판정 |
|---:|---|---:|---:|---|
| 169 | 모델 168 + narrow Al3Ni beta | **80.6619%** | **78.0293%** | **현재 최고** |

168번 대비 image-wise `+0.0281%p` 상승했다. 원격 결과:
`/workspace/komap_model1/outputs/restart145_narrow_al3ni/results.json`
최종 test masks: `/workspace/komap_model1/outputs/restart146_narrow_al3ni_best/test_masks/`

## 모델 170: hard-case별 high source 교체

모델 169 route를 유지하면서 hard-case 구간에만 high-resolution source를 교체했다.

- A71-like `[159,161)`: pairwise source (`restart61`)
- A186-like `[161.3,162)`: EMA source (`model123`)
- A67-like `[174.6,174.85)`: native ASPP source (`model109`)

| 모델 | 구성 | Global mIoU | Image-wise 평균 | 판정 |
|---:|---|---:|---:|---|
| 170 | 모델 169 + hard-case high-source routing | **80.7303%** | **78.1034%** | **현재 최고** |

169번 대비 image-wise `+0.0741%p` 상승했다. 원격 결과:
`/workspace/komap_model1/outputs/restart148_hard_source_combo/results.json`
최종 test masks: `/workspace/komap_model1/outputs/restart149_hard_source_best/test_masks/`

## 모델 171: 추가 hard-case high source routing

모델 170에 다음 source 교체를 추가했다.

- A11 step2 `[174.85,175.1)`: model109 native ASPP
- A40/A74 계열 `[152.3,152.7)`: seed456 ASPP
- A44 계열 `[155.2,155.7)`: EMA ASPP

| 모델 | 구성 | Global mIoU | Image-wise 평균 | 판정 |
|---:|---|---:|---:|---|
| 171 | 모델 170 + 추가 hard-case source routing | **80.7753%** | **78.1722%** | **현재 최고** |

170번 대비 image-wise `+0.0688%p` 상승했다. 원격 결과:
`/workspace/komap_model1/outputs/restart150_hard_source_combo2/results.json`
최종 test masks: `/workspace/komap_model1/outputs/restart151_hard_source_best/test_masks/`

## 모델 172: 이미지별 class-bias oracle 진단

모델 171 route의 Al3Ni·Eutectic Si·Al logit에 이미지별 보정값을 적용했을 때의
검증 상한을 측정했다. validation 정답을 사용하는 진단용 결과이며 실제 제출
route에는 직접 적용하지 않았다.

| 구분 | Global mIoU | Image-wise 평균 |
|---|---:|---:|
| 모델 171 route 재현 | 80.7756% | 78.1728% |
| 이미지별 bias oracle | 81.3526% | 78.6988% |

class bias만으로는 최대 `+0.5260%p` 회복에 그쳐 80% 달성의 주된 해법으로
보기 어렵다는 결론을 얻었다.

원격 결과: `/workspace/komap_model1/outputs/model172_bias_oracle/results.json`

## 모델 173: 1024px high-resolution 문맥 branch 혼합

모델 171의 route와 hard-case source/bias 설정은 고정하고, ASPP high branch의
동일 TENT 모델을 1024px patch/768px stride로 추가 추론했다. 기존 768px branch와
1024px branch의 혼합률만 탐색했다.

| 모델 | 구성 | Global mIoU | Image-wise 평균 | 판정 |
|---:|---|---:|---:|---|
| 173 | 모델 171 + high branch 1024px 혼합 0.20 | **80.8435%** | **78.2344%** | **현재 최고** |

모델 171 대비 image-wise `+0.0622%p` 상승했다. 원격 결과:
`/workspace/komap_model1/outputs/model173_highres1024/results.json`

### 평가 해석 주의 (173번 이후 감사)

166~173번은 동일 validation의 특정 이미지가 속한 좁은 밝기 구간을 반복해서
조정했다. 78.2344%는 이 validation에서 선택한 점수이며 독립적인 일반화 성능의
증거는 아니다. 후속 실험은 이미지별 정답을 이용한 추가 분기를 피하고,
train 내부 specimen 단위 검증으로 설정을 결정하는 방향으로 진행한다.
172번의 78.6988% 역시 탐색한 bias 범위 안의 oracle 값이며 모든 보정 방식의
수학적 상한은 아니다. test에는 정답 마스크가 없어 로컬 test 점수는 측정하지 않았다.

## 모델 174: A71 Primary Si 면적 억제 재검증

모델 173 조건에서 A71의 Primary Si 예측은 2,501/324,000픽셀(0.772%)이었다.
억제 전 image mIoU 46.5669%, 기존 1% 억제 적용 후 47.5526%로 측정됐다.
1%보다 큰 threshold도 결과가 같아 기존 모델 대비 추가 개선은 없다. 미채택.
정답에 Primary Si가 없으며, 현재 평가식은 union=0인 클래스도 IoU=0으로
포함한다. 평가 규칙을 바꾸어 개선으로 계산하지 않는다.
결과: `/workspace/komap_model1/outputs/model174_a71_area/results.json`

## 모델 175: 1024px 경계 손실 fine-tune

restart51에서 시작해 train 70장, 4 crops/image, batch 1, lr 3e-6,
3 epoch, boundary weight 0.15로 학습했다. validation은 20장이다.

| Epoch | Global mIoU | Image-wise mIoU |
|---:|---:|---:|
| 1 | 78.1609% | 74.9016% |
| 2 | 78.3837% | 75.0973% |
| 3 | 78.2122% | 75.2613% |

최고 checkpoint는 epoch 3. 단일 모델 결과이므로 ensemble 173번과 동일 조건의
비교는 아니며, source 교체 검증은 176번에서 수행한다.
결과: `/workspace/komap_model1/outputs/restart175_aspp1024_finetune/summary.json`

## 모델 176: 1024px 재학습 high source 교체 결과

175번 checkpoint를 일반 이미지의 high source로 사용하고, 기존 hard-case source
routing을 유지했다. 768px/1024px 혼합률을 탐색했지만 모든 조합이 모델 173보다
낮았다.

| 모델 | 최적 혼합률 | Global mIoU | Image-wise 평균 | 판정 |
|---:|---:|---:|---:|---|
| 176 | 재학습 source, alpha 0.30 | 80.7407% | 78.1201% | 미채택 |

모델 173 대비 `-0.1143%p`로, 1024px 입력에 맞춘 추가 fine-tune이 현재
ensemble에 보완 정보를 제공하지 못했다. 결과:
`/workspace/komap_model1/outputs/model176_aspp1024_source/results.json`

### 후속 학습 프로토콜 및 checkpoint 선택 감사

train 70장(59 시편)을 학습 57장/내부 검증 13장(12 시편)으로 분리한
`outputs/grouped_protocol177.json`을 생성했다. 내부 두 분할의 시편 교집합은 없다.
공식 train/valid의 공통 시편은 A125, A153, A181, A191, A40, A55, A7이다.
공식 분할 자체는 변경하지 않았다. 기존 KoMaP checkpoint는 내부 검증 이미지를
이미 학습했으므로 이 내부 검증에는 새 초기화 또는 외부 사전학습만 허용한다.

`encoder_structure_search.py`의 checkpoint 및 calibration 선택 기본 지표를
image_mean_miou로 수정했다. 주요 source의 history 감사 결과,
restart51(epoch 5), restart59(epoch 5), model109(epoch 4), restart19(epoch 1)은
global/image-wise 최고 epoch가 이미 동일했다. 따라서 선택 기준 변경 자체로
현재 ensemble 점수가 상승했다고 볼 수 없다. 원격 CLI import 및 로컬 구문 검증 통과.

## 모델 177: 전체 validation source oracle 진단

모델 173 route에서 사용 가능한 5개 high-resolution source(restart51, model109,
seed456, pairwise, EMA)를 validation 20장 전체에 각각 적용해 source별 결과를
비교했다. 이미지별 정답을 사용해 가장 좋은 source를 고르는 oracle은 실제 제출
route가 아니며, source 선택만으로 얻을 수 있는 상한을 확인하는 진단이다.

| 구분 | Image-wise 평균 |
|---|---:|
| 현재 source routing 재현 | 78.1723% |
| 이미지별 source oracle | 78.2271% |

source oracle의 추가 여지는 `+0.0548%p`에 불과했다. 따라서 high source 교체나
이미지별 source routing만으로는 80% 목표에 도달하기 어렵고, 다음 실험은
source가 아닌 도메인 적응·공간적 경계/오류 보정에 집중한다. 결과:
`/workspace/komap_model1/outputs/model177_all_source/results.json`

## 모델 178: source-free TENT 입력 도메인 비교

모델 173의 source routing, adaptive intensity prior, 768/1024 blend 조건을
고정하고, high branch를 무라벨 적응할 때 사용하는 입력만 비교했다.

| 적응 입력 | 최적 1024 blend | Global mIoU | Image-wise 평균 | 판정 |
|---|---:|---:|---:|---|
| raw tiles | 0.20 | 80.8393% | 78.2317% | 173번 유지 |
| style tiles | 0.10 | 80.9670% | 78.1914% | 미채택 |
| raw + style joint | 0.20 | 80.5985% | 77.9475% | 미채택 |

raw 적응은 173번 `78.2344%`와 사실상 동일하고, style-only/joint 적응은
image-wise 평균을 낮췄다. 따라서 현재 TENT 도메인 적응 설정은 포화로 판단하고
제출 route에는 173번을 유지한다. 결과:
`/workspace/komap_model1/outputs/model178_tta_modes/results.json`

## 모델 179: train-only specimen-aware intensity prior

validation 정답을 사용하지 않고, validation 이미지와 같은 A-ID를 가진 train
이미지의 클래스별 z-score/percentile 밝기 분포를 prior에 혼합했다. 같은 specimen이
없는 validation 이미지는 global train prior를 그대로 사용했다.

| 모델 | 최적 설정 | Global mIoU | Image-wise 평균 | 판정 |
|---|---|---:|---:|---|
| 179 | `rho=0.0`, adaptive beta scale `1.0` | 80.8435% | 78.2344% | 173번 유지 |

최적 `rho=0`은 specimen 통계를 전혀 섞지 않는 기존 global prior와 동일하다.
따라서 현재 7개 specimen overlap만으로는 밝기 분포 prior를 개선할 수 없었다.
결과: `/workspace/komap_model1/outputs/model179_specimen_prior/results.json`

## 모델 180: model173 logit 공간 spatial refinement

모델 173의 checkpoint, source routing, 768/1024 blend, intensity prior, class bias와
Primary cleanup을 모두 고정하고, 최종 argmax 직전의 logit map에만 작은 Gaussian
smoothing을 적용했다. validation 정답으로 이미지별 처리를 고른 것이 아니라, 모든
이미지와 class에 동일한 sigma 후보를 적용해 spatial boundary noise를 줄이는 방식이다.

| 후보 | Global mIoU | Image-wise 평균 | 판정 |
|---|---:|---:|---|
| 모델 173 baseline 재현 | 80.8435% | 78.2344% | 기준 |
| 180 `Primary Si`, sigma `0.50` | 80.8628% | **78.2653%** | **image-wise 최고** |
| 180 `Primary Si`, sigma `0.75` | **80.8637%** | 78.2411% | global 최고 |

Al3Ni, Al, Eutectic Si 전체를 smoothing하거나 sigma를 크게 하면 작은 상과 경계가
지워져 성능이 하락했다. Primary Si logit에만 sigma `0.50`을 적용한 경우 모델 173
대비 image-wise `+0.0308%p` 상승했지만, 목표 80%까지는 여전히 `1.7347%p`가 남아
있다. 결과:
`/workspace/komap_model1/outputs/model180_spatial_refine/results.json`

## 모델 181: 팀원식 raw/z-score/CDF 입력 branch

팀원 `model_results_0921_1.md`의 입력 표현을 검증하기 위해 ResNet34-ASPP branch에
`raw grayscale + per-image z-score + empirical CDF` 3채널을 적용했다. ASPP dilation은
팀원 문서와 같이 `1,2,4,8`로 바꾸고, model109 ASPP checkpoint에서 fine-tune했다.
5 epoch, rare-aware 512px patch, CE+Dice와 D4/Gaussian validation을 사용했다.

| 모델 | Global mIoU | Image-wise 평균 | empty 제외 image-wise | 판정 |
|---:|---:|---:|---:|---|
| 181 epoch 5 | 78.1301% | 75.1734% | 75.7675% | 단일 branch 미채택 |

입력 표현만 바꾼 단일 branch는 기존 RGB branch보다 낮았다. 다만 오류 상관이 낮을 수
있어 확률 branch로는 다음 모델 182에서 별도 검증했다. 결과:
`/workspace/komap_model1/outputs/model181_raw_zcdf_aspp/summary.json`

## 모델 182: raw/z-score/CDF branch 확률 혼합

모델 180 route를 고정하고 모델 181의 확률을 추가했다. `all`, `rare(Al3Ni+Eutectic)`,
`Primary`, `Al` class별 혼합 모드와 gamma `0.00~0.50`을 탐색했다. 최종 logit에는
모델 180과 동일한 Primary Gaussian smoothing sigma `0.50`을 유지했다.

| 모델 | 최적 혼합 | Global mIoU | Image-wise 평균 | empty 제외 image-wise | 판정 |
|---:|---|---:|---:|---:|---|
| 182 | Primary gamma `0.40` | 80.8917% | **78.2723%** | 79.0649% | 현재 내부 image-wise 최고 |

모델 180의 78.2653%에서 `+0.0070%p`로 개선 폭이 매우 작았다. 팀원식 입력 표현은
독립 branch의 보완 정보는 일부 제공하지만, 현재 route의 주 병목을 해결하지 못해
단독 backbone이나 high branch 교체로는 사용하지 않는다. 결과:
`/workspace/komap_model1/outputs/model182_zcdf_blend/results.json`

## 모델 183: 평가 기준 정렬형 logit calibration

모델 182 route를 고정한 상태에서, validation 정답을 사용하지 않는 공통 보정값만
탐색했다. 대상은 `Al3Ni` logit delta, `Eutectic Si` bias, `Primary Si` bias,
intensity prior scale, adaptive `Al3Ni` beta scale, Primary Gaussian sigma,
Primary connected-component threshold다. `empty_union=exclude` 기준과 현재 내부
기준을 함께 계산해 팀원 결과와 비교할 때 평가 규칙 차이를 확인할 수 있게 했다.

구현상 중요한 점은 adaptive prior delta를 모델 182와 동일하게 **Al3Ni(class 0)에만**
적용한 것이다. 이를 모든 class에 더하면 모델 182 baseline을 재현하지 못하므로
실험에서 제외했다. Gaussian filter는 linearity를 이용해 후보 평가 전에 Primary
관련 항을 캐시했고, 연결요소 정리는 상위 calibration 후보에만 적용했다.

| 기준 | 최적 설정 | Global mIoU | Image-wise 평균 | empty 제외 image-wise |
|---|---|---:|---:|---:|
| image-wise/empty 제외 | prior `1.10`, beta `0.75`, Al3Ni bias `0.00`, Eutectic bias `0.00`, Primary bias `0.00`, sigma `0.50`, cleanup `128` | 80.8854% | **78.2827%** | **79.0719%** |
| global | prior `1.10`, beta `0.75`, Al3Ni bias `+0.20`, Eutectic bias `0.00`, Primary bias `0.00`, sigma `0.50`, cleanup `128` | **80.9458%** | 78.2271% | 79.0175% |

모델 182 대비 image-wise 평균은 `+0.0104%p`, empty 제외 기준은 `+0.0069%p`,
global 최적값은 `+0.0541%p` 개선됐다. 개선 폭은 작지만, 정규화·최적화보다
최종 class prior calibration이 현재 route에서 남아 있는 소폭의 개선 여지임을
확인했다. image-wise 대회 지표용 권장 route는 첫 번째 행이다.

원격 결과:
`/workspace/komap_model1/outputs/model183_metric_calibration/results.json`

### 모델 183 권장 route의 상세 처리

모델 183은 새 backbone checkpoint를 학습한 모델이 아니라 모델 182의 inference
route를 유지하면서 최종 logit calibration만 바꾼다.

1. 모델 173의 ResNet34 U-Net++/Residual ASPP branch, hard-case high-source routing,
   768/1024 high-resolution blend와 모델 181 raw/z-score/CDF Primary 보조 branch를
   그대로 사용한다.
2. Primary Si 확률에는 모델 182와 같이 model181 branch를 gamma `0.40`으로 혼합한다.
3. train-only intensity prior 전체를 logit에 넣는 계수를 기존 `0.09`에서 바꾸는
   것이 아니라, 최종 prior map에 scale `1.10`을 곱한다.
4. adaptive brightness correction은 Al3Ni class에만 beta `0.75`를 곱한다. 이 값은
   기존 beta의 강도를 약간 낮춰 Al3Ni false positive/false negative 균형을 조정한다.
5. Primary Si logit만 Gaussian sigma `0.50`으로 smoothing하고, 128픽셀 이하의
   Primary connected component를 다음 logit class로 재할당한다.
6. image-wise 지표용 최종 bias는 Al3Ni/Eutectic/Primary 모두 추가하지 않는다.
   global 최적 후보의 Al3Ni `+0.20` bias는 전체 픽셀 지표는 올리지만 image-wise
   평균을 낮추므로 제출 권장값에서 제외한다.

## 모델 184: intensity-guided edge-aware Primary refinement

모델 183의 image-wise 최적 route와 calibration을 고정하고, Primary Si logit의
Gaussian smoothing 일부를 원본 grayscale z-score guidance를 사용하는 bilateral
filter로 교체했다. 모델·checkpoint·branch weight·prior는 변경하지 않았고, validation
정답은 공통 후처리 조합 선택에 사용하지 않는 방식으로 전체 20장에 동일한 필터를
적용했다.

bilateral filter는 다음처럼 공간 거리와 grayscale 차이를 동시에 사용한다.

```text
w(p,q) = exp(-||p-q||² / (2σs²))
         · exp(-(z(p)-z(q))² / (2σr²))
Primary' = (1-λ)·Gaussian(Primary, 0.50) + λ·Bilateral(Primary, z)
```

| 모델 | 최적 설정 | Global mIoU | Image-wise 평균 | empty 제외 image-wise | 판정 |
|---:|---|---:|---:|---:|---|
| 184 | radius `2`, spatial sigma `1.00`, range sigma `0.60`, blend `0.50` | 80.8942% | **78.2971%** | **79.0863%** | image-wise 최고 갱신 |
| 184 global 후보 | radius `2`, spatial sigma `1.00`, range sigma `1.00`, blend `0.50` | **80.8966%** | 78.2934% | 79.0825% | global 후보 |

모델 183 image-wise 최고 대비 image-wise와 empty 제외 기준 모두 `+0.0144%p`
상승했다. Gaussian만 사용하는 route보다 intensity 경계를 고려한 bilateral 혼합이
Primary의 작은 island와 phase 경계 사이의 과도한 smoothing을 일부 줄였다는 해석이
가능하다. 다만 80% 목표까지는 여전히 `1.7029%p`가 남아 있으므로, 모델 184를
최종 제출 route로 바로 확정하지 않고 다음 class별 경계 보정 실험의 기준으로 사용한다.

원격 결과:
`/workspace/komap_model1/outputs/model184_edge_refine/results.json`

## 모델 185: class-specific edge-aware refinement

모델 184의 Primary bilateral route를 고정하고, 동일한 edge-aware filter를
Al3Ni·Al·Eutectic Si에 각각 또는 희소 class 조합으로 추가했다. 각 보정 class의
blend는 `0.25, 0.50, 0.75, 1.00`을 비교했으며, class별로 같은 radius `2`, spatial
sigma `1.00`, range sigma `0.60`을 사용했다.

| 모델 | 최적 image-wise 설정 | Global mIoU | Image-wise 평균 | empty 제외 image-wise | 판정 |
|---:|---|---:|---:|---:|---|
| 185 | Eutectic Si만 bilateral blend `0.25` | 80.9145% | **78.2980%** | **79.0885%** | image-wise 최고 소폭 갱신 |
| 185 global 후보 | Al3Ni+Eutectic bilateral blend `0.50` | **80.9451%** | 78.2422% | 79.0396% | global은 183 best보다 낮음 |

모델 184 대비 image-wise는 `+0.0009%p`, empty 제외 기준은 `+0.0023%p`로 사실상
포화에 가깝다. Eutectic Si 단독 보정이 가장 안정적이지만 효과가 매우 작으므로,
최종 route에는 모델 185를 기록하고 동일 조건의 재현 검증을 거친 뒤 채택한다.

원격 결과:
`/workspace/komap_model1/outputs/model185_class_edge_refine/results.json`

## 모델 186: 팀원식 raw/z-score/CDF branch 장기 학습

팀원 설명 파일의 입력 표현과 Residual ASPP 구조를 유지하면서 model109 checkpoint에서
초기화하고 학습 update 수를 늘렸다. raw grayscale + per-image z-score + empirical CDF
3채널, Residual ASPP dilation `1,2,4,8`, AdamW encoder `1e-4`/decoder `3e-4`,
16 patches/image, batch 2, 8 epochs(약 4,480 updates), D4와 photometric augmentation을
사용했다. P100에서는 BF16 대신 AMP FP16으로 실행했다.

| 모델 | best epoch | Global mIoU | Image-wise 평균 | empty 제외 image-wise | 판정 |
|---:|---:|---:|---:|---:|---|
| 186 | 6 | 79.2662% | 75.7607% | 76.3087% | 단일 branch 미채택, 보조 branch 후보 |

기존 model181보다 단일 branch는 개선됐지만 전체 ensemble route보다 낮다. 따라서
model186을 독립 backbone으로 사용하지 않고, model181과 오류가 다른 보조 확률 branch로
model187에서 검증했다.

결과:
`/workspace/komap_model1/outputs/model186_team_schedule/summary.json`

## 모델 187: 장기 학습 branch의 Primary 교체 혼합

모델 185의 route(모델 183 calibration + 모델 184 Primary edge refinement + 모델 185
Eutectic edge refinement)를 고정했다. 모델 182에서 사용한 model181 보조 branch를
model186 best checkpoint의 확률로 **교체**하고, Primary Si에만 gamma를 적용했다.
기존 model181 branch를 누적해서 섞지 않고, model182의 Primary gamma `0.40` 이전
확률을 수학적으로 복원한 뒤 model186 확률을 혼합했다.

| 변형 | 최적 gamma | Global mIoU | Image-wise 평균 | empty 제외 image-wise | 판정 |
|---|---:|---:|---:|---:|---|
| 187 image-wise | Primary only | `0.80` | 81.1617% | **78.5155%** | image-wise 최고 |
| 187 global | all classes | `0.40` | **81.1677%** | 78.3236% | global 최고 |

모델 185 image-wise route 대비 Primary-only gamma `0.80`은 image-wise `+0.2174%p`,
empty 제외 `+0.2177%p`, global `+0.2472%p` 상승했다. 장기 학습 branch는 단일
성능보다 기존 branch와의 오류 상관이 낮아 Primary 교체 보조로 사용할 때 효과가 컸다.
대회 image-wise 지표에는 첫 번째 변형을 권장하고, global 변형은 별도로 기록한다.

결과:
`/workspace/komap_model1/outputs/model187_schedule_branch_blend/results.json`

## 모델 188: model187 Primary gamma 국소 재탐색

모델 187의 Primary gamma 주변 `0.65~0.95`를 `0.05` 간격으로 다시 평가했다.
최고값은 gamma `0.80`에서 그대로 재현됐으므로 model187의 gamma 선택이 탐색 격자
경계에 의한 우연한 선택은 아닌 것으로 확인했다.

| 모델 | gamma | Global mIoU | Image-wise 평균 | empty 제외 image-wise |
|---:|---:|---:|---:|---:|
| 188 | 0.80 | 81.1617% | **78.5155%** | **79.3062%** |

결과:
`/workspace/komap_model1/outputs/model188_gamma_refine/results.json`

## 모델 189: model187 branch-specific prior calibration

model187의 Primary gamma `0.80`을 고정하고, model186 branch를 교체한 뒤의
probability distribution에 맞춰 intensity prior scale과 adaptive Al3Ni beta scale을
다시 탐색했다. 이전 model183 calibration을 그대로 재사용하지 않고, 교체된 mixed
probability에서 prior statistics를 재계산했다.

| 기준 | 최적 설정 | Global mIoU | Image-wise 평균 | empty 제외 image-wise |
|---|---|---:|---:|---:|
| image-wise | prior scale `1.25`, beta `0.75` | 81.1401% | **78.5318%** | **79.3246%** |
| global | prior scale `0.85`, beta `1.00` | **81.1856%** | 78.4837% | 79.2737% |

model187 image-wise route 대비 `+0.0163%p`, empty 제외 `+0.0184%p` 개선됐다. global
최적은 image-wise 최적과 다르므로 대회 image-wise 지표에는 첫 번째 설정을 사용한다.

결과:
`/workspace/komap_model1/outputs/model189_branch_calibration/results.json`

## 모델 190: model189 prior scale 국소 재탐색

model189 image-wise 최적이 prior scale 탐색 상단에 있었기 때문에 prior `1.20~1.40`,
beta `0.60~0.90`을 다시 비교했다. 최적은 prior `1.25`, beta `0.80`으로, model189의
beta `0.75`보다 약간 낮은 추가 상승을 보였다.

| 모델 | 최적 설정 | Global mIoU | Image-wise 평균 | empty 제외 image-wise |
|---:|---|---:|---:|---:|
| 190 | prior scale `1.25`, beta `0.80` | 81.1395% | **78.5334%** | **79.3273%** |

model189 image-wise 대비 `+0.0016%p`로 사실상 calibration 포화 구간이다. 현재
image-wise 권장 설정은 모델 190으로 갱신하고, global 지표는 모델 189의 global
후보를 별도 기록한다.

결과:
`/workspace/komap_model1/outputs/model190_prior_refine/results.json`

## 모델 191: model186 rare-class 추가 혼합

모델 190의 Primary model186 교체 gamma `0.80`과 prior calibration을 고정하고,
model186 probability를 Al3Ni, Eutectic Si, 두 rare class에 각각 소량 추가했다.
Eutectic 추가는 image-wise 평균을 올리지 못했고, Al3Ni 추가 gamma `0.05`가
include image-wise 기준에서만 아주 작은 개선을 보였다.

| 기준 | 최적 설정 | Global mIoU | Image-wise 평균 | empty 제외 image-wise |
|---|---|---:|---:|---:|
| image-wise | Primary `0.80` + Al3Ni `0.05` | 81.1496% | **78.5338%** | 79.3269% |
| empty 제외 | Primary `0.80` + rare 추가 `0.00` | 81.1395% | 78.5334% | **79.3273%** |
| global | Primary `0.80` + rare `0.30` | 81.1768% | 78.4557% | 79.2352% |

모델 190 대비 include image-wise는 `+0.0004%p`로 사실상 동일하다. Al3Ni 추가
gamma를 크게 하면 성능이 바로 낮아져 rare-class 보조는 포화로 판단한다. 대회
지표가 내부 image-wise 평균이면 model191 include 후보를, empty 제외 규칙이면
model190 후보를 사용한다.

결과:
`/workspace/komap_model1/outputs/model191_rare_branch_blend/results.json`

## 모델 192: model186 장기학습 설정의 seed 456 재검증

팀원 문서의 장기 학습 일정이 seed에 덜 민감한지 확인하기 위해 model109 checkpoint에서
같은 raw grayscale + per-image z-score + empirical CDF 3채널, Residual ASPP
dilation `1,2,4,8`, 512 patch, 16 patches/image, 8 epochs 설정을 seed `456`으로
다시 학습했다. P100에서는 BF16 대신 AMP FP16을 사용했다.

| 모델 | best epoch | Global mIoU | Image-wise 평균 | empty 제외 image-wise | 판정 |
|---:|---:|---:|---:|---:|---|
| 192 | 5 | 78.4413% | 74.9734% | 75.5204% | model186보다 낮아 미채택 |

model186 seed `2028` branch보다 단일 성능이 낮고, 기존 ensemble의 보조 branch로
사용할 가치가 있는지 별도 확인하기 위해 model193에서 확률 혼합을 추가로 평가했다.

결과:
`/workspace/komap_model1/outputs/model192_team_schedule_seed456/summary.json`

## 모델 193: seed 456 branch를 현재 최고 route에 교체 혼합

model191의 후처리와 calibration을 고정한 뒤, model192 확률을 Primary에 gamma `0.80`으로
교체하고 Al3Ni/Eutectic Si 추가 혼합 gamma를 `0~0.30`에서 탐색했다. 최적 image-wise
후보는 rare 추가 gamma `0.00`이었으므로, 새 seed의 rare-class 보정은 효과가 없었다.

| 기준 | 최적 설정 | Global mIoU | Image-wise 평균 | empty 제외 image-wise |
|---|---|---:|---:|---:|
| image-wise | model192 Primary `0.80`, rare 추가 `0.00` | 80.9139% | **78.3244%** | 79.1183% |
| global | model192 Primary `0.80`, Al3Ni 추가 `0.20` | **80.9205%** | 78.2958% | 79.0859% |

model191 image-wise 최고 `78.5338%`보다 `0.2094%p` 낮았고, model190의 기존
model186 Primary branch route도 재현하지 못했다. 따라서 seed 456 branch는 채택하지
않고, 현재 image-wise 제출 후보는 model191을 유지한다.

결과:
`/workspace/komap_model1/outputs/model193_seed456_rare_blend/results.json`

## 모델 194: 팀원식 512→768→native 다중 해상도 학습

팀원 `model_results_0921_1.md`의 핵심 차이인 512, 768, native-resolution 단계 전환을
재현했다. raw grayscale + per-image z-score + empirical CDF 3채널과 Residual ASPP
dilation `1,2,4,8`을 유지하고, validation은 모든 stage에서 768 patch/576 stride로
동일하게 평가했다. seed는 `2028`이며 P100에서는 AMP FP16을 사용했다.

| stage/epoch | 학습 patch | Global mIoU | Image-wise 평균 | empty 제외 image-wise |
|---|---:|---:|---:|---:|
| 0/1 | 512 | 78.1946% | 75.2074% | 75.2074% |
| 0/2 | 512 | 77.7932% | 74.3151% | 74.3151% |
| 0/3 | 512 | 78.8450% | 75.7574% | 75.7574% |
| 1/1 | 768 | 78.9288% | 75.6144% | 75.6144% |
| 1/2 | 768 | 79.4423% | 76.4174% | 76.4174% |
| 1/3 | 768 | 79.2527% | 76.0712% | 76.0712% |

native stage 진입 시 batch `1`이 global pooling BatchNorm의 입력 `[1,128,1,1]`을
만들어 중단됐지만, stage 1 best checkpoint는 정상 저장됐다. native stage는 모델195에서
batch `2`로 이어서 재개했다.

결과:
`/workspace/komap_model1/outputs/model194_multistage_seed2028/history.json`

## 모델 195: native stage batch-2 재개 학습

모델194의 stage 1 best checkpoint에서 이어받아 1024 native crop, 8 patches/image,
batch `2`, encoder/decoder learning rate `5e-5/1.5e-4`로 2 epoch을 추가 학습했다.
BatchNorm 입력 조건을 만족하도록 batch 2를 사용했고, best checkpoint는 native stage
epoch 2였다.

| 모델 | Global mIoU | Image-wise 평균 | empty 제외 image-wise | 판정 |
|---:|---:|---:|---:|---|
| 195 | 79.0904% | **76.1109%** | **76.6552%** | 보조 branch 채택 |

model194 stage1 best의 empty 제외 `76.4174%` 대비 선택 지표는 `76.6552%`
(`+0.2378%p`)로 상승했고, include 기준은 `76.1109%`였다. 이 valid probability를
모델196/197 혼합 탐색에 사용했다.

결과:
`/workspace/komap_model1/outputs/model195_native_resume/summary.json`

test 정답은 없으므로 수치 점수는 계산하지 않았고, 동일 best checkpoint로 제출용 mask
10장을 다음 경로에 생성했다.

`/workspace/komap_model1/outputs/model195_native_resume/test_masks/`

## 모델 196: native branch rare-class 혼합

모델195 확률을 기존 model190/191 route의 Primary에 gamma `0.80`으로 교체하고,
Al3Ni와 Eutectic Si에 rare branch를 함께 추가했다. 18개 후보를 평가한 결과 rare
추가 gamma `0.10`이 image-wise 기준 최고였다.

| 기준 | 최적 설정 | Global mIoU | Image-wise 평균 | empty 제외 image-wise |
|---|---|---:|---:|---:|
| image-wise | Primary `0.80` + rare `0.10` | 81.1628% | **78.5386%** | 79.3288% |
| global | Primary `0.80` + rare `0.30` | **81.1937%** | 78.5063% | 79.2854% |

결과:
`/workspace/komap_model1/outputs/model196_native_rare_blend/results.json`

## 모델 197: native branch gamma 국소 재탐색

모델196 주변에서 Primary replacement gamma `0.70~0.90`, rare mode(Al3Ni+Eutectic,
단일 rare class)와 rare gamma `0.05~0.15`를 80개 조합으로 재탐색했다. image-wise
최적은 Primary `0.75`와 두 rare class 추가 `0.075`였다.

| 기준 | 최적 설정 | Global mIoU | Image-wise 평균 | empty 제외 image-wise |
|---|---|---:|---:|---:|
| image-wise | Primary `0.75` + rare `0.075` | 81.1673% | **78.5564%** | **79.3477%** |
| global | Primary `0.70` + rare `0.15` | **81.1871%** | 78.5546% | 79.3424% |

model196 대비 image-wise 평균은 `+0.0178%p`, 모델191 대비 `+0.0226%p` 상승했다.
현재 내부 image-wise 제출 후보는 모델197 image-wise 설정으로 갱신한다. Global 기준도
model189의 81.1856%를 소폭 넘어섰지만, 두 metric의 최적 설정은 다르므로 대회 metric
정의에 맞춰 선택한다.

결과:
`/workspace/komap_model1/outputs/model197_native_narrow_blend/results.json`

## 모델 214: 새 branch의 희귀 클래스 전용 교체 탐색

모델197의 Primary 경로는 고정하고, Al3Ni/Eutectic Si만 모델207(이미지 IoU
fine-tune), 모델209(boundary/Tversky), 모델211(test-domain distillation) 확률로
교체하는 156개 조합을 정확한 spatial refinement 경로로 평가했다. 새 branch를 희귀
클래스에 붙이는 모든 경우보다 기존 model195 rare branch가 최선이었다.

| 기준 | 최적 설정 | Global mIoU | Image-wise 평균 | empty 제외 image-wise | 판정 |
|---|---|---:|---:|---:|---|
| 전체 기준 | Primary model195 `0.75`, Al3Ni/Eutectic model195 `0.075` | 81.1673% | **78.5564%** | **79.3477%** | 미채택, model197과 동일 |

결과:
`/workspace/komap_model1/outputs/model214_rare_branch_probe/results.json`

## 모델 216: model195 native branch 픽셀 잔차 보정

model195를 고정한 뒤 train split의 라벨로 작은 residual MLP를 학습했다. 입력은
모델 확률 logit, grayscale, local contrast, gradient, entropy, normalized coordinate이며,
valid에서는 보정 강도 alpha를 `0~1.0`으로 탐색했다. 학습된 보정은 train에는 맞았지만
valid에서 기준 alpha `0.0`보다 모두 낮아 branch로 채택하지 않았다.

| alpha | Global mIoU | Image-wise 평균 | empty 제외 image-wise |
|---:|---:|---:|---:|
| 0.0 (기준) | 79.0904% | 76.1109% | 76.6552% |
| 0.1 | 78.7172% | 75.7622% | 76.2942% |
| 0.2 | 78.2788% | 75.3276% | 75.8467% |

결과:
`/workspace/komap_model1/outputs/model216_pixel_refiner_native/summary.json`

## 모델 217: model195 native 표현의 label-free BN-affine TENT

모델195의 네트워크 가중치는 고정하고, 각 valid 이미지의 네 모서리 patch에서
BatchNorm affine 파라미터만 entropy minimization으로 1/3/5 step 적응했다. 입력 표현은
raw grayscale, per-image z-score, empirical CDF이며, valid 전체를 이미지별로 독립
적응했다. standalone branch에서는 baseline보다 상승했지만 model197 route에 연결하기
전에는 별도 검증이 필요하다.

| 기준 | 설정 | Global mIoU | Image-wise 평균 | empty 제외 image-wise |
|---|---|---:|---:|---:|
| image-wise | 5 step, lr `2e-5` | 79.6656% | **76.7163%** | 76.7163% |
| empty 제외 | 1 step, lr `2e-5` | 79.2963% | 76.3116% | **76.8635%** |
| model195 기준 | no adaptation | 79.0904% | 76.1109% | 76.6552% |

결과:
`/workspace/komap_model1/outputs/model217_native_tent/results.json`

## 모델 219: TENT rare-class route 결합

model217의 image-wise 최적 5-step TENT branch를 model195와 결합했다. model195를
Primary에 유지하고, Al3Ni와 Eutectic Si 각각에 TENT 확률을 20% 추가한 조합이 exact
spatial refinement 평가에서도 최고였다.

| 기준 | 설정 | Global mIoU | Image-wise 평균 | empty 제외 image-wise |
|---|---|---:|---:|---:|
| image-wise/empty 제외 | Primary model195 `0.65`; Al3Ni/Eutectic model217 `0.20` | **81.2407%** | **78.6088%** | **79.3963%** |
| model197 기준 | Primary model195 `0.75`; rare model195 `0.075` | 81.1673% | 78.5564% | 79.3477% |

image-wise 평균은 model197 대비 `+0.0524%p` 상승했다. 다만 이 선택은 20장 valid의
후처리/branch 조합 탐색에서 나온 값이므로 독립 test 점수는 아니다. 결과 파일의
상위 33개 exact spatial refinement 평가 안에서도 위 설정이 image-wise 최고였다.

결과:
`/workspace/komap_model1/outputs/model219_tent_branch_fusion/results.json`

## 모델 220: native TENT 적응 step/LR 확장 탐색 (부분 완료)

model195 native 표현의 BN-affine TENT에서 10/20-step 설정을 비교했다. 단독 branch는
10-step, lr `5e-5`가 최고였지만 image-wise mIoU `76.8408%`로 model217의 5-step
최고 `76.7163%`보다 `+0.1245%p`에 그쳤다. 20-step 설정은 모두 10-step 최고보다 낮아
추가 30-step 탐색을 중단했다. 별도 결합 탐색은 계속했다.

| TENT 설정 | Global mIoU | Image-wise 평균 | empty 제외 |
|---|---:|---:|---:|
| 10 step, lr `1e-5` | 79.6695% | 76.7964% | 76.7964% |
| 10 step, lr `2e-5` | 79.6812% | 76.8057% | 76.8057% |
| **10 step, lr `5e-5`** | **79.7236%** | **76.8408%** | **76.8408%** |
| 20 step, lr `1e-5` | 79.4676% | 76.7003% | 76.7003% |
| 20 step, lr `2e-5` | 79.4849% | 76.7202% | 76.7202% |
| 20 step, lr `5e-5` | 79.5492% | 76.7921% | 76.7921% |

결과는 `/workspace/komap_model1/outputs/model220_native_tent_extended/run.log` 및
완료된 설정의 `valid_probs_*.npz`에 기록했다.

## 모델 221: 10-step TENT rare-branch 결합

모델220의 10-step, lr `2e-5` TENT branch를 model195 기반 route와 결합했다. 210개
cheap 후보와 33개 exact spatial refinement 평가에서 rare-class TENT 혼합량은 기존과
같이 Al3Ni/Eutectic Si 각각 `0.20`이었고, Eutectic edge blend를 `0.25`에서 `0.15`로
낮춘 조합이 최고였다.

| 선택 기준 | 설정 차이 | Global mIoU | Image-wise 평균 | empty 제외 image-wise |
|---|---|---:|---:|---:|
| image-wise | Primary gamma `0.65`, Eutectic edge blend `0.15`, Al bias `0.0` | 81.2386% | **78.6226%** | **79.4104%** |
| global | Primary gamma `0.85`, Al bias `+0.10`, Eutectic edge blend `0.25` | **81.2847%** | 78.6047% | 79.3941% |

image-wise 개선 폭은 모델219 대비 `+0.0138%p`로 작으며, validation 20장에 대한
탐색 결과다.

결과: `/workspace/komap_model1/outputs/model221_tent10_branch_fusion/results.json`

10-step lr `5e-5` branch의 희귀 혼합 gamma를 `0.40`까지 늘린 모델224가 image-wise
`78.6610%`로 갱신했다. 더 높은 gamma(model226)와 20-step branch gamma 탐색(model225)은
20-step 탐색은 모델225에서 `78.6921%`를 기록했다. 추가 10-step gamma(model226)와
10/20-step 확률 평균 branch(model228)는 진행 중이다.

## 모델 222: 10-step TENT lr 확장 branch 결합

모델220의 단독 최고 branch(10-step, lr `5e-5`)를 model195 기반 route와 결합했다.
희귀 클래스 혼합량은 Al3Ni/Eutectic Si 각각 `0.20`이며, image-wise route의
Eutectic edge blend는 `0.15`다.

| 선택 기준 | Global mIoU | Image-wise 평균 | empty 제외 image-wise | 모델221 대비 |
|---|---:|---:|---:|---:|
| image-wise | 81.2429% | **78.6257%** | **79.4136%** | **+0.0031%p** |
| global | **81.2906%** | 78.6086% | 79.3982% | global `+0.0059%p` |

image-wise 최고는 모델221과 동일하게 rare gamma `0.20`이었고, 이후 모델224에서
혼합량을 확장한 설정이 이 점수를 추가로 높였다.

결과: `/workspace/komap_model1/outputs/model222_tent10_lr5e5_branch_fusion/results.json`

## 모델 223: 20-step TENT rare-branch 결합

단독 image-wise 성능은 10-step branch보다 낮았던 20-step, lr `5e-5` TENT 출력을
rare-class branch로 결합했다. 단독 점수만으로는 보이지 않았던 상보성이 있어 모델222를
넘었다. image-wise 최적은 rare gamma `0.20`, Primary gamma `0.65`, prior beta scale
`0.70`, Eutectic edge blend `0.25`였다.

| 선택 기준 | Global mIoU | Image-wise 평균 | empty 제외 image-wise |
|---|---:|---:|---:|
| image-wise | 81.2577% | **78.6469%** | **79.4336%** |
| global | **81.2952%** | 78.6311% | 79.4205% |

모델222 대비 image-wise `+0.0212%p` 상승했다. 10-step lr `5e-5`의 혼합 강도
확장(model224)에서 추가 상승했다.

결과: `/workspace/komap_model1/outputs/model223_tent20_rare_branch_fusion/results.json`

## 모델 224: 10-step TENT rare 혼합 강도 확장

모델222의 10-step, lr `5e-5` TENT branch를 고정하고 rare-class gamma 탐색 범위를
`0.15~0.40`으로 확장했다. 275개 평가 기록(242 cheap + 33 exact) 중 최고는
Al3Ni/Eutectic Si 양쪽 gamma `0.40`으로, 기존 상한 `0.20`을 넘었다.

| 선택 기준 | Primary gamma | Rare gamma | prior / beta | Global mIoU | Image-wise 평균 | empty 제외 |
|---|---:|---:|---|---:|---:|---:|
| image-wise | 0.75 | 0.40 / 0.40 | 1.20 / 0.80 | 81.3181% | **78.6610%** | **79.4423%** |
| global | 0.75 | 0.40 / 0.40 | 1.25 / 0.80, Al bias +0.10 | **81.3542%** | 78.6583% | 79.4412% |

image-wise는 모델223보다 `+0.0141%p` 상승했다. 20-step branch와 gamma 확장 조합인
모델225는 추가로 `78.6921%`까지 올렸다. 모델226에서는 10-step branch의 gamma를
`0.40~0.80`으로 더 높였지만 image-wise 최고는 `78.6547%`로 낮아졌다.
10/20-step 확률 평균 branch(model228)는 평가 중이다.

결과: `/workspace/komap_model1/outputs/model224_tent_rare_gamma_expansion/results.json`

## 모델 225: 20-step TENT rare 혼합 강도 확장

모델223에서 사용한 20-step, lr `5e-5` TENT branch의 rare gamma를 `0.15~0.40`으로
넓혔다. 최고 image-wise는 두 rare class gamma `0.40`, Primary gamma `0.75`, prior
scale `1.25`, beta `0.80`, Eutectic bias `+0.15`, edge blend `0.25`였다.

| 선택 기준 | Global mIoU | Image-wise 평균 | empty 제외 image-wise |
|---|---:|---:|---:|
| image-wise | 81.3033% | **78.6921%** | **79.4693%** |
| global | **81.3495%** | 78.6824% | 79.4636% |

모델224의 image-wise 최고보다 `+0.0311%p` 상승했다. gamma `0.40`보다 더 큰 혼합은
아직 시험하지 않았다.

결과: `/workspace/komap_model1/outputs/model225_tent20_rare_gamma_expansion/results.json`

## 모델 226: 10-step TENT rare gamma 고강도 확장

10-step lr `5e-5` branch에서 rare gamma를 `0.40~0.80`으로 추가 탐색했다. image-wise
최고는 다시 gamma `0.40`에서 나왔고, gamma `0.50`의 global 최고도 기존 model224보다
낮았다. gamma를 더 높이는 방향은 채택하지 않는다.

| 선택 기준 | Rare gamma | Global mIoU | Image-wise 평균 | empty 제외 image-wise |
|---|---:|---:|---:|---:|
| image-wise | 0.40 / 0.40 | 81.2946% | 78.6547% | 79.4363% |
| global | 0.50 / 0.50 | 81.3467% | 78.6405% | 79.4191% |

결과: `/workspace/komap_model1/outputs/model226_tent10_rare_gamma_high/results.json`

## 모델 227: TENT adaptation-length probability soup

model220의 10-step/lr `5e-5`와 20-step/lr `5e-5` validation probability를 50:50으로
평균하고 클래스 합을 재정규화했다. 20장 모두 생성됐고 최대 probability-sum 오차는
`2.4e-7`이었다. 단독 image-wise mIoU는 `76.8400%`로 10-step branch와 사실상 같았다.

결과: `/workspace/komap_model1/outputs/model227_tent_probability_soup/valid_probs.npz`

## 모델 228: snapshot-soup rare-branch 결합

227 probability soup를 model195 기반 route와 결합하고 rare gamma `0.20~0.60`을
탐색했다. image-wise 최고는 Primary gamma `0.85`, 두 rare gamma `0.50`, prior `1.25`,
beta `0.80`, Eutectic bias `+0.15`였다.

| Global mIoU | Image-wise 평균 | empty 제외 image-wise | 모델225 대비 |
|---:|---:|---:|---:|
| 81.3109% | 78.6709% | 79.4445% | `-0.0212%p` |

snapshot soup는 모델225를 넘지 못해 제출 후보로 채택하지 않는다.

결과: `/workspace/komap_model1/outputs/model228_tent_snapshot_soup_fusion/results.json`

## 모델 229: 전체 이미지 공유 TENT (실패)

20장 validation의 입력만 사용해 shared BN-affine TENT를 5 global steps, lr `1e-5`로
적용했다. image-wise mIoU는 `75.7464%`로 model195 단독 `76.1109%`보다 낮았다.
checkpoint 대비 모든 BN affine weight/bias 변화가 정확히 `0`이었지만 running mean/var는
크게 이동했고, 후반 image의 순차 BN 통계가 결과를 악화시킨 것으로 확인됐다.

결과: `/workspace/komap_model1/outputs/model229_global_tent/`

## 모델 230: 누적 global AdaBN (미채택)

모델229의 마지막 이미지 편향을 줄이도록 BN running statistics를 validation 20장에 대해
cumulative average로 다시 계산했다. affine 파라미터는 고정했다. image-wise mIoU는
`75.6943%`로 개선되지 않았다. 이 확률을 model195 rare branch에 gamma `0.075/0.20/0.40`
혼합해 exact-evaluate한 최고도 `78.5424%`로 model197의 `78.5564%`보다 낮았다.

결과: `/workspace/komap_model1/outputs/model230_global_adabn/`

## 모델 231: per-image D4-view TENT (미채택)

model195에서 각 validation 이미지의 네 corner tile을 8개 D4 변환으로 적응했다.
entropy loss는 변환별 평균이고 BN moments는 한 이미지의 모든 변환에 대해 누적 평균했다.
단독 image-wise는 `76.6154%`로 per-image native TENT 최고보다 낮았다.

이 branch를 model195 rare class에 gamma `0.075/0.20/0.40`으로 exact 결합했을 때
최고는 gamma `0.40`의 `78.6677%`였다. 모델225(`78.6921%`)를 넘지 못해 채택하지 않는다.

| 평가 | Global mIoU | Image-wise 평균 | empty 제외 |
|---|---:|---:|---:|
| 단독 D4-TENT | 79.2903% | 76.6154% | 76.6154% |
| model195 + D4-TENT rare gamma 0.40 | 81.2784% | 78.6677% | 79.4415% |

결과: `/workspace/komap_model1/outputs/model231_d4_tent/`

## 모델 232: D4 marginal-entropy TENT (단독 미채택; branch 결합 평가 중)

model195 checkpoint를 초기값으로 사용해 valid 이미지마다 네 corner tile을 8개 D4
변환에 통과시켰다. 변환별 확률을 원 방향으로 정렬한 뒤 marginal prediction의 entropy를
최소화하도록 BN affine 파라미터를 3 step, lr `2e-5`로 적응했다. activation 메모리를
줄이기 위해 marginal entropy gradient를 계산한 뒤 view별 graph를 replay하는 방식을 썼다.

| 평가 | Global mIoU | Image-wise 평균 | empty 제외 image-wise |
|---|---:|---:|---:|
| 단독 model232 | 79.3212% | 76.6507% | 76.6507% |
| image-wise 최고 model225 | 81.3033% | **78.6921%** | **79.4693%** |

단독 branch 점수는 model225보다 `2.0414%p` 낮아 단독 채택하지 않는다. 독립 점수만으로
상보성을 배제하지 않기 위해 model232 probability를 model195 기반 route의 rare-class
source로 바꿔 넣은 model233 결합 평가도 수행했다. 최고점은 model225보다 낮았다.

model232 class별 global IoU는 Al3Ni `77.8663%`, Primary Si `87.3274%`, Al `93.5485%`,
Eutectic Si `58.5424%`다. 예측 probability와 단독 평가는
`/workspace/komap_model1/outputs/model232_d4_marginal_tent/`에 저장했다.

## 모델 233: model232 D4 branch rare-class 결합 (미채택)

model232 probability를 model195 기반 route의 Al3Ni/Eutectic Si branch에 넣고 rare gamma,
Primary gamma, intensity prior calibration 및 Eutectic edge/bias 설정을 탐색했다. 최고는
두 rare class gamma `0.40`, Primary gamma `0.75`, prior `1.25`, beta `0.80`, Eutectic
bias `+0.15`, edge blend `0.25`였다.

| 모델 | Global mIoU | Image-wise 평균 | empty 제외 image-wise | model225 대비 |
|---|---:|---:|---:|---:|
| model233 | 81.2758% | 78.6796% | 79.4511% | `-0.0125%p` |
| **model225 최고** | **81.3033%** | **78.6921%** | **79.4693%** | — |

따라서 model232 branch는 보완 source로도 채택하지 않는다. 상세 후보 기록은
`/workspace/komap_model1/outputs/model233_d4_marginal_tent_fusion/results.json`에 있다.

## 모델 234: model195 native checkpoint 저학습률 supervised continuation

model195 best checkpoint에서 이어서 1024 crop, 8 patches/image, batch 2로 4 native-stage
epoch을 학습했다. encoder/decoder learning rate는 `1e-5/3e-5`, seed `2029`였고,
checkpoint 선택 metric은 대회 목표에 맞춰 image-wise mIoU(include empty class)로 설정했다.

| Epoch | Global mIoU | Image-wise 평균 | empty 제외 image-wise |
|---:|---:|---:|---:|
| 1 | 78.9743% | 75.8140% | 76.3610% |
| 2 | 79.0028% | 75.8026% | 76.3427% |
| **3 (best)** | **79.5660%** | **76.4220%** | **76.9816%** |
| 4 | 79.0099% | 75.8386% | 76.3832% |

epoch3 checkpoint는 model195 단독 image-wise `76.1109%`보다 `+0.3111%p` 높다. 다만
최고 앙상블 route보다 낮아 단독 대체 branch로 채택하지 않고, model235에서 보완성을
별도 평가했다.

결과: `/workspace/komap_model1/outputs/model234_native_low_lr_continue/`

## 모델 235: model234 supervised branch exact route probe (미채택)

model234 probability를 model195 기반 route와 결합해 16개 조합을 exact spatial refinement까지
평가했다. Primary source/gamma와 Al3Ni·Eutectic Si의 model234 혼합량을 바꾸고 기존 최고
route의 prior/bias/edge calibration은 고정했다.

| 최고 기준 | 설정 요약 | Global mIoU | Image-wise 평균 | empty 제외 |
|---|---|---:|---:|---:|
| image-wise | Primary model195 `0.75`; Al3Ni model234 `0.20`; Eutectic model234 `0.40` | 81.2596% | **78.5801%** | **79.3591%** |
| global | Primary model195 `0.75`; 두 rare class model234 `0.40` | **81.2850%** | 78.5352% | 79.3008% |
| model225 최고 | Primary model195 `0.75`; 20-step TENT rare branch `0.40` | 81.3033% | **78.6921%** | **79.4693%** |

model235 image-wise 최고는 model225보다 `-0.1120%p` 낮다. 따라서 model234의 supervised
continuation은 단독 checkpoint로는 model195보다 나아졌지만, 이번 exact route에서 보완
효과를 보이지 않아 현재 ensemble에는 채택하지 않는다.

구현: `model235_native_branch_probe.py`
결과: `/workspace/komap_model1/outputs/model235_native_branch_probe/results.json`

## 모델 236: MatSSL-inspired ResNet34 비지도 도메인 적응

model195의 ResNet34 인코더를 시작점으로 사용했다. train split 이미지 70장만 사용해
256×256 patch 쌍을 만들고, 다단계 feature를 sigmoid-gated pooling으로 결합한 뒤
NT-Xent contrastive loss로 30 epoch 사전학습했다. 마스크 라벨은 이 단계에서 사용하지
않았고 valid/test 이미지도 읽지 않았다. 최저 contrastive loss는 epoch 27의 `0.03731`이다.
이는 MatSSL 논문의 metallography 영역 아이디어를 차용한 변형이며, 논문 설정을 그대로
재현한 것은 아니다.

결과: `/workspace/komap_model1/outputs/model236_matssl_ssl/`
구현: `model236_matssl_inspired_ssl.py`

## 모델 237: model236 인코더 이식 후 supervised fine-tune (미채택)

model236 best encoder의 216개 텐서를 model195 segmentation checkpoint에 이식한 뒤,
1024 patch, 8 patches/image, batch 2, 4 epoch으로 지도 미세조정했다. encoder/decoder LR은
`1e-5/3e-5`이며, valid image-wise mIoU를 checkpoint 선택 기준으로 썼다.

| Epoch | Global mIoU | Image-wise 평균 | empty 제외 image-wise |
|---:|---:|---:|---:|
| 1 | 78.6663% | 75.4692% | 76.0065% |
| 2 | 78.8499% | 75.7168% | 76.2568% |
| 3 | 78.7754% | 75.7391% | 76.2776% |
| **4 (best)** | **79.4761%** | **76.2376%** | **76.8022%** |

model234의 지도 continuation best `76.4220%`보다 단독 성능이 `0.1844%p` 낮다. 클래스별
global IoU는 Al3Ni `77.3257%`, Primary Si `88.5090%`, Al `93.4469%`, Eutectic Si `58.6229%`다.

결과: `/workspace/komap_model1/outputs/model237_matssl_finetune/`
초기화 도구: `model237_transfer_ssl_encoder.py`

## 모델 238: model237 분기 16개 exact route probe (미채택)

model237 확률을 model195 기반 route와 결합해 Primary source/gamma 및 두 rare class 혼합률
조합 16개를 실제 spatial refinement까지 적용해 valid에서 평가했다. image-wise 최고는
Primary model195 `0.85`, Al3Ni model237 `0.20`, Eutectic Si model237 `0.40`이었다.

| 평가 | Global mIoU | Image-wise 평균 | empty 제외 image-wise |
|---|---:|---:|---:|
| model238 best image-wise route | 81.2253% | 78.5496% | 79.3288% |
| model225 현재 최고 | 81.3033% | **78.6921%** | **79.4693%** |

model238은 model225 대비 image-wise 기준 `-0.1425%p`이므로 현재 최고 route에는 넣지 않는다.
후보 전체 기록은 `/workspace/komap_model1/outputs/model238_matssl_branch_probe/results.json`에
저장했다.

구현: `model238_matssl_branch_probe.py`

## 모델 239: zero-init stride-1 native-resolution stem (결합 평가 중)

팀원 보고서에서 제안한 32채널 native-resolution feature 보존 아이디어를 적용했다.
model195의 ResNet34/U-Net++/ASPP 경로와 pretrained weight는 그대로 두고, 입력 해상도에서
동작하는 3→32→32 convolution branch를 추가해 마지막 decoder head에 주입했다. 새 주입
projection은 0으로 초기화해 학습 시작 시 기존 모델의 출력이 정확히 보존되도록 했다.
검증은 기존처럼 8-way D4 TTA와 Gaussian overlap blending을 사용했다.

1024 patch, 8 patches/image, batch 2, 4 epoch으로 미세조정하고 image-wise mIoU로 선택했다.

| Epoch | Global mIoU | Image-wise 평균 | empty 제외 image-wise |
|---:|---:|---:|---:|
| 1 | 79.7670% | 76.7253% | 76.7253% |
| **2 (best)** | **79.6581%** | **76.7430%** | **76.7430%** |
| 3 | 79.3088% | 76.3796% | 76.3796% |
| 4 | 79.2686% | 76.3518% | 76.3518% |

단독 image-wise 기준으로 model234보다 `+0.3210%p`, model195보다 `+0.6321%p`다. 모델225
route의 추가 branch로 유효한지는 model240 exact 결합 평가 후 확정한다.

결과: `/workspace/komap_model1/outputs/model239_native_stem/`
구현: `model239_native_stem_finetune.py`

## 모델 240: model239 rare-class branch 결합 (미채택)

model239 probability를 model195 기반 route에 넣고 16개 exact 후보를 평가했다. 최선은
Primary model195 `0.75`, Al3Ni model239 `0.20`, Eutectic Si model239 `0.40`이었다.

| Global mIoU | Image-wise 평균 | empty 제외 image-wise | model225 대비 image-wise |
|---:|---:|---:|---:|
| 81.2663% | 78.6186% | 79.3967% | `-0.0735%p` |

단독 분기의 상승이 최고 조합의 상승으로 이어지지 않아 model225 route에 채택하지 않았다.
결과: `/workspace/komap_model1/outputs/model240_native_stem_branch_probe/results.json`

## 모델 241: Al3Ni/Eutectic class-wise TENT·native stem 혼합 (미채택)

model225의 20-step TENT 확률과 model239 확률을 각각 Al3Ni/Eutectic Si source로 독립 선택해
32개 exact 후보를 비교했다. image-wise 최고는 기존 model225와 동일하게 두 rare class 모두
TENT source `217`, gamma `0.40`이었고, 즉 native stem은 주 지표를 개선하지 못했다.

| 선택 기준 | 설정 | Global mIoU | Image-wise 평균 | empty 제외 |
|---|---|---:|---:|---:|
| image-wise | Primary 195 `0.75`; Al3Ni/Eutectic TENT `0.40/0.40` | 81.3033% | **78.6921%** | **79.4693%** |
| global | Primary 195 `0.85`; Al3Ni 239 `0.40`; Eutectic TENT `0.40` | **81.3367%** | 78.6621% | 79.4359% |

global 기준 후보도 model225의 global 최고 `81.3495%`를 넘지 못했다. 따라서 model239를
현재 최고 앙상블에 포함하지 않는다.
결과: `/workspace/komap_model1/outputs/model241_classwise_stem_tent_probe/results.json`
구현: `model241_classwise_stem_tent_probe.py`

## 모델 242: model239 low-LR continuation + EMA 0.99

팀원 학습 설정의 EMA `0.99`를 분리 검증했다. model239 best에서 1024 patch, 8
patches/image, batch 2로 4 epoch 이어 학습했으며 encoder/decoder/native branch LR은
`5e-6/1.5e-5/1e-4`였다. 각 epoch에서 raw와 EMA 모델을 모두 D4+Gaussian valid 평가했다.

| 가중치 선택 | Best epoch | Global mIoU | Image-wise 평균 | empty 제외 |
|---|---:|---:|---:|---:|
| raw | 4 | 79.7975% | 76.8283% | 76.8283% |
| **EMA 0.99** | **4** | **79.8448%** | **76.8640%** | **76.8640%** |

EMA가 raw보다 `+0.0357%p`, model239 best보다 `+0.1210%p` 높았지만, 최고 route 보완성은
다음 model243에서 주 지표 향상으로 이어지지 않았다.

결과: `/workspace/komap_model1/outputs/model242_native_stem_ema/`
구현: `model242_ema_continuation.py`

## 모델 243: model242 EMA와 20-step TENT class-wise 결합 (미채택)

Al3Ni와 Eutectic Si 각각에 model217 TENT 또는 model242 EMA를 선택하도록 32개 exact 후보를
비교했다. image-wise 최고는 model225와 동일한 model217 TENT `0.40/0.40` 조합으로,
model242를 넣어도 최고 image-wise `78.6921%`가 갱신되지 않았다. global 기준 최고는 두
rare class에 model242 EMA `0.40/0.40`, Primary model195 `0.85`였으나 `81.3698% / 78.6250%`
로 model225 global 최고 `81.3495%`보다 global은 `+0.0203%p`, image-wise는 낮았다.

결과: `/workspace/komap_model1/outputs/model243_classwise_ema_tent_probe/results.json`
구현: `model243_classwise_ema_tent_probe.py`

## 모델 244: model242 EMA teacher supervised distillation (완료)

student는 model239 best, frozen teacher는 model242 best EMA로 초기화했다. train crop에서
동일 입력에 대한 teacher softmax(T=2)를 student에 전달하고, hard-label weighted CE+Dice에
`0.2 × T² × KL`을 더해 4 epoch 학습했다. teacher/student 모두 train split만 사용했고,
valid는 epoch 선택 및 평가에만 사용했다.

| Epoch | Global mIoU | Image-wise 평균 | empty 제외 |
|---:|---:|---:|---:|
| 1 | 79.8082% | 76.7487% | 76.7487% |
| 2 | 79.8452% | 76.7947% | 76.7947% |
| 3 | 79.8210% | 76.7811% | 76.7811% |
| **4 (best)** | **79.8969%** | **76.8332%** | **76.8332%** |

model239 student보다 `+0.0902%p` 올랐고 model242 EMA teacher의 `76.8640%`보다는
`0.0308%p` 낮다. 현재 최고 route에 대한 보완성은 model245에서 평가한다.

결과: `/workspace/komap_model1/outputs/model244_supervised_distillation/`
구현: `model244_supervised_distillation.py`

## 모델 245: distilled student와 20-step TENT class-wise 결합 (미채택)

model244와 model217 TENT의 Al3Ni/Eutectic source를 독립 선택하는 32개 exact 후보를
평가했다. image-wise 최고는 model225와 동일한 양 class TENT gamma `0.40/0.40`으로
`78.6921%`를 유지했으며, distilled student는 주 지표를 개선하지 못했다. global 최고는
Primary model195 `0.85`, Al3Ni model244 `0.40`, Eutectic model217 `0.40`에서 `81.3635%`
였지만 image-wise는 `78.6709%`로 model225 best보다 `0.0212%p` 낮았다.

결과: `/workspace/komap_model1/outputs/model245_classwise_distill_tent_probe/results.json`
구현: `model245_classwise_distill_tent_probe.py`

## 모델 246: latest ensemble의 조건부 라우팅 grouped-CV (미채택)

model245 exact 후보 32개를 대상으로 raw 밝기·질감, 확률 entropy/margin, 예측 클래스 면적을
조건으로 쓰는 1단계 threshold router를 탐색했다. valid 20장을 A-ID 기준 19개 그룹으로 묶어
leave-one-specimen-group-out으로 route를 학습·평가했으며, GT는 각 fold의 학습 쪽에서만
후보 선택에 사용했다. 평가는 이미지별 네 클래스 IoU 평균(빈 union은 코드상 IoU 0)을 썼다.

전체 valid에서 밝기 threshold `gray_mean < 165.55`이면 model245 기준 route, 그 외에는
distilled model244 rare branch를 쓰는 in-sample route가 `78.7912%`로 고정 model225 route의
`78.6921%`보다 `+0.0991%p` 높아 보였다. 그러나 그룹 held-out score는 `78.6206%`로 고정
route 대비 `-0.0714%p`였고, 그룹 bootstrap 95% 차이 구간 `[-0.1814, +0.1347]%p`가 0을
포함했다. valid에 맞춘 밝기 threshold를 채택할 근거가 없어 미채택했다.

결과: `/workspace/komap_model1/outputs/model246_grouped_route_cv/results.json`
구현: `model246_grouped_route_cv.py`

## 모델 247: 금속학적 형태 조건을 추가한 grouped-CV router (미채택)

모델246 후보·분할 방식을 유지하고, raw 밝기/texture 외에 Primary/Eutectic 예측상의
connected-component 수, 큰 component 수, 경계 밀도, 클래스 가중 gradient 통계를 추가했다.
이는 입자 크기·분산·계면 복잡도 변화에 따른 조건부 전문가 선택 가능성을 진단한다.

전체 valid 최적 rule은 여전히 `gray_mean < 165.55`였으며, 금속학적 형태 피처를 더한
그룹 held-out image-wise mIoU는 `78.5775%`로 model225 고정 route 대비 `-0.1146%p`였다.
그룹 bootstrap 95% 차이 구간 `[-0.2465, +0.1094]%p`도 0을 포함했다. 형태 피처가 held-out
fold에서 일관되게 선택되지 않아 현재는 라우팅에 채택하지 않는다.

결과: `/workspace/komap_model1/outputs/model247_metallurgical_route_cv/results.json`
구현: `model246_grouped_route_cv.py` (`--feature-set metallurgical`)

## 금속공학 기반 후속 아이디어 (아직 성능 실험 전)

- Al-Si 문헌에서는 미변형 eutectic Si의 판상/flake 형태와 Sr 개질 시 미세 fibrous 형태가
  구별되며, Feret diameter·circularity 분포로 그 차이를 정량화한다. 따라서 학습/추론 시
  이미지별 밝기만이 아니라 Si feature 크기·원형도·연결성 proxy를 이용한 domain/route 진단이
  물리적으로 타당한 다음 가설이다.
- Al-Si-Ni 합금에서 Al3Ni는 응고 조건에 따라 plate-like에서 fishbone 형태로 바뀔 수 있어,
  Al3Ni branch 라우팅은 component의 길이/분기성/방향성 통계를 조건으로 할 수 있다.
- 다상 Al-Si tomography 연구는 Si·aluminide 및 그 계면의 pixel-wise weight를 사용해 어려운
  경계를 보강하고 phase-network connectivity/Euler number까지 평가했다. 우리 데이터는 2D
  광학 이미지라 그대로 재현할 수 없지만, 계면 가중 손실과 object/topology 진단의 근거가 된다.
- 산화/노이즈가 있는 Al-Si micrograph에서는 clean/noisy 데이터 혼합 학습이 noisy 영상 성능을
  개선한 보고가 있어, 촬영·에칭 상태별 augmentation/domain grouping을 추가 검증할 가치가 있다.

참고: [Al-Si Si-phase extraction (2025)](https://doi.org/10.1016/j.mtcomm.2024.111468),
[Al3Ni morphology vs solidification conditions (2020)](https://doi.org/10.1002/adem.202000503),
[Al-Si tomography segmentation with interface-weighted loss (2019)](https://pmc.ncbi.nlm.nih.gov/articles/PMC6928157/),
[Al-Si semantic segmentation and object-measurement comparison (2024)](https://doi.org/10.1007/s11665-024-10442-5).

## 모델 248: model225 최고 route 오류 사례 감사

model225의 원본 검증 확률과 실제 route를 재현해 점수 `78.6921%`가 일치하는지 확인하고,
클래스별·이미지별 오류를 집계했다. 네 클래스의 이미지 평균 IoU는 Al3Ni `75.30%`,
Primary Si `86.07%`, Al `93.75%`, Eutectic Si `59.65%`였다. A71의 Primary Si는 정답과
예측 모두 없어 대회 계산 규칙대로 IoU 0으로 들어가며, A71 최종 점수는 `46.64%`였다.
다음 취약 사례는 A153(`72.71%`, Eutectic Si IoU `33.94%`), A74(`73.97%`, Eutectic Si
IoU `41.61%`)였다. 따라서 다음 학습 가설은 clDice 단독 개선보다는 고해상도에서 침상 경계
정보가 추가되는지 검증하는 것으로 좁혔다.

결과: `/workspace/komap_model1/outputs/model248_best_route_error_audit/results.json`
구현: `model248_best_route_error_audit.py`

## 모델 249: Eutectic Si clDice 미세조정 (미채택)

model242 EMA 가중치를 시작점으로 train split만 이용해 1024×1024 patch, 이미지당 8 crops,
batch 2, 4 epoch을 학습했다. weighted CE+Dice에 Eutectic Si soft-clDice(`0.10`, 4회
skeletonization)를 추가했다. 최고는 epoch 2의 단독 image-wise `76.6056%`(global `79.6226%`)
로, 기존 주 route보다 낮았다. clDice branch가 Eutectic의 추가 정보를 주는지 model250/252에서
확률 수준으로 따로 확인했다.

결과: `/workspace/komap_model1/outputs/model249_eutectic_cldice/summary.json`
구현: `model249_eutectic_cldice_finetune.py`

## 모델 250: clDice branch exact fusion (미채택)

기존 주 route와 model249 clDice, model244 distillation, model217 TENT rare-class 확률을 포함한
후보 73개를 exact valid 평가했다. image-wise 최고는 model225 route를 그대로 재현한
`78.6921%`; global 최고 후보는 `81.3635%`였지만 image-wise `78.6709%`로 낮았다. 따라서
clDice branch를 섞는 것만으로는 개선 근거가 없었다.

결과: `/workspace/komap_model1/outputs/model250_cldice_branch_probe/results.json`
구현: `model250_cldice_branch_probe.py`

## 모델 251: clDice branch coarse calibration shortlist

Eutectic source(20-step TENT, distillation, clDice), gamma와 Eutectic bias를 바꾼 120개 조합을
저비용으로 선별했다. coarse 점수 1위는 model244 source, gamma `0.40`, bias `+0.15`의
`78.5297%`였고, 상위 후보를 다음 exact 단계로 넘겼다. 이 단계 점수는 후보 선별용 근사치이며
최종 성능으로 채택하지 않는다.

구현·원시 기록: `model252_cldice_calibration_search.py`,
`/workspace/komap_model1/outputs/model252_cldice_calibration_search/results.json`

## 모델 252: clDice branch exact calibration (미채택)

coarse 상위 후보 24개와 model225 기준 route를 동일한 exact evaluator로 비교했다. 최고
image-wise는 Eutectic source model249 gamma `0.40`, Al3Ni source model217 gamma `0.40`,
Primary model195 gamma `0.75` 조합의 `78.6986%`(global `81.3425%`)였다. model225 기준
`78.6921%`보다 겨우 `+0.0065%p`라 seed·검증 표본 변동보다 훨씬 작고, 실질적인 성능 개선으로
판정하지 않았다. 현재 채택 route는 계속 model225다.

결과: `/workspace/komap_model1/outputs/model252_cldice_calibration_search/results.json`
구현: `model252_cldice_calibration_search.py`

## 모델 253: 고정 3배 해상도 fine-tuning (미채택)

팀원 `~/Downloads/EXPERIMENTS.md`는 EffNet-B4에서 crop 기준 2배 확대 `78.19%`, 3배 확대
평균 `78.79%`, 4배 `79.01%`를 보고한다. 3배가 2배보다 일관되게 높았지만 4배와의 차이는
시드 편차보다 작다고 명시되어 있다. 다만 해당 문서는 빈 클래스를 image 평균에서 제외하는
정책을 주 지표로 사용하므로, 빈 클래스를 IoU 0으로 포함하는 우리 공식 목표 수치와 직접
비교할 수 없다. 우리 기존 1024 patch는 짧은 변이 1024보다 작을 때만 확대하므로 실제 확대율이
이미지별로 달라, 이 결과를 직접 재현하지 못했다.

이에 model242 EMA를 시작점으로 원본 약 480–512px 시야에 해당하는 고정 3배 확대,
1536×1536 patch, 이미지당 4 crops, batch 1로 3 epoch fine-tuning했다. 최고 checkpoint는
epoch 2였으며 D4 TTA·stride 864 평가에서 image-wise mIoU `74.9705%`, global mIoU
`77.3867%`였다. 이는 기존 model242 EMA의 image-wise `76.8640%`보다 낮다. model253의
확률 지도를 생성해 class-wise fusion 가능성을 별도로 검증했다.

결과: `/workspace/komap_model1/outputs/model253_highres_scale/summary.json`
구현: `model253_highres_scale_finetune.py`

## 모델 254: 고해상도 분기 class-wise fusion 및 시편 그룹 교차검증 (미채택)

model253 3배 분기와 model255의 matched 1배 대조 분기를 기존 최고 model225 route에
클래스별 30%·50% 혼합하는 사전 지정 후보 35개를 exact evaluator로 비교했다. 검증 누수를
줄이기 위해 동일 시편(A###)의 이미지들을 한 그룹으로 묶어 leave-one-specimen-out 선택도 했다.
새 분기를 포함한 후보 중 최고는 Al3Ni만 model255를 30% 섞은 `78.5902%`였고,
model225 기준 `78.6921%`보다 `-0.1020%p` 낮았다. model253 3배 분기의 최고 후보도
`78.5871%`에 그쳤다. 19개 held-out 시편 모두에서 기존 model225 route가 선택되어
교차검증 개선량은 `0.0000%p`(그룹 bootstrap 95% CI `[0.0000, 0.0000]`)였으며,
개선된 이미지도 없었다. 따라서 해상도 분기는 현재 앙상블에 넣지 않는다.

결과: `/workspace/komap_model1/outputs/model254_highres_route_probe/results.json`
구현: `model254_highres_route_probe.py`

## 모델 255: matched 1배 해상도 대조 fine-tuning

고해상도 효과와 학습량 효과를 분리하기 위해 model253과 같은 model242 EMA 초기화,
seed 253, 이미지당 4 crops, batch 1, 3 epoch, scheduler horizon 4를 사용하고 patch를
512×512, scale을 1배로 맞췄다. model253의 1536px/3배 crop과 원본상 시야가 비슷하도록
설정한 대조군이다. 최고 epoch 3에서 image-wise mIoU `75.4500%`, empty-class 제외
image-wise `76.0248%`, global mIoU `78.8276%`를 얻었다. model253보다 `+0.4796%p` 높아
이번 짧은 fine-tuning 조건에서는 3배 입력이 유리하지 않았다. 다만 두 분기 모두 model242
EMA의 `76.8640%` 및 최고 model225 route의 `78.6921%`보다 낮아 채택하지 않았다.

또한 scale 1 / patch 1024 / stride 576 설정의 model253 추론 코드 결과를 기존 model242
EMA 확률과 대조했다. 20개 valid 이미지에서 probability 최대 차이와 평균 차이는 모두 `0`,
argmax 일치율은 `100%`, 공식 image-wise 점수도 양쪽 모두 `76.8640%`로 추론 경로 parity를
확인했다.

결과: `/workspace/komap_model1/outputs/model255_matched_1x_control/summary.json`
parity 결과: `/workspace/komap_model1/outputs/model256_inference_parity/summary.json`

## 모델 257: 전체 이미지 통계 기반 1배 fine-tuning

Opus 5.5 검토에서 제기된 학습·추론 표현 통계 불일치를 직접 시험했다. 기존 학습은 각 crop에서
z-score/CDF를 다시 계산하는 반면 추론은 전체 resized image에서 계산한다. model242 EMA를
시작점으로 1024×1024 patch, 이미지당 8 crops, batch 2, 동일 학습률·4 epoch을 유지하고,
학습 시 전체 resized image의 통계를 먼저 계산한 뒤 patch를 뽑도록 변경했다. 최고 epoch 4에서
image-wise mIoU `77.2071%`, global mIoU `79.9627%`로 model242 EMA의 `76.8640%`보다
`+0.3431%p` 개선됐다. 정규화 일치가 효과가 있을 수 있음을 보였지만, 단독 성능은 현재 최고
route보다 낮다.

결과: `/workspace/komap_model1/outputs/model257_full_image_stats_1x/summary.json`
구현: `model253_highres_scale_finetune.py --global-representation`

## 모델 258: model257 class-wise route probe 및 시편 그룹 CV (미채택)

model225의 Primary·Al3Ni 설정은 고정하고, model257 확률을 Al3Ni/Eutectic Si/Primary의
부분집합별로 30% 또는 50% 혼합한 14개 exact 후보를 평가했다. in-sample 최고는 Eutectic Si에
model257을 50% 넣은 설정으로 image-wise `78.7154%`, global `81.3674%`였다. model225의
`78.6921%` 대비 raw 상승은 `+0.0233%p`에 불과하다.

19개 시편 그룹 leave-one-specimen-out에서는 평균 변화가 `-0.1266%p`, positive image는
10/20장이었고 그룹 bootstrap 95% CI는 `[-0.3063, +0.0411]%p`로 0을 포함했다. 따라서
검증셋에 맞춘 미세한 후보 순위일 가능성이 높아 route에는 채택하지 않고, 현재 운영 기준은 계속
model225로 둔다. model257의 표현 일치 학습은 유망한 단독 실험이지만, 80% 목표에는 추가 개선이
필요하다.

결과: `/workspace/komap_model1/outputs/model258_fullstats_route_probe/results.json`
구현: `model258_fullstats_route_probe.py`

## 모델 259: 전체 이미지 통계 기반 2배 해상도 fine-tuning (미채택)

model257과 같은 model242 EMA 초기화, 1024×1024 patch, 이미지당 8 crops, batch 2,
동일 학습률·4 epoch을 유지하고 scale만 1배에서 2배로 높였다. z-score/CDF는 inference와
맞춰 전체 resized image에서 계산했다. epoch별 official image-wise mIoU는 `76.6193%`,
`76.8104%`, `76.7059%`, `76.4081%`였고 epoch 2가 최고였다(global `79.9095%`). model257
1배 full-stat 최고 `77.2071%`보다 `-0.3968%p` 낮아, 이 설정에서도 입력 배율 상승은 이득이
없었다.

결과: `/workspace/komap_model1/outputs/model259_fullstats_2x/summary.json`
학습 구현: `model253_highres_scale_finetune.py --scale 2 --global-representation`

## 모델 260: model259 class-wise route probe 및 시편 그룹 CV (미채택)

model259를 각 class 부분집합에 30% 또는 50% 혼합하는 14개 exact 후보를 비교했다. image-wise
최고는 model225 reference `78.6921%` 그대로였고, 19개 시편 그룹 교차검증도 모든 fold에서
model225를 선택했다(held-out delta `0.0000%p`, 95% CI `[0.0000, 0.0000]`). global 기준
후보는 Al3Ni에 model259 50%를 섞은 `81.3242%`였지만 model225 global `81.3033%`보다
`+0.0209%p`일 뿐이며 image-wise는 `78.6424%`로 하락했다. model259 확률은 현재 route에
추가하지 않는다.

결과: `/workspace/komap_model1/outputs/model260_2x_route_probe/results.json`
구현: `model258_fullstats_route_probe.py --source-id 259`

## 모델 261: 전체 이미지 통계 기반 독립 seed 2 (미채택)

model257과 같은 model242 EMA 초기화·1배/1024 patch·8 crops/image·batch 2·학습률·4 epoch을
유지하고 seed만 `261`로 바꿨다. 최고는 epoch 3의 image-wise `77.0962%`, global `79.8903%`로,
model257 seed257의 최고 `77.2071%`보다 `-0.1109%p` 낮았다.

결과: `/workspace/komap_model1/outputs/model261_fullstats_seed261/summary.json`

## 모델 262: model257/261 2-seed 확률 평균 (미채택)

두 seed의 원본 해상도 class probability를 동일 가중치로 평균하고 재정규화했다. 단독 결과는
image-wise `77.1609%`, global `79.9299%`로 최고 단일 seed model257보다 `-0.0462%p` 낮았다.
단순 seed averaging은 이 설정에서 개선을 보이지 않았다.

결과: `/workspace/komap_model1/outputs/model262_fullstats_seed_ensemble/summary.json`
구현: `model262_seed_ensemble_probe.py`

## 모델 263: model262 seed ensemble class-wise route 및 specimen-group CV (미채택)

model262를 30%·50% 비율로 class 부분집합에 넣은 14개 exact 후보를 비교했다. raw image-wise
최고는 model225 대비 `+0.0186%p`인 `78.7107%`였지만, 19개 시편 그룹 held-out 평균 변화는
`-0.1815%p`였고 95% bootstrap CI `[-0.4041, -0.0043]%p`가 전부 음수였다. 20장 중 9장만
향상돼 seed ensemble을 route에 포함하지 않는다.

결과: `/workspace/komap_model1/outputs/model263_seed_ensemble_route_probe/results.json`
구현: `model258_fullstats_route_probe.py --source-id 262`

## 모델 264: model201 gradient/texture branch class-wise route (미채택)

별도 gradient 표현으로 학습한 model201은 단독 image-wise `75.4525%`였으며, 이를 model225
route에 class별로 섞는 14개 exact 후보를 비교했다. 모든 후보와 19개 specimen-group fold에서
model225 reference가 최선으로 선택되어 유효한 보완 효과가 확인되지 않았다.

결과: `/workspace/komap_model1/outputs/model264_gradient_route_probe/results.json`
구현: `model258_fullstats_route_probe.py --source-id 201`

## 모델 265: 전체 이미지 통계 기반 3배 해상도 fine-tuning

기존 model253의 3배/1536 patch/이미지당 4 crops/batch 1/3 epoch 설정을 유지하고,
학습 patch 추출 전에 전체 resized image에서 z-score/CDF 표현을 계산하도록 수정했다.
이는 기존 3배 학습의 crop-local 통계와 inference의 full-image 통계 불일치를 제거한다.
epoch별 official image-wise mIoU는 `74.3811%`, `75.0537%`, `75.0410%`였고
epoch 2가 최고(global `77.5224%`)였다. 기존 crop-local model253 최고 `74.9705%`보다
`+0.0832%p` 개선됐지만, full-stat 1배 model257 `77.2071%`보다 `-2.1534%p` 낮다.
따라서 정규화 불일치는 3배 branch 성능 저하의 일부였지만, 확대 자체의 이득은 확인되지 않았다.

결과: `/workspace/komap_model1/outputs/model265_highres_fullstats_3x/summary.json`
구현: `model253_highres_scale_finetune.py --scale 3 --global-representation`

## 모델 266: model265 30/50% class-wise route 및 specimen-group CV (미채택)

model265를 각 class 부분집합에 `30%` 또는 `50%` 섞는 14개 exact 후보를 model225 route와
비교했다. 최고 후보는 model225 기준 route 그대로였고 image-wise `78.6921%`를 유지했다.
새 분기를 포함한 최고 후보는 `78.5894%`로 `-0.1026%p` 낮았다. 19개 validation 시편 그룹
leave-one-group-out에서 전 fold가 model225 기준 route를 선택했고 held-out delta는
`0.0000%p`(95% CI `[0.0000, 0.0000]`)였다.

결과: `/workspace/komap_model1/outputs/model266_highres_fullstats_3x_route/results.json`

## 모델 267: model265 저혼합률 class-wise route 및 specimen-group CV (미채택)

고해상도 branch의 작은 보완 효과를 확인하기 위해 class 부분집합별 혼합률 `5/10/15/20/25%`
후보 35개와 model225 reference를 exact 평가했다. 최고는 여전히 기준 route `78.6921%`였고,
model265를 Primary Si에 5% 섞은 최고 신규 후보는 `78.6225%`(`-0.0696%p`)였다. 19개
specimen-group held-out fold 모두 model225 route를 선택해 채택 근거가 없었다.

결과: `/workspace/komap_model1/outputs/model267_highres_low_gamma_route/results.json`
구현: `model258_fullstats_route_probe.py --source-id 265 --gammas 0.05 0.10 0.15 0.20 0.25`

## 데이터 분할 specimen overlap 감사

파일명 A-ID 기준 데이터는 train 70장/59시편, valid 20장/19시편, test 10장/10시편이다.
분할은 이미지 단위로는 분리됐지만 시편 단위로는 겹친다: train-valid 7시편
(`A7`, `A40`, `A55`, `A125`, `A153`, `A181`, `A191`), valid-test 1시편(`A63`),
train-test 3시편(`A51`, `A87`, `A167`). 따라서 기존 leave-one-specimen-group-out 결과는
route 파라미터 선택에서 held-out valid group을 분리한 것이며, 해당 시편의 train 이미지가
네트워크 학습에 들어간 7개 train-valid 그룹에 대해서는 완전한 specimen-independent CV가 아니다.
이는 주어진 공식 validation 20장 점수 자체를 무효화하지 않지만, 미관측 시편 일반화 성능의
근거로 grouped route CV를 해석할 때 제한으로 고려해야 한다. 시편 독립 일반화를 검증하려면
각 fold에서 train의 동일 A-ID 이미지도 제외하고 네트워크를 다시 학습해야 하며 아직 수행하지 않았다.

## 모델 268: model265 3배 해상도 continuation (조기 중단)

model265 최고 checkpoint에서 3배/1536 patch 설정을 낮은 학습률로 4 epoch 더 이어가도록
시작했다. 추가 epoch 1은 `75.0316%`, epoch 2는 `74.8117%`로 model265 최고 `75.0537%`를
넘지 못했고 하락해 epoch 3 도중 중단했다. 3배 branch의 undertraining 가설을 지지하지 않아
고해상도 continuation은 더 진행하지 않았다. 남은 체크포인트/기록은 서버에 보존했다.

결과: `/workspace/komap_model1/outputs/model268_highres_fullstats_3x_extended/history.json`

## 모델 269: model257 1배 full-stat continuation (미채택)

model257 최고 checkpoint에서 1024 patch·8 crops/image·batch 2·full-image 통계 정규화를
유지하고 학습률을 절반으로 낮춰 4 epoch 이어 학습했다. epoch별 image-wise mIoU는
`76.9854%`, `76.9703%`, `76.8878%`, `76.9726%`였다. 최고 epoch 1도 model257의
`77.2071%`보다 `-0.2217%p` 낮아, 1배 branch도 추가 fine-tuning으로 개선되지 않았다.

결과: `/workspace/komap_model1/outputs/model269_fullstats_1x_extended/summary.json`

## 모델 270: model269 class-wise route 및 specimen-group CV (미채택)

model269를 30%·50% 비율로 class 부분집합에 섞은 14개 exact 후보를 비교했다. raw 최고는
Eutectic Si에 model269를 50% 혼합한 `78.7209%`로 model225보다 `+0.0289%p`였다.
하지만 19개 valid specimen-group held-out 선택 성능 변화는 `-0.1320%p`, positive image
9/20장이었으며 그룹 bootstrap 95% CI `[-0.3548, +0.0574]%p`가 0을 포함했다. 따라서
검증 표본의 작은 순위 변동으로 보고 model225 route를 유지한다.

결과: `/workspace/komap_model1/outputs/model270_fullstats_extended_route/results.json`

## 모델 271: model265 pixelwise confidence-gated fusion (미채택)

model265가 해당 클래스의 model225 mixed probability보다 높은 픽셀에서만 high-resolution
확률을 반영하는 gate를 Al3Ni/Primary Si/Eutectic Si 부분집합, 확률 margin `0/0.05/0.10`,
blend `0.25/0.50`로 탐색했다. 최고 raw 후보는 Primary Si만 margin `0.05`, blend `0.25`로
반영한 `78.7042%`로 기준 `78.6921%`보다 `+0.0121%p`였고 global mIoU는 오히려 낮았다.
19개 valid specimen-group held-out 선택에서는 평균 변화 `-0.0626%p`, positive image 7/20,
95% CI `[-0.1270, -0.0082]%p`로 전부 음수였다. 따라서 confidence gate는 채택하지 않는다.

결과: `/workspace/komap_model1/outputs/model271_pixelwise_highres_gate/results.json`
구현: `model271_confidence_gated_highres.py`

## 모델 272: 강한 metallography augmentation 1배 fine-tuning (미채택)

model257 full-image-stat checkpoint에서 scale 1, patch 1024, 이미지당 8 crops, batch 2로
4 epoch fine-tuning했다. contrast/brightness/gamma 변화, blur/noise와 함께 임의 회전 및
0.70~1.40배 affine zoom을 추가했다. epoch별 공식 image-wise mIoU(include empty union=0)는
`76.4414%`, `76.9807%`, `76.4905%`, `76.6927%`였고, 최고 epoch 2도 model257의 단독
`77.2071%`보다 `-0.2264%p` 낮았다. 마지막 epoch까지 회복하지 못해 이 강도의 결합 augmentation은
채택하지 않는다. 데이터 변형 가설을 완전히 기각하는 결과는 아니며, affine·photometric 효과를
분리하려면 이후 한 요인씩 비교해야 한다.

결과: `/workspace/komap_model1/outputs/model272_microstructure_augmentation_1x/summary.json`
구현: `model253_highres_scale_finetune.py --global-representation --metallography-augmentation`

## 모델 273: 3배 해상도 장기 fine-tuning (미채택)

model257 full-image-stat checkpoint에서 시작해 3배 확대, 1536 patch(원본 환산 약 512px 시야),
이미지당 4 crops, batch 1, full-image z-score/CDF를 유지하고 8 epoch 학습했다. learning rate도
기존 단기 3배 실험과 동일하게 두어 학습량 효과를 우선 확인했다. epoch별 image-wise mIoU는
`74.8187`, `75.0447`, `75.1917`, `75.4014`, `75.5004`, `75.6836`, `75.6696`, `75.5242%`였고,
epoch 6이 최고(global `78.0989%`, empty 제외 `75.6836%`)다. 총 `2,240 patches/optimizer updates`를 처리한 이번
장기 조건은 model265의 3배 단기 최고 `75.0537%`보다 `+0.6299%p` 개선했지만, model257 단독
`77.2071%`보다 낮다. epoch 7–8에서 plateau/하락해 더 긴 동일 LR 학습의 기대 이득은 작아 보인다.
최종 route 기여 여부는 model225 확률과 혼합한 exact 평가 및 specimen-group 선택 CV로 판정한다.

결과: `/workspace/komap_model1/outputs/model273_highres_3x_longer_finetune/summary.json`
구현: `model253_highres_scale_finetune.py --scale 3 --patch-size 1536 --epochs 8 --global-representation`

## 모델 274: model273 고해상도 branch와 model225 route 결합 (미채택)

model273의 Al3Ni/Primary/Eutectic class-wise 확률 혼합률을 `0.05/0.10/0.20/0.30/0.40/0.50`으로
탐색해 기준 route를 포함한 43개 exact 후보를 평가했다. 기준 route가 image-wise `78.6921%`,
global `81.3033%`로 기존 기록과 정확히 일치하는 것을 확인했다. 신규 고해상도 branch를 포함한
후보 중 최고도 기준 route를 넘지 못했고, specimen-group leave-one-out 19 folds 모두 기준 route를
선택했다(held-out 변화 `0.0000%p`, positive image `0/20`, group bootstrap 95% CI `[0, 0]%p`).
따라서 model273 branch는 현재 route에 추가하지 않는다.

결과: `/workspace/komap_model1/outputs/model274_3x_highres_route/results.json`
구현: `model258_fullstats_route_probe.py` (model220 config 6: 20-step TENT, lr `5e-5` 사용)

## 모델 275: 3배 해상도 uniform-area crop sampling (route 평가 완료)

model273과 비교 시 crop 중심 sampling만 class-focused에서 image-area uniform으로 바꾸고,
나머지 3배/1536 patch, 4 crops/image, batch 1, 8 epoch, full-image z-score/CDF, 초기 checkpoint,
LR와 validation 절차를 고정했다. epoch별 image-wise mIoU는
`74.8558`, `75.3750`, `75.3988`, `75.5759`, `75.2146`, `75.3989`, `75.5271`, `75.4738%`로,
최고 epoch 4는 `75.5759%`(global `78.1866%`)였다. model273 최고 `75.6836%`보다 `-0.1077%p`
낮아 sampling 변경의 독립적인 성능 이득은 확인되지 않았다. epoch2–4에서 짧은 우위가 있었지만
후반에 유지되지 않았다.

결과: `/workspace/komap_model1/outputs/model275_3x_uniform_area_sampling/summary.json`
구현: `model253_highres_scale_finetune.py --sampling-mode uniform_area`

## 모델 276: model275 branch와 model225 route 결합 (미채택)

model275 확률을 Al3Ni/Primary/Eutectic class-wise 비율 `0.05/0.10/0.20/0.30/0.40/0.50`으로
섞은 43개 후보를 평가했다. model225 기준 route는 image-wise `78.6921%`, global `81.3033%`로
기록과 재현 일치했다. 고해상도 branch를 넣은 어떤 후보도 기준 route를 넘지 못했고, 19개
specimen-group fold 모두 기준 route를 선택했다(held-out 변화 `0.0000%p`, positive image `0/20`,
group bootstrap 95% CI `[0, 0]%p`). 따라서 채택하지 않는다.

결과: `/workspace/komap_model1/outputs/model276_3x_uniform_area_route/results.json`
구현: `model258_fullstats_route_probe.py` (model220 config 6: 20-step TENT, lr `5e-5` 사용)

## 모델 277: 3배 해상도 고학습률 fine-tuning (단독·route 미채택)

model257 full-image-stat checkpoint에서 scale 3, patch 1536(원본 환산 약 512px 시야),
이미지당 4 crops, batch 1 설정을 유지하고 encoder/decoder/native-stem 학습률을 각각
`1e-5 / 3e-5 / 1e-4`로 높여 4 epoch 학습했다. epoch별 공식 image-wise mIoU는
`75.6051`, `74.8028`, `75.5712`, `75.7693%`였고 epoch 4가 최고(global `78.4152%`,
Eutectic Si global IoU `56.7832%`)다. model273의 best `75.6836%`보다 `+0.0857%p`지만,
model257 standalone `77.2071%`보다 `-1.4378%p`, 채택 route model225의 `78.6921%`보다
`-2.9228%p` 낮아 단독 모델 개선은 아니다. 공식 검증 확률은 정상 저장됐으며 기존 route에
보완 branch로 넣을 가치가 있는지 exact 혼합 및 specimen-group CV를 추가 확인한다.

결과: `/workspace/komap_model1/outputs/model277_3x_high_lr_finetune/summary.json`
체크포인트: `/workspace/komap_model1/outputs/model277_3x_high_lr_finetune/best.pt`

## 모델 278: model277 고해상도 branch와 model225 route 결합 (미채택)

model277 확률을 Al3Ni/Primary/Eutectic class-wise 혼합률 `0.05/0.10/0.20/0.30/0.40/0.50`으로
탐색한 43개 exact 후보 중 image-wise, empty-class 제외, global mIoU 모두 기존 model225 route가
최고였다(image-wise `78.6921%`, empty 제외 `79.4693%`, global `81.3033%`). 19개 valid specimen
group fold에서도 전부 model225 route를 골랐으며 held-out 변화 `0.0000%p`, positive `0/20`,
group bootstrap 95% CI `[0.0000, 0.0000]%p`였다. 따라서 model277은 단독 점수뿐 아니라 현재
최고 route를 보완하는 데도 유효한 개선을 만들지 못했다.

결과: `/workspace/komap_model1/outputs/model278_3x_high_lr_route/results.json`
구현: `model258_fullstats_route_probe.py` (model220 config 6: 20-step TENT, lr `5e-5`)

## 모델 279: 3배 고해상도 8,000-update 계획 run (4 epoch 조기 종료·미채택)

model257 full-image-stat ResNet34 U-Net++/ASPP checkpoint에서 scale 3, patch 1536(원본 환산
약 512px 시야), 4 crops/image, batch 1, high LR(`1e-5 / 3e-5 / 1e-4`)로 장기 학습을
시작했다. 28 epoch·7,840 updates 계획이었지만, 첫 4 epoch(1,120 updates)에서 image-wise
mIoU가 `76.0337`, `74.8857`, `75.6801`, `75.6798%`로 정체했고 best epoch 1도 model257 단독
`77.2071%`보다 `-1.1734%p`, 최고 route `78.6921%`보다 `-2.6584%p` 낮아 조기 종료했다.
epoch 4 global mIoU는 `78.9642%`였지만 목표 metric인 image-wise mIoU 개선으로 이어지지 않았다.
따라서 8,000-update 학습량 가설은 이 설정에서 검증 완료된 것이 아니며, 이후 고해상도 재시도는
팀의 EfficientNet-B4 계열 및 warmup schedule과 학습 입력 표현을 맞춰 별도 평가해야 한다.

보존 체크포인트: `/workspace/komap_model1/outputs/model279_3x_8000step/best.pt` (epoch 1)
기록: `/workspace/komap_model1/outputs/model279_3x_8000step/history.json` (4 epoch)

## 모델 280: EfficientNet-B4 U-Net 4배 고해상도 (조기 종료, 미채택)

ImageNet pretrained EfficientNet-B4 U-Net을 fresh decoder로 초기화하고, scale 4·1536px crop
(원본 약 384px 시야), uniform-area crop sampling, metallography affine/photometric augmentation,
400-update warm-up + cosine decay로 7,840 optimizer updates를 계획했다. 검증은 4 epoch마다
고정 D4 TTA/overlap inference로 측정하며, test 추론은 최종 검증 뒤로 미뤘다.

검증 image-wise mIoU는 epoch 4/8/12/16/20에서 각각 `70.3792%`, `71.1019%`, `70.6763%`,
`71.3317%`, `71.1283%`였다. 최고 epoch 16의 global mIoU는 `74.7216%`, class global IoU는
Al3Ni `70.6978%`, Primary Si `85.2935%`, Al `91.6581%`, Eutectic Si `51.2371%`다.
epoch 20 global mIoU는 `75.0102%`이나 image-wise는 최고보다 낮았다. 5,600 updates 시점에
최고 route `78.6921%`보다 `-7.3604%p` 낮고 epoch 8 이후 plateau라 판단해 epoch 20에서
중단했다. best checkpoint는 epoch 16, last checkpoint는 epoch 20이며 모두 보존했다.

이 run은 기본 클래스 가중치 `[1.15, 1.45, 0.70, 2.00]`를 사용했다. 팀 고해상도 요약의
CE+Dice 설정에는 클래스 가중치가 명시되지 않아, 다음 EffNet-B4 재현에서는 equal weights를
명시해 이 설정 차이를 분리한다. valid 20장 중 A71 한 장은 정답 Primary Si가 없어 빈 클래스
처리 convention이 점수 비교에 영향을 줄 수 있으나, 이것만으로 현재 성능 격차를 설명하진 못한다.
test 추론은 하지 않았다.

현재 체크포인트: `/workspace/komap_model1/outputs/model280_effb4_4x_highres/last.pt`
진행 기록: `/workspace/komap_model1/outputs/model280_effb4_4x_highres/history.json`

## 모델 281: EfficientNet-B4 U-Net 4배 고해상도, equal-weight CE+Dice (조기 종료, 미채택)

모델280과 같은 seed `280`, 초기화·입력 해상도·patch/FOV·sampling·augmentation·optimizer
schedule·D4 검증을 유지하고 클래스 가중치만 `[1,1,1,1]`로 고정했다. 팀의 요약 설정은 CE+Dice이며 class weighting을
명시하지 않았으므로, 모델280의 `[1.15,1.45,0.70,2.00]` 가중치가 낮은 점수의 원인인지 분리하는
통제 실험이다. 28 epoch(7,840 updates) 계획, valid 4 epoch 간격, test 추론 생략.

epoch 1/2/3/4/5 train loss는 `1.5409` / `0.8108` / `0.6508` / `0.5957` / `0.5574`이며,
epoch 4 image-wise mIoU는 `70.4383%` (global `74.5134%`)다. model280의 같은 epoch 대비
`+0.0591%p`로 거의 차이가 없었다. 더 직접적인 팀 optimizer 설정을 확인해 차등 LR 실험으로
전환하며 epoch 5에서 조기 종료했다. best checkpoint는 epoch 4, last는 epoch 5다. test 추론은
하지 않았다. 출력 경로:
`/workspace/komap_model1/outputs/model281_effb4_4x_unweighted/`

## 모델 282: EfficientNet-B4 U-Net 4배 고해상도, 차등 LR + equal-weight CE+Dice (조기 종료, 미채택)

모델281에서 확인한 class-weight 효과 미미함을 반영해 같은 seed `280`·fresh ImageNet B4·4배
1536px patch·uniform-area sampling·metallography augmentation·equal CE+Dice를 유지한다. 변경점은
팀의 optimizer 문서에서 참고한 encoder LR `1e-4`, decoder LR `3e-4`의 차등 학습률이다. 400-update
warm-up + cosine으로 총 7,840 updates 학습하고 4 epoch마다 D4 검증한다. epoch 1 train loss는
`1.1286`이며, 같은 seed의 단일-LR 무가중치 model281 epoch 1 (`1.5409`)보다 빠르게 낮아졌다.
epoch 4/8/12 train loss는 `0.5057` / `0.4480` / `0.4277`, image-wise mIoU는
`67.6885%` / `69.5036%` / `69.8573%` (epoch 12 global `73.5799%`)다. epoch 12 global class
IoU는 Al3Ni `70.9195%`, Primary Si `83.3019%`, Al `91.9021%`, Eutectic Si `48.1960%`다.
model280 epoch 12보다 `-0.8190%p` 낮고, 최고 route model225보다 `-8.8348%p` 낮다. 12 epoch
(3,360/7,840 updates)에서 회복 폭이 작고 경쟁력이 없어 조기 종료했다. best/last checkpoint는
epoch 12이며 test 추론은 생략했다. 출력 경로:
`/workspace/komap_model1/outputs/model282_effb4_4x_diff_lr/`

## 모델 283: model257 full-stat 체크포인트 + 이미지별 Lovasz-Softmax 미세조정 (route 미채택)

model257의 epoch 4 체크포인트에서 시작해, 1배/1024px patch·이미지당 8 crops·batch 2·full-image
통계 입력·seed 257·encoder/decoder/native 학습률 `5e-6 / 1.5e-5 / 1e-4`를 유지한다.
weighted CE + Dice에 이미지별 정답 존재 클래스만 평균하는 Lovasz-Softmax(`weight=0.25`)를
추가해 대회 image-wise mIoU 목표와 직접 정렬한다. 나머지 구성은 model257과 맞춰 손실 변경 효과를
분리한다. valid 전체에 D4 TTA/overlap 추론을 적용하고 `image_mean_miou` 최고 epoch checkpoint를
선택한다. 현재 train/valid만 사용하며 test 추론은 하지 않았다.

검증 image-wise/global은 epoch 1 `77.3217/79.8848%`, epoch 2 `77.2833/80.2256%`,
epoch 3 `77.2471/80.0534%`, epoch 4 `77.3348/80.0853%`였다. 최고 epoch 4는 model257
standalone `77.2071%`보다 `+0.1277%p`지만 model225 route보다는 낮다.

36개 class-wise route 후보 중 valid 최고는 Primary/Eutectic에 model283 확률을 각각 60%,
Al3Ni는 기존 20-step TENT(model220 config 6)를 40% 섞은 설정으로 image-wise `78.7818%`
(global `81.4019%`)였다. model225의 `78.6921%`보다 raw `+0.0897%p`지만, 19개 시편
leave-one-group-out 평균 delta는 `-0.0121%p`, positive image `11/20`, group bootstrap 95% CI
`[-0.1867, +0.1374]%p`로 0을 포함해 validation 탐색 과적합 가능성이 크므로 route에 반영하지
않는다. route reference는 model225의 `78.6921%`를 정확히 재현했다. 초기 점검용 5-step TENT
입력 probe는 기준 재현에 실패해 유효 결과로 사용하지 않는다.

최고 standalone checkpoint: `/workspace/komap_model1/outputs/model283_model257_imagewise_lovasz/best.pt`
확률/route 결과: `/workspace/komap_model1/outputs/model283_model257_imagewise_lovasz/`
test 추론은 하지 않았다.

## 모델 284: model257 이미지별 Lovasz-Softmax 가중치 0.5 (route 미채택)

모델283과 동일하게 model257 epoch 4에서 초기화하고 seed·입력·crop·optimizer·D4 검증을 고정했으며,
Lovasz 가중치만 `0.25`에서 `0.5`로 높였다. epoch별 image-wise/global mIoU는
`77.3876/79.9540`, `77.3341/80.2805`, `77.3231/80.1526`, `77.3905/80.1543%`로
best epoch 4 standalone은 `77.3905%`다. 모델283의 `77.3348%`보다 `+0.0557%p`지만,
최고 route model225 `78.6921%`보다 낮다.

model225 확률을 정확히 재현하는 입력으로 36개 class-wise 혼합 후보를 비교한 valid 최고는
Primary/Eutectic에 model284를 각각 60%, Al3Ni에 20-step TENT(model220 config 6)를 40% 섞어
`78.7928%`(global `81.4157%`)였다. model225보다 raw `+0.1007%p`이나, 19개 시편
leave-one-group-out 평균 delta `-0.0082%p`, positive image `11/20`, group bootstrap 95% CI
`[-0.1836, +0.1406]%p`로 0을 포함해 채택하지 않는다.

체크포인트·확률·CV 결과: `/workspace/komap_model1/outputs/model284_model257_imagewise_lovasz_w050/`
test 추론은 하지 않았다.

## 모델 285: 이미지별 present-class macro CE + Dice + Lovasz (조기 종료, 미채택)

model257 epoch 4 체크포인트에서 시작해 scale 1·1024 patch·8 crops/image·batch 2·full-image
통계·seed 257·학습률·4 epoch을 고정한다. 변경점은 weighted pixel CE를 각 이미지 내 정답이 존재하는
클래스별 평균 CE로 대체하는 것이며, Dice 및 per-image Lovasz(`weight=0.25`)는 유지한다.
이는 이미지마다 네 클래스 IoU를 동일 비중으로 평균하는 평가 기준에 CE도 정렬해 보는 실험이다.
서버 CUDA smoke test에서 결측 클래스가 있는 입력에 대해 손실 및 gradient 유한성을 확인했다.

epoch 1/2 image-wise mIoU는 `76.2252%` / `75.7108%` (global `78.5956%` / `78.2334%`)로
하락했다. weighted CE 기반 model257 `77.2071%`보다 크게 낮고, train loss는 감소하지만 valid가
악화되어 epoch 2에서 조기 종료했다. best epoch 1 checkpoint 및 두 epoch의 history는 보존했으며,
성능이 기준보다 낮아 valid probability export와 route probe는 수행하지 않았다.

출력 경로: `/workspace/komap_model1/outputs/model285_model257_macro_ce/`

## 모델 286: model257 + 이미지별 present-class Soft-IoU 미세조정 (route 미채택)

model257에서 시작해 scale 1·1024 patch·8 crops/image·batch 2·full-image 통계·seed 257·학습률·
4 epoch을 유지한다. weighted CE + Dice 기반을 보존하면서 이미지별 정답 존재 클래스 Soft-IoU
(`weight=0.5`)를 추가한다. 이전 Soft-IoU 시도는 더 오래된 backbone checkpoint에서 수행됐으므로,
이번에는 현재 강한 full-stat model257 초기화를 사용해 재검증한다.

epoch별 image-wise/global mIoU는 `77.3902/79.9529`, `77.3441/80.2906`, `77.3432/80.1827`,
`77.3950/80.1641%`였으며 best epoch 4 단독은 `77.3950%`로 model257 `77.2071%` 대비
`+0.1878%p` 개선됐다.

model225 route와 정확히 비교한 36개 class-wise 혼합 후보 최고는 Primary/Eutectic에 model286을
각각 60%, Al3Ni에 model220 config 6 TENT를 40% 섞은 `78.8001%`(global `81.4254%`)였다.
model225보다 raw `+0.1080%p`지만 19개 시편 grouped-CV 평균 변화 `-0.0055%p`, positive image
`10/20`, group bootstrap 95% CI `[-0.1831, +0.1476]%p`로 0을 포함해 채택하지 않았다.
route reference는 model225 `78.6921%`를 재현했다. test 추론은 수행하지 않았다.

체크포인트·확률·CV 결과: `/workspace/komap_model1/outputs/model286_model257_imagewise_softiou/`

## 모델 287: model225 component-label / foreground-shape oracle 진단 (완료)

새 학습 전에 region-level class 할당과 Al-vs-secondary 경계 중 어느 쪽에 점수 회복 여지가
있는지 model225 exact route로 검증했다. 예측된 각 secondary class를 따로 4/8-connectivity
component로 나눠 GT majority label을 주는 class-component oracle은 `84.3379/83.9980%`로
baseline `78.6921%`보다 `+5.6458/+5.3060%p`였다. 4-connected 개선의 specimen-group bootstrap
95% CI는 `[+4.6198, +6.6903]%p`다.

GT의 secondary-vs-Al binary shape만 사용하고 phase class는 현 예측을 유지(Al로 놓친 GT secondary만
route 확률로 재분류)한 foreground-shape oracle은 `84.3188%`, delta `+5.6267%p`
(group bootstrap 95% CI `[+4.9819, +6.2686]%p`)였다. 따라서 경계/foreground 분할과 region-level
phase 재분류 양쪽 모두 큰 이론적 headroom이 있지만, 이는 GT 사용 oracle이므로 실제 모델 성능이
아니다. secondary 전체를 하나의 연결 component로 뭉친 naive oracle은 components가 상들을 가로질러
합쳐져 `60.5830%`로 악화했으며 유효한 region oracle로 해석하지 않는다.

이 진단을 근거로 다음은 component-level classifier를 specimen-group CV에서 먼저 시험한다. 통계
결과와 image별 oracle 기록은 다음에 보존했다.

결과: `/workspace/komap_model1/outputs/model287_region_oracle_diagnostic/results_classwise.json`

## 모델 288: model225 region classifier leave-one-specimen-out probe (미채택)

model225 exact route의 class별 connected components에서 예측 확률, 크기/형상, intensity 통계를
추출하고, component GT majority class를 target으로 학습하는 tabular classifier를 19개 specimen
group leave-one-out으로 평가한다. 이 단계는 valid 내부 grouped-CV 진단이며, 결과가 유망하면 train
OOF component를 사용한 별도 학습으로 넘어간다. GT는 각 fold의 classifier 학습에만 쓰고 held-out
specimen에는 사용하지 않는다.

출력 경로: `/workspace/komap_model1/outputs/model288_region_classifier_cv/`

19개 specimen LOSO에서 confidence threshold `0.0/0.60/0.75/0.90`별 image-wise 점수는
`77.5106/78.0725/78.2714/78.6086%`였다. 최고 threshold `0.90`도 model225 `78.6921%`보다
`-0.0835%p`(positive image `1/20`, group bootstrap 95% CI `[-0.2168, +0.0004]%p`) 낮았다.
낮은 threshold일수록 components를 과도하게 재분류해 하락했다. region majority oracle의 큰 headroom을
이 32-feature MLP가 회수하지 못했으므로 이 classifier는 채택하지 않는다.

## 모델 289: 이차상-vs-Al binary boundary loss 미세조정 (route 미채택)

model257 full-stat checkpoint에서 시작해 CE+Dice를 유지하고, 네 logits를 이차상(Al3Ni/Primary/
Eutectic) 대 Al의 binary logit으로 묶은 balanced BCE 및 per-image binary Soft-IoU를 추가한다.
BCE는 GT 경계 주변 3px band에 가중치를 높여 foreground-shape oracle에서 확인한 경계 headroom을
직접 겨냥한다. seed·scale·patch·optimizer는 model257 기준을 유지한다.

출력 경로: `/workspace/komap_model1/outputs/model289_model257_binary_boundary/`

epoch별 image-wise/global mIoU는 `77.4184/80.0075`, `77.2902/80.2992`, `77.2602/80.0908`,
`77.3867/80.1631%`였다. best epoch 1 standalone `77.4184%`는 model257보다 `+0.2113%p`지만,
후속 epoch에서 유지되지 않았다.

model225와 동일한 TENT input을 쓰는 36개 class-wise route 후보의 valid 최고는 Eutectic에
model289를 80% 섞고 Primary/Al3Ni를 model225 설정에 둔 `78.7608%`(global `81.4032%`)였다.
model225보다 raw `+0.0686%p`지만 grouped-CV delta `-0.0709%p`, positive image `14/20`,
group bootstrap 95% CI `[-0.3227, +0.1451]%p`로 0을 포함해 채택하지 않는다. 기준 route 재현은
`78.6921%`로 일치했다. 테스트셋 추론은 하지 않았다.

체크포인트·확률·CV 결과: `/workspace/komap_model1/outputs/model289_model257_binary_boundary/`

## 모델 290: saved prediction scale-oracle 대 1x null-oracle (완료, 미채택)

이미 저장된 같은 ResNet34 계열 1x/2x/3x valid 확률을 이용해 이미지별 GT-best-scale oracle을
구하고, 후보 수를 동일하게 맞춘 여섯 1x checkpoint/loss-variant oracle과 비교한다. 이 oracle은
배율 선택을 실제 구현하는 모델이 아니라, 이미지마다 다른 scale을 고르는 데 유효 headroom이
있는지의 상한 진단이다. 양쪽 비교 모두 valid GT를 oracle 선택에 사용하므로 그 점수는 채택 점수가
아니다.

6개 후보씩 비교한 scale oracle 평균은 `77.6424%`, 1x null oracle 평균은 `77.6972%`였다.
각 그룹의 최고 단일 후보 대비 oracle 선택 이득은 scale `+0.4349%p`, null `+0.3020%p`로,
scale의 추가 이득은 null보다 `-0.0548%p`(specimen-group bootstrap 95% CI
`[-0.2730, +0.1759]%p`)였다. 후보 수 선택 편향을 넘는 image-specific scale headroom은 확인되지 않아
per-image scale router나 추가 high-resolution 학습은 진행하지 않는다.

결과: `/workspace/komap_model1/outputs/model290_scale_oracle_vs_null/results.json`

## 모델 291: 이진 foreground + 조건부 상(phase) 계층형 head (미채택)

model287의 foreground-shape oracle이 보여준 Al/이차상 경계 headroom과 class-component oracle의
상 판별 headroom을 구조적으로 분리해 학습한다. model257에서 시작해 기존 4-class 분류 head를
보조 loss로 보존하고, 공용 decoder feature에 (1) 이차상-vs-Al 이진 head 및 (2) 이차상 픽셀에서만
Al3Ni/Primary/Eutectic을 판별하는 조건부 3-class head를 추가한다. 추론 확률은
`P(class)=P(secondary/Al) × P(phase | secondary)`로 결합한다. 새 head는 model257의 4-class
classifier weight에서 warm-start한다.

train split 70장만 사용하고 model257과 같은 1×·1024 patch·8 crops/image·full-image 통계·D4
검증 조건으로 4 epoch 미세조정했다. epoch별 image-wise mIoU는 `77.1369/77.0169/77.3023/77.2208%`
이며 최고 epoch3 `77.3023%`(global `80.2475%`)로 model257 단독 `77.2071%`보다 `+0.0952%p`
였다.

model225 route를 기준으로 model291을 기존 TENT branch와 class-wise 혼합하는 15개 사전 정의 후보를
평가했다. valid raw 최고 `78.7094%`는 기준 `78.6921%`보다 `+0.0173%p`에 불과했다. specimen-group
LOSO에서 fold별 route 선택 후 held-out image 평균 변화는 `-0.1585%p`, positive image `6/20`,
group bootstrap 95% CI `[-0.3285, -0.0168]%p`로 악화했다. 따라서 이 분기는 route에 채택하지 않고
model225를 최고로 유지한다. valid probability 및 그룹 CV 결과는 아래에 보존했다.

출력: `/workspace/komap_model1/outputs/model291_hierarchical_heads/`
구현: `model291_hierarchical_heads.py`, route 탐색은 `model258_fullstats_route_probe.py`

## 모델 292: 교차-시편 phase-component copy-paste 증강 (미채택)

model291 계층 head는 독립 branch 기준으로만 소폭 개선되고 route CV에서는 미채택됐다. 다음은
구조/loss 재조정 대신 train 시편 다양성을 늘리는 데이터 증강을 시험한다. 서로 다른 시편 그룹의
1024 patch batch pair 사이에서 4-connected Al3Ni/Primary/Eutectic component를 원래 크기로 옮겨
붙이고, 경계 안쪽 5px에서 feature를 feather blend한다. GT mask에는 실제로 붙인 component 영역만
donor phase로 반영한다. 증강은 train split에만 적용하고 valid에는 쓰지 않는다.

model257 초기화, 1×·1024 patch·8 crops/image·full-image 통계·D4 검증 및 동일 optimizer 설정으로
4 epoch 학습했다. epoch별 image-wise mIoU는 `76.8791/77.0478/77.0258/77.0655%`였고, 최고 epoch4
`77.0655%`(global `79.8358%`)는 model257 단독 `77.2071%`보다 `-0.1416%p` 낮았다. 매 epoch
214~235개 component가 붙었지만 검증 향상으로 이어지지 않았다. 기준 branch보다 낮아 model225 route
CV는 수행하지 않고 미채택한다. 로컬/서버 pycompile 및 synthetic tensor로 양방향 cross-specimen
paste smoke test를 통과했으며, 검증 probability도 저장했다.

출력: `/workspace/komap_model1/outputs/model292_cross_specimen_copy/`
구현: `model292_cross_specimen_component_copy.py`

## 모델 293: train-fitted region classifier (미채택)

모델288에서는 valid 20장의 component로 LOSO classifier를 학습해 표본 수가 작았다. 이번에는
model257 checkpoint를 train 70장에 추론해 class별 4-connected secondary component를 생성하고,
각 component의 majority GT class 및 영역/확률/강도 통계로 MLP를 train split에서 학습한다. region
교정은 model257의 예측 component 안에서만 수행한다.

주의: train 예측은 해당 모델이 학습한 이미지의 in-sample 출력이다. valid 내 train/valid 시편 ID
중복도 별도 표시해 해석한다. train 70장에서 180k 이상 predicted secondary components를 추출해
MLP를 학습하고 valid 20장에 적용했다. threshold `0.75/0.85/0.90/0.95`의 image-wise 점수는
`77.2383/77.2221/77.1958/77.1958%`였다. 저장 확률 fp16에 따른 model257 baseline 재현은
`77.2075%`(기록된 full precision summary `77.2071%`)였다. 가장 좋은 raw 개선 `+0.0308%p`는
positive image `6/20`뿐이다.

valid specimen-group threshold 선택 CV는 `+0.0087%p`, positive `5/20`, 95% CI
`[-0.0591, +0.1102]%p`였다. train과 A-ID가 겹치지 않는 valid 13장에서는 `+0.0457%p`, 겹치는
7장에서는 `-0.0599%p`였다. 더 많은 train component로 학습해도 oracle headroom을 회수하지 못했고,
model225 route 결합으로 이어갈 근거가 부족해 미채택한다.

구현: `model293_train_fitted_region_corrector.py`
출력: `/workspace/komap_model1/outputs/model293_train_region_corrector/`

## 모델 294: decoder hidden feature mask-pooled region probe (완료, 미채택)

모델287의 component-label oracle과 모델293의 handcrafted region feature 정체를 바탕으로,
model257을 고정한 채 decoder의 마지막 32-channel hidden map을 각 train GT component 안에서
mask-average/std pooling한다. 기존 확률·형태·밝기 component feature 32개와 hidden pooled feature
64개를 이어 3-class region MLP를 train GT component에서 학습하고, valid의 predicted component에
적용한다. 전체 segmentation network는 고정하며 이는 값싼 go/no-go probe다.

train inference는 full-stat 입력·1x·Gaussian tile overlap을 사용하고 valid segmentation base는
model257의 공식 D4 확률 cache를 그대로 쓴다. Threshold는 specimen-group held-out으로 보고하며,
train-valid A-ID 중복/비중복 그룹 성능도 분리한다. group-CV 이득이 작거나 비중복 시편에서 악화하면
full region-head fine-tuning으로 확장하지 않는다.

학습 component 180,865개, feature 96개. 고정 threshold `0.75/0.85/0.90/0.95`에서
image-wise mIoU는 각각 `77.1366/77.1047/77.0876/77.1110%`로, fp16 확률 기준 model257
baseline `77.2075%`보다 모두 낮았다. Specimen-group held-out threshold 선택도 `77.0531%`, 즉
`-0.1544%p`(positive image `8/20`, group bootstrap 95% CI `[-0.3355, +0.0037]%p`)였다.
시편 비중복 13장 delta `-0.0839%p`, 중복 7장 delta `-0.2855%p`로 양쪽 모두 악화했다.
따라서 frozen learned decoder region head는 확장하지 않고 미채택한다.

구현: `model294_decoder_region_probe.py`
출력 경로: `/workspace/komap_model1/outputs/model294_decoder_region_probe/`

## 모델 295: 이미지 내부 고신뢰 component anchor 진단 (완료, 미채택)

학습형 region 교정기(모델293·294)가 oracle headroom을 회수하지 못한 원인을 좁히기 위해,
학습 없이 같은 이미지 안의 고신뢰 component끼리 비교하는 고정 규칙을 평가한다. 4-connected
secondary component 중 예측 상 평균 확률이 0.90 이상이고 면적이 이미지 component 면적 중앙값 및
32px 이상인 component만 anchor로 삼는다. 위치를 제외하고 확률·불확실성·면적·형태·z/q 밝기/질감
특성을 이미지 내부 robust 표준화한 뒤, anchor 5개 중 4개 이상이 같은 다른 상을 지지할 때만
비-anchor component를 재분류한다. 규칙은 valid 정답을 보지 않고 고정하며, 정답은 제안의 정밀도와
대회식 image-wise mIoU를 사후 진단하는 데만 쓴다. route 학습/valid 튜닝은 하지 않는다.

기준 Model225 route 재현은 `78.6921%`로 일치했다. 고정 교정 후 valid image-wise mIoU는
`70.9398%`(`-7.7523%p`, positive image `3/20`, specimen-group bootstrap 95% CI
`[-12.7669, -3.1122]%p`)로 크게 하락했다. 83,242개 component 중 잘못 예측된 component의
24.20%만 교정됐고, 그 면적 기준 교정률은 14.37%였다. 반면 원래 맞던 component의 35.04%가
오염됐으며, 정답 component 면적의 5.03%가 잘못 바뀌었다. 이미지 내 feature 유사도 투표는 phase
정체성을 보존하지 못하므로 이 계열의 label-free post-hoc 교정은 중단한다.

구현: `model295_within_image_anchor_diagnostic.py`
출력 경로: `/workspace/komap_model1/outputs/model295_within_image_anchor/`

## 모델 296: region 학습용 train in-sample 예측 편향 진단 (완료)

모델293의 MLP는 model257이 학습에 사용한 train 이미지의 예측 component로 학습됐다. 이 예측이
valid보다 지나치게 쉬우면 region classifier가 train에서만 잘못된 분포를 배운다. model257과 동일한
1x·1024 patch·576 stride·D4 조건으로 train 70장을 다시 추론하고, saved valid 확률과 4-connected
secondary component의 GT-majority 오류율, 면적 가중 오류율, confidence·entropy·margin 분포를
비교한다. valid 시편 중 train A-ID 중복/비중복도 따로 본다. 이 진단은 정답으로 모델을 튜닝하지 않으며,
큰 train-valid gap이 확인될 때만 OOF region 학습 비용을 검토한다.

train in-sample component 오류율은 개수 기준 `27.32%`, 면적 가중 `2.99%`였으나, valid는 각각
`37.10%`, `7.12%`였다. valid 평균 confidence는 train `0.675` 대비 `0.658`, entropy는 `0.747` 대비
`0.769`로 valid에서 더 어렵고 불확실했다. valid 중 train A-ID 비중복 13장은 component 오류율
`31.73%`(면적 가중 `6.52%`), 중복 7장은 `43.59%`(`7.64%`)였다. 특히 비중복 valid에서도
in-sample train보다 오류율이 `4.41%p`, 면적 오류가 `3.54%p` 높아 단순 시편 중복만의 효과는 아니다.
모델293 MLP 학습 입력의 오류율은 valid보다 낮고 면적 기준으로 2.4배 차이가 나므로, OOF component
예측으로 region classifier를 재평가할 근거가 있다. 이 결과만으로 성능 향상을 주장하지 않는다.

기존 `model96_4fold_ensemble` 체크포인트는 재사용할 수 없다. 해당 legacy fold runner는 train과 valid를
합친 90장으로 fold를 나누고, RGB 입력의 구형 model6을 학습했다. 이는 train 70장만으로 만든
model257의 누수 없는 OOF 예측도 아니고, 현재 입력/모델 branch와도 다르므로 새 train-only fold 모델이
필요하다.

구현: `model296_region_train_valid_shift_audit.py`
출력 경로: `/workspace/komap_model1/outputs/model296_region_shift_audit/`

## 모델 297: train-only 시편 그룹 OOF segmentation branch (완료, 단독 미채택)

모델293 region corrector의 in-sample 편향을 평가하기 위해 train 70장만 4개 A-ID 그룹 fold로 나눈다.
각 fold 모델은 ImageNet ResNet34 초기화만 사용하고 held-out 시편의 mask를 보지 않은 채 8 epoch
학습한다. Model257과 같은 raw/z/CDF full-image 통계 경로를 쓰되, 계산량을 제한한 512px patch로
학습·D4 Gaussian overlap 추론한다.
각 train 이미지의 held-out 확률을 OOF cache에 저장하고, 같은 fold 모델들의 valid 확률 평균도 만든다.
valid mask는 학습/epoch selection에 전혀 쓰지 않는다. 먼저 OOF 및 valid 단독 점수를 확인한 뒤
region corrector나 model225 route 혼합으로 진행할지 결정한다.

네 fold의 held-out image mIoU는 `75.3545%` (18장), `76.2916%` (18장), `81.0904%` (17장),
`76.3342%` (17장)이다. fold image 수를 반영한 전체 OOF 평균은 `77.2264%`; fold 단순평균은
사용하지 않는다. 네 fold valid probability ensemble은 image-wise `74.2546%`로, model225
route보다 낮아 단독 채택하지 않는다. OOF 수치는 train 시편 그룹 교차검증 결과라 valid 최고점과
직접 비교하지 않는다. validation mask는 학습·epoch selection·fold 결정에 쓰지 않았다.

구현: `model297_train_only_oof_branch.py`
출력 경로: `/workspace/komap_model1/outputs/model297_train_only_oof/`

## 모델 298: train-only OOF region corrector (완료, 개선 미미·미채택)

모델297의 held-out train 확률로 component MLP를 학습하고, specimen-group meta-CV로 confidence
threshold `0.60`을 선택했다. 전체 OOF component로 재학습해 모델297 valid fold-ensemble에 적용한
결과 valid image-wise mIoU는 `74.2546%`에서 `74.2897%`로 `+0.0351%p` 변했다(positive 9/20,
specimen-group bootstrap 95% CI `[-0.0848, +0.1831]%p`). Train meta-CV 변화도 `+0.0354%p`
(95% CI `[-0.0416, +0.1290]%p`)로 0을 포함한다. valid 중 train과 A-ID가 겹치지 않는 13장의
변화는 `+0.0403%p`, 겹치는 7장은 `+0.0253%p`다. 개선 근거가 불확실해 단독 branch로 미채택하고,
model225 route에 대한 소량 class-wise 혼합만 specimen-group held-out으로 추가 확인한다.

구현: `model298_oof_region_corrector.py`
출력 경로: `/workspace/komap_model1/outputs/model298_oof_region_corrector/`

## 모델 299: Model298 OOF 보정 branch와 Model225 route 결합 (완료, 미채택)

Model225의 Primary·Al3Ni 설정을 고정하고, Model298 corrected probability를 Al3Ni/Primary/
Eutectic 부분집합에 `5/10/20/30%` 섞은 28개 exact 후보를 비교했다. evaluator에서 Model225 기준
image-wise `78.6921%` 재현을 확인했다. in-sample 최고는 Al3Ni에만 Model298을 5% 섞은
`78.6077%`로 기준보다 `-0.0844%p` 낮았다. 19개 specimen-group leave-one-out 선택은 전 fold에서
Model225 기준을 골랐고 held-out delta `0.0000%p`, positive image `0/20`, group bootstrap 95% CI
`[0, 0]%p`였다. 따라서 Model298은 현재 route에 섞지 않는다.

구현: `model258_fullstats_route_probe.py --source-id 298`
결과: `/workspace/komap_model1/outputs/model299_model298_route_probe/`

## 모델 300: 회전 불확실도 가중 Mean-Teacher consistency (완료, 미채택)

모델297의 각 fold checkpoint를 초기값으로 1 epoch fine-tune했다. held-out train fold의 입력만
image-only dataset으로 제공하고, 해당 mask와 valid mask는 학습 중 열지 않았다. EMA teacher의
0/90/180/270도 예측을 역회전해 평균하고, view 간 KL divergence에 따른 `exp(-4 × KL)` confidence로
student consistency MSE를 가중했다. `lambda=0.5` fold1 pilot과 gradient 기여를 키운 `lambda=5`
네-fold 실험을 수행했다.

lambda=5의 stored OOF 기준 image-wise mIoU는 Model297 `77.2266%`에서 `77.2124%`로
`-0.0142%p` 변했다(positive 32/70, specimen-group bootstrap 95% CI
`[-0.0756, +0.0447]%p`). fold별 OOF 변화는 `-0.0036, +0.0671, +0.0112, -0.1373%p`로
시편에 따라 일관되지 않았다. valid 4-fold probability ensemble은 Model297의 `74.2546%`에서
`74.2646%`로 `+0.0100%p`였으나 positive 12/20, group bootstrap 95% CI
`[-0.0420, +0.0566]%p`로 0을 포함한다. 따라서 이 consistency 설정은 성능 개선 근거가 없어
미채택한다.

구현: `model300_rotation_consistency_pilot.py`
결과: `/workspace/komap_model1/outputs/model300_rotation_consistency_pilot/aggregate_lambda5.json`

## 모델 301: Model257 boundary-enhanced U-Net++ (구현만 완료, 미학습·대체됨)

Al-Si 미세조직 분할 논문의 Boundary Enhancement Module에서 형태학적 경계 추출과
다중 dilation 문맥 결합 아이디어를 가져와 Model257의 fine-resolution U-Net++ decoder
stage `x0_1/x0_2/x0_3`에 공유 적용한다. dilation rate는 `1/3/5/7`이고, residual projection은
zero-init해 기존 checkpoint 예측을 보존한다. Model257 epoch 4 checkpoint를 실제로 불러온 CPU
검증에서 초기 logits 최대 차이 `0.0`을 확인했다. 학습은 scale 1, patch 1024, 8 crops/image,
batch 2, weighted CE+Dice, 4 epoch 설정으로 준비했다. [RUCS-BE-MT 논문](https://www.sciencedirect.com/org/science/article/pii/S1546221826001864)

Model257 checkpoint는 모든 train 시편으로 학습된 가중치이고, 원 구현은 valid mIoU로 epoch을
선택한다. 따라서 이 checkpoint로 grouped OOF 성능을 주장할 수 없으며, 반복적인 valid 선택에 따른
낙관 편향도 우려된다. 이 방식의 학습은 시작하지 않았고, held-out train groups를 지킨 Model297
fold checkpoint에서 이어 학습하는 Model302로 대체한다. 아래는 구현·parity 확인뿐이며 성능 개선을
의미하지 않는다.

구현: `model301_boundary_enhanced_finetune.py`
기준 checkpoint: `/workspace/komap_model1/outputs/model257_full_image_stats_1x/best.pt`

## 모델 302: Model297 train-only grouped OOF + boundary enhancement (P100 완료, 미채택)

Model297의 네 specimen-group OOF checkpoint를 각각 그대로 이어 받아, 같은 fold의 train groups에만
Model301 boundary enhancement를 4 epoch 미세조정한다. 각 checkpoint가 보지 않은 held-out groups로
OOF 예측을 만들고, fold별 검증 checkpoint 선택은 하지 않고 고정된 마지막 epoch을 쓴다. valid masks는
네 fold 학습·예측이 모두 끝난 뒤 최종 측정에만 열어, validation 기반 epoch 선택 누수를 막는다.

비교의 주 지표는 Model297 대비 paired train-OOF image-wise mIoU 변화와 specimen-group bootstrap
95% CI다. valid 4-fold probability ensemble 변화는 별도 참고값으로 보고 OOF 근거 없이 채택하지 않는다.
Model297 fold1 checkpoint로 실제 state-dict 호환을 확인했고, baseline 대비 초기 logits 최대 차이 `0.0`을
확인했다. 로컬·P100 원격 py_compile 및 CPU smoke test를 통과했고, 로컬 Mac MPS에서도 batch 2,
128×128 forward/backward와 65% allocator cap 설정을 smoke-test해 finite logits/gradient를 확인했다.
사용자가 P100 실행을 선택했다. 첫 GPU0 시도는 65% 메모리 한도(`10.33GiB`)에서 약 `10.56GiB` 할당
중 OOM으로 첫 epoch 전 종료됐으며, 부분 폴더를 `model302_boundary_oof_gpu0_oom/`에 보존했다.
이후 GPU6에 고정하고 메모리 한도를 90%로 높여 네 fold 전체를 완료했다. 1024 patch, 8 crops/image,
batch 2, 4 epochs/fold이며 fold별 학습 loss는 아래와 같다.

| Fold | held-out 이미지 | Epoch train loss (1→4) | 경과시간 |
|---|---:|---|---:|
| 1 | 18 | 0.48423 → 0.47072 → 0.45742 → 0.45132 | 938.5초 |
| 2 | 18 | 0.49507 → 0.47446 → 0.46279 → 0.45736 | 901.9초 |
| 3 | 17 | 0.52612 → 0.50776 → 0.49577 → 0.49180 | 912.0초 |
| 4 | 17 | 0.47127 → 0.46218 → 0.44861 → 0.44421 | 914.7초 |

최종 train-only grouped OOF image-wise mIoU는 Model297 `77.2266%`에서 Model302 `77.4562%`로
`+0.2296%p`였다(positive images `41/70`, specimen-group bootstrap 95% CI
`[-0.3177, +0.7440]%p`). 신뢰구간이 0을 포함해 개선은 불확실하다. 20장 valid 4-fold probability
ensemble은 `74.2546%`에서 `74.9864%`로 `+0.7318%p`였지만, 이는 보조 결과로만 본다. 기존 최고
valid score `78.7209%`를 넘지 못했으므로 Model302는 최고 모델로 채택하지 않는다. valid masks는
네 fold의 학습과 예측이 모두 끝난 뒤 점수 계산에만 사용했다.

실행기는 `auto/cuda/mps/cpu` 장치를 지원한다.

기존 Model287 valid oracle의 image별 진단에서 A71은 기준 `46.64%`/class-component oracle `55.71%`로
oracle headroom이 상대적으로 작다. A153(`72.71→81.88%` shape oracle), A74(`73.97→80.57%`),
A11 step2(`74.00→81.98%`), A67(`74.46→81.53%`)에는 더 큰 shape/class-region 회복 여지가 있다.
이는 valid GT를 쓴 비실현 상한일 뿐이며 학습이나 route 선택에 사용하지 않는다. 한 장만 크게 고쳐
80%를 기대하기보다 여러 hard image에서 paired OOF 개선이 일관적인지 본다.

구현: `model302_boundary_oof_finetune.py`
기준 체크포인트: `/workspace/komap_model1/outputs/model297_train_only_oof/fold{1..4}/last.pt`
기준 OOF: `/workspace/komap_model1/outputs/model297_train_only_oof/train_oof_probs.npz`
결과: `/workspace/komap_model1/outputs/model302_boundary_oof/results.json`
fold 로그: `/workspace/komap_model1/outputs/model302_boundary_oof_gpu6.log`

## 모델 303: Model302 low-LR grouped OOF continuation (P100 완료, 보완성 평가 중)

Model302 네 fold의 마지막 checkpoint에서 각 fold별 4 epoch을 추가 미세조정한다(누적 8 epoch).
encoder/decoder/native/boundary 학습률은 Model302 설정의 1/4로 낮추고, 같은 held-out specimen groups와
같은 train OOF 이미지에서 Model302와 paired 비교한다. valid는 최종 4-fold ensemble 측정에만 사용하며
학습·epoch 선택에는 사용하지 않는다. 목적은 Model302의 네 fold 모두 마지막 epoch까지 train loss가
계속 내려간 데서 추가 학습 여지를 확인하는 것이다. 긍정 결과가 있어도 grouped OOF와 group bootstrap
95% CI를 기준으로 채택 여부를 판단한다.

P100 GPU6, 1024 patch, 8 crops/image, batch 2, 90% memory cap으로 네 fold를 완료했다. fold1/2/3/4의
추가 4 epoch train loss는 `0.44789→0.44439`, `0.45261→0.44964`, `0.48468→0.48421`,
`0.43368→0.43711`이었다. train-only grouped OOF image-wise mIoU는 Model302 `77.4562%`에서
Model303 `77.5096%`로 `+0.0533%p` 변했고 70장 중 46장이 개선됐다. specimen-group bootstrap 95% CI는
`[+0.0155, +0.0909]%p`로 양수지만 개선폭은 작다. valid 4-fold ensemble은 `75.0472%`로
Model302 대비 `+0.0608%p`였다. 현재 최고 route `78.7209%`에는 미치지 못해 단독 채택하지 않는다.
Model304에서 이 branch를 Model225 route에 class-wise 5–50% 혼합한 43개 exact 후보로 평가했다.
기준 route는 `78.6921%`로 재현됐고 in-sample 최고도 기준 route 그대로였다. 19-group leave-one-out은
모든 fold에서 Model225 기준을 선택해 held-out delta `0.0000%p`(positive `0/20`, bootstrap CI
`[0,0]%p`)였다. Model303 branch는 현재 route를 보완하지 못해 미채택한다.

구현: `model302_boundary_oof_finetune.py` (`--model-id`/baseline ID/LR multiplier 추가)
결과 경로: `/workspace/komap_model1/outputs/model303_boundary_oof_8ep/`
실행 로그: `/workspace/komap_model1/outputs/model303_boundary_oof_8ep_gpu6.log`

## 모델 304: Model303 boundary branch의 Model225 route 보완성 검증 (P100 완료, 미채택)

Model303 fold-ensemble 확률을 기존 최고 route인 Model225에 class-wise로 5–50% 혼합하는 43개
exact 후보를 검증했다. Model225 기준 image-wise mIoU `78.6921%`는 재현됐고, in-sample 최고도
기준 route 자체였다. 19개 specimen-group leave-one-out에서도 모든 fold가 Model225를 선택해 held-out
변화 `0.0000%p`, positive image `0/20`, group-bootstrap 95% CI `[0.0000, 0.0000]%p`였다.
Model303 boundary branch는 최고 route를 보완하지 못했다.

결과: `/workspace/komap_model1/outputs/model304_model303_route_probe/results.json`

## 모델 305: Model257 full-data checkpoint + boundary enhancement (완료, 미채택)

Model257의 전체 train 학습 checkpoint에 zero-init shared multi-dilation boundary module을 붙여 4 epoch
미세조정했다(1024 patch, 8 crops/image, batch 2, weighted CE+Dice). `--fixed-final-epoch` 모드로
훈련 중 valid mask 평가와 checkpoint 선택을 생략하고 마지막 epoch을 저장한 뒤에만 valid를 측정했다.
train loss는 `0.35199 → 0.34778 → 0.34061 → 0.33993`으로 감소했지만, valid image-wise mIoU는
Model257 `77.2075%`에서 `76.6797%`로 `-0.5278%p` 하락했다(9/20 images 개선). 클래스별 image-mean
IoU 변화는 Al3Ni `-1.1917%p`, Primary Si `-0.2739%p`, Al `-0.0173%p`, Eutectic Si `-0.6289%p`로
주요 두 Si 상이 악화했다. 단독 모델로 미채택하며, 확률 보완 여부만 Model306에서 확인한다.

결과: `/workspace/komap_model1/outputs/model305_model257_boundary_fixed4/summary.json`
체크포인트: `/workspace/komap_model1/outputs/model305_model257_boundary_fixed4/last.pt`
구현: `model301_boundary_enhanced_finetune.py --fixed-final-epoch`

## 모델 306: Model305 boundary branch의 Model225 route 보완성 검증 (P100 완료, 미채택)

Model305 전체 train 학습 branch를 기존 Model225 class-wise route의 후보 source로 추가하고 43개 exact
혼합·보정 후보를 평가했다. In-sample 최고는 Eutectic Si에 Model305 확률을 50% 혼합한 후보로
`78.6938%`였지만, 기준 `78.6921%`보다 겨우 `+0.0017%p`였다. 19개 specimen-group leave-one-out
결과는 `-0.2601%p`, positive `1/20`, bootstrap 95% CI `[-0.5508, -0.0483]%p`로 악화했다.
따라서 Model305 branch는 route에 포함하지 않는다.

결과: `/workspace/komap_model1/outputs/model306_model305_route_probe/results.json`

## 모델 307: 확률만 쓰는 image-wise expected-IoU decoder (P100 완료, 미채택)

Model303 train grouped-OOF 확률만으로 이미지별 class bias를 좌표 탐색했다. 각 후보는 저해상도 확률의
plug-in ratio-of-expectations mIoU를 최대화하고, Al은 고정 기준 클래스로 두었다. 모든 이미지의 bias를
확정한 뒤에만 train GT mask를 열어 채점했다. 따라서 mask를 통한 per-image bias 선택은 하지 않았다.

OOF image-wise mIoU가 `77.5096%`에서 `76.4843%`로 `-1.0252%p` 하락했고, positive image는 `10/70`,
specimen-group bootstrap 95% CI는 `[-1.3485, -0.6934]%p`였다. Eutectic Si class-mean IoU 변화가
`-2.4764%p`로 가장 컸다. 보조 측정인 공식 valid도 `75.0472%`에서 `73.7333%`로 `-1.3139%p`
하락했다. 확률만으로 선택한 기대-IoU surrogate가 실제 image-wise mIoU로 일반화하지 않아 미채택한다.
class 확률의 보정 오차와 픽셀 간 공간 상관을 surrogate가 반영하지 못한 탓일 수 있다(가능성에 대한 해석).

구현: `model307_expected_iou_decoder.py`
결과: `/workspace/komap_model1/outputs/model307_expected_iou_decoder/results.json`

## 모델 308: frozen DINOv2 다층 dense feature 잔차 branch (P100 완료, route 미채택)

과거 Model4의 frozen DINOv2-Small + 단순 저해상도 decoder(`42.3769%`)와 구별되도록, Model303의
specimen-group OOF CNN checkpoint는 동결하고 DINOv2-Small의 3/6/9/12번째 block patch feature를
작은 residual decoder로 변환해 CNN logits에 더했다. residual 마지막 층을 0 초기화해 시작 logits가
기준 CNN과 정확히 같음을 확인했다. 사전학습 DINOv2 patch 특징과 중간층 추출은 공식 구현을 사용했다.
[DINOv2 공식 코드·가중치는 Apache-2.0 라이선스](https://github.com/facebookresearch/dinov2#license).

네 fold 모두 Model303의 기록된 group split 및 checkpoint로 시작했고, valid mask는 모든 학습·예측과
checkpoint 결정을 끝낸 뒤에만 평가했다. baseline과 candidate 모두 512px patch, stride 384, Gaussian
overlap 및 D4 TTA를 동일하게 적용했다. 70장 train OOF image-wise mIoU는 같은 512px baseline
`76.7647%`에서 DINO branch `76.9273%`로 `+0.1626%p`였다(41/70 positive, specimen-group bootstrap
95% CI `[+0.0003, +0.3500]%p`). class-mean IoU 변화(Al3Ni, Primary Si, Al, Eutectic Si)는
`[-0.0450, +0.5099, +0.0939, +0.0917]%p`로 Primary Si 개선에 대부분 기댄다.

공식 valid 20장도 같은 512px 조건에서 `73.2189%` → `73.4223%` (`+0.2034%p`, positive 9/20)이지만
group-bootstrap 95% CI `[-0.1164, +0.5795]%p`는 0을 포함한다. valid class별 변화는
`[-0.1810, +0.2102, +0.1092, +0.6750]%p`로 OOF 양상과 다소 다르다. 512px 점수는 기존 1024px
Model303 결과나 최고 route `78.7209%`와 직접 비교할 수 없다. branch를 채택하기 전에 고정 Model225
route에 섞었을 때 specimen-group leave-one-out에서도 이득인지 Model309에서 확인한다.

P100 GPU6에서 약 1626초(27분) 실행했다. 구현: `model308_dinov2_dense_fusion_oof.py`
결과: `/workspace/komap_model1/outputs/model308_dinov2_dense_fusion/results.json`
로그: `/workspace/komap_model1/outputs/model308_dinov2_dense_fusion_gpu6.log`

## 모델 309: Model308 DINO branch의 Model225 route 보완성 (P100 서버 완료, 미채택)

Model308 valid 확률을 기존 Model225 route에 class-wise 혼합하는 43개 exact 후보를 19-specimen-group
leave-one-out으로 평가했다. 첫 시도에는 잘못하여 Model217의 5-step TENT 확률을 route 구성에 넣어
Model225 기준 점수를 재현하지 못했다. 이 결과는 폐기하고, Model225에서 실제 사용한 Model220 config 6
(20-step TENT, lr `5e-5`; `valid_probs_06.npz`)으로 바로잡아 재실행했다. 수정된 evaluator는 Model225
기준 image-wise mIoU `78.6921%`를 정확히 재현했다.

43개 후보 중 in-sample 최고도 기준 Model225 route 그 자체였다. 19개 held-out specimen 모두 기준 route를
선택했고, held-out 변화는 `0.0000%p`, positive image `0/20`, group-bootstrap 95% CI `[0, 0]%p`였다.
따라서 DINO branch는 현 최고 route를 보완하지 못해 미채택한다.

잘못된 입력을 쓴 초기 probe: `/workspace/komap_model1/outputs/model309_model308_route_probe/`
수정 완료 결과: `/workspace/komap_model1/outputs/model309_model308_route_probe_tent20/results.json`

## 모델 310: Model308 DINO branch의 원래 1024px scale 전이 (P100 완료, 미채택)

512px로 학습된 Model308 adapter를 재학습 없이 원래 최고 모델 추론 조건인 1024px patch/stride 864/
Gaussian overlap/D4에 적용했다. 기준으로 다시 생성한 CNN 예측은 저장된 Model303 OOF 및 valid 확률과
평균·최대 absolute 차이가 모두 `0.0`으로 정확히 일치해 scale 비교 parity를 확인했다.

Train OOF image-wise mIoU는 1024px baseline `77.5096%`에서 DINO branch `77.4823%`로
`-0.0272%p`였다(31/70 positive, group-bootstrap 95% CI `[-0.1854, +0.1376]%p`). 공식 valid는
`75.0472%`에서 `75.0194%`로 `-0.0278%p`였다(6/20 positive, 95% CI
`[-0.2604, +0.2329]%p`). 512px 학습 branch는 1024px 적용 시 일반화 이득이 없었으며 Model309 route
CV도 개선되지 않아 DINO residual 접근은 종료·미채택한다.

P100 GPU6 추론은 약 1343초(22분)였다. 채점 단계의 mask 경로 전달 버그는 저장된 전체 확률로만 재채점해
수정했으며, inference를 반복하지 않았다. 구현: `model310_dinov2_1024_scale_probe.py`
결과: `/workspace/komap_model1/outputs/model310_dinov2_1024_scale_probe/results.json`

## 모델 311: Eutectic Si train-OOF hard crop mining (P100 완료, 미채택)

Model303의 네 fold checkpoint를 이어 학습하면서, 각 train 이미지에 대해 해당 specimen을 제외해
만든 grouped-OOF 확률로 Eutectic Si false-negative/false-positive 위치를 mining했다. 이미지당 epoch당
기존 random crop 6개에 OOF hard crop 2개를 추가하되 총 8개 crop 수는 유지했다(1024px patch, batch 2,
4 epoch/fold). Valid mask는 학습·hard crop 선택·fold 판단에 사용하지 않고, 전체 예측이 끝난 뒤 보조
점수 산출에만 사용했다.

70장 grouped OOF image-wise mIoU는 Model303 `77.5096%`에서 `77.5029%`로 `-0.0066%p` 변했다
(positive `35/70`, specimen-group bootstrap 95% CI `[-0.1421, +0.1220]%p`). 사실상 동률이며 개선
증거는 없다. 클래스별 변화(Al3Ni, Primary Si, Al, Eutectic Si)는 `[+0.0355, +0.0036, +0.0232,
-0.0888]%p`로, 집중 개선 대상인 Eutectic Si도 소폭 하락했다. 공식 valid는 `75.0472%`에서
`75.0778%`로 `+0.0306%p`였지만, 20장 보조 valid의 작은 변동만으로 채택하지 않는다. 따라서 OOF hard
crop mining은 현재 설정에서 효과가 확인되지 않아 미채택한다.

P100 GPU6 네 fold 학습·추론에 약 `3871초`(64.5분) 소요. 구현: `model311_eutectic_oof_hard_mining.py`
결과: `/workspace/komap_model1/outputs/model311_eutectic_oof_hard_mining/results.json`
확률: `train_oof_probs.npz`, `valid_fold_ensemble_probs.npz`

## 모델 312: EfficientNet-B4 3배 해상도 (P100, 20/28 epoch 조기 종료)

팀 실험 문서의 3배 확대 결과(3 seeds 평균 `78.79%`)를 현재 B4 학습기에 맞춰 점검한다. Model280의 4배
실험은 계획된 7,840 optimizer updates 중 5,600 updates(20/28 epoch)에서 중단됐고 최고 valid
image-wise mIoU가 `71.3317%`였다. 이번에는 seed·구조·patch 크기·crop sampling·augmentation·LR를
Model280과 맞추고 배율만 4배에서 3배로 바꿔 끝까지 학습한다. 설정은 EfficientNet-B4 U-Net,
1536px patch, 4 crops/image, batch 1, uniform-area sampling, metallography augmentation, LR `5e-5`,
400-update warmup + cosine, 28 epochs(7,840 optimizer updates), D4 TTA/Gaussian overlap이다.
image-wise mIoU를 4 epoch마다 평가·checkpoint 선택에 사용하며 empty union은 0으로 처리한다.
test 추론은 하지 않는다. 다만 이 실행은 팀 점수의 정확한 재현은 아니다. 팀 문서는 3-seed 평균,
whole-image reflect-pad + D4 추론, GT에 없는 클래스를 image-wise 평균에서 제외하는 규칙을 보고한다.
Model312는 단일 seed, 1536/1152 tile + Gaussian overlap 추론을 쓰고, 빈 클래스 IoU를 0으로 둔다.
valid 20장 중 Primary Si가 없는 이미지는 A71 한 장이라 결측 처리만으로 생길 수 있는 차이는 최대
`1.25%p`; 따라서 그 규칙만으로 큰 성능 격차를 설명할 수 없다. Model312 점수는 현재 학습·타일 추론
recipe 안에서의 3배 확대 성능으로 해석하고, 팀 기록과 직접 동급 비교하지 않는다.

epoch4/8/12/16/20 train loss는 `0.6141/0.5230/0.4958/0.4860/0.4849`이고 valid image-wise mIoU는
`69.638/70.953/70.368/70.503/69.676%`다. 최고 epoch8보다 epoch20이 `-1.277%p` 낮고, epoch16에서
회복하지 못한 뒤 epoch20에서 다시 하락했다. epoch20 global class IoU는 Al3Ni `70.40%`, Primary Si
`82.68%`, Al `91.87%`, Eutectic Si `49.39%`다. 같은 seed의 Model280 4배 최고는 `71.332%`
(epoch16)로 본 run과 유사한 낮은 수준이었다. 두 고해상도 B4 run 모두 최고 route Model225
`78.6921%`에 크게 못 미쳐, 후반 8 epoch는 중단했다. best checkpoint는 epoch8, last checkpoint는
epoch20이며 test 추론은 하지 않았다.

실행: P100 GPU2 (`CUDA_VISIBLE_DEVICES=2`)
진행 로그: `/workspace/komap_model1/outputs/model312_effb4_3x_match_gpu2.log`
결과 경로: `/workspace/komap_model1/outputs/model312_effb4_3x_match/`

## 모델 313: Model312 체크포인트의 1792px 타일 추론 (valid only)

Model312 best(epoch8)을 고정하고 scale 3, D4/Gaussian을 유지한 채 추론 tile/stride만
1536/1152에서 1792/1344로 키웠다. fixed class-3 bias 0 기준 image-wise mIoU는
`71.2653%`로 Model312 epoch8 `70.9530%` 대비 `+0.3123%p`였다(global `75.0738%`). valid에 맞춰
bias를 다시 고르면 `71.7080%`가 되지만, 이는 같은 20장에 대한 calibration 점수이므로 일반화 이득으로
보지 않는다. 타일 문맥 확장은 작은 개선만 보였다. valid 확률은 저장했고 test 추론은 생략했다.

결과: `/workspace/komap_model1/outputs/model313_context_1792/`

## 모델 314: Model312 체크포인트의 whole-image D4 추론 (valid only)

팀 문서의 whole-image reflect-pad + D4 추론을 분리 검증하기 위해 Model312 best(epoch8)을 고정하고
추론만 whole-image로 바꿨다(scale 3, 32 배수 reflect pad, D4). fixed class-3 bias 0 기준 image-wise
mIoU는 `70.5038%`(global `74.6954%`)로, 같은 체크포인트의 1536px 타일 기준 `70.9530%`보다
`-0.4492%p`, 1792px 타일보다 `-0.7615%p`였다. 따라서 낮은 단독 점수의 주원인이 타일 문맥 부족이라는
가설은 지지되지 않았다. 큰 valid 이미지(1440×1024)를 포함해 전체 20장 추론이 P100에서 완료됐고,
test mask는 생성하지 않았다. valid 확률을 Model225 route에 추가할지 specimen-group CV를 수행 중이다.

결과: `/workspace/komap_model1/outputs/model314_whole_image/`
구현: `encoder_structure_search.py --inference-mode whole-image`

Model314 whole-image 확률을 Model225 route의 Al3Ni/Primary/Eutectic source로 넣는 15개 exact
class-wise 조합(gamma `0.30/0.50`)도 비교했다. 최고는 기존 Model225 route 그대로 `78.6921%`였고,
19개 specimen-group leave-one-out fold 모두 기존 route를 선택했다(held-out 변화 `0.0000%p`,
positive image `0/20`, group-bootstrap 95% CI `[0.0000, 0.0000]%p`). 따라서 B4 고해상도 branch는
단독 성능뿐 아니라 최고 route 보완성도 확인되지 않아 채택하지 않는다.

route 결과: `/workspace/komap_model1/outputs/model314_whole_image/route_probe.json`

## 모델 315: Model257 ResNet34 native-stem whole-image D4 추론 (P100, 미채택)

Model314와 달리 전체 이미지 통계로 학습한 Model257 체크포인트를 대상으로, `Model239`의
full-resolution native stem을 포함한 정확한 구조로 체크포인트를 strict-load하고 scale 1 전체 이미지에
D4 확률 평균을 적용했다. valid 20장 image-wise mIoU는 `73.9255%`, global mIoU는 `78.1414%`로,
기존 Model257 patch/stride D4 추론의 `77.2071%`보다 `-3.2816%p` 낮았다. 최대 이미지 1440×1024에서
추론은 가능했지만 CUDA peak allocated memory가 약 `15.89 GB`(14.79 GiB)로 높았다. 결과적으로
전체 이미지 문맥을 한 번에 처리하는 것이 이 branch에는 도움이 되지 않았다.

저장 확률을 Model225 운영 route에 넣어 Al3Ni/Eutectic/Primary class-wise 14개 혼합 후보를 추가
비교했다. 20장 in-sample 최고와 19개 specimen-group leave-one-out 모두 기존 Model225 route였고,
CV 변화 `0.0000%p`, positive image `0/20`, group-bootstrap 95% CI `[0.0000, 0.0000]%p`였다.
따라서 Model315 추론은 독립 branch와 route 보완성 양쪽 모두 개선 근거가 없어 미채택한다. test는
추론하지 않았다.

P100 GPU2 추론은 20장 약 72초. 구현: `model315_model257_whole_image.py`
확률·요약: `/workspace/komap_model1/outputs/model315_model257_whole_image_valid/`
route 결과: `/workspace/komap_model1/outputs/model315_wholeimage_route/results.json`

## 모델 316: 클래스 균형 reliable-pixel TENT (P100 완료, 미채택)

Model195를 이미지별로 reset하고 BN affine만 20 step, lr `5e-5`로 unlabeled adaptation했다.
기존 entropy 평균과 달리 각 pseudo-class에서 entropy 하위 75% 픽셀만 선택하고, 선택된 클래스별
평균 entropy를 동일 가중해 다수 클래스가 gradient를 독점하지 않게 했다. 각 이미지 mask는 적응과
추론이 끝난 뒤에만 평가에 사용했다. 이는 EATA의 신뢰도 기반 샘플 선택에서 착안한 segmentation용
pixel-level 변형이며, Fisher anti-forgetting 항은 이미지마다 모델을 reset하므로 적용하지 않았다.

독립 valid branch는 image-wise `76.6111%`, global `79.3284%`로 기존 20-step TENT
(`76.7921%`)보다 낮았다. 확률을 Model225에 넣은 14개 class-wise 후보도 기준 route
`78.6921%`보다 높지 않았다. 19개 specimen-group CV delta는 `-0.0145%p`, positive image `0/20`,
group-bootstrap 95% CI `[-0.0457, 0.0000]%p`로 비개선이었다. 따라서 신뢰 픽셀 선별·클래스 균형
adaptation은 현재 설정에서 기존 TENT보다 유효하지 않아 미채택한다. test는 추론하지 않았다.

P100 GPU2 valid 적응·추론은 약 192초. 구현: `model316_reliable_pixel_tent.py`
확률·요약: `/workspace/komap_model1/outputs/model316_reliable_tent_valid/`
route 결과: `/workspace/komap_model1/outputs/model316_reliable_tent_route/results.json`

## 모델 317: EfficientNet-B4 1배 해상도 재검증 (P100 GPU2, 14/28 epoch 중단)

도메인 적응·증류와 별도로 B4 기본 분할 학습을 다시 확인했다. ImageNet 사전학습 EfficientNet-B4 U-Net,
512px patch, scale 1, 이미지별 균등 4 crop, batch 1, LR `5e-5`, warmup 400 updates, CE+Dice,
동일 클래스 가중치, metallography augmentation, D4/Gaussian 타일 추론을 사용했다. 검증은 4 epoch마다
실행했고 test 추론은 생략했다.

epoch 4/8/12의 fixed-bias image-wise mIoU는 `65.2347/66.7332/66.5161%`였고, 저장된 최고는 epoch 8의
`66.7332%`다. epoch 12에서 개선이 멈춰 14/28 epoch에서 중단했다. `best.pt`는 epoch 8, `last.pt`는
epoch 14까지 저장되어 있으며 최종 calibration/summary와 test 산출물은 만들지 않았다. 현 최고 route
Model225 `78.6921%`에 미달하므로 채택하지 않는다.

이 결과는 팀 EfficientNet-B4 기준선과 직접 같은 조건의 재현으로 보지 않는다. 팀 문서는 원본 면적에
비례해 crop을 배분하고 마지막 checkpoint를 평가에 사용했다고 기록했지만, 이 run의 `uniform_area`는
각 이미지에서 같은 횟수로 crop을 뽑았다. 공식 클래스 색상-이름 매핑은 코드와 범례가 일치함을 확인했다.
따라서 다음 비교는 적응/증류가 아니라 crop 표집 프로토콜과 희귀상(Eutectic Si) 경계 오류를 겨냥한
지도학습 설정에서 시작한다.

결과: `/workspace/komap_model1/outputs/model317_effb4_1x_7840/`

## 모델 318: EfficientNet-B4 3배 해상도 면적비례 crop 표집 (P100 GPU2, 8/28 epoch 중단)

팀 실험의 원본 픽셀 면적 비례 crop 표집을 재현하기 위해 Model312와 같은 seed `280`, EfficientNet-B4,
1536px patch/scale 3, batch 1, 280 crops/epoch, LR `5e-5`, warmup 400 updates,
기본 hard class weights, CE+Dice, metallography augmentation 및 D4/Gaussian 추론을 유지하고 표집 방식만
`uniform_area`에서 `area_weighted`로 변경했다. 기존의 이미지별 고정 배정 대신 매 crop마다 이미지 픽셀
면적에 비례해 이미지를 복원추출한 뒤 이미지 안에서 중심점을 균일 추출한다. 입력 shape와 클래스 인덱스를 확인하는 smoke test를
통과한 뒤 P100 GPU2에서 학습했다.

Model318의 fixed-bias image-wise mIoU는 epoch4 `70.9250%`, epoch8 `68.5019%`였다. 같은 조건의 Model312는
epoch4 `69.6376%`, epoch8 `70.9530%`다. 따라서 면적비례 표집은 epoch4에서 `+1.2874%p`였지만 epoch8에는
Model312보다 `-2.4511%p`로 역전했다. epoch4 global class IoU 변화(Al3Ni/Primary Si/Al/Eutectic Si)는
`-0.42/+2.06/-0.21/+1.28%p`였으나, epoch8에는 `-2.62/-7.34/+0.21/-0.49%p`로 Primary Si가 크게
하락했다. 학습 loss가 낮아져도 validation mIoU가 유지되지 않아 epoch8에서 중단했다. 최고 checkpoint는
epoch4이며, epoch4 valid 확률은 별도 추론으로 저장했다. best checkpoint의 valid상 bias sweep 최고
`71.2385%`(Al bias `-0.35`)는 같은 valid에 맞춘 값이라 독립 성능으로 보지 않는다. test 추론은 하지 않았다.

구현: `improve_five_causes.py`의 `area_weighted` sampling mode
결과: `/workspace/komap_model1/outputs/model318_effb4_3x_area_weighted/`
epoch4 valid 확률: `/workspace/komap_model1/outputs/model318_area_weighted_best_e4_valid/`

## 모델 319: Model318/312 확률의 Model225 route 보완성 (P100 완료, 미채택)

Model318 epoch4 및 Model312 epoch8의 동일 추론 프로토콜(1536px tile, stride 1152, D4/Gaussian) valid
확률을 생성해 Model225 기준 route와 비교했다. 두 B4 branch 중 하나 또는 양쪽을 Al3Ni/Primary Si/
Eutectic Si 클래스별로 30%/50% 혼합하는 35개 exact 후보를 평가했다. 기준 Model225 image-wise
mIoU `78.6921%`를 재현했고, in-sample 최고도 기준 route 자체였다. 19개 specimen-group leave-one-out에서
모든 fold가 기준 route를 선택했으며 held-out delta `0.0000%p`, positive image `0/20`, group-bootstrap
95% CI `[0.0000, 0.0000]%p`였다. 따라서 표집 변경 branch는 단독 성능과 기존 route 보완성 모두 채택 근거가
없다.

구현: `model254_highres_route_probe.py` (candidate ID 인자 추가)
결과: `/workspace/komap_model1/outputs/model319_area_weighted_route_probe/results.json`
Model312 대조 확률: `/workspace/komap_model1/outputs/model319_uniform_area_control_e8_valid/`
test 추론은 생략했다.

## 모델 320: Al3Ni/Eutectic Si 전역 문맥 게이트 grouped-OOF 진단 (P100 CPU, 미채택)

도메인 적응·증류와 별도의 구조 개선 가설로, 이미지 전체 조성/밝기 문맥이 Al3Ni와 Eutectic Si의
혼동 보정에 유효한지 저비용으로 확인했다. Model297의 train-only specimen-group OOF 확률을 고정 입력으로
사용하고, 네 개 specimen-group fold 각각에서 held-out 그룹을 제외해 이진 logistic expert를 학습했다.
비교 대상은 픽셀 단위 확률·밝기·국소 통계만 입력한 expert와, 여기에 이미지 밝기 분위수, 예측 클래스
비율/평균 확률 및 각 상의 밝기·국소 변동 통계를 더한 expert다. 수정은 기준 예측에서 Al3Ni 또는
Eutectic Si로 분류된 픽셀에만 적용했다. 학습 표본도 두 GT 상에서 클래스별 최대 384픽셀로 균형 추출했다.
평가는 train 70장/59개 specimen 그룹 전체의 4-class image-wise mIoU이며, 95% 구간은 그룹 paired
bootstrap 10,000회로 계산했다. 공식 valid와 test는 학습·선택에 사용하지 않았다.

기준 Model297 OOF mIoU `77.2266%`에 비해 pixel-only expert는 `76.4177%`(`-0.8090%p`, 그룹 bootstrap
95% CI `[-1.0453, -0.4854]%p`, 개선 이미지 `12/70`), global-context expert는 `76.3990%`
(`-0.8276%p`, 95% CI `[-1.1180, -0.4655]%p`, 개선 이미지 `14/70`)으로 모두 명확히 하락했다.
전역 문맥 추가와 pixel-only 간 차이도 `-0.0186%p`(95% CI `[-0.2238, +0.1814]%p`)로 사실상
구분되지 않았다. 네 fold 모두 두 logistic optimizer가 수렴했으며, 각 fold 점수도 기준보다 낮았다.

따라서 이 간단한 이미지 문맥 보정에는 일반화 근거가 없고, 이를 근거로 대규모 global-context decoder를
학습할 이유도 부족해 해당 방향은 보류한다. 다만 이 결과는 logistic 후처리 게이트만 검증했으므로,
학습된 multi-scale 문맥 구조 전체가 무효라고 단정하지는 않는다. 다음 개선은 추가 DA/증류보다 지도학습
오류 분석—특히 어려운 표본의 라벨·상 경계 일관성, 희귀 상 혼동의 재현 가능한 train-OOF 보정성—을
우선해 근거가 확인되는 가설만 확장한다.

구현: `model320_global_context_gate.py`
결과: `/workspace/komap_model1/outputs/model320_global_context_gate/results.json`

## 팀 코드 ZIP 분석: `KoMaP_code.zip` (2026-09-26)

압축 파일에는 22개 코드·문서 파일(가중치/checkpoint와 예측 결과는 없음)이 들어 있다. 팀이 기록한
최고 `79.12%`는 3배 EfficientNet-B4 세 시드와 4배 한 시드의 probability ensemble이며, 팀의
`common.py`는 이미지별 네 class IoU 중 union이 0인 항목을 `NaN`으로 만들어 평균에서 제외한다.
우리 기본 기록은 empty union을 0으로 두므로 같은 Model225 route도 `78.6921%`다. 팀 방식처럼 결측
항목을 빼면 `79.4693%`다. 실제로 A71 한 장에서 Primary Si GT와 예측이 모두 없어 이 IoU만 `0/0`이다.
따라서 팀의 `79.12%`와 우리 `78.6921%`를 그대로 비교하면 안 되며, 결측 제외 기준에서는 기존 Model225가
`0.3493%p` 높다. 대회 산식은 결측 class의 처리(0/제외/1)를 명시하지 않으므로, 고정 4-class/empty=0과
empty 제외 값을 함께 기록한다. 팀 최고 점수는 ZIP 안에 prediction/checkpoint가 없어 독립 재채점하지 못했다.

모델·학습 recipe도 다르다. 팀 상위 분기는 SMP U-Net + EfficientNet-B4, scale 3, crop 1536, batch 2,
16,000 optimizer iterations, AdamW LR `3e-4`, unweighted CE+Dice이고, 전체 resized image D4 추론을 쓴다.
증강은 affine scale `0.7–1.4`와 임의 회전, D4, brightness/contrast, gamma, blur/noise다. 반면 우리의
최고 Model225는 ResNet34 기반 지도 branch와 20-step TENT rare-class 확률을 class-wise route로 합친
방식이다. 입력 정규화도 팀은 회색 3채널에 동일 mean/std (`0.449/0.226`)를 쓰고, 우리 RGB 경로는
ImageNet 채널별 통계를 적용해 온 차이가 있다. 우리가 별도로 시험한 B4 3배 runs는 patch 1536이어도
batch 1, hard class weights, LR
`5e-5`, 짧은 학습량/중간 checkpoint 평가였으므로 팀의 16k-step recipe 재현이 아니다(예: Model312는
20/28 epoch=5,600/7,840 planned updates에서 중단, best standalone `70.9530%`). 이 낮은 결과만으로
팀 B4 접근을 배제하지 않는다.

팀 코드의 `queue_round7`/`run_seeds.sh`에는 per-image width 4000 배율 정규화와 uniform image sampling도
후보로 있으나, ZIP의 실험 기록에는 그 결과가 없어 검증된 `79.12%`에 포함된 방법으로 취급하지 않는다.
다음 팀-B4 재검증은 고정 3배·긴 학습·batch 2·강한 affine/광도 증강·grayscale 공통 정규화를 맞춘 단일
branch부터 하고, 결과 확률을 Model225 route에 더해 specimen-group leave-one-out으로 보완성을 판단한다.
그 후에만 width-normalization/uniform-sampling 조합을 확장한다.

## 모델 321: Model303 boundary branch 연속 회전 증강 grouped-OOF (P100 완료, 미채택)

일반적인 D4 변환으로 만들 수 없는 임의 각도 회전이 현미경 시야 방향 변화에 강건성을 더하는지
확인했다. Model303의 specimen-group 4-fold OOF checkpoint에서 각 fold를 4 epoch씩 이어 학습했고,
1024px patch, batch 2, LR 배수 0.25, hard class weights와 CE+Dice는 고정한 채 입력에는 bilinear,
마스크에는 nearest-neighbor로 동일한 무작위 각도(±180°)를 적용했다. 각 train 이미지는 자신이 학습에
들어가지 않은 fold 모델의 예측만 OOF로 받았으며, checkpoint는 사전 고정한 마지막 epoch를 사용했다.
따라서 valid mask는 학습이나 선택에 쓰지 않았다. 점수는 고정 4-class image-wise mIoU, empty union=0이다.

70장 grouped-OOF에서 Model303 `77.5096%` → Model321 `77.2998%`로 `-0.2097%p`였고, 개선 이미지는
`26/70`, specimen-group paired-bootstrap 95% CI는 `[-0.4859, +0.0713]%p`였다. 20장 valid fold ensemble도
Model303 `75.0472%` → Model321 `74.6942%`(`-0.3530%p`)로 하락했다. OOF 구간은 0을 포함하지만 두
평가 모두 평균 변화가 음수이므로 임의 회전은 현 설정에서 채택하지 않는다.

결과: `/workspace/komap_model1/outputs/model321_rotation_aug_oof/results.json`
확률: `train_oof_probs.npz`, `valid_fold_ensemble_probs.npz` (동일 디렉터리)

## Validation mIoU 결측 클래스 정책 전수 재채점 (2026-09-26)

공식 산식의 `0/0` 규칙이 미정이라 앞으로 validation 결과에는 `empty union=0`과
`empty class 제외(NaN skip)`를 함께 적는다. 같은 per-image mean-of-four-class-IoUs 정의를
유지하고, 결측 처리만 바꾼다.

P100의 191개 `results.json`에서 20장 per-image score vector 5,002개를 다시 평균했다.
기존 보고값과 `1e-9` 이상 차이 나는 항목은 0개, 읽기 오류도 0개였다(동일 설정의
`best_image`/`best_exclude`/`records` 중복 포함). 별도로 188개 NPZ를 살펴 20개 validation
이미지 예측이 모두 들어 있는 111개 probability archive를 mask와 다시 대조해 두 정책 점수를
계산했다. 모델 채널 순서 차이는 팀 B4 채널 `[Al, Al3Ni, Primary Si, Eutectic Si]`와 기존
모델 채널 `[Al3Ni, Primary Si, Al, Eutectic Si]`를 분리해 처리했다.

추가 전수 재실행 결과도 `miou_policy_audit/rerun_20260926_user/`에 별도 보관했다. 범위와
오류 건수는 같았고 최고 route는 두 결측 정책 모두 Model270(`78.7209% / 79.4978%`), 최고
단독 archive는 Model289(`77.4181% / 78.0528%`)로 유지됐다. 당시 학습 중인 Model322는
완성된 valid 예측이 없어 두 재집계에 포함되지 않았다.

최고 raw validation route는 Model270이며 `empty=0` **78.7209%**, `empty=skip` **79.4978%**다.
하지만 Model225 기준 route와 비교한 19-specimen-group leave-one-out 변화가 `-0.1320%p`,
95% CI `[-0.3548, +0.0574]%p`, 개선 이미지 `9/20`이므로 채택하지 않았다. 현재 일반화 기준으로
유지하는 Model225는 두 정책에서 각각 **78.6921% / 79.4693%**다. 완전한 단독 probability archive
중 최고는 Model289 `77.4181% / 78.0528%`였으며, 고점 route의 저장 점수 벡터와 혼동하지 않는다.

상세 표와 재채점 스크립트: [miou_policy_audit.md](miou_policy_audit.md),
`recompute_miou_policy.py`, `recompute_miou_recorded_results.py`.

## 모델 322: 팀 EfficientNet-B4 3× grouped-OOF 재현 (완료)

팀 제공 3× 레시피를 4-fold specimen-group CV로 재현했다: EfficientNet-B4 U-Net, crop 1536, batch 2, AdamW `3e-4`, unweighted CE+Dice, affine/D4/광도 증강, fold당 16,000 updates. P100 호환을 위해 BF16을 FP16+GradScaler로 바꾸고, 공식 valid를 checkpoint 선택에 쓰지 않았다.

- 70장 grouped OOF: image-wise mIoU empty-union=0 `78.5351%`, NaN-skip `78.7825%`.
- 20장 valid 단독: empty-union=0 `77.85%`, NaN-skip `78.57%`.
- 약한 Model303 parent와 grouped OOF 비교: 단독 `78.53%` 대 `77.51%`; CV에서 고른 branch 혼합 route `79.35%` (+`1.84%p`, 95% CI `[+1.33,+2.40]%p`, 64/70장 개선). 이는 현재 최고 Model270 route와의 비교가 아니다.
- fold1 18장 preliminary 비교(Model303 parent): empty=0 `−0.40%p` (95% CI `[-1.48,+0.62]`), NaN-skip `+0.56%p` (95% CI `[-1.33,+3.45]`); 단일 fold라 결론 근거로 쓰지 않는다.

구현: [model322_team_b4/KoMaP](model322_team_b4/KoMaP/) · OOF 결과: `/workspace/komap_model1/outputs/model322_team_b4_group_oof/`

## 모델 323: Model321 회전증강 branch를 Model270 route에 추가 (완료, 미채택)

Model322의 train OOF 비교 기준은 Model303 parent이므로, 기존 최고 raw route인 Model270에
보조 branch가 실제로 이득을 주는지 별도로 확인했다. `model323_model270_branch_route_probe.py`로
Al3Ni/Primary/Eutectic subset에 Model321 valid 확률을 30%·50% 섞는 14개 후보와 100% 대체 후보를
평가했다. Model270 기준 확률은 model195 Primary `0.75`, 20-step TENT rare branch(model220 설정,
`valid_probs_06.npz`) Al3Ni `0.40`, model269 Eutectic `0.50` 및 고정 prior/edge 후처리다.
처음에는 5-step TENT archive를 잘못 연결해 baseline이 재현되지 않았으나, 잘못된 실행은
`invalid_tent5_diagnostic.json`으로 격리하고 20-step archive로 다시 계산했다.

정정된 baseline은 `empty=0` `78.7209%`, `empty=skip` `79.4978%`로 Model270 기록과 일치했다.
두 정책 모두 후보 최고는 기준 route 자체였고, 19개 validation specimen-group held-out에서
모든 fold가 기준을 선택했다(평균 변화 `0.0000%p`, positive `0/20`, group-bootstrap 95% CI
`[0.0000, 0.0000]%p`). 따라서 Model321 회전증강 branch는 Model270을 보완하지 못한다.
이는 validation 내부 group-CV이지 독립 test 보장은 아니다.

구현: [model323_model270_branch_route_probe.py](model323_model270_branch_route_probe.py)
결과: [Model321 route probe JSON](model323_model270_model321_fusion_tent20/route_probe.json)
P100 원본: `/workspace/komap_model1/outputs/model323_model270_model321_fusion_tent20/route_probe.json`

후속으로 Al branch까지 포함한 32개 후보를 평가했다. raw 최고는 두 정책 모두 baseline Model270
그대로였고, 19개 validation specimen-group held-out route 선택 변화는 두 정책에서 모두
`-0.0384%p`(positive `0/20`, group-bootstrap 95% CI `[-0.1213, 0.0000]%p`)였다.
19 folds 중 A153만 Al 50% 혼합을 골랐으며, held-out 전체 평균은 오히려 하락했다. 따라서
Model321의 Al-inclusive fusion도 미채택이다. 이 결과는 Model322에 같은 route evaluator를 쓸 때
Al을 후보에서 빼지 않도록 확인해 준다.

4-class 결과: [route_probe_all4.json](model323_model270_model321_fusion_tent20/route_probe_all4.json)
Model322 B4 확률도 생성되면 team channel order를 main 순서로 재배열해 같은 32개 후보를 평가한다.
이 Model323 실험은 우리 Model321 branch와 Model270의 융합이며, 팀원 모델과 우리 모델 간 융합은 아니다.

## 모델 324 후보: GRAF-inspired Primary-Si inter-particle gap weighting (사전검사 후 미실행)

Al-Si 미세조직의 인접 Primary-Si 입자 사이 간극을 morphology로 찾고 gap-region CE를 높이는
GRAF/GRWS 아이디어는 기존 실험 기록에 없었다. 원 논문은 인접 입자 병합을 줄이기 위해 gap 영역
가중치를 사용했고, 저자 실험에서 weight 10을 사용했다. 특허 설명은 각 Primary-Si instance를
개별 팽창한 뒤 쌍별 팽창 교집합을 합쳐 gap mask를 만드는 방식이다. [GRAF 논문](https://doi.org/10.1016/j.commatsci.2023.112620),
[gap-region 방법 특허](https://patents.google.com/patent/CN113808140B/en).

초기 native-resolution train GT 사전 점검에서는 56/70장에 gap 픽셀이 있었지만, 실제 Model302
학습 최소 변 길이 1024와 반경 5px 정의로 train OOF를 다시 점검했다. 70장·59 specimen group에서
Model302의 baseline OOF image-mean mIoU는 `77.4562%`, gap 영역을 정답으로 완벽 교정하는
oracle도 `77.4675%`에 그쳤다(**+0.0113%p**, specimen-group bootstrap 95% CI
`[+0.0076, +0.0153]%p`). 학습 해상도에서 gap 픽셀은 평균 `0.0382%`였고 그중 오분류 비율은
`10.88%`였다. 검증 정답을 이용한 조정 없이 train OOF만 쓴 상한 진단에서도 사전 기준 `+0.3%p`에
크게 못 미쳐 Model324 학습은 취소했다. native-resolution 비율 `0.0929%`는 실제 학습 해상도
효과를 반영하지 못한 예비 통계로만 취급한다.

구현(`model302_boundary_oof_finetune.py`)과 no-training oracle 진단
(`model324_gap_oracle_audit.py`)은 보존한다. oracle 결과가 작아 추가 학습은 하지 않는다.
결과 JSON: `/workspace/komap_model1/outputs/model324_gap_oracle_model302_oof/results.json`.

## Model270 검증 오류 분해 (진단 전용; 모델 학습 아님)

Model270 route를 저장된 확률 archive로 재구성해 기준 image-mean mIoU `78.7209%`를 정확히
재현했다. 검증 20장·19 specimen group에서 클래스별 image-mean IoU는 Al3Ni `75.15%`,
Primary Si `86.04%`, Al `93.74%`, Eutectic Si `59.95%`였다. GT에 존재하지만 예측이 전혀 없는
클래스는 한 건도 없어, 주된 문제는 클래스 누락보다 경계/영역 혼동이다. 한 클래스를 GT로 완벽
교정하는 진단상 상한 기여는 Eutectic Si `+14.04%p`, Al3Ni `+12.98%p`, Al `+7.27%p`,
Primary Si `+4.08%p` 순이었다. 이는 달성 가능한 개선치가 아닌 오류 집중도 참고치다.

GT 경계를 반경 1px 무시하는 비공식 민감도 진단은 `87.21%`(+8.49%p), 2px는 `84.59%`,
3px는 `81.16%`였다. 대회 지표를 바꾸는 결과가 아니라 경계 픽셀 민감성을 보여주는 참고값이다.
진단에서는 validation GT로 모델·route·threshold를 조정하지 않았다. 최저 이미지도 Primary Si
GT가 없는 경우였으며, empty-union=0 정책 때문에 해당 class IoU는 0으로 들어간다.

결과: [model325_model270_error_audit.json](model325_model270_error_audit.json)
구현: [model325_model270_error_audit.py](model325_model270_error_audit.py)

## 모델 325 설계: 팀 B4 재현 + Model270 확률 융합

Model322의 grouped-OOF 학습·validation 확률이 완성되는 즉시, 팀원 제공 EfficientNet-B4 U-Net
레시피의 재현 모델과 우리 Model270 route를 결합한다. 팀 기록상 최고 79.12%는 3× 3-seed와 4×
branch의 앙상블이며 팀 evaluator의 empty-class NaN-skip 값이다(우리의 empty=0 점수와 직접 비교 불가).
공유된 `KoMaP_code.zip`에는 그 checkpoint들이 포함돼 있지 않다.
따라서 우선 Model322의 4-fold 3× B4 재현 예측을 팀 방식 branch로 사용하며, 원본 팀 checkpoint
앙상블과 동일한 가중치라는 뜻은 아니다.
기존 `model323_model270_branch_route_probe.py`의 고정 Model270 구성 및 팀→주 채널 재배열을
재사용하고, 4개 클래스 부분집합·혼합비 0.30/0.50 및 B4 전체 대체를 평가한다. 결측 클래스는
IoU 0과 NaN-skip 정책을 모두 계산하며, image-wise 네 클래스 평균을 유지한다. Validation의
최고 단일 route 점수만으로 채택하지 않고 specimen-group held-out CV 개선과 CI도 함께 판정한다.

자동 순차 실행기: [run_model325_team_b4_fusion_after322.sh](run_model325_team_b4_fusion_after322.sh)
평가 결과는 `/workspace/komap_model1/outputs/model325_team_ours_fusion/route_probe.json`에 저장한다.

## 모델 325 결과: 팀 B4 재현 + Model270 확률 융합 (완료, 일반화 개선 미확정)

첫 실행은 팀 B4 확률 archive의 이미지 키(`A117_step4_sample2`)와 기존 branch archive의 키
(`A117_step4_sample2_image`) 표기 차이로 중단됐다. 의미상 같은 ID임을 확인해 `_image` 접미사 정규화 및
키 집합 일치 검사를 추가했고, 네 archive 모두 동일한 valid 20개 ID를 포함하는 것을 검증한 뒤 재실행했다.
구현: [model323_model270_branch_route_probe.py](model323_model270_branch_route_probe.py)

Model270 기준 valid는 empty-union=0 `78.72%`, NaN-skip `79.50%`다. 32개 후보 중 같은 valid set에서 가장 높은
조합은 Al3Ni·Primary Si·Al 확률에 B4를 각 50% 섞고 Eutectic Si는 기존 Model269 branch로 유지한 route로,
`79.16%` / `80.00%`(반올림; empty=0 / NaN-skip)를 얻었다. 다만 이는 같은 valid를 사용해 고른 in-sample 최고점이다.
19개 specimen-group leave-one-group-out route 선택의 평균 개선은 empty=0 `+0.382%p`(14/20장 개선),
group-bootstrap 95% CI `[-0.111,+0.898]%p`였다. CI가 0을 포함하므로 Model270 대비 안정적인 이득은 입증되지 않았고
앙상블 route는 아직 채택하지 않는다. 전체 결과: `/workspace/komap_model1/outputs/model325_team_ours_fusion/route_probe.json`.

## 모델 326: 팀 B4 recipe + full-image-stat 입력

EfficientNet-B4 U-Net에 `[raw grayscale, full-image z-score, full-image empirical CDF]`를 입력하고, 나머지는 Model322의 3×·crop1536·batch2·AdamW·CE+Dice 레시피로 고정했다. 4-fold specimen-group OOF는 fold당 16,000 updates이며 validation mask로 중간 선택하지 않았다. 이 16,000은 비교 기준을 맞추기 위해 상속한 budget이지 update 수 탐색으로 검증된 최적값이 아니다.

- 단독 valid: empty=0 `78.0400%`, NaN-skip `78.7090%`; 70장 OOF empty=0 `78.5297%`.
- Model322과 OOF paired 차이: empty=0 `−0.0050%p`, 95% group-bootstrap CI `[-0.2850,+0.2915]%p` (개선 35/70); 단독 입력 변경 효과는 확인되지 않았다.
- Model270 class-wise fusion valid: empty=0 `79.1937%`, NaN-skip `80.0186%`; Model270 대비 group-CV `+0.4728%p`, 95% CI `[+0.0933,+0.8838]%p`, 15/20장 개선. 현재 robust한 최고 후보이나 empty=0 80%에는 미달한다.

구현/결과: [model326_hybrid_fullstats](model326_hybrid_fullstats/) · `/workspace/komap_model1/outputs/model326_hybrid_fullstats_group_oof/`

## 모델 327: B4 + Model270 클래스별 혼합비 탐색 (CPU 완료, 미채택)

322의 valid B4 확률과 Model270 route를 이용해 기존 325의 후보군을 미세화한다. Al3Ni·Primary Si·Al의
B4 혼합 가중치를 서로 독립적으로 `{0.30, 0.50, 0.70}`에서 조합하고, Eutectic Si는 검증된 Model269
branch를 그대로 유지한다. 기존 32개 route를 포함해 총 59개 설정을 동일한 image-wise empty=0 지표로
평가하며, 19개 specimen-group leave-one-group-out 선택 및 group bootstrap CI를 함께 본다.
이 후보군은 Model325 탐색에서 얻은 클래스 선택 정보를 사용한 후속 탐색이라 valid 재사용에 따른 탐색 편향이
있을 수 있다. 따라서 raw 최고점만으로 채택하지 않고 grouped held-out 근거와 함께 탐색 결과로 보고한다.

전용 옵션 `--classwise-grid`를 [model323_model270_branch_route_probe.py](model323_model270_branch_route_probe.py)에
추가했고 compile/help 및 동일 20개 이미지 키 집합 preflight를 통과했다. 실행은 `CUDA_VISIBLE_DEVICES=`와
낮은 CPU 우선순위로 시작해 GPU 학습 자원을 점유하지 않았다. 결과:
`/workspace/komap_model1/outputs/model327_b4_classwise_route/route_probe.json`.
후보 59개를 모두 평가했다. in-sample 최고는 Al3Ni B4 `0.50`, Primary Si B4 `0.70`, Al B4 `0.70` 혼합이고
Eutectic Si는 Model269 branch 유지로, valid empty-union=0 image-mean mIoU `79.1898%`(NaN-skip `80.0330%`)다.
이는 같은 valid에서 고른 최고점이며 공식 empty=0 기준으로도 80% 미만이다. specimen-group held-out route 개선은
`+0.4265%p`, positive `15/20`, group-bootstrap 95% CI `[-0.0574,+0.8919]%p`로 0을 포함한다.
18/19 fold가 동일한 클래스별 가중치를 선택했지만 독립 개선 증거가 아직 충분하지 않아 미채택 상태다.

## 모델 328: B4 target-width 4000 + image-uniform crop (P100 실행 중)

팀 코드에 남아 있으나 이전에 검증되지 않은 두 제안(per-image resize width `4000`, crop image sampling
`uniform`)을 묶어 평가한다. 이는 두 요인의 개별 효과를 분리하는 실험은 아니지만, image-wise 평균 지표에 맞춰
이미지별 학습 기회를 같게 하고 미세조직 크기/배율 편차를 정규화하는 후속 후보다. B4 U-Net, 1536 crop,
batch 2, AdamW `3e-4`, CE+Dice 및 16,000 update의 나머지 팀 recipe를 유지하며, Model322와 같은 4-fold
specimen-group OOF 및 20장 valid 평가를 쓴다. 각 fold 학습 중 valid 평가는 끄고 마지막 update를 사용한다.
Model328 단독 평가는 기존 grouped-CV 비교(Model303 parent)로 수행하고, 완료 직후 CPU-only Model270
확률 route 탐색도 자동 실행하도록 후속 실행기를 보완했다. 따라서 새 B4 branch의 단독 성능뿐 아니라
현재 최고 route와의 valid 점수 및 specimen-group held-out 개선·신뢰구간까지 함께 판정한다.

Model326+Model270 route가 empty=0 `80%` 및 grouped-CV 95% CI 하한 `>0` gate를 통과하지 못해 시작했다.
Model328 완료 후 같은 gate로 판정하며, 미통과 시에만 Model329를 이어서 실행한다.

실행기: [run_model328_width4000_uniform_group_oof.sh](run_model328_width4000_uniform_group_oof.sh)
순차 평가기: [queue_model328_after326.sh](queue_model328_after326.sh)

## 모델 329 조건부 설계: full-stat 입력 + width4000 + uniform sampling

Model326의 `[raw, full-image z-score, full-image CDF]` 입력 표현과 Model328의 per-image `target_width=4000`,
image-uniform crop sampling을 결합한다. EfficientNet-B4 U-Net, crop 1536, batch 2, AdamW `3e-4`, CE+Dice,
fold당 16,000 updates 및 specimen-group 4-fold OOF 프로토콜을 유지한다. 이 결합 실험은 개별 효과 분리보다
세 기법의 조합이 80% 목표에 도달하는지 검증하는 후속 시도다.

Model328+Model270 route가 empty-union=0 valid `80%` 이상과 grouped-CV 95% CI 하한 `>0`을 함께 만족하지
못할 때만 실행하도록 Model328 완료 후 queue에 연결했다. 시작 전 P100 GPU 6의 14.5 GiB 여유를 3회 연속
확인하고 100-step 학습 및 full-image 추론 smoke를 통과시킨 뒤 본 학습에 진입한다. 학습·OOF·valid 완료 후
Model270 route를 CPU 평가한다.

초기 기록 정정: `predict.py`의 `scores.json` 필드 `mean_miou`는 NaN-skip이며 공식 empty-union=0이 아니다. 따라서 이 절에 먼저 적었던 fold1 `77.9351% / 76.8802%`와 `+1.0549%p`는 NaN-skip 수치였고 empty=0으로 잘못 표기했다. 아래 값은 저장된 예측 mask와 GT mask를 직접 다시 비교해 official empty=0으로 산출했다.

fold 1의 동일 held-out 18장에서 Model322 `75.9170%` → Model329 `76.9348%`, paired `+1.0178%p` (10/18장 개선)였다. fold 2 18장은 Model322 `77.3068%` → Model329 `78.4686%`, `+1.1618%p` (11/18장 개선)였다. 두 fold 36장 합계는 Model322 `76.6119%` → Model329 `77.7017%`, paired `+1.0898%p`, 21/36장 개선, 30 specimen groups, group-bootstrap 95% CI `[+0.4014,+1.8531]%p`다.

클래스 순서 `[Al, Al3Ni, Primary Si, Eutectic Si]`에서 paired class-IoU 변화는 각각 `+0.192`, `+1.761`, `+0.187`, `+2.219%p`다. 95% group-bootstrap CI는 Al `[-0.135,+0.580]`, Al3Ni `[+0.714,+2.992]`, Primary Si `[-0.317,+0.788]`, Eutectic Si `[+0.853,+3.718]%p`다. 이는 train-only OOF의 두 fold 중간 결과(36/70장)이지 valid 점수나 최종 grouped OOF 결론이 아니며, 전체 Model329 성능 또는 80% 달성을 뜻하지 않는다.

재현 코드: `model340_partial_oof_metric_audit.py`; JSON: `/workspace/komap_model1/outputs/model340_partial_oof_audit/folds1_2.json`.

실행기: [run_model329_hybrid_width4000_uniform_group_oof.sh](run_model329_hybrid_width4000_uniform_group_oof.sh)

## 모델 330: 팀 train 픽셀 빈도 기반 class-prior correction (미채택)

팀원이 설명한 방식대로 train mask 70장의 클래스 픽셀 비율을 구해 역수를 class weight로 두고, 현재 최고 valid route인 Model327의 최종 클래스 확률을 weight로 나눈 뒤 argmax했다. 클래스 순서는 팀 `[Al, Al3Ni, Primary Si, Eutectic Si]`에서 route `[Al3Ni, Primary Si, Al, Eutectic Si]`로 재배열했다. 기존 route의 최종 logits에 `-log(class_weight)`를 더하는 것은 확률 나눗셈 후 argmax와 동치다.

train 픽셀 비율(팀 순서)은 `[57.28%, 26.31%, 9.43%, 6.98%]`, 역빈도 weight는 `[1.746, 3.800, 10.606, 14.333]`이다. 재현 검증에서 보정 전 Model327 점수가 기록과 정확히 일치했다: image-mean mIoU, empty-union=0 기준 `79.1898%`(NaN-skip `80.0330%`). 보정 적용 후 `74.8745%`(NaN-skip `75.6938%`)로 각각 `-4.3153%p`, `-4.3392%p` 하락했다.

클래스별 image-mean IoU(empty=0)는 Al3Ni `76.36→74.41%`, Primary Si `86.37→85.13%`, Al `94.01→91.96%`, Eutectic Si `60.02→48.00%`였다. 특히 이미 약한 Eutectic Si에서 역빈도 weight 나눗셈이 예측을 크게 억제해 악화됐다. 따라서 이 보정은 Model327에 적용하지 않는다. 이는 valid 20장 결과이며 test 점수는 아니다.

모델 329 번호는 이미 full-stat + width4000 + uniform 학습 대기 작업에 배정되어 있어, 이 진단은 모델 330으로 정정했다.
평가 스크립트: [model330_train_prior_correction.py](model330_train_prior_correction.py)
결과: `/workspace/komap_model1/outputs/model330_train_prior_correction/result.json`

## 모델 331: B4 class-wise 혼합률 미세 탐색 (미채택)

157개 route를 비교했다. raw 최고는 Model322 B4를 Al3Ni `0.50`, Primary Si `0.70`, Al `0.80` 혼합하고 Eutectic Si에는 Model269 `0.50`을 유지한 설정이다. valid 20장 점수는 empty=0 `79.1914%`, NaN-skip `80.0359%`로 Model327보다 각각 `+0.0016%p`, `+0.0029%p`뿐이다. Model270 대비 specimen-group CV 변화는 empty=0 `+0.441%p` (95% CI `[-0.070,+0.960]`, 16/20장 개선), NaN-skip `+0.483%p` (95% CI `[-0.112,+1.156]`)로 CI가 모두 0을 포함한다. 따라서 미채택이며, NaN-skip 숫자만으로 80% 달성 또는 robust 개선이라 하지 않는다.

결과: `/workspace/komap_model1/outputs/model331_b4_fine_gamma_route/route_probe.json`

## 모델 332: Model322 grouped-OOF Eutectic Si 오차 기하 감사 (CPU, 진단)

공식 validation은 전혀 열지 않고, Model322의 70장 train grouped-OOF 확률과 train masks만 사용했다(59 specimen groups, class order `[Al, Al3Ni, Primary Si, Eutectic Si]`, empty-union IoU=0). 전체 image-wise mIoU는 `78.5346%`; 이미지별 class-IoU 평균은 Al `92.7727%`, Al3Ni `79.3328%`, Primary Si `86.3145%`, Eutectic Si `55.7186%`다. Eutectic Si class mean의 specimen-group bootstrap 95% CI는 `[52.3836%, 59.1447%]`다.

Eutectic GT 내부 경계를 8-connectivity로 추출하고 2px 팽창한 띠에서 GT Eutectic pixels의 `95.71%`, Eutectic false-negative pixels의 `93.85%`가 나타났다. 반면 Eutectic false-positive pixels 중 그 띠 안에 든 비율은 `36.72%`뿐이다. 즉 얇은 Eutectic 상의 누락은 경계 주변에 집중되지만 오탐의 다수(`63.28%`)는 GT Eutectic 경계에서 2px보다 떨어져 있어 경계 손실만으로 설명되지 않는다. 경계 띠/바깥의 pooled Eutectic IoU는 각각 `65.2259%`/`12.8812%`다.

가장 큰 양방향 혼동은 Al3Ni다. GT Eutectic을 놓친 픽셀은 Al3Ni `746,385`(FN의 약 `53.3%`), Al `402,633`, Primary Si `250,904`개다. 예측 Eutectic 오탐의 원래 GT는 Al3Ni `603,093`(FP의 약 `53.6%`), Al `265,838`, Primary Si `256,151`개다. 따라서 다음 해석은 Eutectic needle boundary와 Al3Ni 질감/상 구분을 함께 봐야 하며, 단순 class bias나 경계 보조 loss의 반복으로 해결된다고 가정하지 않는다. Model307의 확률-only expected-IoU decoder는 OOF 및 valid에서 악화했고, Al3Ni–Eutectic pairwise margin/clDice/boundary 실험도 과거에 robust 개선을 보이지 않아 재실행하지 않는다. 현재 대기 중인 width4000+uniform 실험이 상/조직 크기 표현을 바꾸는 효과를 먼저 검증하며, Model326/328 OOF 완료 뒤 동일 감사를 반복 비교한다.

구현: [model332_eutectic_oof_error_geometry.py](model332_eutectic_oof_error_geometry.py)
결과: [model332_eutectic_error_geometry.json](model332_eutectic_error_geometry.json), `/workspace/komap_model1/outputs/model332_eutectic_error_geometry/audit.json`

Claude Code Opus 자문(읽기 전용)도 반영했다. 제안된 확률-only per-image expected-IoU decoder는 Model307에서 이미 grouped OOF `-1.0252%p`, valid `-1.3139%p`로 악화했고, 이미지별 bias oracle도 Model172에서 최대 `+0.5260%p`에 그쳐 재실행하지 않는다. 지표 정렬형 loss는 Model283/284/286에서 valid standalone 소폭 상승이 있었으나 기존 route와 grouped-CV 일반화 이득은 확인되지 않았다. 현 Model326→328→조건부329 queue가 끝나도 robust 80% gate를 통과하지 못하면 후속 후보(Model333)는 Model322 각 fold checkpoint를 시작점으로 per-image Lovasz fine-tune과 동일 update 수 CE+Dice control을 4-fold paired OOF로 비교한다. 채택 조건은 specimen-group bootstrap CI 하한 `>0`, Eutectic Si class IoU 상승, 다른 class 평균 하락 `0.2%p` 이내이며 official valid는 OOF에서 선택한 설정의 최종 1회 확인에만 쓴다. HRNet seed ensemble은 Model327 OOF 확률 부재로 현 단계에서 뒤로 미룬다.

## 모델 333: Model322 B4 checkpoint paired Lovasz fine-tuning (조건부 대기)

Model326/328/329 이후에도 robust 80% gate를 통과하지 못할 때만 실행하도록 준비했다. 각 fold에서 동일한 Model322 checkpoint로부터 두 branch를 초기화해 loss 효과를 분리한다: control은 CE+Dice, candidate는 CE+Dice+`0.25 ×` per-image present-class Lovasz-Softmax다. 두 arm 모두 EfficientNet-B4 U-Net, ImageNet normalization, gray3 입력, scale 3, crop 1536, batch 2, AdamW `2e-5`, 2,000 updates 및 동일 seed의 deterministic worker를 쓴다. Lovasz 계산량을 제한하기 위해 이미지당 131,072개 픽셀을 균등 복원추출해 surrogate를 계산한다.

4-fold specimen-group OOF를 각각 만들고, `evaluate_group_cv.py`로 Lovasz와 동일-init control의 grouped held-out route를 비교한다. Lovasz 확률은 Model270 route와도 CPU-only로 비교하며, empty-union=0 image-mean mIoU와 group bootstrap CI를 기록한다. valid는 모든 fold 학습·OOF 이후 평가하고 test 추론은 하지 않는다. trainer의 shell/Python 문법, Lovasz forward/backward, 원격 `--help`, Model322 fold checkpoint 766-tensor strict load를 확인했다.

실행기: `run_model333_b4_lovasz_group_oof.sh`; 조건부 대기기: `queue_model333_after328.sh`. 조건부 실행은 Model326→328→329 평가가 끝난 뒤 robust 80% gate를 확인한다. 코드는 `/workspace/komap_model1/model326_hybrid_fullstats/KoMaP/train_model333.py` 및 `imagewise_lovasz.py`에 있다.

## 모델 334: Model322/326 B4 확률 평균 + Model270 class-wise route

Model322 팀 B4와 Model326 full-stat B4의 valid 확률을 이미지별 50:50 평균해 후보 branch를 만들었다. 이를 Model270 route에 넣고 59개 class-wise 설정을 비교했다. 최고 설정은 `[Al3Ni, Primary Si, Al, Eutectic Si]` 순서로 B4 결합 branch를 `[0.50, 1.00, 0.50]` 비율로 섞고, Eutectic Si는 기존 Model269 branch `0.50`을 유지한 구성이다. 나머지 Model270 prior, beta, edge 및 cleanup 설정은 고정했다.

- valid 20장 image-wise mIoU: empty-union=0 `79.2353%`; empty-union 제외(NaN-skip) `80.0687%`.
- Model270 대비 specimen-group leave-one-group-out route 선택 결과: `+0.4642%p`, 15/20장 개선, group-bootstrap 95% CI `[+0.0433,+0.9137]%p` (19 groups).
- 이 route 후보 탐색은 같은 valid 20장의 점수를 활용했으며, NaN-skip 80%를 공식 empty=0 기준의 목표 달성으로 해석하지 않는다. 이 결과만으로 test 일반화가 입증된 것은 아니다.

결과: `/workspace/komap_model1/outputs/model334_b4_equalblend_route/route_probe.json`; 확률 입력: `/workspace/komap_model1/outputs/model334_b4_equalblend_route/candidate/probabilities.npz`

## 모델 335: Model270 × B4 branch 2,401개 exact route 전수 채점

Model322, Model326, 그리고 두 모델의 50:50 평균 확률을 Model270 route의 각 class에 독립적으로 적용하는 모든 discrete 조합을 채점했다. class마다 기존 Model270 유지 또는 세 branch 중 하나를 gamma `0.50`/`1.00`으로 혼합하는 7개 선택지를 두었고, 4개 class 전체 `7^4 = 2,401` routes를 만들었다. Eutectic Si도 포함했다. 모든 조합은 동일한 train-only prior/calibration과 exact Gaussian·bilateral spatial refinement 및 Primary component cleanup을 거친다. 점수 산출은 CUDA를 비활성화한 CPU 16-worker 실행이며 train/inference checkpoint는 새로 만들지 않았다.

| 선택 기준 | Al3Ni | Primary Si | Al | Eutectic Si | Global mIoU | Image-wise empty=0 | Image-wise empty 제외 |
|---|---|---|---|---|---:|---:|---:|
| strict image-wise 최고 | 322/326 평균 `0.50` | 326 `1.00` | 322 `0.50` | Model270 유지 | 81.8794% | **79.2903%** | 80.1248% |
| NaN-skip / global 최고 | 322/326 평균 `0.50` | 326 `1.00` | 322 `1.00` | Model270 유지 | **81.9359%** | 79.2898% | **80.1316%** |

Model270의 기존 Eutectic 설정은 Model269 gamma `0.50`이며 그대로 유지됐다. strict 최고 route는 Model334 대비 empty=0 `+0.0549%p`, NaN-skip `+0.0561%p`; Model270 기준 대비 각각 `+0.5693%p`, `+0.6270%p`였다. Global 최고 route는 Model334 global 최고 `81.8677%`보다 `+0.0682%p`다. Empty-union=0 기준 공식 목표 80%에는 아직 `0.7097%p` 부족하다.

평가 정책 차이 원인을 valid mask·route score 배열로 추가 확인했다. 20장 중 A71 한 장만 strict와 제외 점수가 다르며, 해당 mask에는 Primary Si가 없다. NaN-skip 최고 route에서 A71은 strict `50.5104%`, empty union 제외 `67.3471%`로 전체 점수 차이 `0.8418%p`를 설명한다. 이 Primary Si는 양쪽 모두 GT·예측 union이 0이므로 empty=0 정책 아래 이 클래스 점수는 개선 대상이 아니며, A71의 다른 세 class IoU와 나머지 이미지 개선이 필요하다.

19개 validation specimen group을 leave-one-group-out으로 route 선택한 held-out 비교도 기록했다. Model270 대비 image-wise strict 변화는 `+0.4564%p`(14/20장 양수, group-bootstrap 95% CI `[+0.0456,+0.8978]%p`), NaN-skip은 `+0.5060%p`(15/20장 양수, CI `[+0.0524,+1.0264]%p`)였다. Pooled global 비교 변화는 `+0.5484%p`(CI `[+0.2796,+0.8593]%p`)다. 이 검증은 route 선택 일반화를 보조 확인하지만, 2,401개 grid를 같은 validation pool로 비교했으므로 독립 test 평가가 아니다. Model328/329 같은 다른 branch는 이번 grid에 포함되지 않았다.

구현: `model335_exhaustive_b4_route_search.py`
결과: `/workspace/komap_model1/outputs/model335_exhaustive_b4_routes/results.json`; 로그: `/workspace/komap_model1/outputs/model335_exhaustive_b4_routes/run.log`

## 모델 336: 팀 HRNet-W48 + U-Net++ 레시피 재현 (재실행 중)

비공개 [`CNUKoMaP/HRNet_best`](https://github.com/CNUKoMaP/HRNet_best) commit
`97adcdf337ab78fb7ec1469fe48943ff9a6363ea`의 모델 구조와 핵심 학습법을 별도 디렉터리에 보존해
재현한다. 구조는 ImageNet 사전학습 HRNet-W48 encoder + SMP U-Net++ decoder와 학습 전용
class-agnostic boundary head다. 256px 원본 crop을 512px로 2배 확대하고, 희소상 중심 patch sampling,
unweighted `0.5 CE + 0.5 Dice + 1.0 boundary BCE/Dice`, AdamW `3e-4` (encoder/decoder 동일 LR),
cosine schedule 70 epochs를 적용한다. P100에서는 micro-batch 4, accumulation 2로 effective batch 8을
맞춘다. 추론은 원본 256px tile/170px overlap, Gaussian stitching, horizontal/vertical flip 4-way TTA다.

팀 README의 valid 20장 `81.77%`는 pooled/global confusion-matrix mIoU이고 image-mean 점수가 아니다.
따라서 이번 결과는 우리 공식 이미지별 4-class 산식으로 다시 계산하며, empty-union=0/제외/1 점수와
pooled 점수를 명시적으로 분리한다. 우선 `train`으로만 학습하고 `valid`를 최종 평가하며, 마지막에
`test` 입력에 대한 마스크를 생성한다. 기존 Model270/B4 route에 대한 유효성은 저장한 valid 확률맵으로
별도 탐색하되, valid로 고른 최고점은 독립 test 성능처럼 해석하지 않는다.

기존 HRNet 실험은 Model8 HRNet-W18 단순 FPN, Model14 W32 context head, Model15 W48 context head,
Model48 W18 native patch였으며, 이번의 HRNet-W48 + U-Net++ 조합과 달랐다. 서버 환경에서 HRNet-W48
U-Net++ forward shape `[1,4,64,64]` 확인, 약 76.96M parameter 구조 생성, pretrained encoder 로드를
확인했다. 첫 실행은 epoch 1 뒤 inference checkpoint만 기록하고, resume state 저장 중 `random`/NumPy import
누락으로 중단됐다. optimizer state가 남지 않아 정확한 resume는 불가능했다. RNG import와 seed 초기화를 수정해
새로 시작했고, 새 실행의 epoch 1 train state 저장까지 확인했다. 70-epoch 학습과 official valid 평가는 현재 진행 중이다.

관련 `sppark` commit `e0d71b1`은 현재 DINOv3 global-context + local-detail CNN 코드를 포함하고, 문서에는
HRNet 특징과 stride-2/stride-1 detail 경로가 설계 후보로 적혀 있다. 보관된 `accuracy_v1` CNN baseline은
image-mean `78.4729%`지만, 해당 HRNet 설계는 측정 결과가 아니다. 이번에는 먼저 팀의 완성된 HRNet-W48
레시피를 통제된 branch로 재현한다. 성능 확인 뒤 필요하면 sppark식 원본 전체 문맥/로컬 세부 융합과 Model270
class-wise 확률 route를 차례로 비교한다.

Model336 평가 후에는 Model337 CPU-only route sweep을 잇는다. 네 상마다 Model270 유지 또는 HRNet 확률을
`0.25/0.50/0.75/1.00` 비율로 섞는 `5^4=625` 조합을 정확한 공간 후처리까지 채점한다. 사전에 이미지 순서,
정답 마스크 일치, 클래스 순서 및 확률 합을 확인하며, raw valid 최고점과 specimen-group leave-one-group-out
선택 결과를 나눠 기록한다.

코드/설정: [model336_hrnet_best](model336_hrnet_best/)
route 평가기: [model337_hrnet_model270_routes.py](model337_hrnet_model270_routes.py)
서버 결과 예정 경로: `/workspace/komap_model1/outputs/model336_hrnet_best/`

## 모델 338: Model335 expert 혼합률 fine-sweep

Model335 strict-best route의 네 class 확률원을 고정하고 한 class씩 혼합률을 `0.1`~`1.0`으로 바꾸는 38개 고유 설정을 exact CPU 평가했다. 모델 학습이나 새 추론은 없고, 저장된 확률맵과 동일한 calibration/spatial refinement를 재사용했다.

공식 empty-union=0 최고는 **79.3400%**로, Model335 strict-best `79.2903%`보다 `+0.0497%p` 상승했다. 80% 목표에는 아직 `0.6600%p` 부족하다.

| 상 | 확률원 | 혼합률 |
|---|---|---:|
| Al3Ni | Model322/326 평균 | 0.50 |
| Primary Si | Model326 | 1.00 |
| Al | Model322 | 0.50 |
| Eutectic Si | Model269 | **0.90** |

참고로 NaN-skip 최고는 `80.1685%` (Eutectic gamma `0.80`), pooled global 최고는 `81.9667%` (strict-best와 같은 gamma `0.90` route)다. 공식 순위에는 strict empty=0만 사용한다. Model338 strict-best의 A71 strict score는 `49.6248%`, empty class 제외 score는 `66.1664%`; mask에서 비어 있고 모델도 예측하지 않은 Primary Si가 정책 차이의 원인이다.

19개 specimen group leave-one-out route 선택은 Model270 대비 strict `+0.5884%p`, 18/20 images positive, group-bootstrap 95% CI `[+0.2305,+0.9916]%p`였다. Route 탐색이 같은 valid pool에 의존하므로 독립 test 일반화 점수는 아니다. Model329 branch는 아직 포함되지 않았다.

코드: `model338_fine_gamma_route_search.py`; 결과: `/workspace/komap_model1/outputs/model338_fine_gamma_route_search/results.json`; 로그: `/workspace/komap_model1/outputs/model338_fine_gamma_route_search/run.log`

## 모델 339: Model338 네 클래스 혼합률 동시 탐색 (완료, 채택 보류)

Model338은 한 번에 한 클래스의 혼합률만 바꿨기 때문에 클래스 간 확률 정규화·argmax 상호작용을 탐색하지 못했다. Model338 공식 strict-best의 확률원과 calibration/spatial refinement는 고정하고 네 클래스 혼합률을 동시에 바꾸는 400개 조합을 CPU exact scorer로 비교한다. 기준에는 Model270 원본 route를 포함한다.

| 클래스 | 고정 확률원 | 탐색 gamma |
|---|---|---|
| Al3Ni | Model322/326 평균 | `0.3–0.7`, 0.1 간격 |
| Primary Si | Model326 | `0.7–1.0`, 0.1 간격 |
| Al | Model322 | `0.3–0.7`, 0.1 간격 |
| Eutectic Si | Model269 | `0.7–1.0`, 0.1 간격 |

각 조합을 exact spatial refinement까지 적용해 점수화하고 official image-wise empty-union=0, empty 제외 참고값 및 leave-one-specimen-group-out 선택 결과를 계산한다. 탐색 점수는 같은 valid 20장에서 생성되므로 raw 최고는 validation 선택 편향을 포함하며, grouped held-out CI와 함께 판정한다. 새 학습이나 GPU 추론은 하지 않는다. 사전 5-route smoke/preflight는 통과했다.

401개 route 탐색 공식 최고는 **79.3683%** (NaN-skip `80.2062%`, pooled global `82.0064%`)이며, Model338의 strict `79.3400%`보다 raw valid에서 `+0.0283%p`다. 최고 설정은 Al3Ni Model322/326 평균 `0.6`, Primary Si Model326 `1.0`, Al Model322 `0.7`, Eutectic Si Model269 `0.8`이다.

하지만 Model338 strict route를 고정 비교군으로 둔 19-group leave-one-specimen-out route selection 변화는 `−0.0139%p`, 개선 이미지 `9/20`, group-bootstrap 95% CI `[-0.0613,+0.0352]%p`로 0을 포함한다. 따라서 `+0.0283%p`는 같은 valid set의 raw 탐색 최고일 뿐이고, 일반화 개선 근거가 없어 Model338을 보수적 채택 기준으로 유지한다. Model339 route는 최고 관측값 표에는 남기되 안정적으로 우월한 route로 채택하지 않는다.

탐색은 CPU-only 저우선순위로 완료했다. 당시에는 Model329 branch를 포함하지 않았으며, 이후 완료된 Model329와의 626-setting 결합 탐색은 Model341에 기록한다.

코드: `model339_joint_gamma_route_search.py`; 직접 비교 감사 코드: `model339_grouped_cv_parent_audit.py`; 결과: `/workspace/komap_model1/outputs/model339_joint_gamma_route_search/results.json`; Model338 paired group-CV: `/workspace/komap_model1/outputs/model339_joint_gamma_route_search/cv_vs_model338.json`

## 모델 340: Model329 full OOF 공식 지표 재감사 (완료)

Model329의 초기 fold 비교에서 `predict.py` 결과 JSON의 `mean_miou`가 NaN-skip인데도 empty-union=0으로 잘못 표기된 것을 확인했다. Model340은 Model322와 Model329의 저장된 예측 mask를 각 fold GT와 직접 다시 비교하고, union=0 class IoU를 0으로 두어 공식 image-wise 지표를 재계산했다.

fold1은 18장 기준 Model322 `75.9170%` → Model329 `76.9348%` (`+1.0178%p`, 10/18장 개선), fold2는 18장 기준 `77.3068%` → `78.4686%` (`+1.1618%p`, 11/18장 개선)이다. 두 fold 합계 36/70장은 `76.6119%` → `77.7017%`, paired `+1.0898%p`, 21/36장 개선, 30 specimen groups, group-bootstrap 95% CI `[+0.4014,+1.8531]%p`였다.

`[Al, Al3Ni, Primary Si, Eutectic Si]` 평균 class-IoU 변화는 `[+0.192,+1.761,+0.187,+2.219]%p`다. Al3Ni CI `[+0.714,+2.992]`, Eutectic Si CI `[+0.853,+3.718]%p`는 양수이고, Al/Primary Si CI는 0을 포함한다. 이 partial audit 이후 네 fold 모두 완료해 70장 train OOF와 20장 valid를 재평가했다. Model329 단독 valid 공식 empty=0 image-mean은 **78.8334%** (absent class 제외 참고값 `79.5353%`), class별 `[Al, Al3Ni, Primary Si, Eutectic Si]`는 `[94.052,75.131,86.482,59.669]%`다. Train OOF 단독은 `79.1233%`; OOF에서 선택한 네 클래스 gamma `0.5`는 `79.3974%`다. Model303 parent 대비 specimen-group held-out route 선택은 `+1.8875%p`, 67/70장 개선, 59 groups, group-bootstrap 95% CI `[+1.4921,+2.2940]%p`였다.

Model270 expert 경로에 Model329를 섞은 별도 exact route probe의 strict valid 최고는 **79.2615%** (NaN-skip 참고 `80.07697%`)다. Model270 대비 group-held-out 선택 변화는 `+0.4305%p`, 17/20장 개선, 19 groups, 95% CI `[+0.1571,+0.7144]%p`로 양수지만 80% 목표 및 기존 raw 최고 Model339 `79.3683%`에는 미달한다. 따라서 이 점수만으로 기존 최고를 교체하지 않는다.

Model329 학습/OOF/valid 및 Model270 probe는 완료했다. 이후 Model341–345의 결과는 아래 절에 기록한다. Model336 HRNet-W48/U-Net++는 trainer 오류를 수정해 새 학습을 시작했다.

재현 코드: `model340_partial_oof_metric_audit.py`; partial audit: `/workspace/komap_model1/outputs/model340_partial_oof_audit/folds1_2.json`; full OOF/valid: `/workspace/komap_model1/outputs/model329_fullstats_width4000_uniform_group_oof/results.json`; Model270 route: `/workspace/komap_model1/outputs/model329_model270_fusion/route_probe.json`.

## 모델 341: Model329를 추가한 Model339 클래스별 route 탐색 (완료, 현재 최고)

Model329 grouped-OOF/valid 확률을 기존 Model339 route에 혼합하고, 네 class gamma를 `{0,0.25,0.50,0.75,1.0}`에서 탐색한 625개 조합과 Model338 reference를 exact CPU scorer로 평가했다. Official valid strict 최고는 **79.5407%**이며 Model339 base + Model329 gamma `[0.00,0.50,0.25,0.25]` (order `[Al3Ni, Primary Si, Al, Eutectic Si]`)다. 80% 목표에는 `0.4593%p` 부족하다. Model339 unchanged 대비 specimen-group held-out 변화 `+0.1261%p` (15/20장, 95% CI `[+0.0087,+0.3001]%p`), Model338 대비 `+0.1544%p` (14/20장, CI `[+0.0232,+0.3288]%p`)다. 같은 validation pool에서 선택한 route이므로 독립 test 결과는 아니다.

## 모델 342: train grouped-OOF로 선택하는 Model322/329 blend (완료)

Model322와 Model329의 70장 exactly-once train OOF만으로 gamma `[Al,Al3Ni,Primary Si,Eutectic Si]=[0.50,1.00,0.25,0.75]`를 선택했다. OOF blend는 `79.4337%`; specimen-group held-out은 Model322 대비 `+0.8720%p` (51/70장, 95% CI `[+0.4640,+1.2964]%p`). 고정 gamma valid `78.8936%`로 Model341보다 낮다.

## 모델 343: Model342 expert를 Model341 route에 추가 (완료, 미채택)

625 gamma 조합의 raw valid 최고는 gamma `[0,0,0,0]`의 Model341 그대로 `79.5407%`였다. Group-held-out 변화는 `−0.0089%p` (0/20 개선, 95% CI `[-0.0281,0.0000]%p`)여서 Model342 expert를 추가하지 않았다.

## 모델 344: Model322 + Model333 CE+Dice control OOF blend (완료)

Train OOF에서 선택한 class gamma `[Al,Al3Ni,Primary Si,Eutectic Si]=[1,1,0,1]`다. OOF `78.7587%` (Model322 parent `78.5346%`); specimen-group held-out `+0.2221%p` (48/70장, 59 groups, 95% CI `[+0.1090,+0.3650]%p`). Gamma 고정 valid는 `77.9186%`로 Model341보다 낮다.

## 모델 345: Model344 expert를 Model341 route에 추가 (진행 중)

Model341 strict-best를 고정하고 Model344 train-OOF-fitted expert를 네 class별 gamma `{0,0.25,0.50,0.75,1.0}`로 혼합하는 625-route exact CPU 탐색이다. Gamma 전부 0일 때 Model341의 20개 이미지별 점수와 `79.5407%` 평균을 smoke test에서 정확히 재현했다. Full search는 진행 중이고, 결과 전 최고는 Model341이다.

## 모델 346: Model336 HRNet expert를 Model341 route에 추가 (큐 대기)

Model337은 HRNet을 Model270에만 결합하므로, Model346은 Model341 strict-best route에 Model336 HRNet-W48 + U-Net++ probability map을 네 class별 gamma `{0,0.25,0.50,0.75,1.0}`로 넣는 625-setting exact CPU 탐색이다. Team channel order `[Al, Al3Ni, Primary Si, Eutectic Si]`는 내부 route order `[Al3Ni, Primary Si, Al, Eutectic Si]`로 바꾼다. 모든 probability shape·합, 이미지 순서와 validation target을 확인하고, gamma 전부 0인 설정이 Model341 이미지별 결과와 `79.5407%`를 재현하는지 먼저 검사한다.

Model336 train/inference·Model337 route와 Model345가 끝난 뒤 순차 실행하도록 큐에 연결했다. 결과 전에는 Model341을 최고로 유지한다. 코드: `model346_model336_into_341_route.py`; runner: `queue_model346_after336_and345.sh`; 결과 경로: `/workspace/komap_model1/outputs/model346_model336_into_341_route/results.json`.

## 모델 376: HRNet-W48/U-Net++ 경계 가중 CE (4-fold grouped OOF 완료, 미채택)

Model374의 70장 train OOF에서 잘못 분류된 픽셀의 `86.06%`가 GT phase boundary 1px band 안에 있었고, 그 band의 오류 밀도는 내부의 `8.27배`였다. 이를 겨냥해 모델 구조·patch 입력·optimizer·sampling·CE+Dice·class-agnostic boundary head는 Model374와 그대로 두고, CE에만 GT 경계 거리 기반 pixel weight를 추가한다. 이는 추론 시 후처리나 valid/test 기반 라우팅이 아니다.

각 학습 patch의 GT에서 4-neighbor class boundary까지 거리 `d`를 구하고, `d≤3 px`에서 `1 + λ exp(−d²/(2·1.5²))`를 적용한다. `λ`는 최대 `2.0`까지 첫 10% update 동안 선형 증가시키며, patch별 pixel weight 평균을 1로 정규화해 CE 전체 크기를 유지한다. Dice 및 boundary-head loss는 바꾸지 않는다.

4개 specimen-group fold 각각 70 epochs, HRNet-W48 + U-Net++, 512px patch / input scale 2, 12 patches/image, batch 4 × accumulation 2, AdamW `3e-4`, seed 0으로 학습한다. 각 fold의 held-out train images만 OOF 평가에 쓰고, 중단 복구 checkpoint는 Colab Drive에 별도 저장한다. 공식 valid 20장과 test 10장은 이 실험의 학습·선택·채택 기준에 사용하지 않는다.

- 네 fold OOF mIoU는 Model374 `79.2823%`, Model376 `79.4989%`로 `+0.2166%p` (59-group bootstrap 95% CI `[-0.1182,+0.5688]`)이며, 채택 gate를 통과하지 못했다. Fold1–4 변화는 `+0.7642/−0.1054/+0.2020/−0.0077%p`; Eutectic Si mean IoU `+0.5802%p` (group CI `[+0.0134,+1.2290]`) 외 나머지 class CI는 0을 포함한다. 네 fold OOF maps, paired summary, checkpoint epoch70, Drive fold completion manifests를 확인했다. valid/test labels는 학습·선택에 사용하지 않았다.

## 모델 377: HRNet-W48/U-Net++에 방향성 Eutectic→Al3Ni 손실 추가 (4-fold grouped OOF 완료, 미채택)

Model376과 같은 recipe/fold/seed를 쓰고, GT Eutectic Si 중 axial 4-neighbor GT Al3Ni와 맞닿은 픽셀에서만 `β·[logsumexp(all logits)−logsumexp(logits except Al3Ni)]`를 추가했다. β=`2.0`은 첫 10% update 동안 warm-up했다. 각 fold 70 epochs, 70 train images를 exactly once OOF 예측했다.

- Official empty=0 image mean: Model374 `79.2823%`, Model376 `79.4989%`, Model377 `79.3171%`.
- Model377−374 paired OOF `+0.0348%p`, specimen-group bootstrap 95% CI `[-0.4730,+0.4828]`; fold delta `+0.8826/+0.0049/−0.6467/−0.1498%p`. 효과 크기·3/4 fold·최악 fold·CI 네 채택 gate를 모두 통과하지 못했다.
- Model377−376 `−0.1818%p` (95% CI `[-0.6558,+0.1727]`). Class delta vs374 `[Al −0.0093, Al3Ni −0.2725, Primary Si −0.2028, Eutectic Si +0.6236]%p`.
- 방향성 기전은 Eutectic→Al3Ni 경계 오류 `−7.0755%p`, 역방향 Al3Ni→Eutectic `+6.0569%p`였다. 양방향 swap 합은 Model376 `53.7242%`에서 Model377 `52.7056%`로 `−1.0186%p` (group CI `[-1.7825,−0.2938]`)만 줄어 기전 gate `−2%p`에도 미달했다. 반대방향 오류 증가 등 다른 guard도 통과하지 못했다.
- **미채택**; validation/test 평가로 재선택하지 않는다. 결과: [experiments/results/model377_oof_evaluation.json](experiments/results/model377_oof_evaluation.json).

## 모델 379: Model374/376/377 고정 1/3 uniform OOF 앙상블 (완료, 미채택)

같은 held-out fold의 Model374/376/377 softmax 확률을 동일 가중치 `1/3`로 평균하고 argmax했다. Gamma, class bias, TTA, 후처리/라우팅 탐색은 하지 않았다. 70 train images/59 specimen groups만 쓴 selection-conditional OOF 평가다.

- Model374 `79.2823%` → Model379 `79.6857%`: `+0.4034%p`, paired group-bootstrap 95% CI `[+0.1963,+0.6318]`.
- Fold1–4 변화 `+0.8474/+0.2329/+0.1920/+0.3251%p`; class delta `[Al +0.1101, Al3Ni +0.2829, Primary Si +0.2791, Eutectic Si +0.9415]%p`; Step4 27 images/26 groups는 `+0.6874%p`.
- 사전고정된 여섯 adoption gate 중 다섯은 통과했지만, mIoU 최소 상승 `+0.50%p`에는 미달해 **미채택**. 이 평균을 valid/test 성능이라고 부르지 않는다.
- 재현: [experiments/model379_uniform_oof_ensemble.py](experiments/model379_uniform_oof_ensemble.py); 결과: [experiments/results/model379_uniform_oof_ensemble.json](experiments/results/model379_uniform_oof_ensemble.json).

## 모델 380: HRNet 조건부 라우트의 specimen-group 교차검증 (완료, 미채택)

Model341/339에서 valid 전체로 이미 정해진 anchor route를 고정한 채, Model346의 HRNet gamma336과 Model349의 gamma347 선택만 19개 specimen group LOGO로 교차검증했다. 따라서 과거 anchor 선택 편향은 남아 있으며, 결과는 전체 route pipeline의 독립 성능 추정치가 아니다. GPU 학습·test 데이터는 사용하지 않았다.

- 20장 valid의 conditional anchor image-mean은 `79.8277%`, cross-fit 추가 route는 `79.7539%`로 `−0.0738%p` (paired group-bootstrap 95% CI `[−0.2004,+0.0229]`); group-equal 변화 `−0.0625%p` (CI `[−0.1959,+0.0231]`). 10/20장·9/19 group만 개선해 이득 근거가 없으므로 추가 route는 **미채택·동결**한다.
- gamma336 선택은 A71에서만 full-valid modal choice와 달랐고, gamma347은 A11과 A71에서 달랐다. candidate-vs-anchor Al3Ni class-IoU delta `−0.2300%p`의 per-image 변화는 A71 `−2.4082pp`, A11 `−2.3573/+0.0028pp`였으며, 나머지 17장 평균은 `+0.0095pp`였다. 이는 **cross-fit candidate 전체(모든 group에 gamma347을 추가)와 gamma347=0 reference 간 차이**이지, full-valid 대비 달라진 gamma의 인과 효과 분해가 아니다. 차이가 소수 group에 집중된 것은 관찰되지만 일반화 가능한 routing bias의 증거로 해석하지 않는다.
- 별도 저장 score-matrix 분해에서 full-valid raw gamma347 추가 이득은 `+0.0155%p`였지만 cross-fit 이득은 `−0.0738%p`로, 관측된 선택 민감도 차이는 `+0.0893%p`였다. Cross-fit 선택 route와 full-valid route 자체의 점수 차이는 `−0.1556%p`이며 route coefficient가 다른 곳은 A11 두 장/A71 한 장뿐이었다(이미지별 mIoU 변화 `−0.7093/+0.0440/−2.4464%p`). 이는 같은 20장을 사용한 선택 민감도 진단이지 독립 성능 추정이 아니다. 추가 route 탐색은 동결한다.
- Claude Code Opus 검토도 추가 routing search를 중단하고 고정 anchor를 유지하라고 권고했다. 권장 결론은 “전이 가능한 이득 미확인, 해로움 미확정”이다. 이는 routing bias가 없다는 증명이 아니라 19 groups로 작은 subset 효과를 판별할 수 없다는 뜻이다. Model380 분석 코드/사전등록: [audit](experiments/model380_conditional_route_bias_audit.py), [선택 민감도 분해 코드](experiments/model380_fullvalid_crossfit_selection_decomposition.py), [protocol](experiments/model380_conditional_route_bias_preregistration.md); 결과: [conditional CV JSON](experiments/results/model380_conditional_route_bias_audit.json), [분해 JSON](experiments/results/model380_fullvalid_crossfit_selection_decomposition.json).

## 모델 382: HRNet seed-count control (P100 진행 중)

Model374와 같은 HRNet-W48/U-Net++ 학습 recipe를 seed 1·2, 4개 고정 specimen-group fold에 적용해 seed 변동과 모델 수 효과를 분리한다. 각 조합 70 epochs이며, train-only grouped OOF로만 판정하고 공식 valid/test를 중간 선택에 쓰지 않는다. P100 runner는 순차 실행 중이다. 2026-10-02 14:03 UTC 확인 기준 seed1/fold1–3 학습 및 OOF가 완료됐고 seed1/fold4 epoch `29/70`까지 학습됐다. seed1/fold1 train-OOF는 held-out 18장 기준 image-wise mIoU `75.8063%` (empty union=0)이며, 전체 OOF가 아니므로 모델 간 성능 결론에는 사용하지 않는다. 해당 학습 설정은 `val_every=0`; 공식 valid/test를 학습 중 확인하지 않았다. 최초 OOF가 생기기 전인 07:58:08Z에 E5 보조 arm 문서·평가기 hash-lock을 만들었다. 8개 학습/OOF 결과가 모두 모이기 전까지 앙상블 성능 결론은 내리지 않는다. 사전등록: [Model382 protocol](experiments/model382_seed_count_control_preregistration.md); [E5 보조 사전등록](experiments/model382_five_model_ensemble_addendum.md); [hash-lock receipt](experiments/results/model382_e5_preregistration_lock.json); [평가기](experiments/model382_oof_ensemble_compare.py) / [합성 smoke](experiments/model382_oof_ensemble_compare_smoke.py).

2026-10-02 17:34 UTC 업데이트: seed2/fold1 학습은 70/70 epoch를 완료하고 고정 held-out train fold OOF 추론 중이다. 점수/OOF 산출물은 확인하지 않았다. 별도 v2 provenance addendum은 seed2 OOF 파일 생성 전 17:00:41 UTC에 잠갔고, 70개 train-OOF mask 해시와 fold/group inventory(70 images/59 groups)를 고정했다. Step4 표본은 27장, A71은 이 train-OOF 집합에 없어 A71 제외 결과는 전체 집합과 동일한 민감도로 해석해야 한다. 로컬/P100 문법 검사와 synthetic smoke 및 lock source hash 검증은 통과했다. [v2 addendum](experiments/model382_two_seed_mean_sensitivity_v2_addendum.md), [평가기](experiments/model382_two_seed_mean_sensitivity_v2.py), [mask lock generator](experiments/model382_two_seed_mean_sensitivity_v2_lock.py), [v2 lock receipt](experiments/results/model382_two_seed_mean_sensitivity_v2_lock.json). 이는 성능 결과가 아니며 기존 Model382 primary/E5 protocol을 바꾸지 않는다.

Claude Code Opus 5.5 read-only review (2026-10-02 17:51 UTC; logs, predictions, metrics, valid/test 미열람): 70-mask hash/fold inventory, mIoU/gates, seed-fold manifest 검증과 score archive를 열기 전 preflight 순서는 타당하다고 확인했다. Medium 항목은 (1) A71이 없는 상황에서 `a71_excluded_delta_pp`가 전체 delta와 같은 값이라는 표시가 comparison 결과에 없어 독립 subset처럼 읽힐 수 있음, (2) 기존 seed1 checkpoint/manifest/archive hash가 v1/v2 receipt에 고정되지 않음, (3) 오타난/nonexistent `model382_root`도 lock 생성 검사를 통과할 수 있고 실제 root 경로가 receipt에 기록되지 않음이다. seed1 백업의 28개 파일을 P100 원본과 SHA-256 대조해 모두 일치함을 확인했으므로 최종 분석 직전에도 백업과 재대조한다. Low 항목은 preflight 뒤 입력 재열기(TOCTOU), 점수 계산 뒤 output 경로 충돌을 발견하는 순서다. 추가 권고로 원격 복사 뒤 mtime의 약한 증거성, NPZ byte hash만으로 배열 내용의 차이를 보장하지 못하는 점, A71 부재/failure-path smoke 누락, v1/v2 lock 시각 순서 검증 누락을 기록했다. 이번 run은 lock 생성·모델 산출·최종 평가를 모두 P100의 같은 경로에서 하며 v2 lock은 v1 lock 이후 생성된 것을 확인했다. seed1 원격/백업 해시는 전부 일치했다. 이미 고정한 v2 코드는 수정하지 않고, 위 지점은 결과 해석·재현성 한계로 명시한다.

2026-10-02 17:58 UTC 상태: P100 runner PID `972710`은 활성이고 seed2/fold2가 epoch `10/70`까지 학습 중이다. fold1 final checkpoint와 train-OOF manifest/probability/summary 파일 등 7개를 로컬에 백업했고 P100 원본과 SHA-256이 전부 일치했다. 완료된 fold의 optimizer `train_state.pt`는 이전 seed1 보존 정책과 동일하게 제외했다. 백업: [seed2 fold1](</Users/hyunseok/Library/Application Support/KoMaP/model382_backups/seed2_20261002/README.md>).

2026-10-02 18:03 UTC 개선 순서 정정: 직전 진행 메시지에서 Model387/388/389 신호가 없을 때 Lovasz를 후속 후보로 언급했지만, 기록상 Model333에서 EfficientNet-B4에 대해 이미 CE+Dice 대조군과 paired grouped-OOF를 수행했고 Lovasz 차이는 `+0.0040%p`, 95% CI `[-0.0084,+0.0171]%p`로 사실상 0이며 채택하지 않았다. 따라서 Lovasz를 자동 fallback으로 실행하지 않는다. HRNet OOF 감사가 명확한 metric/loss-specific 가설을 새로 뒷받침할 때에만 encoder transferability를 별도 근거로 재검토하고, 그 외에는 Claude Opus 사용량 제한 해제 뒤 새 비라우팅 후보를 다시 자문한다. 자세한 내용은 아래 [Model333 절](#모델-333-model322-b4-checkpoint-paired-lovasz-fine-tuning-조건부-대기)을 참조한다.

2026-10-02 18:04 UTC 준비 점검: Model387/388/389 분석 스크립트와 synthetic smoke를 다시 compile/run했고 세 smoke 모두 통과했다. 검사는 합성 fixture 전용이며 OOF/valid/test 데이터를 읽지 않았다. P100 runner는 활성이고 seed2/fold2 epoch `14/70`까지 확인했다.

2026-10-02 18:11 UTC 코드 점검(점수/OOF 미열람): P100 runner `972710`은 활성이고 seed2/fold2가 epoch `15/70`, step `100/156`에서 학습 중이다. 원격/로컬 `dataset.py` SHA-256은 동일(`f862a2c1a0d2120316e4e1938c0f6729fa9bf0d92e544e1b9b90969aa5cef861`)이지만 원격 `train.py`(`5ab3ca112ed394466bc4476da13040f0c932fa276a2bb477a5f206f7faadb162`)는 로컬(`ad3cd53d3cf73a28b63c4c9ad202232857e70e5287e858552277c90e290c6869`)과 다르므로 현재 실행을 로컬 코드만으로 완전히 재현할 수 있다고 가정하지 않는다. 원격 trainer에는 선택적 boundary-CE/directional-loss flags가 추가됐지만 runner가 이를 전달하지 않아 default weight는 0이다. 데이터셋은 `self.rng = random.Random(seed)`를 만들고 crop 선택에 이 객체를 사용하지만 `seed_worker`는 module-level `random`/NumPy만 재시드하며 worker별 dataset RNG는 재시드하지 않는다. `num_workers=4`인 현 실행에서 worker별 private RNG stream의 초기 상관/epoch 간 반복 및 checkpoint RNG 복원 범위가 crop 다양성에 미치는 영향을 확인할 코드-level 가설로 기록한다. 아직 실측된 학습 영향은 아니며 현재 run은 중단·수정하지 않는다. Claude Code에 read-only 검토와 합성 DataLoader RNG 재현 시험을 요청한 뒤, 필요하면 source hash를 고정한 matched paired OOF ablation으로만 검증한다.

2026-10-02 18:12 UTC 합성 CPU 검증: 원격의 실제 `seed_worker`와 PyTorch DataLoader를 쓰되 synthetic Dataset만 넣어 확인했다. 현재 worker init에서는 네 worker의 첫 private-RNG 값이 같았고, 다음 iterator/epoch에서도 반복됐다. 별도 임시 `worker_init_fn`에서 `get_worker_info().dataset.rng`를 `torch.initial_seed()`로 재시드하자 worker 간 스트림이 달라지고 다음 iterator에서 새 스트림을 얻었다. 이어 epoch 경계의 main torch RNG 상태를 저장·복원한 restart 합성시험에서 corrected 방식의 다음 epoch shuffle과 private draws가 정확히 일치했다. 이는 RNG 흐름/재개 가능성의 합성 동작 검증일 뿐 실제 crop 분포·학습 성능·전체 학습 checkpoint 재개 동등성의 개선 증거는 아니다. 코드 변경, 재학습, OOF/valid/test 접근은 하지 않았다.

2026-10-02 18:17 UTC 모니터링: P100 runner `972710`과 seed2/fold2 train process가 계속 활성이다. fold2 epoch `21/70`, step `100/156`; 지난 점검 이후 약 6 epoch 진행했고 관측 epoch 시간은 약 105초다. Model382 점수/OOF는 열지 않았다. Claude Code 재검토는 사용량 제한 갱신(기록상 05:50 KST) 이후에 하며, 그 전까지 추가 GPU 실험을 시작하지 않는다.

2026-10-02 18:24 UTC augmentation RNG 추가 점검(합성 CPU 전용): P100 환경의 Albumentations `2.0.8` `Compose`는 `py_random`/NumPy `random_generator`를 별도 보유하고 `set_random_seed()`가 내부 transform에 전파됨을 확인했다. 실제 `build_train_transform()`은 부모 프로세스에서 `seed=None`으로 한 번 만들어지고 현재 `seed_worker`는 `Compose.set_random_seed()`를 호출하지 않는다. Synthetic brightness transform으로 현재 init을 재현했을 때 네 worker의 첫 augmentation 결과가 모두 같고 다음 iterator에서도 반복됐다. dataset-private RNG와 `Compose` 양쪽을 `worker_info.seed`로 초기화하는 임시 worker init에서는 worker/epoch별 값이 달라졌다. 이어 main torch RNG 저장·복원 합성시험에서 고친 worker init이 다음 epoch의 shuffle, dataset RNG, augmentation 결과까지 정확히 replay됨을 확인했다. 실제 run과 같은 `batch_size=4`, `num_workers=4`로 한 추가 합성시험에서도 네 worker의 crop/augmentation RNG sequence가 worker 간 및 iterator 간 반복되고, 고친 초기화에서는 분리됐다. 실제 데이터/학습/점수는 접근하지 않았고 source는 수정하지 않았다. worker RNG 분리 및 augmentation 다양성이 일반화에 미치는 영향은 아직 미측정이며 Claude 검토와 사전등록된 paired OOF 비교가 필요하다.

2026-10-02 18:29 UTC 모니터링: P100 runner `972710` 활성, seed2/fold2 epoch `27/70` 완료. 직전 정기 점검 이후 약 5 epoch 진행했으며 epoch당 약 104–105초다. training log만 확인했고 Model382 OOF/점수는 열지 않았다.

2026-10-02 18:40 UTC 모니터링: P100 runner `972710` 활성, seed2/fold2 epoch `33/70` 완료. training log만 확인했고 Model382 OOF/점수는 열지 않았다. worker별 dataset/Albumentations RNG 재시딩은 아직 candidate hypothesis로만 유지한다.

2026-10-02 18:51 UTC 모니터링: P100 runner `972710` 활성, seed2/fold2 epoch `39/70` 완료. training log만 확인했고 Model382 OOF/점수는 열지 않았다. RNG 동작 결함은 성능 효과가 미측정인 해석 제한/후속 ablation 후보로 기록하며 Model387/388/389의 고정 분석 순서는 유지한다.

2026-10-02 19:01 UTC 모니터링: P100 runner `972710` 활성, seed2/fold2 epoch `45/70` 완료. training log만 확인했고 Model382 OOF/점수는 열지 않았다. Model387/388/389의 고정 분석 실행조건은 여덟 Model382 OOF 완료 뒤 그대로 적용한다.

2026-10-02 19:11 UTC 모니터링: P100 runner `972710` 활성, seed2/fold2 epoch `50/70` 완료 후 epoch `51/70` 진행 중이다. training log만 확인했고 Model382 OOF/점수는 열지 않았다. RNG 수정 후보는 Claude 검토 전 실행하지 않는다.

2026-10-02 19:15 UTC 재사용 smoke 추가: [synthetic worker-RNG regression test](experiments/worker_rng_reseed_smoke.py)를 만들고 P100의 KoMaP Python 환경(Python, PyTorch `2.6.0+cu118`, Albumentations `2.0.8`)에서 실행했다. 현재-init 반복 재현, candidate worker/epoch 스트림 분리, main torch RNG 복원 뒤 다음 epoch replay가 모두 통과했다. 테스트는 CPU synthetic arrays만 사용했고 model382 학습 프로세스·체크포인트·예측·점수에는 접근하지 않았다.

2026-10-02 19:22 UTC 모니터링: P100 runner `972710` 활성, seed2/fold2 epoch `56/70` 완료 후 epoch `57/70` 진행 중이다. training log만 확인했고 Model382 OOF/점수는 열지 않았다.

2026-10-02 19:32 UTC 모니터링: P100 runner `972710` 활성, seed2/fold2 epoch `62/70` 완료. 관측 epoch 시간은 약 104초이며 training log만 확인했다. OOF/점수는 열지 않았다.

2026-10-02 19:34 UTC seed 제어 점검(합성 only): 동일한 `random.seed`/`numpy.random.seed`/`torch.manual_seed`를 두 번 재설정하고 Albumentations `Compose(seed=None)`를 새로 만든 뒤, `Compose.py_random`, `random_generator`, child transform RNG draws가 같지 않음을 확인했다(둘 다 `Compose.seed is None`). 따라서 현재 `--seed`가 Compose 난수열을 완전히 고정한다고 볼 수 없다. worker-specific `set_random_seed(worker_info.seed)` 후보는 이전 synthetic restart replay를 통과했으나, 이 재현성 결함의 학습/mIoU 영향은 미측정이다.

2026-10-02 19:35 UTC smoke 갱신: 위 `Compose(seed=None)` global-seed 재현성 검사를 [synthetic worker-RNG smoke](experiments/worker_rng_reseed_smoke.py)에 포함했다. P100 PyTorch/Albumentations 환경에서 unseeded Compose, 기존 worker duplication, corrected worker/epoch separation 및 next-epoch replay 전체가 통과했다. Smoke는 synthetic-only이며 training/OOF/valid/test artifacts를 열지 않는다.

2026-10-02 19:43 UTC 모니터링: P100 runner `972710` 활성, seed2/fold2 epoch `68/70` 완료. training log만 확인했고 OOF/점수는 열지 않았다.

2026-10-02 19:54 UTC 전환 확인: seed2/fold2가 epoch `70/70`을 완료했고 runner는 fold2 OOF 추론/manifest 단계를 거쳐 seed2/fold3 학습을 시작했다. 추론 실행 사실만 process state로 확인했으며 OOF 로그·산출물·점수는 읽지 않았다. seed2/fold3 training log는 epoch `1/70`, step `50/159`다.

2026-10-02 19:57 UTC 보존/상태: seed2/fold2의 args, final checkpoint, train log, OOF manifest, probability archive, per-image CSV, summary JSON 총 7개를 로컬 [seed2 backup](</Users/hyunseok/Library/Application Support/KoMaP/model382_backups/seed2_20261002/README.md>)에 복사했고 P100 원본 SHA-256 7개와 모두 일치했다. 점수 포함 JSON/CSV/NPZ는 내용 열람 없이 보관·해시만 했다. `train_state.pt`는 seed1·fold1 정책대로 제외했다. README를 folds1–2로 갱신했다. P100 runner 활성, seed2/fold3 epoch `2/70` 완료; training log만 확인했다.

2026-10-02 20:05 UTC 모니터링: P100 runner `972710` 활성, seed2/fold3 epoch `6/70` 완료. training log만 확인했고 OOF/점수는 열지 않았다.

2026-10-02 20:16 UTC 모니터링: P100 runner `972710` 활성, seed2/fold3 epoch `12/70` 완료. training log만 확인했고 OOF/점수는 열지 않았다.

2026-10-02 20:27 UTC 모니터링: P100 runner `972710` 활성, seed2/fold3 epoch `18/70` 완료. training log만 확인했고 OOF/점수는 열지 않았다.

2026-10-02 20:37 UTC 모니터링: P100 runner `972710` 활성, seed2/fold3 epoch `23/70` 완료 후 epoch `24/70` 진행 중이다. training log만 확인했고 OOF/점수는 열지 않았다.

2026-10-02 20:49 UTC 모니터링: P100 runner `972710` 활성, seed2/fold3 epoch `30/70` 완료. training log만 확인했고 OOF/점수는 열지 않았다.

2026-10-02 20:57 UTC worker RNG 재사용 smoke 보완 및 재실행: 기존 추상 probe 대신 실제 `KomapPatchDataset`·`build_train_transform()` 경로에 임시 합성 PNG 한 쌍을 넣어 P100 환경(Python 환경의 PyTorch `2.6.0+cu118`, Albumentations `2.0.8`)에서 검증했다. 기존 `seed_worker`에서는 fork된 4개 worker의 첫 crop 좌표와 augmentation tensor가 같고 다음 epoch에도 재현됐다. 후보에서 `SeedSequence`로 crop RNG와 `Compose.set_random_seed()`용 stream seed를 서로 분리하자 worker별 crop이 갈리고 다음 epoch 스트림도 달라졌다. CPU torch RNG 복원으로 다음 synthetic shuffled epoch replay도 통과했다. 테스트는 임시 합성 데이터만 사용했으며 KoMaP 이미지, 체크포인트, 예측, OOF/score 파일을 열지 않았다. 이는 실제 데이터 파이프라인에서 RNG 상태 중복을 확인한 것이지 mIoU 영향의 증거는 아니다. `train.py`/활성 학습은 수정하지 않았고 CUDA/end-to-end resume는 검증하지 않았다.

2026-10-02 20:58 UTC 모니터링: P100 runner `972710` 활성, seed2/fold3 epoch `34/70` 완료. training log와 process 상태만 확인했고 OOF/점수는 열지 않았다. locked audits 전에는 worker RNG 수정이나 새 GPU 작업을 시작하지 않는다.

2026-10-02 21:09 UTC 모니터링: P100 runner `972710` 활성, seed2/fold3 epoch `40/70` 완료. training log와 process 상태만 확인했고 OOF/점수는 열지 않았다.

2026-10-02 21:14 UTC Claude Code Opus 읽기 전용 리뷰: crop sampler의 `random.Random(seed)`와 Albumentations `Compose(seed=None)` private RNG는 worker-init의 전역 Python/NumPy seeding으로 재설정되지 않으며, `persistent_workers=False`라 worker가 epoch마다 같은 parent 상태에서 출발한다. `Compose.set_random_seed(seed)`도 Albumentations 2.0.8에서 같은 seed를 모든 child에 전파한다. P100 합성 ReplayCompose 시험 400회에서 같은 child seed는 H/V flip 혼합 조합을 0회 만들었고(196회 둘 다 미적용, 204회 둘 다 적용), 서로 다른 파생 child seed에서는 네 조합이 모두 나타났다. 이는 RNG 구현/다양성 문제를 보인 것이며 mIoU 효과는 미측정이다. Opus는 variable-consumption crop RNG와 worker/batch 변경에 불변인 `(run_seed, epoch, sample_index)`별 crop/augmentation stream을 권고했다.

2026-10-02 21:15 UTC Model390 후보와 검증: 첫 로컬 복사본의 trainer/loss source가 P100의 Model382 source와 달랐음을 해시 대조로 발견해 성능 비교에 쓰지 않기로 하고, P100의 실제 Model336/374 training tree를 새 후보의 베이스로 다시 복사했다. candidate `train.py` 차이는 opt-in `--rng-mode`와 epoch 전달, `dataset.py`는 per-item stream 및 nested transform 고유 seed, `model.py`는 synthetic CPU smoke용 `encoder_weights=none` 처리뿐이다. 다른 학습/loss/evaluation core 파일은 P100 source hash와 일치하며 active tree는 수정하지 않았다. P100 CPU에서 합성 전용 regression suite가 legacy bitwise equality, worker 수 0/2/4·batch/order 불변성, epoch 분리/재현, H/V 네 조합, synthetic `train.py` epoch+resume를 모두 통과했다. 임시 합성 PNG/checkpoint만 썼고 CUDA는 비활성화했으며 실제 KoMaP data/OOF/score는 열지 않았다. **성능 효과는 아직 모른다.** 사전등록은 [Model390 protocol](experiments/model390_per_item_rng_preregistration.md), 코드 [candidate](model336_hrnet_rng_candidate), [synthetic smoke](experiments/rng_per_item_candidate_smoke.py); pre-score lock은 [receipt](experiments/results/model390_preregistration_lock.json)다. 모든 기존 Model382/387/388/389 순서와 gate가 끝나기 전 Model390 GPU 학습은 시작하지 않는다.

2026-10-02 21:15 UTC 모니터링: P100 runner `972710` 활성, seed2/fold3 epoch `43/70` 완료 후 epoch `44/70` 진행 중. train log/process만 확인했고 OOF/점수는 열지 않았다.

2026-10-02 21:27 UTC Claude Code Opus Model390 사전 감사 후 보완: resume 시 `rng_mode`가 default legacy로 바뀌는 실수를 막도록 candidate trainer가 checkpoint 저장 args와 현재 training args 전체(`output_dir`, `resume` 제외)를 비교하고, 일치 확인 전 `args.json`을 덮어쓰지 않도록 했다. 사전등록에 처치가 worker/epoch stream 독립성 구조를 바꾼다는 점, 학습 seed/cuDNN·AMP noise가 group CI에 포함되지 않는다는 점, Model382와 고정할 전체 CLI·sampler/main RNG 조건, `--val-every 0`, Model382 OOF runner SHA와 inference 옵션, held-out train symlink 의미, class empty-union=0 및 `_step(4)_` Step4 정의를 추가했다. Model382 OOF runner SHA는 `753c1a…01f69`, Model390 receipt는 현재 소스·문서 해시로 갱신했다. 보완 smoke에서 32개 synthetic item의 worker 0/2/4·batch/order 불변성, full augmentation 14개 node의 고유 seed, legacy bitwise equality, epoch replay, H/V 조합, CPU train checkpoint resume-index/mode mismatch guard가 모두 통과했다. 첫 재검증 시 smoke만 epochs를 바꾸어 resume guard가 의도대로 거부했으며, 동일한 학습 인자로 맞춘 smoke를 재실행해 통과했다. 전부 임시 합성 데이터·CPU만 사용했고 mIoU 효과는 미측정이다.

2026-10-02 21:27 UTC 모니터링: P100 runner `972710` 활성, seed2/fold3 epoch `49/70` 완료 후 epoch `50/70` 진행 중이다. training log/process만 확인했고 Model382 OOF·score는 열지 않았다. 기존 locked 분석과 gate가 끝나기 전 Model390 GPU 학습은 시작하지 않는다.

2026-10-02 21:29 UTC Model390 seed 충돌 여지 축소: Claude가 선택 개선으로 지적한 32-bit child-seed 충돌 가능성을 없애기 위해 per-item/tree seed 생성 폭을 64-bit로 늘렸다. 같은 P100 synthetic suite를 다시 실행해 모두 통과했고 candidate source hash를 갱신했다. 실제 학습·점수 변화는 없다.

2026-10-02 21:41 UTC Model390 paired OOF evaluator 고정: candidate 산출물의 lock 이후 생성 시각, checkpoint/args/OOF hash, inference 설정 및 manifest를 검증하고, Model382 helper·primary preregistration receipt·reference runner의 해시도 잠갔다. 합성 fixture가 empty-union=0, seed1/final gates, paired group bootstrap, manifest 검증 및 lock 이전 artifact 거부를 통과했다(로컬/P100 CPU). 갱신한 lock receipt SHA-256은 `52d4b70168cb3c3cbcc7479e2260348337a5637cc12a96f1dfe742503be5362e`다. KoMaP 예측·OOF 점수는 열지 않았다.

2026-10-02 21:41 UTC P100 모니터링: runner `972710`이 Model382 seed2/fold3를 계속 학습 중이며 epoch `57/70` 완료 후 epoch `58/70`, step `50/159`였다. training log/process만 확인했다. 기존 Model382/387/388/389 locked sequence와 gate가 끝나기 전 Model390 GPU 학습은 시작하지 않는다.

2026-10-02 21:51 UTC Claude Code Opus 읽기 전용 브레인스토밍: `model_results.md`와 사전등록/코드만 보고 75.55%가 현재 점수가 아니라 Model66 초기 raw-valid 값임을 지적했다. 최신 raw-valid 숫자는 Model349 79.9095%지만 valid 20장에 625개 route를 선택한 편향된 값이고, Model374 79.2823% 및 Model379 79.6857%는 train grouped-OOF라 서로 구분해야 한다. 기록상 미세 상경계/Eutectic Si가 약하고, 새 후보로 이미지별 물리 pixel-scale 정규화, 학습/in-sample과 held-out 경계 오류 분해, 고정 1:1 HRNet×B4 OOF 평균을 제시했다. loss/SSL/topology/route 계열 다수는 이미 효과가 없거나 음수라 반복을 피하라고 권했다. 검토는 문서·소스만 읽었으며 OOF/valid/test 산출물은 읽지 않았고 변경을 수행하지 않았다.

2026-10-02 21:51 UTC Model391 train-mask-only 물리 배율 가능성 진단: 사전등록 및 source/input hash lock 뒤 공식 `train.zip`의 70 image/mask pair, 59 specimen group만 사용했다. step 고정효과를 넣은 `log(median axial boundary run px) ~ log(native width)`와 20,000회 group bootstrap에서 slope는 Al3Ni `1.217` (95% CI `[0.762,1.634]`), Eutectic Si `0.838` (`[0.605,1.072]`), Primary Si `0.567` (`[0.119,1.015]`)였다. 사전 gate대로 비-Al 3상 중 2상이 지지되어 width-normalization 후속 가설은 통과했다. 이는 픽셀 폭의 연관성일 뿐 physical FOV 확인이나 mIoU 이득이 아니다. specimen별 조성/형태 confounding 및 촬영 metadata 부재를 제한으로 명시했다. 결과 [JSON](experiments/results/model391_physical_scale_mask_audit.json), [사전등록](experiments/model391_physical_scale_mask_audit.md), [코드](experiments/model391_physical_scale_mask_audit.py), [lock](experiments/results/model391_physical_scale_mask_audit_lock.json). GPU·OOF·valid/test는 사용하지 않았다.

2026-10-02 21:52 UTC P100 모니터링: runner `972710` 활성, Model382 seed2/fold3 epoch `63/70` 완료 후 epoch `64/70` 시작 직전이다. 학습 로그/process만 읽었다.

## 모델 383: boundary-label relaxation (실험 전 gate 보완)

학습은 시작하지 않았다. 최초 no-training gate의 비가산적인 band/full 비율과 top-2 기준을 Claude Opus 지적 뒤 폐기하고, fold3–4에서 Model374/376의 boundary-band 대 outside Shapley 기여를 paired group bootstrap으로 보는 proxy 취소검사로 교체했다. Model383 v2 코드와 합성 smoke는 통과했으며, 실 fold artifact 분석이나 자동 학습 GO는 아직 없다. [보완 문서](experiments/model383_no_training_gate_addendum.md), [gate 코드](experiments/model383_no_training_gate_v2.py).

## 모델 384: B4 train-OOF 수축 라우터 스크리닝 (완료, 미통과)

Model322/Model329의 70장 exactly-once train-OOF 확률만 사용했다. 고정 global baseline은 두 모델 equal probability blend이며, router는 predicted-present class 수·Eutectic Si area fraction·normalized entropy 중 단일 image-level feature threshold 하나로 두 전문가 중 하나를 선택하고 lambda `{0.25,0.50,1.0}`만큼 global 평균에서 수축한다. 공식 valid/test 파일은 열지 않았다. 같은 OOF key 및 fold membership을 4-fold mask 목록과 대조하고, specimen group 59개가 fold 간 분리됐는지 검사했다.

- Global blend OOF `79.2884%`; 수축 router OOF screen `79.2784%`, delta `−0.0101%p`, 59-group bootstrap 95% CI `[−0.2160,+0.1621]%p`, 36/70 images 양수.
- Outer fold delta는 `+0.1075/+0.1425/−0.3838/+0.0777%p`. 선택 feature는 3개 fold에서 normalized entropy, 한 fold에서 predicted-present count로 일관성이 약했다.
- 사전 gate(+0.30pp, CI 하한>0, 4 fold 중 3개 이상 양수) 중 fold 수 조건만 통과해 **스크리닝 미통과**. 단, inner router의 OOF 예측 base model은 outer group으로 학습됐을 수 있어 본 결과는 confirmatory estimate가 아니다. Model322/329 B4 쌍에 대한 추가 conditional routing은 우선순위를 낮춘다. 이는 HRNet 라우팅의 이득/편향을 판정한 결과가 아니다.
- 사전등록: [protocol](experiments/model384_b4_oof_router_screen_preregistration.md); [고정 스크립트](experiments/model384_b4_oof_router_screen.py); [pre-result hash lock](experiments/results/model384_b4_oof_router_screen_preregistration_lock.json); [결과](experiments/results/model384_b4_oof_router_screen.json).

## 모델 385: Step4 Eutectic 내부 오분류 feasibility screen (완료, prototype 후보 중단)

학습 없이 Model374 train-only grouped OOF folds 1–2의 36장/30 specimen group만 사용했다. Step4 이미지 16장/16 groups에서 GT Eutectic Si false-negative 중, 어느 GT phase boundary에서든 원본 mask 격자상 8px 이상 떨어진 픽셀의 비율을 사전등록 기준으로 계산했다. 공식 valid/test 데이터는 열지 않았다.

- Eutectic FN `352,640` pixel 중 내부 FN은 `19,404` pixel, 즉 **5.5025%**였다. Fold별 비율은 fold1 `1.4217%`, fold2 `8.6293%`; fold 안에서 specimen group을 재표집한 bootstrap 95% CI는 `[0.3339,13.5751]%` (20,000회)다. 사전 기준 `>=40%`에 크게 못 미쳐 **multi-prototype/texture auxiliary-loss 후보는 중단**한다. 이 두-fold 진단은 일반화 추정치가 아니다. Opus의 사후 검토에 따라, 이 값은 “FN 중 내부 오류의 몫”일 뿐 GT Eutectic 픽셀을 분모로 둔 거리별 FN율이 아니다. 따라서 이것만으로 경계 오분류가 농축됐다고 결론 내리지 않는다.
- 내부 Eutectic FN은 주로 Primary Si(18,486px)로 예측됐고, 경계 인접 FN은 주로 Al3Ni(180,418px), 이어 Al(96,784px) 및 Primary Si(56,034px)로 예측됐다. 이는 오류 위치·혼동 대상의 기술 통계이지 원인이나 개입 효과의 증명은 아니다.
- Opus 권고: 우선 GT Eutectic 거리 구간별 `FN pixels / GT Eutectic pixels`를 계산해 경계 0–4px의 오류율이 내부(≥8px)보다 실제로 높은지 확인하고, Al3Ni–Eutectic signed-boundary displacement가 무작위 잡음인지 한 방향 편향인지 본다. 후보 선별에 이미 사용한 folds1–2가 아닌 untouched folds3–4를 판정에 써야 한다.
- 따라서 다음 우선순위는 새 prototype 학습이 아니라 Model374/376 folds 3–4 train-OOF probability artifact 복구와 기존 Model383 v2 gate/Model381 기하 진단의 train-only 실행이다. 11:15 UTC 재시도 후에도 기존 Colab CPU의 `/content/drive/MyDrive`가 보이지 않아 Drive 복구는 미완료다. gate 통과 전 학습 GO는 하지 않는다.
- 사전등록/코드/결과: [protocol](experiments/model385_step4_eutectic_interior_screen_preregistration.md), [screen](experiments/model385_step4_eutectic_interior_screen.py), [smoke](experiments/model385_step4_eutectic_interior_screen_smoke.py), [결과 JSON](experiments/results/model385_step4_eutectic_interior_screen.json).

### Model386 follow-up: interior 연결 성분·시편 집중 감사 (완료, 사후 기술 통계)

Opus가 지적한 group 집중을 원본 Model374 argmax train-OOF map과 같은 fold GT로 확인했다. 거리 `>=8px` interior 19,642px는 16 Step4 groups 중 9개에만 있고, 47개의 8-connectivity GT component로 나뉜다. 그중 FN율 `>=90%`인 32 components가 전체 interior support의 `98.44%`를 포함한다. 높은 interior FN율이 A182 하나의 큰 연결영역만으로 생긴 것은 아니지만, 전체 interior support의 `87.21%`와 FN의 `88.24%`는 A182 한 group에 몰려 있고 상위 3 groups가 각각 `94.64%`, `95.67%`를 차지한다. A182 내부 오류 17,123px의 대부분은 Primary Si(16,668px), 나머지 대부분은 Al3Ni(455px)다. 따라서 “관측된 여러 interior component가 거의 모두 빠졌다”는 기술 결론은 가능하지만, 시편 독립성이 약하고 7/16 groups에는 interior support 자체가 없어 이를 일반적인 phase/모델 메커니즘으로 확장하거나 새 학습으로 연결하지 않는다. Model385 prototype 중단 및 Model383 원래 gate는 그대로 유지한다.

재현: [사후 프로토콜](experiments/model386_step4_interior_component_audit_preregistration.md), [분석 코드](experiments/model386_step4_interior_component_audit.py), [Colab runner](experiments/model386_component_colab_runner.py), [합성 smoke](experiments/model386_interior_component_smoke.py), [결과 JSON](experiments/results/model386_step4_interior_component_audit.json).

## 모델 386: Step4 Eutectic 거리별 GT-support FN율 파일럿 (완료, 탐색용만)

Model385의 FN-share 해석에 남은 분모 문제를 확인하기 위해 Model374 grouped train-OOF folds1–2에서 GT Eutectic 픽셀을 분모로 한 거리구간별 FN율을 계산했다. 36장/30 groups 중 Step4 16장/16 groups, 공식 valid/test 미사용이다. Fold1–2는 이미 후보 검토에 사용됐으므로 이 분석은 **탐색용**이고 학습 GO/STOP 근거가 아니다.

| GT 경계 거리 (native px) | GT Eutectic support 비율 | 해당 거리대 Eutectic FN율 | 전체 Eutectic FN 중 비율 |
|---|---:|---:|---:|
| `[0,2)` | 79.63% | 35.52% | 80.95% |
| `[2,4)` | 15.42% | 20.70% | 9.13% |
| `[4,8)` | 3.01% | 51.28% | 4.41% |
| `>=8` | 1.95% | 98.79% | 5.50% |

`<4px` pooled FN율은 `33.12%`, `>=8px`는 `98.79%`였고, 차이는 `−65.67%p` (fold-stratified specimen-group bootstrap 95% CI `[-72.38,-47.20]%p`), 비율은 `0.335` (CI `[0.263,0.448]`)다. 두 fold 각각에서도 `near < interior` 방향이 같았다. 따라서 Model385의 “FN 중 94.5%가 경계 근처”라는 점유율만 보고 경계에서 특별히 실패한다고 해석하면 안 된다. 실제로는 GT Eutectic support의 약 1.95%인 내부 픽셀이 거의 전부 누락되어 miss rate가 가장 높고, 절대 FN 수는 지원 픽셀이 압도적으로 많은 `[0,2)` 구간에 가장 많다. 이는 16개 Step4 groups의 탐색적 기하 통계이며 인과·일반화 증거가 아니다. 기존 Model385의 사전등록된 prototype 중단 판정은 사후 변경하지 않는다.

- 실행 위치는 새 Colab CPU session `komap386cpu`이며, transfer bundle의 remote/local SHA-256이 일치하고 argmax map hash를 manifest와 대조했다. GT mask symlink를 follow해 payload에 포함했으며, 20,000회 group bootstrap을 완료한 뒤 JSON을 내려받고 CPU session을 종료했다.
- 재현: [사전등록](experiments/model386_step4_eutectic_distance_rate_pilot_preregistration.md), [분석 코드](experiments/model386_step4_eutectic_distance_rate_pilot.py), [argmax export](experiments/model386_export_argmax_oof.py), [Colab runner](experiments/model386_colab_runner.py), [합성 smoke](experiments/model386_step4_eutectic_distance_rate_pilot_smoke.py), [결과 JSON](experiments/results/model386_step4_eutectic_distance_rate_pilot.json).

Model386 scale-2 round-trip 합성 검사는 네 native shape에서 GT one-hot을 nearest 2× 확대하고 bilinear 0.5× 축소했을 때 argmax 불일치가 모두 0임을 확인했다. 이는 이상적인 class probability에서 입력/출력 리사이즈만으로 경계 라벨이 손실되지 않는다는 뜻이다. 학습된 feature stride, 실제 logits, receptive field의 해상도 한계를 검증한 결과는 아니므로 이를 배제 근거로 확대하지 않는다. [검사 코드](experiments/model386_scale2_roundtrip_check.py), [결과 JSON](experiments/results/model386_scale2_roundtrip_check.json).

## 모델 387: 해상도 병목·경계 오류 oracle 상한 감사 (사전등록 완료, Model382 OOF 대기)

Claude Code Opus는 Model386 folds1–2 신호가 한 시편 A182에 집중되고 내부 GT support가 작아 고해상도/output-stride 재학습 근거로 부족하다고 검토했다. 따라서 Model382 seed1·2의 train-only OOF에서 모든 class의 native-grid 경계거리별 FN/support와 각 거리대 오류를 전부 GT로 고쳤을 때의 공식 image-wise mIoU 상한을 계산한다. folds3–4를 주 진단 subset, 이미 가설 탐색에 쓴 folds1–2를 별도 탐색 subset으로 분리하고, fold-stratified group bootstrap 및 seed별 재현성을 보고한다. oracle 상한은 실제 달성 점수가 아니며, 이 진단만으로 feature stride가 원인이라고 단정하거나 새 학습을 자동 시작하지 않는다. valid/test는 읽지 않는다.

2026-10-02에 분석 정의를 고정했고 Model382 fold3/4 OOF 산출 전이다. P100은 Model382가 사용 중이므로 Model387은 모든 OOF가 끝난 뒤 CPU에서 실행한다. py_compile, CLI help, 합성 fixture smoke는 통과했다. [사전등록](experiments/model387_resolution_headroom_preregistration.md), [분석 코드](experiments/model387_resolution_headroom_audit.py), [합성 smoke](experiments/model387_resolution_headroom_smoke.py). 결과는 생성 전이다.

## 모델 388: component-balanced minority crop 후보 (E1 완료, OOF headroom 대기)

Opus는 sampler 노출과 같은 held-out component의 오류를 직접 연결하자는 최초 제안을 기각했다. Grouped OOF의 해당 시편은 학습에서 제외되어 component 자체의 train-crop 노출은 0이고, 네 fold 간 비교로도 효과를 식별할 수 없다. 이에 E1(training GT 전용 expected-anchor dose)과 E2(held-out OOF component-bin oracle headroom)를 별개로 보는 계획을 OOF 결과 전에 hash-lock했다. 분석은 낮은 우선순위의 CPU 작업으로만 실행했고 P100 학습을 중단하거나 추가 GPU 학습을 시작하지 않았다.

고정 fold training masks 70장/59 groups에서, minority class를 고른 뒤 pixel 좌표를 고르는 현재 sampler와, `area>=64 px` 8-connectivity component를 균등 선택한 다음 해당 성분 내 pixel을 고르는 후보의 기대 positive-anchor 수를 비교했다. `area<64`는 후보의 positive anchor 중심에서 제외(단, 일반 random crops와 다른 crop 문맥에서는 계속 노출될 수 있음)하고 eligible 성분이 없으면 기존 방식을 fallback한다. 양쪽 모두 fold training GT만 썼고 Model382 OOF·valid·test를 읽지 않았다. P100의 CPU에서 실행한 뒤 JSON SHA-256을 local copy와 대조했다.

| outer fold | class | 64–255 px component 수 / groups | 후보/현재 anchor dose |
|---|---|---:|---:|
| 1 (탐색) | Al3Ni | 9,998 / 44 | 5.493× |
| 1 (탐색) | Eutectic Si | 13,263 / 44 | 2.041× |
| 2 (탐색) | Al3Ni | 9,605 / 44 | 4.444× |
| 2 (탐색) | Eutectic Si | 11,492 / 44 | 2.120× |
| 3 (주 분석) | Al3Ni | 10,642 / 45 | 4.189× |
| 3 (주 분석) | Eutectic Si | 11,551 / 45 | 2.105× |
| 4 (주 분석) | Al3Ni | 9,775 / 44 | 4.432× |
| 4 (주 분석) | Eutectic Si | 12,702 / 44 | 2.039× |

각 fold/class의 총 expected positive-anchor 수는 후보와 기존 방식이 정확히 같았다. E1의 dose 조건(목표 64–255 bin에서 두 held-out folds 모두 `>=2×`)은 두 class 모두 통과한다. 다만 후보는 training pool에서 `area<64` 성분 약 32,800–52,400개에 positive anchor를 배정하지 않으며, 기존 방식에선 해당 작은 성분 중심 anchor가 fold3/4 클래스별 positive anchor의 약 10.2–32.4%였다. 이것은 중요한 trade-off이며 성능 향상 증거가 아니다. 따라서 E2에서 그 작은 성분 및 목표 size-bin의 official mIoU oracle headroom, fold/group/seed 일관성, 사전 gate를 모두 확인하기 전에는 sampler 학습을 시작하지 않는다.

재현 및 결과: [사전등록](experiments/model388_component_sampler_audit_preregistration.md), [분석 코드](experiments/model388_component_sampler_audit.py), [합성 smoke](experiments/model388_component_sampler_audit_smoke.py), [pre-OOF hash lock](experiments/results/model388_preregistration_lock.json), [주 folds3–4 E1 JSON](experiments/results/model388_exposure_folds3_4.json), [탐색 folds1–2 E1 JSON](experiments/results/model388_exposure_folds1_2.json). E2 아직 실행 전이며 Model382 OOF 완료를 기다린다.

Opus safety 검토에 따라 원 후보·E2 분석을 변경하지 않고, non-target bins 해석 및 후속 paired 학습 채택 기준을 보충 addendum으로 고정했다. OOF fold3 파일은 생성된 상태였으나 addendum lock 시각 `13:19:27 UTC`까지 이 감사는 OOF score/prediction 값을 열지 않았다. `<64px` oracle headroom은 GO·veto 어느 쪽에도 쓰지 않고 기술통계로만 둔다. 별도 학습을 하게 되더라도 overall OOF `+0.50%p` 이상, paired group-bootstrap CI 하한 `>0`, 4 folds 중 3개 이상 개선, class-IoU와 tiny-component hit-rate 안전 문턱을 모두 요구한다. [해석·safety addendum](experiments/model388_component_sampler_safety_addendum.md), [addendum lock](experiments/results/model388_safety_addendum_lock.json).

## 모델 389: tiny-anchor mass 보존 component sampler (E1 통과, OOF 미열람)

Model388과 구별되는 조건부 후보로, 클래스별 총 positive-anchor 예산과 `<64px` component의 기존 anchor mass를 보존하고 `>=64px` 쪽 mass만 eligible component들에 균등 분배한다. Model389 별도 사전등록·코드·합성 smoke를 hash-lock한 뒤 P100 CPU에서 E1을 계산했다. 이 분석은 고정 training masks만 사용했으며 Model382 OOF score/prediction, 공식 valid/test는 열지 않았다.

| primary outer fold | Eutectic 64–255px components / groups | bin 전체 expected-anchor 비율 | component별 비율 중앙값 |
|---|---:|---:|---:|
| 3 | 11,551 / 45 | 1.3738× | 1.5474× |
| 4 | 12,702 / 44 | 1.3761× | 1.5220× |

사전등록 E1 기준(각 primary fold에서 pooled·component-median 비율 `>=1.25`, component `>=30`, group `>=5`)은 통과했다. 탐색 folds1/2도 pooled 1.3839×/1.3786×, median 1.5417×/1.5305×다. `<64px` expected-anchor mass는 정확히 1.0×이고 class total budget도 보존됐다. 그 대가로 `>=256px` Eutectic 및 큰 Al3Ni component dose가 낮아지는 zero-sum 재배분이며, 실제 crop coverage와 이미지/group별 노출 이동은 이 audit에서 측정하지 않았다. Claude Opus 독립 검토도 E1을 dose 조작 확인으로만 해석하고, anchor와 crop 노출의 차이·큰 성분 감소·이미지/group 재가중·off-target Al3Ni 영향을 별도 위험으로 기록하라고 권고했다. **성능 향상 증거가 아니며 학습 GO가 아니다.** 다음에는 OOF 값을 보기 전에 필요하면 train-mask-only 보완 dose/coverage audit를 별도 사전등록하고, Model388의 사전고정 비교 결과가 tiny-component guard에서 harm/inconclusive일 때만 Model389의 E2/훈련 분기를 고려한다.

재현: [Model389 protocol](experiments/model389_mass_preserving_sampler_preregistration.md), [audit](experiments/model389_mass_preserving_sampler_audit.py), [synthetic smoke](experiments/model389_mass_preserving_sampler_audit_smoke.py), [pre-score lock](experiments/results/model389_preregistration_lock.json), [primary E1](experiments/results/model389_exposure_folds3_4.json), [exploratory E1](experiments/results/model389_exposure_folds1_2.json).

## 현재 후속 gate (2026-10-03)

- P100에는 현재 학습 프로세스가 없다. Model382의 8개 seed×fold 학습/OOF는 완료됐다. Model395 CPU crop-exposure gate와 Model389 E2 component-miss gate는 모두 NO-GO라 해당 sampler들은 학습하지 않는다.
- Model382의 원래 matched-count B/C 비교는 아직 계산 불가다. 현재 P100에는 Model374 OOF fold1–2만 있고 fold3–4, Model376/377 OOF probability archives가 없다. `model374_fold34_payload.zip`에도 해당 확률 아카이브가 없다. valid/test로 대신하지 않는다.
- Model390 per-item RNG는 별도 사전등록/lock된 candidate다. 현재 lock/source 검증과 synthetic paired-evaluator smoke가 통과했지만, 원 프로토콜은 Model382 matched-count/seed 비교와 Model387/388/389 locked audits가 끝난 뒤 실행하도록 되어 있다. 비교 입력이 회수 불가능해 이 선행조건을 바꾸려면 명시적 addendum이 필요하다. Model390도 아직 학습하지 않았다.
- 새 P100 학습은 위 선행조건을 처리하고 Opus read-only 검토를 받은 뒤에만 결정한다. 공식 valid/test 점수로 후보를 다시 선택하지 않고 test는 계속 봉인한다.

## 과거 개선 순서 (Claude Code Opus 검토, 2026-10-02; 아래는 당시 상태)

1. Model382 학습은 현재 P100에서 고정 순서대로 진행 중이며, 새 GPU 작업은 겹쳐 시작하지 않는다. 2026-10-02 16:32 UTC 확인 시 seed1의 4개 fold OOF가 모두 끝났고 seed2/fold1은 epoch 36/70 도중이었다. seed1의 28개 결과·체크포인트 파일과 `protocol.json`을 별도 local backup에 복사하고, 파일 SHA-256 전체 일치 및 네 fold Model382 manifest/checkpoint verifier를 통과했다([backup 안내](</Users/hyunseok/Library/Application Support/KoMaP/model382_backups/seed1_20261002/README.md>)). 보조 seed-average 분석은 lock된 아티팩트를 P100의 고유 파일 경로로 미러링했으며 SHA-256 대조, 합성 smoke, lock verifier를 통과했다. 여덟 OOF가 준비되면 기존 사전등록 matched-count 비교(374+seed1+seed2 대 374+376+377)를 train-only 70장/59 groups로 계산하며, valid/test로 가중치를 고르지 않는다. A71 민감도를 보고하고, 보조 E5는 별도 gate 통과 시에만 고려한다. 이어 Model387 boundary oracle audit 및 Model388 component-bin E2를 CPU-only로 실행한다. Model389는 별도 후보이며 현재 E1 통과만으로 학습하지 않고, 기존 Model388 비교의 tiny-component 안전성 결과와 Model389 E2 gate를 모두 요구한다. 평가기 [코드](experiments/model382_oof_ensemble_compare.py), [합성 smoke](experiments/model382_oof_ensemble_compare_smoke.py); hash-lock receipt: [Model382](experiments/results/model382_e5_preregistration_lock.json), [Model388](experiments/results/model388_preregistration_lock.json), [Model388 safety](experiments/results/model388_safety_addendum_lock.json), [Model389](experiments/results/model389_preregistration_lock.json).
2. Model383 및 추가 거리기하 진단은 Drive에서 Model374/376의 fold3–4 **train-OOF** 확률맵을 복구한 뒤에만 실행한다. 2026-10-02 15:27 UTC의 `/workspace`·`/root` 및 P100 outputs 재검색에서도 Model374 probabilities는 fold1–2뿐이고, Model376 raw OOF와 Model377 OOF가 없었다(376 test probabilities는 별도 존재). fold3에는 Model374 `args.json`만 있고 checkpoint/OOF는 찾지 못했다. local `model374_fold34_payload.zip`에도 OOF probability archive가 없다. aggregate OOF JSON은 일부 남아 있지만 새 거리구간 분석 및 확률 앙상블 재구성에 충분하지 않다. Model386 folds1–2는 `FN/support`가 경계 근처보다 내부에서 더 높음을 보였으나 탐색용이므로 folds3–4를 대체하지 않는다. test 확률로 대체하지 않는다. Drive mount 재시도 후에도 `/content/drive/MyDrive`는 없었고 gate 통과도 자동 학습 GO가 아니다.
3. 라우팅 편향 완화의 첫 CPU 스크린인 Model384는 B4(Model322/329) pair에서 gate를 통과하지 못해 해당 pair의 추가 router 탐색을 보류한다. 이것으로 HRNet 라우팅 편향이 해결되거나 없다고 결론 내리지는 않는다. 현 P100 inventory의 Model374 OOF는 fold1–2뿐이며 fold3 학습 흔적, Model376 test inference뿐이라 최신 HRNet family router의 train-only 평가자료는 불완전하다. Model382가 끝난 뒤 artifact 복구 여부를 확인하고, HRNet OOF가 완전해질 때만 사전고정된 저자유도 분석을 검토한다. 일반 4-fold OOF nested 분석은 outer-heldout group이 다른 fold 모델 학습에 포함되는 누수를 명시해야 한다.
4. 사용자의 최신 지시(2026-10-03 KST)에 따라 Colab A100 대신 GPU 학습은 P100에서 진행한다. 현재 P100의 Model382 한 작업이 끝나기 전에는 GPU 작업을 겹쳐 띄우지 않는다. `colab new --gpu A100`은 quota/entitlement로 거절됐다. Drive mount가 성공하지 않았으며 `/content/drive/MyDrive`와 영구 저장 위치를 검증하지 못했다. Model386 탐색 진단은 임시 Colab CPU sessions에서 실행하고 결과를 다운로드한 뒤 각 세션을 종료했다. 별도로 큰 payload upload의 CLI 500 오류 원문에 해당 세션의 임시 proxy credential이 포함된 것을 확인해 값을 반복 인용하지 않고 해당 유휴 runtime을 즉시 종료했다. 축소·checksum 검증한 payload는 별도 CPU session에서 성공적으로 처리했고, Drive OAuth credential이나 Drive 파일은 업로드되지 않았다. 현재 Colab active session은 없다.

## 개선 방향 재검토 (Claude Code Opus, 2026-10-02 14:12 UTC)

Opus에 추가 routing/threshold 탐색을 피하고, 70 train images/59 specimen groups에서 검증 가능한 후속 실험을 순위화해 달라고 요청했다. 저장된 Model374 OOF source를 다시 확인한 값은 70장 image-wise 평균 `79.2823%`, image-score sample SD `7.9601%p`, class mean IoU `[Al 92.93, Al3Ni 79.62, Primary Si 86.35, Eutectic Si 58.23]%`다. Boundary top-2 oracle의 전체 오답 중 GT 경계 자체에 놓인 비율은 `66.00%`이고, boundary-fraction 회귀 계수는 `−0.3797` IoU per unit fraction(경계밀도 +10%p와 연관된 약 `−3.80%p`, fold/resolution/step 보정; 관찰 연관이지 인과효과 아님)이다. Opus의 추가 test-size 계산은 이 7.96pp 분산을 독립·동일분포 10장에 대입한 거친 표준오차 추정일 뿐이며, group dependence와 distribution match가 검증되지 않아 확정 예측으로 쓰지 않는다.

- 먼저 현재 Model382의 사전등록 matched-count/seed 비교를 끝내고, 그 다음 이미 lock된 Model387/388/389 OOF audits를 실행한다. 기존 결과 없이 sampler·해상도·경계 개입의 GO를 앞당기지 않는다.
- 향후 in-sample 대 OOF generalization-gap 비교는 학습 난도 판단에는 참고될 수 있지만, in-sample 오류가 높다는 이유만으로 annotation noise/불가역 오류라고 결론내릴 수 없다. 재사용할 경우 사후 기술진단으로 명시하고 seed·group·class 차이를 분리한다.
- 학습곡선 측정 제안은 별도 grouped train-only subset 학습이 필요하며 P100 예산이 크므로 Model382 결과 뒤 재검토한다. valid labels를 train에 병합하는 안은 현 workflow(70장 학습, 20장 valid 점수)를 바꾸고 valid를 독립 점검에 쓸 수 없게 하므로 실행하지 않는다. 공개 대회 페이지는 데이터 분할 70/20/10과 평가식을 적지만 valid label의 최종 재학습 사용 허용 여부를 명시하지 않는다. 사용자가 split 정책을 바꾸기 전에는 기존 분할을 유지한다.
- 목표 `80%`는 그대로 둔다. 작은 OOF/valid 표본의 변동이 크다는 이유로 목표를 낮추거나, 한 번의 valid 최고값을 일반화된 성능으로 간주하지 않는다.

## Model374 OOF 시편 coverage 추가 진단 (탐색 종료, 2026-10-02 15:06 UTC)

Claude Opus의 0-GPU 제안을 따라 기존 Model374 OOF와 train.zip의 사전 고정 grayscale intensity/texture 특징으로, held-out 시편과 해당 fold train 시편 간 최근접 거리의 연관을 확인했다. **사후 탐색 분석**이며 fold 학습셋이 겹치고 bootstrap은 고정 모델만 조건화하므로 인과·확증 결과가 아니다.

- appearance distance와 group OOF mIoU Spearman rho는 fold1 `−0.721`, fold2 `−0.525`, fold3 `−0.099`, fold4 `−0.057`; 평균 `−0.351`, 고정 OOF group-bootstrap CI `[-0.568,−0.084]`다. 그러나 fold1·2만 큰 음의 값이고 fold3·4는 최대 appearance distance가 각각 `1.25/1.27`인데도 관계가 거의 없다. fold1의 상위 거리 2개를 제외해도 rho `−0.725`, fold2에서는 `−0.385`로 약해진다. Step+resolution 거리 rho 평균 `−0.041` (CI `[-0.457,+0.412]`), GT class-mix 거리 평균 `−0.189` (CI `[-0.433,+0.088]`)다.
- 상위 거리 quartile 그룹을 fold별 비-far 그룹 median 점수까지 올린 가상 시나리오는 전체 OOF `+1.74%p`지만, 이는 달성 가능 점수/진짜 상한이 아니다. 실제 증거는 이득이 fold1·2에 집중되고 target 분포에 같은 꼬리가 있는지 불확실하다.
- 별도의 **target input만 보는** 분석에서 valid 20장/19 groups 중 train prefix 중복은 7 groups, OOF 거리 percentile 중앙값은 `55.2%`; valid groups 4개(A44,A7,A153,A59)가 75백분위 이상이고 2개(A153,A59)가 90백분위 이상이다. test 10 groups 중 train prefix 중복은 3개, 중앙값 `25.6%`, 최대 `72.4%`로 75백분위 이상은 0개다. 같은 prefix를 거리 계산에서 제외해도 split별 중앙값은 같았다. Opus는 valid의 4/19 및 2/19가 각각 균등 percentile에서 기대되는 수와 거의 같아 coverage 과잉이 아니고, test에는 far-tail이 없다는 점을 들어 **추가 4-fold coverage learning-curve GPU 실험은 보류**하라고 권고했다.
- target 쪽 분석은 공식 train/valid/test ZIP의 원본 이미지 특징만 사용했고 valid/test mask·예측·점수는 읽지 않았다. 15D 저수준 proxy로 domain shift 부재를 증명하지 않으며 mIoU 예측도 아니다. 이 coverage branch는 여기서 종료한다. 이미 진행 중인 Model382는 계속하고, 이후에는 사전 고정된 Model382 matched-count/seed 비교 및 Model387/388/389 CPU OOF audit 순서를 따른다. 이 OOF audits에서 여러 fold에 일관된 추가 증거가 나오지 않는 한 coverage 계열 추가 학습은 열지 않는다.
- 재현: [exploratory protocol](experiments/model374_train_oof_group_coverage_audit.md), [group-distance audit](experiments/model374_train_oof_group_coverage_audit.py), [followup](experiments/model374_train_oof_group_coverage_followup.py), [target-only audit](experiments/model374_target_group_coverage_audit.py), [group coverage JSON](experiments/results/model374_train_oof_group_coverage_audit.json), [tail/class followup JSON](experiments/results/model374_train_oof_group_coverage_followup.json), [target-only JSON](experiments/results/model374_target_group_coverage_audit.json).

## Claude Code Opus 5.5 후속 실험 설계 리뷰 (2026-10-02 15:50 UTC 이후)

- **기준점 정정:** `75.55%`는 초기 Model66의 raw validation 점수이고 현재 모델 점수가 아니다. 최신 기록상 raw valid 최고는 Model349 `79.9095%`지만 valid 20장에 625개 route 조합을 시험해 선택된 최댓값이라 selection-biased다. Model374 `79.2823%`와 Model379 `79.6857%`는 grouped train-OOF 점수이므로 서로 다른 모델·split/프로토콜 사이의 단순 차이로 valid-generalization gap을 주장하지 않는다. 목표 80%는 유지하며, 현 raw-valid 최고만으로 일반화된 80% 달성이라 하지 않는다.
- **기존 Model382 primary는 그대로 유지한다.** 과거 Model374/376/377 OOF 파일이 끝내 없으면 primary matched-count arm은 “필수 입력 누락으로 평가 불가”라고 기록하며 valid/test로 대체하지 않는다. seed1을 백업한 뒤 동일 P100 학습을 계속한다. seed2 fold가 완료될 때마다 체크포인트와 OOF를 같은 방식으로 별도 보존한다.
- **보조 2-seed 민감도 분석은 결과를 보기 전에 새로 고정했다.** seed1/seed2 확률을 고정 1:1 soft-mean하고 각각의 singleton과 따로 비교한다. 두 비교 모두 Δ≥`+0.20pp`, fold-stratified group-bootstrap 95% CI 하한 `>0`, 어떤 class도 `−1.0pp` 초과 하락하지 않음, 최소 3/4 folds에서 평균이 두 seed 모두보다 높음을 요구한다. 실용 채택 고려는 두 비교 모두 `+0.50pp` 이상일 때만이며, pass는 자동 채택이 아니다. 2026-10-02 16:23 UTC lock 시 seed2/fold1은 학습 중이었고 seed2 OOF 파일은 아직 없었다. 기존 seed1 OOF 점수도 보지 않은 상태로 고정했다. [사전등록](experiments/model382_two_seed_mean_sensitivity_preregistration.md), [평가기](experiments/model382_two_seed_mean_sensitivity.py), [합성 smoke](experiments/model382_two_seed_mean_sensitivity_smoke.py), [lock generator](experiments/model382_two_seed_mean_sensitivity_lock.py), [lock receipt](experiments/results/model382_two_seed_mean_preregistration_lock.json).
- **후속 학습은 Model387/388/389 locked audits 결과 뒤에만 결정한다.** 해상도/얇은 경계의 일관된 병목이면 해상도 개입, rare-component 오류가 sampler gate를 통과하면 sampler 실험을 우선한다. 둘 다 아니면 자동으로 Lovasz를 실행하지 않는다: Model333 B4 grouped-OOF에서 Lovasz의 paired 변화는 `+0.0040%p` (95% CI `[-0.0084,+0.0171]%p`)로 근거가 약했다. HRNet에 재시험하려면 OOF 진단이 별도의 metric/loss-specific 가설을 뒷받침해야 하며, 동일 checkpoint·동일 update의 대조군과 별도 사전등록을 요구한다. 새 비라우팅 아이디어는 Opus 사용량 제한 해제 뒤 검토한다. 과거 제안의 예상 `+0.3–1.0pp`는 이 데이터에서 검증된 수치가 아니다.

## Model392 실행 안전성 수정 및 revision-3 lock (2026-10-02 23:22 UTC)

Claude Code Opus의 read-only 점검에서 Model392 amendment의 pooled `3/4 folds improve` 조건이 구현 gate에서 빠진 것, 완료된 seed1을 건너뛰지 않아 seed2 중단 후 runner 재시작이 막힐 수 있는 것, 공식 valid의 단회 claim을 실패/동시 실행에도 유지해야 하는 점을 확인했다. 점수나 valid/test 파일은 열지 않았다. 수정은 모델 recipe와 수치 gate의 범위를 바꾸지 않는다.

- OOF gate에 pooled 3/4 fold 조건을 추가하고, seed2−seed1 delta 및 fold별 target-width 이미지 수를 보고한다. 분석은 고정 output 경로의 영구 단회 claim을 먼저 만든 뒤 입력을 읽고, 최종 JSON은 임시 파일에서 atomic rename한다. 실패 후 재분석은 이 lock에서 허용하지 않는다.
- full-train runner는 epoch 70과 clean completion log가 함께 확인된 seed를 재시작 때 건너뛴다. epoch 70 checkpoint만 있고 clean marker가 없으면 추가 학습 없이 `--resume`으로 종료기록을 마무리하도록 한다. receipt에 불필요한 새 attempt를 추가하지 않는다.
- 공식 valid evaluator는 체크포인트/OOF gate를 확인한 뒤 valid ID/mask를 읽기 전에 `O_EXCL` claim을 만든다. claim은 오류나 동시 실행에서도 유지되며 재시도에는 새 프로토콜 결정이 필요하다. per-image 점수는 평가 중 stdout에 출력하지 않고 완료된 산출물에만 기록한다.
- 기존 revision-2 lock은 보존했다. P100의 당시 v2 source 8개를 `/workspace/komap_model1/.model392_revision2_backup/`에 복구용으로 복사하고 새 코드를 반영했다. [안전성 addendum](experiments/model392_runtime_safety_addendum.md), [revision-3 lock](experiments/results/model392_widthnorm_runtime_lock.json), [lock 생성기](experiments/create_model392_runtime_lock.py). Revision-3 lock SHA-256: `68e7f2d524e338f24d9e88472d1463aadc69d5cb4daf7a56e0d5997edc3ebc98` (42 files).
- 로컬 및 P100 Python 환경에서 OOF/geometry/valid 합성 스모크, 후보 합성 스모크, py_compile 및 shell syntax가 통과했다. 전부 `CUDA_VISIBLE_DEVICES` 비움 상태의 합성/CPU 검사였고 KoMaP 점수나 공식 valid/test labels를 평가하지 않았다. 로컬에는 `albumentations`가 없어 후보 스모크를 P100에서만 실행했다.
- P100에는 Model392 output tree나 Model392 trainer가 없었다. 23:17 UTC 확인 당시 유일한 GPU 작업은 Model382 seed2/fold4였고 epoch34 완료 후 epoch35 step50/159까지 기록되어 있었다. GPU 이용률·메모리는 조회하지 않았다. Model382 및 선행 OOF/audit gates가 끝나기 전 Model392 학습은 시작하지 않는다.
- 23:27 UTC 모니터링: runner PID `972710` 활성. Model382 seed2/fold4는 epoch39 완료 후 epoch40 step50/159 진행 중이며, train log만 확인했다. Model392 output tree는 여전히 없고 GPU 이용률·메모리는 조회하지 않았다.
- 23:38 UTC 모니터링: runner PID `972710` 활성. Model382 seed2/fold4는 epoch45 완료 후 epoch46 step50/159 진행 중이며, train log만 확인했다. 추가 GPU 작업은 시작하지 않았고 GPU 이용률·메모리는 조회하지 않았다.

## Claude Code Opus 후속 브레인스토밍 재감사 (2026-10-02 23:36 UTC)

두 차례 prompt-only 브레인스토밍에서 나온 후보를 현재 P100 코드와 과거 실험기록에 대조했다. 첫 제안은 대부분 이미 시험된 방향과 겹쳐 추가 실험으로 열지 않는다.

- P100에서 현재 실행 중인 `model336_hrnet_best/komap_seg/dataset.py`를 확인했다. Model382는 이미지마다 12 crop씩 동일하게 할당하고, crop의 50%를 Al3Ni/Eutectic Si positive pixel 중심에 두며 두 희소상 선택확률은 같다. 따라서 Opus의 image-uniform + generic rare-Si sampling 제안은 기존 recipe와 중복이다. component-aware sampler는 별도 Model388/389 사전등록 audit gates가 끝난 뒤에만 검토한다.
- stride-2 raw-input detail skip은 Model121에서 이미 실행됐고, 768/1024/3× 고해상도 branch·finetune도 여러 차례 있었다. uniform-area crop만 바꾼 Model275는 Model273보다 최고 epoch 기준 `−0.1077%p`로 독립 개선을 보이지 않았다. HRNet 전용 변형이 자동으로 새 아이디어가 되지는 않는다.
- 기존 [Model374 train-only grouped-OOF oracle 결과](experiments/results/model374_oof_oracle.json)는 class-presence oracle 기준점과 동일한 `79.282329%`이며 변화 `0.0000%p`, false-present/missed class가 있는 이미지 `0/70`이다. 따라서 presence gate를 새 진단으로 반복하지 않는다. GT-majority component oracle `+6.1651%p`는 비실현 oracle이며, 학습된 component/region corrector들은 별도로 이미 미채택됐다.
- 새로운 group-bootstrap 평가기를 만들 필요도 없다. 잠긴 [Model382 평가기](experiments/model382_oof_ensemble_compare.py)는 각 이미지의 official 4-class mIoU 차이를 먼저 계산한 뒤 specimen group을 fold별 복원추출한다. synthetic scorer smoke가 로컬 CPU에서 통과했다. P100 복사본에는 smoke 파일이 없어 이번 실행은 로컬 synthetic-only로 했으며, 실제 Model382 OOF/결과는 열지 않았다.
- Opus가 제안한 1/4·1/2 GT 격자화는 정식 stride 상한이 아니라 인위적 block-prediction sensitivity probe일 뿐이고, ±1px mask 변형도 중복 주석 없이는 annotation noise 추정이 아니다. 8-way D4 TTA 및 nested logit bias도 과거 TTA/625-route·crossfit 탐색과 중복되므로 새 방향으로 채택하지 않는다.
- Claude의 수정 결론은 현재 근거만으로 새로운 GPU 학습 개입을 정당화할 수 없다는 것이다. 이는 목표 축소가 아니라, 진행 중 Model382 및 잠긴 audit/Model392 gate를 완료해 새로운 train-only 증거를 얻은 뒤 다음 처치를 정하는 순서다. GPU 이용률·메모리는 확인하지 않았다.

## Model393 합성 미세조직 형태 가능성 점검 (CPU 전용, 2026-10-02 23:47 UTC)

`train.zip`의 마스크 70장만 대상으로 사전등록·해시 고정된 형태 기술 통계를 계산했다. 원본 이미지 픽셀, valid/test 데이터, Model382/392 산출물, OOF 예측이나 점수는 읽지 않았고 학습도 시작하지 않았다.

추가로 train image PNG 70개의 헤더 메타데이터를 확인했다. 모두 빈 `Phase_info` 항목만 있고 DPI/물리 pixel-size/배율 필드는 없어서, 이 자료만으로 촬영 배율을 복원할 수 없다.

추가 민감도(기존 Model394 group JSON만 이용): raster-bin 층화 bootstrap CI는 Eutectic-Si `[0.036, 0.390]`, Al3Ni `[-0.039, 0.438]`이다. apparent wavelength group 평균은 q10/q50/q90 `45.46/53.33/64px`; 59개 중 10개가 상한 `64px`에 가깝다. crop별 peak 위치는 원 분석에서 저장하지 않아 정확한 포화율은 모른다. bin 내부 상관 방향도 특히 7-group 최고 해상도 bin에서 불안정하다. 명목 permutation p와 percentile bootstrap이 일관된 추론을 주지 않으므로 전처리를 바꾸지 않는다. 이 사후 민감도는 [재현 코드](experiments/model394_scale_sensitivity_diagnostics.py), [JSON](experiments/results/model394_scale_sensitivity_diagnostics.json)에 있다.

Model394의 고정 label-free radial-spectrum 추정기로 train 이미지 70장만 CPU 분석했다. 59개 group, 네 raster bin, phase fraction rank를 통제한 Eutectic-Si width proxy와 apparent wavelength의 partial-rank 상관은 `ρ=0.217`, one-sided Freedman–Lane permutation `p=0.0434`, group-bootstrap 95% CI `[0.021, 0.402]`였다. Al3Ni 보조값은 `ρ=0.202`, `p=0.0755`, CI `[-0.056, 0.442]`로 불확실하다. Eutectic-Si 결과도 Model393 마스크 통계를 본 뒤 고른 탐색 가설이라 확증 결과가 아니다. image-only scale proxy와 실제 OOF false-negative의 연관은 아직 계산하지 않았으며 배율 정규화 학습도 승인·시작하지 않았다. [프로토콜](experiments/model394_image_scale_mask_association_protocol.md), [분석 코드](experiments/model394_image_scale_mask_association.py), [합성 smoke](experiments/model394_image_scale_mask_association_smoke.py), [잠금](experiments/results/model394_image_scale_mask_association_lock.json), [결과](experiments/results/model394_image_scale_mask_association.json).

Opus 통계 재감사와 저장된 group 결과의 사후 민감도 점검: image-scale feature는 59개 group 중 10개에서 상한 `64px`에 가깝게(≥`60px`) 포화됐고, 전체 중앙값도 `53.3px`다. Eutectic-Si partial-rank `ρ=0.217`은 raster-bin 층화 group bootstrap에서 95% CI `[0.036, 0.390]`; leave-one-bin-out 상관은 `0.132–0.256`이다. 단, bin 내부 Eutectic raw 상관은 `0.319, 0.163, −0.086, 0.144`로 이질적이고 최고 해상도 bin은 7 groups뿐이다. Al3Ni 층화 CI는 `[-0.039, 0.438]`이다. 원 permutation p=`0.0434`는 outcome/direction을 이미 본 뒤의 nominal 값이고, 두 CI 계산과 불일치 가능성이 있어 유의 근거로 쓰지 않는다. Opus의 결론은 이 상관만으로 정규화·리사이즈·재학습은 정당화되지 않으며, 허용되는 다음 일은 고정 Model382 train-OOF에서 FN과 mask 폭/scale proxy가 연결되는지 별도로 보는 것뿐이라는 것이다. 그 분석도 같은 그룹을 재사용하는 탐색이며 독립 재현은 아니다.

P100 모니터(2026-10-03 00:08 UTC): runner PID `972710`은 Model382 seed2/fold4 epoch `61/70`까지 완료했고, epoch 61은 `108.9초`였다. Model382 OOF가 완성된 뒤에만 train-only group-level error association을 별도 audit한다. 사용률·메모리는 조회하지 않았다.

P100 모니터(2026-10-03 00:18 UTC): PID `972710` 활성. Model382 seed2/fold4 epoch `66/70` 완료, epoch 67 step `50/159`; 최근 epoch `108.9초`. 남은 학습은 약 7분 추정이며 OOF 추론은 별도다. 새 학습은 시작하지 않았고 사용률·메모리는 조회하지 않았다.

P100 모니터(2026-10-03 00:28 UTC): Model382 seed2/fold4의 학습이 epoch `70/70`로 완료됐다. runner PID `972710`은 살아 있고, `run.log`상 마지막 fold OOF inference 단계다. OOF 점수/확률은 열지 않았으며 GPU 사용률·메모리는 조회하지 않았다.

P100 진행 재확인(2026-10-03 00:30 UTC): runner PID `972710`은 여전히 활성이고 Model382 fold4 OOF 단계다. 완성 manifest는 8개 중 7개로, 마지막 fold 산출은 아직 최종 고정되지 않았다.

평가 로그 노출 주의: 00:29 UTC 상태 확인 때 OOF evaluator log의 마지막 per-image mIoU 줄들이 도구 출력에 우발적으로 포함됐다. 해당 개별 값들은 요약·기록·의사결정에 사용하지 않는다. 이후에는 OOF evaluator/train output 로그 내용 대신 process 및 산출물 metadata만 확인한다. 노출된 OOF 일부를 포함한 후속 오류 연관은 확증 분석으로 주장하지 않는다. 기존 사전등록 gate는 수치/구현을 바꾸지 않고 그대로 실행하되, 결과 공개 시 이 노출을 명시한다.

P100 확인(2026-10-02 23:58 UTC): 추적 중인 Model382 runner PID `972710`은 seed2/fold4 epoch `56/70`까지 진행했고, epoch 56은 `108.0초`였다. 다음 개선 브레인스토밍에서 Opus가 새 후보로 이미지별 겉보기 미세조직 스케일 정규화를 제안했다. 고정 ×2와 달리 test 입력만으로 스케일을 정해 Eutectic-Si의 얇은 구조를 모델 격자에서 더 잘 표현하자는 가설이다. 다만 원본 pixel scale이 없고 실제 미세조직 차이와 배율 차이를 혼동할 수 있다. ±1px mask jitter는 annotation noise의 추정치가 아니라는 점도 기존 원칙대로 유지한다. 추가 학습 전, Model382 OOF가 완성된 뒤 train-only로 (a) mask 폭과 raw raster diagonal의 group-level 관계, (b) label-free 이미지 텍스처 척도가 mask 폭/OOF FN과 연결되는지 먼저 검증한다. 이 탐색 역시 이미 train-mask 형태를 본 뒤 생긴 가설이므로 confirmatory가 아니다. gate가 불분명하면 물리 스케일 metadata 또는 독립 이중 주석을 확보하기 전까지 스케일 개입을 보류한다.

mask 폭과 raster diagonal의 추가 사후 탐색을 코드로 재현했다. 그룹 평균 Spearman rho는 Eutectic-Si `0.574` (59-group bootstrap 95% CI `[0.354, 0.758]`), Al3Ni `0.504` (`[0.256, 0.712]`)이다. 관계는 관찰되지만 acquisition 배율, 실제 형태 차이, 그룹 조성의 혼재를 분리하지 못한다. 그래서 다음 단계는 학습이 아니라 image-only apparent-scale proxy와 폭/완성될 Model382 OOF FN의 관계를 train-only에서 점검하는 것이다. [재현 코드](experiments/model393_scale_width_resolution_followup.py), [JSON](experiments/results/model393_scale_width_resolution_followup.json).

Opus의 후속 프로토콜 재감사에서는 held-out GT mask를 입력으로 쓴 합성 이미지 평가를 파라미터 누수와 구분했다. mask→image renderer 정의상 held-out mask를 conditioning하는 것 자체는 모델 파라미터 누수가 아니지만, 해당 결과로 renderer·checkpoint·합성 비율·필터를 고르면 그 그룹이 사실상 validation이 된다. 더 근본적으로, 합성 이미지의 segmenter IoU는 renderer의 mask 정합성만 볼 뿐 실제 이미지 성능이나 augmentation 이득을 입증하지 않으며, mask 조건을 쉽게 드러내는 합성물이 오히려 높은 점수를 얻을 수도 있다. Stage A는 별도 보고하는 plausibility screen에 그쳐야 한다. Stage B에서 합성 샘플은 outer-train mask/prior만으로 만들고, 설정 선택은 nested inner-group split에서만 해야 하며 outer test는 잠긴 설정으로 한 번만 평가한다. nested 분할이 불가능하면 Stage A/tuning 그룹과 Stage B 평가 그룹을 완전히 분리한다. 따라서 synthetic 학습은 현재 보류한다.

해상도별 보조 요약(사후 탐색, 잠긴 JSON에서 계산): `675x480`은 17장/16 groups, Eutectic-Si q50 폭 proxy 중앙값 `2.0px`, Al3Ni `2.8px`; `1230x875`은 20장/15 groups, 각각 `2.0/5.6px`; `1278x909`은 24장/21 groups, `2.8/4.39px`; `1440x1024`는 9장/7 groups, `2.8/8.79px`다. 표본 수가 작고 상 조성도 resolution bin마다 달라 물리 scale 효과와 시편 조성 효과를 분리할 수 없다. 이는 resolution 혼란 가능성을 뒷받침하지만, 물리적 폭이나 인과 효과를 추정하지 않으며 학습 GO도 아니다.

- All four phases pass the predeclared support screen: represented in 58–59 of 59 specimen groups. Image-median area fractions are Al `58.4%`, Al3Ni `25.6%`, Primary Si `9.6%`, and Eutectic Si `7.1%`.
- Median connected-component counts/image are Al `150`, Al3Ni `1,138`, Primary Si `28`, and Eutectic Si `893`; median component areas are respectively `52`, `21`, `912`, and `35` pixels. Eutectic-Si's per-image median of the pixel-width q50 proxy has q10/q50/q90 `2/2/4 px`; this is a raster-label descriptor, not a physical length or annotation-noise estimate.
- Raw image dimensions vary substantially: `675x480` (17 images), `1230x875` (20), `1278x909` (24), and `1440x1024` (9). Without physical pixel scale, pixel-size comparisons can be confounded by resolution/acquisition scale.
- Support gate passed, but the protocol has no automatic training-GO rule. Any later synthetic pilot needs its own fold-local source lock and grouped-OOF gate; this audit's aggregate findings make a design adapted to them exploratory/selection-conditional. [Protocol](experiments/model393_synthetic_mask_feasibility_protocol.md), [audit code](experiments/model393_synthetic_mask_feasibility_audit.py), [lock](experiments/results/model393_synthetic_mask_feasibility_lock.json), [results](experiments/results/model393_synthetic_mask_feasibility.json). Lock SHA-256: `e7477968cefbb725019e18885a599937a6b095ce8e3766923bf1a98e0fbca0e6`.
- P100 monitor at 23:48 UTC: runner PID `972710` active on Model382 seed2/fold4, epoch `51/70` (step `100/159`); epoch 50 took `109.1s`. Model392 is not training. No other GPU job was started; GPU utilization/memory was not queried.
- Claude Code Opus read-only review: **conditional GO to design a falsifiable pilot, not a GO to train**. Main risks are 1–2 px Eutectic-Si structures being corrupted by a renderer/downsampling, unknown physical scale across four raster sizes, mask/image memorization with roughly 50 train images per fold, and generated masks breaking observed phase adjacency. Opus proposed a fold-local synthetic-image consistency gate first; only if it passes, compare real+strong augmentation, the same recipe plus synthetic samples, and a compute/step-matched real-repeat control under grouped OOF. That proposal still needs a leakage/selection audit and a separately frozen protocol; no model or score is authorized by this screen. No gain estimate is made.

## Model382 완료 및 후속 CPU 진단 (2026-10-03)

- P100 Model382 seed1/seed2의 4-fold train-OOF 8개가 모두 완료됐다. 현재 P100에는 이 프로젝트의 학습 프로세스가 없다. 이어진 분석은 전부 `CUDA_VISIBLE_DEVICES=""` CPU-only이며 valid/test 파일이나 라벨을 열지 않았다.
- 고정된 Model382 two-seed 보조 분석은 첫 실행에서 기존 evaluator의 provenance 경로 버그로 종료됐다. `seed` 문자열이 이미 `seed1`/`seed2`인데 `seed` 접두어를 다시 붙여 `seedseed1`/`seedseed2` checkpoint 경로를 만들었다. 잠금 evaluator·metric·threshold·bootstrap·gate는 수정하지 않고, 별도 recovery wrapper가 이 8개 경로만 바로잡아 해시를 확인했다. 복구 결과는 [JSON](experiments/results/model382_two_seed_mean_sensitivity_v2_recovery.json); wrapper [코드](experiments/model382_two_seed_mean_sensitivity_v2_recovery.py), SHA-256 `c3f10395c390f64e663d8741a49b0a6dbeaa0691002b0775b7fe39d1e8a4d183`.
- 공식 image-wise 4-class mIoU(empty union=0), 70 train-OOF images/59 groups에서 seed1·seed2 확률의 1:1 평균은 `79.1622%`. seed1 단독 대비 `+0.2433%p` (group-bootstrap 95% CI `[+0.0800,+0.4167]`), seed2 단독 대비 `+0.2022%p` (`[+0.0310,+0.4074]`)다. 점추정과 CI 조건은 충족했지만 4개 fold 중 두 seed 모두를 이긴 fold는 `2/4`뿐이라 사전등록 secondary/adoption gate는 **실패**했다. 더구나 진행 로그에 일부 per-image OOF 값이 노출됐으므로 이 전체 분석은 탐색적이며 확증 근거가 아니다.
- Model387 boundary/oracle audit의 [folds3–4 결과](experiments/results/model387_resolution_headroom_folds3_4.json)와 [folds1–2 탐색 결과](experiments/results/model387_resolution_headroom_folds1_2.json)는 둘 다 2 seeds × 34/36 images, 각각 29/30 groups를 포함한다. folds1–2는 Model385/386 가설 생성에 사용된 탐색 subset이다. folds3–4 OOF mIoU는 seed1 `80.842%`, seed2 `80.831%`; folds1–2는 `77.103%`, `77.193%`다. 이 숫자는 해당 train-OOF subset의 기술통계이지 valid/test 점수가 아니다.
- Folds3–4에서 Al3Ni의 `<4px` FN율은 두 seed `15.1–15.4%`, `>=8px`는 `0.34–0.36%`; Primary Si는 각각 `19.3–19.7%` 대 `1.37–2.65%`로 경계 오류 집중이 반복됐다. Eutectic Si는 반대로 `<4px` `24.1–25.1%`, `>=8px` `88.9–92.5%`였지만 deep-interior support가 작으므로 별도 분포/그룹 집중 확인이 필요하다. 0–2px 띠의 perfect-correction oracle mIoU 상한은 `+15.675–15.729%p`로 크지만 실제 가능한 개선량의 예측이 아니다. 코드 [Model387](experiments/model387_resolution_headroom_audit.py), preregistration [문서](experiments/model387_resolution_headroom_preregistration.md); fold4 per-image OOF 노출 때문에 이 audit 역시 확증 분석으로 주장하지 않는다.
- Model387 검증 중 P100에 없던 Model382 lock-bound smoke 파일은 로컬 잠금 해시와 일치하는 사본으로 복구했다. 해당 보조 smoke는 별도 visualization 모듈이 P100에 없어 import 단계에서 실패했으나, Model387 자체 synthetic smoke 및 Model382 audit의 해시 검증은 통과했다. 평가 코드는 OOF 계산 전에 모든 lock/hash 및 manifest 검사를 통과했다.
- Model388 E2 folds3–4는 `2026-10-03 00:48 UTC` 현재 P100 CPU에서 실행 중이다. Model388 preregistration/safety-addendum lock과 코드 hash 검증, synthetic smoke가 통과했다. 정해진 area-bin headroom/support/dose gate만 적용하고 결과로 새 학습을 자동 시작하지 않는다. Model389 E2는 Model388의 사전 조건 확인 뒤에만 진행 여부를 결정한다.

### Model388 E2 완료 (2026-10-03 00:53 UTC)

- 첫 실행은 OOF 점수 계산 후 집계 코드의 nested-key 버그(`cell_rows` 원소에 다시 `row["cells"][cell]` 접근)로 중단되어 결과 파일을 만들지 못했다. 잠금된 평가 코드·처치·metric·20,000회 group bootstrap은 수정하지 않고, [별도 recovery wrapper](experiments/model388_component_sampler_audit_recovery.py)가 각 cell 요약에 기존 집계기가 기대하는 nested view만 추가해 재실행했다. 합성 smoke와 두 Model388 lock 검증을 통과했다. 최초 실패와 기존 per-image OOF 로그 노출 때문에 결과는 탐색적으로만 해석한다.
- [E2 결과 JSON](experiments/results/model388_headroom_folds3_4_recovery.json), 34 train-OOF images/29 groups per seed; official image-wise 4-class mIoU, empty union=0. **Al3Ni 64–<256px**: seed1 oracle gain `+0.942%p`, 99.5% CI `[+0.588,+1.358]`; seed2 `+0.965%p`, `[+0.599,+1.383]`. Folds3/4 seed1 gain `+0.799/+1.085%p`, seed2 `+0.840/+1.089%p`; maximum specimen-group share `13.2%/12.4%`. Per-fold support는 `2,698/3,565` components, `14/15` groups다. Training-GT-only E1 dose ratio는 `4.189/4.432×`, class-total expected anchor delta는 fold별 `0`.
- **Eutectic Si 64–<256px**: seed1 oracle gain `+2.087%p`, 99.5% CI `[+1.628,+2.519]`; seed2 `+1.995%p`, `[+1.563,+2.417]`. Folds3/4 seed1 `+2.063/+2.112%p`, seed2 `+1.857/+2.134%p`; maximum group share `8.3%/8.4%`. Per-fold support는 `4,785/3,634` components, `14/15` groups다. E1 dose ratio `2.105/2.039×`, class-total expected anchor delta는 `0`.
- 따라서 두 `64–<256px` cells는 사전 고정 E2 headroom·fold 방향·seed 재현·99.5% CI·support·largest-group-share 및 E1 dose 기준을 수치상 통과했다. 이는 해당 GT component를 완벽히 고쳤을 때의 **oracle 상한**이지 sampler 학습이 점수를 올린다는 증거가 아니다. Candidate는 eligible image/class에서 `<64px` positive anchor dose를 0으로 재분배하므로 tiny-component 성능 위험을 실제 paired OOF에서 감시해야 한다. 별도 training preregistration/명시적 순차 결정 전에는 P100 학습을 시작하지 않는다. Model389 E2는 Model388 training comparison에서 tiny-component 안전성 문제가 관측될 때만 조건부로 검토한다.
- 현재 P100에는 학습이나 audit 프로세스가 없다. Recovery SHA-256 `1d4c0ec6369cf1e7197404cb17245135ba7f98a1aa06312e3c3460d3e39e194f`; 결과 SHA-256 `02918ca8a05a2d1e706277f08eb0710b06aa0050b2b71695cd24a49943be00f7`. Original locked audit SHA-256 `ca6f13bf5ed8813efc881173d14facf21db4eb6b104ca5cc37358cd4cd14cb4e`; preregistration lock `bf9b900a7895a5770cdeaccf8334741c6f525ea27a4be14f940e573e59819df9`; safety addendum lock `f795bf33b144a1c71863a6df4b35ddc6497a78df66f738791fea43c8306bd491`.

## Model374 Drive recovery 및 Model392 실행 순서 확정 (2026-10-03)

- Model374 fold3–4의 문서상 Drive 위치 `MyDrive/KoMaP/model374_fold34_retry_20261001`를 `gdrive3`에서 **파일명·크기·수정 시각·Drive MD5만** 확인했다. 두 fold 모두 epoch-70 checkpoint pair와 completion 파일은 있지만, `oof/` 바로 아래에는 `valid_probabilities.npz`, `valid_per_image.csv`, `valid_summary.json` 등 `valid_*` 산출물만 있었다. 아카이브/manifest 내용은 읽지 않았다. train-fold OOF 확률 파일을 찾지 못했으며, 기존 Colab session `komap374b`는 현재 CLI에서 찾을 수 없다. [metadata inventory](experiments/results/model374_drive_metadata_inventory.json).
- 따라서 Model382 matched-count B/C 대비는 **필요한 Model374 fold3–4 및 Model376/377 train-OOF 확률 아카이브가 없어 추정 불가**로 종료한다. valid/test나 재학습한 옛 run으로 대체하지 않는다. 기존 Model382 same-seed/fold 8개 reference는 P100에 남아 있으며, Model392 비교에는 이 고정 reference를 사용한다.
- Model382 two-seed 확률 평균은 train-OOF `79.1622%`였으나 두 seed 모두를 이긴 fold가 `2/4`라 사전 gate를 통과하지 못했다. 이미 일부 aggregate/per-image OOF 값이 노출된 개발 결과이므로 다음 분석도 독립 확증으로 표현하지 않는다. Model387 audit 완료, Model395 crop-dose 및 Model389 rare-component audit는 각각 NO-GO이며, 실행 대기 중인 sampler GPU branch는 없다.
- Claude Code Opus의 현 증거 재검토는 Model390 RNG stream 교정보다 **Model392 이미지별 폭 정규화**를 다음 한 가지 실험으로 권고했다. 실제 RNG 다양성과 mIoU 개선은 측정되지 않았고, 폭 정규화 효과도 아직 가설이다. 675px 이미지의 native crop/tile이 약 135–140px로 바뀌는 큰 개입이며 물리적 배율 효과와 시편 형태 차이를 분리할 수 없다. 예상 향상치를 주장하지 않는다. Model390은 별도 후보로 보류하고 두 처치를 섞지 않는다.
- 누락된 비교를 “완료”로 가장하지 않고 unestimable로 닫는 sequence addendum을 작성했다. 기존 Model392 revision-3의 42-file runtime lock (`68e7f2d5…c3ebc98`), 모델 구조·recipe·추정량·bootstrap·수치 gate는 그대로 보존했다. [Model392 sequence-closure addendum](experiments/model392_sequence_closure_addendum.md)와 P100에서 생성·재검증한 [closure lock](experiments/results/model392_sequence_closure_lock.json)은 83개 파일 및 Model382의 8개 seed×fold checkpoint/args/manifest/OOF reference를 고정한다. Closure-lock SHA-256: `3df753de47ab636390ef806237018cd3c302636c0b59cc1dd2ea5236532a551e`.
- P100에서 runtime lock 검증, 8개 fold의 train/held-out-train-OOF 링크, 새 runner 문법, closure lock 재검증이 통과했다. Runner는 한 번에 한 fold만 학습하고, 매 OOF 추론 직전 링크를 다시 검증한다. `val_every=0`; fold별 OOF inference 로그는 여덟 manifest가 모두 완료되기 전에 열지 않는다. gate 통과 시에만 full-train seed1/2와 official valid 20장 단회 평가로 이어지며, gate 실패 시 그 단계와 test는 실행하지 않는다.
- 이 addendum/closure 결정 시점에는 Model392 출력 디렉터리와 학습 프로세스가 없었다. Lock 생성기가 Model382 baseline 파일의 SHA-256을 검증했으며 Model392 후보 점수는 아직 없다. 다음 paired OOF 결과는 baseline OOF 및 일부 train-only audit 노출에 조건화된 개발 결과로 보고한다.

## Model392 완료 결과 및 현재 후속 상태 (2026-10-04)

- P100에서 사전등록된 Model392 HRNet-W48/U-Net++ 폭 정규화 실험의 2 seeds × 4 specimen-group folds 학습과 exactly-once train-OOF가 완료됐다. `model392_widthnorm_two_seed_confirmatory.json`의 claim은 `2026-10-04 02:38:34 UTC`, 결과는 `02:46:24 UTC`에 생성됐다. 이 결과는 70 train images/59 groups에 대한 development-only paired OOF이며 공식 valid/test를 읽지 않았다.
- 같은 seed/fold Model382 대비 seed1 delta는 `+0.0586%p` (group-bootstrap 95% CI `[-0.5578,+0.6381]`), seed2는 `+0.1499%p` (`[-0.4123,+0.6655]`)다. 두 seed equal-weight paired delta는 `+0.1042%p`, CI `[-0.4175,+0.6007]`로 사전등록 기준 `+0.50%p`와 CI 하한 `>0`을 모두 충족하지 못했다. `valid_evaluation_authorized=false`이므로 full-train 및 official-valid 단계는 실행하지 않았다.
- Pooled fold deltas는 fold1 `+1.0041`, fold2 `-0.1482`, fold3 `-0.9984`, fold4 `+0.5214%p`로 2/4 folds만 양수다. Equal-weight class deltas는 Al `-0.0911`, Al3Ni `+0.2584`, Primary Si `+0.3170`, Eutectic Si `-0.0674%p`; Step4 delta는 `+0.8268%p`다. 따라서 평균 이득은 작고 fold-3 하락이 커서 width normalization을 채택하지 않는다. 이는 물리적 scale 효과를 입증/반증하는 독립 추정이 아니라, 이미 노출된 train-OOF 기반 개발 결과다.
- Model397 fixed B4/HRNet blend의 사전 고정 train-OOF 결과는 `79.9859%`로 B4 `79.1233%`, HRNet seed-mean `79.2085%`보다 높았으나 Model392 primary gate 실패로 `valid_stage_allowed=false`; official validation 점수로 취급하지 않는다. 별도 error-type 감사에서는 Eutectic Si Step4의 FN과 FP가 함께 커서 단순한 전역 과소예측으로 설명되지 않았다.
- 최신 official-valid 관측 최고는 계속 Model364 `79.9430%`이며 80%보다 `0.0570%p` 낮다. P100 재확인에서 KoMaP 학습/평가 프로세스는 실행 중이지 않았다. 다음 후보 Model399는 fold-local weighted-CE/prior-correction 아이디어 초안이며 아직 Claude 검토·사전등록을 통과하지 않아 학습하지 않았다. P100 전용 Python 환경에서 crop-prior audit synthetic smoke를 CUDA 비활성 상태로 재실행해 통과했다. Claude Code 주간 한도 초기화 전에는 검토 응답을 받지 못했다.

### Model399 crop-prior precision audit 완료 (2026-10-04)

- 잠금된 CPU-only 진단은 4개 specimen-group fold × 2개 coordinate-cap seed에서 native crop 70 epoch Monte Carlo와 실제 resize/Albumentations 10-epoch probe를 수행했다. P100 PID `1483836`은 결과 JSON을 쓰고 정상 종료(exit 0)했다. ten-epoch 결과의 모든 training-mask SHA-256이 one-epoch 기준 JSON과 정확히 일치했다. 결과는 [JSON](experiments/results/model399_crop_prior_audit_aug10.json), SHA-256 `d553338b83a53f6364233331212ecde090f13366e605370d21e786e7d2cfd19d`; 사전등록 SHA-256 `66ee18c97af02953d75cdac19bd42fd67f70d1ddde6c71fef65a894563ad4b02`이다.
- class order `[Al, Al3Ni, Primary Si, Eutectic Si]`에서 10-epoch rare-centered `q` 평균은 `[58.0593,25.4020,9.2792,7.2596]%`, uniform-crop 평균은 `[57.9633,25.0441,9.9642,7.0284]%`였다. rare-centered minus uniform은 `[+0.096,+0.358,−0.685,+0.231]%p`; Al3Ni·Primary Si·Eutectic Si 방향은 8개 fold/seed cell 모두 같았다. Rare-centered `q/p` pooled-source 평균은 `[1.0135,0.9655,0.9836,1.0414]`로 source prior와 차이가 작다.
- Ten-epoch fold-local normalized `1/q` weights의 class 평균은 `[0.2280,0.5218,1.4271,1.8231]`, cell ranges는 Al `[0.2234,0.2312]`, Al3Ni `[0.4881,0.5506]`, Primary Si `[1.3841,1.4660]`, Eutectic Si `[1.8078,1.8346]`다. One-epoch 대비 mean q 차이는 `[-0.1902,+0.2686,−0.0577,−0.0206]%p`; normalized weight의 최대 cellwise 절대 차이는 `[0.0063,0.0180,0.0461,0.0490]`였다. Ten-epoch normalized `1/q`와 `1/p`의 최대 cellwise 절대 차이는 `[0.0035,0.0340,0.0728,0.0965]`다. 따라서 sampler-aware prior 보정이 one-epoch 추정 잡음에 좌우될 우려는 줄었고, 실제 crop/source prior 차이는 작다는 쪽의 증거가 강화됐다. 하지만 이는 class weighting/보정의 mIoU 효과를 측정하지 않았으며 어떤 학습도 승인하지 않는다.
- Image-wise metric에 맞춰 source `p_imgmean`도 비교했다. Rare-centered `q/p_imgmean` 평균은 `[1.0026,1.0102,0.9585,0.9995]`, uniform crop은 `[1.0009,0.9959,1.0292,0.9677]`였다. Normalized `1/q`와 `1/p_imgmean` 간 최대 cellwise 절대차는 `[0.0062,0.0196,0.0690,0.0443]`로 pooled `p` 대비 Al3Ni/Eutectic Si에서 더 가까웠다. `p_imgmean` 역시 image-wise IoU를 최적화하는 정확한 target prior는 아니므로 설명적 비교로만 취급한다.
- Recall Loss의 primary paper와 pinned [공식 구현](https://github.com/PotatoTian/recall-semseg/tree/a3f7fc3d12468ac41e78740122d8820c087c7aa6)을 읽었다. 논문 평가는 Synthia/Cityscapes의 DeepLab 계열이다. pinned [`runningScore`](https://github.com/PotatoTian/recall-semseg/blob/a3f7fc3d12468ac41e78740122d8820c087c7aa6/ptsemseg/metrics.py#L18-L36)는 이미지 전체 confusion matrix를 누적하고 class IoU에 `nanmean`을 적용하므로 pooled/global mIoU이며 empty class를 제외한다. 이는 KoMaP의 이미지별 4-class empty=0 지표와 다르다. RecallCE 구현은 매 mini-batch hard-argmax false-negative rate로 pixelwise CE를 가중하므로 Model392의 4-patch batch mix에 민감할 수 있고, device/empty-or-single-error-class 처리도 그대로 이식할 수 없다. 더 중요한 점은 reference trainer의 static WeightedCE/BalancedCE class-count branch가 `valloader`를 순회하고, Synthia YAML은 `train`/`val`을 별도 split으로 지정한다는 것이다. 이는 static CE ablation에서의 label leakage caveat이며 YAML의 main `RecallCE` run까지 같은 문제라고 단정하지 않는다. Static ablation 계산은 재현하지 않고, 모든 KoMaP 가중치는 outer-train mask에서만 계산해야 한다. [paper](https://arxiv.org/abs/2106.14917), [loss source](https://github.com/PotatoTian/recall-semseg/blob/a3f7fc3d12468ac41e78740122d8820c087c7aa6/ptsemseg/loss/recall_loss.py#L10-L39), [trainer source](https://github.com/PotatoTian/recall-semseg/blob/a3f7fc3d12468ac41e78740122d8820c087c7aa6/train.py#L828-L865), [Synthia config](https://github.com/PotatoTian/recall-semseg/blob/a3f7fc3d12468ac41e78740122d8820c087c7aa6/configs/deeplab_synthia.yml#L5-L20). 어떤 Recall Loss/weighted-CE 변형도 아직 Claude가 검토하거나 학습 승인하지 않았다.
- 이 진단은 train images/masks만 사용했고 OOF 예측·mIoU·official valid/test·GPU를 전혀 읽거나 사용하지 않았다. Model399는 여전히 Claude Opus의 단일 방향/no-go 검토와 잠긴 paired grouped-OOF protocol 전까지 학습하지 않는다.

## Model400: Model396-D2 단발 valid 평가 (2026-10-04/05)

- Claude 사용량이 소진된 상태라 별도 호출 없이 독립 검토했다. Model396의 train-only grouped OOF D2 gate는 통과했지만, OOF에서는 이미지마다 held-out fold 1개를 썼고 valid route는 두 모델군 모두 4-fold 확률 평균을 쓴다는 estimand 차이를 확인했다. 재사용된 valid 20장은 개발용 참고값으로만 보고, 고정 50:50 route를 한 번 평가하는 것으로 승인했다. [사전등록/독립 검토](experiments/model400_model396_d2_valid_ensemble_preregistration.md).
- 고정 route는 기존 Model329 EfficientNet-B4 4-fold 확률 아카이브와 Model382 HRNet-W48 seed2 4-fold 확률 평균을 0.5:0.5로 섞고, 마지막 argmax를 적용한다. HRNet은 patch 512, input scale 2, overlap 170, flip4 TTA, Gaussian tile stitching으로 P100 GPU 7에서 예측했다. 평가는 공식 image-wise 4-class mIoU, empty union=0으로 valid 20장에 수행했고 test 입력은 사용하지 않았다.
- 결과는 **79.3227%**, 기록된 Model364 `79.9430%`보다 `−0.6202%p`다. 목표 `80%`에 `0.6773%p` 부족하므로 현재 최고를 갱신하지 못했고 후보로 채택하지 않는다. class image-mean IoU는 Al `94.4481%`, Al3Ni `74.8952%`, Primary Si `86.7512%`, Eutectic Si `61.1964%`다. 최저 개별 이미지 mIoU는 A71 `39.700%`다.
- 투명성/복구 이력: 첫 inference는 첫 맵 저장 직후 progress JSON의 `flush` 인자 위치 버그로 멈췄다. 해당 맵은 해시·정규화 확인 후 보존하고 별도 recovery lock 아래 나머지 19개 맵만 생성했다. 첫 scorer 실행도 같은 출력 버그로 A117 라벨 한 장의 IoU를 메모리에서 계산한 직후 멈췄다(숫자 미출력, aggregate/result 미생성). 기존 claim을 보존하고 같은 후보·metric만 완결하는 별도 score-recovery lock 및 JSON logging shim으로 복구했다. 새 후보나 파라미터 탐색은 없었고 최종 result는 한 파일만 생성됐다.
- valid를 반복 사용한 결과이고 OOF와 최종 4-fold 추론 대상도 같지 않으므로 selection-conditional 개발 점수다. 이 값은 독립 test/generalization 성능이 아니다. 후속 개선 선택은 train-only grouped OOF로 돌아간다.
- 산출물: [결과 JSON](experiments/results/model400_model396_d2_valid_once.json), [base lock](experiments/results/model400_model396_d2_valid_lock.json), [inference recovery lock](experiments/results/model400_inference_recovery_lock.json), [score recovery lock](experiments/results/model400_score_recovery_lock.json), [recovery receipt](experiments/results/model400_score_recovery_receipt.json). Result SHA-256 `697e6d0b83be50ba59edce48a22c34184a8abe37f77515724688a3190ae8f2d7`; base lock `11b98f83e3b9fa653658bc69a1dfe9a6d5ff3feed0d0aaffb2b347e1afcfbda6`; inference-recovery lock `ef8fec2b3fd7ef581ad3203a71a05ae118c5e4819f39036b460a5cb44b17120e`; score-recovery lock `d4d9f5dba11db53366965ee95cd44139dfadad56af392184a8652a5f6938c1bd`.

## Model401: Model397 고정 blend 단발 valid 평가 (2026-10-05)

- 고정 확률 조합은 `0.50 * Model329 B4 4-fold + 0.25 * Model392 HRNet seed1 4-fold + 0.25 * Model392 HRNet seed2 4-fold`이며, 각 HRNet checkpoint의 저장된 preprocessing, patch 512, scale 2, overlap 170, flip4 TTA 및 Gaussian tile stitching을 사용했다. 마지막에 probability argmax를 적용했다.
- 공식 image-wise 4-class mIoU(empty union=0)를 valid 20장 모두에서 계산했다. 결과 `79.4181708850679%`; 저장된 per-image 벡터의 평균 재계산 `79.41817088506788%`와 일치한다. Class image-means(Al/Al3Ni/Primary Si/Eutectic Si)는 `94.3031748/75.0060975/86.9066895/61.4567217%`. Model364 대비 `−0.5248018%p`, 80% 목표보다 `0.5818291%p` 낮다. 최고 official-valid 관측은 Model364 `79.9429727%` 그대로다.
- Model397 grouped train-OOF `79.9858620%`는 별도 OOF 추정량이며 valid 결과가 아니다. 이 추가 valid 평가는 이미 반복 재사용된 20장에 대한 selection-conditional 기술값으로, 독립 test/generalization 성능으로 주장하지 않는다. `test_used=false`이며 test 입력/라벨은 열지 않았다.
- 복구·실행 이력: 첫 inference는 첫 이미지 두 seed 맵을 저장한 뒤 logger의 `json.dumps(..., flush=True)` 인자 버그로 중단됐다. 첫 맵은 보존했고, 별도 recovery lock 아래 남은 19장만 생성했다. 총 20장 × 2 seed 평균 맵 40개, manifest와 recovery receipt를 확인한 후 공식 scorer를 단 한 번 실행했다. 추론은 P100에서 약 66분 걸렸다(8 checkpoint × flip4 TTA × 타일 추론); 시작 전 비용 산정과 logger 확인이 부족했던 운영상 지연이다. 로컬로 가져온 결과·lock·manifest·claim·receipt의 SHA-256은 P100 원본과 일치한다.
- 후속 개선 방향은 Al3Ni/Primary Si 경계에서 같은 상 내부 방향을 예측하는 보조 head가 기존의 강한 boundary-localization 및 target-support 근거에 비춰 검토 후보로 남아 있다. 다만 mIoU 이득은 입증되지 않았고 Eutectic Si의 deep-interior 오류를 해결하지 못한다. Claude Code 자문은 한 번 요청했으나 weekly limit에 막혀 응답을 받지 못했다(초기화 안내: 2026-10-06 23:00 KST); 우회 호출은 하지 않았으며 해당 자문 없이 새 학습을 승인하지 않았다.
- 산출물: [결과 JSON](experiments/results/model401_model397_blend_valid_once.json), [base lock](experiments/results/model401_model397_blend_valid_lock.json), [recovery lock](experiments/results/model401_model397_blend_recovery_lock.json), [inference manifest](outputs/model401_model397_blend_valid_once/inference_manifest.json), [recovery receipt](outputs/model401_model397_blend_valid_once/recovery_receipt.json). Result SHA-256 `7a165024ab090c5a64eb032dbfd7af323f37d5dda032c20753881c89f5c0a58f`; base lock `2b92bdddf5d22980b1019e4c289f9cea41ca40719e228f9fdec8dd1101f819cf`; recovery lock `298ffa85ec00c8e0495a918f9308f65c3f85a9540e07aff97db0f316546a13f0`.

## Model402: Model397 grouped-OOF component·경계 오류 구조 진단 (2026-10-05)

- 고정된 Model397의 B4/HRNet grouped train-OOF 확률만 재사용해 70장/59 specimen group에서 8-connectivity GT 성분 크기와 GT 경계까지 거리별 FN을 분석했다. 주요 blend C는 `0.50*B4 + 0.25*HRNet seed1 + 0.25*seed2`이며 image-wise 4-class mIoU, empty-union=0으로 재계산한 `79.9858620%`가 기존 Model397 값과 일치한다. B4 단독 `79.1232674%`, HRNet 두 seed 평균 `79.2085382%`; 모두 train-OOF 결과이지 official valid 점수가 아니다.
- C의 class image-mean IoU는 Al `93.0701%`, Al3Ni `80.9244%`, Primary Si `86.9460%`, Eutectic Si `59.0029%`다. Eutectic Si GT component 중 1–64 native px의 `89.83%`에 FN이 하나 이상 있고 support-weighted FN rate는 `33.94%`; 64–256px component는 miss rate `99.19%`, FN rate `21.42%`다. 각 cell을 완벽히 고친다고 가정한 image-mIoU oracle delta는 각각 `+2.622%p` (group-bootstrap 95% CI `[+2.067,+3.228]`)와 `+2.138%p` (`[+1.885,+2.395]`)다. 256–1024px Eutectic component도 miss rate `99.97%`지만 oracle delta는 `+1.608%p`다.
- 고정 OOF arm 간 Eutectic Si class mean은 B4 `56.988%`, HRNet seed 평균 `58.046%`, C `59.003%`로, blend가 B4 대비 `+2.015%p` 높다. 다만 작은 GT 성분 miss는 여전히 약 89–91%, 64–256px 성분 miss는 약 99%여서 class-level 보완만으로 조각난 성분 문제가 해결되지는 않았다. 이는 고정 train-OOF 내부 차이이며 valid generalization 증거가 아니다.
- 경계 거리 분석에서는 Al3Ni GT 경계 0–2px FN rate `27.35%`, oracle delta `+4.445%p` (95% CI `[+3.899,+5.026]`); Primary Si는 FN rate `27.38%`, delta `+1.313%p` (`[+1.044,+1.623]`)다. Eutectic Si도 경계 0–2px FN rate `30.53%`, oracle delta `+5.222%p` (`[+4.639,+5.834]`)로 경계 오류가 크다. 경계에서 8px 이상 떨어진 Eutectic GT pixel의 FN rate는 `91.55%`지만 support가 작아 oracle delta는 `+0.192%p` (`[+0.074,+0.383]`)다. Eutectic 오류를 전부 boundary 현상으로 설명할 수 없지만, 내부 오류의 점수 영향이 크다고도 단정할 수 없다.
- **개선 판단:** 단일 global prior/bias 조정으로 세 오류를 동시에 해결할 근거는 없다. Al3Ni·Primary Si는 경계 특화 보정/방향 head 가설을, Eutectic Si는 작은 성분과 질감·내부 구조를 겨냥한 class-specific hard-example 또는 전문가 가설을 각각 분리해 검증할 근거가 생겼다. Oracle delta는 이상적 상한이지 기대 성능 향상이 아니며 서로 다른 cell의 delta를 더해서는 안 된다. 기존 Model395 sampler exposure gate 실패도 이 진단으로 뒤집히지 않는다. 별도 Claude 검토와 사전등록 paired grouped-OOF gate 전에는 GPU 학습하지 않는다.
- Model402는 CPU-only train-OOF 사후 진단이다. valid/test를 읽지 않았고 training을 시작하지 않았다. 20,000회 group bootstrap은 고정 OOF 예측에 조건부이며 독립 확인이 아니다. 산출물은 [결과 JSON](experiments/results/model402_model397_component_boundary_audit.json), [lock](experiments/results/model402_model397_component_boundary_audit_lock.json), [claim](experiments/results/model402_model397_component_boundary_audit.json.claim), [분석 프로토콜](experiments/model402_model397_component_boundary_audit.md), [실행 코드](experiments/model402_model397_component_boundary_audit.py)다. Result SHA-256 `e42822133d6a60b4a19164aed97e8bf8f7bb3434f726635b288ad24d0d3a0bcf`; lock `495693d8752e30d2479c0c75ea1b46868556cb09f406d306c2dd875405b8c146`; claim `69b2fce35d633630c6fbcd995e521178b1cebef43e928f6d175f831c4fc1b381`.
