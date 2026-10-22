# 12개·10개 FP16판 일반화 가능성 감사 및 제한된 검증

사용자는 이미지가 train70/valid20/test10 합계 100장뿐이며 추가 정답은 없다고 확인했다. 이번 요청은 이전 GPU 종료 요청 이후의 새로운 검증 요청이다. 기존 후보·가중치·계수·후처리는 동결하고 추가 fold/계수 선택·학습은 하지 않는다.

엄밀한 결론 범위: 고정한 12/10판에 독립 정답 holdout이 없으므로 일반화 개선을 입증할 수 없다. 아래 실행은 학습 노출 감사, 이미 사용한 valid의 조건부 안정성 진단, 정답 없는 test의 출력 변화 진단이다. bootstrap CI를 선택 편향 교정이나 일반화 성적이라고 부르지 않는다.

## 고정 비교

- 후보: audit_20261008/bundles/cpu2_fp16_12 및 cpu2_fp16_10. 12판은 326 fold1·3, 10판은 326 모두 제거·Al3Ni gamma329=0.8. TENT5·OpenCV2, 나머지는 이전 설정 그대로다.
- 품질 비교의 주 참조: 동일 저장 FP16/TENT5인 cpu2_fp16_t5(14개). cpu2_lite15(동일 OpenCV2의 15 FP32), lite15, 보관 원본349도 부 비교한다. 동시기 FP16_14와 두 후보의 PNG/GT 점수를 읽고 재채점한다.
- train/valid/test의 파일 SHA·디코딩 픽셀 SHA·specimen(A숫자) ID 중복 및 masks 수를 확인한다. specimen 정의는 기존 prepare_group_cv.py의 group_id 함수와 일치하는지 확인한다.
- 결과를 보기 전에 확인한 overlap: train-valid 7 specimen(A125,A153,A181,A191,A40,A55,A7), valid는 20장·19 specimen. train 미노출 valid는 13장·12 specimen이고 후보 선택에는 여전히 노출됐다. 해당 부분집합을 성능과 무관하게 고정한다.

## CPU 진단

- 기존 마스크와 GT를 다시 읽어 four-class image mIoU(empty union=0)를 재계산한다.
- valid 전체, train specimen 중복 부분, train specimen 비중복 부분에서 평균·클래스별·최악 이미지·개선/악화 개수를 보고한다.
- specimen을 복원추출하는 paired cluster bootstrap: seed=20261008, 20,000회, percentile 95% 구간. 뽑힌 specimen에 속한 이미지를 모두 포함하고 image-weighted mean을 계산한다. 후보는 모든 bootstrap에서 고정한다. 구간은 동일 표본에 조건부이며 선택 편향은 미교정이다.
- leave-one-specimen-out 평균 변화 범위와 가장 영향 큰 그룹을 계산한다. 이는 새 모델 선택이나 독립 cross-validation이 아니다.
- 후보별 사전 평균 하락 ≤0.01%p·최악 이미지 하락 ≤0.10%p의 기존 품질 한도를 참고하되 새로운 bootstrap 채택 문턱을 결과 뒤 만들지 않는다.
- 326/329의 실제 args·CV symlink·fold manifest·checkpoint SHA를 감사한다. 전체 후보는 329 네 fold를 함께 사용하므로 train70의 각 이미지는 다른 세 fold 모델의 훈련에 들어갔다. 기존 single-heldout-fold OOF를 전체 12/10 후보의 OOF 성적으로 사용하지 않는다. HRNet 전체 train 노출도 코드와 args로 확인한다.

## test10 원본 재추론

- GPU 매뉴얼을 재확인했다. 서버 상태를 확인한 후 비어 있는 P100 GPU7 한 장에서 12판, 10판 순서로 실행한다. flock과 각 run/component 전 GPU memory·util·현재 inference 검사를 적용한다. 다른 작업은 종료하거나 조작하지 않는다.
- 신규 출력 root: /workspace/model349_compress/generalization_20261008/runs. 기존 결과는 덮어쓰지 않는다. 후보 코드/설정/가중치를 수정하지 않는다.
- 기존 lite15 및 lite14 test10와 입력/calibration SHA가 같은지 확인하고 이미지별·클래스별 mask agreement를 계산한다. 정답 없는 test로 IoU·성능 유지·일반화 개선을 주장하지 않는다. 한 모델을 정답으로 취급하지 않는다.
- test10 중 train 또는 valid와 specimen을 공유하는 4장과 나머지 6장을 구분해 출력 변화를 보고한다. 각 이미지의 학습/선택 노출 상태를 기록한다.
- 시간은 12/10 각각 단회 참고값이며 속도 확정은 하지 않는다. test 결과를 보고 후보/계수/후처리를 재선택하지 않는다.

## 종료와 기록

새 GPU 훈련·cross-validation 재학습은 이번에 하지 않는다. 현재 checkpoint로는 누수 없는 전체 후보 OOF를 만들 수 없으므로 그 한계를 증거와 함께 기록한다. 결과와 코드/설정/입력 해시·GPU 상태·로그를 한국어로 저장하고, 독립 일반화 검증 여부를 false로 명시한다. 기존 릴리스·가중치·작업 트리 변경을 보존한다.
