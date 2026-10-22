# 네트워크 OOF 입력을 이용한 잔차 보정 후보

앞선 두 잔차 후보가 기각된 뒤 설계한다. 기존 base14는 유지하며 81% 목표는 미달이다. 같은 valid를 반복 개발에 사용했다는 한계는 계속 적용된다.

## 감사한 입력

기존 Model397의 Model329 B4 및 Model392 HRNet 두 seed의 시편 그룹 OOF 확률 12개를 원래 lock SHA와 대조했다. 70장·59시편·4fold(18,18,17,17)가 정확히 한 번씩 포함된다. 각 fold의 학습·제외 시편은 겹치지 않는다. B4 훈련 split, HRNet 실제 train/valid 링크, args와 run receipt SHA, OOF manifest의 seed/fold/확률 SHA를 확인했다. 감사는 GPU 및 공식 valid/test 입력을 사용하지 않았다. 이는 기존 입력의 검증 결과이고 새 모델 성적이 아니다.

## 고정 후보

확률 조합은 기존 Model397의 0.5 B4 + 0.25 HRseed1 + 0.25 HRseed2를 유지한다. 원래 구현의 정규화와 클래스 순서를 확인하고 main 순서 [Al3Ni, Primary Si, Al, Eutectic Si]로 변환한다. 실제 배열과 train GT로 기존 raw OOF 79.9858620081%를 재현한 뒤 사용한다. 확률 합·유한 값·shape·이미지 ID·원본 train 이미지와 GT SHA를 검사한다.

학습 입력은 원본 gray와 FP32 log(max(p,1e-7))이다. 원본 피클의 학습 이미지 예측이나 전체 train70 GT로 fit한 prior를 OOF 입력에 섞지 않는다. 정답은 제공 train70만 사용한다. valid/test 정답은 trainer에 전달하지 않는다.

첫 unweighted 잔차와 같은 CNN 11,876params, zero-init, delta 상한 ±1, seed81349, 8epochs, 1120updates, CE/Dice/잔차 L2, 샘플러와 증강을 사용한다. 수학적으로 같은 CE 계산과 엄격한 CUDA 결정성 검사를 유지한다. 최종 epoch8만 평가하며 epoch/seed/잔차 상한을 valid로 선택하지 않는다. 단순 가중 CE는 개선되지 않았으므로 OOF 입력 효과와 섞지 않는다.

## 평가와 한계

학습 입력은 Model397 raw OOF log-prob이며 배포 대상은 고정 exh14의 precleanup refined logits다. 네트워크와 후처리의 입력 분포 차이는 명시적인 위험이다. OOF 입력을 쓰는 것이 성능 향상을 보장하지 않는다. 기존 valid20 부모 fields의 SHA와 zero-refiner20 마스크 exact 검사를 통과한 뒤 최종 모델을 평가한다. 부모 출력 재사용 시간은 보정/eval 시간으로만 기록한다.

네트워크 확률은 그룹 OOF지만 새 보정기는 그 전체 70장 정답을 학습한다. 이 사실만으로 전체 pipeline의 독립 OOF 성능이라고 부르지 않는다. 추가 시편 교차검증도 교사가 다른 fold의 정답을 학습한 간접 관계가 남으면 nested OOF라고 부르지 않는다. 공식 valid20은 반복 사용했으므로 독립 일반화 증거가 아니다.

81%가 관측되면 원본 영상부터 전체 fresh 재현, PNG/GT 독립 공식 채점, 원본 source/weights 불변, test10 마스크와 일치율(정답 없음, test mIoU 없음), 한국어 재현 레시피를 확인한 뒤 완료를 판정한다. 유휴 GPU와 기존 lock을 검사하고 다른 사용자 작업·기존 dirty tree·릴리스를 보존한다.

현재 문서는 아직 실행 완료나 새 점수를 주장하지 않는다.

## 클래스 동률 처리

최초 재현에서 main 순서 argmax가 기존 team 순서 argmax와 동률 픽셀의 우선순위를 달리해 약0.0000037%p 차이가 났다. 원래 team argmax를 계산한 뒤 label ID를 main으로 변환해 기존 점수를 엄격하게 재현한다. 학습용 main log-prob 자체는 유지하고 해당 logit argmax의 점수와 동률 픽셀 수도 별도 기록한다. 이미 생성된70개 features는 새 계산 결과와 모든 배열이 정확히 같은지 확인해 재사용하며 덮어쓰지 않는다. 최초 실패 source/log/receipt를 보존한다.
