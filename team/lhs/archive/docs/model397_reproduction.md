# Model397 재현 자료

[Model397 학습·OOF 재현 코드 ZIP](../bundles/Model397_train_route_code.zip)

Model397은 Model329 EfficientNet-B4/U-Net과 Model392 HRNet-W48/U-Net++ 두 계열의 train-only grouped-OOF 확률을 `0.50 × B4 + 0.25 × HRNet seed1 + 0.25 × HRNet seed2`로 고정 평균한 조합이다. 마스크/데이터/체크포인트/확률 맵은 ZIP에 넣지 않았으며, 대회에서 별도로 제공하는 `train.zip`과 ImageNet encoder 사전학습 가중치는 외부에서 준비해야 한다.

기록값은 공식 empty-union=0, 이미지별 4-class mIoU 기준 **79.9859% train-OOF**다. Model392의 사전등록 two-seed gate가 실패하여 Model397의 공식 valid 평가는 승인되지 않았다. 그러므로 79.9859%는 official validation 또는 test 점수가 아니며, 80% 달성으로 해석하면 안 된다. ZIP 안의 단일 파이프라인은 네 그룹 fold를 만들고 4개의 B4 및 8개의 HRNet fold/seed 모델을 순차 학습한 뒤 OOF 점수를 계산한다. `valid.zip`/`test.zip` 및 그 경로는 실행하지 않는다.

폴더 구성, 고정 학습 설정, 실행법, 의존성 및 해석상 제한은 ZIP의 `README.md`에 정리되어 있다. 재현 보고서에는 원래 사전등록과 reference OOF/gate JSON도 포함했다. 실제 실행 환경 및 seed 차이에 따라 수치는 달라질 수 있다.
