# Model401 확률의 릴리스 route 보조원 CPU 검사

- 부모: 서버 `/workspace/model349_compress/exh14_base_search_20261008/runs/control14_valid20`의 원본 릴리스 14가중치 valid20 확률 캐시. 원본 bundle·캐시 변경 없음.
- 보조원: `/workspace/komap_model1/outputs/model329_fullstats_width4000_uniform_group_oof/valid_ensemble/probabilities.npz`(Model401 잠금 SHA `80ebfddad10abb1bb5f1d7af8472f477fd5b6435b3c7fef0bfa51e21e1314b05`)와 Model401 저장 HRNet seed1/2 평균확률 각20개. `0.50 B4 + 0.25 HR1 + 0.25 HR2`; 팀→route 채널순서 변환.
- 부모 cache를 원본 실행기의 `predict_fixed`로 재현해 PNG exact 및 저장 mIoU 일치, 보조원 argmax를 Model401 valid79.4181708851%와 비교하는 gate가 통과해야 혼합 점수를 본다.
- 고정 혼합: 릴리스 원래 gamma329/336/347까지의 `mixed` 확률에 Model401 class0(Al3Ni)/class2(Al)를 `replace_classes`로 추가. `(gamma0,gamma2)`는 `(0,0)`, `(0.25,0.25)`, `(0.50,0.50)`, `(0.75,0.75)`, `(1,1)`, `(1,0)`, `(0,1)`, `(0.5,1)`, `(1,0.5)`의 9개. 이후 원래 `route.predict` 후처리 그대로.
- valid20 official image-wise 4class IoU(empty=0), 시편 leave-one-out 9설정 선택과 릴리스 대비 클래스·20이미지 차이를 기록한다. 부모 확률, 원본14가중치, Model401 원자료, 기존 base 포인터 보존. test·GT 학습 없음. 반복valid 탐색·독립 일반화 한계.
