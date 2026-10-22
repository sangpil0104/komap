# GPU3 병행 실행 변경 기록

2026-10-08 새 학습 결과를 보기 전에 실행 GPU/대기 순서만 변경한다. GPU3/4/6이 비었음을 확인했고 TTA는 GPU7에서 재개했다. 기존 대기 PID2148796에 학습/child가 없음을 확인 후 우리 대기 프로세스만 중단해 중복을 막았다. 원래 소스/프로토콜/예약 기록은 보존한다.

train70, 부모 HR336, seed81336, 손실과 learning rate, 4epoch/최종checkpoint 선택, paired crop/augmentation/optimizer, valid20 평가 방식은 동일하다. 새 루트 accuracy81_hrlovasz_gpu3_20261008에서 GPU3 하나만 사용한다. TTA의 결과로 레시피를 선택하지 않는다. TTA는 GPU7에서 이어간다. GPU3 유휴 3회 및 compute 없음 검사와 전용 lock을 적용하고, 학습 중 다른 GPU3 compute가 나타나면 우리 학습만 중단한다. 서로 다른 가설의 실험이며 원래의 GPU7 대기 학습은 실행하지 않는다.

기존 protocol의 GPU7/선행 TTA 정상종료 및81%미달 조건은 이 실행 순서 변경으로 대체한다. 새 raw81% 결과도 PNG 독립재채점/원본불변/test10 마스크 검증이 필요하다. 서로 다른 GPU의 소요시간을 속도개선 근거로 사용하지 않는다.
