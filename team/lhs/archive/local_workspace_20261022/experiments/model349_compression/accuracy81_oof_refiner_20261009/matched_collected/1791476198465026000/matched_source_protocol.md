# OOF 잔차 모델의 같은 네트워크 입력 평가

base14로 전이한 OOF 잔차 후보가79.999389%로 기각된 뒤 정의한 후속 진단이다. 목표81%는미달이고같은valid반복개발평가이며독립일반화증거가아니다. 모델을추가학습하거나epoch/seed/잔차상한/가중치를valid로선택하지않는다.

검증된최종epoch8.pt(SHA9fd39362251f737c092222317b7e775e533f89902135cc4839df739ce24ed63a)를Model401의같은Model329 B4/Model392 HRNet 두seed의4fold평균valid확률에적용한다. 기존0.5B4+0.25HRseed1+0.25HRseed2를유지한다. 모든확률SHA를기존Model401manifest와대조하고raw baseline79.4181708851%를기존argmax우선순위/공식4클래스empty0지표로정확하게재현한뒤사용한다.

학습과같이FP32log(max(p,1e-7))+rawgray를입력으로사용한다. Model349 prior/공간필터/cleanup을섞지않는다. 예측은원래team순서argmax후mainlabel변환으로동률우선순위를보존한다. zero-refiner20원래Model401마스크exact가필수다. 이평가로학습원397와전이대상base14의네트워크·후처리차이를줄였을때효과를본다. 결과가좋아져도분포차이가이전손실의원인임을확정할수없다.

원본확률을읽기전용재사용하고보정/eval시간만기록한다. 기존parent14와Model401전체source/weights/입력/릴리스는변경하지않는다. 유휴GPU/기존lock확인,타사용자보존,우리작업만관리한다. 결과PNG/GT를독립채점하고이미지/클래스차이를기록한다.81%관측시전체raw fresh재현/test10마스크(정답없음/test mIoU없음)/원본불변/한국어재현자료확인후목표달성을판정한다. 현재후속진단은아직실행하지않았다.
