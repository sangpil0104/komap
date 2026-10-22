# HR336 입력 배율 탐색

2026-10-08 현재 결합 TTA·paired Lovász 결과가 아직 완료되지 않은 시점에 고정한다. 두 실험의 구현/소스/학습은 변경하지 않는다. GPU7 기존 TTA PID2158061의 종료와 정상 completion을 확인한 뒤에만 동일 GPU에서 실행한다. 이전 TTA raw가81% 이상이면 이 실험을 건너뛰고 독립 검증을 우선한다. 다른 사용자 작업에 신호하지 않는다. GPU7 점유 확인/유휴3회/기존9개lock을 적용하며 다른compute 등장 시 우리worker만 중단한다.

현재 HR336은 scale2의512 network px /256 native px 문맥을 사용한다. 같은 가중치로 scale1(더 넓은 native 문맥)과 scale3(더 촘촘한 native 표현)를 시험한다. 성능 개선은 가설이며 학습 scale2와 달라 분포 변화로 악화될 수 있다. 새로운 학습/외부데이터/체크포인트/사전학습 다운로드 없음. HR336 원본20 이미지만 다시 추론하며 identity TTA, network patch512/batch2는 유지한다. 실제 inference 코드에서 overlap은 native px이므로 기존170/256의 비율을 유지하도록 scale1 overlap340, scale3 overlap114로 조정한다. scale3에170을 그대로 쓰면171px native tile의 stride가1px이 되어 부당한 계산 폭증이 생기므로 이를 실행 전에 확인하고 방지했다. 이 설정 변경도 결과 전에 동결한다.

평가 후보는 scale1 HR336 치환, scale3 치환, 기존2·신규1·신규3 확률을동일 비율1/3 평균한 HR336 치환의3개다. 기존2는 대조군이다. 다른13개 가중치/fold/클래스별융합/고해상도 경로/TENT5/후처리 모두 유지한다. 고정3개 결과를 한 번 평가하고 개별이미지/클래스/배율계수를 따로 고르지 않는다.

변경 없는6개분기는 기존원본control14_valid20의 raw확률을SHA 동결·검증 후 재사용한다. baseline336 포함 기존확률의 source/input/calibration/config provenance를 확인하고 기존base를 재조립해 metrics의80 IoU가 정확히 같은지 먼저 확인한다. 이 실험은 raw component 재추론+검증된분기replay다. **전체앙상블fresh재추론 또는 독립일반화 평가로 부르지 않는다.** 시간은신규HRcomponent+재조립구간만 기록하고 전체모델시간/속도개선으로부르지않는다. 81% 이상 관측 시 전체 rawvalid20 fresh재현과PNG/GT 독립재채점·test10마스크·원본불변검증이 필요하다.

공식산식은 이미지별4클래스IoU 평균후20장평균,빈union=0. 평균·클래스·이미지변화/최악손실을기록한다. valid반복개발평가·배율선택한계와test정답없음을명시한다. 원본14 checkpoint/소스/기존릴리스/기존부분산출물을보존한다. 신규결과는별도accuracy81_hrscale_20261008루트에만쓴다.
