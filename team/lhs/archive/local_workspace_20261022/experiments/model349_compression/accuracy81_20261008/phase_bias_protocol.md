# Al3Ni·Primary Si 전역 상대 logit 보정 탐색

이전 전역보정은 prior_scale/Al/eut_bias를 변경했고 Al3Ni와 Primary Si의 상대 bias를 독립 조정하지 않았다. 다음2개의 공통계수로 검토한다. 이 설계는 이전 valid결과를 이미 본 뒤 정한 탐색이다. 새 학습/외부데이터/GPU추론은 없으며 original14 source/가중치는 변경하지 않는다.

원래 priors/bias/Gaussian/bilateral 처리 후, cleanup/Primary presence 규칙 전에 Al3Ni logit 및 Primary Si logit에 각각상수를더한다. Al/Eutectic logits는 그대로이며 이미지/클래스별특별계수나 특정이미지예외는없다. 각bias[-0.2,-0.1,-0.05,0,0.05,0.1,0.2]의7×7=49개를결과전에고정한다. 기존14개fold/융합/라우팅/TENT5/후처리파라미터는유지한다.

baseline00의20마스크/80 IoU가 기존best14 replay와정확히같아야한다. 전역상수는확률calibration/판정경계를변경하며 새로운network가아니다. 최고도별도함수경로에서재계산해일치를확인한다. 공식산식은이미지별4class IoU평균후20장평균,빈union=0. 평균/class/image 변화,조건부groupbootstrap,고정49후보내시편제외선택진단을보존한다. GT는채점에만쓰고모델학습에는쓰지않는다.

같은valid의반복탐색성적으로선택편향을제거하지못한다. 캐시조합을whole raw/inference 재현/독립일반화성적으로보고하지않는다. 원본영상기준으로81%관측시전체fresh 원본재추론/PNG독립채점/test10·원본불변·재현레시피를확인한다. test정답없음/mIoU주장없음. 기존릴리스/타GPU작업/작업트리변경을보존한다.
