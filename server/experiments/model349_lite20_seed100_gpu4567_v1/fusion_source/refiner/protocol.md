# OOF 보정기의 밝기 입력 두 고정 비교

같은네트워크입력평가도악화된뒤정의한다. 원본밝기의시편차이의존성을확인하는탐색이며원인이밝기라고확정하지않는다. 다음두후보는각각정확히한번8epochs최종모델만평가한다. 선택값의독립일반화보장을주장하지않는다.

원본 Model397 OOF70 features/GT,unweighted loss/seed81349/8epochs/1120updates/11,876params/delta±1/샘플러/증강을유지한다. 바꾸는것은원본gray입력의변환뿐이다. zscore후보는각원본전체이미지의gray평균/표준편차를사용하고z를±3clip후(z/6+.5)로저장입력범위[0,1]을맞춘다. std<1e-6이면.5상수다. no-gray후보는gray를항상.5로설정해network의centered입력이0이된다. 정답에따라정규화값을바꾸지않는다. crop전전체이미지통계로학습하고추론에서도전체이미지통계를사용한다.

기존고정Model401확률의raw baseline79.4181708851%재현및zero-refiner20maskexact가필수다. prior/cleanup을적용하지않고같은networkraw입력에만먼저평가한다. epoch/seed/잔차상한을valid로고르지않는다. 각trainer학습자료는train70만이다. 보정evaluator만validGT를읽는다.

두후보를한GPU에서순차실행하고원본확률/14parentsourceweights/SHA/기존릴리스를보존한다. 입력정규화합성검사(affine밝기/상수/범위/finite),sourceSHA,학습receipt,PNG/GT독립공식채점을남긴다. 원본확률재사용이며전체fresh시간을추정하지않는다.81%관측시전체fresh독립재현/test10마스크와일치율(정답없음/test mIoU없음)/원본불변/한국어재현자료확인후목표완료를판정한다.현재학습또는새점수는아직없다.
