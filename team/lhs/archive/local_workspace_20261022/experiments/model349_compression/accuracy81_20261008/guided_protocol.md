# 영상 경계 기반 guided logit refinement 탐색

Kaiming He et al., Guided Image Filtering (ECCV2010)의 gray guidance 수식5·6·8을 독립 구현한다. 원논문: https://people.csail.mit.edu/kaiming/publications/eccv10guidedfilter.pdf . 공개 외부 이미지/사전학습/데이터를 사용하지 않고 제공 원본 valid 영상의 grayscale 밝기[0,1]만 guidance로 쓴다. 이 후처리의 효과는 가설이고 영상 텍스처가 실제 class 경계와 다르면 악화될 수 있다.

원래base14의 모든fold/가중치/계수/라우팅/TENT5/priors/bias/Gaussian·bilateral를 유지한다. 기존logit 공간의Gaussian·bilateral 후, cleanup 및Primary presence 규칙 전에 4개logit 모두에같은guidedfilter를적용해원래refined logits와혼합한다. image/class별계수선택없음. border는 기존native배열처리와유사하게OpenCV REFLECT_101로정하고원논문clip-window와border차이가있음을기록한다.

결과 전에 고정한 후보는 radius[1,2,4] × epsilon[1e-4,1e-3,1e-2] × blend[0.25,0.5,1.0]의27개와 기존기준blend0의1개,총28개다. epsilon은guidance[0,1]분산단위다. radius0identity,constant입력필드보존,constantguidance의2회boxmean,유한값/shape를합성으로검사한다. 기존base20마스크/80 IoU와정확일치해야채점한다. 최고는별도함수경로로다시계산해일치를확인한다.

공식산식은이미지별4classIoU평균후20장평균,빈union=0이다. 저장확률의탐색성적이며whole raw재추론/독립일반화성적으로주장하지않는다. 모든입력영상/SHA·캐시/SHA·코드/프로토콜을동결하고원본을변경하지않는다. mean/class/image 변화,조건부그룹bootstrap,고정28후보의시편제외선택진단을보존한다. 같은valid로설계·선택한편향은제거되지않는다. 유의미한후보또는81%관측시전체원본재추론/독립PNG채점/재현확인후판정한다. GPU/타작업/기존릴리스변경없음. test정답없음/점수주장없음.
