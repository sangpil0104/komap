# B4 가로4000 현재3fold의 train70 추가 학습

HR336의 통계 적응·짧은 미세조정·학습 문맥 확대는 성능을 개선하지 못했다. 원본 HR336도 CE 클래스 가중치가 없으므로 class-weight 누락으로 손실을 설명하지 않는다. 다른 주요 분기 B4 Model329를 대상으로 제공 학습 자료의 활용을 늘리는 고정 후보를 정의한다. 기존 fold 선택 탐색이나 중지된 base12 학생 증류와 다른 실험이며 아직 성능 증거가 없다.

현재 base14에 있는 Model329 fold1,3,4를 모두 보존한다. 세 초기 checkpoint를 각각 train70 전체로400 optimizer updates 추가 학습하고 최종 세 checkpoint를 함께 교체한다. fold별 valid 결과를 보고 선택하거나 fold를 제거하지 않는다. 각 부모는 그룹 fold 학습 모델이지만 추가 학습 이후에는 모두 train70을 보므로 새 모델을 OOF라고 부르지 않는다.

가로4000 full-image 통계/zcdf3를 원래 helper로 계산한다. 원본 실행 스크립트의 crop1536/batch2/uniform image sampling/CE+Dice(eps1)를 유지한다. original affine/D4/brightness/contrast/gamma/blur/noise augmentation을 별도 dataset에 복원한다. 각 원본 학습 이미지의 가로4000 전체 영상으로 mean/std/CDF를 계산하고 증강 crop에 적용한다. crop으로 통계를 다시 추정하지 않는다. 같은 seed20261009와 sample별 고정seed로 세 초기모델에 같은 image/crop/augmentation stream을 적용한다. train70/GT만 사용하고 valid/test 정답·외부 데이터·새 사전학습 다운로드는 학습하지 않는다.

원본 pretrained state에서 optimizer를 새로 시작한다. AdamW LR1e-5/wd1e-2,cosine400updates,clip1,FP16AMP/FP32loss,GradScaler1024. microbatch2의 추가 학습에서 BN running mean/variance/counter를 고정하며 affine는 학습한다. 구조·파라미터 수·fold 수·저장 dtype(FP32)를 유지한다. 마지막step400만 사용하고 중간checkpoint를valid로채점하지않는다. 매100updates에optimizer/RNG/state를보존한다. fold별최대2400초이며상한/외부GPU점유/OOM/nonfinite시우리생성자식만중지하고부분자료보존/자동재학습없음.

한 GPU3에서 세 fold를 순차 실행한다. 기존PID 종료 및 lock, GPU유휴3회를 확인한다. 먼저 synthetic batch2/crop1536 AMP forward/backward/optimizer step과원본checkpoint불변/BN고정/finite검사를통과해야실제학습한다. 메모리 실패시crop/batch를즉석으로바꾸거나valid기준을완화하지않고실패를기록한다.

세 최종checkpoint를검사하고기존14개중329의3개만바꾼별도묶음을만든다. 다른11개/identityTTA/TENT5/라우팅/계수/후처리는유지한다. 전체valid20을원본영상부터새실행하고공식image별4IoU(emptyunion0)→20장평균을PNG/GT로독립재채점한다. class/image변화·최악손실·실제단회시간·checkpoint bytes·source/원본불변을기록한다. 반복valid개발이므로일반화보장을하지않는다.81%관측시전체재현·독립채점·원본불변/test10마스크(점수없음)/한국어레시피를완료한뒤목표완료를검토한다. 기존base/릴리스/미추적파일은보존한다.
