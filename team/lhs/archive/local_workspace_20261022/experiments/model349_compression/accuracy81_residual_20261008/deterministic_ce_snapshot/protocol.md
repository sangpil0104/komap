# 고정 앙상블 + 작은 잔차 보정 모델

전역 계수·공간 필터·TTA·단기 HR336 학습 결과를 확인한 뒤 설계한 후속 탐색이다. 원본 valid 성능을 보장하지 않는다. 기존 train70 teacher 캐시는 lite14 구성으로 현재 exh14의 fold/322/347과 달라 그대로 학습 입력으로 쓰지 않는다. 현재 원본 exh14의 고정 코드·14가중치·계수로 train70을 다시 추론한다. 원본 입력/SHA·부모14 checkpoint/source/config/calibration SHA를 동결한다.

모든 학습 정답은 제공 train70만 사용한다. valid/test 입력·정답은 trainer에 전달하지 않는다. 새 외부데이터/사전학습 다운로드 없음. 앙상블을 학생 모델로 대체하지 않고 앙상블의 pre-cleanup logits에 잔차를 더한다. 원본 priors/bias/Gaussian/bilateral와 이후 cleanup/Primary presence 규칙을 그대로 유지한다. 부모 train 예측은 부모가 이미 학습한 이미지에 대한 것이므로 이 자료를 held-out/strict OOF라 부르지 않는다. 같은 valid를 반복 사용했고 train-valid 시편 중복도 있어 독립 일반화 증거가 아니다.

입력은 원본 refined logits 4개 및 원본 grayscale1개다. logit은 pixel별4class 평균을 빼고[-8,8]clip/4,gray는[0,1]에서(값−0.5)×2로 바꾸어 CNN 입력을 만든다. 원래 logits 자체는 FP32로 보존해 초기출력이 부모와 정확히 같게 한다. CNN은 Conv3×3(5→32)/SiLU/Conv3×3(32→32)/SiLU/Conv3×3(32→4), normalization/dropout없음, 마지막conv zero-init이다. 11,876params이며 delta=tanh(output)로[-1,1]에 제한한다. 새 network의class order도 [Al3Ni,PrimarySi,Al,EutecticSi]다. output=부모logits+delta 뒤 원래 cleanup을 적용한다.

결과 전에 고정한 학습은 seed81349,8epochs,70×16native256crops/epoch,batch8,AdamW LR1e-3/wd1e-4,cosine8epoch,FP32,gradclip1이다. 50% positive crops에서 train GT Al3Ni/Eutectic 중 존재하는 class를 동일확률로 고르고, 나머지는uniform crops다. 모든features/GT에동일 H/Vflip·rot90를적용한다. epoch/index별seed를 고정하고 shuffled loader generator/RNG/optimizer/checkpoint를 보존한다. CE0.5+Dice0.5+0.05mean(delta²)를 최소화한다. 최종epoch8만valid 평가하고중간epoch/seed/LR/잔차상한을valid로선택하지않는다. training cap20분/foreigncompute/nonfinite/OOM시우리작업만중단하고부분산출물보존.

검증은 GPU7에서현재parent14로valid20 전체fresh재추론한후,최종잔차CNN을full-native영상에적용한다. parent와초기zero-refiner의마스크가모두exact여야성적을사용한다. CNN의유한forward/backward,zeroidentity,4classpalette,실제teacher precleanup 추출/원래마스크exact를합성 및실제부모산출물에서검사한다. 전체validPipeline 시간(teacherfresh+fields+refiner+fusion)과checkpoint크기를기록한다. 두단계source/입력/weights/SHA와새11,876params checkpoint/SHA를보존한다.

공식평가는이미지별4IoU평균후20장평균,빈union0. PNG/GT독립채점/클래스·이미지차이/최악손실을기록한다. 81%관측도전체fresh 독립재현·원본불변·test10마스크(정답없음/mIoU주장없음)와재현레시피를확인한뒤목표달성을검토한다. 신뢰구간/선택진단은같은valid조건부일뿐일반화보장이아니다.

GPU7의기존TTA/배율controller종료/완료를확인하고기존9개lock+전용lock,유휴3회/compute없음검사를통과한뒤한GPU순차로teacher/train/valid를실행한다. GPU3의경계학습은별도진행이며변경하지않는다. 타사용자작업/원본가중치/기존릴리스/dirty tree 보존. 신규파일은accuracy81_residual_20261008에만작성한다.

## GPU 변경 기록

train70 분기·마스크·학습 입력 추출을 완료하고 parent_masks_exact=true를 확인한 뒤 타사용자가 GPU7을 사용했다. 학습 시작 전 유휴 대기 controller2167874만 종료하고 GPU3을 3회 유휴 확인해 이어간다. 학습 recipe/모델/목적함수/seed/학습자료는 변경하지 않는다. trainer GPU 점유 검사와 inference GPU 영수증만 KOMAP_PHYSICAL_GPU를 사용하도록 바꿨다. 이전 코드·protocol·lock은 migration_snapshot_gpu7에 보존한다. resume_gpu3.py는 기존 teacher train70/features를 SHA 검증해 재사용하고 미완료 학습 및 fresh valid20만 순차 실행한다. 물리 GPU 변경은 기록하며 성능 개선이나 재현 성적을 미리 주장하지 않는다.
