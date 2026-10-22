# HR336 경계 보조 손실을 유지한 추가 학습

CE/Dice 4epoch control80.1856429%, Lovász80.1806589%를 확인한 뒤 설계한 후속 탐색이다. 두 모델 모두 원래 HR336의 boundary auxiliary loss를 제외했으며 Eutectic Si 평균이 각각 약0.45%p 악화됐다. 보조손실 제외가 원인이라고 단정하지 않고, 같은 조건에서 복원한 후보를 검증한다. 반복valid개발평가/사후가설이며 독립 확증·일반화보장은 아니다.

기존 HR336 원본 부모에서 다시 시작한다. train70 정답만 사용하고 외부데이터/추가사전학습/valid·test학습을 하지 않는다. 이전 control과 동일한seed81336,4epochs,70×12crops,scale2/patch512,batch2×accum4,encoderLR5e-6/decoder2e-5,AdamWwd1e-4,cosine4epoch,BN통계고정,augmentation/RNG/FP16AMP+FP32loss/clip1을 유지한다. 같은 stream을 재현하며 기존 control을 재학습하지 않는다.

차이는 objective와 boundary head gradient다. Segmentation0.5CE+0.5Dice에 원래 HR336의class-agnostic boundary head loss1.0을 더하고 그head도학습한다. GT3×3 이웃에둘이상클래스가있는픽셀을경계band로두고,원래정의대로batch neg/pos로positiveweight한BCE + 전체batch softDice(eps1)를쓴다. 보조loss는학습만사용하고 inference 모델/14개수는유지한다. 구현은기존원본training losses.py의BoundaryHeadLoss/gt_boundary_band와동일식을별도파일로복원하고CPU synthetic/부모forward·backward smoke를확인한다. 기존원본소스를변경하지않는다.

최종epoch4(420optimizerupdates)만전체 원본valid20재추론한다. 중간epoch를valid로선택하지않는다. 다른13checkpoint/fold/융합/라우팅/TENT5/후처리/identityTTA는유지한다. 원래base와이미완료된동일stream CE/Dice control둘다비교해boundary복원효과와추가학습전체효과를구분한다. 코드·train140파일SHA·부모SHA·환경·final checkpoint SHA·시간·20PNG/공식4class IoU(빈union0)를보존한다.

GPU3의직전paired 실험완료/sourceunchanged receipt와PID2157571 종료를확인한뒤GPU3유휴3회/기존학습lock+새전용lock을획득하고실행한다. 타GPU작업/타프로세스를변경하지않는다. 외부GPU3 compute등장/nonfinite/OOM/40분trainingcap시우리작업만중단하며부분checkpoint보존. 신규독립루트accuracy81_hrboundary_20261008에만쓴다. 원본14/기존릴리스/이전결과보존.

81%관측도독립PNG채점·전체 원본재현·원본불변/test10마스크생성확인후목표달성을검토한다. test정답없음/mIoU주장금지. 81% 미달이면실험근거를기록하고다음방법을검토한다.
