# 정확도용 HR336 추가학습: CE/Dice 대조 vs CE/Dice/Lovasz

2026-10-08 TTA 결과/신규학습결과 확인 전 고정. 기존라이브TTA PID2141791의정상종료/완료receipt를기다리고최고raw점수가81%미만일때만진행. GPU7유휴/모든기존lock확인. 타사용자프로세스/원본파일보존,한GPU순차사용. 별도고유루트로기존학습/릴리스덮어쓰기없음.

제공train70이미지/정답만학습. valid/test파일을학습코드에서읽지않는다. 외부데이터/새사전학습다운로드없음. frozenModel349_exh14 HR336 원본FP32checkpoint를두arm공통초기화. 기존모델은이미train70으로학습됐을수있고독립OOF이력인증이없으므로신규OOF/일반화인증주장금지.

Control:0.5CE+0.5Dice. Treatment:0.5CE+0.25Dice+0.25LovaszSoftmax(각patch/4클래스평균). Lovasz는IoU surrogate이며native full-image 공식empty-union0 mIoU와수학적으로동일한학습목표라는주장없음. 참고원논문 https://openaccess.thecvf.com/content_cvpr_2018/html/Berman_The_LovaSz-Softmax_Loss_CVPR_2018_paper.html . 독립구현이며외부코드복제없음.

두arm모두4epochs,70×12crops/epoch,patch512/native256,input_scale2,기존trainaugmentation/positivecrop0.5,seed81336, microbatch2×accum4=effective8. encoderLR5e-6/decoder2e-5,AdamWwd1e-4,cosine4epochs,FP16AMP+FP32loss,clip1. 작은microbatch/단기추가학습에서기존BN통계를유지하도록모든BN eval,기존training-onlyboundaryhead동결/손실미사용은두arm동일. 모델나머지는train. epoch/index고정randomseed로crop/augmentation순서를짝지으며forward전에pairedtorchseed고정. optimizer/RNG/checkpoint는별도저장. 학습loss/업데이트수/소요시간기록,nonfinite/OOM즉시실패/부분결과보존. arm별최대40분,자동재학습없음.

마지막epoch4만평가한다. valid로checkpoint/epoch/LR선택없음. 기존14개model의HR336checkpoint만arm최종판으로교체하고identityTTA의base14공식레시피로원본valid20추론. 다른13원본checkpoint/fold/계수/TENT5/후처리유지. 두arm및부모비교로추가학습효과와손실변경효과를구분한다. image별4IoU(emptyunion0)→20장평균,PNG독립재채점/클래스·이미지변화/최악손실/conditionalgroupbootstrap. 동일valid의추가개발평가이므로독립일반화증거아님.

raw81%이상도재현/독립PNG채점/원본불변확인/test10마스크생성과정답없음명시후에만81%목표완료검토. 미달이면결과를보존해다음가설근거로사용한다. test mIoU주장금지.
