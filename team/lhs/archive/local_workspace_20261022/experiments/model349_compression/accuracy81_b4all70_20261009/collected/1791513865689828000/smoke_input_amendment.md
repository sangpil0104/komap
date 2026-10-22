# 학습 전 입력 검사 수정

첫 합성 검사에서 independent normal random 3채널과 random target의 AMP backward가 nonfinite여서 optimizer step 전 종료했다. 실제400updates 학습은 시작하지 않았다. 실패 당시 source/preflight/로그/receipt를 attempt1_failed_synthetic에 보존했다.

같은1536/batch2/부모fold1/loss/GradScaler1024로 기존전처리를거친train70의고정sample0,1을읽어별도진단했다. finite loss/모든gradient/clipnorm/실제step/BNbuffers불변/원본SHA불변이통과했다. peakallocated13,936,979,968bytes였다. 이두입력의결과만으로모든향후crop의수치안전성을보장하지않는다.

입력게이트만고정traincrop0,1로수정한다. synthetic검사라고보고하지않고train GT를읽었다고명시한다. valid/test GT를읽지않으며gradient검사기준/loss scale/loss/학습recipe/400updates/epoch선택은바꾸지않는다. 진단의in-memory모델은폐기됐고세실제학습은각원래checkpoint에서새로시작한다. 실제학습중nonfinite/OOM/외부점유는계속즉시실패처리한다. valid성적을보고조정한수정이아니다.
