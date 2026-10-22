# soft IoU 고정 E 비교

train70 fold1 held18 독립 채점. Dice 4 epoch와 같은 예산의 soft IoU 4 epoch를 비교했다. 고정 E는 78.100906% → 78.100519%, −0.000388%p. HR 단독은 −0.025295%p. 개선이 없어 다른 fold 확대는 미채택한다. 입력 SHA 불변 확인. 공식 valid20/test 점수가 아니며 81% 목표는 미달성이다.
