# HR seed2 부분집합 비교

선행세prefix후보미통과이후,seed2 유지fold 크기1~3의모든부분집합14개와control을비교한다. family비율및R/F고정,남은fold평균후FP16반올림. 이미원본재추론한4fold맵사용하며추가GPU추론없음. mean−0.01%p/최악−0.10%p gate와최소가중치선택. 시편제외안에서후보전체gate및선택을다시수행. 반복valid에대한후속탐색이며독립일반화미검증.
