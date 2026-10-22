# 정확도 탐색4: 전역 공간 후처리100개

2026-10-08 결과 전 grid 고정. GPU7은타사용자작업/우리재개watcher대기중이므로Mac CPU만사용. 기존14fold/융합/priors/bias/TENT/원본확률고정.

PrimarySi gaussian_blend[0,.25,.5,.75,1] × EutecticSi eut_edge_blend[-.25,0,.125,.25,.5] × cleanup_threshold[0,32,128,256] =100후보. negative Eutectic blend는bilateral완화반대방향의고정unsharp로그확률보정이며원본입력에동일적용한다. 정답/파일명에따라이미지별처리선택없음. 기존setting .5/.25/128포함하고20마스크/IoU정확재현을필수로한다.

비선형prior_terms와공간필터를영상별1회계산하고선형결합+원래cleanup/PrimarySiabsence rule을후보별적용한다. 선정후보는원래route.predict함수로20장을다시계산해IoU전부정확일치확인. emptyunion0의4class/image평균→20image평균. 클래스/이미지손실/조건부group-bootstrap/전체고정grid specimen제외선택진단보고. 이미반복개발한valid/cache이므로탐색결과이며독립일반화/원본전체새재추론/81달성으로주장하지않는다. 기존모델/릴리스보존.
