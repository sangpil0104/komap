lite16_N3_seed414

사용자 채택: R9+E6+F1=16개. E HR392 seed1 fold2만 N3 seed414/epoch25/last.pt로 교체. 이전 lite16 및 릴리스 보존.

valid20 fresh 80.6284614846% (+0.0738181969%p), 최악 이미지−0.076428%p. valid는 반복 개발 점수. test GT 없음. N3는 full-image raw/Z±4/CDF RGB8입력, scale2/single-view, 개별확률FP16 및 가족평균FP16. 나머지HR392는 widthnorm/flip4. E 비율 .5 B4ws +.25(seed1 family) +.25(seed2 family), F/R규칙 동일.

ANA 실행: `bash ~/komap/lite16_N3_adopted_20261013/run_lite16_N3.sh INPUT_IMAGES NEW_OUTPUT 0`. 원본 부모 코드 번들과 선택 자산이 필요합니다. 로컬 code.zip은 코드/설정/manifest/검증기록만 포함하며 가중치는 포함하지 않습니다. 서버 assets에는 선택 E/F 가중치가 연결되어 있고 R은 기존 부모 번들에서 선택9개를 읽습니다.

전처리 valid20/test10 전체 픽셀 일치 및 16가중치 SHA 확인. 패키징된 전처리·E직접평균·F는 기존 independent fresh 참조와 CPU대조. 변경한 portable 실행기의 새 GPU end-to-end smoke는 현재326 학습을 방해하지 않도록 미실행입니다.

L2 후속: 새N3 baseline 대비4시드 모두 평균+.0124~+.0225%p지만최악이미지−.1129~−.1275%p. A153/A63 손실 때문에 아직 추가교체 미채택. 최고valid시드 선택하지 않음.
