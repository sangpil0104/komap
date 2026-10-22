# Model349 lite20: 8개 시드 가중치

[가중치 ZIP 파일](model349_lite20_seeds3407-3414_weights.zip)은 Git LFS에 저장되어 있습니다.
파일 화면의 Download 버튼으로 내려받거나 Git LFS가 설치된 환경에서 `git lfs pull`을 실행하세요.

시드 **3407~3414**의 HRNet336 `last.pt` 8개와 각 `args.json`, 가중치별 SHA256, manifest를 포함합니다.
기존 R/E/F 앙상블의 `R/weights/model336_hrnet_best/last.pt` 슬롯에 들어가는 모델입니다.
전체 앙상블 실행에는 나머지 공통 가중치와 기존 추론 코드도 필요합니다.

원본 가중치·설정·메타데이터의 바이트를 유지하고 ZIP 압축 방식을 **표준 ZIP-LZMA**로 바꿨습니다.
ZIP 내부 모든 파일의 SHA256이 원래 ZIP의 동일 파일과 일치하는지 검증했습니다.
가중치의 정밀도나 저장 형식을 변환하지 않았습니다. ZIP 자체의 SHA256은 재압축으로 변경됩니다.

## 검증 및 압축 해제

```bash
sha256sum -c model349_lite20_seeds3407-3414_weights.zip.sha256
python -m zipfile -e model349_lite20_seeds3407-3414_weights.zip model349_lite20_seed8_weights
```

Python 표준 라이브러리(권장: conda komap) 또는 LZMA ZIP을 지원하는 7-Zip으로 압축을 해제하세요.
일부 기본 unzip 프로그램은 ZIP-LZMA를 지원하지 않습니다.

[기존 Release의 원본 ZIP 분할본과 복원 스크립트](https://github.com/CNUKoMaP/sppark/releases/tag/model349-lite20-seed8-weights-20261011)도 그대로 유지됩니다.
