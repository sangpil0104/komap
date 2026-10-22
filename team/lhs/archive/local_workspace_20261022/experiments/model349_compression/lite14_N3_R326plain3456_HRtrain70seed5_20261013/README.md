# lite14 — N3 / R326 plain3456 / HRNet train70 seed5

가중치 14개: R 8개 + E 5개 + F 1개. 기존 lite15를 보존한 별도 패키지입니다.

| 경로 | 구성 | 가중치 수 |
|---|---|---:|
| R | ResNet34 2개, EfficientNet-B4 5개, HRNet-W48 1개 | 8 |
| E | B4ws fold1·3, N3 seed414, HR392 seed1 fold4, HR392 train70 seed5 | 5 |
| F | 기존 epoch8 잔차 보정기 | 1 |

E 확률 = 0.50 × B4ws 두 fold 평균 + 0.25 × (N3 + HR392 seed1 fold4) 평균 + 0.25 × HR392 train70 seed5.

`hr_group_overrides["2"]`가 기존 seed2 fold1·4 그룹을 단일 train70 checkpoint로 대체합니다. 두 fold checkpoint는 포함하거나 로드하지 않습니다. seed5 checkpoint는 한 번만 추론합니다. R과 F의 설정은 lite15와 같습니다.

독립 저장확률 평가 valid20: 80.7207553045%, lite15 대비 +0.0381998870%p. 5개 seed 중 valid 점수를 비교해 선택한 개발 결과입니다. 독립 일반화 점수로 해석하지 않습니다. 원본 이미지 E/F 재현 결과는 `valid20_comparison.json`에, test10 마스크 변경량은 `test10_comparison.json`에 기록합니다. test 정답/정확도를 사용하지 않습니다.

## 실행

`parent_package`와 `assets`를 포함한 ANA 서버의 완전 패키지에서 실행:

```bash
python verify_package.py
python infer_compact.py --parent-package ./parent_package --assets ./assets --input INPUT_IMAGES --output NEW_OUTPUT --gpus 0
```

`NEW_OUTPUT`은 아직 없는 디렉터리를 지정합니다. 일반 입력 이미지 추론에는 GT가 필요하지 않습니다. 14개 checkpoint와 metadata SHA는 `selected_assets.json`에 고정되어 있습니다.

코드 ZIP은 가중치를 제외합니다. 서버 패키지에는 SHA 검증된 14개 실제 checkpoint가 별도 복사되어 있고 기존 lite15나 원본 파일에 대한 symlink는 없습니다. weights 약 2.03 GB(2,033,791,825 bytes).

검증 완료: 새 E/F 소스 추론 valid20 출력은 저장 후보와 20/20장 픽셀 exact, 80.7207553045%. Test10은 lite15 대비 26,642/9,074,964px (0.2935769%) 변경. 정답을 읽지 않았으며 test 정확도는 계산하지 않습니다.

전체 14개 가중치 포함 서버 패키지: `/home/sangpil/komap/lite14_N3_R326plain3456_HRtrain70seed5_20261013/`
