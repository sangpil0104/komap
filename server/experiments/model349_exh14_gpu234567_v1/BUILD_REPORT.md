# 준비 및 사전 점검 기록

준비일: 2026-10-09 (Asia/Seoul).

**학습·실제 추론·TENT는 실행하지 않았습니다.**
GitHub exh14는 고정 추론 패키지이므로 valid20 평가 실행 환경을 준비했습니다.

## 출처와 보존

- 저장소: https://github.com/CNUKoMaP/lhs
- commit: `1704fff772a713101a2d63467b4d9321420ada3f`
- 파일: `Model349_exh14_code.zip` (`downloads/` 보존, SHA-256은 `provenance.json`).
- ZIP의 모든 파일과 실행용 `source/`의 대응 파일을 바이트 단위로 비교하여 일치 확인.
- 기존 릴리스에서 복사한 FP32 체크포인트 14개 전부 GitHub manifest 해시와 일치.
- 필요한 B4 `args.json` 7개를 별도 복사하고 `local_weights.json`에 출처·해시 기록.
- 검증 이미지·마스크 각 20개를 별도 복사하고 `data_manifest.json`에 출처·해시 기록.
- 기존 모델·실험 코드·이전 결과·conda 패키지를 수정하지 않음.

## 통과한 점검

1. conda `komap` 및 CUDA 가용성. 물리 GPU 2~7을 UUID로 변환한 후 6개 장치만 노출 확인.
2. 원본 recipe manifest, 체크포인트, B4 설정, 검증 데이터 해시와 이미지/마스크 크기·색상 확인.
3. 전체 14개 모델을 CPU에서 실제 구성하여 `load_state_dict`의 엄격 로딩 통과.
   가중치의 FP32 dtype·유한값 검사. forward/backward/optimizer 호출 없음.
4. 지표 합성 검사: Primary Si 없음 + 나머지 클래스 IoU 0.8 → image mIoU 60%.
   정답과 같은 예측이라도 한 클래스가 없으면 75%.
5. 서로 다른 이미지 크기의 합성 valid20에서 이미지 평균 67.5% 및 보고서·비교 이미지 20개 생성.
   원본 점수와 재계산 점수가 다르면 결과 저장을 거부하는 동작 확인.
6. 비교 이미지의 좌상/우상/좌하/우하 배치와 오류 위치의 정답 색·정답 위치의 흰색을 픽셀 단위 확인.
   `checks/layout_preview_synthetic.png`는 육안 확인용 합성 예측이며 실제 모델 결과가 아님.
7. 기존 결과 번호를 덮어쓰지 않는 새 폴더 할당 확인.
8. 셸·Python 문법 및 이 실행기가 만든 프로세스 그룹의 부모·자식 종료 동작 확인.

상세 기록: `checks/preflight_*.json`, `checks/preflight.log`, `checks/export_cpu.json`,
`checks/export_cpu.log`, `checks/runtime_cpu.json`.

## 실제 실행에서 확인할 사항

GPU forward, 전체 추론 시간, RTX4090에서의 최종 image mIoU는 사용자가 실험을 시작한 뒤 측정됩니다.
원본 P100 기록 점수를 로컬 재현 결과로 기재하지 않았습니다.
원본 추론 코드를 수정하지 않았으며, 다중 GPU 구성요소 스케줄링도 원본 실행기를 사용합니다.

사용자 실행: `./start_nohup.sh` 또는 `./run.sh evaluate --execute`.
