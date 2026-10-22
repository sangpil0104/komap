# lite13 retuned GPU 3~6 빌드 검증

검증일: 2026-10-14 (한국 시간). **학습과 실제 모델 추론은 시작하지 않았습니다.**

- GitHub 고정 커밋: `b6fa9f84eb6415a1a4258b717467350355237e6c`.
- ZIP SHA-256: `23a7a085eb075fb4303fb5505dc99c1003abe5804af635b20a292f857184fb69`.
- GitHub에서 다시 다운로드했고, 제공된 SHA-256 및 압축 해제 원본의 바이트 일치를 확인했습니다.
- `source/`에만 지표, 터미널 진행률, 실행 기록, 추론 타이머 연결을 적용했습니다.
- 최종 구성과 retuned JSON/YAML 설정은 원본과 동일합니다.
- 기존 결과·모델·실험 코드는 변경하지 않았습니다.

| 검사 | 결과 |
|---|---|
| 준비된 학습 계획 | 학습 33개, 최종 가중치 13개, 전체 51단계 |
| 실행 진입점 | 21개 스크립트의 원래 인자와 `--help` 실행 성공 |
| 초기 가중치 | ResNet34 / B4 / HRNet ImageNet 사전학습 가중치 CPU offline 로딩 성공, forward/backward 0회 |
| 데이터 | train70 / valid20 / test10, 190개 파일 원본 해시 일치 |
| OOF | 59개 시편 그룹, held-out 18/18/17/17, 학습과 held-out의 시편 중복 없음 |
| 단위 검사 | 5개 통과: image mIoU, 비교 그림 위치/색, 실패 단계 완료 방지, 자식 종료까지 시간 측정, 고정 계획 |
| 결과 내보내기 | 합성 fixture로 비교 그림 20장, 13개 가중치 슬롯, test10 파일, 추론시간 표·JSON·CSV 생성 성공 |
| GPU | UUID 매핑으로 물리 3·4·5·6만 지정, CUDA 장치 4개 확인 |
| 셸·파이썬 | 셸 문법 검사와 모든 실행 Python AST 검사 통과 |
| 사전 점검 | `./run.sh check` 통과 |

결과 내보내기 검사는 합성 라벨과 가중치 자리표시자 파일을 사용했습니다. 모델 성능을 측정한 검사가 아닙니다.
임시 fixture는 검사 후 삭제했으며 실제 `results/`에 가짜 실험 결과를 저장하지 않았습니다.

실행 환경은 기존 `komap`을 사용합니다: PyTorch 2.7.1+cu128, torchvision 0.22.1+cu128,
SMP 0.5.0, timm 1.0.29, Albumentations 2.0.8. GitHub의 PyTorch 2.8 환경과 버전이 다르며
환경을 덮어쓰지 않았습니다. import/인자/CPU 초기화 검사를 통과했지만 실제 전체 학습을 완료한 검증은 아닙니다.

세부 기록: `checks/preflight.json`, `runtime_cpu.json`, `design.log`, `export_cpu.json`, `source_integrity.json`.

추론시간은 사용자가 실행한 학습이 완료된 후 valid20 / test10의 새 추론에서 측정됩니다.
프로세스 시작·모델 로딩·전처리·R/E/F·후처리·출력 저장을 포함한 전체 시간과 장당 평균을 보고합니다.
학습·채점·보고서 생성은 해당 시간에서 제외하며, R/E/F 단계별 시간도 함께 보존합니다.

사용자 시작 명령:

```bash
cd /Data3/sppark/komap/experiments/model349_lite13_retuned_reproduction_gpu3456_v1
./start_nohup.sh
```
