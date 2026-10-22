# 빌드 검증 결과

2026-10-11, conda `komap`. **학습은 시작하지 않았습니다.** 기존 실험/모델/결과는 수정하지 않았습니다.

- 계획: 100개 학습, seed 3407~3506, GPU 물리 4~7, 총122단계. R/E92 → 선택/OOF → F8 → 선택/패키징.
- Python AST: 656개 파일 통과. shell 실행기 2개 `bash -n` 통과.
- 학습 명령: 100개 모두 seed/출력/부모/feature 경로 검사; 서로 다른17종 recipe는 argparse에서 중단하여 실제 파서 통과 확인.
- CPU 검사8개 통과: 배분·DAG, 학습 인자, 336/347 결합, empty-class=0, active route/20 weights, 변조된 완료 파일 거부, 최고 단일 후보 보존, 모의 GPU4~7 UUID 배정/완료 작업 재개.
- HRNet과 B4의 ImageNet 초기 가중치를 오프라인 cache에서 CPU로 로드. optimizer step과 GPU 학습은 수행하지 않음.
- 기존 valid20 확률로 R route와 E/F를 다시 계산: **image mIoU 0.8025584552762022**, 기존 stage2 예측과20장 모두 픽셀 단위 일치.
- 2×2 비교 이미지20장 생성 확인. 비교 이미지 생성 전후 원래 후보 완료 기록의 SHA256 동일함을 확인.
- 기준 모델을 사용한20개 체크포인트 패키징 및 전체163개 패키지 파일의 엄격한 해시 검증 통과.
- 소스864개/고정 입력2949개 해시 및 train70/valid20 mask와 시편별 fold 검사 통과. `./run.sh check` 완료.

검증 기록은 `checks/`에 있습니다. 실제 새 시드 학습 및 새 체크포인트의 GPU 추론은 사용자가 시작한 실행에서 수행됩니다. 최종 결과를 저장하기 전, 선택된 패키지의 원본 이미지부터 시작하는 valid20 추론과 탐색 예측의 완전 일치를 검사하도록 연결했습니다.

실행:

```bash
cd /Data3/sppark/komap/experiments/model349_lite20_seed100_gpu4567_v1
./start_nohup.sh
tail -F logs/nohup_latest.log
```
