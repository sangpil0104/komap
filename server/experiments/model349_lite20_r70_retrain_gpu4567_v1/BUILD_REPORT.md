# lite20_r70 빌드 검증

2026-10-11 KST. **r70 학습과 GPU 추론은 시작하지 않았습니다.** 기존 seed100 탐색을 중단하거나 기존 소스·모델·결과를 변경하지 않았습니다.

GitHub 커밋 `34c46604a699a9067605540295f647d0d5cca443`에서 요청 ZIP을 받았습니다. ZIP 크기는2,488,372 bytes, Git blob SHA1과 ZIP CRC를 확인했으며911개 항목을 별도 `upstream/`에 풀었습니다. SHA256은 `d65995562b6505d09e6429f8f535c2f11b74558bb14074753d39567c7ab0cb72`입니다.

구성은 새 B4 학습3개, 총9개 단계입니다. seed0, train70, 각21,000 iterations이며, 나머지14개 가중치는 완료된 `model_results_20261011_1`의 lite20과 공유합니다. 결과 비교는17개 r70 대20개 lite20입니다.

통과한 검사:

- 실행 소스395개 Python AST, shell 실행기2개 문법 검사.
- 세 학습 명령의 실제 argparse 검사. 모델 생성/학습 전에 중단해 인자만 검증.
- CPU 검사7개: recipe/학습 DAG, full70 경로와 기존 fold 경로, 빈 클래스0/동률 우선순위, 기존 실험 잠금 감지, 완료 파일 변조 거부, 단계 의존성을 반영한 ETA, 모의 GPU4~7 배정 및 완료 단계 재개.
- 기존 fold 가중치를 **검증용 대역**으로 사용한17개/20개 패키징, 전체 해시와 공통14개 가중치 불변 검사.
- B4 ImageNet 초기화 cache 및 세 체크포인트의 CPU strict-load 검사.
- 새 upstream R 추론 코드로 기존 valid20 확률을 재계산: R image mIoU **0.8001013505955459**, 기존20개 예측 마스크와 픽셀 단위 완전 일치.
- 기존 예측을 양쪽에 넣은 결과 저장 통합 검사: 최종 image mIoU **0.8025584552762022**, 차이0/부트스트랩 구간0, 비교 이미지120장, test10 마스크, Markdown/JSON/CSV 및17개 패키지 저장.
- 소스530개/고정 입력355개 해시, train70/valid20/test10 구성과 학습 전용 데이터 경로 검사. `./run.sh check` 통과.

검증 산출물은 모두 `checks/`에 있습니다. `checks/package_fixture_*`의 가중치와 `checks/export_fixture_*`의 보고서는 검증용이며 **새 r70 학습 결과가 아닙니다**. 새 r70 성적은 사용자가 학습을 시작한 뒤에만 생성됩니다.

현재 seed100 탐색의 `outputs/search.lock`이 활성 상태임을 읽기만 하는 검사로 확인했습니다. r70 실행기를 지금 시작하면 그 잠금이 풀리고 GPU4~7이 비워질 때까지 대기합니다.

```bash
cd /Data3/sppark/komap/experiments/model349_lite20_r70_retrain_gpu4567_v1
./start_nohup.sh
tail -F logs/nohup_latest.log
```

중단·재개와 결과 배치는 `README.md`에 정리했습니다. 학습 중간 iteration 재개는 제공하지 않으며, 미완료 단계는 새 attempt로 다시 실행합니다. 완료 단계와 이전 시도는 보존됩니다.
