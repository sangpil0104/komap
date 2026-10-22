# 과거 실험 실행 스크립트

이 네 launcher는 실험 작업공간에 있었지만 첫 GitHub archive에서 빠져 provenance와 source 목록을 완성하기 위해 보존했다. 현재 Model336 이후의 GPU queue에는 포함되지 않는다.

| 파일 | 당시 목적 | 재실행 주의사항 |
|---|---|---|
| `run_stronger_sequence.sh` | Model82–90에 해당하는 순차 backbone 실험 | 과거 `/proc/435828` PID 종료를 기다리고 GPU 3을 선택한다. 실행 전에 process guard와 device를 수정해야 한다. |
| `run_effv2_recovery.sh` | EfficientNetV2-S 복구 실험(Model84) | 과거 `/proc/436112` PID 종료를 기다리고 GPU 3을 선택한다. process guard와 device를 수정해야 한다. |
| `queue_model326_after325.sh` | Model322/325 완료 후 Model326 실행 | 이전 output 경로와 GPU 6 여유 메모리 확인을 사용하므로 경로·선행 조건을 확인해야 한다. |
| `run_model326_metric_aligned_group_oof.sh` | 이전 metric-aligned EfficientNet-B4 grouped-OOF 변형 | 4 fold × 16,000 update의 고비용 recipe이며 후속 선정된 Model326 full-image-stat pipeline과 다르다. |

대회 데이터, checkpoint, 생성 probability map은 포함하지 않는다. 코드와 경로를 현 환경에 맞게 조정해야 할 수 있으므로 이 스크립트들을 확인 없이 실행하지 않는다.
