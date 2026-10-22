# 실험 기록 보관

최종 제출본(lite13) 이전의 실험 자료입니다. 최종 제출본은 저장소 최상위 [`final_lite13/`](../final_lite13/)을 보세요.

| 경로 | 내용 |
|---|---|
| [`bundles/`](bundles/) | 실험 단계별 재현 번들: Model349 exh14·lite12~23·fusion805, Model364, Model397 코드 ZIP과 SHA-256, 학습 보조 파일 |
| [`experiments/`](experiments/) | Model3xx 실험 스크립트와 학습 코드(Model322 B4, Model326, Model336 HRNet 등). 색인은 [`SOURCE_INDEX.md`](experiments/SOURCE_INDEX.md) |
| [`results/`](results/) | 모델별 결과 JSON·CSV |
| [`docs/`](docs/) | 실험 결과 정리 문서: [전체 모델 결과](docs/model_results.md), [mIoU 기준별 최고 모델](docs/best_models_by_miou.md), 재현 안내, 사전등록 문서 |
| [`README_실험기록_20261014.md`](README_실험기록_20261014.md) | 최종판 정리 전 저장소 README |
| [`local_workspace_20261022/`](local_workspace_20261022/) | 로컬 작업 폴더의 코드·문서 백업. 나머지 파일은 릴리스 [`local-workspace-archive-20261022`](https://github.com/CNUKoMaP/lhs/releases/tag/local-workspace-archive-20261022). [백업 안내](local_workspace_20261022/README_백업안내.md) |

이 폴더의 점수는 대부분 검증용(valid) 20장에서 반복 비교해 고른 값이라, 독립적인 일반화 성능의 근거가 아닙니다.
