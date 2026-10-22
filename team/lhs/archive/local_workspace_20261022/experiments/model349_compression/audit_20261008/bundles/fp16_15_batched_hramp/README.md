# Model349 D3 표현모델 제거 경량 추론

원본 Model349의 고정 라우팅·후처리를 유지하고 322·347·기초 보조 분기와 표현모델 181을 생략합니다. 가중치 파일은 15개입니다.

실행: `python -B infer_lite.py --input /path/to/images --output /path/to/new_run --gpus 0`

정답 마스크가 있으면 `--gt /path/to/masks`를 추가합니다. 점수는 이미지별 4클래스 mIoU 평균이며 결측 클래스 IoU=0입니다.

검증 20장 재추론 점수와 비교 결과는 생성 실험 기록을 참조하세요. 시험 정답 점수는 확인되지 않았습니다.
