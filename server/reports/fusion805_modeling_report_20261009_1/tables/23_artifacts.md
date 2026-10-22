**표 23. 산출물과 확인 방법**

| 항목 | 경로/역할 |
| --- | --- |
| 본 보고서 | 모델링_결과서_Fusion805.doc / .docx / .md / .pdf |
| 그림 | figures/*.png (300dpi), *.svg, catalog.json |
| 표 | tables/*.md, *.csv; Word 문서에는 편집 가능한 표로 삽입 |
| 검증 20장 비교 | appendix/stage1 및 appendix/stage2, 각 20개 PNG |
| 수치·출처 | evidence/audited_metrics.json, pixel_changes.json, GitHub 사본 |
| 분석·생성 소스 | source/analyze.py, figures.py, report.py, render.py |
| 실행 소스 보존 | evidence/execution_source/ |
| 원본 모델 보존 위치 | results/model_results_20261009_2/selected_model/ |
| 검증 기록 | validation/; 문서 변환·이미지·링크·해시 점검 |

보고서 ZIP에 다중 GB 학습 가중치와 전체 train 원본 데이터는 중복 포함하지 않는다. 가중치 manifest와 기존 경로는 보존한다.
