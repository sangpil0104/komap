experiments - 실험 모델 (모델 이름이랑 소스코드)
results - 결과 md파일
AGENTS.md - 실험 어케할지 기본 설정?

이정도만 보쇼

## 최신 실험 결과

- [전체 결과 목록](results/RESULTS_INDEX.md)
- [1002 Model349 CE 클래스 가중치 제거 결과](results/model_results_1002_1/model_results_1002_1.md): valid20 image mIoU **80.040537%**.
- [1002 웹 보고서](reports/model_results_1002_1_web_v1/index.html): 모델 구조·적용 사항·클래스별 조합과 검증 비교 이미지 20장. 폴더 전체를 내려받아 `index.html`을 브라우저에서 열면 됩니다.
- [0930 Model349 학습·조합 결과](results/model_results_0930_1/model_results_0930_1.md)
- [CE 클래스 가중치 제거 실험 구성](experiments/model349_no_class_weights_v1/README.md)

결과 지표는 이미지별 4클래스 IoU 평균을 구한 뒤 검증 이미지에 대해 평균한 image mIoU입니다. 정답·예측에 모두 없는 클래스는 IoU=0입니다. 기존 학습 모델과 데이터는 로컬에 보존하며, GitHub에는 보고서·지표·실험 소스와 웹 보고서용 비교 이미지를 저장합니다.

- [HRNet_taeho_no_gaussian 추론 ablation](experiments/hrnet_no_gaussian_inference_v1/README.md)
- [완료된 조건 비교표](experiments/hrnet_no_gaussian_inference_v1/COMPARISON.md)
- [이미지별 변화와 연산량 진단](experiments/hrnet_no_gaussian_inference_v1/DIAGNOSTICS.md)
