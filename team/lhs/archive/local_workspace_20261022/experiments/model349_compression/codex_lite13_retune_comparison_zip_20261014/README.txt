lite13 vs 계수 재조정 비교 — 2026-10-14

먼저 images/00_valid_summary.png, images/01_test_summary.png를 확인하세요.
images/valid: 20장 각각 전체 비교 1장 + 변화 집중 확대 2장.
images/test: 10장 각각 전체 비교 1장 + 변화 집중 확대 2장.

valid 그림: 입력 / 정답 / lite13 / 재조정 / 변경 위치 강조 / 오류 변화.
test 그림: 입력 / lite13 / 재조정 / 변경 위치 강조 / 변경된 픽셀의 새 클래스.
변경 강조: 노랑은 실제 변경, 분홍은 가시성을 위해 변경 주변 1픽셀만 확대 표시.
오류 변화: 초록=기존 오류 복구, 빨강=새 오류, 보라=오류 간 라벨 변경.
클래스 색상은 assets의 원본 팔레트를 그대로 사용합니다: {'Al₃Ni': [25, 76, 153], 'Primary Si': [76, 178, 76], 'Al': [153, 127, 76], 'Eutectic Si': [204, 204, 204]}

valid20 평균 80.72098350% -> 80.77934057% (+0.05835706%p).
test10 변경 38,776픽셀 (0.427285%).
수치 전체: valid_comparison.json / test_comparison.json / valid_image_scores.txt.
assets에는 원본 입력, 모델별 라벨 마스크, valid 정답이 있습니다. test 정답은 없습니다.

양쪽 모두 13개 동일 모델 가중치. 재조정 설정은 valid20에서 선택한 retune_full로 고정:
{
  "R": {
    "al_gamma": 1.0,
    "al3ni_gamma": 1.0,
    "eut_edge_blend": 0.125
  },
  "E": {
    "wb4": 0.4,
    "g1share": 0.7,
    "eclasses": [
      0,
      2
    ],
    "use_f": true
  }
}

측정: 각 valid 이미지의 4개 클래스 IoU를 평균한 뒤 20장 평균(빈 union=0).
valid20은 반복 개발 및 계수 선택에 사용됐으므로 일반화 성능 향상으로 주장하지 않습니다.
test10에는 정답이 없으므로 변경량/시각 비교만 제공합니다. test 보고 후 추가 계수 선택은 하지 않았습니다.
저장 확률 CPU 재결합입니다. 원래 lite13 valid20/test10 마스크 exact를 확인했습니다.
valid 재조정 점수는 기존 retune_full과 일치합니다. 새 GPU 원본 추론 검증은 하지 않았습니다.
test 기존 E는 분해·재구성 차이 0; B4 비교에는 선택 실험과 동일 FP16 fold 캐시 평균을 사용합니다.
해당 B4 캐시는 원래 fresh 가족 확률과 최대 0.00048828 차이가 있을 수 있습니다.
기존 모델/가중치/릴리스와 GPU 시드 학습은 변경하지 않았습니다.
