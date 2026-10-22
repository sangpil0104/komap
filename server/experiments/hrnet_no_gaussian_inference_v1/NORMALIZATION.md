# 추론 시 디코더 정규화 통계 혼합

기존 HRNet 체크포인트의 가중치, BatchNorm running mean/variance, 업데이트 횟수는 보존한다.
모델은 `eval()`과 inference mode에서 실행하며 optimizer나 역전파를 사용하지 않는다.
선택한 BatchNorm2d 층의 출력 계산에만 아래 식을 적용한다. 입력 이미지 사이에
추정 통계를 전달하거나 저장된 통계를 갱신하지 않는다.

각 입력 텐서의 샘플·채널별로 공간 축 H×W에서 평균 `μ_input`과 분산 `v_input`을 구한다.
분산은 `torch.var_mean(..., correction=0)`이다. **배치 축은 합치지 않는다.**
타일 추론에서는 각 타일, 전체 이미지 추론에서는 각 전체 이미지가 한 샘플이다.

```text
μ_mix = (1 − α) μ_saved + α μ_input
v_mix = (1 − α) v_saved + α v_input
        + α (1 − α) (μ_saved − μ_input)^2
y = BN_weight × (x − μ_mix) / sqrt(v_mix + eps) + BN_bias
```

`BN_weight`, `BN_bias`, `eps`는 학습된 모듈의 기존 값이다. 위 분산식에는 두 평균의
차이로 생기는 혼합 분포의 분산 항도 포함한다. 앞선 층의 변경이 이후 층의 입력에
반영되므로, 모든 통계를 원래 네트워크에서 미리 계산해 두는 방식과는 다르다.

- `scope=decoder`: 디코더의 BatchNorm2d에만 적용한다. 인코더는 저장된 통계를 사용한다.
- `scope=encoder` 또는 `all`: 효과가 어느 부분에서 생기는지 비교하는 대조 실험이다.
- `statistic=mean`: 평균만 혼합하고 저장된 분산을 사용한다.
- `statistic=variance`: 평균은 유지하고 분산만 선형 혼합한다. 평균 차이 항은 넣지 않는다.
- `subset=last`: `decoder.blocks.x_0_<depth>`만 적용한다.
- `subset=except_last`: 마지막 블록을 제외한다.
- `subset=high_resolution`: 디코더 블록 키의 마지막 인덱스가 `depth−1` 이상인 블록이다.
- `subset=low_resolution`: 그보다 작은 인덱스의 블록이다.

입력 영상의 gamma 보정 계수와 여기의 `α`는 별개다. 예를 들어 입력 gamma 0.9와
정규화 α=0.05는 각각 영상 명암 변환과 특징맵 통계 혼합 비율을 뜻한다.

구현은 `normalization_adapter.py`에 있다. forward hook은 매 forward가 끝나면 제거한다.
`checks/normalization_real_model_preservation.json`에서 실제 HRNet의 모든 파라미터와
버퍼 해시가 추론 전후 동일함을 확인했다. 원본 체크포인트와 학습 소스의 파일 해시도
각 실행 시 검사한다. 하위 범위 확장 전후의 기본 동작은 bitwise 동일하게 검증했다.

α와 적용 범위는 valid20 결과를 보고 선택한 추론 하이퍼파라미터다. 독립 데이터에서
검증한 일반화 성능으로 해석하지 않는다. BatchNorm 출력 계산을 바꾸는 조건이므로,
단순한 출력 마스크 후처리와 구분해서 보고한다.
