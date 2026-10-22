# 클래스별 결합과 후처리

클래스 순서는 [Al3Ni, Primary Si, Al, Eutectic Si]. 팀 순서 [Al, Al3Ni, Primary Si, Eutectic Si]의 확률을 [1,2,0,3] 채널 인덱스로 재배열한다.

| 단계 | Al3Ni | Primary Si | Al | Eutectic Si |
| --- | --- | --- | --- | --- |
| Model339 | 326 / 0.70 | 326 / 1.00 | 322 / 0.60 | 269 / 0.70 |
| Model341 | 329 / 0.50 | 329 / 1.00 | — | — |
| Model346 | — | 336 / 0.25 | 336 / 0.75 | 336 / 0.75 |
| Model349 | — | — | 347 / 0.25 | — |

해당 클래스만 \(\widetilde P_c=(1-\gamma)P_c+\gamma Q_c\)로 교체한 뒤, 네 클래스 전체를 \(P_k=\widetilde P_k/\sum_j\widetilde P_j\)로 정규화한다. 다른 클래스도 재정규화의 영향을 받으므로 각 단계 계수를 곱해 전체 가중치로 해석하지 않는다.

Model339 내부 순서는 Primary Si → Al3Ni → Eutectic Si → Al. 이후 Model329, Model336, Model347 순서로 각 클래스 인덱스의 0이 아닌 계수를 처리한다.

| 처리 | 정확한 설정 |
| --- | --- |
| 밝기 prior | Z-score prior0.85 + CDF prior0.15; 적응형 차이 항은 Al3Ni에 적용 |
| Logit 보정 | log(max(P,1e-7)) + 0.09×prior; prior_scale1.25, beta_scale0.80 |
| 클래스 bias | 영상 통계에 따른 기본 bias + al_delta0.00, eut_delta+0.15 |
| Primary Si 공간 필터 | Gaussian(sigma0.50)0.50 + bilateral0.50 |
| Eutectic Si 공간 필터 | 원래 logit0.75 + bilateral0.25 |
| Bilateral | Z-score guide, radius2, spatial sigma1.0, range sigma0.60 |
| 연결 성분 정리 | Primary Si의 128픽셀 이하 성분을 가장 점수 높은 다른 클래스로 재할당 |
| 영상 면적 규칙 | 정리 후 Primary Si 비율이 1% 이하이면 남은 해당 픽셀도 다른 클래스로 재할당 |

작은 Primary Si가 제거될 수 있는 규칙이므로 추론 방법의 일부로 공개한다. 영상별 source 선택·bias 조건은 source_snapshot의 highres1024_search.py 및 hard_source_search.py에 보존되어 있다.

그림: [Model349 전체 구조](../figures/09_model349_pipeline.png), [클래스별 결합 계수](../figures/12_classwise_fusion.png)
