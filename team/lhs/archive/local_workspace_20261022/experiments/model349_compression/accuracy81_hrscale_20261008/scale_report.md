# HR336 배율 component replay 검증

빈 union=0인 이미지별4클래스평균 후20장평균. PNG/GT 독립재채점과실행metrics일치 확인. 동일valid반복개발평가이며 독립일반화 증거가 아니다.

| 구성 | image mIoU | base 대비 | 최악이미지손실 | 측정 구간 시간 |
|---|---:|---:|---:|---:|
| base14 | 80.2312313% | — | — | 별도환경단회 |
| baseline_replay_valid20 | 80.2312297% | -0.0000016%p | -0.0002881%p (A44_step4_sample3) | 0.00s (assembly only) |
| mean123_valid20 | 79.9252238% | -0.3060076%p | -0.9421040%p (A191_step4_sample3) | 2.84s (CPU averaging only; excludes all GPU work) |
| scale1_valid20 | 78.0854409% | -2.1457904%p | -8.9127804%p (A11_step2_sample4) | 30.16s (new HR336 component only; excludes replayed branches and fusion) |
| scale3_valid20 | 78.8111609% | -1.4200704%p | -6.9128047%p (A42_step2_sample4) | 200.98s (new HR336 component only; excludes replayed branches and fusion) |

가중치14개/1,755,428,956바이트 유지. HR336만 새 입력 배율로 원본 재추론하고 변경 없는6분기의SHA검증된확률을재사용했다. 전체fresh재추론/독립일반화성적이 아니다. 시간은 신규component 또는CPU평균구간만이며 전체모델시간/속도개선근거가 아니다. 기존릴리스/원본가중치보존. test mIoU 없음.


같은 재사용 분기/실행 환경의 대조군은80.2312297%다. 표의 canonical 원본80.2312313%는 별도2GPU 실행으로 아주 작은 수치 차이가 있다. 아래는 동일 replay 대조군 대비 비교다.

| 후보 | mIoU | 동일 대조군 대비 |
|---|---:|---:|
| mean123_valid20 | 79.9252238% | -0.3060059%p |
| scale1_valid20 | 78.0854409% | -2.1457888%p |
| scale3_valid20 | 78.8111609% | -1.4200688%p |

모든후보가낮아채택하지않는다. 입력배율과native문맥이학습배율2와달라생기는오류가커졌다는쪽의결과이며 학습을배율에맞춰다시수행한처치까지기각한것은아니다.
