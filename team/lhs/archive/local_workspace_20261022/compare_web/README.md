# KoMaP corrected ensemble valid error audit

`valid.zip`의 20개 샘플에 대해 공식 클래스 매핑을 적용한 corrected ensemble(model57 계열) 원본, ground truth, 예측 mask, 픽셀 단위 error map을 비교하는 정적 갤러리입니다.

실행:

```bash
python3 -m http.server 8765 --directory compare_web
```

브라우저에서 <http://127.0.0.1:8765>를 엽니다.

기능:

- 원본 / 예측 mask side-by-side
- ground truth / error map 확인
- 샘플별 mIoU, error rate, error pixel 수
- 전체 valid global mIoU, image-wise mean mIoU와 클래스별 IoU
- mask overlay 투명도 조절
- 이전·다음 이동, 이미지 선택, 썸네일 이동
- 키보드 `←` / `→` 탐색

Error map은 초록색이 정답 일치, 빨간색이 오분류입니다. 클래스 색상 매핑은 평가 파이프라인 감사가 끝나기 전까지 주의해서 해석합니다.
