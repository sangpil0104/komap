# 외부 원본 및 출처

- `CNUKoMaP/HRNet_best`, commit `97adcdf337ab78fb7ec1469fe48943ff9a6363ea`: Model336 HRNet-W48 + U-Net++ 재현에 사용한 원본 recipe. 비공개 저장소다.
- `CNUKoMaP/sppark`, commit `e0d71b1`: 전역 문맥/국소 세부 설계 아이디어를 검토하는 데 참고했다. 비공개 저장소이며 HRNet detail-path 관련 내용은 측정된 결과가 아니라 제안으로 취급했다.

P100 환경에 맞춘 Model336 수정과 해당 실험의 평가는 [전체 모델 결과](model_results.md) 및 [`model336_hrnet_best/README.md`](../experiments/model336_hrnet_best/README.md)에 기록되어 있다. upstream 팀의 pooled mIoU를 대회의 image-wise 평가 지표와 혼동하지 않는다.
