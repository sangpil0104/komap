# Upstream references and provenance

- `CNUKoMaP/HRNet_best`, commit `97adcdf337ab78fb7ec1469fe48943ff9a6363ea`: source recipe for the Model336 HRNet-W48 + U-Net++ reproduction. Repository is private.
- `CNUKoMaP/sppark`, commit `e0d71b1`: reviewed for global-context/local-detail design ideas. Repository is private; its HRNet detail-path discussion was treated as a proposal, not as a measured result.

Model336 changes for the P100 environment and its experiment-specific evaluation are documented in `docs/model_results.md` and `experiments/model336_hrnet_best/README.md`. Do not interpret the upstream team's pooled mIoU as the competition's image-wise metric.
