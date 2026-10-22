# Model336 — HRNet_best recipe on our official image-mean evaluator

Source implementation: private [`CNUKoMaP/HRNet_best`](https://github.com/CNUKoMaP/HRNet_best), commit `97adcdf337ab78fb7ec1469fe48943ff9a6363ea`.
Related design source: [`CNUKoMaP/sppark`](https://github.com/CNUKoMaP/sppark), commit `e0d71b1`.

This isolated copy preserves the teammate's HRNet-W48 + SMP U-Net++ model,
2× 512px patch recipe, rare-phase-centered sampling, class-unweighted
CE+Dice, training-only class-agnostic boundary head, and overlap-tile/flip
inference. The training entrypoint only adds FP16 autocast and gradient
accumulation so a P100 can use micro-batch 4 with effective batch 8. It also
stores epoch-level optimizer/RNG state separately from the inference checkpoint
so an interrupted run can resume without changing the released checkpoint format.

Important evaluation correction: upstream `evaluate.py` reports pooled/global
mIoU. `evaluate_image_mean.py` instead reports the official image-mean
four-class mIoU under empty-union=0, skip, and one policies, plus pooled mIoU
as a diagnostic only. Checkpoints and probability maps are written to a new
Model336 output directory; no prior experiment is overwritten.

The inspected `sppark` source has a DINOv3 global-context + local-detail CNN
hybrid and a design proposal for HRNet features plus stride-2/stride-1 local
detail. Its archived `accuracy_v1` CNN baseline reports 78.4729% image-mean
mIoU; the HRNet design section is not a measured HRNet result. We first test
the complete released HRNet-W48 recipe as a controlled branch. If useful,
follow-up work can test the `sppark`-style full-image context/local-detail
fusion around HRNet, then check class-wise fusion with Model270 using the
saved validation probabilities.

The queue also runs a CPU-only 625-route search (Model337) after Model336
validation/test inference. Each of the four classes independently keeps the
Model270 probability or blends in the HRNet probability at gamma
0.25/0.50/0.75/1.00. The route evaluator verifies image order, exact GT-mask
alignment, class order, and probability normalization before scoring; it
reports grouped leave-one-specimen-out route selection alongside raw valid
scores to expose selection optimism.

Training is scheduled after the existing Model326→328→329→333 GPU queue so
experiments continue to use the P100 serially. Data root on the server is
`/workspace/komap_model1/data` (the extracted train/valid/test ZIPs).
