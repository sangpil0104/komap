# Model404 v5: scorer correction before training

## Revision note

v4 generalized the tiny-component candidate to Al3Ni and Eutectic Si. Before any calibration or optimizer step, a synthetic scorer test found that the class-name formatter used a dictionary view incorrectly. v4's source lock is preserved; its waiting queue was stopped while Model403 still held the P100. v5 changes only the scorer entry point and queue/runner lock references. Loss, paired trainer, calibration, fold inventory, augmentation, training schedule, and adoption gates remain the v4 protocol. No Model404 v4/v5 optimizer step or calibration step has occurred.

The v5 scorer shim calls the immutable v4 OOF evaluator and corrects the hit-rate class-name formatting before that evaluator aggregates paired group-bootstrap metrics. Synthetic checks cover both target classes and all four component-size guards.

## Fixed protocol inherited from v4

- A CPU-only fixed train-OOF confusion audit (70 images/59 specimen groups; no mIoU tuning) supports applying the loss to both target classes: among native 1–<64 px Al3Ni component pixels, the Model397 blend predicts Al for `26.26%` and Eutectic Si for `12.86%`; among tiny Eutectic-Si pixels, `21.91%` are predicted as Al3Ni and `11.46%` as Al. Majority component recall is below 50% for `40.83%` of tiny Al3Ni and `31.90%` of tiny Eutectic components; complete misses are `25.18%` and `21.51%`. These are selection-conditional train-OOF diagnostics, not validation/test estimates. [Audit result](results/model404_tiny_component_confusion_audit.json).
- Target classes: Al3Ni (class 1) and Eutectic Si (class 3), native GT component area `[1,64)` px, approximated on augmented training masks by `[4,16)`, `[16,64)`, `[64,256)` px at nominal 2× scale. Crop-border components are ignored.
- Candidate uses component-normalized `-log(mean p(true class))`, averaged equally by nonempty crop×class×size-bin cells, alongside CE+Dice and boundary loss. Compute-matched control uses only CE+Dice and boundary loss.
- Both arms fine-tune each Model392 seed1 fold checkpoint for 8 epochs with identical per-item-seeded sampling, optimizer, scheduler, and training parameters. Fold coefficient is fixed by 8-batch outer-train-only decoder-gradient calibration with zero optimizer steps.
- Paired 4-fold specimen-group OOF, 70 training images/59 groups, official image-wise mean of four class IoUs, empty-union=0. Paired group bootstrap 20,000 draws. Official valid/test remain unread; test stays sealed.
- Adoption gates unchanged: overall `>=+0.20%p`, paired CI lower `>0`, positive folds `>=3/4`, worst fold `>=−0.30%p`; targeted Al3Ni/Eutectic class-IoU deltas nonnegative and all other classes `>=−0.50%p`; Step4 `>=−0.30%p`; tiny and medium component-hit deltas nonnegative with CI lower `>=−5%p` for both target classes.

The v5 lock supersedes v4 before all GPU calibration/training. The queue starts only after Model403 releases its P100 file lock.
