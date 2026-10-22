# Model404 v6: parent non-regression gate

## Revision note

An independent scorer audit found that v5's adoption checks could pass when the component candidate beat the compute-matched continuation control but both were worse than the Model392 seed1 checkpoint used for initialization. Before any v5 calibration or optimizer step, its waiting queue was stopped. v6 preserves all v5 objective, training, calibration, fold, OOF, component-hit and treatment-control rules, and adds a preregistered point-estimate non-regression requirement versus the fixed Model392 seed1 OOF parent. v5 lock/source files are preserved. No Model404 calibration or optimizer step has occurred.

## Locked v6 addition

The scorer already computes paired candidate-vs-Model392-seed1 image-wise mIoU and its 20,000-draw specimen-group bootstrap interval. The v6 overall adoption gate additionally requires the candidate's paired image-mean mIoU point delta versus that initialization parent to be `>=0.0%p`. The confidence interval is reported, not separately thresholded. This prevents accepting a positive treatment-vs-control effect that leaves the actual fine-tuned model below its initialization parent on the same fixed OOF set.

## Inherited v5 protocol (unchanged)

- Target classes: Al3Ni and Eutectic Si native components `[1,64)` px; augmented-mask bins `[4,16)`, `[16,64)`, `[64,256)` at nominal 2× scale, excluding crop-edge components.
- Candidate adds component-normalized `-log(mean p(true class))`, equally averaged over nonempty crop×class×size-bin cells; CE+Dice and boundary loss remain. Control is compute-matched CE+Dice plus boundary continuation.
- Four fixed specimen-group folds; Model392 seed1 initialization; 8 epochs per arm/fold; item-seeded matched data order. Component weight is fixed by 8 outer-train-only gradient probes, zero optimizer steps.
- OOF: 70 train images/59 groups; official image-wise mean over four class IoUs, empty-union=0; paired group bootstrap 20,000 draws. Valid/test remain unread; test sealed.
- Existing gates remain: candidate-vs-control `>=+0.20%p`, paired CI lower `>0`, positive folds `>=3/4`, worst fold `>=−0.30%p`; Al3Ni/Eutectic class deltas nonnegative, other classes `>=−0.50%p`; Step4 `>=−0.30%p`; both size-bin hit deltas nonnegative with CI lower `>=−5%p` for both target classes. v6 adds the parent point non-regression gate above.
- Seed1 gate passage is not final adoption; seed2 must reproduce the locked gate before considering official validation.

The v6 source receipt supersedes v5 before calibration/training. Its P100 queue waits for Model403's existing lock.
