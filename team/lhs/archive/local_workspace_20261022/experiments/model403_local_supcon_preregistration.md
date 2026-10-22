# Model403 — class-balanced local supervised contrastive auxiliary loss

**Status (2026-10-05 11:07 UTC):** Model403 was stopped after completing fold3 epoch 22/70 because its preregistered adoption gate had become mathematically unreachable: completed fold1 and fold2 paired mIoU deltas were both negative, while the gate requires at least 3/4 positive folds. Folds 1–2 each completed 70 epochs and all 18 held-out predictions (36/70 images total); fold3 OOF inference and fold4 were not run. This is not a full OOF score and is not an official-valid result. The fold3 `last.pt` and `train_state.pt` checkpoints were preserved at epoch 22. No valid/test data was used. The queued Model404 v6 began outer-train-only gradient calibration at 11:07:39 UTC after Model403 released the lock; calibration is not Model404 optimizer training. Claude Code review remains unavailable until the weekly quota reset; independent review is recorded in [model403_independent_review.md](model403_independent_review.md).

**Audit note:** The implementation-history paragraph immediately below is a pre-launch record. Its pooled 0.0062 decision, Claude review gate, and “no training has occurred” statement were superseded before launch by the fold-local coefficients and independent review outcome below.

**Partial OOF observation:** fold 1's 18-image `empty=0` image-mean mIoU was 76.8158% (the evaluator also printed `empty=skip` 77.8339%). This is one held-out training fold, not the paired 70-image result or official validation; it has not been used for tuning or a stop/adopt decision.

Implementation checks completed on 2026-10-05 KST: CPU loss/gradient smoke passed; the remote trainer/scorer CLI and fold runner parse; the fixed baseline archives contain the expected 18/18/17/17 OOF images. The paired scorer was run with Model392 seed1 predictions as both candidate and reference over all 70 train images: both scored 78.9774938%, delta and group-bootstrap interval were exactly zero, and the gate correctly failed. An initial P100 attempt on device 0 stopped on memory allocation; retrying the exact batch-4, 512-pixel forward/backward smoke with only device 7 visible passed, with 6.62 GiB peak allocated. An initial seed-403 fold-1 probe at auxiliary weight 0.05 found gradient ratios 7.1119–7.9887 (median 7.7131), suggesting weight 0.0065. Before training, projection-head initialization was isolated with `torch.random.fork_rng`; CPU and P100 assertions confirmed it leaves both global RNG streams unchanged while producing the same initialized head, preserving Model392's data-loader shuffle/worker RNG stream. With actual training seed 1 and probe weight 0.0065, four train-only batches per outer fold gave pooled median decoder-gradient ratio 1.050471 (fold medians 1.0840/1.0459/1.0988/1.0392; 16-batch range 0.9195–1.2113). The unit-median coefficient is 0.0061877, so the fixed training weight was amended to 0.0062 before any optimizer step. A direct four-fold check at the exact final weight 0.0062 gave pooled median ratio 0.994654 (fold medians 0.9839/0.9836/1.0658/0.9706; range 0.7903–1.2027). This is loss-scale calibration only, not mIoU evidence. Every probe records zero optimizer steps and no valid/test reads. Final-weight seed-1 outputs: [fold 1](results/model403_supcon_gradient_calibration_final_fold1.json), [fold 2](results/model403_supcon_gradient_calibration_final_fold2.json), [fold 3](results/model403_supcon_gradient_calibration_final_fold3.json), [fold 4](results/model403_supcon_gradient_calibration_final_fold4.json). The seed-1 0.0065 derivation outputs are [fold 1](results/gradient_calibration_seed1_fold1.json), [fold 2](results/gradient_calibration_seed1_fold2.json), [fold 3](results/gradient_calibration_seed1_fold3.json), [fold 4](results/gradient_calibration_seed1_fold4.json). Earlier seed-403 checks remain archived: [fold 1](results/model403_supcon_gradient_calibration_fold1.json), [fold 2](results/model403_supcon_gradient_calibration_fold2.json), [fold 3](results/model403_supcon_gradient_calibration_fold3.json), [fold 4](results/model403_supcon_gradient_calibration_fold4.json). The original calibration code/result are [script](model403_supcon_gradient_calibration.py) and [JSON](results/model403_supcon_gradient_calibration.json). Claude must review the intervention and this scaling rationale before training. No Model403 optimizer step/training has occurred. No official-valid or test data was read.

## Independent review outcome

Codex reviewed this experiment independently at the user's direction. The trainer follows the Model392 HRNet-W48/U-Net++ architecture, width-normalized patch sampling, CE+Dice and boundary losses, optimizer/scheduler, checkpoint contract, and inference path. The trainer has no valid/test loader. OOF inference uses `valid` only for each held-out training-group directory; scoring reads the 70 training masks and fixed OOF probability archives. The scorer implements image-wise mean of the four class IoUs with empty-union=0 and passed an exact 70-image self-comparison against Model392 (zero delta).

Review found one OOF-control concern: the originally pooled gradient-scale coefficient was derived from training-only probes across all four folds, so each fold's held-out masks indirectly contributed through the other folds' probes. Before any optimizer step, this was amended to a fold-local coefficient, derived only from that fold's outer-train probe. The four locked coefficients are 0.0059964661, 0.0062145050, 0.0059155784, and 0.0062546380; source JSON hashes and the exact formula are in [the fold-weight lock](results/model403_fold_specific_weight_lock.json). This is a leakage-control amendment, not a performance-selected change. **Decision: GO** for this exact, fold-local treatment.

The intervention remains exploratory: pulling same-class embeddings together may suppress real within-class morphology/domain variation, especially for Eutectic Si. Four folds and one seed limit precision. The preregistered paired group-bootstrap, per-fold, per-class, and Step4 gates remain unchanged; OOF evidence is not an official-valid score or proof of reaching 80%.

## Rationale and hypothesis

Model402's train-only audit found that small Eutectic-Si components are often missed, while earlier boundary losses did not produce a robust cross-group gain. The next experiment changes decoder feature geometry rather than adding another boundary target: a lightweight projection head receives a supervised local contrastive loss, with an equal cap on sampled pixels per present class and per patch. The segmentation architecture, CE+Dice objective, boundary auxiliary head, augmentation, and width-normalized sampling remain Model392's recipe. This is a hypothesis, not an expected gain or a direct reproduction of PSCL.

Important limitation from the metallurgy-specific PSCL paper: its method is a two-stage pretraining/fine-tuning design with image- and multiscale patch-level objectives plus feature-similarity-based supervised-assisted sampling; its ablation reports a large drop when the sampling strategy is removed. Model403 is only a low-cost, fully supervised online contrastive auxiliary loss and omits those components. The paper motivates investigation but does not validate this treatment or imply transferable gains ([paper](https://www.nature.com/articles/s41598-025-32855-5), [authors' code](https://github.com/neulmc/PSCL)). Independent review judged this simplified treatment worth one gated OOF experiment, while flagging the risk that global same-class positives suppress within-class morphology.

The four-fold specimen-group split and the single seed are fixed to the existing Model397 OOF protocol. The control is the already-recorded Model392 seed-1 OOF prediction for each same held-out image. Only the 70 training images/masks are permitted; no validation/test images are read.

## Fixed treatment

- U-Net++ / HRNet-W48 ImageNet initialization; input patch 512, input scale 2, width normalization; 12 samples/image; batch 4 with accumulation 2; 70 epochs; AdamW `3e-4`, weight decay `1e-4`; boundary-head weight 1.0; seed 1.
- Projection head: decoder channels → 128 → 64, 1×1 convolutions. Nearest-neighbour downsample masks to the decoder grid; sample at most 16 positions/class/image; normalized embeddings; temperature 0.1. Each fold uses its own coefficient from the fold-specific lock, computed as `0.0065 / that fold's outer-train median gradient ratio` (folds 1–4: 0.0059964661, 0.0062145050, 0.0059155784, 0.0062546380). This calibrates loss scale without held-out/valid/test masks or mIoU. Projector initialization uses a forked RNG so baseline data-loader shuffle/worker RNG streams are preserved.
- The projection head is training-only. The saved inference checkpoint contains the standard Model392 segmentation network and preprocessing metadata.
- Exactly one recipe is evaluated. No loss-weight sweep, route search, threshold tuning, or validation selection.

## Evaluation and decision gate

Score exactly-once held-out train images using image-wise mean of the four class IoUs and empty-union=0. Compare paired image deltas and specimen-group bootstrap intervals against the same Model392 seed-1 fold predictions; report per-fold, per-class, and Step4 deltas.

Adopt only if the overall gain is at least +0.20 percentage points, the paired specimen-group 95% CI lower bound is above 0, at least 3/4 folds improve, Eutectic-Si does not regress, and Step4 does not regress by more than 0.30 points. Otherwise stop this intervention without tuning on valid. Any missed runtime/fold inventory or non-finite loss invalidates the run.

## Conditional final fit for the 80% validation objective

Only if the train-only OOF gate passes, fit this same locked training procedure once on all 70 train images for 70 epochs, seed 1. Use the median of the four fold-local coefficients, `0.006105485549762077`, fixed now from the fold-weight lock; do not derive it from valid performance. The trainer must remain train-only. After the checkpoint is complete, evaluate official `valid` exactly once with patch 512, overlap 170, scale 2, and flip4 TTA. Report `image_miou_empty_zero` (four-class image mean, empty union = 0), verify all 20 valid images were scored, and make no threshold, route, calibration, or checkpoint selection from that result. If it is below 80%, record the miss and continue with new train-only experiments; do not tune this recipe against the same valid set. Do not read `test.zip` as part of this experiment.

## Archived optional Claude Code review prompt (not a run gate)

Submit once the Claude Code quota resets; do not ask about GPU occupancy.

```text
Independently review the preregistered Model403 intervention before any GPU
training. Official metric is image-wise mean of four class IoUs, empty union=0.
The current raw valid leader is Model364 at 79.943% on a repeatedly reused
20-image valid set; do not recommend any valid/test tuning. Model397 fixed
grouped train-OOF blend is 79.986% on 70 train images / 59 specimen groups, not
an official valid score. Model402 train-OOF diagnostics identify small
Eutectic-Si component misses; prior clDice, hard-crop, generic boundary loss,
directional loss, hierarchical heads, region correctors, DINOv2 feature branch,
and mean-teacher rotation consistency did not provide an adoptable robust gain.

Model403 keeps the Model392 HRNet-W48/U-Net++ recipe, seed1, fixed four-fold
specimen-group splits, and 70-epoch schedule; it adds only a decoder projection
head plus class-balanced pixel supervised-contrastive loss (weight 0.0062,
temperature 0.1, up to 16 sampled positions/class/image). The initial 0.05
weight was rejected before training because a seed-403 outer-train-only probe
measured a median auxiliary/main decoder-gradient norm ratio of 7.7131. A
seed-1, four-fold train-only probe at weight 0.0065 measured pooled median
ratio 1.050471; the unit-median coefficient is 0.0061877, rounded to fixed
weight 0.0062. Directly probing that final coefficient gave pooled median
ratio 0.994654 across all four folds. This is loss-scale
calibration, not an mIoU-selected value. Projector initialization is RNG-isolated
so baseline DataLoader order is preserved. It is an online
supervised auxiliary objective, NOT a reproduction of PSCL. The metallography
PSCL paper uses two-stage pretraining/fine-tuning, image and multiscale patch
losses, and feature-similarity-assisted sampling; the paper reports sampling
strategy as especially important. KoMaP has no separate unlabeled training
pool. The Model403 scorer was validated by a 70-image Model392-seed1
self-comparison (delta exactly zero), but no Model403 optimizer step has run.

Return GO or NO-GO for this exact treatment before training. If NO-GO, propose
at most one materially better, testable loss/feature intervention that uses
only outer-train-fold data and does not repeat the listed failed experiments.
Do not claim expected score gains. Check whether the class-balanced sampler,
positive/negative definition, projection-head gradient scale, and chosen loss
weight are methodologically defensible. Do not request or use valid/test,
route/calibration search, or GPU occupancy information. If GO, give only any
necessary non-GPU statistical pass/futility conditions; group bootstrap, fold
deltas, per-class and Step4 safeguards are already preregistered.
```
