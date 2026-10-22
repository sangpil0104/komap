# Model403 independent review

Review date: 2026-10-05. This is a post-launch code/method review, not an amendment to the locked Model403 recipe or its OOF decision gate.

## Current run evidence

- The live OOF runner is `experiments/run_model403_local_supcon_oof.sh` (current SSH session is tracked by the workspace; it was restarted on 2026-10-05 after verifying a saved checkpoint).
- Fold 1 completed 70/70 epochs. Fold 2 had reached epoch 53 at 08:40 UTC; at 09:24 UTC no trainer/runner process or held-out fold-2 archive existed. The persisted `train_state.pt` was verified at epoch 53, the OOF lock was available, and physical GPU7 was idle. The same runner/recipe was relaunched with `nohup` at 09:28 UTC and the trainer was confirmed live at 09:29 UTC, resuming as `cuda:0` under `CUDA_VISIBLE_DEVICES=7`. The reason for the earlier stop is unknown.
- Fold 2 has not yet completed after the restart. Only fold 1's held-out probability archive exists so far. There is no aggregate Model403 OOF score yet.
- The 76.82% reported for fold 1 is not the official validation score and is not a full-OOF result. Do not use it to tune this run.

## Independent method review

The current function `supervised_local_contrastive` samples up to 16 decoder-grid positions per present class per crop, then concatenates sampled embeddings across the minibatch. Every other sampled embedding with the same class ID becomes a positive; the loss does not constrain positives by spatial distance, connected component, image identity, feature similarity, or prediction uncertainty. “Local” therefore describes the sampled decoder-grid pixels, not a local-neighborhood positive-pair rule.

This is a plausible but material risk for metallography: two regions with the same phase label can have different morphology/texture, and all-pairs class attraction may suppress that within-class structure. The fixed coefficient was calibrated by gradient magnitude, which controls relative optimization scale but does not establish that the objective improves mIoU. These are hypotheses to check against the completed grouped OOF result, not reasons to interrupt or alter the preregistered run.

Relevant primary literature supports treating sample construction as a design variable: *Exploring Cross-Image Pixel Contrast for Semantic Segmentation* uses semi-hard sample selection and segmentation-aware hard anchors ([ICCV 2021 paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Wang_Exploring_Cross-Image_Pixel_Contrast_for_Semantic_Segmentation_ICCV_2021_paper.pdf)); *Weakly Supervised Semantic Segmentation by Pixel-to-Prototype Contrast* describes hard-pixel and semi-hard prototype sampling ([CVPR 2022 paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Du_Weakly_Supervised_Semantic_Segmentation_by_Pixel-to-Prototype_Contrast_CVPR_2022_paper.pdf)). Their settings differ from fully supervised KoMaP, so this motivates a candidate, not an expected gain.

## Metric audit

On 2026-10-05 I inspected the scorer's underlying implementation. It forms per-image confusion matrices after probability argmax; class IoU uses zero where union is zero. It then averages the four class IoUs within each image and averages images, matching the official image-wise mIoU with `empty=0`. Candidate/reference deltas are paired by image; the 20,000-draw bootstrap resamples whole specimen groups within each fixed fold. The Model403 scorer also enforces the 70-image/59-group inventory and the baseline self-comparison test was exactly zero. This confirms the gate's metric implementation; no Model403 aggregate result exists yet.

## Independent assessment and next candidate

The local OOF evidence points to a more specific failure than generic feature separation. Model397's fixed 70-image/59-group OOF blend is 79.9859% (development OOF, not official valid); Eutectic-Si image-mean IoU is 59.00%. In its component audit, 89.83% of 1–64 native-pixel Eutectic-Si components and 99.19% of 64–256-pixel components contain at least one false negative; their perfect-correction image-mIoU oracle deltas are +2.622 and +2.138 pp, respectively. Those are diagnostic upper bounds, not expected gains. Separately, Al3Ni and Primary-Si false negatives cluster near GT boundaries. This argues for different error mechanisms rather than one global calibration change.

Model403's code-level limitation is now explicit: `supervised_local_contrastive` draws up to 16 decoder-grid pixels per present class per crop, then makes every sampled same-class pair a positive. It does not sample or balance connected components, hard pixels, image identity, spatial neighborhoods, or feature-similar positives. Consequently it can pull distinct same-phase morphologies together, and its class cap does not guarantee representation of the many small/medium Eutectic-Si components implicated by the audit. Nearest-neighbor label reduction to the decoder grid also means this objective is not a direct small-component recall loss. This is a risk assessment, not evidence that Model403 will fail; keep its locked OOF run unchanged.

Two prior results constrain the follow-up: Model311's Eutectic hard-crop experiment was essentially neutral/slightly negative on its paired OOF comparison (`−0.0066 pp`, 95% group-bootstrap CI `[-0.1421,+0.1220]`), and Model395's component-anchor sampler passed its anchor-count check but failed the pretraining crop-pixel exposure gate. Do not simply repeat either crop-sampling idea.

If Model403 fails its fixed gate, the strongest distinct candidate is one train-only, full-resolution component-normalized recall term for Al3Ni/Eutectic-Si GT regions, defined on the augmented 512-pixel masks and segmentation logits rather than changing crop selection. Before any GPU run, a CPU-only preflight should report per-fold/class/area-bin component counts, the realized loss mass per component and specimen group, and how crop truncation/augmentation changes each component; cap weights and require minimum support using outer-train data only. Freeze one coefficient from gradient-scale probes that include the complete CE+Dice+boundary objective, then compare once against the same-fold Model392 baseline with paired group bootstrap and the existing class/fold/Step4 safeguards. This candidate is not yet preregistered and must not be launched from partial Model403 scores.

No score gain is claimed. A read-only Claude Code review was attempted after confirming the CLI's `claude.ai` login, but the CLI returned its weekly-limit message before producing a review. This is my own source-and-results review, not a Claude opinion.
