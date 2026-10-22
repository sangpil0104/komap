# Model392 candidate: per-image pixel-scale normalization

This is an isolated candidate derived from the P100 Model382 HRNet-W48 +
U-Net++ source tree. It adds a train-split modal-width normalization option;
it does not alter the active Model382 source or its running process. When
training, pass `--rng-mode legacy` so the only intended change from Model382
is the scale-aware crop/tile geometry.

Model391 used only the 70 official training image/mask pairs and found
step-adjusted boundary-run slopes versus native image width of 1.217 for
Al3Ni (group-bootstrap 95% CI 0.762–1.634) and 0.838 for Eutectic Si
(0.605–1.072). However, the four width strata contain disjoint specimen
groups, so this is a between-specimen association confounded with morphology,
not within-specimen evidence of magnification. It motivates only a falsifiable
grouped-OOF experiment; it does not prove physical field of view or predict
an mIoU gain.

With `--normalize-width`, the trainer derives `target_width` from the mode of
native widths in that run's `train/images` only (ties choose the smaller
width). Every image uses effective input scale
`base_input_scale * target_width / native_width`. Its source crop is sized as
`round(patch_size / effective_input_scale)`, so equal physical proportions
map to the same model patch size. The resolved `target_width` is stored in
`args.json` and checkpoint preprocessing metadata. Evaluation automatically
uses the saved value and rejects conflicting explicit overrides. `predict.py`
infers each checkpoint's own target width, which allows safe inference-time
averaging of fold checkpoints whose training-fold modes differ.

Inference uses the same per-image effective scale and scales native overlap
in proportion to native tile size, preserving the Model382 overlap/tile
ratio. The result remains mapped to each image's original dimensions.

When width normalization is enabled, training rejects any `--val-every` value
other than 0; the legacy quick-validation forward does not honor per-image
scale and must not expose fold/official validation scores during training.

This is not authorized to train yet. The preregistered Model382 comparison,
Model387/388/389 audits and their gates must finish first. If later run, use
the locked four specimen-group folds, 70 epochs, last checkpoint only,
train-only OOF, and no official valid/test access before the adoption gate.
