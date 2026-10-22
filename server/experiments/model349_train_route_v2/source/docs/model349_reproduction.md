# Model349 training and routing bundle

This is a **code-only** bundle. A third party provides the original labeled
`train.zip` (70 image/mask pairs) and `valid.zip` (20 pairs). The pipeline
trains its own checkpoints, generates its own validation probabilities, and
rebuilds the class-wise route through Model349. It does not require a copy of
our trained checkpoints, Model183 cache, or validation probability archives.

## Run

On a Linux/CUDA machine, extract the ZIP and put the two supplied dataset
archives in one directory. From the extracted directory:

```bash
bash experiments/setup_model349_environment.sh
KOMAP_DATA_DIR=/absolute/path/to/archives bash experiments/run_model349_pipeline.sh preflight
KOMAP_DATA_DIR=/absolute/path/to/archives bash experiments/run_model349_pipeline.sh train
```

`KOMAP_DATA_DIR` must contain `train.zip` and `valid.zip`; the default is
`./datasets`. Set `KOMAP_PYTHON` to use an existing compatible virtual
environment. The setup script pins the Python packages used on our P100.
`preflight` checks the archive counts and all four grouped-CV partitions
without running GPU training.
ImageNet encoder initializations are downloaded by their libraries if not
already cached, so the training host needs access to those weights or a
prepopulated standard library cache. These are public pretrained **initial
weights**, not our competition-trained checkpoints.

The runner makes a fresh `runs/model349_train.*` directory. It never
overwrites a previous run. Within that directory, `outputs/` holds the newly
trained checkpoints, prediction archives, route caches and result JSONs;
`logs/` holds one log per stage. The final result is
`outputs/model349_model347_into_346_route/results.json` and prints the
validation image-wise four-class mIoU with empty unions counted as zero.
`KOMAP_RUN_ROOT` can set a new output directory; an existing directory is
rejected to avoid accidental overwrites. `KOMAP_WORKERS` controls CPU route
search concurrency.

## What gets trained

1. ResNet34/U-Net++ base and its ASPP lineage:
   Model6 → 62 → 97 → 98 → 109.
2. Model183's base-route sources: restart51/59/61, Model123 EMA and
   Model181 representation. The route47 probability cache and Model183
   validation entries are regenerated from these new checkpoints.
3. Native/full-stat lineages: Model109 → 194 → 195 → 239 → 242 → 257 → 269;
   Model220 TENT adapts Model195 at inference time.
4. Grouped-CV EfficientNet-B4 branches Model322, 326 and 329 (four folds,
   16,000 updates per fold), plus Model336 HRNet-W48/U-Net++ and the
   weighted-CE Model347 fine-tune.
5. CPU class-wise route searches Model335 → 338 → 339 → 341 → 346 → 349.

The included `model303_progress.json` is only the historical grouped-CV
split manifest; it contains no checkpoint or validation prediction. The
pipeline uses `valid.zip` for checkpoint selection and route selection, so
its reported score is **validation**, not an untouched test estimate.

The archived Model349 validation score is 79.9095189% using the official
image-wise, four-class, empty=0 calculation. Training from scratch is
stochastic and may select slightly different checkpoints/routes, so this
bundle should be used to reproduce the **procedure**, not to guarantee
bit-identical checkpoints or that exact number. Original Model194 attempted
a 1024px third stage that exceeded P100 memory; this runner skips that
unsuccessful stage and continues from the stage-1 best checkpoint.

Training is very compute-intensive, particularly twelve B4 folds. Run it on
an otherwise available GPU, monitor logs, and follow the department GPU
server manual. No test masks, dataset archives, old trained checkpoints or
cached validation predictions are uploaded to GitHub with this ZIP.
