# Model349 reproduction bundle

> **The ZIP alone is not a turnkey end-to-end run.** It contains the source
> code and a command to reproduce Model349's validation route **after** the
> required data, checkpoints/probability archives, and Model183 metadata are
> supplied. It does not by itself train all component models or generate all
> eight probability sources. The code-only archive excludes competition data,
> trained weights, cached predictions, and a frozen environment.

## What this reproduces

Model349 is the current raw validation leader under the official image-wise,
four-class mIoU metric with empty-union set to zero. Its recorded strict valid
score is **79.9095189%** on the 20 labeled validation images. This bundle
contains the code and configuration needed to reproduce Model349's exact
625-route validation search from the original probability archives. It is not
a single neural-network checkpoint: Model349 is a route over eight probability
sources and Model183's per-image route metadata.

Class order used by the route is `[Al3Ni, Primary Si, Al, Eutectic Si]`:

1. Model341 is the selected Model339 route with Model329 probabilities.
2. Model346 adds Model336 HRNet-W48/U-Net++ probabilities with gamma
   `[0, 0, 1.0, 0.25]`.
3. Model349 adds Model347 weighted-CE HRNet probabilities with gamma
   `[0, 0, 0.25, 0]`.

The Model349 grouped held-out route-selection delta against Model346 was
`-0.0864076` percentage points (10/20 images; 95% group-bootstrap CI
`[-0.2355669,+0.0220142]`). Treat 79.9095% as a raw validation result, not an
independent test estimate or a proven generalization gain.

## Run the validation reproduction

Unzip this archive at the competition project root, preserving its paths. Use
the competition's authorized dataset and the matching cached probability and
route-metadata files listed below. On the recorded P100 environment:

```bash
cd /workspace/komap_model1
/workspace/venv-komap/bin/python \
  experiments/model346_model336_into_341_route.py \
  --root /workspace/komap_model1 \
  --expert-id 347 --anchor-model346 \
  --workers 4 --bootstrap 20000 \
  --output outputs/model349_reproduction_rerun.json
```

The scorer verifies that gamma-zero reproduces Model346's stored class and
mean-score baseline before evaluating the route grid. It writes the selected
gamma, raw strict valid mIoU, per-class means, and specimen-group CV/bootstrap
results to the output JSON. Compare with
`results/outputs/model349_model347_into_346_route/results.json`.

## Component-model creation code included

The archive now includes the component training/inference sources and the
recorded launchers, not just the final route scorer:

| Component family | Included code |
|---|---|
| ResNet34-derived sources (Models 195/217/269) and Model183 route metadata | `experiments/historical_scripts/` (including the original trainer, TTA/adaptation, branch, and calibration implementations) |
| EfficientNet-B4 / grouped OOF (Models 322/326/329) | `experiments/model322_team_b4/KoMaP/`, `experiments/model326_hybrid_fullstats/KoMaP/`, and matching launchers in `experiments/runners/` |
| HRNet-W48/U-Net++ (Models 336/347) | `experiments/model336_hrnet_best/` plus `experiments/runners/queue_model347_weighted_hrnet_after346.sh` |
| Route selection and scoring (Models 339/341/346/349) | `experiments/model341_model329_joint_route_search.py`, `experiments/model346_model336_into_341_route.py`, `experiments/runners/`, and the archived reference JSON files |

Those scripts can train or infer the component models when run with the
matching data, dependencies, initialization, configuration, and output paths.
Training produces the model checkpoints; inference (and, for the TENT branch,
per-image test-time adaptation) turns those checkpoints into the probability
archives consumed by the route scorer. A checkpoint listed as an input to a
continuation is the *parent model's training output*, not an unrelated
artifact—for example Model347 starts from Model336, while Model269 continues
Model257. Rebuilding a lineage therefore means running its training stages in
order and passing each stage's checkpoint forward.

The scripts are separate experiment entrypoints; the bundle does **not**
provide a single command that trains every lineage in dependency order and
routes them automatically. Some lineages are warm-start continuations, so
their parent stage must be trained first. The command above reproduces the
exact route once the listed cached probabilities and metadata are present.

## Required external inputs

The archive intentionally excludes competition images/masks, weights, cached
probability tensors, and pickled route metadata. Restore the following files
under the same project-root layout before running the command:

```text
data/valid/masks/*_mask.png
outputs/model183_metric_calibration/entries.pkl
outputs/model195_native_resume/valid_probs.npz
outputs/model220_native_tent_extended/valid_probs_06.npz
outputs/model269_fullstats_1x_extended/valid_probs.npz
outputs/model322_team_b4_group_oof/valid_ensemble/probabilities.npz
outputs/model326_hybrid_fullstats_group_oof/valid_ensemble/probabilities.npz
outputs/model329_fullstats_width4000_uniform_group_oof/valid_ensemble/probabilities.npz
outputs/model336_hrnet_best/valid_eval/valid_probabilities.npz
outputs/model347_hrnet_inverse_ce_finetune/valid_eval/valid_probabilities.npz
outputs/model339_joint_gamma_route_search/results.json
outputs/model341_model329_joint_route_search/results.json
outputs/model346_model336_into_341_route/results.json
```

To regenerate missing probability files instead of restoring them, train the
corresponding component lineage to create its checkpoints, then run its
inference entrypoint with the matching preprocessing configuration. This
requires the authorized train/valid images and masks, dependencies, and any
base/pretrained initialization required by the first stage. None of these
data or weight artifacts are embedded in this code-only archive. The P100
virtual environment used for the recorded run is `/workspace/venv-komap`;
package versions are not frozen by this bundle.

## Scope boundary

This bundle reproduces Model349's validation route search and score when given
the recorded validation artifacts. It does not claim a one-command
from-checkpoint test submission pipeline: test labels are unavailable, test
probability maps are not included, and the eight source branches must each
produce test probabilities using their matching model/inference settings.
