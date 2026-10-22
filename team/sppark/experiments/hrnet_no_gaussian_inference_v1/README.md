# HRNet_taeho_no_gaussian: inference-only ablation

The checkpoint is fixed to the completed no-Gaussian experiment identified in
`baseline.json`. No model training, optimizer update, or checkpoint selection is performed.
Original models, their source code, and past reports remain untouched; hashes are
recorded in `preservation.json`. Python runs exclusively in conda `komap`.

The stopping deadline is **2026-09-30 01:30 KST / 2026-09-29 16:30 UTC**.
The user additionally requests stopping at 8% remaining Codex plan allowance.
There is no account-quota tool exposed to this session; that condition requires
a user notification. The goal token counter is not the account's remaining allowance.

The primary metric is the equal-weight average over official valid20 images of
their four-class IoU average. Empty union always receives IoU=0. Global mIoU is
not computed. The historical no-Gaussian report used empty IoU=1; recomputing
its saved confusion matrices with the required policy gives **78.917287950658%**.

`run.py` varies only inference and reuses identical network forward passes when
comparing probability restoration, spatial tile weights, or averaging rules.
Every arm specifies a control and its exact changed fields. The first group
has 15 arms: baseline; area/bicubic/antialiased restoration; resizing logits
before softmax; Gaussian sigma .25/.1875/.0625; uniform/Hann tile weights;
no/horizontal/vertical flip TTA; logit tile averaging; geometric TTA averaging.

`postprocess.py` evaluates fixed, label-independent spatial filters and class
decision weights on cached probabilities. All settings apply uniformly across
images. Ground-truth masks enter only the scoring/reporting path. Hyperparameters
are explored and selected on valid20, so improvements are validation findings,
not independent evidence of test-set improvement. Per-image deltas and class
IoUs should be inspected alongside the average score.

Artifacts for each completed condition are placed in a new
`results/model_results_MMDD_N` directory: summary, per-image metrics, predictions,
and 20 original-resolution 2x2 comparisons. The top-right panel shows errors in
ground-truth colors and correct pixels in white; the other panels contain the
original (top-left), prediction (bottom-left), and target (bottom-right).
Each panel header displays image mIoU and 100%−image mIoU.

`COMPARISON.md` and `comparison.json` collect completed arms only. Raw probability
caches and per-run configuration/source snapshots live under `outputs/`.
Progress and elapsed/remaining time are printed by tqdm to the terminal/logs.
No localhost dashboard is used.

Example inference command (not training):

```bash
conda run -n komap --no-capture-output python -B -u \
  experiments/hrnet_no_gaussian_inference_v1/run.py --gpu 7 --batch 4
```

Initially, one physical GPU was shared, capped at 36% of that device's memory.
The ongoing DDP training then waited for that device: observed training update
time rose from roughly .19 to .74 seconds in the sampled before/during intervals.
The study switched to independent-image inference across GPUs 0–7 so each
condition can finish in a shorter interval. The same 36% per-device memory cap
applies. `V00_parallel_control` reproduced **all 20 probability maps bit for bit**;
see `checks/parallel_equivalence.json`. No training process was stopped or changed.
The interrupted single-GPU phase retains its completed and partial outputs;
only its unfinished conditions are scheduled into a new run.

`parallel_inference.py` supports ordinary tile inference, full-image context,
the extra frozen-network transformations, and size-dependent scale policies.
`inference_queue.py` executes those jobs sequentially with the requested deadline.
The running queue was originally launched under the temporary name `queue.py`;
that file was renamed to avoid shadowing Python's standard-library queue module.
The initial plan-generation import error occurred before any affected condition
was launched and was fixed without changing model weights or old results.

`reproduce.py --id <condition> --export-only` exports the dependency graph of a
selected inference recipe. Without `--export-only`, it recomputes that recipe
from raw images and compares the result to cached exploration outputs.
The synthetic prelaunch check establishes bitwise equality with the preserved
original sliding-window implementation and verifies all eight transform inverses.

Additional reports:

- `RANKING.md`: completed-condition ranking and descriptive paired intervals.
- `DIAGNOSTICS.md`: image-size cohorts, per-image changes, and network input pixel-work proxy.
- `SELECTION_SENSITIVITY.md`: retrospective specimen-group omission and bootstrap selection sensitivity; explicitly not independent validation.
- `BOUNDARY_<ID>.md`: boundary/interior errors and small-component recall for selected conditions.
- `analysis/<UTC timestamp>/`: standalone SVG/PNG plots, with the score snapshot that generated them.
- `checks/audit_*.json`: independent score, probability-cache, panel-pixel, and preservation verification.

`queue_followup.py` uses named queue logs and terminates only its own child process
group at the requested cutoff. Successive GPU jobs remain sequential. CPU-only
analysis of saved probabilities can run alongside a GPU inference job.

`parallel_reproduce_expandable.py --id <ID>` reconstructs a recipe from raw images
across GPUs 0–7, then compares its probabilities and masks against the exploration
cache. Cached probabilities never enter the reconstruction's inference path.
For a **later user-started reproduction after the search deadline**, use:

```bash
conda run -n komap --no-capture-output python -B -u \
  experiments/hrnet_no_gaussian_inference_v1/manual_reproduce.py \
  --id GP05_sharpen --allow-after-deadline
```

`GP05_sharpen` is an intermediate completed example; use the ID selected in the
final report for the final recipe. The manual flag is never used by the active
search queues. This command performs inference and verification, not training.
