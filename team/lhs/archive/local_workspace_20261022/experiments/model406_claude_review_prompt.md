# Claude Code 독립 검토 요청 — Model406 v3 작은 성분 교정

Claude Code 주간 한도가 초기화되면 결과 lock이 완성된 뒤 한 번만 제출한다. 초기화 전 quota 우회·재시도는 하지 않는다. 읽기 전용으로 검토하고 파일 수정, 학습 실행, 공식 validation/test 접근, GPU 점유 관련 질문은 금지한다.

```text
Act as an independent semantic-segmentation researcher. Review the completed,
locked Model406 v3 paired grouped-OOF experiment in this repository. Read only
the train-only OOF artifacts and the listed source/protocol files. Do not edit
files, run training, inspect official validation/test images or masks, or tune
thresholds, blend weights, or routing. Do not ask about GPU utilization.

Task metric: for each image, average IoU over the four classes, assign IoU=0
when a class union is empty, then average images. The repeatedly reused
20-image official validation leader is selection-conditioned and must not be
looked at or tuned further; test remains sealed. Treat grouped train-OOF as
development evidence, not as a fresh independent estimate.

Why this experiment exists:
- Model404 v6 improved native 1–63 px GT-component hit rate by +1.822 pp for
  Al3Ni and +1.715 pp for Eutectic Si, but zero-overlap tiny predicted
  components increased by 30.9 and 24.6 per image, respectively. Its
  compute-matched image-wise OOF gain was only +0.0319 pp and failed the
  preregistered +0.20 pp adoption gate.
- A fixed Model405 branch replacement had −0.0158 pp OOF change and was NO-GO.
- Model406 candidate adds the calibrated Model404 v4 balanced recall term for
  Al3Ni/Eutectic-Si GT components and a top-0.1% target-class hard-negative
  probability penalty outside a one-pixel same-class GT halo. The negative
  term selects pixels, not connected components; do not assume it reduced
  false islands without checking the locked OOF diagnostic.

Interpret the loss size bins carefully. Its `[4,16), [16,64), [64,256)` areas
are measured after resize/crop/augmentation, not in native mask pixels. A
train-mask-only geometric audit at target width 1278 estimated that the
combined `[4,256)` bin maps to native `[0.279,17.85)` px for 675-px-wide
images, capturing about 81.6% of native 1–63 px Al3Ni components and 64.3% of
Eutectic-Si components; for 1440-px images it maps to `[1.27,81.25)` and
includes some native 64–255 px components. These are pre-crop/pre-augmentation
approximations, not actual optimizer exposure. The current v3 recipe is fixed;
assess this only as an interpretation/follow-up issue, not grounds to alter
the locked run without a demonstrated defect.

The paired arms start from the same fixed Model392 seed-1 checkpoint per fold
and continue for eight epochs with identical architecture, per-item seeded
data order, augmentation, optimizer, schedule, and inference. The control is
CE+Dice+boundary only. Candidate auxiliary coefficients were calibrated on
outer-train crops only, with zero optimizer steps. The 4-fold split holds out
specimen groups; exact settings and artifact hashes are in the v3 OOF lock.
The chained v1/v2/v3 source locks also include the Model392 parent checkpoints
and parent OOF probability maps used by the scorer; verify this provenance
chain before assessing the parent-regression gate.

Read these authoritative artifacts:
- experiments/model406_component_hardnegative_preregistration_v3.md
- experiments/model406_component_hardnegative_loss.py
- experiments/model406_component_hardnegative_train.py
- experiments/model406_component_hardnegative_v3_lock.py
- experiments/score_model406_component_hardnegative_oof.py
- experiments/score_model406_component_hardnegative_oof_v3.py
- experiments/model406_tiny_component_confusion_audit.py
- the locked Model406 v3 OOF result JSON and its claim/lock named by the
  runner (inspect only after verifying the complete OOF lock)
- model_results.md entries for Models 402–406 and related component studies

The live experiment and its result/lock JSON files are on P100 under
`/workspace/komap_model1`; they are not necessarily present in this local
workspace. Before invoking this prompt locally, after the experiment and
secondary audit finish, copy only the result JSON, one-shot claim, source/OOF
lock JSON, and secondary component-confusion summary JSON into the matching
local `experiments/results/` paths. Verify each copied file's SHA-256 against
the P100 copy. Do not copy train images/masks or full probability archives for
this review unless a specific audit question cannot be answered from the
locked summaries; never transfer official validation/test artifacts.

Return concise, evidence-grounded answers:
1. Is there a critical mechanistic, leakage, split, or scoring defect that
   invalidates the completed comparison?
2. Does the observed candidate-control result pass every preregistered gate?
   Separate official image-wise mIoU, class/Step4 guards, tiny GT-component
   recall, unmatched tiny predicted-component count, and pixel FP burden.
3. If NO-GO, recommend at most one distinct next experiment that addresses
   the measured residual error without changing or rescuing this run. State a
   concrete grouped-OOF comparison and stop/adoption criteria. Do not forecast
   an mIoU gain.

The preregistered gate requires at least +0.20 pp image-wise mIoU, paired
specimen-group bootstrap 95% CI lower bound >0, at least 3/4 positive folds,
worst fold >=−0.30 pp, inherited class/Step4/component-hit/parent safeguards,
and no increase in zero-overlap 1–63 native-pixel predicted components for
either target class. A pass authorizes only an independently locked seed-2
grouped-OOF confirmation, not an official-valid look. Distinguish connected
components from physical particles: raster fragments and 8-connectivity do
not prove metallurgical instance identity. Cite primary literature only when
it directly supports a methodological point, and label transfer to KoMaP as
a hypothesis rather than evidence.

One domain-adjacent follow-up lead, only if the locked diagnosis shows
residual class confusion rather than predominantly isolated false-positive
islands: Xie et al., “Hard example learning based on neural collapse for
class-imbalanced semantic segmentation,” Applied Soft Computing 171 (2025),
112755 (DOI 10.1016/j.asoc.2025.112755). It uses an ETF classifier and
prototype-similarity-weighted hard pixels on 24 grayscale steel micrographs
with four phases. This is not a component loss, uses another alloy/modality
and dataset size, and is not direct evidence for KoMaP. Model403 global
supervised-contrastive learning was NO-GO; assess whether this is genuinely a
distinct and worthwhile representation/hard-example mechanism, and recommend
it only if it better matches the locked residuals than the component-level FP
or scale-aware alternative. Still recommend at most one follow-on experiment.
```
