# Al3Ni–Eutectic phase-pair component consistency probe

> 한국어 안내: 이 파일은 평가 전 고정한 실험 protocol 원문을 보존하기 위해 영어로 유지합니다. 분석 결과와 현재 판정은 [모델 결과 기록](model_results.md)을 참고하십시오.

Status: no-go after the preregistered grouped-OOF feasibility check. No official
validation labels or predictions were used.

## Motivation and gate

Model367 found that 62.39% of Model329 Eutectic false-positive pixels lie
outside the Eutectic ground-truth 2 px boundary band; Al3Ni is the dominant
confused class. Earlier Model151 removed small Al3Ni/Eutectic components and
did not help, and Model370 tested small-component Eutectic cleanup. This probe
tests a different hypothesis: a predicted Al3Ni/Eutectic region may be
internally phase-pure enough that its mean class evidence is more reliable
than individual pixel argmaxes.

Before looking at predictions under this rule, a label-only feasibility check
on the 70 grouped-OOF training masks found 84,583 8-connected components in the
Al3Ni∪Eutectic union. Components with at least 70% of their pixels belonging
to one of the two phases cover 88.24% of all pixels in that union. This passes
the predeclared structural screen (>70% coverage); it does not establish that
predicted components are similarly pure or that postprocessing improves mIoU.

## Frozen rule

For each predicted segmentation, form 8-connected components of pixels whose
argmax is Al3Ni or Eutectic Si. For each component, average the model's two
probability channels over its pixels and relabel the entire component to the
larger-mean class. Exact ties go to Al3Ni (class index 1). Other classes and
pixels are unchanged. No component-size cutoff, confidence threshold, class
prior, or coefficient is searched.

## Evaluation and decision

- Use only exactly-once grouped train-OOF predictions and their train masks;
  never read official-validation labels for this analysis.
- Model329 is the initial CPU feasibility check. Model374 and its fixed 50:50
  blend with Model329 can be assessed only after the corresponding grouped OOF
  maps exist.
- Fold 1 is exploratory. The directional replication set is folds 2–4, with
  paired specimen-group bootstrap intervals.
- First compute a GT-informed component-majority diagnostic on folds 2–4. This
  is not an upper bound for arbitrary corrections: it forces each predicted
  component to one of two phases and leaves other errors untouched. If its gain
  is below +0.5 percentage points, stop without promoting the rule.
- Keep the deployable rule only if its folds 2–4 pooled image-wise mIoU delta
  (four classes, empty union = 0) has bootstrap 95% CI lower bound above zero,
  pooled gain at least +0.3 pp, every individual fold delta is positive, and
  Eutectic Si class IoU does not decrease. Otherwise discard it.

Any score on folds used to motivate this fixed rule remains exploratory; this
probe is not a nested-CV estimate. The predeclared folds 2–4 result is the
directional check, not proof of independent generalization.

## Result: stop (2026-10-01)

The GT-informed diagnostic was applied to Model329's exactly-once grouped OOF
maps for folds 2–4 only (52 images, 44 filename-proxy groups). It assigns each
predicted Al3Ni∪Eutectic connected component to the GT-majority phase within
that component, so it is strictly diagnostic and cannot be deployed.

| Folds 2–4, image-wise mIoU (empty=0) | Score |
|---|---:|
| Model329 OOF | 79.8809% |
| GT-informed component-majority diagnostic | 72.4695% |
| Paired delta | −7.4114 pp |

The paired group-bootstrap 95% CI is `[−8.7642, −6.0835] pp`; only 2/52 images
improved. Fold deltas were −7.5672 pp (fold 2), −7.8675 pp (fold 3), and
−6.7902 pp (fold 4). Mean per-image Al3Ni IoU fell from 80.988% to 75.394%,
and Eutectic Si from 57.371% to 33.319%; Al and Primary Si were unchanged.
The structural purity screen on GT components did not transfer to predicted
components: forcing a predicted region to one phase destroys useful local
phase variation. The +0.5 pp feasibility gate failed decisively, so the frozen
probability-voting rule was not run or promoted. This rejects whole-component
phase homogenization, not boundary-aware or pixel-level phase correction.

Reproduction: `experiments/model374_phase_pair_component_audit.py`, using
Model329 fold-specific probability archives and the corresponding grouped
training masks only. The audit records archive SHA-256 values in its JSON.
