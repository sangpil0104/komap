"""Read-only analysis of completed accuracy_v1 runs; writes beside this file.

Run with numpy, pandas, pillow, matplotlib available. No model inference/training.
Existing metrics, predictions, model code and checkpoints are never rewritten.
"""
from __future__ import annotations

import base64
import hashlib
import html
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/komap-analysis-mpl")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

HERE = Path(__file__).resolve().parent
OUT = HERE.parent
ROOT = OUT.parent
BASE = ROOT / "baseline"
CLASS_NAMES = ["Primary Si", "Al3Ni", "Eutectic Si", "Al"]
FOCAL = "A71_step2_sample1"
inputs = set()


def read_csv(path):
    path = Path(path)
    inputs.add(path)
    return pd.read_csv(path)


def read_json(path):
    path = Path(path)
    inputs.add(path)
    return json.loads(path.read_text())


def write_json(name, value):
    (HERE / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))


comparison = read_csv(OUT / "reports/comparison.csv")
manifest = read_csv(BASE / "data/manifest.csv").set_index("image_id")
frames, paths = {}, {}
for row in comparison.itertuples():
    name = row.experiment.replace(" (previous review)", "")
    paths[name] = Path(row.source)
    frames[name] = read_csv(row.source).set_index("image_id").sort_index()
baseline = frames["baseline / D4"]
ids = baseline.index
assert len(ids) == 20 and ids.is_unique
groups = manifest.loc[ids].group_id.to_numpy()
unique_groups = np.unique(groups)
rng = np.random.default_rng(2026)
draws = rng.integers(0, len(unique_groups), size=(50000, len(unique_groups)))
counts = np.array([(groups == g).sum() for g in unique_groups])
statistics, image_records, details = [], [], {}
baseline_cms = np.stack([json.loads(x) for x in baseline.confusion_matrix])

for name, original_frame in frames.items():
    assert set(original_frame.index) == set(ids)
    frame = original_frame.loc[ids]
    cms = np.stack([json.loads(x) for x in frame.confusion_matrix])
    assert np.array_equal(cms.sum(axis=2), baseline_cms.sum(axis=2))
    intersection = np.diagonal(cms, axis1=1, axis2=2)
    gt = cms.sum(axis=2)
    union = gt + cms.sum(axis=1) - intersection
    ious = np.divide(intersection, union, out=np.full(intersection.shape, np.nan), where=union > 0) * 100
    calculated = np.nanmean(ious, axis=1)
    assert np.allclose(calculated, frame.mIoU_exclude.to_numpy(), atol=1e-10)
    # Diagnostic only: fixed GT support, so absent-class false positives are not penalized here.
    gt_present = np.nanmean(np.where(gt > 0, ious, np.nan), axis=1)
    pooled = cms.sum(axis=0)
    pooled_iou = np.diag(pooled) / (pooled.sum(axis=0) + pooled.sum(axis=1) - np.diag(pooled)) * 100
    delta = frame.mIoU_exclude - baseline.mIoU_exclude
    group_sums = np.array([delta.to_numpy()[groups == g].sum() for g in unique_groups])
    bootstrap = group_sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
    ci = np.quantile(bootstrap, [0.025, 0.975])
    record = dict(experiment=name, miou_exclude=float(calculated.mean()),
                  delta_vs_baseline_d4_pp=float(delta.mean()),
                  improved_images=int((delta > 1e-8).sum()),
                  worsened_images=int((delta < -1e-8).sum()),
                  median_delta_pp=float(delta.median()),
                  bootstrap_low_pp=float(ci[0]), bootstrap_high_pp=float(ci[1]),
                  delta_without_A71_pp=float(delta.drop(FOCAL).mean()),
                  miou_without_A71=float(frame.drop(FOCAL).mIoU_exclude.mean()),
                  gt_present_diagnostic=float(gt_present.mean()),
                  pooled_miou=float(pooled_iou.mean()),
                  miou_empty_one=float(frame.mIoU_one.mean()),
                  miou_empty_zero=float(frame.mIoU_zero.mean()))
    for c in range(4):
        record[f"class_{c}_iou"] = float(np.nanmean(ious[:, c]))
        record[f"class_{c}_contributing_images"] = int((union[:, c] > 0).sum())
        record[f"class_{c}_iou_common19"] = float(frame.drop(FOCAL)[f"IoU_{c}"].mean())
    statistics.append(record)
    for i, image_id in enumerate(ids):
        image_records.append(dict(experiment=name, image_id=image_id, group_id=groups[i],
                                  miou=float(calculated[i]), delta_pp=float(delta.iloc[i]),
                                  gt_present_diagnostic=float(gt_present[i]),
                                  primary_si_gt_pixels=int(gt[i, 0]),
                                  primary_si_pred_pixels=int(cms[i].sum(axis=0)[0])))
    details[name] = dict(pooled_confusion=pooled.tolist(),
                         pooled_recall_percent=(pooled / pooled.sum(axis=1)[:, None] * 100).tolist(),
                         absent_union=[dict(image_id=str(ids[i]), class_id=int(c))
                                       for i, c in zip(*np.where(union == 0))])

stats = pd.DataFrame(statistics).set_index("experiment").sort_values("miou_exclude", ascending=False)
stats.to_csv(HERE / "metric_analysis.csv")
pd.DataFrame(image_records).to_csv(HERE / "image_analysis.csv", index=False)

training, curves, d4_effects = [], {}, []
for name in ["baseline", "fullframe", "boundary", "seed2027", "seed2028"]:
    run = BASE / "run" if name == "baseline" else OUT / "runs" / name / "official"
    status = read_json(run / "status.json")
    assert status["state"] == "completed" and status["update"] == 4500
    inputs.add(run / "metrics.jsonl")
    log = [json.loads(x) for x in (run / "metrics.jsonl").read_text().splitlines()]
    valid = pd.DataFrame([x for x in log if "validation_miou" in x])
    train = pd.DataFrame([x for x in log if "loss" in x])
    curves[name] = (valid, train)
    best = valid.loc[valid.validation_miou.idxmax()]
    stages = valid.groupby("stage", sort=False).validation_miou.max().to_dict()
    training.append(dict(experiment=name, minutes=float(status["elapsed_seconds"] / 60),
                         best_update=int(best["update"]), best_stage=str(best["stage"]),
                         best_miou=float(best.validation_miou),
                         best_weights="raw" if best.validation_raw >= best.validation_ema else "ema",
                         final_raw=float(valid.iloc[-1].validation_raw),
                         final_ema=float(valid.iloc[-1].validation_ema), stage_best=stages))
    delta = frames[f"{name} / D4"].mIoU_exclude - frames[f"{name} / plain"].mIoU_exclude
    d4_effects.append(dict(experiment=name, delta_pp=float(delta.mean()),
                           improved_images=int((delta > 0).sum()), worsened_images=int((delta < 0).sum())))
pd.DataFrame(training).to_csv(HERE / "training_analysis.csv", index=False)
pd.DataFrame(d4_effects).to_csv(HERE / "d4_effects.csv", index=False)
overlap = sorted(set(manifest.loc[manifest.source_split == "train", "group_id"]) & set(groups))
seed_std = float(np.std([stats.loc[f"{n} / plain", "miou_exclude"]
                         for n in ["baseline", "seed2027", "seed2028"]], ddof=1))
write_json("analysis.json", dict(
    suite=read_json(OUT / "suite_status.json"), statistics=stats.reset_index().to_dict("records"),
    training=training, d4_effects=d4_effects, confusion=details,
    bootstrap=dict(resamples=50000, seed=2026, clusters=len(unique_groups),
                   unit="A-prefix filename group", weighting="image-weighted mean after cluster resampling",
                   limitations="Exploratory percentile intervals; same validation used for selection, no multiple-comparison adjustment; physical specimen independence unconfirmed."),
    official_train_valid_prefix_overlap=overlap, seed_plain_sample_sd_pp=seed_std,
    test_evaluation=read_json(OUT / "ensembles/seed_ensemble/official/test/metrics_summary.json")))

plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": "white", "savefig.facecolor": "white"})
COLORS = {"baseline": "#526579", "fullframe": "#16827c", "boundary": "#aa6c22",
          "seed2027": "#9b63a1", "seed2028": "#356adb", "seed_ensemble": "#da664f"}


def save(fig, name):
    fig.savefig(HERE / f"{name}.png", dpi=160, bbox_inches="tight")
    fig.savefig(HERE / f"{name}.svg", bbox_inches="tight")
    plt.close(fig)


d4_names = [name for name in stats.index if name.endswith(" / D4")]
fig, axes = plt.subplots(1, 2, figsize=(14, 5.4), layout="constrained")
y = np.arange(len(d4_names))
scores = stats.loc[d4_names, "miou_exclude"].to_numpy()
axes[0].barh(y, scores, color=[COLORS[n.split(" / ")[0]] for n in d4_names])
axes[0].set(yticks=y, yticklabels=d4_names, xlim=(0, 86), xlabel="Validation image-mean mIoU (%)",
            title="Completed experiments (20 images, all D4)")
axes[0].invert_yaxis()
for i, value in enumerate(scores):
    axes[0].text(value + .3, i, f"{value:.4f}", va="center", fontsize=9)
all_delta = stats.loc[d4_names, "delta_vs_baseline_d4_pp"].to_numpy()
rest_delta = stats.loc[d4_names, "delta_without_A71_pp"].to_numpy()
axes[1].barh(y - .17, all_delta, height=.32, color="#356adb", label="All 20 images")
axes[1].barh(y + .17, rest_delta, height=.32, color="#dd8844", label="Other 19 (sensitivity check)")
axes[1].set(yticks=y, yticklabels=[n.split(" / ")[0] for n in d4_names],
            xlabel="Difference vs baseline + D4 (percentage points)", title="A71 changes the ranking")
axes[1].invert_yaxis()
axes[1].axvline(0, color="#73808c", lw=.8)
axes[1].legend(loc="lower left")
save(fig, "score_comparison")

fig, axes = plt.subplots(1, 2, figsize=(13.5, 5), layout="constrained")
for name, (valid, train) in curves.items():
    axes[0].plot(valid["update"], valid.validation_miou, marker=".", lw=1.5, color=COLORS[name], label=name)
    axes[1].plot(train["update"], train.loss.rolling(10, min_periods=1).mean(), lw=1.2, color=COLORS[name], label=name)
axes[0].set(title="Checkpoint validation: max(raw, EMA), no D4", xlabel="Optimizer update", ylabel="mIoU (%)", ylim=(68, 80))
axes[1].set(title="Training loss: 10-log-record rolling mean", xlabel="Optimizer update", ylabel="Loss", ylim=(.2, 1.1))
axes[0].legend(ncol=2, fontsize=8)
axes[1].text(.02, .97, "Boundary uses an additional loss term.\nStage schedules differ; loss values are not\na direct architecture comparison.",
             transform=axes[1].transAxes, va="top", fontsize=8)
for ax in axes:
    ax.grid(alpha=.18)
save(fig, "training_curves")

case_names = ["seed2028 / D4", "fullframe / D4", "boundary / D4", "seed2027 / D4", "seed_ensemble / D4"]
image_order = baseline.mIoU_exclude.sort_values().index
matrix = np.array([(frames[n].mIoU_exclude - baseline.mIoU_exclude).loc[image_order].to_numpy() for n in case_names])
fig, ax = plt.subplots(figsize=(15, 4.7), layout="constrained")
im = ax.imshow(matrix, cmap="RdBu", vmin=-13, vmax=13, aspect="auto")
ax.set(xticks=np.arange(20), xticklabels=image_order, yticks=np.arange(5), yticklabels=case_names,
       title="Per-image change vs baseline + D4 (percentage points; all 20 images)")
plt.setp(ax.get_xticklabels(), rotation=65, ha="right", fontsize=8)
for i in range(5):
    for j in range(20):
        ax.text(j, i, f"{matrix[i,j]:+.1f}", ha="center", va="center", fontsize=8,
                color="white" if abs(matrix[i,j]) > 6 else "#253442")
fig.colorbar(im, ax=ax, shrink=.7, label="mIoU change (pp)")
save(fig, "per_image_changes")

fig, ax = plt.subplots(figsize=(11, 4), layout="constrained")
common = stats.loc[d4_names, [f"class_{c}_iou_common19" for c in range(4)]].to_numpy()
common -= stats.loc["baseline / D4", [f"class_{c}_iou_common19" for c in range(4)]].to_numpy(dtype=float)
im = ax.imshow(common, cmap="RdBu", vmin=-1.7, vmax=1.7, aspect="auto")
ax.set(xticks=np.arange(4), xticklabels=CLASS_NAMES, yticks=np.arange(len(d4_names)), yticklabels=d4_names,
       title="Class IoU change on the same 19 images with all classes present (pp)")
for i in range(len(d4_names)):
    for j in range(4):
        ax.text(j, i, f"{common[i,j]:+.3f}", ha="center", va="center", color="white" if abs(common[i,j]) > 1 else "#253442")
fig.colorbar(im, ax=ax, label="IoU change (pp)")
save(fig, "class_changes")

# Scientific prediction panels: original files are opened read-only.
case_ids = [FOCAL, "A11_step1_sample4", "A67_step1_sample1"]
fig, axes = plt.subplots(3, 4, figsize=(14, 9), layout="constrained")
for i, image_id in enumerate(case_ids):
    source_files = [Path(manifest.loc[image_id, "image_path"]), Path(manifest.loc[image_id, "mask_path"]),
                    OUT / "evaluations/baseline/official/valid/d4_1/predictions" / f"{image_id}_mask.png",
                    OUT / "evaluations/seed2028/official/valid/d4_1/predictions" / f"{image_id}_mask.png"]
    for j, path in enumerate(source_files):
        inputs.add(path)
        with Image.open(path) as image:
            axes[i, j].imshow(np.asarray(image.convert("RGB")))
        axes[i, j].axis("off")
        if i == 0:
            axes[i, j].set_title(["Input", "Ground truth", "Baseline + D4", "Seed2028 + D4"][j])
        if j == 0:
            axes[i, j].text(0, -.05, image_id, transform=axes[i, j].transAxes, fontsize=9)
        if j >= 2:
            name = "baseline / D4" if j == 2 else "seed2028 / D4"
            axes[i, j].text(0, -.05, f"mIoU: {frames[name].loc[image_id, 'mIoU_exclude']:.3f}%", transform=axes[i, j].transAxes, fontsize=9)
fig.suptitle("Error cases: green = primary Si; blue = Al3Ni; gray = eutectic Si; brown = Al", fontsize=11)
save(fig, "prediction_cases")

table_rows = []
for name in ["baseline", "fullframe", "boundary", "seed2027", "seed2028", "seed_ensemble"]:
    plain = f"{stats.loc[name+' / plain', 'miou_exclude']:.4f}" if name != "seed_ensemble" else "—"
    d4 = stats.loc[name + " / D4"]
    table_rows.append(f"| {name} | {plain} | {d4.miou_exclude:.4f} | {d4.delta_vs_baseline_d4_pp:+.4f} | {int(d4.improved_images)}/20 |")

text = """# 정확도 개선 실험 분석 — 2026-09-20

현재 설정의 최고 검증 점수는 seed2028 + D4의 **78.7858%**다. 기존 모델 + D4의 78.6865%보다 **0.0993%p** 높지만, 광범위한 성능 개선으로 판단하기에는 근거가 부족하다. 네 학습 모두 4,500 updates를 완료했고, 상태 로그의 학습 시간 합계는 **38.87분**이다. 평가·프로세스 시작 시간은 이 합계에서 제외된다.

## 1. 같은 추론 조건의 비교

수치는 공식 검증 20장의 이미지별 mIoU 평균이며, 픽셀 정확도(accuracy)가 아니다. 변화량과 개선 이미지 수의 기준은 모두 **기존 모델 + D4**다. 이전 comparison.html의 변화량은 기존 plain 모델 대비이므로 기준이 다르다.

| 실험 | 기본 mIoU (%) | D4 mIoU (%) | 기존 D4 대비 (%p) | 개선 이미지 |
|---|---:|---:|---:|---:|
""" + "\n".join(table_rows) + """

![점수 비교](score_comparison.png)

## 2. 최고 점수 해석: A71 한 장의 영향

`A71_step2_sample1`은 초정 Si 정답 픽셀이 0개다. 기존 D4 모델은 780개 픽셀을 초정 Si로 잘못 예측하여 해당 클래스 IoU가 0이고 4개 클래스를 평균한다. seed2028과 fullframe은 초정 Si 예측이 0개라 정답과 예측의 합집합이 비어 해당 클래스를 평균에서 제외한다. 이는 기존에 정해 놓은 `empty_union=exclude` 계산 규칙이며 계산 오류는 아니다.

- A71 점수: 기존 D4 **40.7059%** → seed2028 D4 **53.2620%**.
- 같은 A71의 실제 존재하는 3개 클래스만 비교하면 기존 **54.2746%** → seed2028 **53.2620%**로 낮아진다.
- 초정 Si 오검출 제거는 실제 개선이지만, 약 12.56%p의 이미지 점수 상승에는 평균 대상 변경이 크게 작용한다. 이 한 장의 전체 평균 기여는 **+0.6278%p**다.
- 나머지 19장 평균: 기존 D4 **80.6855%**, seed2028 D4 **80.1291%**로 **−0.5563%p**다. fullframe도 나머지 19장에서는 **−0.4972%p**다.
- 나머지 19장 비교는 민감도 분석이다. A71을 공식 검증에서 삭제하거나 기존 평가 규칙을 바꾸지 않았다.

초정 Si의 기존 86.3947% → seed2028 91.2817% 또한 그대로 4.89%p 개선으로 해석하면 안 된다. 기존은 A71의 0점을 포함한 20장 평균이고 새 모델은 19장 평균이다. 같은 19장으로 비교하면 **90.9418% → 91.2817%, +0.3399%p**다.

![동일 이미지 집합의 클래스 변화](class_changes.png)

## 3. 개선 효과와 실패한 후보

- **D4는 일관된 개선**: 5개 단일 모델에서 각각 +0.1818~+0.2573%p, 이미지 15~18/20장에서 개선됐다. 기존 모델에서도 +0.2136%p다. 원본 코드상 8개 변환을 추론하므로 plain보다 계산량이 늘어난다; 실제 지연시간을 이번 분석에서 새로 측정하지는 않았다.
- **fullframe**: 같은 seed의 기본 추론에서 +0.1131%p지만, D4 적용 후 기존 대비 +0.0845%p이며 A71의 영향이 크다. 학습은 11.25분으로 다른 CNN 실험의 약 9.1~9.3분보다 길다. 우선 채택할 확실한 근거는 부족하다.
- **boundary=0.05**: 기본 −0.1590%p, D4 −0.1908%p. 이번 설정은 mIoU 개선에 실패했다. 별도의 경계 정확도 지표는 계산하지 않았으므로 경계 품질 자체의 결론은 내리지 않는다.
- **3-seed 앙상블**: 기존 D4 대비 −0.1248%p, 최고 seed2028 D4 대비 −0.2241%p. 기본 모델 3개의 확률을 합치는 비용을 고려하면 현재 평가 기준으로 채택할 이득이 없다. seed2027의 낮은 점수가 원인의 일부일 가능성은 있으나, 특정 멤버의 인과 효과는 제외 실험 없이는 확정할 수 없다.
- **seed 변동**: 같은 설정의 seed 2026/2027/2028 기본 점수는 78.4729/77.7883/78.5990%, 표본 표준편차 0.4362%p다. 이번 설정 변경의 작은 점수 차이보다 크다. 3개 seed만으로 전체 변동을 정확히 추정할 수는 없다.

보조 진단인 전체 픽셀 합산 mIoU는 앙상블 81.6370%, 기존 D4 81.4989%로 순위가 달라진다. 이는 이미지를 같은 비중으로 평균하는 기본 평가 지표와 다른 값이며, 점수가 더 높다는 이유로 기준을 교체하지 않는다.

## 4. 클래스별 병목과 실패 사례

기존 D4 모델의 클래스 IoU는 초정 Si 86.3947%, Al3Ni 74.4029%, 공정 Si 59.8481%, Al 94.1002%다. **공정 Si와 Al3Ni가 우선 분석 대상**이다. seed2028 D4는 Al3Ni 72.6949%, 공정 Si 59.1609%로 오히려 낮아졌다. 동일 19장 비교에서도 Al3Ni는 −1.6367%p, 공정 Si는 −0.7282%p다.

seed2028 D4의 주요 악화 이미지는 `A67_step1_sample1` **−7.3242%p**, `A11_step1_sample4` **−3.0857%p**다. seed2027 D4는 A11에서 **−9.1085%p**다. A11에서 seed2028은 Al3Ni의 실제 면적 7.54%를 3.52%로 작게 예측하고, 공정 Si의 실제 면적 8.18%를 11.00%로 크게 예측한다. Al3Ni ↔ 공정 Si 혼동을 정답 마스크와 함께 검토할 가치가 있다.

![전체 20장의 변화](per_image_changes.png)

![입력·정답·예측 비교](prediction_cases.png)

## 5. 학습 상태와 불확실성

fullframe/boundary/seed2028의 최고 체크포인트는 update 4250의 raw 가중치다. seed2027은 update 4500의 EMA 가중치다. 마지막 raw 점수와 최고값의 차이가 작아 마지막 평가 구간에서 큰 개선은 관찰되지 않는다. 이 로그만으로 더 긴 학습이 항상 무효라고 결론 내리지는 않는다.

![학습 곡선](training_curves.png)

이미지 ID의 A-prefix 19개 그룹을 묶어 50,000회 paired bootstrap(고정 seed 2026)을 수행한 탐색적 95% 구간은 seed2028 vs 기존 D4 **[−1.2203, +1.8128]%p**, fullframe **[−0.7767, +1.4892]%p**다. 둘 다 0을 포함한다. 검증셋으로 모델을 선택한 효과와 여러 후보를 비교한 효과를 보정하지 않은 구간이라 최종 일반화 성능의 보증은 아니다.

공식 train/valid에는 A-prefix 7개(A125, A153, A181, A191, A40, A55, A7)가 겹친다. 기존 audit에도 기록된 사항이다. 접두사가 실제 동일 시편임이 확인된 것은 아니므로 데이터 누출이라고 단정하지 않는다. 새로운 시편에 대한 성능을 판단하려면 실제 시편 관계 확인 및 group CV가 필요하다. 현재 테스트 10장은 정답이 없어 정확도를 계산할 수 없으며, PNG 생성·형식 검사를 완료했다는 뜻이다.

## 6. 현재 선택과 다음 확인 순서

1. **현재 검증 규칙의 최대 점수가 목적**이면 seed2028 + D4가 후보 1위다. 기존 D4와 사실상 근소한 차이라는 점을 함께 기록한다.
2. **넓은 이미지 범위에서의 안정성 판단**을 우선하면 기존 모델 + D4를 기준 모델로 유지한다. 이 판단 역시 독립 시험에서 입증된 우열을 뜻하지 않는다.
3. 다음 학습 전에 A71의 빈 클래스 처리와 A11/A67의 Al3Ni·공정 Si 정답/예측을 검토한다. 규칙을 검증 최고점에 맞춰 사후 변경하거나 검증 이미지를 학습에 추가하지 않는다.
4. 설정을 확정할 다음 실험은 동일 group fold에서 baseline과 후보를 비교하는 것이다. 큰 구조 변경이나 더 긴 학습에 앞서 현재 0.1%p 수준 차이가 반복되는지 확인하는 편이 결과 해석에 도움이 된다.

이번 분석은 새 학습·새 테스트 추론을 실행하지 않았다. 학습된 모델과 기존 소스·결과는 보존하고, 분석 파일만 이 폴더에 추가했다.

## 산출물과 재현

- `metric_analysis.csv`: 기본·보조 지표, 같은 19장 클래스 비교, paired bootstrap 구간.
- `image_analysis.csv`: 각 이미지의 변화와 초정 Si 정답·예측 픽셀 수.
- `training_analysis.csv`, `d4_effects.csv`, `analysis.json`: 학습·D4·혼동 행렬 상세 수치.
- `input_hashes.json`: 사용한 CSV, 로그, 이미지, 상태 파일의 SHA-256.
- `*.png`, `*.svg`: 공유 가능한 그림. `report.html`은 그림을 내장하여 단독으로 열 수 있다.
- `analyze.py`: 읽기 전용 분석 스크립트. numpy/pandas/pillow/matplotlib가 있는 Python으로 실행한다. 차트 라이브러리는 이번에 `/tmp/komap-result-analysis-20260920`에만 설치했으며 komap conda 환경은 바꾸지 않았다.
"""
(HERE / "REPORT.md").write_text(text)

# Self-contained HTML with embedded figures; no network dependencies.
body = []
in_table = False
in_list = False


def inline(value):
    import re
    value = html.escape(value)
    value = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", value)
    return re.sub(r"`(.+?)`", r"<code>\1</code>", value)


for line in text.splitlines():
    if in_table and not line.startswith("|"):
        body.append("</tbody></table></div>")
        in_table = False
    is_list = line.startswith("- ") or (len(line) > 2 and line[0].isdigit() and line[1:3] == ". ")
    if in_list and not is_list:
        body.append("</ul>")
        in_list = False
    if not line:
        continue
    if line.startswith("!["):
        alt, fname = line[2:].split("](", 1)
        data = base64.b64encode((HERE / fname[:-1]).read_bytes()).decode()
        body.append(f'<figure><img alt="{html.escape(alt)}" src="data:image/png;base64,{data}"></figure>')
    elif line.startswith("|"):
        if not in_table:
            body.append('<div class="scroll"><table><thead><tr>' + "".join(f"<th>{inline(c.strip())}</th>" for c in line.strip("|").split("|")) + "</tr></thead><tbody>")
            in_table = True
        elif not set(line).issubset(set("|-: ")):
            body.append("<tr>" + "".join(f"<td>{inline(c.strip())}</td>" for c in line.strip("|").split("|")) + "</tr>")
    elif line.startswith("#"):
        level = len(line) - len(line.lstrip("#"))
        body.append(f"<h{level}>{inline(line[level:].strip())}</h{level}>")
    elif is_list:
        if not in_list:
            body.append("<ul>")
            in_list = True
        body.append(f"<li>{inline(line[2:] if line.startswith('- ') else line[3:])}</li>")
    else:
        body.append(f"<p>{inline(line)}</p>")
if in_list:
    body.append("</ul>")
(HERE / "report.html").write_text('''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>정확도 개선 실험 분석</title><style>
body{font:16px/1.8 system-ui,sans-serif;color:#203344;background:#f4f7fb;max-width:1160px;margin:40px auto;padding:0 22px}h1{font-size:30px}h2{font-size:23px;margin-top:44px;border-bottom:1px solid #d3dce7;padding-bottom:9px}strong{color:#154d86}code{background:#e6edf4;padding:2px 5px;border-radius:4px;overflow-wrap:anywhere}li{margin:9px 0}img{display:block;width:100%;height:auto}figure{margin:25px 0;background:white;border:1px solid #d3dce7;border-radius:8px;padding:8px}.scroll{overflow:auto}table{border-collapse:collapse;width:100%;background:white}th,td{text-align:right;padding:10px;border-bottom:1px solid #d3dce7;white-space:nowrap}th:first-child,td:first-child{text-align:left}@media(max-width:600px){body{font-size:15px;margin:20px auto;padding:0 14px}h1{font-size:24px}}
</style><main>''' + "\n".join(body) + "</main></html>")
write_json("input_hashes.json", {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(inputs)})
print(stats[["miou_exclude", "delta_vs_baseline_d4_pp", "delta_without_A71_pp"]].to_string())
print(f"Report: {HERE / 'report.html'}")
