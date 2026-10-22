"""Read-only analysis of completed augmentation experiments; outputs stay here."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/komap-analysis-mpl")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

HERE = Path(__file__).resolve().parent
OUT = HERE.parent
ROOT = OUT.parent
BASE = ROOT.parent / "accuracy_v1/baseline"
NAMES = ["baseline", "photo_mild", "noise_mild", "scale_mild"]
COLORS = ["#64748b", "#16847b", "#d18335", "#b44957"]
FOCAL = "A71_step2_sample1"
inputs = {}


def register(path):
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            digest.update(chunk)
    inputs[str(path)] = digest.hexdigest()
    return path


def csv(path):
    return pd.read_csv(register(path))


def js(path):
    return json.loads(register(path).read_text())


def dump(name, value):
    (HERE / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))


table = csv(OUT / "reports/comparison.csv")
frames = {(r.experiment, r.inference): csv(r.source).set_index("image_id").sort_index()
          for r in table.itertuples()}
base = frames["baseline", "d4"]
assert len(base) == 20 and base.index.is_unique
manifest = csv(BASE / "data/manifest.csv").set_index("image_id")
groups = manifest.loc[base.index].group_id.to_numpy()
unique = np.unique(groups)
draws = np.random.default_rng(2026).integers(0, len(unique), size=(50000, len(unique)))
counts = np.array([(groups == g).sum() for g in unique])
base_cm = np.stack([json.loads(v) for v in base.confusion_matrix])
common = (base_cm.sum(axis=2) > 0).all(axis=1)
records, image_records, focal_records, class_records = [], [], [], []

for name in NAMES:
    for mode in ["plain", "d4"]:
        frame = frames[name, mode]
        assert frame.index.equals(base.index)
        ref = frames["baseline", mode]
        cm = np.stack([json.loads(v) for v in frame.confusion_matrix])
        assert np.array_equal(cm.sum(axis=2), base_cm.sum(axis=2))
        diag = np.diagonal(cm, axis1=1, axis2=2)
        union = cm.sum(axis=1) + cm.sum(axis=2) - diag
        iou = np.divide(diag, union, out=np.full(diag.shape, np.nan), where=union > 0) * 100
        score = np.nanmean(iou, axis=1)
        assert np.allclose(score, frame.mIoU_exclude, atol=1e-10)
        assert (union > 0).all(), "Unexpected empty-union change requires fresh interpretation"
        delta = frame.mIoU_exclude - ref.mIoU_exclude
        sums = np.array([delta.to_numpy()[groups == g].sum() for g in unique])
        boot = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
        low, high = np.quantile(boot, [.025, .975])
        pooled = cm.sum(axis=0)
        pooled_iou = np.diag(pooled) / (pooled.sum(axis=0) + pooled.sum(axis=1) - np.diag(pooled)) * 100
        records.append(dict(experiment=name, inference=mode, miou=float(score.mean()),
                            delta_pp=float(delta.mean()), improved_images=int((delta > 0).sum()),
                            worsened_images=int((delta < 0).sum()), median_delta_pp=float(delta.median()),
                            bootstrap_low_pp=float(low), bootstrap_high_pp=float(high),
                            delta_common19_pp=float(delta.to_numpy()[common].mean()),
                            pooled_miou=float(pooled_iou.mean())))
        for image_id in base.index:
            image_records.append(dict(experiment=name, inference=mode, image_id=image_id,
                                      miou=float(frame.loc[image_id, "mIoU_exclude"]), delta_pp=float(delta.loc[image_id])))
        if mode == "d4":
            for c, label in enumerate(["primary_si", "al3ni", "eutectic_si", "al"]):
                class_records.append(dict(experiment=name, class_name=label, iou=float(frame[f"IoU_{c}"].mean()),
                                          delta_pp=float((frame[f"IoU_{c}"] - ref[f"IoU_{c}"]).mean()), images=20))
            focal_cm = np.array(json.loads(frame.loc[FOCAL, "confusion_matrix"]))
            focal_records.append(dict(experiment=name, image_id=FOCAL,
                                      miou=float(frame.loc[FOCAL, "mIoU_exclude"]),
                                      delta_pp=float(delta.loc[FOCAL]), primary_si_false_positive_pixels=int(focal_cm.sum(0)[0])))

stats = pd.DataFrame(records)
stats.to_csv(HERE / "metric_analysis.csv", index=False)
pd.DataFrame(image_records).to_csv(HERE / "image_analysis.csv", index=False)
pd.DataFrame(class_records).to_csv(HERE / "class_analysis.csv", index=False)
pd.DataFrame(focal_records).to_csv(HERE / "A71_analysis.csv", index=False)

training, curves, effects = [], {}, []
base_cfg = yaml.safe_load(register(BASE / "run/resolved_config.yaml").read_text())
base_env = js(BASE / "run/environment.json")
for name in NAMES:
    run = BASE / "run" if name == "baseline" else OUT / "runs" / name / "seed2026/official"
    status = js(run / "status.json")
    assert status["state"] == "completed" and status["update"] == status["total_updates"] == 4500
    cfg = yaml.safe_load(register(run / "resolved_config.yaml").read_text())
    env = js(run / "environment.json")
    assert cfg["seed"] == base_cfg["seed"] == 2026
    for key in ["model", "train", "loss", "sampling", "evaluation", "inference"]:
        assert cfg[key] == base_cfg[key], key
    for key in ["code_sha256", "dataset_sha256", "world_size", "effective_batch", "train_ids", "validation_ids"]:
        assert env[key] == base_env[key], key
    if name != "baseline":
        checkpoint_hash = inputs[str(register(run / "best.pt"))]
        for mode in ["plain", "d4"]:
            provenance = js(OUT / "evaluations" / name / "seed2026/official" / mode / "provenance.json")
            assert provenance["checkpoint_sha256"] == checkpoint_hash
            assert provenance["frozen_code_sha256"] == env["code_sha256"]
            assert provenance["dataset_hash"] == env["dataset_sha256"]
            assert provenance["selected_weights"] == "raw"
            assert not set(provenance["train_ids"]) & set(provenance["validation_ids"])
    log = [json.loads(v) for v in register(run / "metrics.jsonl").read_text().splitlines()]
    valid = pd.DataFrame([v for v in log if "validation_miou" in v])
    curves[name] = valid
    best = valid.loc[valid.validation_miou.idxmax()]
    stage_max = valid.groupby("stage", sort=False).validation_miou.max().to_dict()
    training.append(dict(experiment=name, minutes=float(status["elapsed_seconds"] / 60),
                         best_update=int(best["update"]), best_validation=float(best.validation_miou),
                         final_raw=float(valid.iloc[-1].validation_raw),
                         final_ema=float(valid.iloc[-1].validation_ema), **stage_max))
    d4_delta = frames[name, "d4"].mIoU_exclude - frames[name, "plain"].mIoU_exclude
    effects.append(dict(experiment=name, d4_gain_pp=float(d4_delta.mean()), improved_images=int((d4_delta > 0).sum())))
pd.DataFrame(training).to_csv(HERE / "training_analysis.csv", index=False)
pd.DataFrame(effects).to_csv(HERE / "d4_effects.csv", index=False)
previous = csv(ROOT.parent / "accuracy_v1/outputs/reports/comparison.csv")
overlap = sorted(set(manifest.loc[manifest.source_split == "train", "group_id"]) & set(groups))
dump("analysis.json", dict(metrics=records, training=training, d4=effects, A71=focal_records,
    total_training_minutes=sum(r["minutes"] for r in training if r["experiment"] != "baseline"),
    previous_best=previous.sort_values("miou", ascending=False).iloc[0].to_dict(),
    bootstrap=dict(seed=2026, resamples=50000, clusters=len(unique), unit="A-prefix filename group",
                   method="paired cluster percentile; image-weighted resampled mean",
                   limitations="Exploratory: no selection/multiple-comparison correction, one training seed, physical independence not confirmed"),
    train_valid_prefix_overlap=overlap, unchanged_model_engine=base_env["code_sha256"],
    no_training_or_inference_run=True, no_metric_definition_changes=True))

plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": "white", "savefig.facecolor": "white"})


def save(fig, name):
    for ext in ["png", "svg"]:
        fig.savefig(HERE / f"{name}.{ext}", dpi=160, bbox_inches="tight")
    plt.close(fig)


fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), layout="constrained")
x = np.arange(4)
for offset, mode, color in [(-.18, "plain", "#93a7bc"), (.18, "d4", "#1c847d")]:
    values = [float(frames[n, mode].mIoU_exclude.mean()) for n in NAMES]
    axes[0].bar(x + offset, values, width=.34, label=mode, color=color)
    for i, value in enumerate(values):
        axes[0].text(i + offset, value + 1, f"{value:.3f}", ha="center", fontsize=8)
axes[0].set(xticks=x, xticklabels=NAMES, ylim=(0, 88), ylabel="Validation mIoU (%)", title="Same 20 images, same seed and training schedule")
axes[0].legend(loc="lower left")
d4 = stats[stats.inference == "d4"].set_index("experiment").loc[NAMES[1:]]
values = d4.delta_pp.to_numpy()
axes[1].errorbar(values, np.arange(3), xerr=np.array([values-d4.bootstrap_low_pp.to_numpy(), d4.bootstrap_high_pp.to_numpy()-values]),
                 fmt="o", color="#245982", capsize=5)
axes[1].axvline(0, color="#8996a5", lw=1)
axes[1].set(yticks=np.arange(3), yticklabels=NAMES[1:], xlabel="Difference vs baseline + D4 (percentage points)",
            title="Exploratory paired group bootstrap: 95% interval")
axes[1].invert_yaxis()
axes[1].grid(axis="x", alpha=.15)
save(fig, "score_comparison")

fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), layout="constrained")
for name, color in zip(NAMES, COLORS):
    v = curves[name]
    axes[0].plot(v["update"], v.validation_miou, marker=".", label=name, color=color)
    axes[1].plot(v["update"], v.validation_miou, marker=".", label=name, color=color)
axes[0].set(xlabel="Optimizer update", ylabel="mIoU (%)", ylim=(65, 80), title="Validation max(raw, EMA), no D4")
axes[1].set(xlabel="Optimizer update", ylabel="mIoU (%)", xlim=(3990, 4520), ylim=(76, 79), title="Final stage detail (native image, 500 updates)")
for ax in axes:
    ax.grid(alpha=.15)
    ax.legend(fontsize=8)
for step in [2500, 4000]:
    axes[0].axvline(step, color="#aaaaaa", ls="--", lw=.8)
save(fig, "training_curves")

fig, axes = plt.subplots(1, 2, figsize=(14, 5), layout="constrained")
class_df = pd.DataFrame(class_records)
mat = np.array([class_df[class_df.experiment == n].delta_pp.to_numpy() for n in NAMES[1:]])
im = axes[0].imshow(mat, cmap="RdBu", vmin=-2, vmax=2, aspect="auto")
axes[0].set(xticks=np.arange(4), xticklabels=["Primary Si", "Al3Ni", "Eutectic Si", "Al"],
            yticks=np.arange(3), yticklabels=NAMES[1:], title="Class IoU change vs baseline + D4 (pp)")
for i in range(3):
    for j in range(4):
        axes[0].text(j, i, f"{mat[i,j]:+.3f}", ha="center", va="center", color="white" if abs(mat[i,j]) > 1.2 else "#233444")
fig.colorbar(im, ax=axes[0], shrink=.65)
for i, name in enumerate(NAMES[1:]):
    delta = frames[name, "d4"].mIoU_exclude - base.mIoU_exclude
    axes[1].scatter(np.arange(20), delta, label=name, s=25, color=COLORS[i+1], alpha=.8)
axes[1].axhline(0, color="#777777", lw=.8)
axes[1].set(xticks=np.arange(20), xticklabels=base.index, ylabel="mIoU change (pp)", title="All 20 image differences")
plt.setp(axes[1].get_xticklabels(), rotation=75, ha="right", fontsize=7)
axes[1].legend(fontsize=8)
save(fig, "class_and_image_changes")

report = """# Augmentation 실험 결과 분석 — 2026-09-20

**이번 세 augmentation 후보 중 최고는 `photo_mild + D4` 78.7203%지만, 기존 D4 대비 +0.0338%p로 사실상 근소한 차이다. 노이즈와 배율 변화는 이번 강도·학습 일정에서 성능을 낮췄다.** 기본 augmentation을 유지하고, 밝기 강화는 반복 확인할 후보로 남기는 판단이 적절하다.

## 동일 조건의 결과

세 모델 모두 seed 2026, GPU 8장, effective batch 8, 4,500 updates(2,500/1,500/500)를 완료했다. 원본 모델·학습 엔진의 코드 해시, 데이터 해시, 학습/검증 ID, 모델/손실/학습률/일정/평가 설정이 기존 모델과 동일함을 확인했다. mIoU는 공식 검증 20장의 이미지별 평균이며 픽셀 accuracy와 다르다.

| 실험 | 기본 추론 mIoU (%) | D4 mIoU (%) | 기존 D4 대비 (%p) | 개선 이미지 |
|---|---:|---:|---:|---:|
| 기존 모델 | 78.4729 | 78.6865 | 기준 | — |
| 밝기·대비·감마 강화 (`photo_mild`) | 78.5445 | **78.7203** | **+0.0338** | 8/20 |
| 약한 노이즈 (`noise_mild`) | 77.5921 | 77.7765 | −0.9100 | 5/20 |
| 소폭 배율 변화 (`scale_mild`) | 77.4215 | 77.5778 | −1.1087 | 4/20 |

![점수와 탐색적 신뢰구간](score_comparison.png)

기존 accuracy_v1까지 포함한 기록상 최고는 여전히 `seed2028 + D4` 78.7858%다. photo_mild는 이보다 0.0655%p 낮다. 다만 seed2028 최고점에는 앞선 분석에서 확인한 A71 빈 클래스 제외 효과가 있으므로, 이 기록만으로 광범위한 우열을 단정하지 않는다.

## 밝기 강화: 소폭 상승, 교체 근거는 약함

기본 추론 +0.0717%p, D4 +0.0338%p다. D4 기준 8장은 개선되고 12장은 악화했으며, 이미지별 변화의 중앙값은 −0.1234%p다. 개선된 일부 이미지가 평균을 조금 끌어올린 결과다.

- 가장 크게 개선: A42_step2_sample4 +1.5376%p, A11_step2_sample4 +1.5009%p, A67_step1_sample1 +1.1246%p.
- 크게 악화: A59_step3_sample1 −0.9968%p, A186_step2_sample2 −0.8559%p, A181_step1_sample1 −0.6875%p.
- 모든 클래스가 존재하는 동일 19장에서도 +0.0420%p로 차이가 작다.
- 탐색적 paired group bootstrap 95% 구간은 **[−0.2910, +0.3620]%p**로 0을 포함한다.
- 전체 픽셀을 합산한 보조 mIoU는 기존 D4 81.4989% → 81.4418%로 소폭 낮다. 이 값은 이미지별 평균인 주 지표와 다른 지표이며 기준을 바꾸지 않았다.

따라서 이번 설정을 명확한 개선으로 채택하기보다 같은 seed의 대조군과 추가 seed/group fold에서 반복 확인할 후보로 남긴다.

## 노이즈와 배율 변화: 현재 설정은 제외할 근거가 있음

노이즈는 검증 이미지 15/20장에서 악화했고, 배율 변화는 16/20장에서 악화했다. A71을 제외한 동일 19장에서도 각각 −0.7232%p, −0.9283%p로 낮아져 특정 한 장만의 문제가 아니다.

| 클래스 | 기존 D4 IoU (%) | 밝기 강화 변화 (%p) | 노이즈 변화 (%p) | 배율 변화 (%p) |
|---|---:|---:|---:|---:|
| 초정 Si | 86.3947 | +0.1739 | −0.5336 | −0.6332 |
| Al₃Ni | 74.4029 | +0.0462 | −1.3179 | −1.6791 |
| 공정 Si | 59.8481 | −0.0315 | −1.7178 | −1.9555 |
| Al | 94.1002 | −0.0532 | −0.0705 | −0.1670 |

**주요 손실은 원래 취약했던 공정 Si와 Al₃Ni에 집중됐다.** 합산 혼동행렬에서도 공정 Si를 Al₃Ni로 잘못 예측한 비율이 기존 9.5771%에서 노이즈 10.4506%, 배율 10.6006%로 늘었다. 이 패턴은 미세 조직 구분이 악화됐음을 보여준다. 노이즈/보간이 미세 특징을 손상했을 가능성은 있지만 이번 로그만으로 원인을 확정할 수 없다. 적용 강도, 최종 단계 길이, seed가 달라질 때도 항상 해롭다고 일반화하지 않는다.

![클래스별·이미지별 변화](class_and_image_changes.png)

탐색적 95% 구간은 노이즈 **[−1.6153, −0.2601]%p**, 배율 **[−1.6998, −0.5552]%p**다. 해당 검증셋에서는 하락 방향이 일관되지만, 반복 학습 간 변동을 포함하는 구간은 아니다.

## A71의 빈 클래스 영향 재확인

이번에는 **모든 모델이 동일한 20장 × 4개 클래스 집계**를 사용했고, exclude/one/zero 점수가 모두 같다. 앞선 seed2028/fullframe의 평균 분모 변화가 이번 결과를 만들지는 않았다. A71에는 초정 Si 정답이 0개지만 모든 모델이 초정 Si를 일부 예측해 이 클래스가 0점으로 포함된다.

| 모델 (D4) | A71 mIoU (%) | 초정 Si 오검출 픽셀 |
|---|---:|---:|
| 기존 | 40.7059 | 780 |
| 밝기 강화 | 40.5845 | 722 |
| 노이즈 | 36.2479 | 4,181 |
| 배율 변화 | 36.1695 | 7,125 |

노이즈/배율에서는 초정 Si 오검출도 늘고 전체 분할 점수도 낮아졌다. 단, 초정 Si IoU는 모두 0점이므로 오검출 픽셀 증가량과 mIoU 하락량을 직접 비례 관계로 해석하지 않는다.

## 학습과 시간

| 실험 | 학습 로그 시간 | 최고 체크포인트 | 최고 가중치 |
|---|---:|---:|---|
| 밝기 강화 | 9.17분 | update 4,250 | raw |
| 노이즈 | 9.09분 | update 4,250 | raw |
| 배율 변화 | 9.06분 | update 4,500 | raw |

합계 **27.32분**이다. 이 값은 학습 중 검증을 포함하며 후속 plain/D4 평가 및 프로세스 시작 시간은 제외한다.

밝기 강화는 patch512 단계 최고 77.6993%, patch768 단계 최고 78.1098%로 기존 단계 최고 75.1918%, 76.4152%보다 높았다. 하지만 마지막 전체 영상 단계까지 학습하면 기존과의 차이는 +0.0717%p로 줄었다. 초기 검증 점수만으로 최종 개선을 판단하면 안 되는 사례다. 배율 후보는 마지막 update가 최고점이므로 로그만 보고 완전히 수렴했다고 단정할 수는 없다.

![학습 곡선](training_curves.png)

D4는 이번에도 모든 후보에서 개선됐다: 밝기 +0.1758%p(16/20장), 노이즈 +0.1844%p(17/20장), 배율 +0.1563%p(18/20장). 다만 노이즈/배율의 학습 성능 하락을 상쇄하지는 못했다.

## 현재 선택과 다음 확인

1. **기존 augmentation + D4를 기준으로 유지한다.** 기존 augmentation을 모두 제거하는 실험은 하지 않았으므로 augmentation 자체가 불필요하다는 결론은 아니다.
2. **이번 noise_mild/scale_mild는 현재 학습 설정에서 채택하지 않는다.** 같은 설정의 학습을 더 반복하기보다 원인이나 설정을 바꾸는 명확한 가설이 생겼을 때 재검토한다.
3. **photo_mild는 보류 후보**다. 실제 채택 전 control과 photo_mild를 같은 추가 seed 또는 같은 group fold에서 비교한다. 현재 +0.0338%p만으로 기존 모델을 교체할 근거는 약하다.
4. 새 구조/손실을 제안하기 전에 A11/A71의 Al₃Ni·공정 Si 정답과 예측을 계속 검토한다. 검증 정답을 학습에 섞거나 평가 규칙을 최고점에 맞춰 바꾸지 않는다.

## 해석 범위·보존

이번 비교는 seed 2026 한 번의 학습과 이미 모델 선택에 사용된 동일 검증 20장에 기반한다. A-prefix 19개 그룹을 단위로 50,000회 paired bootstrap(seed 2026)을 수행했고, 그룹 재표집 후 이미지 수로 가중한 평균 차이의 2.5/97.5 백분위를 사용했다. 모델 선택·다중 비교 효과는 보정하지 않은 탐색적 구간이다. 공식 train/valid에서 A-prefix 7개가 겹치며, 접두사가 물리적으로 동일한 시편인지 확인되지는 않았다. 독립 시편의 성능을 입증한 결과로 해석하지 않는다.

기존 소스·모델·결과는 수정하지 않았다. 이 폴더에 분석 스크립트, Markdown 보고서, CSV/JSON, PNG/SVG 그림만 추가했다. 새 학습·추론이나 서버·브라우저 실행은 하지 않았다. `input_hashes.json`에는 실제 읽은 결과 파일과 체크포인트의 SHA-256을 저장했다. `analyze.py`는 NumPy/Pandas/PyYAML/Matplotlib가 있는 Python에서 재현 가능하다. 차트는 기존 임시 분석 환경을 사용해 conda 환경을 바꾸지 않았다.
"""
(HERE / "REPORT.md").write_text(report)
# Confirm analyzed inputs did not change during report creation.
for path, expected in list(inputs.items()):
    assert register(path) and inputs[path] == expected, path
dump("input_hashes.json", inputs)
dump("verification.json", dict(recomputed_metrics_match=True, same_validation_ids=True,
    same_ground_truth_pixel_counts=True, same_model_loss_training_config=True,
    matching_checkpoint_hashes=True, unchanged_inputs=True, empty_union_switch=False,
    completed_training_runs=3, no_browser_or_server=True))
print(stats[["experiment", "inference", "miou", "delta_pp", "improved_images"]].to_string(index=False))
print(f"Report: {HERE / 'REPORT.md'}")
