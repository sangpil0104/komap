"""Analyze saved predictions and logs only; no model training or inference."""
from pathlib import Path
import hashlib
import json
import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/komap-analysis-mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE.parent
PROJECT = HERE.parents[3]
EXPERIMENTS = PROJECT / 'experiments'
BASE = EXPERIMENTS / 'accuracy_v1/baseline'
inputs = {}


def register(path):
    path = Path(path)
    digest = hashlib.sha256()
    with path.open('rb') as f:
        for data in iter(lambda: f.read(8 << 20), b''):
            digest.update(data)
    inputs[str(path)] = digest.hexdigest()
    return path


def read_json(path):
    return json.loads(register(path).read_text())


def read_csv(path):
    return pd.read_csv(register(path))


suite = read_json(OUT / 'suite_status.json')
assert suite['state'] == 'completed' and len(suite['completed']) == 10
table = read_csv(OUT / 'reports/comparison.csv')
audit = read_json(HERE / 'checkpoint_audit.json')
assert len(audit) == 10
for row in audit:
    inputs.update(row['hashes'])
names = sorted(row['experiment'] for row in audit)
frames = {}
for row in table.itertuples():
    frames[row.experiment, row.inference] = read_csv(row.source).set_index('image_id').sort_index()
    if row.experiment != 'baseline':
        directory = Path(row.source).parent
        provenance = read_json(directory / 'provenance.json')
        assert not provenance['request']['smoke_test']
        assert provenance['request']['checkpoint_sha256'] == next(
            a['hashes'][str(OUT / 'runs' / row.experiment / 'seed2026/official/best.pt')]
            for a in audit if a['experiment'] == row.experiment)
        for relative, digest in provenance['artifacts'].items():
            artifact = directory / relative
            register(artifact)
            assert inputs[str(artifact)] == digest
        read_json(directory / 'metrics_summary.json')
previous = read_csv(EXPERIMENTS / 'accuracy_v1/outputs/reports/comparison.csv')
for row in previous.itertuples():
    if row.experiment.split(' / ')[0] in ('seed2028', 'fullframe', 'seed_ensemble'):
        name, mode = row.experiment.split(' / ')
        frames['previous_' + name, mode.lower()] = read_csv(row.source).set_index('image_id').sort_index()
aug = read_csv(EXPERIMENTS / 'augmentation_v1/outputs/reports/comparison.csv')
for mode in ('plain', 'd4'):
    path = aug[(aug.experiment == 'photo_mild') & (aug.inference == mode)].iloc[0].source
    frames['previous_photo_mild', mode] = read_csv(path).set_index('image_id').sort_index()

reference = frames['baseline', 'd4']
ids = reference.index
cms = np.stack([json.loads(v) for v in reference.confusion_matrix])
common = (cms.sum(2) > 0).all(1)
assert common.sum() == 19
manifest = read_csv(BASE / 'data/manifest.csv').set_index('image_id')
group_ids = manifest.loc[ids, 'group_id'].to_numpy()
groups = sorted(set(group_ids))
rng = np.random.default_rng(20260920)
draws = rng.integers(0, len(groups), size=(30000, len(groups)))
counts = np.array([(group_ids == g).sum() for g in groups])


def interval(delta):
    totals = np.array([delta[group_ids == g].sum() for g in groups])
    means = totals[draws].sum(1) / counts[draws].sum(1)
    return np.quantile(means, [.025, .975]).tolist()


metric_rows, image_rows, class_rows = [], [], []
for (name, mode), frame in frames.items():
    assert frame.index.is_unique and frame.index.equals(ids)
    cm = np.stack([json.loads(v) for v in frame.confusion_matrix])
    assert np.array_equal(cm.sum(2), cms.sum(2))
    assert np.array_equal(cm.sum((1, 2)), (frame.H * frame.W).to_numpy())
    diag = np.diagonal(cm, axis1=1, axis2=2)
    union = cm.sum(1) + cm.sum(2) - diag
    ious = np.divide(diag, union, out=np.full(diag.shape, np.nan), where=union > 0) * 100
    score = np.nanmean(ious, axis=1)
    np.testing.assert_allclose(score, frame.mIoU_exclude, atol=1e-8)
    np.testing.assert_allclose(ious, frame[[f'IoU_{c}' for c in range(4)]], atol=1e-8)
    pooled = cm.sum(0)
    g_iou = np.diag(pooled) / (pooled.sum(0) + pooled.sum(1) - np.diag(pooled)) * 100
    ref = frames['baseline', mode]
    delta = score - ref.mIoU_exclude.to_numpy()
    low, high = interval(delta)
    row = dict(experiment=name, inference=mode, image_miou=float(score.mean()), global_miou=float(g_iou.mean()),
        delta_baseline_pp=float(delta.mean()), median_delta_pp=float(np.median(delta)),
        improved_images=int((delta > 1e-8).sum()), worsened_images=int((delta < -1e-8).sum()),
        common19_miou=float(score[common].mean()), common19_delta_pp=float(delta[common].mean()),
        bootstrap_group_low=low, bootstrap_group_high=high,
        empty_one=float(np.nan_to_num(ious, nan=100).mean()), empty_zero=float(np.nan_to_num(ious, nan=0).mean()),
        a71_miou=float(frame.loc['A71_step2_sample1', 'mIoU_exclude']),
        d4_gain_pp=float(frame.mIoU_exclude.mean() - frames[name, 'plain'].mIoU_exclude.mean())
            if (name, 'plain') in frames else None)
    if (name, mode) in [(r.experiment, r.inference) for r in table.itertuples()]:
        original = table[(table.experiment == name) & (table.inference == mode)].iloc[0]
        assert abs(original.image_mIoU - row['image_miou']) < 1e-8
        assert abs(original.global_mIoU - row['global_miou']) < 1e-8
    metric_rows.append(row)
    for c, label in enumerate(('Primary Si', 'Al3Ni', 'Eutectic Si', 'Al')):
        class_rows.append(dict(experiment=name, inference=mode, class_id=c, class_name=label,
            image_mean_iou=float(np.nanmean(ious[:, c])), contributing_images=int(np.isfinite(ious[:, c]).sum()),
            global_iou=float(g_iou[c]), global_precision=float(pooled[c, c] / pooled[:, c].sum() * 100),
            global_recall=float(pooled[c, c] / pooled[c].sum() * 100),
            predicted_fraction=float(pooled[:, c].sum() / pooled.sum() * 100),
            ground_truth_fraction=float(pooled[c].sum() / pooled.sum() * 100)))
    for i, image_id in enumerate(ids):
        image_rows.append(dict(experiment=name, inference=mode, image_id=image_id,
            group_id=group_ids[i], miou=float(score[i]), delta_baseline_pp=float(delta[i]),
            all_gt_classes=bool(common[i])))
metrics = pd.DataFrame(metric_rows)
classes = pd.DataFrame(class_rows)
images = pd.DataFrame(image_rows)
metrics.to_csv(HERE / 'metric_analysis.csv', index=False)
classes.to_csv(HERE / 'class_analysis.csv', index=False)
images.to_csv(HERE / 'image_analysis.csv', index=False)

training, events = [], []
for name in names:
    folder = OUT / 'runs' / name / 'seed2026/official'
    status = read_json(folder / 'status.json')
    assert status['state'] == 'completed' and status['update'] == 2000
    history = [json.loads(s) for s in register(folder / 'metrics.jsonl').read_text().splitlines()]
    valid = [s for s in history if 'validation_miou' in s]
    assert len(valid) == 10 and all(np.isfinite(x['loss']) for x in history if 'loss' in x)
    item = next(x for x in audit if x['experiment'] == name)
    assert abs(max(v['validation_miou'] for v in valid) - item['best_plain_miou']) < 1e-8
    assert abs(item['best_plain_miou'] - frames[name, 'plain'].mIoU_exclude.mean()) < 1e-8
    row = {k: v for k, v in item.items() if k != 'hashes'}
    row['minutes'] = status['elapsed_seconds'] / 60
    for stage in range(3):
        stage_values = [x['validation_miou'] for x in valid if x['stage'].startswith(f's{stage}_')]
        row[f'stage{stage}_best'] = max(stage_values)
    training.append(row)
    events.extend(dict(experiment=name, **v) for v in valid)
training = pd.DataFrame(training)
training.to_csv(HERE / 'training_analysis.csv', index=False)
events = pd.DataFrame(events)
events.to_csv(HERE / 'validation_history.csv', index=False)

pairs = [
    ('ASPP', 'e00_ft_control', 'e05_aspp'), ('ASPP + hard', 'e01_hard', 'e06_aspp_hard'),
    ('ASPP + weighted', 'e02_weighted', 'e07_aspp_weighted'),
    ('ASPP + hard + weighted', 'e04_hard_weighted', 'e08_aspp_hard_weighted'),
    ('Hard sampling', 'e00_ft_control', 'e01_hard'),
    ('Weighted CE', 'e00_ft_control', 'e02_weighted'),
    ('Dice/Lovasz', 'e00_ft_control', 'e03_overlap_loss'),
    ('Longer full-frame stage', 'e08_aspp_hard_weighted', 'e09_aspp_long_tail')]
effects = []
for factor, source, target in pairs:
    delta = (frames[target, 'd4'].mIoU_exclude - frames[source, 'd4'].mIoU_exclude).to_numpy()
    low, high = interval(delta)
    effects.append(dict(factor=factor, reference=source, candidate=target, delta_pp=float(delta.mean()),
        bootstrap_group_low=low, bootstrap_group_high=high))
effects = pd.DataFrame(effects)
effects.to_csv(HERE / 'paired_effects.csv', index=False)

request = read_json(OUT / 'evaluations/e00_ft_control/seed2026/d4/request.json')
train_groups = set(manifest.loc[request['train_ids'], 'group_id'])
shared_groups = sorted(train_groups & set(groups))
summary = dict(completed_experiments=10, updates_per_experiment=2000,
    validation_images=20, validation_groups=len(groups), shared_train_validation_groups=shared_groups,
    image_id_train_validation_overlap=[], training_minutes_sum=float(training.minutes.sum()),
    metrics_recomputed_from_confusion_matrices=True, evaluated_artifact_hashes_valid=True,
    checkpoint_provenance_valid=True, group_bootstrap_replicates=30000,
    bootstrap_note='Exploratory paired A-ID cluster bootstrap; not adjusted for model selection, training seeds, or train/validation group overlap.',
    full_training_or_new_inference_in_this_analysis=False)
(HERE / 'analysis.json').write_text(json.dumps(summary, indent=2))

# Export static plots using the already available temporary analysis environment.
plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
d4 = metrics[metrics.inference == 'd4'].set_index('experiment')
ordered = ['previous_seed2028', 'previous_fullframe', 'previous_photo_mild', 'baseline'] + list(
    d4.loc[names].sort_values('image_miou', ascending=False).index)
fig, axes = plt.subplots(1, 2, figsize=(14, 7), layout='constrained')
colors = ['#4477AA'] * 3 + ['#228833'] + ['#CC6677'] * 10
positions = np.arange(len(ordered))
for ax, column, title in zip(axes, ('image_miou', 'global_miou'), ('Image-wise mIoU (primary)', 'Global mIoU (secondary)')):
    values = d4.loc[ordered, column]
    ax.scatter(values, positions, c=colors, s=45, zorder=3)
    ax.axvline(d4.loc['baseline', column], c='#228833', ls='--', lw=1)
    for y, value in enumerate(values):
        ax.annotate(f'{value:.4f}', (value, y), xytext=(7, 0), textcoords='offset points', va='center', fontsize=9)
    ax.set_yticks(positions, ordered if ax is axes[0] else [])
    ax.invert_yaxis()
    ax.set_xlim(values.min() - .05, values.max() + .17)
    ax.set_xlabel('mIoU (%) — zoomed scale')
    ax.set_title(title)
    ax.grid(axis='x', alpha=.2)
fig.suptitle('Same 20 validation images, D4 inference: no refinement candidate beats the parent')
for ext in ('png', 'svg'):
    fig.savefig(HERE / f'score_comparison.{ext}', dpi=170)
plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(14, 5), layout='constrained')
for name, color in [('e00_ft_control', '#4477AA'), ('e05_aspp', '#228833'), ('e08_aspp_hard_weighted', '#CC6677'), ('e09_aspp_long_tail', '#AA3377')]:
    frame = events[events.experiment == name]
    axes[0].plot(frame['update'], frame.validation_miou, '.-', label=name, c=color)
axes[0].axhline(frames['baseline', 'plain'].mIoU_exclude.mean(), c='black', ls='--', label='Parent before fine-tuning')
axes[0].axvline(800, c='grey', ls=':', alpha=.6)
axes[0].axvline(1400, c='grey', ls=':', alpha=.6)
axes[0].set(xlabel='Additional optimizer updates', ylabel='Plain validation mIoU (%)', title='Patch stages lose accuracy; full-frame stage recovers it')
axes[0].legend(fontsize=8)
axes[0].grid(alpha=.2)
axes[0].text(.02, .02, 'Stage lines: defaults 800 / 1400; e09 uses 600 / 1000', transform=axes[0].transAxes, fontsize=8)
selected = d4.loc[names].sort_values('image_miou', ascending=False)
y = np.arange(10)
axes[1].errorbar(selected.delta_baseline_pp, y,
    xerr=np.array([selected.delta_baseline_pp - selected.bootstrap_group_low,
                   selected.bootstrap_group_high - selected.delta_baseline_pp]),
    fmt='o', color='#CC6677', capsize=3)
axes[1].axvline(0, c='black', ls='--')
axes[1].set_yticks(y, selected.index)
axes[1].invert_yaxis()
axes[1].set(xlabel='D4 change vs parent (percentage points)', title='Exploratory 95% paired group bootstrap intervals')
axes[1].grid(axis='x', alpha=.2)
for ext in ('png', 'svg'):
    fig.savefig(HERE / f'training_and_uncertainty.{ext}', dpi=170)
plt.close(fig)

# Keep the report template below numeric claims close to their source data.
def md_table(headers, rows):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---'] * len(headers)) + ' |'] +
        ['| ' + ' | '.join(str(v) for v in row) + ' |' for row in rows])

rank_table = md_table(['실험 (D4)', '이미지 평균 mIoU', 'Global mIoU', '기준 대비 pp', '향상/악화 이미지'],
    [[name, f'{d4.loc[name,"image_miou"]:.4f}%', f'{d4.loc[name,"global_miou"]:.4f}%',
      f'{d4.loc[name,"delta_baseline_pp"]:+.4f}', f'{int(d4.loc[name,"improved_images"])}/{int(d4.loc[name,"worsened_images"])}'] for name in ordered])
class_table = md_table(['D4 모델', '초정 Si', 'Al3Ni', '공정 Si', 'Al'],
    [[name] + [f'{v:.4f}' for v in classes[(classes.experiment == name) & (classes.inference == 'd4')].image_mean_iou]
     for name in ['baseline', 'e00_ft_control', 'e05_aspp', 'e02_weighted', 'e08_aspp_hard_weighted']])
effect_table = md_table(['변경', '비교', 'D4 변화 pp'],
    [[r.factor, f'{r.reference} → {r.candidate}', f'{r.delta_pp:+.4f}'] for r in effects.itertuples()])
train_table = md_table(['실험', '시간(분)', '선택 update', '가중치', '512 단계 최고', '768 단계 최고', '전체 영상 최고'],
    [[r.experiment, f'{r.minutes:.2f}', r.best_update, r.best_weights, f'{r.stage0_best:.4f}', f'{r.stage1_best:.4f}', f'{r.stage2_best:.4f}'] for r in training.itertuples()])
e08 = d4.loc['e08_aspp_hard_weighted']
weak = images[(images.experiment == 'e08_aspp_hard_weighted') & (images.inference == 'd4')].sort_values('delta_baseline_pp')
image_table = md_table(['이미지', 'e08 mIoU', '기준 대비 pp'],
    [[r.image_id, f'{r.miou:.4f}', f'{r.delta_baseline_pp:+.4f}'] for r in weak.head(5).itertuples()])
control = training[training.experiment == 'e00_ft_control'].iloc[0]
report = f'''# refinement_v1 완료 결과 분석

분석일: 2026-09-20. 기준: 공식 validation 20장, 클래스 순서 초정 Si / Al3Ni / 공정 Si / Al. 모든 점수는 %, 변화량은 percentage point(pp)입니다.

## 결론

**10개 모두 정상 완료했지만, 이번 미세조정에서는 정확도 개선이 확인되지 않았습니다. 기존 모델을 유지합니다.**

- 기준 CNN+D4: 이미지 평균 **78.6865**, Global **81.4989**.
- 새 실험 최고는 구조를 바꾸지 않은 e00 추가 학습 대조군: **78.6336**, 기준 대비 **-0.0529 pp**.
- 새 구조 후보 최고 e05 ASPP: **78.6293**, 기준 대비 **-0.0571 pp**. 대조군과 차이는 **-0.0042 pp**로 매우 작습니다.
- 기대했던 e08 ASPP+hard+weighted: **78.4935**, 기준 대비 **-0.1930 pp**. 목표 클래스까지 악화했습니다.
- 이전 모든 기록의 이미지 평균 최고는 seed2028+D4 **78.7858**입니다. 이번 10개가 이를 갱신하지 못했습니다.
- 모든 새 후보가 plain에서도 기존 plain **78.4729**를 넘지 못했습니다.

![점수 비교](score_comparison.png)

## 전체 순위

{rank_table}

`previous_`는 이전 실험입니다. `selection.json`은 기준 모델과 refinement 후보만 비교하므로 baseline을 선택합니다. 이전 seed2028이나 photo_mild를 포함한 전체 과거 기록의 자동 순위 파일이 아닙니다.

평가 지표에 따라 이전 모델의 순위도 다릅니다. 기존 seed 앙상블의 Global mIoU는 **{d4.loc['previous_seed_ensemble','global_miou']:.4f}**로 높지만 이미지 평균은 **{d4.loc['previous_seed_ensemble','image_miou']:.4f}**입니다. 주 지표인 이미지 평균을 우선한다면 이것을 새 최고 모델로 바꾸면 안 됩니다.

## 무엇이 효과가 있었는가

대조군 또는 나머지 조건이 같은 후보끼리 비교했습니다.

{effect_table}

ASPP 유무를 짝지은 네 비교 모두 D4에서 약 0.001~0.006 pp 하락했습니다. 이 실험에서는 추가 문맥 블록의 이득을 관찰하지 못했으며, 이 미세한 차이만으로 ASPP 일반의 유효성을 판단할 수 없습니다. 실제 체크포인트의 context projection norm은 0보다 커서 블록이 전혀 학습되지 않은 상황은 아닙니다.

가중 CE의 대조군 대비 변화는 -0.1074 pp로, ASPP 변경보다 영향이 컸습니다. hard sampling도 -0.0419 pp였습니다. 이번 가중치와 샘플링 비율을 그대로 확대할 근거는 없습니다. e09는 전체 영상 업데이트를 늘렸지만 앞선 패치 단계 길이도 함께 줄였으므로, 전체 영상 학습만의 독립 효과로 해석하지 않습니다.

모든 새 후보에서 D4가 plain보다 **{d4.loc[names,'d4_gain_pp'].min():.4f}~{d4.loc[names,'d4_gain_pp'].max():.4f} pp** 높았습니다. 기존에도 D4 이득은 +0.2136 pp였으므로, 정확도를 우선하는 현재 추론 설정에서는 D4를 유지할 근거가 있습니다.

## 학습 기록에서 보이는 원인 후보

{train_table}

위 단계별 최고 점수는 **plain 검증**입니다. 기준 모델은 이미 전체 영상 학습을 끝낸 가중치입니다. 이를 다시 512 패치 학습에 넣자 대조군 점수가 첫 검증에서 **{control.stage0_best:.4f}**로 내려갔고, 800 update 시점에는 **77.3352**까지 낮아졌습니다. 마지막 전체 영상 단계에서 **{control.stage2_best:.4f}**로 회복했지만 시작점 78.4729에는 못 미쳤습니다. 다른 후보도 유사한 흐름입니다.

이는 **완성된 전체 영상 모델을 패치 학습부터 다시 시작한 일정이 적합하지 않았을 가능성**을 지지합니다. 손실/분포 변화와 일정의 효과가 섞여 있으므로 과적합이나 학습률 문제 중 하나로 원인을 확정할 수는 없습니다. 단순히 더 오래 학습하면 나아진다는 증거도 없습니다.

학습 엔진은 추가 학습을 시작할 때 최고 점수를 -1로 초기화하고, 200 update마다 평가한 새 가중치 중 최고를 저장합니다. 부모 가중치를 update 0 후보로 등록하지 않았기 때문에 `best.pt`라는 이름이어도 부모 모델보다 낮을 수 있습니다. 버전별 비교표가 부모 모델을 별도로 포함하므로 이번에는 이를 확인할 수 있었습니다. `best.pt`의 completed=false는 최종 update 이전에 선택됐기 때문이며, 완료 여부는 `last.pt`와 status의 2000/2000으로 확인했습니다.

![학습 곡선과 불확실성](training_and_uncertainty.png)

## 클래스와 이미지별 변화

아래 값은 이미지별 클래스 IoU 평균입니다.

{class_table}

e08의 기준 대비 변화는 초정 Si **-0.1683**, Al3Ni **-0.3700**, 공정 Si **-0.1942**, Al **-0.0396 pp**입니다. 개선 목표였던 Al3Ni·공정 Si가 함께 하락했습니다. 이 결론은 단순한 클래스 평균뿐 아니라 Global 클래스 IoU에서도 동일합니다. Global precision/recall 및 예측 면적 비율은 `class_analysis.csv`에 기록했습니다.

e08은 20장 중 {int(e08.improved_images)}장 개선, {int(e08.worsened_images)}장 악화했습니다. 하락이 큰 사례:

{image_table}

A71을 뺀 동일 19장에서도 e08은 **{e08.common19_delta_pp:+.4f} pp**이고, 대조군도 **{d4.loc['e00_ft_control','common19_delta_pp']:+.4f} pp**입니다. 이번 실패가 A71 한 장에만 의존하지는 않습니다.

과거 seed2028은 기준 대비 +0.0993 pp였지만, A71을 뺀 19장에서는 **{d4.loc['previous_seed2028','common19_delta_pp']:+.4f} pp**였습니다. A71에는 정답 초정 Si가 없고, seed2028의 예측에서도 초정 Si가 없어 해당 클래스가 exclude 평균에서 제외됐습니다. 따라서 seed2028의 최고 점수를 넓은 이미지 집합에서의 일관된 개선으로 해석하면 안 됩니다. 기준 모델과 seed2028 모두 계속 보관하는 것이 타당합니다.

## 검증 범위와 동료 결과

- 저장된 confusion matrix로 {len(frames)}개 모델/추론 조합의 이미지 평균·Global 점수를 재계산했고 CSV와 일치했습니다. 해상도, 동일 ID 20장, 동일 GT 클래스 픽셀 수, 평가 산출물 해시와 체크포인트 연결도 확인했습니다.
- 새 실험은 모두 seed2026 한 번씩입니다. 부트스트랩은 검증 {len(groups)}개 A-ID를 단위로 짝지어 30,000회 재추출한 탐색용 구간입니다. 모델/체크포인트 선택, 반복 실험, 학습 seed 변동을 보정하지 않으므로 공식 유의성 검정으로 해석하지 않습니다.
- 이미지 ID의 train/validation 중복은 없습니다. 다만 공식 분할에서 A-ID 그룹 {len(shared_groups)}개가 양쪽에 존재합니다: {', '.join(shared_groups)}. 같은 A-ID가 연관된 시편이라는 가정에서 독립 시편 일반화 성능으로 확장해 해석하는 데 한계가 있습니다.
- 동료 residual ASPP 문서의 **Global 79.48 / 이미지 평균 75.83**과 현재 기준의 **81.4989 / 78.6865**는 기록상 현재 기준 쪽이 높습니다. 사용자는 이미지가 같다고 확인했으나 정확한 평가 목록·클래스 매핑·평가 코드·TTA가 확인되지 않아 통제된 모델 간 비교는 아닙니다.
- 테스트 10장에는 정답이 없어 여기서는 테스트 정확도를 주장하지 않습니다.

## 다음 실험 우선순위

1. **현재 모델 교체 보류.** 주 지표 관측 최고 seed2028+D4와 비교 기준 baseline+D4를 보존합니다. 이번 후보 중 채택할 근거는 없습니다.
2. **패치 단계를 생략한 전체 영상 미세조정부터 작은 대조 실험.** 동일한 부모 가중치에서 시작해 update 0을 평가 후보에 포함하고, 짧은 전체 영상 학습만 비교하는 것이 이번 기록에서 가장 직접적인 다음 가설입니다. 성공을 보장하는 제안은 아닙니다.
3. ASPP를 재검토한다면 기존 네트워크를 잠시 고정하고 새 블록만 학습하거나, 기존 가중치와 새 블록의 학습률을 분리하는 조건을 대조군과 비교합니다. 이번에는 모든 학습 가능한 가중치에 같은 낮은 학습률을 사용했습니다.
4. 공정 Si/Al3Ni 혼동이 큰 이미지의 원본·라벨 경계를 점검한 뒤, 새 검증 분할이나 그룹별 재학습에서 재현되는 개선인지 확인합니다. 공식 70장으로 이미 학습한 부모 가중치를 다른 fold로 그대로 가져가면 누출될 수 있으므로 그룹 CV의 부모 모델은 fold별로 새로 학습해야 합니다.

## 시간과 보존

실험별 학습 시간은 {training.minutes.min():.2f}~{training.minutes.max():.2f}분, 합계 **{training.minutes.sum():.2f}분**입니다. 이는 status 기록의 학습 실행 시간 합이며 GPU 프로세스 시작과 학습 후 plain/D4 평가 등 전체 명령의 모든 대기 시간을 포함한 벽시계 시간은 아닙니다.

이번 분석은 기존 모델·코드·결과를 수정하지 않았습니다. 새 학습·추론, 웹 서버 또는 브라우저 실행 없이 이 폴더에 보고서·차트·분석 코드·CSV/JSON만 추가했습니다. 차트에는 기존 임시 분석 환경을 사용해 komap conda 환경을 변경하지 않았습니다. `input_hashes.json`에 읽은 입력 파일 해시를 보관합니다.
'''
(HERE / 'REPORT.md').write_text(report)
(HERE / 'input_hashes.json').write_text(json.dumps(inputs, indent=2))
print(json.dumps(summary, indent=2))
print('Report:', HERE / 'REPORT.md')
