"""Reproduce this read-only analysis; writes only to this new report directory."""
import csv
import hashlib
import json
from pathlib import Path
import shutil
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from tqdm import tqdm

OUT = Path(__file__).resolve().parent
PROJECT = OUT.parents[1]
SUITE = PROJECT / 'experiments/selfsup_unet_v1/outputs/suite_20260924T113719_442712Z'
SOURCES = {a: PROJECT / 'results' / f'model_results_0924_{n}' for a, n in zip('ABC', (5, 6, 7))}
CLASSES = ['primary_si', 'al3ni', 'eutectic_si', 'al']
NAMES = ['Primary Si', 'Al3Ni', 'Eutectic Si', 'Al']
PALETTE = np.array([[76, 178, 76], [25, 76, 153], [204, 204, 204], [153, 127, 76]], dtype=np.uint8)
SPECIAL = 'A71_step2_sample1'


def read(path):
    return json.loads(path.read_text())


def write(name, obj):
    path = OUT / name
    with path.open('x', encoding='utf-8') as handle:
        json.dump(obj, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def rgb(path):
    with Image.open(path) as image:
        return np.asarray(image.convert('RGB'))


def labels(array):
    result = np.full(array.shape[:2], 255, dtype=np.uint8)
    for c, color in enumerate(PALETTE):
        result[np.all(array == color, axis=-1)] = c
    if (result == 255).any():
        raise ValueError('Unknown mask palette')
    return result


def ious(cm, policy='one'):
    tp = cm.diagonal().astype(float)
    union = cm.sum(0) + cm.sum(1) - tp
    fill = {'one': 1., 'zero': 0., 'exclude': np.nan}[policy]
    return np.divide(tp, union, out=np.full(4, fill), where=union > 0) * 100


def main():
    started = time.monotonic()
    if (OUT / 'analysis.json').exists():
        raise FileExistsError('Preserve this analysis; create a separate directory to rerun')
    summaries = {a: read(p / 'summary.json') for a, p in SOURCES.items()}
    data = {a: {r['image_id']: r for r in read(p / 'per_image.json')} for a, p in SOURCES.items()}
    saved_dataset = read(SUITE / 'dataset.json')
    manifest_path = PROJECT / 'data/processed/manifest.csv'
    assert saved_dataset['manifest_sha256'] == digest(manifest_path)
    with manifest_path.open() as handle:
        manifest = list(csv.DictReader(handle))
    valid = {r['image_id']: r for r in manifest if r['source_split'] == 'valid'}
    assert len(valid) == 20
    for a in 'ABC':
        assert set(data[a]) == set(valid) == set(saved_dataset['ids']['valid'])
    source_audit = []
    for relative, expected in read(SUITE / 'source_hashes.json').items():
        assert digest(SUITE / 'source' / relative) == expected, relative
    for row in tqdm(manifest, desc='Verify original data hashes', unit='image'):
        for kind in ('image', 'mask'):
            if row[kind + '_path']:
                assert digest(Path(row[kind + '_path'])) == row[kind + '_sha256']
        assert saved_dataset['file_hashes'][row['image_id']] == [row['image_sha256'], row['mask_sha256']]
    for a, folder in SOURCES.items():
        destination = OUT / 'valid_mask_comparison' / a
        destination.mkdir(parents=True)
        recalculated = []
        provenance = read(folder / 'provenance.json')
        assert digest(Path(provenance['checkpoint'])) == provenance['checkpoint_sha256']
        for relative, expected in read(folder / 'source_hashes.json').items():
            assert digest(folder / 'source' / relative) == expected
        for image_id, row in tqdm(valid.items(), desc=f'Verify {a}: masks + panels', unit='image'):
            original = rgb(Path(row['image_path']))
            target_rgb = rgb(Path(row['mask_path']))
            prediction_rgb = rgb(folder / 'masks' / f'{image_id}.png')
            target, prediction = labels(target_rgb), labels(prediction_rgb)
            cm = np.bincount((4 * target.astype(np.int64) + prediction).ravel(), minlength=16).reshape(4, 4)
            assert np.array_equal(cm, data[a][image_id]['confusion_matrix'])
            for policy in ('one', 'zero', 'exclude'):
                np.testing.assert_allclose(np.nanmean(ious(cm, policy)), data[a][image_id]['miou_' + policy], rtol=0, atol=1e-10)
            recalculated.append(ious(cm))
            error = np.full_like(target_rgb, 255)
            wrong = target != prediction
            error[wrong] = target_rgb[wrong]
            h, w = target.shape
            positions = [(14, 90), (14 + w + 16, 90), (14, 90 + h + 16 + 38),
                         (14 + w + 16, 90 + h + 16 + 38)]
            figure = folder / 'valid_mask_comparison' / f'{image_id}.png'
            with Image.open(figure) as image:
                for (x, y), expected in zip(positions, [original, error, prediction_rgb, target_rgb]):
                    assert np.array_equal(np.asarray(image.crop((x, y, x+w, y+h))), expected)
            copied = destination / figure.name
            shutil.copy2(figure, copied)
            assert digest(copied) == digest(figure)
            source_audit.append({'arm': a, 'image_id': image_id, 'confusion_verified': True,
                                 'four_native_panels_verified': True, 'comparison_sha256': digest(figure)})
        np.testing.assert_allclose(np.mean(recalculated), summaries[a]['image_miou'], atol=1e-10, rtol=0)
        for c in range(4):
            np.testing.assert_allclose(np.mean(recalculated, axis=0)[c], summaries[a]['iou_' + CLASSES[c]], atol=1e-10, rtol=0)
    pairs = {}
    for a, b in [('B','A'), ('C','A'), ('B','C')]:
        differences = {k: data[a][k]['image_miou'] - data[b][k]['image_miou'] for k in valid}
        values = np.array(list(differences.values()))
        pairs[a+'-'+b] = {'mean_pp': float(values.mean()), 'median_pp': float(np.median(values)),
                          'images_improved': int((values > 0).sum()), 'images_worse': int((values < 0).sum()),
                          'per_image_pp': differences}
    cm = np.asarray(data['C'][SPECIAL]['confusion_matrix'])
    prediction_c = labels(rgb(SOURCES['C'] / 'masks' / f'{SPECIAL}.png'))
    y, x = np.where(prediction_c == 0)
    special = {'image_id': SPECIAL, 'gt_primary_pixels': int(cm[0].sum()),
               'false_primary_pixels': int(cm[:, 0].sum()), 'false_primary_gt_class_counts': cm[:, 0].tolist(),
               'bbox_xy_inclusive': [int(x.min()), int(y.min()), int(x.max()), int(y.max())],
               'empty_class_transition_image_pp': 100 / 4,
               'empty_class_transition_dataset_pp': 100 / 4 / 20,
               'counterfactual_C_miou_if_only_empty_class_term_restored': summaries['C']['image_miou'] + 1.25,
               'diagnostic_only_not_an_official_score': True,
               'excluding_A71_mean_diagnostic': {a: float(np.mean([r['image_miou'] for k,r in data[a].items() if k != SPECIAL])) for a in 'ABC'},
               'primary_present19_mean': {a: float(np.mean([r['iou_primary_si'] for k,r in data[a].items() if k != SPECIAL])) for a in 'ABC'}}
    completed = read(SUITE / 'completed.json')
    loss_trends = {}
    for name in ['A_supervised', 'B_ssl', 'B_finetune', 'C_extra']:
        entries = [json.loads(line) for line in (SUITE / name / 'training.jsonl').read_text().splitlines()]
        loss_trends[name] = {'first20_logged_loss_mean': float(np.mean([r['loss'] for r in entries[:20]])),
                             'last20_logged_loss_mean': float(np.mean([r['loss'] for r in entries[-20:]])),
                             'records': len(entries)}
    analysis = {'summaries': summaries, 'pairs': pairs, 'A71': special, 'loss_trends': loss_trends,
                'stages': completed['stages'], 'budget': read(PROJECT / 'results/model_results_0924_8/budget.json'),
                'suite_wall_seconds': completed['this_invocation_wall_seconds'], 'seed': 2026}
    write('analysis.json', analysis)
    write('verification.json', {'original_images_hash_verified': 100, 'available_masks_hash_verified': 90,
                              'prediction_masks_verified': 60, 'native_panels_verified': 240,
                              'source_snapshots_verified': True, 'checkpoints_hash_verified': 3, 'images': source_audit})
    with (OUT / 'summary.csv').open('x', newline='') as handle:
        fields = ['arm', 'image_miou'] + ['iou_' + c for c in CLASSES] + ['boundary_error_2px', 'outer_error_32px']
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows({'arm': a, **summaries[a]} for a in 'ABC')
    with (OUT / 'per_image_comparison.csv').open('x', newline='') as handle:
        fields = ['image_id', 'A', 'B', 'C', 'B-A', 'C-A', 'B-C']
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for image_id in valid:
            row = {'image_id': image_id, **{a: data[a][image_id]['image_miou'] for a in 'ABC'}}
            row.update({p: value['per_image_pp'][image_id] for p,value in pairs.items()})
            writer.writerow(row)
    figures = OUT / 'figures'
    figures.mkdir()
    colors = {'A':'#356bb3', 'B':'#ad5898', 'C':'#2a8a6a'}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    for i, a in enumerate('ABC'):
        value = summaries[a]['image_miou']
        axes[0].bar(i, value, color=colors[a])
        axes[0].text(i, value+1, f'{value:.2f}', ha='center')
    axes[0].set(xticks=range(3), xticklabels=['A supervised','B SSL + supervised','C extra supervised'], ylim=(0,100), ylabel='image mIoU (%)', title='Official valid20; empty-class IoU = 1')
    positions = np.arange(4)
    for offset, a in [(-.18,'B'),(.18,'C')]:
        values = [summaries[a]['iou_'+c] - summaries['A']['iou_'+c] for c in CLASSES]
        axes[1].bar(positions+offset, values, width=.36, label=a+' minus A', color=colors[a])
    axes[1].axhline(0,color='black',lw=.6)
    axes[1].set(xticks=positions, xticklabels=NAMES, ylabel='Class IoU difference (percentage points)', title='C Primary Si decline includes the A71 empty-class effect')
    axes[1].legend()
    fig.savefig(figures/'summary.png', dpi=160)
    plt.close(fig)
    order = sorted(valid, key=lambda k:pairs['B-A']['per_image_pp'][k])
    fig, axes = plt.subplots(1, 2, figsize=(12, 8), constrained_layout=True)
    for axis, key, color in zip(axes,['B-A','C-A'],[colors['B'],colors['C']]):
        axis.barh(range(20), [pairs[key]['per_image_pp'][k] for k in order], color=color)
        axis.axvline(0, color='black', lw=.8)
        axis.set(yticks=range(20), yticklabels=order, xlabel='image mIoU difference (percentage points)', title=key+' (all 20 images)')
        axis.invert_yaxis()
    fig.savefig(figures/'per_image_deltas.png', dpi=150)
    plt.close(fig)
    crop = (548, 0, 582, 27)
    row = valid[SPECIAL]
    images = [rgb(Path(row['image_path'])), rgb(Path(row['mask_path']))]
    images += [rgb(SOURCES[a]/'masks'/f'{SPECIAL}.png') for a in 'ABC']
    fig, axes = plt.subplots(1, 5, figsize=(12, 3), constrained_layout=True)
    for axis, array, title in zip(axes, images, ['Original','Ground truth','A: 0 Primary FP','B: 0 Primary FP','C: 46 Primary FP']):
        axis.imshow(array[crop[1]:crop[3],crop[0]:crop[2]], interpolation='nearest')
        axis.set_title(title, fontsize=10)
        axis.axis('off')
    fig.suptitle('A71 top-edge diagnostic crop: x=[548,582), y=[0,27), native coordinates', fontsize=11)
    fig.savefig(figures/'A71_top_edge.png', dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    for a, name in [('A','A_supervised'),('B','B_finetune'),('C','C_extra')]:
        history = [json.loads(line) for line in (SUITE/name/'training.jsonl').read_text().splitlines()]
        steps = np.array([h['update'] for h in history]) + (4000 if a=='C' else 0)
        losses = np.array([h['loss'] for h in history])
        axes[0].plot(steps,losses,alpha=.2,color=colors[a])
        axes[0].plot(steps[9:],np.convolve(losses,np.ones(10)/10,mode='valid'),label=a,color=colors[a])
    history = [json.loads(line) for line in (SUITE/'B_ssl/training.jsonl').read_text().splitlines()]
    steps = np.array([h['update'] for h in history]); losses = np.array([h['loss'] for h in history])
    axes[1].plot(steps,losses,alpha=.3,color=colors['B'])
    axes[1].plot(steps[9:],np.convolve(losses,np.ones(10)/10,mode='valid'),color=colors['B'])
    axes[0].set(xlabel='Supervised updates (C continues A)',ylabel='Training CE + Dice',title='Training loss, 10-record moving mean')
    axes[1].set(xlabel='SSL updates',ylabel='Masked-pixel L1 (intensity range 0-1)',title='Reconstruction improves; segmentation gain is not implied')
    axes[0].legend()
    fig.savefig(figures/'training_curves.png',dpi=160)
    plt.close(fig)
    write('provenance.json', {'suite':str(SUITE), 'source_results':{a:str(p) for a,p in SOURCES.items()},
                             'analysis_script_sha256':digest(Path(__file__)),
                             'source_summary_sha256':{a:digest(p/'summary.json') for a,p in SOURCES.items()},
                             'manifest_sha256':digest(manifest_path), 'training_or_inference_performed':False,
                             'comparison_images':'byte-identical copies, original sources preserved',
                             'elapsed_seconds':time.monotonic()-started})
    print(json.dumps({'report_directory':str(OUT),'audit_passed':True,'elapsed_seconds':time.monotonic()-started},indent=2))


if __name__ == '__main__':
    main()
