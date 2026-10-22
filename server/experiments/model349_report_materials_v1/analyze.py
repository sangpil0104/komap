"""Read-only report analysis: data EDA and frozen-cache inference ablations."""
from pathlib import Path
import gc
import hashlib
import json
import pickle
import sys

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter
from tqdm import tqdm

ROOT = Path('/Data3/sppark/komap')
HERE = Path(__file__).resolve().parent
OUT = HERE / 'outputs/analysis_20261007_1'
RUN = ROOT / 'experiments/model349_train_route_v2/outputs/runs/model349_train_20260930T143930_234475Z'
SOURCE = RUN / 'source_snapshot'
sys.path[:0] = [str(ROOT / 'experiments/model349_mask_export_v1'), str(SOURCE / 'experiments'), str(SOURCE / 'experiments/historical_scripts')]
import export_masks as original
import model335_exhaustive_b4_route_search as route
import model198_fast_branch_search as refine
from local_image_metrics import class_ious

CLASSES = original.PALETTE
NAMES = ['Al3Ni', 'Primary Si', 'Al', 'Eutectic Si']


def save(name, obj):
    (OUT / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')


def decode(path):
    a = np.asarray(Image.open(path).convert('RGB'), dtype=np.uint32)
    code = (a[:, :, 0] << 16) | (a[:, :, 1] << 8) | a[:, :, 2]
    p = CLASSES.astype(np.uint32)
    codes = (p[:, 0] << 16) | (p[:, 1] << 8) | p[:, 2]
    labels = np.full(code.shape, 255, dtype=np.uint8)
    for c, value in enumerate(codes):
        labels[code == value] = c
    assert np.all(labels < 4), path
    return labels


def eda():
    records = []
    hist = {}
    paths = [(split, p) for split in ['train', 'valid', 'test']
             for p in sorted(((ROOT / 'dataset' if split == 'test' else RUN / 'data') / split / 'images').glob('*_image.png'))]
    for split, p in tqdm(paths, desc='Dataset EDA', unit='image'):
        a = np.asarray(Image.open(p).convert('L'), dtype=np.uint8)
        h = np.bincount(a.ravel(), minlength=256)
        hist.setdefault(split, []).append((h / h.sum()).tolist())
        name = original.canonical(p.name)
        item = {'split': split, 'image': name, 'specimen': name.split('_')[0],
                'path': str(p), 'sha256': original.digest(p),
                'width': a.shape[1], 'height': a.shape[0], 'pixel_count': int(a.size),
                'gray_mean': float(a.mean()), 'gray_std': float(a.std())}
        if split != 'test':
            mask = p.parent.parent / 'masks' / p.name.replace('_image.png', '_mask.png')
            labels = decode(mask)
            assert labels.shape == a.shape
            counts = np.bincount(labels.ravel(), minlength=4)
            item.update(mask_path=str(mask), mask_sha256=original.digest(mask),
                        class_pixels=counts.tolist(), class_fraction=(counts/a.size).tolist())
        records.append(item)
    summary = {}
    for split in ['train', 'valid', 'test']:
        rows = [r for r in records if r['split'] == split]
        dims = {}
        for r in rows:
            key = f"{r['width']}×{r['height']}"
            dims[key] = dims.get(key, 0) + 1
        s = {'images': len(rows), 'specimens': sorted({r['specimen'] for r in rows}),
             'dimensions': dims, 'total_pixels': sum(r['pixel_count'] for r in rows),
             'gray_mean_mean': float(np.mean([r['gray_mean'] for r in rows])),
             'gray_std_mean': float(np.mean([r['gray_std'] for r in rows]))}
        if split != 'test':
            pixels = np.array([r['class_pixels'] for r in rows])
            frac = np.array([r['class_fraction'] for r in rows])
            s.update(pixel_fraction=(pixels.sum(0)/pixels.sum()).tolist(),
                     image_fraction_mean=frac.mean(0).tolist(),
                     image_fraction_median=np.median(frac, axis=0).tolist(),
                     image_fraction_q25=np.quantile(frac, .25, axis=0).tolist(),
                     image_fraction_q75=np.quantile(frac, .75, axis=0).tolist(),
                     present_images=(pixels>0).sum(0).tolist(),
                     absent_images={NAMES[c]: [r['image'] for r in rows if r['class_pixels'][c]==0] for c in range(4)})
        summary[split] = s
    overlap = {f'{a}/{b}': sorted(set(summary[a]['specimens']) & set(summary[b]['specimens']))
               for a,b in [('train','valid'), ('train','test'), ('valid','test')]}
    save('eda.json', {'records': records, 'summary': summary, 'overlap': overlap,
                      'mean_image_histograms': {k: np.mean(v,axis=0).tolist() for k,v in hist.items()},
                      'classes': NAMES})


def cleanup(pred, logits, threshold):
    pred = refine.cleanup_primary(pred, logits, threshold)
    if float((pred == 1).mean()) <= .01:
        mask = pred == 1
        other = logits.copy()
        other[1] = -np.inf
        pred[mask] = other[:, mask].argmax(axis=0).astype(np.uint8)
    return pred


def ablations():
    config, g329, g336, g347 = original.settings()
    archives = {k: original.load_archive(p, k != '269') for k,p in original.archive_paths().items()}
    with (RUN / 'outputs/model183_metric_calibration/entries.pkl').open('rb') as f:
        entries = pickle.load(f)
    names = sorted(archives['347'])
    definitions = {
        'full': ['Model349 전체', '기존 클래스별 결합 + prior·bias + 공간 필터 + Primary Si 정리'],
        'no_post': ['모든 최종 보정 제거', '동일한 클래스별 결합 확률의 argmax; prior·bias·공간 필터·정리 제거'],
        'prior_bias_only': ['prior·bias만 유지', '동일한 결합; prior·bias 유지, 공간 필터와 정리 제거'],
        'no_cleanup': ['Primary Si 정리 제거', '동일한 결합·prior·bias·공간 필터 유지; 연결 성분·1% 규칙 제거'],
        'no_spatial': ['공간 필터 제거', '동일한 결합·prior·bias·Primary Si 정리 유지; Gaussian·bilateral 제거'],
        'no_prior': ['밝기 prior 제거', 'prior_scale=0; 클래스 bias, 공간 필터, Primary Si 정리는 유지'],
        'equal_mean': ['7개 확률원의 균등 평균', '기본 확률과 Model269/322/326/329/336/347을 각 1/7로 평균; 원래 후처리 유지'],
    }
    cases = {k: [] for k in definitions}
    source_cases = {k: [] for k in archives}
    for name, item in tqdm(list(zip(names, entries)), desc='Frozen-cache comparisons', unit='image'):
        gt = decode(RUN / f'data/valid/masks/{name}_mask.png')
        assert np.array_equal(gt, item['target'])
        sources = {k: v[name] for k,v in archives.items()}
        base = route.undo_model181_primary_blend(item)
        mixed = route.make_mixed({'base':base, 'sources':sources}, config)
        for key, gammas in [('329', g329), ('336', g336), ('347', g347)]:
            for c,gamma in enumerate(gammas):
                if gamma:
                    mixed = route.replace_classes(mixed, sources[key], (c,), gamma)
        weights = route.bilateral_weights(item['z'])
        bp,delta = refine.prior_terms(item,mixed)
        prior = config['prior_scale']*bp
        prior[0] += config['prior_scale']*config['beta_scale']*item['params']['beta']*delta
        logits=np.log(np.maximum(mixed,1e-7))+.09*prior
        logits[2]+=item['params']['al_bias']+config['al_delta']
        logits[3]+=item['params']['eut_bias']+config['eut_delta']
        spatial=logits.copy()
        spatial[1]=config['gaussian_blend']*gaussian_filter(logits[1],sigma=.5,mode='reflect')+(1-config['gaussian_blend'])*refine.apply_bilateral(logits[1],weights)
        spatial[3]=(1-config['eut_edge_blend'])*logits[3]+config['eut_edge_blend']*refine.apply_bilateral(logits[3],weights)
        preds={
            'full': route.predict(item,mixed,config,weights,True),
            'no_post': mixed.argmax(0),
            'prior_bias_only': logits.argmax(0),
            'no_cleanup': spatial.argmax(0),
            'no_spatial': cleanup(logits.argmax(0).astype(np.uint8),logits,config['cleanup_threshold']),
            'no_prior': route.predict(item,mixed,{**config,'prior_scale':0.},weights,True),
            'equal_mean': route.predict(item,np.mean(np.stack([base]+[sources[k] for k in ['269','322','326','329','336','347']]),axis=0),config,weights,True),
        }
        expected=np.asarray(Image.open(original.VALID_RESULT / 'predicted_labels' / f'{name}.png'))
        assert np.array_equal(preds['full'],expected), name
        assert np.array_equal(cleanup(spatial.argmax(0).astype(np.uint8),spatial,config['cleanup_threshold']),expected), name
        for k,p in preds.items():
            ious=class_ious(p,gt)
            cases[k].append({'image':name,'image_miou':float(ious.mean()),'class_iou':ious.tolist()})
        for k,p in sources.items():
            ious=class_ious(p.argmax(0),gt)
            source_cases[k].append({'image':name,'image_miou':float(ious.mean()),'class_iou':ious.tolist()})
    summary=[]
    for k,rows in cases.items():
        summary.append({'id':k,'label':definitions[k][0],'definition':definitions[k][1],
                        'image_miou':float(np.mean([r['image_miou'] for r in rows])),
                        'class_iou':np.mean([r['class_iou'] for r in rows],axis=0).tolist(),'images':rows})
    baseline=summary[0]['image_miou']
    assert abs(baseline-.8006879890006227)<1e-12
    for row in summary:
        row['delta_pp']=(row['image_miou']-baseline)*100
    save('ablations.json',{'cases':summary,'selection':'All original weights and route parameters frozen; no search or retraining.',
                           'source_argmax':[{'model':k,'image_miou':float(np.mean([r['image_miou'] for r in rows])),
                                             'class_iou':np.mean([r['class_iou'] for r in rows],axis=0).tolist()} for k,rows in source_cases.items()],
                           'baseline_pixel_identical_images':20})
    del entries,archives
    gc.collect()


if __name__ == '__main__':
    OUT.mkdir(parents=True,exist_ok=False)
    eda()
    ablations()
    save('sources.json',{'run':str(RUN),'metrics':str(original.VALID_RESULT/'metrics.json'),
                         'input_archive_hashes':{str(p):original.digest(p) for p in original.archive_paths().values()},
                         'script_sha256':original.digest(Path(__file__))})
    print('ANALYSIS_DONE',OUT,flush=True)
