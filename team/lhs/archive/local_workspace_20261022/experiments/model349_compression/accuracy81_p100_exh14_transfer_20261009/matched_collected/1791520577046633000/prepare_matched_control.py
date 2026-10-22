"""Prepare a same-executable P100 original-route control; CPU only."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')
from pathlib import Path
import hashlib
import json
import shutil
import yaml

ROOT = Path(__file__).resolve().parent
PARENT = ROOT.parent
SOURCE = PARENT / 'Model349_exh14'
OUTPUTS = Path('/workspace/komap_model1/outputs')


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def load_result(name):
    return json.loads((OUTPUTS / name / 'results.json').read_text())


def main():
    bundle = ROOT / 'matched_control_bundle'
    assert not bundle.exists()
    shutil.copytree(SOURCE, bundle, ignore=shutil.ignore_patterns(
        'weights', 'results', 'search', '__pycache__'))
    code = (bundle / 'infer_lite.py').read_text()
    needle = '    sources = dict(sources)\n    for key in'
    assert code.count(needle) == 1
    code = code.replace(needle, "    sources = dict(sources)\n    sources['b4mean'] = 0.5 * (sources['322'] + sources['326'])\n    for key in")
    (bundle / 'infer_lite.py').write_text(code)
    cfgpath = bundle / 'configs/model349_lite.yaml'
    cfg = yaml.safe_load(cfgpath.read_text())
    route = load_result('model339_joint_gamma_route_search')['best_image_empty_zero']
    base_keys = list(cfg['route']['base'])
    cfg['route']['base'] = {key: route[key] for key in base_keys}
    cfg['route']['gamma329'] = load_result('model341_model329_joint_route_search')['best_image_empty_zero']['gamma329']
    cfg['route']['gamma336'] = load_result('model346_model336_into_341_route')['raw_best_gamma336']
    cfg['route']['gamma347'] = load_result('model349_model347_into_346_route')['raw_best_gamma347']
    cfg['inference']['b4_fold_ids'] = {name: [1, 2, 3, 4] for name in ('322', '326', '329')}
    cfg['inference']['ema_pruned'] = False
    cfg['validation_image_miou'] = None
    cfgpath.write_text(yaml.safe_dump(cfg, sort_keys=False))
    paths = []
    for name in ('model322_team_b4_group_oof', 'model326_hybrid_fullstats_group_oof',
                 'model329_fullstats_width4000_uniform_group_oof'):
        paths.extend(f'weights/{name}/fold{fold}/last.pt' for fold in (1, 2, 3, 4))
    paths.extend(f'weights/{name}/{file}' for name, file in (
        ('model336_hrnet_best', 'last.pt'), ('model347_hrnet_inverse_ce_finetune', 'last.pt'),
        ('model269_fullstats_1x_extended', 'best.pt'),
        ('restart51_aspp_768_finetune', 'best.pt'), ('model109_aspp_native', 'best.pt'),
        ('restart59_aspp768_seed456', 'best.pt'), ('restart61_pairwise_margin', 'best.pt'),
        ('model123_ema', 'best.pt')))
    assert len(paths) == 20
    rows = []
    for rel in paths:
        candidate = OUTPUTS / Path(*Path(rel).parts[1:])
        assert candidate.is_file(), candidate
        dest = bundle / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.symlink_to(candidate)
        args = candidate.parent / 'args.json'
        if args.exists():
            shutil.copy2(args, dest.parent / 'args.json')
        rows.append({'path': rel, 'source_path': str(candidate),
                     'sha256': sha(candidate), 'bytes': candidate.stat().st_size})
    (bundle / 'checkpoint_manifest.json').write_text(json.dumps(
        {'source':'P100 outputs; original Model349 route under exh14 executable',
         'count':20,'bytes':sum(r['bytes'] for r in rows),'checkpoints':rows},indent=2)+'\n')
    (bundle / 'recipe_manifest.json').write_text(json.dumps({
        'kind':'matched P100 original route control',
        'adapter':'adds 0.5*(322+326) b4mean source required by original route',
        'source_exh14_inference_sha256':sha(SOURCE/'infer_lite.py'),
        'control_inference_sha256':sha(bundle/'infer_lite.py'),
        'original_p100_result':'/workspace/komap_model1/outputs/model349_model347_into_346_route/results.json',
        'independent_generalization_verified':False},indent=2)+'\n')
    (ROOT / 'matched_control_prepared.json').write_text(json.dumps({
        'route':cfg['route'],'inference':cfg['inference'],
        'checkpoint_count':len(rows),'checkpoint_bytes':sum(r['bytes'] for r in rows),
        'checkpoint_sha256':{r['path']:r['sha256'] for r in rows},
        'original_p100_cached_image_miou_percent':100*load_result('model349_model347_into_346_route')['raw_best_valid_image_mean_miou_empty0'],
        'control_inference_sha256':sha(bundle/'infer_lite.py')},indent=2)+'\n')
    print('prepared control',len(rows),sum(r['bytes'] for r in rows))


if __name__=='__main__':
    main()
