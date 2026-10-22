"""Eight independent r70 packages, strict hashes, fresh inference and image-first reports."""
import argparse, copy, os, shutil, subprocess, sys
from pathlib import Path
import numpy as np
from priority_common import (ROOT, REPO, R70, SEED_RUN, IDS, PREVIEW, read, write, sha,
                             verify_files, files_under, new_attempt, environment, require_komap)

def load_report_tools(implementation):
    sys.path.insert(0, str(implementation))
    from verify_model import verify
    from export_results import score_variant
    from report_utils import allocate_result
    return verify, score_variant, allocate_result

def replace_hr336(source, destination, candidate, verify):
    """Copy first: editing manifests or weights cannot change an existing package."""
    verify(source)
    key = 'weights/model336_hrnet_best/last.pt'
    assert candidate['group'] == '336' and set(candidate['context']['probabilities']) == {'336'}
    assert set(candidate['context']['weights']) == {'R/'+key}
    checkpoint = Path(candidate['checkpoint'])
    assert candidate['context']['weights']['R/'+key] == str(checkpoint)
    assert sha(checkpoint) == candidate['artifacts'][str(checkpoint)]
    before = read(source/'compact/assets_manifest.json')
    assert before['count'] == 17
    assert sum('_full70/' in r['path'] for r in before['checkpoints'] if r['namespace']=='R') == 3
    assert not any('/fold' in r['path'] for r in before['checkpoints'] if r['namespace']=='R')
    shutil.copytree(source, destination)
    target = destination/'Model349_exh14'/key
    shutil.copy2(checkpoint, target)
    selected = copy.deepcopy(before)
    changes = 0
    for row in selected['checkpoints']:
        if row['namespace'] == 'R' and row['path'] == key:
            row.update(sha256=sha(target), bytes=target.stat().st_size, source_checkpoint=str(checkpoint))
            changes += 1
    assert changes == 1
    selected['bytes'] = sum(r['bytes'] for r in selected['checkpoints'])
    write(destination/'compact/assets_manifest.json', selected)
    release = destination/'Model349_exh14'
    rmanifest = read(release/'checkpoint_manifest.json')
    rmanifest['checkpoints'] = [{k:v for k,v in r.items() if k!='namespace'}
                                for r in selected['checkpoints'] if r['namespace']=='R']
    rmanifest['bytes'] = sum(r['bytes'] for r in rmanifest['checkpoints'])
    write(release/'checkpoint_manifest.json', rmanifest)
    recipe = read(release/'recipe_manifest.json')
    recipe['files_sha256']['checkpoint_manifest.json'] = sha(release/'checkpoint_manifest.json')
    write(release/'recipe_manifest.json', recipe)
    write(destination/'seed_replacement.json', dict(id=candidate['id'], seed=candidate['seed'],
        checkpoint=str(checkpoint), checkpoint_sha256=sha(checkpoint),
        base_package=str(source), changed_checkpoint='R/'+key, fixed_checkpoint_count=16))
    (destination/'README.md').write_text(
        f'# lite20 r70 + HR336 seed {candidate["seed"]}\n\n'
        '17 checkpoints: full-train70 B4 322/326/329; only HR336 differs from the r70 package.\n'
        'Verify: `python -B verify_model.py` in conda komap.\n'
        'Inference: `python -B compact/infer_compact.py --config compact/config.json '
        '--parent-package . --assets assets --input /path/to/images --output /new/output --gpus 4,5,6,7`\n')
    manifest = read(destination/'model_manifest.json')
    manifest['variant'] = f'lite20_r70_HR336_seed{candidate["seed"]}'
    manifest['files_sha256'] = {str(p.relative_to(destination)): sha(p) for p in sorted(destination.rglob('*'))
                                if p.is_file() and p != destination/'model_manifest.json'}
    write(destination/'model_manifest.json', manifest)
    verification = verify(destination)
    old = {(r['namespace'], r['path']):r['sha256'] for r in before['checkpoints']}
    fixed = [r for r in selected['checkpoints'] if (r['namespace'], r['path']) != ('R',key)]
    assert len(fixed) == 16 and all(old[(r['namespace'],r['path'])] == r['sha256'] for r in fixed)
    return verification

def evaluate(work, identifier):
    require_komap()
    if os.environ.get('PRIORITY_USER_STARTED') != '1': raise RuntimeError('Use the user-started priority launcher.')
    import torch
    assert os.environ['KOMAP_PHYSICAL_GPU'] in ['4','5','6','7'] and torch.cuda.device_count() == 1
    state = read(work/'state.json'); r70_run = Path(state['r70_run'])
    verify, score_variant, _ = load_report_tools(r70_run/'implementation')
    candidate = read(SEED_RUN/f'completed/{identifier}.json')
    assert identifier in IDS and candidate['seed'] == 3407+IDS.index(identifier)
    verify_files(candidate['artifacts'])
    source = Path(read(r70_run/'completed/package.json')['models']['lite20_r70'])
    attempt = new_attempt(work/'seed_evaluations'/identifier)
    package = attempt/'model'
    replace_hr336(source, package, candidate, verify)
    inference = attempt/'inference'
    command = [sys.executable, '-B', '-u', str(package/'compact/infer_compact.py'),
               '--config', str(package/'compact/config.json'), '--parent-package', str(package),
               '--assets', str(package/'assets'), '--input', str(r70_run/'implementation/data/valid/images'),
               '--output', str(inference), '--gpus', os.environ['CUDA_VISIBLE_DEVICES']]
    write(attempt/'command.json', command)
    subprocess.run(command, check=True, env=environment(), cwd=attempt)
    assert read(inference/'completion.json')['gt_read'] is False
    assert read(inference/'completion.json')['allow_retrained'] is False
    metrics = score_variant(inference, attempt/'scored')
    result = dict(id=identifier, seed=candidate['seed'], package=str(package), inference=str(inference),
                  scored=str(attempt/'scored'), metrics=metrics, original_REF=candidate['image_miou'],
                  attempt=str(attempt), physical_gpu=int(os.environ['KOMAP_PHYSICAL_GPU']))
    result['files_sha256'] = files_under(attempt)
    write(work/f'evaluated/{identifier}.json', result)
    print(f'EVALUATED {identifier}: R={metrics["R"]["image_miou"]:.9f}, REF={metrics["stage2"]["image_miou"]:.9f}', flush=True)

def export(work, *, results_root=REPO/'results'):
    state = read(work/'state.json'); r70_run = Path(state['r70_run'])
    verify, _, allocate_result = load_report_tools(r70_run/'implementation')
    rows = [read(work/f'evaluated/{identifier}.json') for identifier in IDS]
    for row in rows: verify_files(row['files_sha256'], Path(row['attempt']))
    preview = read(PREVIEW)
    assert preview['run'] == str(SEED_RUN) and preview['count'] == 20 and preview['baseline_pixel_exact']
    old = {r['id']:r for r in preview['candidates']}
    for row in rows:
        previous = old[row['id']]
        assert previous['seed'] == row['seed'] and abs(previous['REF_image_miou']-row['original_REF']) < 1e-12
        row['previous_R'] = previous['R_image_miou']
        row['delta_R_pp'] = 100*(row['metrics']['R']['image_miou']-row['previous_R'])
        row['delta_REF_pp'] = 100*(row['metrics']['stage2']['image_miou']-row['original_REF'])
    rows.sort(key=lambda r:(-r['metrics']['stage2']['image_miou'], r['seed']))
    best = rows[0]; best_r = max(rows, key=lambda r:r['metrics']['R']['image_miou'])
    result = allocate_result(Path(results_root))
    (result/'INCOMPLETE').write_text('Export in progress. Only complete.json certifies a finished report.\n')
    for row in rows:
        shutil.copytree(row['scored'], result/f'seeds/seed_{row["seed"]}')
        write(result/f'seeds/seed_{row["seed"]}/model_reference.json', dict(
            package=row['package'], manifest_sha256=sha(Path(row['package'])/'model_manifest.json'),
            seed_checkpoint=read(SEED_RUN/f'completed/{row["id"]}.json')['checkpoint']))
    for directory in ['valid_mask_comparison', 'predicted_labels', 'predicted_masks']:
        shutil.copytree(Path(best['scored'])/'stage2'/directory, result/directory)
    for name in ['metrics.json', 'per_image.csv']:
        shutil.copy2(Path(best['scored'])/'stage2'/name, result/name)
    shutil.copytree(best['package'], result/'selected_model'); verify(result/'selected_model')
    shutil.copytree(work/'implementation', result/'source_snapshot')
    shutil.copy2(PREVIEW, result/'previous_8_seed_R_metrics.json')
    write(result/'summary.json', dict(seed_run=str(SEED_RUN), r70_run=str(r70_run),
          best_REF_seed=best['seed'], best_R_seed=best_r['seed'],
          candidates=[{k:v for k,v in r.items() if k!='files_sha256'} for r in rows]))
    lines = ['# Model349 lite20 r70: 기존 8개 시드 B4 교체 평가', '',
        f'R+E+F 최고: **seed {best["seed"]}, {best["metrics"]["stage2"]["image_miou"]*100:.6f}%**.',
        f'R 단독 최고: **seed {best_r["seed"]}, {best_r["metrics"]["R"]["image_miou"]*100:.6f}%**.', '',
        '기존 HR336 시드 3407~3414 각각에 동일한 새 full-train70 B4 322/326/329를 적용했습니다.',
        'R의 B4 fold 가중치 6개 대신 full70 가중치 3개를 사용합니다. E의 B4 fold 구성은 그대로 유지합니다.',
        '각 패키지는 R8+E8+F1=17개입니다. r70 기준에서 HR336 하나만 시드별로 교체했으며 나머지 16개 SHA256은 일치합니다.',
        '시드별로 원본 valid20 이미지부터 R/E/F 전체를 새로 추론했습니다. 추가 학습이나 시드별 B4 재학습은 없습니다.',
        '지표: 이미지별 네 클래스 IoU를 먼저 평균하고 valid20 평균. union=0은 IoU=0. global mIoU는 계산하지 않습니다.', '',
        '| Seed | 기존 fold R (%) | 새 r70 R (%) | R 차이 (%p) | 기존 fold R+E+F (%) | 새 r70 R+E+F (%) | 최종 차이 (%p) |',
        '|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f'| {r["seed"]} | {r["previous_R"]*100:.6f} | {r["metrics"]["R"]["image_miou"]*100:.6f} | {r["delta_R_pp"]:+.6f} | {r["original_REF"]*100:.6f} | {r["metrics"]["stage2"]["image_miou"]*100:.6f} | {r["delta_REF_pp"]:+.6f} |')
    lines += ['', '같은 valid20에서 비교·순위를 매긴 탐색 결과입니다. 독립 test 성능으로 해석하지 않습니다.',
        '`valid_mask_comparison/`은 최종 최고 시드 20장입니다. 각 시드의 R/R+E/R+E+F 결과와 2×2 비교 그림은 `seeds/seed_번호/` 아래에 있습니다.',
        '그림: 좌상 원본, 우상 정답색 오류(정답 픽셀 흰색), 좌하 예측, 우하 정답. image mIoU와 100%-image mIoU를 표시합니다.',
        '`selected_model/`은 R+E+F 최고 조합의 17개 가중치와 추론 코드입니다. 8개 전체 모델은 summary.json의 package 경로에 보존됩니다.',
        '이 비교로 원래 100개 시드 탐색의 기준 모델이나 계획은 변경하지 않습니다. 후속 단계에서 원래 run을 완료된 작업부터 재개합니다.',
        f'r70 학습 run: `{r70_run}`', f'원래 시드 탐색 run: `{SEED_RUN}`', '']
    (result/(result.name+'.md')).write_text('\n'.join(lines))
    (result/'INCOMPLETE').unlink()
    write(result/'complete.json', dict(ok=True, candidates=8, valid_images=20,
          panels=500, best_REF_seed=best['seed'], best_R_seed=best_r['seed'], workflow=str(work)))
    receipt = dict(result=str(result), files_sha256=files_under(result))
    write(work/'evaluation_export.json', receipt)
    print(f'EIGHT-SEED RESULT={result}', flush=True)
    return result

def main():
    p = argparse.ArgumentParser(); p.add_argument('--work', required=True, type=Path)
    p.add_argument('--task', choices=IDS); p.add_argument('--export', action='store_true'); a=p.parse_args()
    require_komap()
    if a.export: export(a.work)
    elif a.task: evaluate(a.work, a.task)
    else: p.error('Specify --task or --export')

if __name__ == '__main__': main()
