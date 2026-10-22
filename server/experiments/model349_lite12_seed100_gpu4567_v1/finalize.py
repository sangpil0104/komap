"""Compare complete fixed-route candidates and preserve a twelve-weight model."""
from __future__ import annotations
import argparse
import csv
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path
import shutil
from tqdm import tqdm
import yaml
from common import ROOT, REPO, MODELS, read_json, write_json, sha
from engine import evaluate

def select_combination(records,evaluate_fn):
    groups = {model:sorted([r for r in records if r['model']==model],
                          key=lambda r:(-r['image_miou'],r['id']))[:2] for model in MODELS}
    cache = {}; trace = []
    def selection(active):
        return {k:v for record in active.values() for k,v in record['selection'].items()}
    def score(active):
        chosen = selection(active); key = tuple(sorted(chosen.items()))
        if key not in cache: cache[key] = evaluate_fn(chosen)
        return cache[key]
    active = {}; best = score(active)
    for record in sorted(records,key=lambda r:(-r['image_miou'],r['id']))[:1]:
        trial = {record['model']:record}; value = score(trial)
        if value>best: active,best = trial,value
    for iteration in range(2):
        improved = False
        for model in MODELS:
            for option in [None]+groups[model]:
                trial = dict(active); trial.pop(model,None)
                if option: trial[model] = option
                value = score(trial)
                if value>best+1e-12: active,best,improved = trial,value,True
            trace.append(dict(iteration=iteration,component=model,image_miou=best,
                              selected={k:v['id'] for k,v in active.items()}))
        if not improved: break
    return active,selection(active),best,trace

def export_model(active,directory,score):
    directory.mkdir()
    for name in ['src','configs']:
        shutil.copytree(ROOT/'frozen'/name,directory/name)
    shutil.copy2(ROOT/'frozen/infer_lite.py',directory/'infer_lite.py')
    replacements = {k:v for record in active.values() for k,v in record['replacements'].items()}
    rows = []
    for row in tqdm(read_json(ROOT/'local_weights.json')['checkpoints'],desc='Save 12 selected weights',unit='weight'):
        relative = row['relative']; source = Path(replacements.get(relative,ROOT/relative))
        target = directory/relative; target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source,target)
        if '/fold' in relative:
            shutil.copy2(ROOT/'frozen'/Path(relative).parent/'args.json',target.parent/'args.json')
        rows.append(dict(path=relative,source=str(source),sha256=sha(target),bytes=target.stat().st_size))
    if len(rows)!=12: raise AssertionError('Selected model must contain 12 checkpoints')
    write_json(directory/'selected_weights.json',dict(count=12,checkpoints=rows))
    write_json(directory/'selection.json',dict(image_miou=score,selected={k:r['id'] for k,r in active.items()}))
    # Configuration and inference code are copied byte-for-byte from upstream.
    # Measured scores belong in selection.json, never in the frozen recipe.
    (directory/'README.md').write_text('# Selected Model349 lite12 FP16\n\n'
        f'Validation image mIoU: {score*100:.6f}%. Twelve checkpoints; fixed GitHub lite12 inference.\n\n'
        'Use conda komap. Example: `python -B infer_lite.py --input /path/to/images '
        '--gt /path/to/masks --output /path/to/new_output --gpus 4`.\n\n'
        'Selected on validation; not an independent test score. Local starting checkpoints differ from upstream published hashes.\n')

def finalize(run):
    completed = [read_json(p) for p in (run/'completed').glob('*.json')]
    training = [r for r in completed if r['kind']=='train']
    records = [r for r in completed if r['kind'] in ['train','ensemble'] and 'image_miou' in r]
    baseline = read_json(run/'completed/baseline_score.json')['image_miou']
    print(f'Final fixed-route selection: {len(records)} candidates',flush=True)
    max_evaluations = 1 + min(1,len(records)) + 2*sum(1+min(2,sum(r['model']==m for r in records)) for m in MODELS)
    with tqdm(total=max_evaluations,desc='Evaluate combinations (upper bound)',unit='combination',mininterval=5) as progress:
        def score_fn(selection):
            value = evaluate(run,selection,quiet=True)['image_miou']; progress.update(); return value
        active,selection,score,trace = select_combination(records,score_fn)
        progress.total = progress.n; progress.refresh()
    date = datetime.now(ZoneInfo('Asia/Seoul')).strftime('%Y%m%d'); index = 1
    while True:
        directory = REPO/'results'/f'model_results_{date}_{index}'
        try: directory.mkdir(); break
        except FileExistsError: index += 1
    measured = evaluate(run,selection,directory,comparisons=True)
    if abs(measured['image_miou']-score)>1e-12: raise AssertionError('Final score mismatch')
    export_model(active,directory/'selected_model',score)
    shutil.copytree(read_json(run/'completed/baseline_score.json')['evaluation'],directory/'baseline')
    for record in tqdm(records,desc='Preserve candidate comparisons',unit='candidate'):
        destination = directory/'candidates'/record['id']
        shutil.copytree(record['evaluation'],destination)
        write_json(destination/'training_record.json',record)
    with (directory/'leaderboard.csv').open('w',newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['candidate','model','seed_or_seeds','image_miou_percent','delta_from_local_baseline_pp'])
        for record in sorted(records,key=lambda r:(-r['image_miou'],r['id'])):
            writer.writerow([record['id'],record['model'],record.get('seed',record.get('seeds')),
                             record['image_miou']*100,(record['image_miou']-baseline)*100])
    write_json(directory/'search_trace.json',trace)
    summary = dict(training_planned=100,training_completed=len(training),candidates=len(records),
                   image_miou=score,local_baseline_image_miou=baseline,run=str(run),
                   selected={k:r['id'] for k,r in active.items()},checkpoint_count=12,
                   upstream_score_is_not_local_baseline=True)
    write_json(directory/'search_summary.json',summary)
    shutil.copy2(ROOT/'provenance.json',directory/'provenance.json')
    implementation = run/'implementation'
    shutil.copytree(implementation,directory/'source_snapshot',
                    ignore=lambda path,names: [n for n in names if n=='__pycache__' or
                                               (Path(path)==implementation and n=='weights')])
    report = (f'# Model349 lite12 FP16 시드 탐색 결과\n\n'
        f'학습 {len(training)}/100개, 최종 lite12 후보 {len(records)}/84개.\n\n'
        f'최종 image mIoU: **{100*score:.6f}%**, 로컬 기준: **{100*baseline:.6f}%**, '
        f'차이 **{100*(score-baseline):+.6f}%p**.\n\n'
        '검증 20장 각각 4-class IoU 평균을 구하고 20장에 동일 비중으로 평균합니다. 빈 union=0.\n\n'
        'GitHub lite12의 326 fold1·3 / 329 fold1·2·3·4 / HRNet336 / Native269 / 고해상도 4개, '
        '결합 계수·후처리·TENT5를 고정했습니다. 선택 모델도 FP16 저장 가중치 12개입니다.\n\n'
        'GitHub 가중치는 제공되지 않아 로컬 보존 가중치의 FP16 사본에서 비교했습니다. '
        'GitHub 기록 80.1255610%를 로컬 기준 점수로 사용하지 않았습니다. '
        '검증셋으로 시드를 선택했으므로 독립 test 성능이 아닙니다.\n\n'
        f'선택: `{summary["selected"]}`\n\n'
        '`valid_mask_comparison/`: 좌상 원본 / 우상 오류 / 좌하 예측 / 우하 정답. '
        '맞은 픽셀은 흰색, 틀린 픽셀은 정답 색이며 상단에 image mIoU와 100%−image mIoU를 표시합니다.\n\n'
        '`candidates/`: 84개 후보별 보고서·비교 이미지(완료된 후보만). '
        '`selected_model/`: 선택된 12개 가중치·추론 코드. `source_snapshot/`: 학습·실행 코드.\n')
    (directory/f'{directory.name}.md').write_text(report)
    write_json(directory/'complete.json',summary)
    write_json(run/'latest_result.json',dict(path=str(directory),training_completed=len(training),image_miou=score))
    print(f'RESULT_DIRECTORY={directory}',flush=True)
    return directory

if __name__=='__main__':
    p = argparse.ArgumentParser(); p.add_argument('--run',type=Path,required=True)
    finalize(p.parse_args().run.resolve())
