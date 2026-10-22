"""Select a fixed-size Model349 ensemble and save all evaluated candidates."""
from __future__ import annotations
import argparse
import csv
from datetime import datetime
from zoneinfo import ZoneInfo
import shutil
from pathlib import Path
from tqdm import tqdm
import yaml
from common import ROOT, REPO, BASE, PALETTE, COMPONENT_DIRS, read_json, write_json, sha
from engine import evaluate, baseline_cache

def records_at(run):
    return [r for p in sorted((run/'completed').glob('*.json'))
            if 'image_miou' in (r:=read_json(p))]

def select_combination(records, evaluate_fn):
    """Start at the best measured candidate; at most two coordinate passes.

    A fine-tuned HRNet candidate carries its parent; changing 336 clears a
    previously paired 347. Thus we do not silently retain stale parent metadata.
    """
    groups={}
    for record in records: groups.setdefault(record['model'],[]).append(record)
    for model in groups:
        groups[model]=sorted(groups[model],key=lambda r:(-r['image_miou'],r['id']))[:2]
    cache={(): evaluate_fn({})}; trace=[]
    def selection(active):
        result={}
        for model in ['123','269','326','329','336','347']:
            if model in active: result.update(active[model]['selection'])
        return result
    def score(active):
        chosen=selection(active); fingerprint=tuple(sorted(chosen.items()))
        if fingerprint not in cache: cache[fingerprint]=evaluate_fn(chosen)
        return cache[fingerprint]
    active={}; best=cache[()]
    # Every individually evaluated candidate remains eligible, even if a
    # subsequent coordinate search settles in a different local optimum.
    for record in sorted(records,key=lambda r:(-r['image_miou'],r['id']))[:1]:
        if record['image_miou']>best:
            active={record['model']:record}; best=score(active)
    for pass_index in range(2):
        improved=False
        for model in ['123','269','336','347','329','326']:
            proposals=[]
            for option in [None]+groups.get(model,[]):
                trial=dict(active); trial.pop(model,None)
                if model=='336' and trial.get('347',{}).get('parent_id'):
                    trial.pop('347')
                if option:
                    trial[model]=option
                    if model=='347' and option.get('parent_id'): trial.pop('336',None)
                proposals.append((score(trial),trial))
            value,trial=max(proposals,key=lambda p:p[0])
            if value>best+1e-12:
                active=trial; best=value; improved=True
            trace.append(dict(pass_index=pass_index,component=model,image_miou=best,
                              selected={k:v['id'] for k,v in active.items()}))
        if not improved: break
    return active,selection(active),best,trace

def export_model(active, directory, score):
    directory.mkdir()
    shutil.copytree(ROOT/'frozen/src',directory/'src')
    shutil.copytree(ROOT/'frozen/configs',directory/'configs')
    shutil.copy2(ROOT/'frozen/infer.py',directory/'infer.py')
    config_path=directory/'configs/model349.yaml'
    config=yaml.safe_load(config_path.read_text())
    config['original_validation_image_miou']=config['validation_image_miou']
    config['validation_image_miou']=score
    config['seed_search_selected']={k:r['id'] for k,r in active.items()}
    config_path.write_text(yaml.safe_dump(config,sort_keys=False,allow_unicode=True))
    # Keep the original palette and fixed route. Score is provenance, not a recipe change.
    replacements={}
    for model in ['123','269','326','329','336','347']:
        if model in active: replacements.update(active[model]['replacements'])
    mappings={}
    for component,paths in replacements.items():
        name=COMPONENT_DIRS[component]
        if component in ['326','329']:
            for fold,path in enumerate(paths,1): mappings[f'weights/{name}/fold{fold}/last.pt']=Path(path)
        else:
            basename='last.pt' if component in ['336','347'] else 'best.pt'
            mappings[f'weights/{name}/{basename}']=Path(paths[0])
    provenance=[]
    for row in tqdm(read_json(ROOT/'weights_manifest.json'),desc='Save 23 selected weights',unit='weight'):
        relative=row['path']; source=mappings.get(relative,BASE/row['original_relative_path'])
        target=directory/relative; target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source,target)
        if '/fold' in relative:
            shutil.copy2(source.parent/'args.json',target.parent/'args.json')
        provenance.append(dict(path=relative,source=str(source),sha256=sha(target)))
    assert len(provenance)==23
    write_json(directory/'selected_weights.json',provenance)
    write_json(directory/'selection.json',dict(image_miou=score,selected={k:r['id'] for k,r in active.items()}))
    (directory/'README.md').write_text('# Selected Model349 seed-search ensemble\n\n'
        f'Validation image mIoU: {score*100:.6f}%. Fixed route; 23 weight files.\n\n'
        'Activate `conda activate komap`, then run `python -B infer.py --help`.\n'
        'This model was selected using validation; its score is not an independent test score.\n')

def finalize(run):
    records=records_at(run)
    print(f'Final selection from {len(records)} complete candidates',flush=True)
    active,selection,score,trace=select_combination(records,lambda s:evaluate(s,quiet=True)['image_miou'])
    date=datetime.now(ZoneInfo('Asia/Seoul')).strftime('%Y%m%d')
    index=1
    while True:
        directory=REPO/'results'/f'model_results_{date}_{index}'
        try: directory.mkdir(); break
        except FileExistsError: index+=1
    evaluated=evaluate(selection,directory,comparisons=True)
    assert abs(evaluated['image_miou']-score)<1e-12
    completed=[read_json(p) for p in (run/'completed').glob('*.json')]
    training=[r for r in completed if r['kind']=='train']
    write_json(directory/'search_trace.json',trace)
    write_json(directory/'search_summary.json',dict(training_planned=100,training_completed=len(training),
               complete_candidates=len(records),selection=selection,run=str(run),image_miou=score,
               selected={k:r['id'] for k,r in active.items()}))
    with (directory/'leaderboard.csv').open('w',newline='') as f:
        writer=csv.writer(f); writer.writerow(['candidate','model','seed_or_seeds','image_miou_percent','delta_pp','parent'])
        for r in sorted(records,key=lambda r:(-r['image_miou'],r['id'])):
            writer.writerow([r['id'],r['model'],r.get('seed',r.get('seeds')),r['image_miou']*100,
                             (r['image_miou']-read_json(baseline_cache()/'baseline/metrics.json')['image_miou'])*100,r.get('parent_id','')])
    (directory/'candidates').mkdir()
    for record in tqdm(records,desc='Preserve candidate reports',unit='candidate'):
        shutil.copytree(record['evaluation'],directory/'candidates'/record['id'])
        write_json(directory/'candidates'/record['id']/'training_record.json',record)
    export_model(active,directory/'selected_model',score)
    delta=(score-read_json(baseline_cache()/'baseline/metrics.json')['image_miou'])*100
    text=(f'# Model349 시드 탐색 결과\n\n계획: 100개 학습 / 완료: {len(training)}개 / '
          f'최종 model349 형태로 평가한 후보: {len(records)}개.\n\n'
          f'최종 image mIoU: **{score*100:.6f}%**; 기존 대비 **{delta:+.6f}%p**.\n\n'
          '지표: 각 이미지의 4개 클래스 IoU 평균을 validation 20장에 대해 평균. 빈 클래스 IoU=0.\n\n'
          '결합 계수와 후처리는 고정했습니다. 동일 검증셋으로 후보를 선택한 점수이며 독립 test 성능은 아닙니다.\n\n'
          f'선택한 후보: `{ {k:r["id"] for k,r in active.items()} }`\n\n'
          '추가 Model123 시드 40개는 현재 라우팅에서 검증 2장의 예측에 영향을 줍니다.\n\n'
          'B4는 완성된 4-fold 세트만 평가했습니다. 중단 시 미완료 작업은 재개할 수 있습니다.\n\n'
          '`selected_model/`: 선택된 23개 가중치와 추론 소스. `leaderboard.csv`: 후보 순위.\n\n'
          '`valid_mask_comparison/`: 좌상 원본, 우상 오류, 좌하 예측, 우하 정답. '
          '맞은 픽셀은 흰색, 틀린 픽셀은 정답 색. 상단에는 image mIoU와 100%−image mIoU 표시.\n\n'
          '`candidates/`: 후보별 보고서와 20장 비교 이미지.\n')
    (directory/f'{directory.name}.md').write_text(text)
    write_json(run/'latest_result.json',dict(path=str(directory),training_completed=len(training),image_miou=score))
    print(f'RESULT_DIRECTORY={directory}',flush=True)
    return directory

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--run',type=Path,required=True)
    finalize(p.parse_args().run.resolve())
