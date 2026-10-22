"""Add the three fixed R B4-329 weights, preserving the prior 56 models."""
import csv
import hashlib
import io
import json
import zipfile
from pathlib import Path

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'repository'
RESULT=Path('/Data3/sppark/komap/results/model_results_20261012_1')
MODEL=RESULT/'selected_model'
ASSETS=json.loads((MODEL/'lite20/selected_assets_lite20.json').read_text())
COMPLETE=json.loads((RESULT/'complete.json').read_text())
SEARCH=Path(COMPLETE['run'])
BASE=Path(json.loads((SEARCH/'plan.json').read_text())['baseline'])
RELATIVE='models/B4-329/329_fixed_seed0_folds134.zip'
PREFIX='329_fixed_seed0_folds134'


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def dump(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')


def main():
    existing=json.loads((OUT/'manifest.json').read_text())
    assert existing['checkpoint_count']==56
    assert RELATIVE not in existing['files']
    selected=[r for r in ASSETS['checkpoints'] if r['namespace']=='R' and 'model329_' in r['path']]
    assert len(selected)==3
    metadata={r['namespace']+'/'+r['path']:r for r in ASSETS['metadata']}
    tasks={t['id']:t for t in json.loads((BASE/'plan.json').read_text())['tasks']}
    files={};hashes={};rows=[];text_members={}
    for row in selected:
        path=MODEL/'Model349_exh14'/row['path'];fold=int(path.parent.name.removeprefix('fold'))
        assert fold in [1,3,4] and sha(path)==row['sha256']
        args=path.with_name('args.json');params=json.loads(args.read_text())
        assert params['seed']==0 and params['iters']==16000 and params['encoder']=='efficientnet-b4'
        args_key='R/'+str(Path(row['path']).with_name('args.json'))
        assert sha(args)==metadata[args_key]['sha256']
        task_id=f'329_fixed_fold{fold}'
        for source in [path,args]:
            member=f'{PREFIX}/fold{fold}/{source.name}';files[member]=source;hashes[member]=sha(source)
        recipe=tasks[f'train_expert_b4_fold{fold}']
        text_members[f'{PREFIX}/fold{fold}/original_training_recipe.json']=(json.dumps(recipe,ensure_ascii=False,indent=2)+'\n').encode()
        rows.append(dict(family='B4-329',group='329',candidate_id='329_fixed_seed0',task_id=task_id,
            training_seed=0,fold=fold,hr_slot=None,archive=RELATIVE,member=f'{PREFIX}/fold{fold}/last.pt',
            checkpoint_sha256=row['sha256'],checkpoint_bytes=row['bytes'],deployed_weight_key='R/'+row['path'],
            source_checkpoint=str(path),candidate_ensemble_image_miou=None,
            provenance='Exact checkpoint from the final deployed package; held fixed throughout seed100'))
    rows.sort(key=lambda r:r['fold'])
    info=dict(id='329_fixed_seed0',family='B4-329',group='329',kind='fixed_component',archive=RELATIVE,
        training_seed=0,folds=[1,3,4],checkpoint_count=3,checkpoints=rows,
        image_miou=None,standalone_image_miou=None,
        score_scope='No standalone or seed-comparison score measured for B4-329; fixed in all seed100 candidates',
        ensemble_context_scores=dict(baseline_REF_image_miou=COMPLETE['baseline_image_miou'],
                                     final_REF_image_miou=COMPLETE['image_miou']),
        source_result=str(RESULT),source_baseline_run=str(BASE),
        source_checkpoint_manifest_sha256=sha(MODEL/'lite20/selected_assets_lite20.json'),
        no_seed_search_for_this_component=True)
    text_members[f'{PREFIX}/manifest.json']=(json.dumps(info,ensure_ascii=False,indent=2)+'\n').encode()
    text_members[f'{PREFIX}/SHA256SUMS.txt']=''.join(f'{v}  {k.removeprefix(PREFIX+"/")}\n' for k,v in hashes.items()).encode()
    text_members[f'{PREFIX}/README.md']='''# Model349 lite20 — R B4-329 (seed0, folds 1/3/4)

model_results_20261012_1의 최종 모델이 사용하는 B4-329 가중치 3개입니다.
각 fold의 last.pt 및 args.json은 최종 패키지의 원본 파일과 SHA256이 일치합니다.
기존 train fold 모델을 100회 시드 탐색 내내 고정해 사용했습니다.

UNet / efficientnet-b4 / zcdf3 / image_global / target_width4000 / uniform.
각 fold: seed0, 16,000 iterations, crop1536, batch2, lr3e-4, weight_decay1e-2, CE+Dice.
배치 위치: Model349_exh14/weights/model329_fullstats_width4000_uniform_group_oof/fold{1,3,4}/.
폴더별 last.pt와 args.json을 함께 사용하세요.

B4-329 단독 또는 seed 비교 점수는 측정하지 않았습니다.
이 가중치를 고정 사용한 전체 R+E+F의 기준 점수는 80.255846%, 최종 조합은 80.479068%입니다.
두 수치는 전체 앙상블의 valid20 image mIoU입니다.
각 이미지의 4개 클래스 IoU 평균 후 valid20 평균, empty union=0입니다.

original_training_recipe.json은 원래 학습 계획입니다. 경로 템플릿은 원본 서버 경로 기준입니다.
해제한 이 폴더에서 sha256sum -c SHA256SUMS.txt로 확인할 수 있습니다.
'''.encode()
    destination=OUT/RELATIVE;destination.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(destination,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=1) as archive:
        for name,source in files.items():archive.write(source,name)
        for name,data in text_members.items():archive.writestr(name,data)
    with zipfile.ZipFile(destination) as archive:
        for name,expected in hashes.items():
            with archive.open(name) as stream:assert hashlib.file_digest(stream,'sha256').hexdigest()==expected
        for name,expected in text_members.items():assert archive.read(name)==expected
    info.update(archive_sha256=sha(destination),archive_bytes=destination.stat().st_size,archive_verified=True)
    dump(OUT/'metadata/additional_models.json',[info])
    saved=json.loads((OUT/'metadata/checkpoints.json').read_text());assert len(saved)==56
    saved.extend(rows);dump(OUT/'metadata/checkpoints.json',saved)
    csv_path=OUT/'metadata/checkpoints.csv'
    with csv_path.open(newline='') as stream:fields=csv.DictReader(stream).fieldnames
    buffer=io.StringIO();writer=csv.DictWriter(buffer,fieldnames=fields,lineterminator='\n');writer.writeheader()
    writer.writerows({k:r.get(k) for k in fields} for r in saved);csv_path.write_text(buffer.getvalue())
    existing.update(checkpoint_count=59,candidate_count=28,model_archive_count=29,
        additional_fixed_checkpoint_count=3,additional_fixed_groups={'B4-329':3})
    existing['files'][RELATIVE]=dict(bytes=info['archive_bytes'],sha256=info['archive_sha256'])
    dump(OUT/'manifest.json',existing)
    p=OUT/'SHA256SUMS.txt';p.write_text(p.read_text()+f'{info["archive_sha256"]}  {RELATIVE}\n')
    readme=OUT/'README.md';text=readme.read_text()
    text=text.replace('# Model349 lite20 — 5개 계열의 시드 탐색 모델 전달','# Model349 lite20 — 6개 계열 모델 전달')
    text=text.replace('총 **56개 가중치**를 모두 담았습니다.','총 **56개 가중치**를 담았습니다. 여기에 최종 모델의 고정 **B4-329 3개**를 추가해 현재 **총 59개 가중치 / 29개 모델 ZIP**입니다.')
    text=text.replace('평가 단위에 맞춘 **28개 ZIP**이며, 각 ZIP은','시드 탐색 모델은 평가 단위에 맞춘 **28개 ZIP**이며, 각 ZIP은')
    text=text.replace('이 저장소의 5개 계열만으로','이 저장소의 6개 계열만으로')
    section=f'''## 추가된 B4-329 — 고정 seed0, fold 1·3·4

[B4-329 가중치 3개 ZIP 다운로드]({RELATIVE}) — 약 {info['archive_bytes']/1e6:.1f} MB.

| 모델 | 가중치 수 | fold | 학습 seed | 학습 길이 | 역할 |
|---|---:|---|---:|---|---|
| R B4-329 | 3 | 1, 3, 4 | 0 | 각 16,000 iterations | 100회 시드 탐색에서 고정 사용 |

`model_results_20261012_1/selected_model`의 실제 배포 가중치와 `args.json`을 그대로 담았으며 SHA256을 검증했습니다.
전처리는 zcdf3 / image_global / target_width4000 / uniform이고, 모델은 UNet / efficientnet-b4입니다.
적용 경로는 `Model349_exh14/weights/model329_fullstats_width4000_uniform_group_oof/fold{{1,3,4}}/last.pt`입니다.

**이 B4-329에 대해 별도의 단독 성능 또는 시드별 최고 점수는 측정하지 않았습니다.**
기준 전체 R+E+F의 80.255846%와 최종 조합의 80.479068%는 이 가중치를 고정 사용한 앙상블 점수입니다.
이번 파일은 기존 R의 fold 가중치이며, E B4 warmstart 및 별도 r70 full-train70 B4-329와 구분됩니다.
100회 탐색의 후보 수는 28개로 그대로이고, 고정 모델 ZIP 1개가 추가됐습니다.

B4-329만 받으려면 저장소에서 다음을 실행하세요.

```bash
git lfs pull --include="{RELATIVE}"
python verify_download.py --family B4-329 --deep
```

'''
    assert '## 다운로드와 검증' in text;text=text.replace('## 다운로드와 검증',section+'## 다운로드와 검증',1)
    total=sum(r['bytes'] for r in existing['files'].values())/1024**3
    text=text.replace('약 **7.89 GiB**',f'약 **{total:.2f} GiB**')
    text=text.replace('`metadata/checkpoints.csv/json`: 56개 가중치','`metadata/checkpoints.csv/json`: 59개 가중치')
    text=text.replace('- `metadata/candidates/`: 후보별 valid20 이미지 점수와 출처.',
        '- `metadata/candidates/`: 후보별 valid20 이미지 점수와 출처.\n- `metadata/additional_models.json`: 추가한 고정 B4-329 가중치 3개와 앙상블 문맥 정보.')
    readme.write_text(text)
    dump(ROOT/'build_verified.json',info)
    print(json.dumps(dict(ok=True,archive=RELATIVE,bytes=info['archive_bytes'],sha256=info['archive_sha256'],folds=[1,3,4],checkpoint_count=3,total_repository_checkpoints=59),indent=2))


if __name__=='__main__':main()
