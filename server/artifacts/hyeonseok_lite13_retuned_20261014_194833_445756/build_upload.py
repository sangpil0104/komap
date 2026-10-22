"""Package exact selected weights and source; append a complete ensemble to the handoff repo."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import csv,difflib,hashlib,json,re,shutil,subprocess,zipfile
from tqdm import tqdm

ROOT=Path(__file__).resolve().parent;REPO=ROOT/'repository';TASK=json.loads((ROOT/'task.json').read_text())
RESULT=Path(TASK['source_result']);MODEL=RESULT/'selected_model'
PREFIX='Model349_lite13_retuned_retrained_20261014';ARCHIVE=TASK['archive']

def read(p):return json.loads(Path(p).read_text())
def write(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def main():
    parent=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()
    complete=read(RESULT/'complete.json');assert complete['ok'] and complete['weights']==13
    metrics=read(RESULT/'metrics.json');times=read(RESULT/'inference_timing.json')
    assert metrics['count']==20 and abs(sum(x['image_miou'] for x in metrics['images'])/20-metrics['image_miou'])<1e-12
    assert metrics['image_miou']==complete['image_miou']
    baseline_manifest=read(REPO/'manifest.json');baseline_checkpoints=read(REPO/'metadata/checkpoints.json')
    files=read(MODEL/'files_sha256.json');preservation={}
    for name,digest in tqdm(files.items(),desc='Verify selected model',unit='file',mininterval=5):
        assert sha(MODEL/name)==digest,name;preservation[str(MODEL/name)]=digest
    assert len([name for name in files if name.startswith('weights/') and name.endswith('.pt')])==13
    for split,count in [('valid',20),('test',10)]:
        assert times[split]['images']==count and times[split]['inventory_verified']
        assert abs(times[split]['seconds_per_image']-times[split]['wall_seconds']/count)<1e-12
    stage_rows=[read(p) for p in (RESULT/'stage_receipts').glob('*.json')]
    training={r['artifact_sha256']:r for r in stage_rows if r.get('artifact','').endswith('.pt')}
    assert len(training)==33
    checkpoints=[]
    for p in sorted((MODEL/'weights').rglob('*.pt')):
        rel=str(p.relative_to(MODEL));digest=files[rel];receipt=training[digest];cmd=receipt['command']
        seed=int(cmd[cmd.index('--seed')+1]) if '--seed' in cmd else 81349 if 'refiner/' in rel else None
        assert seed is not None
        match=re.search(r'/fold(\d+)/',rel);fold=int(match.group(1)) if match else None
        checkpoints.append(dict(family='Lite13-retuned',group=rel.split('/')[1],candidate_id='lite13_retuned_20261014',
            task_id='lite13_'+receipt['name'],training_seed=seed,fold=fold,hr_slot=None,archive=ARCHIVE,member=PREFIX+'/'+rel,
            checkpoint_sha256=digest,checkpoint_bytes=p.stat().st_size,deployed_weight_key=rel.removeprefix('weights/'),
            source_checkpoint=str(p),original_training_checkpoint=receipt['artifact'],candidate_ensemble_image_miou=metrics['image_miou']))
    write(ROOT/'originals_sha256.json',preservation)
    extras=ROOT/'extras';extras.mkdir();shutil.copy2(ROOT/'predict_portable.py',extras/'predict_portable.py')
    reportdir=REPO/'reports/model_results_20261014_1';reportdir.mkdir(parents=True)
    reports=['model_results_20261014_1.md','metrics.json','per_image.csv','inference_timing.json','inference_timing.csv','environment.freeze.txt','complete.json','pipeline_completion.json']
    for name in reports:shutil.copy2(RESULT/name,reportdir/name)
    portable_readme=f'''# Model349 lite13 retuned — 2026-10-14 학습 모델

전체 앙상블 valid20 image mIoU **{metrics['image_miou']*100:.6f}%**. R7 + E5 + F1 = **13개 가중치**입니다.
선행/OOF를 포함한 33개 학습 작업에서 생성했으며 이 패키지는 최종 추론에 필요한 13개를 포함합니다.

| 데이터 | 전체 추론시간 | 장당 평균 |
|---|---:|---:|
| valid 20장 | {times['valid']['wall_seconds']:.3f}초 | {times['valid']['seconds_per_image']:.3f}초 |
| test 10장 | {times['test']['wall_seconds']:.3f}초 | {times['test']['seconds_per_image']:.3f}초 |

RTX 4090 GPU 3~6에서 R 구성요소를 분배하고 E는 GPU 3, F는 CPU에서 실행했습니다.
전체 시간은 프로세스 준비·모델 로딩·전처리·R(TTA/TENT)/E/F·후처리·출력 저장·임시 정리를 포함합니다.
학습, 정답 채점과 보고서 생성은 제외했습니다. 별도 warm-up 없이 split마다 1회 측정했습니다.
장당 값은 전체 시간/장수이며 단일 이미지 지연시간이 아닙니다.

## 실행

기존 `komap` conda 환경을 활성화하세요. 검증 환경은 Python {{python_version}},
PyTorch 2.7.1+cu128 / torchvision 0.22.1+cu128 / SMP 0.5.0 / timm 1.0.29입니다.
`results/environment.freeze.txt`는 이번 실험 환경입니다. `code/environment.yml`은 upstream 환경(Pytorch 2.8) 기록입니다.

```bash
conda activate komap
python predict_portable.py --verify-only
python predict_portable.py --images /path/to/images --output /path/to/new_output --gpus 0
# 4개 GPU 사용 예:
python predict_portable.py --images /path/to/images --output /path/to/another_output --gpus 3,4,5,6
```

`fusion/predicted_labels/이름.png`와 `fusion/predicted_masks/이름_mask.png`에 예측을 저장합니다.
`predict_portable.py`는 이번 가중치와 코드를 원본 manifest로 검증한 후 실행합니다.
보존된 원래 `predict.sh`에는 학습 서버의 conda 경로가 있으므로 다른 서버에서는 위 portable 실행기를 사용하세요.

## 구성과 지표

R: restart61, model269, B4-326 fold2, B4-329 fold1·3·4, HRNet347.
E: B4 warmstart fold1·3, HRNet414 N3, HRNet392 seed1 fold4, HRNet392 train70 seed5. F: 새 OOF로 학습한 보정기.
R `al_gamma=1`, `al3ni_gamma=1`, `eut_edge_blend=0.125`; E 비율 `0.4 / 0.42 / 0.18`입니다.
각 모델의 seed/fold/해시는 `model_manifest.json`에 있습니다.

image mIoU는 이미지별 네 클래스 IoU 평균을 20장에 대해 평균하며 빈 클래스 IoU는 0입니다.
개별 구성 모델의 단독 점수는 측정하지 않았습니다. test10은 정답이 없어 점수가 없습니다.
일부 R checkpoint 선택과 원본 결합 비율 조정에 valid20이 사용되므로 독립 test 성능으로 해석하지 않습니다.
공개 80.779378%는 원래 가중치의 참조 점수이며 이번 실측값은 위 80.471931%입니다.

원본 가중치/코드는 파일 단위 그대로 보존했습니다. 새 portable 실행기와 문서/결과만 추가했습니다.
`files_sha256.json`은 원래 모델/코드 해시, `bundle_sha256.json`은 이 전달 패키지의 파일 해시입니다.
학습 33개 중 중간/OOF 전용 가중치는 이 패키지에 포함되지 않습니다. 데이터 원본도 포함되지 않습니다.
'''
    version=subprocess.check_output(['/home/sppark/miniconda3/envs/komap/bin/python','-c','import platform; print(platform.python_version())'],text=True).strip()
    portable_readme=portable_readme.replace('{python_version}',version)
    (extras/'README_TRANSFER.md').write_text(portable_readme)
    descriptor=dict(id='lite13_retuned_20261014',family='Lite13-retuned',kind='complete_ensemble',archive=ARCHIVE,
        checkpoint_count=13,checkpoints=checkpoints,image_miou=metrics['image_miou'],score_scope='complete R7+E5+F1 ensemble, valid20 image mIoU, empty union=0',
        training_count=33,source_result=str(RESULT),source_code_repository='CNUKoMaP/lhs',source_code_commit=read(RESULT/'source_snapshot/provenance.json')['commit'],
        inference_times={s:{k:times[s][k] for k in ['images','wall_seconds','seconds_per_image','images_per_second','scope','execution']} for s in ['valid','test']})
    write(extras/'model_manifest.json',descriptor)
    entries={str(p.relative_to(MODEL)):p for p in MODEL.rglob('*') if p.is_file()}
    entries.update({p.name:p for p in extras.iterdir() if p.is_file()})
    entries.update({'results/'+name:reportdir/name for name in reports})
    bundle_hashes={name:sha(p) for name,p in entries.items()};write(extras/'bundle_sha256.json',bundle_hashes);entries['bundle_sha256.json']=extras/'bundle_sha256.json'
    archive=REPO/ARCHIVE;archive.parent.mkdir(parents=True)
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=1,allowZip64=True) as z:
        for name,p in tqdm(sorted(entries.items()),desc='Create model ZIP',unit='file',mininterval=5):z.write(p,PREFIX+'/'+name)
    assert archive.stat().st_size<2*1024**3
    with zipfile.ZipFile(archive) as z:
        assert len(z.namelist())==len(entries) and len(set(z.namelist()))==len(entries)
        for name,p in tqdm(sorted(entries.items()),desc='Verify ZIP contents',unit='file',mininterval=5):
            h=hashlib.sha256()
            with z.open(PREFIX+'/'+name) as f:
                for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
            assert h.hexdigest()==sha(p),name
    digest=sha(archive);size=archive.stat().st_size
    descriptor.update(archive_sha256=digest,archive_bytes=size,archive_verified=True)
    write(REPO/'metadata/complete_models.json',[descriptor]);write(reportdir/'model_manifest.json',descriptor)
    shutil.copy2(extras/'README_TRANSFER.md',reportdir/'README.md')
    manifest=baseline_manifest.copy();manifest['files']=dict(baseline_manifest['files']);manifest['files'][ARCHIVE]=dict(bytes=size,sha256=digest)
    manifest['checkpoint_count']=72;manifest['model_archive_count']=30;manifest['complete_model_checkpoint_count']=13
    manifest['complete_model_bundles']=[dict(id=descriptor['id'],archive=ARCHIVE,checkpoint_count=13,image_miou=metrics['image_miou'])]
    write(REPO/'manifest.json',manifest);write(REPO/'metadata/checkpoints.json',baseline_checkpoints+checkpoints)
    csv_path=REPO/'metadata/checkpoints.csv'
    with csv_path.open() as f:fields=next(csv.reader(f))
    with csv_path.open('a',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writerows(checkpoints)
    with (REPO/'SHA256SUMS.txt').open('a') as f:f.write(digest+'  '+ARCHIVE+'\n')
    p=REPO/'verify_download.py';text=p.read_text();text=text.replace("'HRNet392','B4-329']","'HRNet392','B4-329','Lite13-retuned']")
    text=text.replace("if args.family=='B4-329':", "if args.family in ['B4-329','Lite13-retuned']:")
    text=text.replace("B4-329 was fixed, without a seed ranking. Use --family B4-329 without --best.","This model is fixed and has no seed ranking; omit --best for this family.")
    text=text.replace("if extra.exists():candidates+=json.loads(extra.read_text())", "if extra.exists():candidates+=json.loads(extra.read_text())\n        complete=root/'metadata/complete_models.json'\n        if complete.exists():candidates+=json.loads(complete.read_text())")
    p.write_text(text)
    p=REPO/'README.md';text=p.read_text();text=text.replace('# Model349 lite20 — 6개 계열 모델 전달','# Model349 모델 전달 — lite13 retuned와 lite20 시드 탐색',1)
    text=text.replace('현재 **총 59개 가중치 / 29개 모델 ZIP**입니다.','lite20 전달분은 **59개 가중치 / 29개 모델 ZIP**입니다. 새 lite13 완성 모델 13개를 포함해 저장소 전체는 **72개 가중치 / 30개 모델 ZIP**입니다.',1)
    section=f'''## 새 모델: lite13 retuned — 2026-10-14

**[13개 가중치와 추론 코드 ZIP 다운로드]({ARCHIVE})** — {size/1e9:.3f} GB.
[사용법·구성](reports/model_results_20261014_1/README.md) · [전체 결과서](reports/model_results_20261014_1/model_results_20261014_1.md).

전체 R7+E5+F1 앙상블의 valid20 image mIoU는 **{metrics['image_miou']*100:.6f}%**입니다.
선행/OOF 포함 33개 학습 작업에서 얻은 최종 13개 가중치이며, 이 ZIP 하나에 추론에 필요한 모델이 모두 있습니다.

| 데이터 | 실측 전체 추론시간 | 장당 평균 |
|---|---:|---:|
| valid 20장 | {times['valid']['wall_seconds']:.3f}초 | {times['valid']['seconds_per_image']:.3f}초 |
| test 10장 | {times['test']['wall_seconds']:.3f}초 | {times['test']['seconds_per_image']:.3f}초 |

RTX 4090, R은 GPU 3~6에 분배, E는 GPU 3, F는 CPU입니다. 모델 로딩·전처리·TTA/TENT·후처리·저장까지 포함한 전체 경과시간입니다.
학습·채점·보고서 생성은 제외하며 장당 값은 전체 시간/이미지 수입니다. test 정답이 없어 test 점수는 없습니다.
일부 checkpoint 선택과 원본 비율 조정에 valid20을 사용했습니다. 공개 원래 가중치의 80.779378%와 이번 재학습 점수는 구분합니다.

```bash
git lfs pull --include="{ARCHIVE}"
python verify_download.py --family Lite13-retuned --deep
unzip {ARCHIVE}
cd {PREFIX}
conda activate komap
python predict_portable.py --verify-only
python predict_portable.py --images /path/to/images --output /path/to/new_output --gpus 0
```

아래 lite20 후보별 점수와는 별도 실험입니다. 기존 59개 가중치 파일은 그대로 보존했습니다.

'''
    text=text.replace('## 계열별 최고 후보',section+'## lite20 계열별 최고 후보',1)
    text=text.replace('약 **8.10 GiB**',f'약 **{sum(r["bytes"] for r in manifest["files"].values())/(1024**3):.2f} GiB**')
    text=text.replace('`metadata/checkpoints.csv/json`: 59개 가중치','`metadata/checkpoints.csv/json`: 72개 가중치')
    text += '\n- `metadata/complete_models.json`: lite13 retuned 전체 모델 13개, 구성과 실측 점수/추론시간.\n'
    p.write_text(text)
    assert read(REPO/'metadata/checkpoints.json')[:59]==baseline_checkpoints
    for name,value in baseline_manifest['files'].items():assert manifest['files'][name]==value
    write(ROOT/'build_verified.json',dict(ok=True,parent=parent,archive=ARCHIVE,bytes=size,sha256=digest,checkpoints=13,
        total_checkpoints=72,original_checkpoint_bytes=1952143546,archive_members=len(entries),score=metrics['image_miou'],
        portable_launcher=str(extras/'predict_portable.py'),old_model_files_unchanged=True))
    print(json.dumps(read(ROOT/'build_verified.json'),indent=2))

if __name__=='__main__':main()
