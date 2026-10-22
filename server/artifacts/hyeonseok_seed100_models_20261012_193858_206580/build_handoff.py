"""Package all 56 requested seed-search checkpoints and documented scores."""
import csv
import hashlib
import io
import json
import shutil
import threading
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'repository'
PROJECT = Path('/Data3/sppark/komap')
RESULT = PROJECT / 'results/model_results_20261012_1'
COMPLETE = json.loads((RESULT/'complete.json').read_text())
RUN = Path(COMPLETE['run'])
IMPL = RUN / 'implementation'
RECORDS = {p.stem:json.loads(p.read_text()) for p in (RUN/'completed').glob('*.json')}
PLAN = json.loads((RUN/'plan.json').read_text())
BASE = Path(PLAN['baseline'])
BASE_SCORE = COMPLETE['baseline_image_miou']
FAMILIES = {'347':'HRNet347', '326':'B4-326', '322':'B4-322', 'E_b4':'B4-warmstart', 'E_hr':'HRNet392'}
EXPECTED = {'347':12, '326':8, '322':4, 'E_b4':24, 'E_hr':8}


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def dump(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2)+'\n')


def csv_text(rows, fields):
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n"); writer.writeheader()
    writer.writerows({k:r.get(k) for k in fields} for r in rows)
    return buffer.getvalue()


def members(candidate):
    return [candidate] if candidate['kind']=='train' else [RECORDS[k] for k in candidate['deps']]


def parent_info(candidate):
    if candidate['group']!='347':
        return None
    task_id = candidate.get('parent_task')
    path = Path(candidate['context']['weights'].get('R/weights/model336_hrnet_best/last.pt',
                BASE/'retrained_model/Model349_exh14/weights/model336_hrnet_best/last.pt'))
    return dict(task_id=task_id or 'baseline_336', seed=RECORDS[task_id]['seed'] if task_id else 0,
                checkpoint_sha256=sha(path), source_checkpoint=str(path), included_in_this_repository=False)


def build_candidate(candidate, progress, lock):
    group=candidate['group']; family=FAMILIES[group]; identifier=candidate['id']
    name=identifier+(f'_seed{candidate["seed"]}' if candidate['kind']=='train' else '')
    relative=f'models/{family}/{name}.zip'
    destination=OUT/relative; destination.parent.mkdir(parents=True,exist_ok=True)
    expected_files={}; weight_rows=[]; metadata={}; files={}
    for task in members(candidate):
        checkpoint=Path(task['checkpoint']); sub=f'{name}/{task["id"]}'
        args=checkpoint.with_name('args.json')
        for file in [checkpoint,args]:
            expected=task['artifacts'][str(file)]
            assert sha(file)==expected, file
            member=f'{sub}/{file.name}'; files[member]=file; expected_files[member]=expected
        command=Path(task['attempt'])/'train_command.json'
        member=f'{sub}/train_command.json'; files[member]=command; expected_files[member]=sha(command)
        bn=checkpoint.with_name('bn_freeze_check.json')
        if bn.exists():
            expected=task['artifacts'][str(bn)]; assert sha(bn)==expected
            files[f'{sub}/{bn.name}']=bn;expected_files[f'{sub}/{bn.name}']=expected
        weight_rows.append(dict(family=family,group=group,candidate_id=identifier,task_id=task['id'],
            training_seed=task['seed'],fold=task.get('fold'),hr_slot=task.get('slot'),
            archive=relative,member=f'{sub}/{checkpoint.name}',
            checkpoint_sha256=task['artifacts'][str(checkpoint)],checkpoint_bytes=checkpoint.stat().st_size,
            deployed_weight_key=task['weight_key'],source_checkpoint=str(checkpoint),
            candidate_ensemble_image_miou=candidate['image_miou']))
    info=dict(id=identifier,group=group,family=family,archive=relative,
              image_miou=candidate['image_miou'],image_miou_percent=100*candidate['image_miou'],
              delta_baseline_percentage_points=100*(candidate['image_miou']-BASE_SCORE),
              baseline_image_miou=BASE_SCORE,image_count=20,
              score_scope='Full R+E+F ensemble on valid20 with this candidate inserted; not a standalone model score',
              fold_score_policy='Fold members are evaluated together; no separately measured per-fold score',
              hr336_parent=parent_info(candidate),checkpoints=weight_rows,
              images=candidate['images'])
    metadata[f'{name}/candidate.json']=(json.dumps(info,ensure_ascii=False,indent=2)+'\n').encode()
    metadata[f'{name}/README.md']=(f'# {family} / {identifier}\n\n'
        f'valid20 전체 R+E+F image mIoU: {100*candidate["image_miou"]:.6f}%.\n'
        '개별 모델 단독 점수가 아닙니다. fold 구성원은 함께 평가했습니다.\n'
        '각 task 폴더에 원본 last.pt, args.json, train_command.json을 담았습니다.\n'
        'candidate.json의 체크포인트별 seed/fold/hr_slot/deployed_weight_key를 확인하세요.\n'
        'HRNet347의 평가에는 candidate.json에 명시된 HR336 부모 구성도 필요합니다.\n'
        'args.json과 train_command.json의 절대경로는 원본 학습 당시 기록입니다.\n'
        '일반 ZIP Deflate 형식이며, 저장소 verify_download.py로 검증할 수 있습니다.\n').encode()
    checksums=''.join(f'{value}  {member.removeprefix(name+"/")}\n' for member,value in expected_files.items())
    metadata[f'{name}/SHA256SUMS.txt']=checksums.encode()
    with zipfile.ZipFile(destination,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=1,allowZip64=True) as archive:
        for member,file in files.items():
            info_zip=zipfile.ZipInfo.from_file(file,member);info_zip.compress_type=zipfile.ZIP_DEFLATED
            info_zip._compresslevel=1
            with file.open('rb') as src,archive.open(info_zip,'w') as dest:
                for block in iter(lambda:src.read(4*1024*1024),b''):
                    dest.write(block)
                    with lock:progress.update(len(block))
        for member,data in metadata.items():archive.writestr(member,data)
    with zipfile.ZipFile(destination) as archive:
        for member,expected in expected_files.items():
            with archive.open(member) as stream:
                assert hashlib.file_digest(stream,'sha256').hexdigest()==expected,member
        for member,expected in metadata.items():assert archive.read(member)==expected
    assert destination.stat().st_size<2**31
    info.update(archive_bytes=destination.stat().st_size,archive_sha256=sha(destination),archive_verified=True)
    dump(OUT/f'metadata/candidates/{identifier}.json',info)
    print(f'VERIFIED {relative}: {len(weight_rows)} weights, {info["archive_bytes"]:,} bytes',flush=True)
    return info


def main():
    candidates=sorted([r for r in RECORDS.values() if r.get('candidate') and r['group'] in FAMILIES],key=lambda r:r['id'])
    targets=[t for c in candidates for t in members(c)]
    assert len(targets)==56 and len({t['id'] for t in targets})==56
    assert {g:sum(t['group']==g for t in targets) for g in FAMILIES}==EXPECTED
    assert COMPLETE['ok'] and COMPLETE['training_count']==100
    total=sum(Path(t['checkpoint']).stat().st_size+Path(t['checkpoint']).with_name('args.json').stat().st_size+
              (Path(t['attempt'])/'train_command.json').stat().st_size+
              (Path(t['checkpoint']).with_name('bn_freeze_check.json').stat().st_size if Path(t['checkpoint']).with_name('bn_freeze_check.json').exists() else 0)
              for t in targets)
    lock=threading.Lock()
    with tqdm(total=total,desc='Package 56 model weights',unit='B',unit_scale=True,mininterval=10) as progress:
        with ThreadPoolExecutor(max_workers=3) as pool:
            infos=[future.result() for future in as_completed([pool.submit(build_candidate,c,progress,lock) for c in candidates])]
    infos.sort(key=lambda r:(list(FAMILIES).index(r['group']),-r['image_miou'],r['id']))
    weights=[w for c in infos for w in c['checkpoints']]
    best={family:next(r for r in infos if r['group']==group) for group,family in FAMILIES.items()}
    dump(OUT/'metadata/all_candidates.json',infos)
    dump(OUT/'metadata/best_per_family.json',best)
    dump(OUT/'metadata/checkpoints.json',weights)
    (OUT/'metadata/checkpoints.csv').write_text(csv_text(weights,['family','candidate_id','task_id','training_seed','fold','hr_slot','archive','member','checkpoint_sha256','checkpoint_bytes','deployed_weight_key']))
    (OUT/'metadata/candidate_scores.csv').write_text(csv_text(infos,['family','id','image_miou','image_miou_percent','delta_baseline_percentage_points','archive','archive_sha256']))
    dump(OUT/'metadata/parent_dependencies.json',{r['id']:r['hr336_parent'] for r in infos if r['hr336_parent']})
    code_manifest=json.loads((IMPL/'code_manifest.json').read_text())
    source_archive=OUT/'source_snapshot.zip'
    with zipfile.ZipFile(source_archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
        for name,expected in code_manifest.items():
            assert sha(IMPL/name)==expected,name
            archive.write(IMPL/name,'source_snapshot/'+name)
        for name in ['code_manifest.json','input_manifest.json']:
            archive.write(IMPL/name,'source_snapshot/'+name)
    with zipfile.ZipFile(source_archive) as archive:
        for name,expected in code_manifest.items():
            with archive.open('source_snapshot/'+name) as stream:
                assert hashlib.file_digest(stream,'sha256').hexdigest()==expected
    shutil.copy2(RUN/'environment.freeze.txt',OUT/'environment.freeze.txt')
    shutil.copy2(RUN/'plan.json',OUT/'metadata/original_plan.json')
    shutil.copy2(RESULT/'model_results_20261012_1.md',OUT/'metadata/original_final_report.md')
    shutil.copy2(RESULT/'metrics.json',OUT/'metadata/original_final_metrics.json')
    files={r['archive']:dict(bytes=r['archive_bytes'],sha256=r['archive_sha256']) for r in infos}
    files['source_snapshot.zip']=dict(bytes=source_archive.stat().st_size,sha256=sha(source_archive))
    manifest=dict(repository='CNUKoMaP/giveItToHyeonSeok',checkpoint_count=56,candidate_count=len(infos),
        training_counts={FAMILIES[g]:n for g,n in EXPECTED.items()},source_run=str(RUN),
        source_result=str(RESULT),baseline_image_miou=BASE_SCORE,final_combination_image_miou=COMPLETE['image_miou'],
        originals_preserved=True,all_archives_verified=True,files=files)
    dump(OUT/'manifest.json',manifest)
    (OUT/'SHA256SUMS.txt').write_text(''.join(f'{v["sha256"]}  {k}\n' for k,v in files.items()))
    write_readme(infos,best,manifest)
    dump(ROOT/'build_verified.json',manifest)
    print(f'COMPLETE: 56 checkpoints / {len(infos)} candidate ZIPs / {sum(f["bytes"] for f in files.values()):,} bytes',flush=True)


def member_labels(candidate):
    return ', '.join((f'slot{w["hr_slot"]}/' if w['hr_slot'] else '')+
                     (f'fold{w["fold"]}: ' if w['fold'] else '')+str(w['training_seed']) for w in candidate['checkpoints'])


def write_readme(infos,best,manifest):
    total_gib=sum(f['bytes'] for f in manifest['files'].values())/1024**3
    lines=['# Model349 lite20 — 5개 계열의 시드 탐색 모델 전달', '',
        '100회 학습 시드 탐색에서 생성한 **HRNet347 12개, B4-326 8개, B4-322 4개, B4 warmstart 24개, HRNet392 8개**, 총 **56개 가중치**를 모두 담았습니다.',
        '평가 단위에 맞춘 **28개 ZIP**이며, 각 ZIP은 원본 `last.pt`, `args.json`, 학습 명령, 점수/seed/fold/적용 위치, SHA256을 포함합니다.',
        'B4 warmstart는 BN 통계 고정 확인 기록도 포함합니다. 기존 full-train70 r70 실험은 이번 전달 범위와 별도입니다.', '',
        '## 계열별 최고 후보', '',
        '**아래 점수는 해당 후보를 적용한 전체 R+E+F 앙상블의 valid20 image mIoU입니다. 개별 모델 단독 성능이 아닙니다.**',
        '기존 lite20 기준은 **80.255846%**입니다. HRNet347은 명시된 HR336 부모와 함께 교체했으며, fold 모델들은 묶음으로 평가했습니다.',
        '최고 후보는 각 계열에서 이번에 새로 학습한 후보 중 선정했습니다. 기존 가중치가 더 높은 경우도 아래에 표시합니다.', '',
        '| 계열 | 가중치 수 / 후보 수 | 최고 후보 | 학습 seed (fold/slot) | image mIoU | 기준 대비 %p |',
        '|---|---:|---|---|---:|---:|']
    for group,family in FAMILIES.items():
        b=best[family];n=sum(r['group']==group for r in infos)
        lines.append(f'| {family} | {EXPECTED[group]} / {n} | [{b["id"]}]({b["archive"]}) | {member_labels(b)} | **{b["image_miou_percent"]:.6f}%** | {b["delta_baseline_percentage_points"]:+.6f} |')
    lines += ['',
        '- **HRNet347 최고**: `347_05`, 학습 seed **3424**, 부모 HR336=`336_05` (seed **3412**). 부모와 함께 사용한 평가 점수입니다.',
        '- **B4-326 최고**: `326_set1`의 fold1 seed **3457**, fold2 seed **3458**을 함께 사용합니다.',
        '- **B4-322 최고**: `322_set2`, fold3 seed **3465**입니다.',
        '- **B4 warmstart 최고**: `E_b4_set3`, fold1~4 seed **3479, 3480, 3481, 3482**입니다.',
        '- **HRNet392 최고**: `E_hr_set1`, slot1/fold2=**3495**, slot1/fold4=**3496**, slot2/fold1=**3497**, slot2/fold4=**3498**입니다.',
        '- `slot1/slot2`는 기존 모델 경로의 `seed1/seed2` 슬롯을 뜻합니다. 실제 학습 난수 seed는 위의 34xx 값입니다.',
        '- 새 B4 warmstart 및 HRNet392 후보는 모두 기존 기준보다 낮아 최종 선택에서는 **기존 E 가중치를 유지**했습니다.',
        '- 최종 조합은 HRNet347 `347_07`(seed3426), B4-326 `326_set1`, B4-322 `322_set2` 등을 사용해 **80.479068%**입니다. 조합 선택 때문에 HRNet347의 계열별 최고(`347_05`)와 최종 선택(`347_07`)이 다릅니다.',
        '- 이 저장소의 5개 계열만으로 최종 20개 모델 전체가 구성되지는 않습니다. R의 다른 구성 모델, HR336, F 및 기준 E 가중치는 별도입니다.', '',
        '## 다운로드와 검증', '',
        f'가중치 ZIP 전체 크기는 소스 ZIP을 포함해 약 **{total_gib:.2f} GiB**입니다. 모델 ZIP은 Git LFS에 저장됩니다.',
        '위 표의 최고 후보 링크 또는 아래 전체 목록에서 파일 페이지의 **Download raw file**로 ZIP을 받을 수 있습니다.',
        '전체 clone은 Git LFS를 설치한 환경에서 다음과 같이 합니다.', '',
        '```bash', 'git lfs install', 'git clone https://github.com/CNUKoMaP/giveItToHyeonSeok.git', 'cd giveItToHyeonSeok', 'python verify_download.py --deep', '```', '',
        '최고 후보 5개만 받으려면:', '', '```bash',
        'GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/CNUKoMaP/giveItToHyeonSeok.git',
        'cd giveItToHyeonSeok',
        'git lfs pull --include="'+','.join(best[f]['archive'] for f in FAMILIES.values())+'"',
        'python verify_download.py --best --deep', '```', '',
        '일반 ZIP(Deflate) 형식이며 폴더 구조를 유지해 압축을 해제하세요. `last.pt` 옆에 해당 `args.json`을 둡니다.',
        '`candidate.json`과 `metadata/checkpoints.csv`의 `deployed_weight_key`로 원래 앙상블의 적용 위치를 확인할 수 있습니다.',
        'HRNet347은 완전한 모델 가중치이므로 자체 로딩에 부모 파일은 필요하지 않지만, 표의 앙상블 점수를 재현하려면 함께 평가한 HR336 부모 가중치도 필요합니다. 부모 정보/해시는 `metadata/parent_dependencies.json`에 있습니다.',
        'HR336 seed3411/3412는 이전 [8개 HR336 가중치 ZIP](https://github.com/CNUKoMaP/sppark/blob/main/model349_lite20_seeds3407-3414_weights.zip)에도 포함되어 있습니다. 해당 저장소 접근 권한이 필요합니다.', '',
        '## 지표와 학습 조건', '',
        '- valid 이미지마다 네 클래스 IoU를 평균한 뒤 valid20의 평균을 계산합니다. 정답과 예측에 모두 없는 클래스는 IoU=0입니다. global mIoU는 계산하지 않았습니다.',
        '- 동일한 valid20에서 반복 비교/선택한 개발용 점수이며, 독립 test 성능이 아닙니다.',
        '- HRNet347: HR336 체크포인트에서 12 epochs 미세조정, 역빈도 CE 가중치.',
        '- B4-326: zcdf3 / image_global / scale3 / area, fold1·2 각각 16,000 iterations.',
        '- B4-322: gray3 / imagenet / scale3 / area, fold3 16,000 iterations.',
        '- B4 warmstart: zcdf3 / image_global / target_width4000 / uniform, 기존 B4에서 BN 통계 고정, 각 fold 2,000 iterations.',
        '- HRNet392: UnetPlusPlus / tu-hrnet_w48, patch512 / input_scale2 / width 정규화, 각 모델 70 epochs.',
        '- 실행 환경은 conda `komap`, 사용 GPU는 물리 4~7입니다. 학습 인자의 상세값은 각 원본 `args.json` 및 `train_command.json`에 있습니다.', '',
        '## 전체 후보 목록', '']
    for group,family in FAMILIES.items():
        lines += [f'### {family}', '', '| 후보 / ZIP | 학습 seed (fold/slot) | HR336 부모 | 가중치 수 | image mIoU |', '|---|---|---|---:|---:|']
        for r in infos:
            if r['group']!=group:continue
            parent=r['hr336_parent'];parent_text=(f'{parent["task_id"]} / seed{parent["seed"]}' if parent else '—')
            lines.append(f'| [{r["id"]}]({r["archive"]}) | {member_labels(r)} | {parent_text} | {len(r["checkpoints"])} | {r["image_miou_percent"]:.6f}% |')
        lines.append('')
    lines += ['## 함께 제공한 기록', '',
        '- `metadata/best_per_family.json`: 계열별 최고 후보와 구성원.',
        '- `metadata/candidate_scores.csv`, `metadata/all_candidates.json`: 28개 후보의 전체 점수.',
        '- `metadata/checkpoints.csv/json`: 56개 가중치의 seed/fold/slot/적용 경로/SHA256.',
        '- `metadata/candidates/`: 후보별 valid20 이미지 점수와 출처.',
        '- `source_snapshot.zip`: 해당 탐색에서 동결한 모델/학습/추론 소스. 원본 소스 해시와 대조했습니다.',
        '- `environment.freeze.txt`: 실제 conda komap의 패키지 버전.',
        '- `metadata/original_plan.json`, `metadata/original_final_report.md`: 원래 100회 계획과 최종 결과.',
        '- `manifest.json`, `SHA256SUMS.txt`, `verify_download.py`: ZIP 및 압축 내부 가중치 검증.',
        '- 학습 데이터 및 나머지 앙상블 가중치는 포함하지 않았습니다. 원본 코드와 명령의 서버 절대경로는 수신 환경에 맞게 지정해야 합니다.', '']
    (OUT/'README.md').write_text('\n'.join(lines))


if __name__=='__main__':
    main()
