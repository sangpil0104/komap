"""Collect finished B4 continuation; verify saved weights and independently score PNGs."""
from pathlib import Path
import hashlib,importlib.util,json,shlex,subprocess,tarfile,time
import numpy as np
ROOT=Path(__file__).resolve().parent
REMOTE='/workspace/model349_compress/'+ROOT.name
SSH=['ssh','-p','20670','root@168.188.128.36']
stamp=str(time.time_ns())
code=r'''import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
from pathlib import Path
import json,hashlib,tarfile,torch
torch.set_num_threads(2)
r=Path(%r);done=json.loads((r/'completion.json').read_text())
assert done['finished'] and done['whole_fresh_pipeline'] and done['parent_source_weights_unchanged']
assert done['folds']==[1,3,4] and done['checkpoint_count']==14
audits={};sequence=[]
for f in [1,3,4]:
 d=r/'training'/('fold'+str(f));receipt=json.loads((d/'completion.json').read_text())
 assert receipt['finished'] and receipt['updates']==400 and receipt['train_images_seen']==70
 assert not receipt['valid_read'] and not receipt['test_read']
 p=d/'last.pt';assert hashlib.sha256(p.read_bytes()).hexdigest()==receipt['checkpoint_sha256']
 parent=r.parent/'Model349_exh14/weights/model329_fullstats_width4000_uniform_group_oof'/('fold'+str(f))/'last.pt'
 old=torch.load(parent,map_location='cpu',weights_only=False);new=torch.load(p,map_location='cpu',weights_only=False)
 assert old.keys()==new.keys()
 assert all(a.shape==new[k].shape and a.dtype==new[k].dtype for k,a in old.items())
 assert all(torch.isfinite(a).all() for a in new.values())
 bn=[k for k in old if k.endswith(('running_mean','running_var','num_batches_tracked'))]
 assert bn and all(torch.equal(old[k],new[k]) for k in bn)
 changed=sum(not torch.equal(a,new[k]) for k,a in old.items());assert changed>0
 sequence_file=d/'sample_sequence.json'
 assert hashlib.sha256(sequence_file.read_bytes()).hexdigest()==receipt['sample_sequence_sha256']
 samples=json.loads(sequence_file.read_text());assert len(samples)==800 and len({x['image'] for x in samples})==70
 sequence.append(samples)
 audits[str(f)]={'checkpoint_sha256':receipt['checkpoint_sha256'],'saved_dtype_and_shapes_match_parent':True,'bn_buffers_exact':len(bn),'changed_tensors':changed,'all_tensors_finite':True}
 del old,new
assert sequence[0]==sequence[1]==sequence[2]
(r/'saved_checkpoint_audit.json').write_text(json.dumps(audits,indent=2)+'\n')
files=[p for p in r.iterdir() if p.is_file() and p.suffix in ('.json','.py','.md','.log')]
files += [p for p in (r/'training').rglob('*.json')]
for folder in ['runs','attempt1_failed_synthetic']:
 files += [p for p in (r/folder).rglob('*') if p.is_file() and 'probabilities' not in p.relative_to(r/folder).parts]
files += [p for p in (r/'bundle').rglob('*') if p.is_file() and not p.is_symlink() and '__pycache__' not in p.parts and p.suffix in ('.py','.json','.yaml')]
a=r/('collection_'+%r+'.tar.gz')
with tarfile.open(a,'x:gz') as t:
 for p in sorted(set(files)):t.add(p,arcname=str(p.relative_to(r)),recursive=False)
print(json.dumps({'archive':str(a),'sha256':hashlib.sha256(a.read_bytes()).hexdigest(),'saved_checkpoint_audit':audits}))
'''%(REMOTE,stamp)
metadata=json.loads(subprocess.check_output(SSH+['/workspace/venv-komap/bin/python -c '+shlex.quote(code)],text=True))
archive=ROOT/Path(metadata['archive']).name
subprocess.run(['scp','-P','20670','root@168.188.128.36:'+metadata['archive'],str(archive)],check=True)
assert hashlib.sha256(archive.read_bytes()).hexdigest()==metadata['sha256']
snapshot=ROOT/'collected'/stamp;snapshot.mkdir(parents=True)
with tarfile.open(archive) as t:
    for m in t.getmembers():
        assert m.isfile() and not Path(m.name).is_absolute() and '..' not in Path(m.name).parts
    t.extractall(snapshot,filter='data')
spec=importlib.util.spec_from_file_location('audit',ROOT.parent/'audit_20261008/analyze_runs.py')
audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)
gt=Path('/tmp/claude-501/-Users-hyunseok-orca-workspaces-KoMaP-gurnard/8c455a47-b7eb-4c6e-b73d-ddf6da07fc20/scratchpad/gt/valid/masks')
canonical=ROOT.parent/'exh14_base_search_20261008/exh14_valid20'
control=sorted((ROOT.parent/'accuracy81_residual_20261008/collected').glob('*/runs/teacher_valid20'))[-1]
candidate=snapshot/'runs/b4all70_valid20'
ref,c,score=(audit.score(p,gt) for p in [canonical,control,candidate])
lock=audit.load(snapshot/'preflight_source_lock.json')
assert all(hashlib.sha256((snapshot/p).read_bytes()).hexdigest()==h for p,h in lock.items())
complete=audit.load(snapshot/'completion.json')
report={'miou_percent':score['miou']*100,'class_iou_percent':(np.array(score['classes'])*100).tolist(),
    'vs_canonical_base':audit.pair(canonical,candidate,ref,score),
    'vs_previous_same_gpu_control':audit.pair(control,candidate,c,score),
    'completion':complete,'fresh_inference':audit.load(candidate/'completion.json'),
    'archive':metadata,'source_snapshot_matches_preflight':True,'independent_png_scoring_agrees':True,
    'independent_generalization_verified':False,'test_used':False}
audit.dump(ROOT/'comparison.json',report)
lines=['# B4 세 fold의 train70 추가 학습 검증','','이미지별4클래스IoU 평균 후valid20평균,emptyunion0. 독립PNG/GT채점과모든80IoU 일치 확인. 같은valid 반복개발평가이며독립일반화증거가아니다.','',
    f"원본 base {100*ref['miou']:.7f}%, 새 후보 {100*score['miou']:.7f}%, 변화 {report['vs_canonical_base']['delta_pp']:+.7f}%p.",'',
    f"14가중치 {complete['checkpoint_bytes']:,}bytes. 전체freshvalid 단회 {report['fresh_inference']['elapsed_seconds']:.2f}초. 속도개선으로판정하지않는다. 기존3fold를모두유지했고새파라미터수/구조/FP32dtype변경없음. 학습후세모델은train70을보므로OOF모델이아니다.",'',
    '최종checkpoint의키/shape/dtype/finite와BN buffer exact를원본과대조했고모델tensor실제변경/3fold동일800sample순서를확인했다. 원본source/14weights 보존. test정답및test mIoU없음.','',
    '| 클래스 | base 대비 |','|---|---:|']
for i,name in enumerate(['Al3Ni','Primary Si','Al','Eutectic Si']):lines.append(f"| {name} | {report['vs_canonical_base']['class_delta_pp'][i]:+.7f}%p |")
lines+=['','| 이미지 | 새 mIoU | base 대비 |','|---|---:|---:|']
for name in sorted(score['images']):lines.append(f"| {name} | {100*score['images'][name]['image_miou']:.7f}% | {report['vs_canonical_base']['per_image_delta_pp'][name]:+.7f}%p |")
lines+=['','비교는전체추가학습효과다. train70의노출만분리한동일계산량대조군은없다. 최초random3채널합성입력의비유한gradient와자연전처리train입력게이트수정근거를실패snapshot/수정문서에보존했다. loss scale/비유한값판정은완화하지않았다.','']
(ROOT/'report.md').write_text('\n'.join(lines))
print(json.dumps({'miou_percent':report['miou_percent'],'delta_pp':report['vs_canonical_base']['delta_pp'],'worst_image':report['vs_canonical_base']['worst_image'],'worst_delta_pp':report['vs_canonical_base']['worst_image_delta_pp']},indent=2))
