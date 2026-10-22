"""Copy immutable completed Model401 HR maps, checking every manifest hash."""
from pathlib import Path
import hashlib,json,shlex,subprocess,tarfile
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'hr392_inputs'; OUT.mkdir(exist_ok=True)
assert not list(OUT.iterdir()), 'Refusing to overwrite any collected input'
remote='/workspace/komap_model1/outputs/model401_model397_blend_valid_once'
script='''from pathlib import Path
import json,hashlib,tarfile
r=Path(%r); m=json.loads((r/'inference_manifest.json').read_text()); files=[r/'inference_manifest.json',r/'recovery_receipt.json']; rows=[]
assert m['status']=='INFERENCE_COMPLETE_NOT_SCORED', m['status']
assert len(m['image_ids'])==20 and m['test_inputs_excluded'] is True
for seed in [1,2]:
 for row in m[f'hrnet_seed{seed}_mean_maps']:
  p=r/f'hrnet_seed{seed}_mean_valid'/(row['image_id']+'.npz')
  assert hashlib.sha256(p.read_bytes()).hexdigest()==row['sha256'],p
  files.append(p);rows.append({'path':str(p.relative_to(r)),'sha256':row['sha256']})
assert len(rows)==40
a=Path('/workspace/model349_compress/accuracy81_20261008/hr392_inputs.tar.gz')
with tarfile.open(a,'x:gz') as t:
 for p in files:t.add(p,arcname=str(p.relative_to(r)),recursive=False)
print(json.dumps({'archive':str(a),'sha256':hashlib.sha256(a.read_bytes()).hexdigest(),'maps':rows}))
'''%remote
receipt=json.loads(subprocess.check_output(['ssh','-p','20670','root@168.188.128.36','python3 -c '+shlex.quote(script)],text=True))
archive=ROOT/'hr392_inputs.tar.gz'
subprocess.run(['scp','-P','20670','root@168.188.128.36:'+receipt['archive'],str(archive)],check=True)
assert hashlib.sha256(archive.read_bytes()).hexdigest()==receipt['sha256']
with tarfile.open(archive) as t:
 for m in t.getmembers():assert m.isfile() and not Path(m.name).is_absolute() and '..' not in Path(m.name).parts
 t.extractall(OUT,filter='data')
for row in receipt['maps']:assert hashlib.sha256((OUT/row['path']).read_bytes()).hexdigest()==row['sha256']
(ROOT/'hr392_collection.json').write_text(json.dumps(receipt,indent=2)+'\n')
print('Verified 40 completed maps, server and local manifest SHA-256 exact')
