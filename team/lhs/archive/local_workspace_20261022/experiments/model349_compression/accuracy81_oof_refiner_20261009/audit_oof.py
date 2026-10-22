"""Read-only network-OOF input integrity and group separation audit."""
from pathlib import Path
import hashlib,json,re,time
import numpy as np
ROOT=Path('/workspace/komap_model1');OUT=Path(__file__).resolve().parent
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def group(n):return re.match(r'^(A\d+)_',n).group(1)
def main():
 lockpath=ROOT/'experiments/results/model397_fixed_b4_model392_blend_lock.json';lock=json.loads(lockpath.read_text())
 assert sha(lockpath)=='8e9e4250332f260c52124f1d7ff19ac21ee51e602d1b7e2a33c0a6b0b7e52d57'
 allids=set();allgroups={};rows=[]
 for fold in range(1,5):
  held={p.name.removesuffix('_mask.png') for p in (ROOT/f'data_model322_group_cv/oof_fold{fold}/masks').glob('*_mask.png')}
  train={p.name.removesuffix('_mask.png') for p in (ROOT/f'data_model322_group_cv/train_fold{fold}/masks').glob('*_mask.png')}
  hg={group(x) for x in held};tg={group(x) for x in train}
  assert held and train and held.isdisjoint(train) and hg.isdisjoint(tg) and len(held|train)==70
  assert allids.isdisjoint(held);allids|=held
  for g in hg:assert g not in allgroups;allgroups[g]=fold
  paths=[k for k in lock['locked_files'] if k.endswith('probabilities.npz') and f'fold{fold}_oof/' in k];assert len(paths)==3
  hashes={}
  for k in paths:
   p=ROOT/k;meta=lock['locked_files'][k];assert p.stat().st_size==meta['size_bytes'] and sha(p)==meta['sha256']
   with np.load(p,allow_pickle=False) as z:assert set(z.files)==held
   hashes[k]=meta['sha256']
  p=ROOT/f'outputs/model329_fullstats_width4000_uniform_group_oof/fold{fold}/args.json'
  assert sha(p)==lock['locked_files'][str(p.relative_to(ROOT))]['sha256']
  assert json.loads(p.read_text())['train_splits']==f'train_fold{fold}'
  for seed in [1,2]:
   data=ROOT/f'data_model382_seed{seed}_fold{fold}'
   assert (data/'valid').resolve()==(ROOT/f'data_model322_group_cv/oof_fold{fold}').resolve()
   assert (data/'train').resolve()==(ROOT/f'data_model322_group_cv/train_fold{fold}').resolve()
   for filename in ['args.json','model392_run_receipt.json']:
    receipt=ROOT/f'outputs/model392_widthnorm/seed{seed}/fold{fold}'/filename
    assert sha(receipt)==lock['locked_files'][str(receipt.relative_to(ROOT))]['sha256']
   m=json.loads((ROOT/f'outputs/model392_widthnorm/seed{seed}/fold{fold}_oof/model392_manifest.json').read_text())
   assert m['fold']==fold and m['seed']==seed and m['model']==392
   key=f'outputs/model392_widthnorm/seed{seed}/fold{fold}_oof/valid_probabilities.npz'
   assert m['oof_probabilities_sha256']==hashes[key]
  rows.append({'fold':fold,'held_images':len(held),'train_images':len(train),'held_groups':sorted(hg),'train_and_held_groups_disjoint':True,'probability_archive_sha256':hashes})
 assert len(allids)==70 and len(allgroups)==59
 result={'time':time.time(),'model397_lock_sha256':sha(lockpath),'images':70,'groups':59,'all_image_ids':sorted(allids),'folds':rows,'official_valid_or_test_inputs_opened':False,'gpu_used':False,'scope':'network component grouped-OOF inputs verified; no new refiner or independent full-pipeline generalization result'}
 (OUT/'oof_input_audit.json').write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps({'images':70,'groups':59,'fold_counts':[x['held_images'] for x in rows],'probability_archives_verified':12,'official_valid_or_test_inputs_opened':False}))
if __name__=='__main__':main()
