"""Read-only, GT-free comparison of completed fresh E maps to saved E."""
from pathlib import Path
import json,hashlib
import numpy as np
from PIL import Image
ROOT=Path('/workspace/model349_compress/accuracy81_stage1_fresh_20261009')
DATA=Path('/workspace/komap_model1')
SRC=DATA/'outputs/model401_model397_blend_valid_once'
B4=DATA/'outputs/model329_fullstats_width4000_uniform_group_oof/valid_ensemble/probabilities.npz'
manifest=json.loads((ROOT/'runs/release_valid/input_manifest.json').read_text())
names={r['name'] for r in manifest['images']}
completed=[line.strip() for line in (ROOT/'expert_valid.log').read_text().splitlines() if line.strip() in names]
assert len(completed)==len(set(completed))
def labels(p):return np.array([2,0,1,3],dtype=np.uint8)[p[[2,0,1,3]].argmax(0)]
rows=[]
with np.load(B4) as b4:
 for name in completed:
  path=ROOT/f'runs/expert_valid/{name}.npy'
  before=hashlib.sha256(path.read_bytes()).hexdigest()
  fresh=np.load(path)
  assert before==hashlib.sha256(path.read_bytes()).hexdigest()
  parts=[b4[name].astype(np.float32)]
  for seed in [1,2]:
   with np.load(SRC/f'hrnet_seed{seed}_mean_valid/{name}.npz') as z:parts.append(z[name].astype(np.float32))
  reference=(.5*parts[0]+.25*parts[1]+.25*parts[2])[[1,2,0,3]]
  assert fresh.shape==reference.shape and np.isfinite(fresh).all()
  old=labels(reference);new=labels(fresh)
  stored=np.asarray(Image.open(Path('/workspace/model349_compress/accuracy81_oof_zgray_20261009')/f'matched_source/control/predicted_labels/{name}.png'))
  assert np.array_equal(old,stored)
  rows.append({'name':name,'fresh_sha256':before,'probability_max_abs_diff':float(np.abs(fresh-reference).max()),'probability_mean_abs_diff':float(np.abs(fresh-reference).mean()),'mask_changed_pixels':int(np.count_nonzero(old!=new)),'pixels':int(old.size)})
report={'completed_images':len(rows),'gt_read':False,'whole_E_complete':(ROOT/'runs/expert_valid/completion.json').exists(),'rows':rows}
(ROOT/'expert_progress_audit.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
