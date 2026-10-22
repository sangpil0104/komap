"""Compare completed fresh E outputs to frozen family-fold replay, without GT."""
from pathlib import Path
import json
import numpy as np
r=Path('/workspace/model349_compress/fusion805_compression_round2_20261010');old=r.parent/'fusion805_compression_20261010';fresh=r/'fresh23_gpu0';rows=[]
files=sorted(p for shard in [0,1] for p in (fresh/f'expert{shard}').glob('*.npy'))
with np.load('/workspace/komap_model1/outputs/model329_fullstats_width4000_uniform_group_oof/valid_ensemble/probabilities.npz',allow_pickle=False) as archive:
 for path in files:
  actual=np.load(path,allow_pickle=False);name=path.stem;expected=.5*archive[name].astype(np.float32)
  for seed,folds in [(1,[2,4]),(2,[1,4])]:
   parent=r/'seed1_maps' if seed==1 else old/'fold_maps';shard=next(parent/f'shard{i}' for i in [0,1] if (parent/f'shard{i}/{name}_fold1.npy').exists())
   value=sum(np.load(shard/f'{name}_fold{fold}.npy',allow_pickle=False).astype(np.float32)*.5 for fold in folds).astype(np.float16).astype(np.float32)
   expected=expected+.25*value
  expected=expected[[1,2,0,3]];assert actual.shape==expected.shape and np.isfinite(actual).all()
  delta=np.abs(actual-expected);priority=np.array([2,0,1,3],np.uint8)
  a=priority[actual[priority].argmax(0)];b=priority[expected[priority].argmax(0)]
  rows.append({'name':name,'probabilities_exact':bool(np.array_equal(actual,expected)),'max_probability_difference':float(delta.max()),'changed_probability_values':int(np.count_nonzero(delta)),'expert_label_changed_pixels':int(np.count_nonzero(a!=b))})
result={'completed_images':len(rows),'expected_images':20,'gt_read':False,'is_final_validation':False,'images':rows};(fresh/'partial_probability_parity.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
