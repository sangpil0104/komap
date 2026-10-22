import os
os.environ.update(CUDA_VISIBLE_DEVICES="",OMP_NUM_THREADS="2",OPENBLAS_NUM_THREADS="2")
from pathlib import Path
import json,hashlib
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parent
DATA=Path("/workspace/komap_model1")
PALETTE=np.array([(153,127,76),(25,76,153),(76,178,76),(204,204,204)],np.uint8)
rows=[];confusion=np.zeros((4,4),np.int64);counts=np.zeros((4,4),np.int64)
# columns: GT pixels, E error pixels, E errors with any family correct, E errors with unanimous family label
for fold in range(1,5):
 paths=[DATA/f"outputs/model329_fullstats_width4000_uniform_group_oof/fold{fold}_oof/probabilities.npz"]+[DATA/f"outputs/model392_widthnorm/seed{s}/fold{fold}_oof/valid_probabilities.npz" for s in [1,2]]
 archives=[np.load(p) for p in paths]
 try:
  assert all(set(z.files)==set(archives[0].files) for z in archives)
  for name in sorted(archives[0].files):
   parts=[z[name].astype(np.float32) for z in archives];prob=.5*parts[0]+.25*parts[1]+.25*parts[2]
   rgb=np.asarray(Image.open(DATA/f"data/train/masks/{name}_mask.png").convert("RGB"));truth=np.full(rgb.shape[:2],255,np.uint8)
   for c,color in enumerate(PALETTE):truth[np.all(rgb==color,axis=-1)]=c
   assert np.all(truth<4)
   pred=prob.argmax(0);family=np.stack([x.argmax(0) for x in parts]);any_correct=(family==truth).any(0);unanimous=(family==family[0]).all(0);error=pred!=truth
   cf=np.bincount((truth.astype(np.int64)*4+pred).ravel(),minlength=16).reshape(4,4);confusion+=cf
   n=np.zeros((4,4),np.int64)
   for c in range(4):
    mask=truth==c;n[c]=[np.count_nonzero(mask),np.count_nonzero(mask&error),np.count_nonzero(mask&error&any_correct),np.count_nonzero(mask&error&unanimous)]
   counts+=n;rows.append({"name":name,"fold":fold,"counts":n.tolist(),"confusion":cf.tolist()});print(len(rows),name,flush=True)
 finally:
  for z in archives:z.close()
assert len(rows)==70
result={"scope":"train70 grouped OOF error diagnostic only; no candidate or validation score","class_order":["Al","Al3Ni","Primary Si","Eutectic Si"],"count_columns":["GT pixels","E errors","E errors where any individual family is correct","E errors with unanimous family prediction"],"counts":counts.tolist(),"confusion_GT_by_pred":confusion.tolist(),"rows":rows,"valid_read":False,"test_read":False,"gpu_used":False}
(ROOT/"result.json").write_text(json.dumps(result,indent=2)+"\n")
(ROOT/"completion.json").write_text(json.dumps({"images":70,"finished":True})+"\n")
print(json.dumps({"counts":counts.tolist()},indent=2))
