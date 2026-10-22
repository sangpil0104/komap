"""Verify direct family assembly and F against the independent fresh reference."""
from pathlib import Path
import json,sys,importlib.util
import numpy as np,torch
from PIL import Image
K=Path.home()/'komap';O=K/'lite16_N3_adopted_20261013';F=K/'codex_lite16_N3_L2_fresh_valid_20261013';P=K/'Model349_lite20_b4ws_bundle/Model349_fusion805'
spec=importlib.util.spec_from_file_location('fu',P/'infer_fusion.py');fu=importlib.util.module_from_spec(spec);spec.loader.exec_module(fu);torch.set_num_threads(2);ref=fu.Residual().eval();ref.load_state_dict(torch.load(O/'assets/refiner/epoch8.pt',map_location='cpu',weights_only=False)['state_dict']);man=json.loads((F/'R_fresh/input_manifest.json').read_text());rt=lambda a:a.astype(np.float16).astype(np.float32);rows=[]
for row in man['images']:
 n=row['name'];load=lambda tag:np.load(F/f'fresh_probabilities/{tag}/{n}.npy').astype(np.float32)
 b=load('b4ws');h1=rt(.5*load('N3')+.5*load('hr_s1_f4'));h2=rt(.5*load('hr_s2_f1')+.5*load('hr_s2_f4'));e=(.5*b+.25*h1+.25*h2)[[1,2,0,3]];assert np.array_equal(e,np.load(F/f'N3/expert/{n}.npy'))
 gray=torch.from_numpy(fu.prepare_gray(np.asarray(Image.open(K/f'data/valid/images/{n}_image.png').convert('L'),np.float32)/255)[None,None]);r=np.asarray(Image.open(F/f'R_fresh/predicted_labels/{n}.png'))
 with torch.no_grad():v,_=ref(torch.from_numpy(np.log(np.maximum(e,1e-7))[None]),gray)
 ex=fu.labels(e);fl=fu.labels(v.numpy()[0]);pred=np.where(fl==2,2,np.where(np.isin(ex,[0,2]),ex,r)).astype(np.uint8);assert np.array_equal(pred,np.asarray(Image.open(F/f'N3/fusion/predicted_labels/{n}.png')))
 rows.append(n)
(O/'assembly_verification.json').write_text(json.dumps({'images':rows,'expert_probability_arrays_exact':True,'final_masks_exact':True,'new_GPU_inference':False,'reference':'independent fresh R/E/F N3 seed414','remaining_portable_GPU_smoke':'not run while R326 training'},indent=2));print('ALL20_EXPERT_AND_FINAL_EXACT')
