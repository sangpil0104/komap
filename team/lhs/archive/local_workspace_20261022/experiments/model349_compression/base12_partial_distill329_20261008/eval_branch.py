"""Raw-image student329 prediction plus explicitly cached other-component fusion."""
import os
os.environ.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1')
import argparse,importlib.util,json,subprocess,sys,time
from pathlib import Path
import numpy as np
import torch
import cv2
ROOT=Path('/workspace/model349_compress/base12_partial_distill329_20261008')
OLD=Path('/workspace/model349_compress/audit_20261008/runs/cpu2_fp16_12')
def module(path):
 spec=importlib.util.spec_from_file_location('infer',path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def main(key):
 assert os.environ['CUDA_VISIBLE_DEVICES']=='7';torch.set_num_threads(2);cv2.setNumThreads(2)
 bundle=ROOT/'bundles'/key;m=module(bundle/'infer_lite.py');manifest=json.loads((OLD/'input_manifest.json').read_text());manifest['config']=m.load_config(bundle/'configs/model349_lite.yaml')
 work=ROOT/'cached_diagnostics'/key;work.mkdir(parents=True,exist_ok=False);m.write_json(work/'input_manifest.json',manifest);(work/'probabilities').mkdir()
 for c in ['326','base','336','269']:(work/'probabilities'/c).symlink_to(OLD/'probabilities'/c,target_is_directory=True)
 out=work/'probabilities/329';out.mkdir();sys.path.insert(0,str(bundle/'src'));from b4_zcdf import load_member,load_gray,predict_probs
 member=load_member(str(bundle/'weights/model329_fullstats_width4000_uniform_group_oof/fold1'),'cuda');t=time.monotonic();uuid=os.environ['PILOT_GPU_UUID']
 for row in manifest['images']:
  assert m.sha(row['path'])==row['sha256'];processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader'],text=True)
  if sum(uuid in line for line in processes.splitlines())>1:raise RuntimeError('Foreign GPU7 compute; stopping own branch inference')
  values=predict_probs([member],load_gray(row['path']),tta=False,device='cuda').cpu().numpy().astype(np.float16)[m.TEAM_TO_MAIN];np.save(out/(row['name']+'.npy'),values);print(key,row['name'],flush=True)
 m.write_json(out/'complete.json',{'component':'329','images':20,'elapsed_seconds':time.monotonic()-t})
 # Release model before CPU fusion; this is not a complete-model wall-time measurement.
 del member;torch.cuda.empty_cache();m.assemble(work,Path('/workspace/komap_model1/data/valid/masks'))
 m.write_json(work/'scope.json',{'scope':'student329 raw image inference plus frozen other4 components cached fusion','raw_full_model_inference':False,'checkpoint_count':9,'branch_seconds':json.loads((out/'complete.json').read_text())['elapsed_seconds']})
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--candidate',required=True);main(p.parse_args().candidate)
