"""FP16 storage for selected E tensors; preserve BN statistics and all originals."""
from pathlib import Path
import json,hashlib,copy,shutil,os,argparse
os.environ.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
import torch
ROOT=Path(__file__).resolve().parent;torch.set_num_threads(2)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def main():
 parser=argparse.ArgumentParser();parser.add_argument('--parent-package',type=Path,required=True);parser.add_argument('--original-assets',type=Path,required=True);parser.add_argument('--refiner',type=Path,required=True);parser.add_argument('--output',type=Path,default=ROOT/'assets');parser.add_argument('--manifest-output',type=Path,default=ROOT/'selected_assets.json');args=parser.parse_args();parent=args.parent_package
 out=args.output;out.mkdir(exist_ok=False);manifest=json.loads((ROOT/'original_assets.json').read_text());original=copy.deepcopy(manifest);rows=[]
 for row in manifest['checkpoints']:
  relative=row['path'];source=(parent/'Model349_exh14' if row['namespace']=='R' else args.original_assets)/relative
  if relative=='refiner/epoch8.pt':source=args.refiner
  assert sha(source)==row['sha256']
  if row['namespace']=='R':continue
  dest=out/relative;dest.parent.mkdir(parents=True,exist_ok=True)
  if relative=='refiner/epoch8.pt':dest.symlink_to(source);continue
  checkpoint=torch.load(source,map_location='cpu',weights_only=False);state=checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint;converted=[]
  for key,tensor in list(state.items()):
   if isinstance(tensor,torch.Tensor) and tensor.dtype==torch.float32 and not key.endswith(('running_mean','running_var')):
    half=tensor.half();assert torch.isfinite(tensor).all() and torch.isfinite(half).all();converted.append({'key':key,'elements':tensor.numel(),'max_abs_error':float((half.float()-tensor).abs().max()) if tensor.numel() else 0.});state[key]=half
  assert converted;torch.save(checkpoint,dest);reloaded=torch.load(dest,map_location='cpu',weights_only=False);again=reloaded['state_dict'] if 'state_dict' in reloaded else reloaded;assert list(again)==list(state)
  for key in state:assert torch.equal(again[key],state[key]) and again[key].dtype==state[key].dtype
  original_row=copy.deepcopy(row);row['bytes']=dest.stat().st_size;row['sha256']=sha(dest);rows.append({'path':relative,'original_bytes':original_row['bytes'],'stored_bytes':row['bytes'],'original_sha256':original_row['sha256'],'stored_sha256':row['sha256'],'converted_tensors':converted});assert sha(source)==original_row['sha256'];print(relative,row['bytes'],flush=True)
 for row in manifest['metadata']:
  if row['namespace']=='R':continue
  source=args.original_assets/row['path'];assert sha(source)==row['sha256'];dest=out/row['path'];dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,dest)
 manifest['bytes']=sum(x['bytes'] for x in manifest['checkpoints']);manifest['storage']='Selected E FP16 tensors; BN running statistics unchanged; inference restored to FP32';args.manifest_output.write_text(json.dumps(manifest,indent=2)+'\n')
 assert len(rows)==8
 (ROOT/'build_completion.json').write_text(json.dumps({'checkpoints':23,'original_bytes':original['bytes'],'stored_bytes':manifest['bytes'],'E_converted_checkpoints':8,'originals_unchanged':True,'BN_running_statistics_preserved':True,'inference_precision':'FP32','validation_passed':False,'rows':rows},indent=2)+'\n')
if __name__=='__main__':main()
