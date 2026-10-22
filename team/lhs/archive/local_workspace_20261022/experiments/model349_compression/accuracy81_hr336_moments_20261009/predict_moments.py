"""Generate HR336 per-image moment candidates without opening any GT."""
import os
os.environ.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1')
from pathlib import Path
import sys,json,time,subprocess
import numpy as np
import torch
from moments import Moments
from fields_io import SOURCE,sha,write
ROOT=Path(__file__).resolve().parent;PRIOR=ROOT.parent/'accuracy81_residual_20261008/runs/teacher_valid20'
sys.path.insert(0,str(SOURCE/'src'))
from komap_seg.checkpoint import load_model
from komap_seg.dataset import build_eval_transform,load_image_rgb
from komap_seg.inference import predict_probs_tta
def main():
 torch.set_num_threads(2);manifest=json.loads((PRIOR/'input_manifest.json').read_text());cfg=manifest['config'];h=cfg['inference']['hrnet']
 assert len(manifest['images'])==20;checkpoint=SOURCE/'weights/model336_hrnet_best/last.pt';before=sha(checkpoint)
 inventory=json.loads((ROOT/'bn_inventory.json').read_text());assert inventory['checkpoint_sha256']==before
 model=load_model(checkpoint,torch.device('cuda:0'));collector=Moments(model);transform=build_eval_transform()
 out=ROOT/'probabilities';out.mkdir();(ROOT/'moment_stats').mkdir()
 for kind in ['baseline','alpha025','alpha1']:(out/kind).mkdir()
 rows=[];start=time.monotonic();baseline_exact=0
 for row in manifest['images']:
  name=row['name'];assert sha(Path(row['path']))==row['sha256'];collector.restore()
  image=transform(image=load_image_rgb(Path(row['path'])))['image'].float()
  def predict():return predict_probs_tta(model,image,patch_size=h['patch'],overlap=h['overlap'],batch_size=h['batch_size'],tta=False,input_scale=h['input_scale'],device=torch.device('cuda:0'),tta_mode=h['tta']).numpy()
  t=time.monotonic();collector.begin();value=predict();collector.end();collect_seconds=time.monotonic()-t
  baseline=value.astype(np.float16)[[1,2,0,3]];expected=np.load(PRIOR/'probabilities/336'/f'{name}.npy')
  if not np.array_equal(baseline,expected):
   write(ROOT/'baseline_mismatch.json',{'image':name,'pixels_changed':int(np.count_nonzero(baseline!=expected)),'max_abs_diff':float(np.max(np.abs(baseline.astype(float)-expected.astype(float))))});raise AssertionError('Original HR336 probability replay differs; investigate before scoring')
  baseline_exact+=1;np.save(out/'baseline'/f'{name}.npy',baseline)
  names=sorted(collector.stats);means=[];vars=[];counts=[];offsets=[0]
  for n in names:
   s,q,count=collector.stats[n];mean=s/count;variance=(q/count-mean.square()).clamp_min(0)
   means.append(mean.cpu().numpy());vars.append(variance.cpu().numpy());counts.append(count);offsets.append(offsets[-1]+len(mean))
  np.savez_compressed(ROOT/'moment_stats'/f'{name}.npz',names=np.asarray(names),mean=np.concatenate(means),variance=np.concatenate(vars),counts=np.asarray(counts),offsets=np.asarray(offsets))
  timing={}
  for kind,alpha in [('alpha025',.25),('alpha1',1.)]:
   collector.apply(alpha);t=time.monotonic();value=predict();timing[kind]=time.monotonic()-t
   assert np.isfinite(value).all();np.save(out/kind/f'{name}.npy',value.astype(np.float16)[[1,2,0,3]])
  collector.restore()
  apps=subprocess.check_output(['nvidia-smi','-i',os.environ.get('KOMAP_PHYSICAL_GPU','3'),'--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip().splitlines();assert len(apps)<=1,'Foreign compute; terminate our inference only'
  rows.append({'image':name,'original_probability_exact':True,'observed_bn_layers':len(names),'unobserved_bn_layers':[n for n in collector.layers if n not in collector.stats],'collect_seconds':collect_seconds,'candidate_predict_seconds':timing,'moment_stats_sha256':sha(ROOT/'moment_stats'/f'{name}.npz')})
  print('MOMENTS',name,len(names),flush=True)
 assert baseline_exact==20 and sha(checkpoint)==before;collector.assert_original();collector.unchanged_non_bn()
 write(ROOT/'gpu_completion.json',{'images':20,'baseline_probability_maps_exact':True,'original_bn_buffers_restored':True,'parameters_non_bn_buffers_unchanged':True,'checkpoint_unchanged':True,'checkpoint_sha256':before,'valid_gt_opened':False,'test_used':False,'actual_tta':'identity','elapsed_seconds':time.monotonic()-start,'rows':rows,'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated()})
if __name__=='__main__':main()
