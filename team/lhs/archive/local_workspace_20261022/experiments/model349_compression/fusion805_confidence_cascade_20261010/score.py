"""Separate gate calculation from frozen per-image PNG metrics."""
from pathlib import Path
import json,numpy as np
ROOT=Path(__file__).resolve().parent
def main():
 dec=json.loads((ROOT/'decisions.json').read_text());data=json.loads((ROOT.parent/'fusion805_compression_round3_20261010/result.json').read_text());assert dec['names']==data['names'];keys=data['keys'];v=np.array(data['class_iou'])[1];refs=['original27','reference25','reference23','reference23_cache'];rows=[]
 for cfg in dec['configs']:
  values=np.array([v[keys.index('reference23_cache' if cfg['use_full_lite23'][n] else dec['cheap']),j] for j,n in enumerate(dec['names'])]);image=values.mean(1);gates={}
  for ref in refs:
   delta=(image-v[keys.index(ref)].mean(1))*100;gates[ref]={'mean_pp':float(delta.mean()),'worst_pp':float(delta.min()),'pass':bool(delta.mean()>=-.01-1e-10 and delta.min()>=-.1-1e-10)}
  rows.append(dict(cfg,mean_percent=float(values.mean()*100),gates=gates,quality_pass=all(x['pass'] for x in gates.values()),compute_reduction_candidate=cfg['cheap_images']>=5))
 result={'configurations':rows,'repeated_valid':True,'prediction_raw_inference_verified':False,'speed_measured':False,'weights_retained':23,'selection_bias_remaining':True};(ROOT/'result.json').write_text(json.dumps(result,indent=2)+'\n')
 for row in rows:print(row['key'],row['cheap_images'],row['mean_percent'],row['quality_pass'],row['gates']['reference23'])
if __name__=='__main__':main()
