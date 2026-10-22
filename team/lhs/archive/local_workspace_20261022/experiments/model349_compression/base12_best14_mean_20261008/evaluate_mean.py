"""Fixed 50:50 fused probability blend before shared postprocess; cached CPU replay."""
import os
os.environ.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1')
import ast,hashlib,importlib.util,json,sys,time,types
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent;WORK=ROOT.parents[2];REVIEW=ROOT.parent/'exhaustive_review_20261008'
SCRATCH=Path('/tmp/claude-501/-Users-hyunseok-orca-workspaces-KoMaP-gurnard/8c455a47-b7eb-4c6e-b73d-ddf6da07fc20/scratchpad')
os.environ['SLIM']=str(SCRATCH/'slim');os.environ['GT']=str(SCRATCH/'gt/valid/masks')
source=WORK/'Model349_reproduction_code/experiments/historical_scripts/highres1024_search.py'
for node in ast.parse(source.read_text()).body:
 if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='MATRIX' for t in node.targets):matrix=np.array(ast.literal_eval(node.value.args[0]),dtype=np.float32);break
stub=types.ModuleType('highres1024_search');stub.MATRIX=matrix;sys.modules['highres1024_search']=stub
os.chdir(WORK/'Model349_lite12_base')
p=ROOT.parent/'exhaustive_20261008/exhaustive_local.py';spec=importlib.util.spec_from_file_location('cached',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
# Preserve every fusion operation. Extract the fields at the exact postprocess boundary.
fn=next(n for n in ast.parse(p.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='predict')
assert isinstance(fn.body[-1],ast.Return)
fn.name='mixed_fields';fn.body[-1]=ast.Return(value=ast.Tuple(elts=[ast.Name(id=k,ctx=ast.Load()) for k in ['item','mixed','s']],ctx=ast.Load()))
code=ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[]));ns=dict(m.__dict__);exec(compile(code,str(p),'exec'),ns);mixed_fields=ns['mixed_fields']
import cv2
cv2.setNumThreads(2)
configs={'base12':(0,5,15,1,0,1,0,0,7),'best14':(12,3,13,1,1,1,0,1,7)}
ious={k:[] for k in ['base12','best14','mean50']};ref=np.load(REVIEW/'replay_iou.npz');names=m.NAMES
assert list(ref['names'])==names
postkeys=['prior_scale','beta_scale','al_delta','eut_delta','gaussian_blend','eut_edge_blend','cleanup_threshold']
input_hashes={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in [p,source]}
t=time.monotonic()
for index,name in enumerate(names):
 z=m.load(name);a,pa,sa=mixed_fields(z,configs['base12']);b,pb,sb=mixed_fields(z,configs['best14'])
 assert all(sa[k]==sb[k] for k in postkeys) and a['params']==b['params']
 assert np.isfinite(pa).all() and np.isfinite(pb).all()
 fields={'base12':pa,'best14':pb,'mean50':np.float32(.5)*pa+np.float32(.5)*pb}
 for key,prob in fields.items():
  pred=m.route.predict(a,prob,sa,z['weights'],exact=True);gt=z['gt'];values=[]
  for cid in range(4):
   union=np.count_nonzero((pred==cid)|(gt==cid));values.append(np.count_nonzero((pred==cid)&(gt==cid))/union if union else 0.)
  ious[key].append(values)
  if key in configs:
   assert np.array_equal(np.array(values),ref[key][index]),(name,key,'endpoint scoring changed')
   old=cv2.imread(str(REVIEW/'cached_masks'/key/(name+'.png')),cv2.IMREAD_UNCHANGED);assert np.array_equal(old,pred),(name,key,'endpoint mask changed')
  else:
   out=ROOT/'cached_masks/mean50';out.mkdir(parents=True,exist_ok=True);cv2.imwrite(str(out/(name+'.png')),pred)
 print(name,round(time.monotonic()-t,1),flush=True)
arr={k:np.array(v) for k,v in ious.items()};np.savez(ROOT/'iou.npz',names=np.array(names),**arr)
mean=arr['mean50'];base=arr['base12'];other=arr['best14'];delta=(mean.mean(1)-base.mean(1))*100;worst=int(delta.argmin())
result={'scope':'cached probability replay; NOT raw image inference / independent generalization','blend_best14_weight':0.5,'blend_location':'final fused 4-class mixed probabilities, before common prior/spatial/cleanup','unique_checkpoints':16,'endpoint_iou_and_masks_exact':True,'miou_percent':{k:float(v.mean()*100) for k,v in arr.items()},'class_order':['Al3Ni','Primary Si','Al','Eutectic Si'],'class_iou_percent':(mean.mean(0)*100).tolist(),'vs_base12':{'mean_delta_pp':float(delta.mean()),'class_delta_pp':((mean-base).mean(0)*100).tolist(),'worst_image':names[worst],'worst_delta_pp':float(delta[worst]),'quality_gate_pass':bool(delta.mean()>=-.01 and delta.min()>=-.10),'rising_images':int((delta>0).sum()),'falling_images':int((delta<0).sum())},'per_image':{n:{'base12_percent':float(base[i].mean()*100),'best14_percent':float(other[i].mean()*100),'mean50_percent':float(mean[i].mean()*100),'delta_vs_base12_pp':float(delta[i]),'class_delta_vs_base12_pp':((mean[i]-base[i])*100).tolist()} for i,n in enumerate(names)},'source_hashes':input_hashes,'elapsed_cpu_replay_seconds':time.monotonic()-t,'raw_inference_seconds':None,'checkpoint_bytes':None,'test_accuracy':None,'independent_generalization_verified':False}
(ROOT/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
lines=['# base12 + best14 1:1 확률 평균 결과','','공통 후처리 직전 최종 융합 확률을 1:1 평균했다. 기존 확률 캐시를 쓴 CPU 재생이며 원본 영상 추론/독립 일반화 검증이 아니다. 양 끝 단독 구성의 20장 마스크와 IoU가 이전 재채점과 정확히 같음을 확인했다. 비율 탐색은 하지 않았다.','','| 구성 | 가중치 | 캐시 image mIoU |','|---|---:|---:|']
for k in arr:lines.append(f'| {k} | {16 if k=="mean50" else 12 if k=="base12" else 14} | {result["miou_percent"][k]:.7f}% |')
lines+=['',f'base12 대비 평균 {delta.mean():+.7f}%p, 최악 {delta.min():+.7f}%p ({names[worst]}). 품질 기준 통과: {result["vs_base12"]["quality_gate_pass"]}. 평균 손실≤0.01%p/최악 손실≤0.10%p. 상승 {int((delta>0).sum())}장, 악화 {int((delta<0).sum())}장.','','| 이미지 | base12 | best14 | 1:1 평균 | base12 대비 (%p) |','|---|---:|---:|---:|---:|']
for n,row in sorted(result['per_image'].items(),key=lambda x:-x[1]['delta_vs_base12_pp']):lines.append(f"| {n} | {row['base12_percent']:.5f}% | {row['best14_percent']:.5f}% | {row['mean50_percent']:.5f}% | {row['delta_vs_base12_pp']:+.5f} |")
lines+=['','고유 가중치가 16개로 늘어 경량화 후보가 아니다. 공통 경로를 재사용한 실제 코드/속도는 검증하지 않았다. 80.2 후보 자체가 같은 valid로 선택되었고 평균판 평가도 동일 valid이므로, 품질 기준을 통과하더라도 신규 시편 개선을 증명하지 않는다. 모델 파일 tensor 평균이나 두 mIoU 점수의 산술 평균과 구분한다. 기존 base/릴리스는 유지한다.','']
(ROOT/'report.md').write_text('\n'.join(lines));assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in input_hashes.items());print(json.dumps({k:result[k] for k in ['miou_percent','vs_base12','unique_checkpoints']},ensure_ascii=False,indent=2))
