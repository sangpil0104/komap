import torch,json,hashlib
from pathlib import Path
D=Path(__file__).resolve().parent;K=Path('/workspace/komap_model1/outputs')
paths={**{f'R329_f{f}':K/f'model329_fullstats_width4000_uniform_group_oof/fold{f}/last.pt' for f in (1,3,4)},**{f'B4ws_f{f}':K/f'claude_b4_329_warmstart_bnfreeze/fold{f}/last.pt' for f in (1,3)}}
records={};schemas={};encoder_sha={}
for tag,p in paths.items():
 raw=torch.load(p,map_location='cpu',weights_only=False)
 sd=raw
 for key in ('state_dict','model_state_dict','model','net'):
  if isinstance(raw,dict) and key in raw and isinstance(raw[key],dict):sd=raw[key];break
 tensors={k:v for k,v in sd.items() if torch.is_tensor(v)};assert tensors,tag
 enc={k:v for k,v in tensors.items() if k.startswith(('encoder.','module.encoder.'))};assert enc,(tag,list(tensors)[:10])
 nbytes=lambda x:sum(v.numel()*v.element_size() for v in x.values())
 digest=hashlib.sha256()
 for k,v in sorted(enc.items()):digest.update(k.encode());digest.update(v.numpy().tobytes())
 schemas[tag]={k:(list(v.shape),str(v.dtype)) for k,v in enc.items()};encoder_sha[tag]=digest.hexdigest();records[tag]={'checkpoint':str(p),'file_bytes':p.stat().st_size,'tensor_bytes':nbytes(tensors),'encoder_tensor_bytes':nbytes(enc),'decoder_and_head_bytes':nbytes(tensors)-nbytes(enc),'encoder_state_sha':digest.hexdigest(),'encoder_keys':len(enc),'other_keys':len(tensors)-len(enc)}
 del raw,sd,tensors,enc
report={'models':records,'encoder_shapes_compatible':all(s==next(iter(schemas.values())) for s in schemas.values()),'all_encoder_values_identical':len(set(encoder_sha.values()))==1,'groups':{}}
for group,keys in [('R329',[f'R329_f{f}' for f in (1,3,4)]),('B4ws',[f'B4ws_f{f}' for f in (1,3)]),('all_five',list(records))]:
 before=sum(records[k]['tensor_bytes'] for k in keys);after=records[keys[0]]['encoder_tensor_bytes']+sum(records[k]['decoder_and_head_bytes'] for k in keys)
 report['groups'][group]={'before_tensor_bytes':before,'shared_encoder_tensor_bytes':after,'theoretical_saved_bytes':before-after,'theoretical_saved_percent':100*(before-after)/before,'encoders_after':1,'decoders_after':len(keys),'ensemble_output_members_after':len(keys),'performance_preservation_verified':False,'requires_decoder_realignment_training':True}
report['interpretation']='Architecture and tensor-storage audit only. Different trained encoders cannot be shared losslessly; separate head outputs are still retained. One file is not one model. Exact deployed checkpoint/protocol equivalence must be checked before any training or release.'
(D/'shared_encoder_inventory.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
