"""Exercise actual lite20 R routing/launcher with stub predictions and processes."""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import importlib.util,shutil,sys,tempfile
import numpy as np
from PIL import Image
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from experiment import ROOT,read_json,write_json

spec=importlib.util.spec_from_file_location('lite20_infer',ROOT/'frozen/infer_lite.py')
inference=importlib.util.module_from_spec(spec);spec.loader.exec_module(inference)
cfgpath=ROOT/'compact_source/selected_R_lite20.yaml';config=inference.load_config(cfgpath)
assert set(inference.high_candidates(config))=={'restart51','pairwise'}
assert inference.b4_fold_ids(config,'322')==[3]
assets=read_json(ROOT/'compact_source/selected_assets_lite20.json')
expected={r['path'] for r in assets['checkpoints'] if r['namespace']=='R'}
assert len(expected)==11

with tempfile.TemporaryDirectory(prefix='lite20_routing_',dir=ROOT/'checks') as directory:
    work=Path(directory);inputs=work/'images';inputs.mkdir();out=work/'base';out.mkdir()
    rows=[]
    for i,levels in enumerate([(152,153,152,153),(174,175,175,175),(159,160,159,160),
                                (100,100,100,100),(161,162,161,162),(155,156,155,156)]):
        a=np.array(levels,np.uint8).reshape(2,2)
        path=inputs/f'case{i}_image.png';Image.fromarray(np.repeat(a[...,None],3,axis=-1)).save(path)
        rows.append(dict(name=f'case{i}',path=str(path)))
    chosen=[]
    def adapted(checkpoint,*args):chosen.append(Path(checkpoint).parent.name);return object()
    def predict(model,im,*args):return np.full((4,2,2),.25,np.float32)
    with patch('tent_style_search.adapt_model',side_effect=adapted), \
         patch('style_tta_search.predict_array',side_effect=predict), \
         patch('style_tta_search.transform_image',side_effect=lambda raw,*args:raw), \
         patch('torch.cuda.empty_cache'):
        inference.base_worker(rows,out,'cpu',config,work/'weights')
    assert chosen==['restart51_aspp_768_finetune','restart51_aspp_768_finetune',
                    'restart61_pairwise_margin','restart51_aspp_768_finetune',
                    'restart51_aspp_768_finetune','restart51_aspp_768_finetune']
    fake=work/'package';fake.mkdir();(fake/'configs').mkdir()
    shutil.copy2(ROOT/'frozen/configs/calibration.json',fake/'configs/calibration.json')
    for name in expected:
        p=fake/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('fixture; never loaded')
    launches=[];queried=[]
    class Process:
        def __init__(self,command,**kwargs):launches.append((command,kwargs['env']['CUDA_VISIBLE_DEVICES']))
        def poll(self):return 0
    def query(command,**kwargs):queried.append(command[2]);return '0, 0\n'
    targets=[f'FAKE-UUID-GPU-{g}' for g in range(2,8)]
    args=SimpleNamespace(config=cfgpath,input=inputs,output=work/'release',gpus=','.join(targets),
                         limit=None,gt=None,keep_probabilities=True)
    with patch.object(inference,'ROOT',fake),patch.object(inference.subprocess,'Popen',Process), \
         patch.object(inference.subprocess,'check_output',side_effect=query), \
         patch.object(inference.time,'sleep'),patch.object(inference,'assemble',return_value=[]):
        inference.run(args)
    manifest=read_json(args.output/'input_manifest.json')
    assert set(manifest['required_weights'])==expected
    assert len(launches)==7 and {gpu for command,gpu in launches}==set(targets)
    assert set(queried)==set(targets)
    assert all(command[4]=='--_worker' for command,gpu in launches)
write_json(ROOT/'checks/lite20_routing.json',dict(ok=True,training_started=False,gpu_inference_started=False,
    predictions_stubbed=True,processes_stubbed=True,pruned_band_cases=6,required_r_weights=11,
    components_launched=7,physical_gpu_uuid_targets=[2,3,4,5,6,7]))
print('PASS: lite20 brightness routing, R11 inventory and GPU2~7 launcher propagation; stubs only.')
