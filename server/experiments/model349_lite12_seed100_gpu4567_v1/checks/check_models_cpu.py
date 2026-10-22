"""Load all twelve packaged inference models on CPU; no forward/backward call."""
import sys,gc,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common import ROOT, read_json, write_json
import torch
import numpy as np
from tqdm import tqdm
from engine import frozen
from komap_seg.checkpoint import load_model
from b4_zcdf import load_member
from tent_style_search import load_aspp
from model253_highres_scale_finetune import Stride1NativeStemASPP

torch.set_num_threads(2)
records=[]
for row in tqdm(read_json(ROOT/'local_weights.json')['checkpoints'],desc='CPU load 12 inference models',unit='model'):
    relative=row['relative']; path=ROOT/relative
    payload=torch.load(path,map_location='cpu',weights_only=True)
    state=payload.get('state_dict',payload.get('model',payload))
    assert all(t.dtype==torch.float16 for t in state.values() if torch.is_floating_point(t))
    assert all(torch.isfinite(t).all() for t in state.values() if torch.is_floating_point(t))
    if 'fold' in relative:
        # The portable loader expects args.json alongside weights. Use a temporary
        # view, exactly as the runtime worker does, without modifying weights.
        from engine import make_weight_view
        import tempfile
        with tempfile.TemporaryDirectory() as temp:
            view=make_weight_view(Path(temp)/'view',{})
            model=load_member(str(view.parent/Path(relative).parent),'cpu')['model']
    elif 'model336' in relative: model=load_model(path,torch.device('cpu'))
    elif 'model269' in relative:
        model=Stride1NativeStemASPP(); model.load_state_dict(payload['model'])
    else: model=load_aspp(path,torch.device('cpu'))
    assert next(model.parameters()).dtype==torch.float32
    records.append(dict(path=relative,strict_load=True,stored_float_dtype='float16',runtime_dtype='float32'))
    del payload,state,model; gc.collect()
write_json(ROOT/'checks/cpu_model_load.json',dict(training_started=False,forward_calls=0,gpu_used=False,models=records))
print('PASS: twelve FP16 checkpoints load strictly into FP32 CPU models; zero forward/backward calls.')
