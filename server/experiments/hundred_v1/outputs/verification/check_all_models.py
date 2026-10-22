"""Real-data forward/backward/optimizer checks for all 100 configs; no saved models."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.dont_write_bytecode=True
sys.path[:0]=[str(ROOT),str(ROOT/'vendor')]
import gc,json
import torch
from tqdm import tqdm
from alphase.data.splits import select_rows
from alphase.utils import amp_context
from hundredlab.catalog import catalog
from hundredlab.workspace import config
from hundredlab.models import build_model,configure_trainable
from hundredlab.optimization import optimizer_for
from hundredlab.dataset import HundredDataset
from hundredlab.losses import segmentation_loss

torch.set_num_threads(4)
device=torch.device('cuda:0')
results=[]
for name in tqdm(catalog(),desc='100 model contracts',unit='model'):
    cfg=config(name)
    train,_=select_rows(cfg,'official')
    largest=max(train,key=lambda r:int(r['width'])*int(r['height']))
    model=build_model(cfg).to(device)
    record=dict(experiment=name,parameters=sum(p.numel() for p in model.parameters()),stages=[])
    for si,stage in enumerate(cfg['train']['stages']):
        configure_trainable(model,cfg,stage)
        dataset=HundredDataset([largest],cfg,stage,si)
        batch=dataset[1]
        x=batch['detail'][None].to(device);t=batch['target'][None].to(device)
        optimizer,scheduler=optimizer_for(model,cfg,stage)
        model.train();optimizer.zero_grad(set_to_none=True)
        with amp_context(device,cfg['train']['precision']):out=model(x)
        loss,_=segmentation_loss(out,t,cfg['loss'])
        assert torch.isfinite(loss),name
        loss.backward()
        trainable=[(n,p) for n,p in model.named_parameters() if p.requires_grad]
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for n,p in trainable),(name,[n for n,p in trainable if p.grad is None])
        assert any(p.grad.abs().sum()>0 for n,p in trainable),name
        optimizer.step();scheduler.step()
        record['stages'].append(dict(shape=list(x.shape),loss=float(loss.detach()),trainable_parameters=sum(p.numel() for n,p in trainable)))
        del optimizer,scheduler,x,t,out,loss,trainable,dataset,batch
    results.append(record)
    del model
    gc.collect();torch.cuda.empty_cache()
(ROOT/'outputs/verification/all_model_contracts.json').write_text(json.dumps(results,indent=2))
print('All 100 models passed native/patch data, loss, gradient, and optimizer checks.')
