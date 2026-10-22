"""Read-only mixing of saved BN statistics with per-input statistics at inference."""
import torch


def predict(model,x,spec):
    amount=float(spec['amount']);scope=spec.get('scope','encoder')
    if not 0<=amount<=1:raise ValueError('Normalization blend must be in [0,1]')
    if scope not in ('encoder','decoder','all'):raise ValueError(scope)
    statistic=spec.get('statistic','both');subset=spec.get('subset','all')
    if statistic not in ('both','mean','variance'):raise ValueError(statistic)
    if subset not in ('all','last','except_last','high_resolution','low_resolution'):raise ValueError(subset)
    if subset!='all' and scope!='decoder':raise ValueError('Decoder subsets require decoder scope')
    last_depth=model.decoder.depth if subset!='all' else None
    if amount==0:return model(x)
    def hook(module,inputs,output):
        if module.training:raise RuntimeError('Frozen inference requires eval mode')
        value=inputs[0].float()
        current_var,current_mean=torch.var_mean(value,dim=(-2,-1),correction=0,keepdim=True)
        saved_mean=module.running_mean[None,:,None,None].float()
        saved_var=module.running_var[None,:,None,None].float()
        mean=(1-amount)*saved_mean+amount*current_mean
        variance=(1-amount)*saved_var+amount*current_var+amount*(1-amount)*(saved_mean-current_mean).square()
        if statistic=='mean':variance=saved_var
        elif statistic=='variance':
            mean=saved_mean;variance=(1-amount)*saved_var+amount*current_var
        result=(value-mean)*torch.rsqrt(variance+module.eps)
        if module.affine:result=result*module.weight[None,:,None,None]+module.bias[None,:,None,None]
        return result.to(output.dtype)
    handles=[]
    for name,module in model.named_modules():
        if isinstance(module,torch.nn.BatchNorm2d) and (scope=='all' or name.startswith(scope+'.')):
            if subset!='all':
                if not name.startswith('decoder.blocks.'):continue
                layer=int(name.split('.')[2].split('_')[-1])
                if subset=='last' and layer!=last_depth:continue
                if subset=='except_last' and layer==last_depth:continue
                if subset=='high_resolution' and layer<last_depth-1:continue
                if subset=='low_resolution' and layer>=last_depth-1:continue
            if module.running_mean is None or module.running_var is None:raise ValueError('Expected saved BN statistics')
            handles.append(module.register_forward_hook(hook))
    if not handles:raise ValueError(f'No BatchNorm2d layers matched {scope}')
    try:return model(x)
    finally:
        for handle in handles:handle.remove()
