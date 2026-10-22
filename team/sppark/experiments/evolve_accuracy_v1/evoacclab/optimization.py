import math
import torch


def optimizer_for(model, cfg, stage):
    options = cfg['train']
    groups = {}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        rate = stage['new_lr']
        if name.startswith('additions.'):
            rate = stage.get('adapter_lr', rate)
        elif name.startswith(('detail.', 'encoder.')):
            rate *= stage.get('encoder_multiplier', 1.)
        decay = 0. if p.ndim <= 1 or name.endswith('bias') else options['new_weight_decay']
        key = (rate, decay)
        if key not in groups:
            groups[key] = dict(params=[], names=[], lr=rate, weight_decay=decay)
        groups[key]['params'].append(p)
        groups[key]['names'].append(name)
    kind = options['optimizer'].lower()
    if kind == 'sgd':
        optimizer = torch.optim.SGD(list(groups.values()), momentum=.9, nesterov=True)
    else:
        cls = torch.optim.AdamW if kind == 'adamw' else torch.optim.Adam
        optimizer = cls(list(groups.values()), betas=tuple(options.get('betas', [.9, .999])), amsgrad=options.get('amsgrad', False))
    warmup, total = stage['warmup_updates'], stage['max_updates']

    def factor(step):
        if warmup and step < warmup:
            return (step + 1) / warmup
        progress = min(1., max(0., (step - warmup) / max(1, total - warmup)))
        schedule = options.get('schedule', 'cosine')
        if schedule == 'constant':
            return 1.
        if schedule == 'poly':
            return .01 + .99 * (1 - progress) ** .9
        return .01 + .99 * (1 + math.cos(math.pi * progress)) / 2

    return optimizer, torch.optim.lr_scheduler.LambdaLR(optimizer, factor)
