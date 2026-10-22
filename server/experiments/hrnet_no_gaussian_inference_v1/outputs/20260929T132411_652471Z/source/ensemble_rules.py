"""Fixed per-pixel rules for combining same-checkpoint inference variants."""
import numpy as np


def combine(values,arm):
    weights=np.asarray(arm['weights'],np.float32)
    if len(weights)!=len(values) or (weights<0).any() or weights.sum()<=0:raise ValueError('Invalid ensemble weights')
    weights/=weights.sum();rule=arm.get('aggregation',{'kind':'arithmetic'})
    def arithmetic():
        q=None
        for value,weight in zip(values,weights):q=weight*value if q is None else q+weight*value
        return q
    kind=rule['kind']
    if kind=='arithmetic':return arithmetic()
    if kind=='geometric':
        q=None
        for value,weight in zip(values,weights):
            term=weight*np.log(np.maximum(value,1e-8));q=term if q is None else q+term
        q=np.exp(q-q.max(0,keepdims=True));return q/q.sum(0,keepdims=True)
    if kind=='margin_gate':
        baseline=values[0];ordered=np.partition(baseline,-2,axis=0)
        margin=ordered[-1]-ordered[-2]
        return np.where((margin<rule['threshold'])[None],arithmetic(),baseline)
    stack=np.stack(values)
    if kind=='median':
        q=np.median(stack,axis=0);return q/q.sum(0,keepdims=True).clip(1e-8)
    if kind=='entropy':
        if rule['strength']==0:return arithmetic()
        certainty=1+(stack*np.log(stack.clip(1e-8))).sum(1)/np.log(4)
    elif kind=='margin':
        ordered=np.partition(stack,-2,axis=1);certainty=ordered[:,-1]-ordered[:,-2]
    elif kind=='most_confident':
        index=stack.max(1).argmax(0)
        return np.take_along_axis(stack,index[None,None].repeat(4,axis=1),axis=0)[0]
    else:raise ValueError(kind)
    logits=rule['strength']*certainty+np.log(weights.clip(1e-8))[:,None,None]
    dynamic=np.exp(logits-logits.max(0,keepdims=True));dynamic/=dynamic.sum(0,keepdims=True)
    return (dynamic[:,None]*stack).sum(0)
