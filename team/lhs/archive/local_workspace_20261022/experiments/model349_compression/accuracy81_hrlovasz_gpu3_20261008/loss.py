"""Independent per-patch Lovasz extension for four-class softmax IoU."""
import torch
import torch.nn.functional as F

def lovasz_softmax(logits, targets):
 probs=logits.float().softmax(1);losses=[]
 for p,y in zip(probs,targets):
  y=y.reshape(-1);p=p.reshape(p.shape[0],-1)
  classes=[]
  for c in range(p.shape[0]):
   fg=(y==c).float();errors=(fg-p[c]).abs();errors,order=errors.sort(descending=True);fg=fg[order]
   n=fg.sum();intersection=n-fg.cumsum(0);union=n+(1-fg).cumsum(0)
   jaccard=1-intersection/union.clamp_min(1);grad=torch.cat([jaccard[:1],jaccard[1:]-jaccard[:-1]])
   classes.append(torch.dot(errors,grad))
  losses.append(torch.stack(classes).mean())
 return torch.stack(losses).mean()

def dice_loss(logits, targets):
 p=logits.float().softmax(1);gt=F.one_hot(targets,logits.shape[1]).permute(0,3,1,2).float()
 inter=(p*gt).sum((0,2,3));den=(p+gt).sum((0,2,3));return 1-((2*inter+1e-6)/(den+1e-6)).mean()

def objective(logits,targets,arm):
 ce=F.cross_entropy(logits.float(),targets);dice=dice_loss(logits,targets)
 if arm=='control':return .5*ce+.5*dice
 if arm=='lovasz':return .5*ce+.25*dice+.25*lovasz_softmax(logits,targets)
 raise ValueError(arm)
