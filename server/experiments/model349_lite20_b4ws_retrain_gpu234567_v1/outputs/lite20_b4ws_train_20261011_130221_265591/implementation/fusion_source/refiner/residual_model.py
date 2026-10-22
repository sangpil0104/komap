"""Zero-initialized, bounded native pixel correction of parent logits."""
import torch
import torch.nn as nn
class Residual(nn.Module):
 def __init__(self):
  super().__init__();self.net=nn.Sequential(nn.Conv2d(5,32,3,padding=1),nn.SiLU(),nn.Conv2d(32,32,3,padding=1),nn.SiLU(),nn.Conv2d(32,4,3,padding=1))
  nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)
 def forward(self,logits,gray):
  centered=(logits-logits.mean(1,keepdim=True)).clamp(-8,8)/4
  features=torch.cat([centered,(gray-.5)*2],1);delta=self.net(features).tanh()
  return logits+delta,delta
def smoke():
 torch.manual_seed(81349);torch.set_num_threads(1);m=Residual();assert sum(p.numel() for p in m.parameters())==11876
 logits=torch.randn(2,4,17,19);gray=torch.rand(2,1,17,19);y=torch.randint(0,4,(2,17,19))
 output,delta=m(logits,gray);assert torch.equal(output,logits) and torch.count_nonzero(delta)==0
 loss=nn.functional.cross_entropy(output,y)+.05*delta.square().mean();loss.backward()
 assert torch.isfinite(loss) and torch.isfinite(m.net[-1].weight.grad).all() and m.net[-1].weight.grad.abs().sum()>0
 assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
 with torch.no_grad():m.net[-1].bias.fill_(20);_,delta=m(logits,gray);assert delta.max()<=1 and delta.min()>=-1
if __name__=='__main__':smoke();print('RESIDUAL_ZERO_IDENTITY_BOUNDED_FINITE_GRAD_PARAMETER_COUNT_PASSED')
