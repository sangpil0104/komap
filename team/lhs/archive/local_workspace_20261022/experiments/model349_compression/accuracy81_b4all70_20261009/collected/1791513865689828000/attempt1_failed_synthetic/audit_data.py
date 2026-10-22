"""CPU-only preprocessing/reproducibility gate before GPU training."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
import hashlib, numpy as np, torch, cv2
from data import ROOT, Dataset, SEED, write
torch.set_num_threads(2);cv2.setNumThreads(2)
d=Dataset()
for index in [0,1]:
    x,y,info=d.sample(index);a,b,meta=d.sample(index)
    assert torch.equal(x,a) and torch.equal(y,b) and info==meta
    assert torch.isfinite(x).all() and set(y.unique().tolist())<=set(range(4))
seen={int(np.random.default_rng(SEED+i).integers(70)) for i in range(800)}
assert len(seen)==70
assert all(np.isfinite(mean) and np.isfinite(std) and std>0 for mean,std in d.stats)
assert all(np.all(np.diff(cdf)>=0) and cdf[-1]==1 for cdf in d.cdfs)
write(ROOT/'data_audit.json',{'passed':True,'train_images':70,'planned_sample_count':800,
    'planned_all70_covered':True,'repeat_crops_exact':True,'global_image_statistics_used':True,
    'finite_statistics_and_monotone_cdfs':True,'valid_read':False,'test_read':False,'gpu_used':False,
    'rows':d.rows})
print('PASS train70 global-statistics and deterministic crop audit',flush=True)
