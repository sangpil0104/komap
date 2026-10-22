"""Synthetic training feasibility gate; no valid/test images or GT."""
import os
os.environ.update(OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', NO_ALBUMENTATIONS_UPDATE='1')
from pathlib import Path
import sys, hashlib, json, time
import torch
ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent / 'Model349_exh14'
sys.path.insert(0, str(SOURCE / 'src'))
from komap_seg.model import build_model
from loss import objective
torch.set_num_threads(2)
torch.manual_seed(81336)
checkpoint = SOURCE / 'weights/model336_hrnet_best/last.pt'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
before = sha(checkpoint)
saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
model = build_model(**saved['model_config'], encoder_weights=None)
model.load_state_dict(saved['state_dict'])
model.cuda().train()
for n, p in model.named_parameters():
    if n.startswith('boundary_head.'):
        p.requires_grad_(False)
for m in model.modules():
    if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
        m.eval()
buffers = {n: b.clone() for n, b in model.named_buffers()}
opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=2e-5, weight_decay=1e-4)
scaler = torch.amp.GradScaler('cuda', init_scale=1024)
image = torch.randn(2, 3, 1024, 1024, device='cuda')
target = torch.randint(0, 4, (2, 1024, 1024), device='cuda')
start = time.monotonic()
with torch.autocast('cuda', dtype=torch.float16):
    logits = model(image)
assert logits.shape == (2, 4, 1024, 1024)
loss = objective(logits, target, 'control')
assert torch.isfinite(loss)
scaler.scale(loss / 4).backward()
scaler.unscale_(opt)
assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
scaler.step(opt)
scaler.update()
torch.cuda.synchronize()
assert all(torch.equal(b, buffers[n]) for n, b in model.named_buffers())
assert sha(checkpoint) == before
result = {'passed': True, 'batch': 2, 'patch': 1024, 'native_crop': 512,
          'loss': loss.item(), 'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated(),
          'elapsed_seconds': time.monotonic() - start, 'bn_buffers_unchanged': True,
          'original_checkpoint_unchanged': True, 'valid_read': False, 'test_read': False}
(ROOT / 'smoke_memory.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result), flush=True)
