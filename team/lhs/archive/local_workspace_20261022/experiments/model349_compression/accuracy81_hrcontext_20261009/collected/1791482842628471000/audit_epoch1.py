"""Read-only checkpoint invariant audit; no images, GT, or CUDA."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
from pathlib import Path
import hashlib, json
import torch
ROOT = Path(__file__).resolve().parent
torch.set_num_threads(2)
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
progress = json.loads((ROOT / 'training/context/progress.json').read_text())
assert progress and progress[0]['epoch'] == 1 and progress[0]['updates'] == 105
parent = ROOT.parent / 'Model349_exh14/weights/model336_hrnet_best/last.pt'
candidate = ROOT / 'training/context/epoch1.pt'
before = sha(candidate)
recipe = json.loads((ROOT / 'training/context/recipe.json').read_text())
assert sha(parent) == recipe['parent_sha256']
assert not recipe['valid_read_during_training'] and not recipe['test_read_during_training']
original = torch.load(parent, map_location='cpu', weights_only=False)['state_dict']
changed = torch.load(candidate, map_location='cpu', weights_only=False)['state_dict']
assert original.keys() == changed.keys()
bn = [k for k in original if k.endswith(('running_mean', 'running_var', 'num_batches_tracked'))]
boundary = [k for k in original if k.startswith('boundary_head.')]
assert bn and boundary
assert all(torch.equal(original[k], changed[k]) for k in bn + boundary)
assert all(torch.isfinite(v).all() for v in changed.values())
updates = [k for k in original if not torch.equal(original[k], changed[k])]
assert updates and sha(candidate) == before and sha(parent) == recipe['parent_sha256']
result = {'epoch': 1, 'optimizer_updates': 105, 'candidate_sha256': before,
          'candidate_bytes': candidate.stat().st_size, 'bn_buffers_checked': len(bn),
          'boundary_tensors_checked': len(boundary), 'changed_tensors': len(updates),
          'bn_buffers_unchanged': True, 'boundary_head_unchanged': True,
          'all_tensors_finite': True, 'parent_checkpoint_unchanged': True,
          'candidate_read_only': True, 'valid_read': False, 'test_read': False,
          'intermediate_checkpoint_scored': False}
(ROOT / 'epoch1_invariant_audit.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result), flush=True)
