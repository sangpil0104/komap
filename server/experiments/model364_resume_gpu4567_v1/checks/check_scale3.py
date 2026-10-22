"""Reproduce the actual Model361 first-image inference, without training."""
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
recipe = json.loads((ROOT / 'pipeline_tasks.json').read_text())
parent = Path(recipe['parent_run'])
uuid = subprocess.check_output(['nvidia-smi', '-i', '4', '--query-gpu=uuid',
                                '--format=csv,noheader'], text=True).strip()
os.environ['CUDA_VISIBLE_DEVICES'] = uuid
sys.path.insert(0, str(ROOT / 'source/experiments/model336_hrnet_best'))
import torch
from tqdm import tqdm
from komap_seg.dataset import KomapFullDataset
from komap_seg.checkpoint import load_model
from komap_seg.inference import predict_probs_tta

torch.set_num_threads(2)
dataset = KomapFullDataset(parent / 'data/valid/images', parent / 'data/valid/masks')
sample = dataset[0]
device = torch.device('cuda:0')
model = load_model(parent / 'outputs/model361_hrnet_scale3_finetune/last.pt', device).eval()
task = next(t for t in recipe['tasks'] if t['id'] == 'evaluate_model361')
argv = task['argv']
overlap = int(argv[argv.index('--overlap') + 1])
assert overlap == 114
with torch.inference_mode():
    for _ in tqdm(range(1), desc='Actual Model361 scale3 flip4 inference', unit='image'):
        probs = predict_probs_tta(model, sample['image'], patch_size=512, overlap=overlap,
                                  batch_size=2, input_scale=3.0, device=device, tta_mode='flip4')
assert probs.shape == (4, *sample['mask'].shape)
assert torch.isfinite(probs).all()
torch.testing.assert_close(probs.sum(0), torch.ones_like(probs[0]), atol=1e-4, rtol=1e-4)
report = {'training_started': False, 'checkpoint': 'Model361 existing last.pt',
          'image': sample['stem'], 'shape': list(probs.shape), 'input_scale': 3.0,
          'native_tile': 171, 'overlap': overlap, 'stride': 57,
          'actual_first_validation_image_flip4': 'passed', 'physical_gpu': 4}
(ROOT / 'reports/scale3_validation.json').write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
