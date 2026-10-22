"""Parse the Model361 recipe and run inference-only model checks on requested GPUs."""
import gc
import importlib.util
import json
import sys
from pathlib import Path

import torch
import segmentation_models_pytorch as smp
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'adapters'))
sys.path.insert(0, str(ROOT / 'source/experiments/model336_hrnet_best'))
from komap_seg.model import build_model
from komap_seg.inference import predict_probs_tta

spec = importlib.util.spec_from_file_location('model361_train', ROOT / 'source/experiments/model336_hrnet_best/train.py')
module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
tasks = json.loads((ROOT / 'pipeline_tasks.json').read_text())['tasks']
task = next(t for t in tasks if t['id'] == 'train_model361')
sys.argv = task['argv']
args = module.parse_args()
recipe = json.loads((ROOT / 'source/config/model364_recipe.json').read_text())['training']
mapping = {'architecture': 'architecture', 'encoder_name': 'encoder_name',
           'patch_size': 'patch_size', 'input_scale': 'input_scale',
           'samples_per_image': 'samples_per_image', 'batch_size': 'batch_size',
           'accumulation_steps': 'accumulation_steps', 'epochs': 'epochs',
           'learning_rate': 'lr', 'encoder_learning_rate_multiplier': 'encoder_lr_mult',
           'weight_decay': 'weight_decay', 'boundary_head_weight': 'boundary_head_weight', 'seed': 'seed'}
for key, attr in mapping.items():
    assert getattr(args, attr) == recipe[key], (key, getattr(args, attr), recipe[key])
assert args.ce_class_weights is None and args.val_every == 0
report = []
for gpu in tqdm([4, 5, 6, 7], desc='Inference-only GPU checks', unit='GPU'):
    device = torch.device(f'cuda:{gpu}')
    for family in ['b4', 'hrnet']:
        model = (smp.Unet(encoder_name='efficientnet-b4', encoder_weights='imagenet',
                         in_channels=3, classes=4) if family == 'b4' else build_model()).eval().to(device)
        size, batch = (1536, 2) if family == 'b4' else (512, 4)
        torch.cuda.reset_peak_memory_stats(device)
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.float16):
            result = model(torch.zeros(batch, 3, size, size, device=device))
        assert result.shape == (batch, 4, size, size) and torch.isfinite(result).all()
        if family == 'hrnet' and gpu == 4:
            with torch.inference_mode(), torch.autocast('cuda', dtype=torch.float16):
                p = predict_probs_tta(model, torch.zeros(3, 32, 32), patch_size=64,
                                      overlap=16, batch_size=2, input_scale=2.0,
                                      device=device, tta_mode='d4')
            assert p.shape == (4, 32, 32) and torch.isfinite(p).all()
            torch.testing.assert_close(p.sum(0), torch.ones(32, 32), rtol=1e-3, atol=1e-3)
        row = {'gpu': gpu, 'family': family, 'input': [batch, 3, size, size],
               'output': list(result.shape), 'forward_peak_gib': torch.cuda.max_memory_allocated(device) / 2**30}
        print(json.dumps(row), flush=True); report.append(row)
        del model, result; gc.collect(); torch.cuda.empty_cache()
(ROOT / 'reports/model_forward.json').write_text(json.dumps(
    {'training_started': False, 'model361_recipe_parser': 'passed', 'hrnet_d4_inference': 'passed',
     'gpu_forwards': report}, indent=2))
