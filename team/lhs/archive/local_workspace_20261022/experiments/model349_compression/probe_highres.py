"""For every image, run TENT + 768 high-res prediction with EACH high-res candidate checkpoint
and save the pre-auxiliary probability (style-weighted raw/style mix), so any routing map can be
scored offline. Run from a Model349 bundle root."""
import json, sys, time
from pathlib import Path
import numpy as np
import torch
from PIL import Image
ROOT = Path.cwd(); sys.path.insert(0, str(ROOT / 'src'))
import yaml
from hard_source_search import route_params
from highres1024_search import choose_source
from style_tta_search import predict_array, transform_image
from tent_style_search import adapt_model
from types import SimpleNamespace

images, out = Path(sys.argv[1]), Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
config = yaml.safe_load((ROOT / 'configs/model349.yaml').read_text())
cal = json.loads((ROOT / config['calibration']).read_text())
tm, ts = cal['train_brightness_mean'], cal['train_brightness_std']
tent = config['inference']['tent']
W = ROOT / config['weights_dir']
CAND = {'restart51': 'restart51_aspp_768_finetune', 'model109': 'model109_aspp_native', 'seed456': 'restart59_aspp768_seed456',
        'pairwise': 'restart61_pairwise_margin', 'ema': 'model123_ema'}
keyof = SimpleNamespace(**{f'source_{k}': k for k in CAND})
device = torch.device('cuda:0')
for p in sorted(images.glob('*.png')):
    name = p.stem.removesuffix('_image')
    raw = np.asarray(Image.open(p).convert('RGB'), dtype=np.uint8)
    gray = raw[..., 0].astype(np.float32); mean, std = float(gray.mean()), float(gray.std())
    params = dict(route_params(mean, std), pair_weight=0.0)
    style = transform_image(raw, 'standardize', tm, ts)
    arrays, t = {}, {}
    for k, folder in CAND.items():
        t0 = time.monotonic()
        model = adapt_model(str(W / folder / 'best.pt'), raw, device, True, tent['steps'], tent['lr'])
        hi = [predict_array(model, im, 768, 576, device, True, 1, 'd4', 'gaussian') for im in (raw, style)]
        pre = (1 - params['style_weight']) * hi[0] + params['style_weight'] * hi[1]
        pre /= np.maximum(pre.sum(0, keepdims=True), 1e-7)
        arrays[k] = pre.astype(np.float16); t[k] = time.monotonic() - t0
        del model; torch.cuda.empty_cache()
    np.savez(out / f'{name}.npz', **arrays,
             z=((gray - mean) / max(std, 1e-5)).astype(np.float32))
    (out / f'{name}.json').write_text(json.dumps({'params': params, 'routed': choose_source(mean, keyof), 'mean': mean,
                                                  'timing': t, 'prior_z_stats': cal['priors']['z'], 'prior_q_stats': cal['priors']['q']}))
    print('DONE', name, choose_source(mean, keyof), {k: round(v, 1) for k, v in t.items()}, flush=True)
