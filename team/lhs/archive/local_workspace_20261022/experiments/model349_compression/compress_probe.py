"""Dump decomposed Model349 probabilities so compressed variants can be scored offline.

Run from the release bundle root (it imports the bundle's own src/):
    python compress_probe.py --component 322 --images <valid/images> --out <dir>

Per image it writes <out>/<component>/<name>.npz with float16 arrays in the main
class order [Al3Ni, Primary Si, Al, Eutectic Si] and records wall time per unit.

  B4 (322/326/329): f{fold}_v{n}  mean of the first n D4 views, n in {1,2,4,8},
                    views ordered [0,4,2,6,1,5,3,7] (id, hflip, rot180, ...)
  HRNet (336/347):  v1 (identity), v2 (+hflip), v4 (original flip4)
  269:              v1 (no TTA), v8 (original D4)
  base:             every ingredient of the base mixture, with and without TENT
"""
import argparse, json, os, sys, time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT / 'src'))
import yaml

TEAM_TO_MAIN = [1, 2, 0, 3]
B4_DIRS = {'322': 'model322_team_b4_group_oof', '326': 'model326_hybrid_fullstats_group_oof',
           '329': 'model329_fullstats_width4000_uniform_group_oof'}
HR_DIRS = {'336': 'model336_hrnet_best', '347': 'model347_hrnet_inverse_ce_finetune'}
VIEW_ORDER = [0, 4, 2, 6, 1, 5, 3, 7]


def f16(a):
    return np.asarray(a, dtype=np.float32).astype(np.float16)


def b4_member_views(mem, gray, device, mod):
    """Per-view probabilities (C,H,W) at original resolution for one fold member."""
    H, W = gray.shape
    scale = mod.member_scale(mem, W)
    g = mod.resize_gray(gray, scale)
    h, w = g.shape
    x = torch.from_numpy(mod.make_input(g, mem['norm'], mem['input'], mem['stats']))[None].to(device)
    x = F.pad(x, (0, (-w) % 32, 0, (-h) % 32), mode='reflect')
    out = []
    for k in VIEW_ORDER:
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16):
            logits = mem['model'](mod._d4(x, k))
        p = mod._d4_inv(logits.float().softmax(1), k)[:, :, :h, :w]
        if scale != 1:
            p = F.interpolate(p, size=(H, W), mode='area' if scale > 1 else 'bilinear')
        out.append(p[0])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--component', required=True)
    ap.add_argument('--images', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    config = yaml.safe_load((ROOT / 'configs/model349.yaml').read_text())
    weights = ROOT / config['weights_dir']
    device = torch.device('cuda:0')
    out = args.out / args.component
    out.mkdir(parents=True, exist_ok=True)
    paths = sorted(p for p in args.images.iterdir() if p.suffix.lower() == '.png')
    timing = {}
    c = args.component

    if c.endswith('soup') and c[:3] in B4_DIRS:
        # uniform weight average of the 4 fold checkpoints -> one network, one forward pass per view
        base_c = c[:3]
        mod = __import__('b4_gray' if base_c == '322' else 'b4_zcdf')
        members = [mod.load_member(str(weights / B4_DIRS[base_c] / f'fold{i}'), str(device)) for i in range(1, 5)]
        soup = members[0]
        sds = [m['model'].state_dict() for m in members]
        avg = {}
        for k, v in sds[0].items():
            avg[k] = (sum(sd[k].float() for sd in sds) / len(sds)).to(v.dtype) if v.is_floating_point() else v
        soup['model'].load_state_dict(avg)
        soup['model'].eval()
        for p in paths:
            name = p.stem.removesuffix('_image')
            gray = mod.load_gray(str(p))
            t0 = time.monotonic()
            views = b4_member_views(soup, gray, device, mod)
            timing[name] = time.monotonic() - t0
            np.savez(out / f'{name}.npz', **{f'v{n}': f16(torch.stack(views[:n]).mean(0).cpu().numpy()[TEAM_TO_MAIN]) for n in (1, 2, 4, 8)})
            print('DONE', c, name, round(timing[name], 1), flush=True)

    elif c in B4_DIRS:
        mod = __import__('b4_gray' if c == '322' else 'b4_zcdf')
        members = [mod.load_member(str(weights / B4_DIRS[c] / f'fold{i}'), str(device)) for i in range(1, 5)]
        for p in paths:
            name = p.stem.removesuffix('_image')
            gray = mod.load_gray(str(p))
            arrays, t0 = {}, time.monotonic()
            for fi, mem in enumerate(members, 1):
                views = b4_member_views(mem, gray, device, mod)
                for n in (1, 2, 4, 8):
                    arrays[f'f{fi}_v{n}'] = f16(torch.stack(views[:n]).mean(0).cpu().numpy()[TEAM_TO_MAIN])
            timing[name] = time.monotonic() - t0
            np.savez(out / f'{name}.npz', **arrays)
            print('DONE', c, name, round(timing[name], 1), flush=True)

    elif c in HR_DIRS:
        from komap_seg.checkpoint import load_model
        from komap_seg.dataset import build_eval_transform, load_image_rgb
        from komap_seg.inference import predict_probs, _TTA_OPS
        model = load_model(weights / HR_DIRS[c] / 'last.pt', device)
        tf = build_eval_transform()
        h = config['inference']['hrnet']
        for p in paths:
            name = p.stem.removesuffix('_image')
            im = tf(image=load_image_rgb(p))['image'].float()
            t0 = time.monotonic()
            views = []
            for fwd, inv in _TTA_OPS:
                views.append(inv(predict_probs(model, fwd(im), patch_size=h['patch'], overlap=h['overlap'],
                                               batch_size=h['batch_size'], input_scale=h['input_scale'], device=device)))
            timing[name] = time.monotonic() - t0
            np.savez(out / f'{name}.npz', **{f'v{n}': f16(torch.stack(views[:n]).mean(0).numpy()[TEAM_TO_MAIN]) for n in (1, 2, 4)})
            print('DONE', c, name, round(timing[name], 1), flush=True)

    elif c == '269':
        from model253_highres_scale_finetune import Stride1NativeStemASPP, predict_scaled
        model = Stride1NativeStemASPP().to(device)
        model.load_state_dict(torch.load(weights / 'model269_fullstats_1x_extended/best.pt', map_location='cpu', weights_only=False)['model'])
        model.eval()
        n = config['inference']['native']
        for p in paths:
            name = p.stem.removesuffix('_image')
            t0 = time.monotonic()
            v1 = predict_scaled(model, p, n['patch'], n['stride'], n['scale'], device, True, 'none')
            t1 = time.monotonic()
            v8 = predict_scaled(model, p, n['patch'], n['stride'], n['scale'], device, True, n['tta'])
            timing[name] = {'v1': t1 - t0, 'v8': time.monotonic() - t1}
            np.savez(out / f'{name}.npz', v1=f16(v1), v8=f16(v8))
            print('DONE', c, name, flush=True)

    elif c == 'base':
        import calibrate_al
        import model183_metric_calibration as cache
        from highres1024_search import choose_source
        from hard_source_search import route_params
        from prior_representation_search import percentile_image
        from style_tta_search import predict_array, transform_image
        from tent_style_search import adapt_model
        cal = json.loads((ROOT / config['calibration']).read_text())
        tm, ts = cal['train_brightness_mean'], cal['train_brightness_std']
        branch_models = [calibrate_al.load_model(str(weights / d / 'best.pt'), device) for d in ['model97_native_patch', 'model98_hard_negative']]
        branch_models.append(cache.load_aspp(weights / 'model109_aspp_native/best.pt', device))
        pair = cache.load_aspp(weights / 'restart61_pairwise_margin/best.pt', device)
        rep = cache.load_rep(weights / 'model181_raw_zcdf_aspp/best.pt', device)
        cand = {'restart51': 'restart51_aspp_768_finetune', 'model109': 'model109_aspp_native', 'seed456': 'restart59_aspp768_seed456',
                'pairwise': 'restart61_pairwise_margin', 'ema': 'model123_ema'}
        sargs = SimpleNamespace(**{f'source_{k}': str(weights / v / 'best.pt') for k, v in cand.items()})
        tent = config['inference']['tent']
        for p in paths:
            name = p.stem.removesuffix('_image')
            raw = np.asarray(Image.open(p).convert('RGB'), dtype=np.uint8)
            gray = raw[..., 0].astype(np.float32)
            mean, std = float(gray.mean()), float(gray.std())
            arrays, t = {}, {}
            t0 = time.monotonic()
            for i, (m, patch, stride) in enumerate(zip(branch_models, (640, 512, 512), (512, 384, 384))):
                arrays[f'branch{i}'] = f16(calibrate_al.predict_probabilities(m, p, patch, stride, device, True, 2, True))
            t['branches'] = time.monotonic() - t0; t0 = time.monotonic()
            arrays['pair'] = f16(cache.predict_rgb(pair, p, 768, 576, device, True, 1, 'd4', 'gaussian'))
            t['pair'] = time.monotonic() - t0; t0 = time.monotonic()
            arrays['extra'] = f16(cache.predict_rep(rep, p, 512, 384, device, True, 2))
            t['extra'] = time.monotonic() - t0
            ckpt = choose_source(mean, sargs)
            style = transform_image(raw, 'standardize', tm, ts)
            for tag, steps in (('tent', tent['steps']), ('notent', 0)):
                t0 = time.monotonic()
                model = adapt_model(ckpt, raw, device, True, steps, tent['lr'])
                t[f'adapt_{tag}'] = time.monotonic() - t0
                for size, stride in ((768, 576), (1024, 768)):
                    for kind, im in (('raw', raw), ('style', style)):
                        t0 = time.monotonic()
                        arrays[f'high{size}_{kind}_{tag}'] = f16(predict_array(model, im, size, stride, device, True, 1, 'd4', 'gaussian'))
                        t[f'high{size}_{kind}_{tag}'] = time.monotonic() - t0
                del model; torch.cuda.empty_cache()
            meta = {'z': ((gray - mean) / max(std, 1e-5)).astype(np.float32), 'q': percentile_image(gray).astype(np.float32)}
            np.savez(out / f'{name}.npz', **arrays, **meta)
            (out / f'{name}.json').write_text(json.dumps({'params': route_params(mean, std), 'source': Path(ckpt).parent.name,
                                                          'prior_z_stats': cal['priors']['z'], 'prior_q_stats': cal['priors']['q'],
                                                          'timing': t}))
            timing[name] = t
            print('DONE base', name, {k: round(v, 1) for k, v in t.items()}, flush=True)
    else:
        raise SystemExit(f'unknown component {c}')
    (out / '_timing.json').write_text(json.dumps(timing, indent=1))


if __name__ == '__main__':
    main()
