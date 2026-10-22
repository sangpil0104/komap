"""Train F (11,876-parameter residual refiner) on features from build_features.py. Never reads valid/test.

    python train_refiner.py --parent-package inference --features FEATURES --out NEW_DIR [--device cuda]

Same recipe as the published F (inference/f/training_recipe.json): seed 81349,
8 epochs x 140 minibatches (70 images x 16 random 256-px crops, batch 8), AdamW lr 1e-3 wd 1e-4, cosine,
0.5 CE + 0.5 Dice + 0.05 |delta|^2, gray z-score input, last epoch kept (no selection). Writes NEW_DIR/epoch8.pt.
"""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse, hashlib, json, math, random, sys, time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
SEED = 81349


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''): h.update(b)
    return h.hexdigest()


class Crops(Dataset):
    def __init__(self, root, prepare_gray, allow_subset=False):
        self.epoch = 0; self.rows = []
        man = json.loads((root / 'manifest.json').read_text()); assert man['split'] == 'train' and (allow_subset or len(man['images']) == 70)
        for row in man['images']:
            p = root / (row['name'] + '.npz'); assert sha(p) == row['features_sha256'], p
            with np.load(p) as z: logits = z['logits']; gray = z['gray']; target = z['target']
            assert logits.dtype == np.float32 and logits.shape == (4, *target.shape) and gray.shape == target.shape and np.isfinite(logits).all()
            gray = prepare_gray(gray)
            self.rows.append((logits, gray, target, {c: np.flatnonzero(target == c) for c in [0, 3]}))

    def __len__(self): return len(self.rows) * 16

    def __getitem__(self, index):
        logits, gray, target, coords = self.rows[index // 16]; rng = np.random.default_rng(SEED + self.epoch * 100000 + index); h, w = target.shape; p = 256
        active = [c for c in coords if len(coords[c])]
        if active and rng.random() < .5:
            cls = active[rng.integers(len(active))]; center = coords[cls][rng.integers(len(coords[cls]))]; cy, cx = divmod(int(center), w)
            y = int(np.clip(cy - rng.integers(p), 0, h - p)); x = int(np.clip(cx - rng.integers(p), 0, w - p))
        else:
            y = int(rng.integers(h - p + 1)); x = int(rng.integers(w - p + 1))
        a = logits[:, y:y + p, x:x + p]; b = gray[None, y:y + p, x:x + p]; c = target[y:y + p, x:x + p]
        if rng.random() < .5: a = a[..., ::-1]; b = b[..., ::-1]; c = c[..., ::-1]
        if rng.random() < .5: a = a[..., ::-1, :]; b = b[..., ::-1, :]; c = c[::-1, :]
        k = int(rng.integers(4)); a = np.rot90(a, k, axes=(-2, -1)); b = np.rot90(b, k, axes=(-2, -1)); c = np.rot90(c, k)
        return torch.from_numpy(np.ascontiguousarray(a)), torch.from_numpy(np.ascontiguousarray(b)), torch.from_numpy(np.ascontiguousarray(c)).long()


def objective(logits, target, delta):
    p = logits.softmax(1); gt = torch.nn.functional.one_hot(target, 4).permute(0, 3, 1, 2).float()
    inter = (p * gt).sum((0, 2, 3)); den = (p + gt).sum((0, 2, 3)); dice = 1 - ((2 * inter + 1e-6) / (den + 1e-6)).mean()
    ce = -torch.nn.functional.log_softmax(logits, dim=1).gather(1, target[:, None]).mean()
    return .5 * ce + .5 * dice + .05 * delta.square().mean()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--parent-package', type=Path, required=True)
    ap.add_argument('--features', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--allow-subset', action='store_true', help='smoke test only: train on fewer than 70 images')
    a = ap.parse_args()
    sys.path.insert(0, str(a.parent_package / 'f'))
    from residual_model import Residual
    from gray_input import prepare_gray, MODE
    assert MODE == 'zscore'
    torch.set_num_threads(2); random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True; torch.use_deterministic_algorithms(True)
    a.out.mkdir(parents=True, exist_ok=False); ds = Crops(a.features, prepare_gray, a.allow_subset)
    gen = torch.Generator().manual_seed(SEED); loader = DataLoader(ds, batch_size=8, shuffle=True, num_workers=0, generator=gen, pin_memory=True)
    assert len(loader) == math.ceil(len(ds) / 8) and (a.allow_subset or len(loader) == 140)
    dev = torch.device(a.device); model = Residual().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4); sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=8)
    start = time.monotonic(); updates = 0; rows = []
    for epoch in range(1, 9):
        ds.epoch = epoch; model.train(); total = 0.
        for logits, gray, target in loader:
            logits = logits.to(dev); gray = gray.to(dev); target = target.to(dev); opt.zero_grad(set_to_none=True)
            pred, delta = model(logits, gray); loss = objective(pred, target, delta); assert torch.isfinite(loss); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.); opt.step(); updates += 1; total += loss.item()
        sched.step()
        torch.save({'state_dict': {k: v.detach().cpu() for k, v in model.state_dict().items()}, 'seed': SEED, 'epoch': epoch, 'parameters': 11876}, a.out / f'epoch{epoch}.pt')
        rows.append({'epoch': epoch, 'loss': total / len(loader), 'updates': updates}); print(json.dumps(rows[-1]), flush=True)
    assert updates == 8 * len(loader) and (a.allow_subset or updates == 1120)
    (a.out / 'completion.json').write_text(json.dumps({'epochs': 8, 'updates': updates, 'progress': rows, 'final_checkpoint': 'epoch8.pt',
        'final_sha256': sha(a.out / 'epoch8.pt'), 'features_manifest_sha256': sha(a.features / 'manifest.json'),
        'elapsed_seconds': time.monotonic() - start, 'device': str(dev), 'valid_or_test_read': False}, indent=2) + '\n')


if __name__ == '__main__':
    main()
