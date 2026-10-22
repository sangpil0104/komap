"""Render test masks for selected component subsets with the exhaustive-search fusion (no GT needed).
Run from the lite14 bundle root with PROBE, HIGH, OUT set."""
import json, os, sys
from pathlib import Path
import numpy as np
from PIL import Image
os.environ.setdefault('GT', '/nonexistent')
sys.path.insert(0, str(Path.cwd()))
import exhaustive as ex

COMBOS = {'full_offline': (15, 15, 15, 1, 1, 1, 7, 1, 15), 'lite14': (0, 15, 15, 1, 0, 1, 0, 0, 7),
          'base12': (0, 5, 15, 1, 0, 1, 0, 0, 7), 'top802': (12, 3, 13, 1, 1, 1, 0, 1, 7)}
OUT = Path(os.environ['OUT'])
for name in ex.NAMES:
    z = {}
    for c in ('322', '326', '329'):
        d = np.load(ex.PROBE / c / f'{name}.npz'); z[c] = [d[f'f{i}_v1'].astype(np.float32) for i in range(1, 5)]
    for c in ('336', '347', '269'):
        z[c] = np.load(ex.PROBE / c / f'{name}.npz')['v1'].astype(np.float32)
    b = np.load(ex.PROBE / 'base' / f'{name}.npz')
    z['branches'] = [b[f'branch{i}'].astype(np.float32) for i in range(3)]; z['pair'] = b['pair'].astype(np.float32); z['q'] = b['q']
    h = np.load(ex.HIGH / f'{name}.npz')
    z['high'] = {k: h[k].astype(np.float32) for k in ['restart51'] + ex.BANDS}; z['z'] = h['z']
    z['meta'] = json.loads((ex.HIGH / f'{name}.json').read_text())
    z['weights'] = ex.route.bilateral_weights(h['z']); z['zero'] = np.zeros_like(z['336'])
    for tag, cb in COMBOS.items():
        pred = ex.predict(z, cb).astype(np.uint8)
        (OUT / tag).mkdir(parents=True, exist_ok=True)
        Image.fromarray(pred).save(OUT / tag / f'{name}.png')
    print('done', name, z['meta']['routed'], flush=True)
