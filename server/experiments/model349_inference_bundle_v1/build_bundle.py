"""Build a portable inference-only derivative; never edits the recorded run."""
from __future__ import annotations

import ast
import hashlib
import importlib.metadata as md
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
from PIL import Image
from tqdm import tqdm
import yaml

ROOT = Path('/Data3/sppark/komap')
HERE = Path(__file__).resolve().parent
RUN = ROOT / 'experiments/model349_train_route_v2/outputs/runs/model349_train_20260930T143930_234475Z'
SOURCE = RUN / 'source_snapshot/experiments'
EXPORT = ROOT / 'experiments/model349_mask_export_v1/outputs/20261007_1'
BUNDLE = ROOT / 'results/model349_inference_bundle_20261007_1'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + '\n')


class Slicer:
    """Select required definitions and their module globals, keeping function bodies."""
    def __init__(self):
        self.paths = {}
        for base in [SOURCE / 'historical_scripts', SOURCE]:
            self.paths.update({p.stem: p for p in base.glob('*.py')})
        self.paths['b4_gray'] = SOURCE / 'model322_team_b4/KoMaP/common.py'
        self.paths['b4_zcdf'] = SOURCE / 'model326_hybrid_fullstats/KoMaP/common.py'
        for p in (SOURCE / 'model336_hrnet_best/komap_seg').glob('*.py'):
            self.paths['komap_seg.' + p.stem] = p
        self.mods = {}
        self.selected = {}
        self.records = []

    def get(self, module):
        if module not in self.mods:
            path = self.paths[module]
            source = path.read_text()
            tree = ast.parse(source)
            symbols = {}
            for node in tree.body:
                if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                    symbols[node.name] = node
                elif isinstance(node, (ast.Import, ast.ImportFrom)):
                    for alias in node.names:
                        name = alias.asname or (alias.name.split('.')[0] if isinstance(node, ast.Import) else alias.name)
                        new = ast.Import(names=[alias]) if isinstance(node, ast.Import) else ast.ImportFrom(module=node.module, names=[alias], level=node.level)
                        ast.copy_location(new, node)
                        symbols[name] = new
                elif isinstance(node, (ast.Assign, ast.AnnAssign)):
                    targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                    for target in targets:
                        for name in ast.walk(target):
                            if isinstance(name, ast.Name):
                                symbols[name.id] = node
            # The historical factory also exposed unrelated architectures. Keep only
            # the ASPP branch actually called by Model349; the network is unchanged.
            if module == 'encoder_structure_search':
                symbols['build_model'] = ast.parse('''
def build_model(candidate: str, pretrained: bool, checkpoint: str | None):
    if candidate != "aspp":
        raise ValueError("This Model349 bundle includes only its ASPP architecture")
    model = ASPPUNetPP(pretrained=False)
    if checkpoint is not None:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(payload["model"], strict=True)
    return model
''').body[0]
            self.mods[module] = (source, symbols)
        return self.mods[module]

    def add(self, module, symbol):
        source, symbols = self.get(module)
        if symbol in self.selected.setdefault(module, set()):
            return
        if symbol not in symbols:
            raise KeyError((module, symbol))
        self.selected[module].add(symbol)
        node = symbols[symbol]
        if isinstance(node, ast.ImportFrom):
            if node.module in self.paths:
                self.add(node.module, node.names[0].name)
        elif isinstance(node, ast.Import):
            if node.names[0].name in self.paths:
                raise RuntimeError(f'Explicit module attribute roots needed: {module}:{symbol}')
        else:
            for dep in ast.walk(node):
                if isinstance(dep, ast.Name) and isinstance(dep.ctx, ast.Load) and dep.id in symbols:
                    self.add(module, dep.id)
                if isinstance(dep, ast.ImportFrom) and dep.module in self.paths:
                    for alias in dep.names:
                        self.add(dep.module, alias.name)

    def emit(self):
        out = BUNDLE / 'src'
        out.mkdir()
        for module, selected in sorted(self.selected.items()):
            source, symbols = self.get(module)
            nodes = []
            seen = set()
            for name in sorted(selected, key=lambda k: (symbols[k].lineno, k)):
                n = symbols[name]
                if id(n) not in seen:
                    nodes.append((name, n))
                    seen.add(id(n))
            chunks = ['# Model349 inference dependency; extracted from the recorded source snapshot.',
                      '# Unused training/search entrypoints are intentionally omitted.',
                      'from __future__ import annotations\n']
            # Imports first, then original-order definitions. Modified factory last.
            nodes.sort(key=lambda row: (0 if isinstance(row[1], (ast.Import, ast.ImportFrom)) else
                                       (2 if module == 'encoder_structure_search' and row[0] == 'build_model' else 1), row[1].lineno))
            for name, n in nodes:
                if isinstance(n, (ast.Import, ast.ImportFrom)) or (module == 'encoder_structure_search' and name == 'build_model'):
                    chunks.append(ast.unparse(n))
                else:
                    start = min([n.lineno] + [d.lineno for d in getattr(n, 'decorator_list', [])])
                    chunks.append('\n'.join(source.splitlines()[start - 1:n.end_lineno]))
            path = out / (module.replace('.', '/') + '.py')
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('\n\n'.join(chunks) + '\n')
            self.records.append({'module': module, 'source': str(self.paths[module].relative_to(RUN / 'source_snapshot')),
                                 'source_sha256': sha(self.paths[module]), 'selected_symbols': sorted(selected),
                                 'derived_sha256': sha(path)})
        (out / 'komap_seg/__init__.py').write_text('"""HRNet inference dependencies for Model349."""\n')
        write_json(BUNDLE / 'provenance/source_manifest.json', self.records)


def build():
    BUNDLE.mkdir(parents=True, exist_ok=False)
    slicer = Slicer()
    roots = {
        'calibrate_al': ['load_model', 'predict_probabilities'],
        'model183_metric_calibration': ['load_aspp', 'load_rep', 'make_mixed', 'predict_rgb', 'predict_rep'],
        'model187_schedule_branch_blend': ['undo_model181_primary_blend'],
        'highres1024_search': ['choose_source'],
        'hard_source_search': ['route_params'],
        'prior_representation_search': ['percentile_image'],
        'style_tta_search': ['predict_array', 'transform_image'],
        'tent_style_search': ['adapt_model'],
        'model253_highres_scale_finetune': ['Stride1NativeStemASPP', 'predict_scaled'],
        'model335_exhaustive_b4_route_search': ['make_mixed', 'replace_classes', 'bilateral_weights', 'predict', 'undo_model181_primary_blend'],
        'b4_gray': ['load_member', 'load_gray', 'predict_probs'],
        'b4_zcdf': ['load_member', 'load_gray', 'predict_probs'],
        'komap_seg.checkpoint': ['load_model'],
        'komap_seg.dataset': ['build_eval_transform', 'load_image_rgb'],
        'komap_seg.inference': ['predict_probs_tta'],
    }
    for module, names in roots.items():
        for name in names:
            slicer.add(module, name)
    slicer.emit()
    print(f'Inference-only dependency modules: {len(slicer.selected)}', flush=True)

    checkpoints = [a for a in json.loads((EXPORT / 'preservation.json').read_text()) if a['path'].endswith('.pt')]
    manifests = []
    for record in tqdm(checkpoints, desc='Copy Model349 weights', unit='file'):
        src = Path(record['path'])
        relative = src.relative_to(RUN / 'outputs')
        dst = BUNDLE / 'weights' / relative
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        assert sha(dst) == record['sha256']
        manifests.append({'path': str(dst.relative_to(BUNDLE)), 'bytes': dst.stat().st_size,
                          'sha256': record['sha256'], 'original_relative_path': 'outputs/' + str(relative)})
        if (src.parent / 'args.json').exists():
            args = json.loads((src.parent / 'args.json').read_text())
            if 'fold' in src.parent.name:
                # Only values consulted by load_member; training paths are not needed.
                args = {k: args[k] for k in ['arch', 'encoder', 'norm', 'scale', 'target_width', 'input', 'feat_stats'] if k in args}
                write_json(dst.parent / 'args.json', args)
    write_json(BUNDLE / 'provenance/weights_manifest.json', manifests)

    route = json.loads((EXPORT / 'route.json').read_text())
    route['base'] = {k: v for k, v in route['base'].items() if k in [
        'primary_source', 'primary_gamma', 'al3ni_source', 'al3ni_gamma', 'al_source', 'al_gamma',
        'eutectic_source', 'eutectic_gamma', 'prior_scale', 'beta_scale', 'al_delta', 'eut_delta',
        'gaussian_blend', 'eut_edge_blend', 'cleanup_threshold']}
    stats = []
    for p in sorted((RUN / 'data/train/images').glob('*_image.png')):
        gray = np.asarray(Image.open(p).convert('L'), dtype=np.float32)
        stats.append((float(gray.mean()), float(gray.std())))
    calibration = {'priors': json.loads((EXPORT / 'priors.json').read_text()),
                   'train_brightness_mean': float(np.mean([x[0] for x in stats])),
                   'train_brightness_std': float(np.mean([x[1] for x in stats])),
                   'source_train_image_count': len(stats)}
    write_json(BUNDLE / 'configs/calibration.json', calibration)
    config = {'model': 'Model349', 'original_run': RUN.name, 'validation_image_miou': 0.8006879890006227,
              'class_order': ['Al3Ni', 'Primary Si', 'Al', 'Eutectic Si'],
              'palette': [[25, 76, 153], [76, 178, 76], [153, 127, 76], [204, 204, 204]],
              'calibration': 'configs/calibration.json', 'weights_dir': 'weights', 'route': route,
              'inference': {'device': 'cuda', 'threads': 2, 'probability_storage_dtype': 'float16',
                            'b4_folds': 4, 'b4_tta': 'd4',
                            'native': {'patch': 1024, 'stride': 576, 'scale': 1.0, 'tta': 'd4'},
                            'hrnet': {'patch': 512, 'overlap': 170, 'batch_size': 2, 'input_scale': 2.0, 'tta': 'flip4'},
                            'tent': {'steps': 5, 'lr': 1e-5}}}
    (BUNDLE / 'configs/model349.yaml').write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True))

    # A pinned runtime dependency closure, independent of the repository's editable package.
    pending = ['torch', 'torchvision', 'numpy', 'Pillow', 'scipy', 'tqdm', 'segmentation-models-pytorch',
               'timm', 'albumentations', 'opencv-python-headless', 'PyYAML', 'packaging']
    versions = {}
    while pending:
        name = canonicalize_name(pending.pop())
        if name in versions:
            continue
        dist = md.distribution(name)
        versions[name] = dist.version
        for raw in dist.requires or []:
            req = Requirement(raw)
            if req.marker is None or req.marker.evaluate({'extra': ''}):
                pending.append(req.name)
    requirements = ['# Exact installed Model349 inference dependencies (Linux x86_64).',
                    '--extra-index-url https://download.pytorch.org/whl/cu128']
    requirements += [f'{name}=={version}' for name, version in sorted(versions.items())]
    (BUNDLE / 'requirements.lock.txt').write_text('\n'.join(requirements) + '\n')
    conda_list = json.loads(subprocess.check_output(['/home/sppark/miniconda3/bin/conda', 'list', '-n', 'komap', '--json'], text=True))
    python_version = next(x['version'] for x in conda_list if x['name'] == 'python')
    env = {'name': 'komap', 'channels': ['defaults'], 'dependencies': [f'python={python_version}', 'pip', {'pip': ['-r requirements.lock.txt']}]}
    (BUNDLE / 'environment.yaml').write_text(yaml.safe_dump(env, sort_keys=False))
    environment = BUNDLE / 'environment'
    environment.mkdir()
    # Preserve a full environment definition as well as the smaller install definition.
    raw_export = yaml.safe_load(subprocess.check_output(['/home/sppark/miniconda3/bin/conda', 'env', 'export', '-n', 'komap'], text=True))
    raw_export.pop('prefix', None)
    for dep in raw_export['dependencies']:
        if isinstance(dep, dict) and 'pip' in dep:
            dep['pip'] = [line for line in dep['pip'] if 'alphase' not in line and '-e ' not in line and ' @ file:' not in line]
            dep['pip'].insert(0, '--extra-index-url https://download.pytorch.org/whl/cu128')
    (environment / 'environment.full.yaml').write_text(yaml.safe_dump(raw_export, sort_keys=False))
    write_json(environment / 'conda-packages.json', [x for x in conda_list if x['name'] != 'alphase'])
    freeze = (RUN / 'environment.freeze.txt').read_text().splitlines()
    (environment / 'original_training.freeze.txt').write_text('# Original training package inventory; unrelated editable repository excluded.\n' + '\n'.join(x for x in freeze if not x.startswith('-e ')) + '\n')
    write_json(environment / 'runtime_versions.json', {'python': sys.version, 'packages': versions,
               'platform': 'Linux x86_64', 'torch_cuda': '12.8', 'reference_gpu': 'NVIDIA GeForce RTX 4090',
               'fresh_environment_install_tested': False})
    for filename in ['infer.py', 'run.sh', 'verify.py', 'README.md']:
        shutil.copy2(HERE / filename, BUNDLE / filename)
    (BUNDLE / 'run.sh').chmod(0o755)
    print(BUNDLE, flush=True)


if __name__ == '__main__':
    build()
