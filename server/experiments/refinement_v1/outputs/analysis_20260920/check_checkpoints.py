"""Read checkpoints on CPU without modifying training artifacts."""
import hashlib
import json
from pathlib import Path
import torch

torch.set_num_threads(2)
HERE = Path(__file__).resolve().parent
OUT = HERE.parent
records = []
for folder in sorted((OUT / 'runs').glob('*/seed2026/official')):
    best = torch.load(folder / 'best.pt', map_location='cpu', weights_only=False, mmap=True)
    last = torch.load(folder / 'last.pt', map_location='cpu', weights_only=False, mmap=True)
    cfg = best['resolved_config']
    selected = best['ema_model'] if best['selected_weights'] == 'ema' else best['model']
    context = selected.get('context.project.weight')
    assert last['completed'] and last['optimizer_step'] == 2000
    assert best['world_size'] == 8
    assert len(set(best['train_ids'])) == 70 and len(set(best['validation_ids'])) == 20
    assert not set(best['train_ids']) & set(best['validation_ids'])
    hashes = {}
    for filename in ('best.pt', 'last.pt'):
        digest = hashlib.sha256()
        with (folder / filename).open('rb') as f:
            for data in iter(lambda: f.read(8 << 20), b''):
                digest.update(data)
        hashes[str(folder / filename)] = digest.hexdigest()
    records.append(dict(experiment=folder.parents[1].name,
        best_update=best['optimizer_step'], best_stage=best['stage_index'],
        best_weights=best['selected_weights'], best_plain_miou=best['best_metric'],
        last_update=last['optimizer_step'], completed=last['completed'],
        world_size=best['world_size'], source_sha256=cfg['refinement']['source_sha256'],
        parent_sha256=cfg['refinement']['parent_sha256'], dataset_sha256=best['dataset_hash'],
        context_projection_norm=None if context is None else float(context.norm()), hashes=hashes))
    del best, last, selected, context
(HERE / 'checkpoint_audit.json').write_text(json.dumps(records, indent=2))
print('Checkpoint audit: 10 full runs completed; models read on CPU only.')
