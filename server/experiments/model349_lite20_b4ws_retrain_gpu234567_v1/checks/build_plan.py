"""Create the lite20 training DAG from the verified previous full retraining recipe."""
from pathlib import Path
import json

ROOT=Path(__file__).resolve().parents[1]
OLD=ROOT.parent/'model349_fusion805_lite23_b4ws_retrain_gpu234567_v1'
plan=json.loads((OLD/'plan.json').read_text())
assets=json.loads((ROOT/'compact_source/selected_assets_lite20.json').read_text())
expected={row['path'] for row in assets['checkpoints'] if row['namespace']=='R'}
plan['release_slots']=[slot for slot in plan['release_slots'] if slot['path'] in expected]
plan['tasks']=[t for t in plan['tasks'] if t['id'] not in ['train_restart59','train_model322_fold4']]
for t in plan['tasks']:
    if t['id']=='train_model109':t['family']='ancestor'
plan.update(training_count=35,stage_count=51,deployed_checkpoint_count=20,
            model='Model349_lite20_b4ws',pruned_high_bands=['model109','seed456'],
            removed_training_tasks=['train_restart59','train_model322_fold4'],
            model109_retained_as_training_ancestor=True)
assert {r['path'] for r in plan['release_slots']}==expected and len(expected)==11
assert len(plan['tasks'])==51 and sum(t['role']=='train' for t in plan['tasks'])==35
(ROOT/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
print('lite20: 35 training jobs, 51 stages, R11 + E8 + F1 = 20 deployed weights')
