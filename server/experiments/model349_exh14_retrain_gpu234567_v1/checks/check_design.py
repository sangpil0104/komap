"""Parse all 14 actual training commands without entering training."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import os
import subprocess
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiment import ROOT, read_json, write_json, environment
from scheduler import render, validate_tasks

plan = read_json(ROOT / 'plan.json')
validate_tasks(plan['tasks'])
tasks = {t['id']:t for t in plan['tasks']}
assert len(tasks) == 14
assert tasks['train_model109']['deps'] == []
assert tasks['train_model269']['deps'] == []
assert tasks['train_restart51']['deps'] == ['train_model109']
assert tasks['train_restart59']['deps'] == ['train_model109']
assert tasks['train_restart61']['deps'] == ['train_restart51']
assert tasks['train_model347']['deps'] == ['train_model336']
original = {t['id']:t for t in read_json(ROOT / 'original_recipe.json')['tasks']}
for task in tasks.values():
    before, after = original[task['id']]['argv'], task['argv']
    if '_fold' in task['id']:
        assert after[-1] == '--deterministic_workers'
        after = after[:-1]
    assert len(before) == len(after)
    for old, new in zip(before, after):
        if old != new:
            assert old.startswith('{run}/data') or any(f'{p}/best.pt' in old for p in
                                                     ['model98_hard_negative','model257_full_image_stats_1x'])
            assert new.startswith('{source}/../')

parsed = ROOT / 'checks/parsed_arguments'
parsed.mkdir(exist_ok=True)
env = environment()
env['CUDA_VISIBLE_DEVICES'] = ''


def parse(task):
    # These future output paths are only parsed, never created or executed.
    argv = render(task['argv'], ROOT/'source', ROOT/'checks/never_executed_run', 4)
    output = parsed / f"{task['id']}.json"
    command = [sys.executable, '-B', str(ROOT/'checks/parse_only.py'), argv[0], str(output), *argv[1:]]
    result = subprocess.run(command, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    (parsed / f"{task['id']}.log").write_text(result.stdout)
    if result.returncode:
        raise RuntimeError(f"{task['id']}: {result.stdout}")
    return task['id']


with ThreadPoolExecutor(max_workers=4) as pool:
    complete = list(pool.map(parse, plan['tasks']))
assert not (ROOT/'checks/never_executed_run').exists()
assert read_json(parsed/'train_model336.json')['epochs'] == 70
assert read_json(parsed/'train_model347.json')['epochs'] == 12
assert read_json(parsed/'train_restart61.json')['epochs'] == 4
for task in plan['tasks']:
    if '_fold' in task['id']:
        args = read_json(parsed/f"{task['id']}.json")
        assert args['iters'] == 16000 and args['deterministic_workers'] and args['no_valid_eval']
write_json(ROOT/'checks/design.json', dict(ok=True, training_started=False, commands_parsed=len(complete),
           original_hyperparameters_preserved=True, new_parent_dependencies=True,
           no_training_entrypoint_entered=True))
print('PASS: all 14 actual commands parse; original hyperparameters and new-parent dependencies verified.')
