"""Real subprocess scheduling with synthetic text artifacts; no ML execution."""
import copy
import json
import os
from pathlib import Path
import sys
import tempfile
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiment import ROOT, GPUS, read_json, write_json
from scheduler import run_tasks, render


def alive(pid):
    try:
        return Path(f'/proc/{pid}/stat').read_text().split(') ')[1].split()[0] != 'Z'
    except FileNotFoundError:
        return False


def simulate(tasks, destination, modes=None, **kwargs):
    destination.mkdir()
    fixture = copy.deepcopy(tasks)
    for task in fixture:
        for key in ['requires','produces']:
            task[key] = render(task.get(key, []), ROOT/'source', destination, 4)
    spec = destination/'fixture.json'
    write_json(spec,fixture)
    for task in fixture:
        task['argv'] = [str(ROOT/'checks/fake_worker.py'), (modes or {}).get(task['id'],'ok'), str(spec), task['id']]
    return run_tasks(fixture, destination, ROOT/'source', GPUS, {g:f'TEST-GPU-{g}' for g in GPUS},
                     sys.executable, os.environ.copy(), poll_interval=.01, report_interval=120, **kwargs)


with tempfile.TemporaryDirectory(prefix='exh14_scheduler_') as temporary:
    parent = Path(temporary)
    tasks = read_json(ROOT/'plan.json')['tasks']
    events = simulate(tasks, parent/'success')
    by_id = {t['id']:t for t in tasks}
    completed, active, seen, max_active = set(), {}, set(), 0
    for event in events:
        if event['event']=='started':
            assert set(by_id[event['task']]['deps']) <= completed
            gpu = event['gpu']
            if gpu is None:
                assert event['cuda_visible_devices']==''
            else:
                assert gpu in GPUS and gpu not in active
                assert event['cuda_visible_devices']==f'TEST-GPU-{gpu}'
                active[gpu]=event['task'];seen.add(gpu);max_active=max(max_active,len(active))
        elif event['event']=='finished':
            completed.add(event['task'])
            if event['gpu'] is not None:assert active.pop(event['gpu'])==event['task']
    assert len(completed)==53 and seen==set(GPUS) and max_active==6
    base=[dict(id='bad',serial_index=0,priority=0,kind='gpu',deps=[],argv=[],produces=[]),
          dict(id='sibling',serial_index=1,priority=0,kind='gpu',deps=[],argv=[],produces=[]),
          dict(id='dependent',serial_index=2,priority=0,kind='gpu',deps=['bad'],argv=[],produces=[])]
    try:
        simulate(base,parent/'failure',{'bad':'fail','sibling':'linger'})
    except RuntimeError as error:
        assert 'exit 7' in str(error)
    else:
        raise AssertionError('Failed worker was accepted.')
    failed=[json.loads(line) for line in (parent/'failure/events.jsonl').read_text().splitlines()]
    assert not any(e.get('task')=='dependent' for e in failed)
    for event in failed:
        if event['event']=='started':assert not alive(event['pid'])
    child=int((parent/'failure/fixture.child_pid').read_text())
    for _ in range(100):
        if not alive(child):break
        time.sleep(.01)
    assert not alive(child)
    missing=copy.deepcopy(base[:1]);missing[0]['produces']=['{run}/missing.pt']
    try:
        simulate(missing,parent/'missing',{'bad':'missing'})
    except RuntimeError as error:
        assert 'required output' in str(error)
    else:
        raise AssertionError('Missing checkpoint was accepted.')
    # Busy GPUs must be withheld, including a case with no immediately free GPU.
    calls=[]
    def idle():
        calls.append(1)
        return [] if len(calls)==1 else [7]
    waited=simulate(base[:1],parent/'waiting',idle_gpus=idle)
    assert [e['gpu'] for e in waited if e['event']=='started']==[7]
write_json(ROOT/'checks/scheduler.json',dict(ok=True,training_started=False,synthetic_only=True,
    tasks=53,gpus=sorted(seen),max_concurrent=6,dependency_order=True,exclusive_gpu_assignment=True,
    failure_descendant_cleanup=True,missing_checkpoint_rejected=True,busy_gpus_waited=True))
print('PASS: all 53 dependencies, GPUs 2~7, concurrency 6, failure cleanup, output checks, GPU waiting.')
