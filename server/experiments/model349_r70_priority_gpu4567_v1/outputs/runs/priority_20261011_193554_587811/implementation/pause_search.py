"""Drain active tasks before stopping only the pinned seed-search coordinator."""
import os, signal, shutil, time
from pathlib import Path
from tqdm import tqdm
from priority_common import SEED, SEED_RUN, read, write, records, process, alive, lock_busy, sha, tail

def find_coordinator():
    matches = []
    for path in Path('/proc').glob('[0-9]*'):
        info = process(int(path.name))
        if not info or info['state'] in ['Z', 'X']: continue
        args = info['args']
        if str(SEED/'run.py') in args and 'train' in args and '--execute' in args:
            if '--resume' in args and Path(args[args.index('--resume')+1]).resolve() != SEED_RUN:
                raise RuntimeError('Another seed run is active; refusing to signal it.')
            matches.append(info)
    if len(matches) > 1: raise RuntimeError('Multiple seed coordinators; refusing ambiguous stop.')
    if matches and Path(read(SEED/'outputs/latest_run.json')['run']) != SEED_RUN:
        raise RuntimeError('Active search is not the pinned run.')
    return matches[0] if matches else None

def signal_identity(identity, signum):
    if alive(identity):
        try: os.kill(identity['pid'], signum)
        except ProcessLookupError: pass

def stop_coordinator(identity):
    # TERM is queued before CONT, so the existing Python cleanup handler runs on wake-up.
    signal_identity(identity, signal.SIGTERM)
    signal_identity(identity, signal.SIGCONT)
    deadline = time.monotonic()+30
    while alive(identity) and time.monotonic() < deadline: time.sleep(.2)
    if alive(identity): raise RuntimeError(f'Coordinator {identity["pid"]} did not stop; later stages blocked.')

def pause(work):
    work = Path(work); archive = work/'paused_search'; archive.mkdir(exist_ok=True)
    coordinator = find_coordinator(); workers = []
    if coordinator:
        write(archive/'coordinator_before_stop.json', coordinator)
        print(f'DRAIN: stop assigning new tasks; coordinator PID={coordinator["pid"]}. Active tasks finish normally.', flush=True)
        try:
            signal_identity(coordinator, signal.SIGSTOP)
            deadline = time.monotonic()+5
            while alive(coordinator) and process(coordinator['pid'])['state'] != 'T':
                if time.monotonic() > deadline: raise RuntimeError('Coordinator did not suspend.')
                time.sleep(.05)
            if alive(coordinator):
                children = Path(f'/proc/{coordinator["pid"]}/task/{coordinator["pid"]}/children').read_text().split()
                for pid in children:
                    row = process(int(pid))
                    if row and alive(row):
                        # inventory() can have a short-lived nvidia-smi child when suspended.
                        if Path(row['args'][0]).name == 'nvidia-smi':
                            continue
                        if str(SEED_RUN/'implementation/worker.py') not in row['args'] or str(SEED_RUN) not in row['args']:
                            raise RuntimeError(f'Unexpected child: {row}; refusing to guess task ownership.')
                        row['task'] = row['args'][row['args'].index('--task')+1]; workers.append(row)
            write(archive/'active_tasks.json', workers)
            status = read(SEED_RUN/'status.json')
            logs = {r['task']: r['log'] for r in status.get('active', [])}
            with tqdm(total=len(workers), desc='Save active seed tasks', unit='task', mininterval=5) as bar:
                start = time.monotonic(); last = -1e9
                while True:
                    remaining = [r for r in workers if alive(r)]
                    bar.update(len(workers)-len(remaining)-bar.n)
                    if not remaining: break
                    if time.monotonic()-last > 20:
                        print(f'DRAIN elapsed={(time.monotonic()-start)/60:.1f}m | active={len(remaining)} | per-task progress/ETA below', flush=True)
                        for row in remaining:
                            print(f'[{row["task"]}] {tail(logs.get(row["task"], "")) if row["task"] in logs else "finishing current task"}', flush=True)
                        last = time.monotonic()
                    time.sleep(1)
        finally:
            stop_coordinator(coordinator)
    elif lock_busy(SEED/'outputs/search.lock'):
        raise RuntimeError('Seed lock held without a recognized coordinator; refusing to kill unknown processes.')
    if lock_busy(SEED/'outputs/search.lock'):
        raise RuntimeError('Seed workers still hold the lock; r70 cannot start.')
    # Keep old weights/cache/source in place; archive receipts and immutable checkpoint identities.
    done = records(SEED_RUN)
    snapshot = archive/f'records_{time.time_ns()}'; snapshot.mkdir()
    for filename in ['plan.json', 'status.json', 'implementation_manifest.json', 'events.jsonl']:
        path = SEED_RUN/filename
        if path.exists(): shutil.copy2(path, snapshot/filename)
    shutil.copytree(SEED_RUN/'completed', snapshot/'completed')
    checkpoints = {row['checkpoint']: row['artifacts'][row['checkpoint']] for row in done.values() if 'checkpoint' in row}
    for path, expected in tqdm(checkpoints.items(), desc='Verify saved checkpoints', unit='model', mininterval=5):
        if sha(path) != expected: raise ValueError(f'Checkpoint changed: {path}')
    result = dict(seed_run=str(SEED_RUN), training_completed=sum(r['kind']=='train' for r in done.values()),
                  saved_records=str(snapshot), completed_tasks=sorted(done), checkpoint_sha256=checkpoints,
                  unfinished_drained_tasks=[r['task'] for r in workers if r['task'] not in done],
                  resume_granularity='completed task; incomplete attempts are retained and rerun')
    write(work/'pause_complete.json', result)
    print(f'SEARCH SAVED / STOPPED: {result["training_completed"]}/100 complete.', flush=True)
    return result
