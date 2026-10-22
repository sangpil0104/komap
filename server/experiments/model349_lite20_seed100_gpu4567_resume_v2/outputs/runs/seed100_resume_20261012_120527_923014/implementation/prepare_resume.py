"""Prepare an isolated continuation from verified receipts; never start training."""
import fcntl
import shutil
from pathlib import Path

from common import ROOT, GPUS, read_json, write_json, sha, stamp, records
from run import check, snapshot, verify_hashes, verify_completed

ORIGIN = ROOT.parent / 'model349_lite20_seed100_gpu4567_v1/outputs/runs/seed100_20261011_170427_031422'


def import_saved_work(origin, destination):
    """Keep receipt bytes/absolute artifact paths; copy caches before any writes."""
    saved = records(origin)
    for name in ['completed', 'jobs', 'field_cache', 'evaluations']:
        (destination / name).mkdir(exist_ok=True)
    for path in sorted((origin / 'completed').glob('*.json')):
        target = destination / 'completed' / path.name
        if target.exists():
            raise FileExistsError(target)
        shutil.copy2(path, target)
    # new_attempt uses mkdir(exist_ok=False), so these historical links cannot
    # be reused as writable attempts. Interrupted work starts in attempt_2.
    historical = []
    for job in sorted((origin / 'jobs').iterdir()):
        if not job.is_dir():
            continue
        target = destination / 'jobs' / job.name
        target.mkdir()
        for attempt in sorted(job.glob('attempt_*')):
            (target / attempt.name).symlink_to(attempt.resolve(), target_is_directory=True)
            historical.append(str(attempt))
    cache_counts = {}
    for name in ['field_cache', 'evaluations']:
        cache_counts[name] = 0
        for cache in sorted((origin / name).iterdir()):
            if not (cache / 'complete.json').is_file():
                continue
            # Copy, not links: cache locks/reevaluation must not write into the
            # old run. Cached metrics retain immutable old artifact paths.
            shutil.copytree(cache, destination / name / cache.name)
            cache_counts[name] += 1
    count = sum(r['kind'] == 'train' for r in saved.values())
    return dict(
        origin_run=str(origin), completed_tasks=len(saved),
        completed_training=count, remaining_training=100-count,
        completion_receipt_sha256={p.name: sha(p) for p in sorted((origin / 'completed').glob('*.json'))},
        copied_caches=cache_counts, historical_attempts=historical,
        reuse_policy='Verified completed tasks only; unfinished tasks restart in a new attempt directory',
    )


def main():
    info = check()
    (ROOT / 'outputs').mkdir(exist_ok=True)
    with (ROOT / 'outputs/search.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        target = ROOT / 'resume_target.json'
        if target.exists():
            run = Path(read_json(target)['run'])
            verify_hashes(read_json(run / 'implementation_manifest.json'), run / 'implementation', 'Verify prepared source')
            verify_completed(run)
            print(f'ALREADY PREPARED: {run}; no training started', flush=True)
            return
        verify_hashes(read_json(ORIGIN / 'implementation_manifest.json'), ORIGIN / 'implementation', 'Verify original source')
        if read_json(ORIGIN / 'plan.json') != read_json(ROOT / 'plan.json'):
            raise ValueError('Resume must preserve the original seed plan')
        verify_completed(ORIGIN)
        run = ROOT / 'outputs/runs' / f'seed100_resume_{stamp()}'
        run.mkdir(parents=True)
        snapshot(run)
        provenance = import_saved_work(ORIGIN, run)
        provenance.update(prepared=stamp(), gpus=GPUS, training_started=False,
                          original_implementation_manifest_sha256=sha(ORIGIN / 'implementation_manifest.json'),
                          fix='Normalize command argv to str before JSON logging and subprocess execution')
        write_json(run / 'resume_provenance.json', provenance)
        archive = run / 'origin_state'; archive.mkdir()
        for name in ['status.json', 'events.jsonl', 'environment.freeze.txt', 'implementation_manifest.json']:
            shutil.copy2(ORIGIN / name, archive / name)
        verify_completed(run)
        completed = records(run)
        prior = read_json(ORIGIN / 'status.json')
        write_json(run / 'status.json', dict(
            phase='prepared_not_started', training_completed=provenance['completed_training'],
            planned_training=100, completed=list(completed),
            pending=[t['id'] for t in read_json(run / 'plan.json')['tasks'] if t['id'] not in completed],
            gpus=GPUS, active=[], total_elapsed_seconds=prior['total_elapsed_seconds'],
            estimated_remaining_seconds=prior.get('estimated_remaining_seconds'), origin_run=str(ORIGIN),
        ))
        write_json(run / 'preflight.json', info)
        write_json(target, dict(run=str(run), origin_run=str(ORIGIN), prepared=provenance['prepared']))
        write_json(ROOT / 'outputs/latest_run.json', dict(run=str(run)))
        print(f'PREPARED {provenance["completed_training"]}/100; remaining={provenance["remaining_training"]}', flush=True)
        print(f'RUN_DIRECTORY={run}\nNo training started. Start manually using ./start_nohup.sh', flush=True)


if __name__ == '__main__':
    main()
