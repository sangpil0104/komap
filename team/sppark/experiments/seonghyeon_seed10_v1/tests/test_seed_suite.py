import json
import os
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import pytest
import torch

from report import allocate_results, comparison, scores
from run import is_complete, run_queue
from runtime import seed_estimate, suite_eta, write_json
from train import CropDataset, seed_all, worker_init
from vendor.common import PALETTE


def test_image_first_metric_and_empty_class_policy():
    # One tiny perfect image and a much larger entirely wrong image have equal weight.
    small = np.array([[0, 1, 2, 3]])
    big = np.tile(small, (100, 10))
    first, second = scores(small, small), scores((big + 1) % 4, big)
    assert (first["image_miou"] + second["image_miou"]) / 2 == 0.5
    target = np.array([[0, 1, 3]])
    row = scores(target, target)
    assert row["image_miou"] == 1.0 and row["image_miou_zero"] == 0.75
    assert row["absent_classes"] == [2]
    row = scores(np.array([[2, 1, 3]]), target)
    assert row["image_miou"] == 0.5  # Primary Si has FP: it is zero, not absent.


def test_comparison_preserves_every_native_pixel(tmp_path):
    target = np.tile(np.array([0, 1, 2, 3], dtype=np.uint8), (13, 25))
    pred = target.copy()
    pred[3, 6] = 0  # Actual Primary Si; the error must be green.
    original = np.full((*target.shape, 3), 81, np.uint8)
    path = tmp_path / "comparison.png"
    comparison(path, original, pred, target, scores(pred, target)["image_miou"])
    saved = np.array(Image.open(path))
    h, w = target.shape
    assert np.array_equal(saved[36:36+h, :w], original)
    assert np.array_equal(saved[h+72:h*2+72, :w], PALETTE[pred])
    assert np.array_equal(saved[h+72:h*2+72, w:], PALETTE[target])
    errors = saved[36:36+h, w:]
    assert np.array_equal(errors[3, 6], PALETTE[2])
    assert np.all(errors[pred == target] == 255)


def test_eta_schedules_two_waves_instead_of_adding_gpu_times():
    status = dict(state="training", iterations=100, iteration=50, eval_every=100,
                  valid_count=20, seconds_per_iteration=1, plain_seconds_per_image=1,
                  d4_seconds_per_image=8, plain_evaluations_done=0)
    remaining, total = seed_estimate(status)
    assert (remaining, total) == (230, 280)
    assert suite_eta([status] * 6, pending=4, slots=6) == 510
    final = dict(status, state="valid_d4", iteration=100, plain_evaluations_done=1, eval_image=10)
    assert seed_estimate(final)[0] == 80
    assert suite_eta([{"state": "loading"}] * 6, 4, 6) is None


def test_numbered_results_never_overwrite(tmp_path):
    first, second = allocate_results(tmp_path), allocate_results(tmp_path)
    assert first != second and first.is_dir() and second.is_dir()


def test_seed_reproduces_multiworker_augmentation_without_training(tmp_path):
    image = np.random.default_rng(4).integers(0, 256, (80, 96), dtype=np.uint8)
    mask = np.tile(np.arange(96, dtype=np.uint8) % 4, (80, 1))
    ip, mp = tmp_path / "image.png", tmp_path / "mask.png"
    Image.fromarray(image).save(ip)
    Image.fromarray(PALETTE[mask]).save(mp)

    def batches(seed):
        seed_all(seed)
        ds = CropDataset([(str(ip), str(mp))], dict(seed=seed, crop=32, target_width=96, iters=4, bs=2))
        loader = torch.utils.data.DataLoader(ds, batch_size=2, num_workers=2,
                    worker_init_fn=worker_init, generator=torch.Generator().manual_seed(seed))
        return [(x.clone(), y.clone()) for x, y in loader]

    a, b, c = batches(7), batches(7), batches(8)
    assert all(torch.equal(x, z) and torch.equal(y, w) for (x, y), (z, w) in zip(a, b))
    assert any(not torch.equal(x, z) for (x, _), (z, _) in zip(a, c))


def test_parallel_queue_logs_failures_and_finishes_other_seeds(tmp_path, capsys):
    session, result = tmp_path / "session", tmp_path / "result"
    session.mkdir()
    result.mkdir()
    (result / "metrics").mkdir()
    fake = tmp_path / "fake_worker.py"
    # This is a subprocess protocol fixture, not a model or training run.
    fake.write_text('''
import json, os, pathlib, sys, time
job = json.loads(pathlib.Path(sys.argv[1]).read_text())
seed = job['seed']; run = pathlib.Path(job['run']); result = pathlib.Path(job['result'])
started = time.time()
print('fixture progress', seed, flush=True)
time.sleep(0.2)
(run/'seen.json').write_text(json.dumps(dict(gpu=os.environ['CUDA_VISIBLE_DEVICES'], started=started, ended=time.time())))
if seed == 3: raise SystemExit(7)
rows = [dict(image=f'image_{i}.png') for i in range(20)]
metric = dict(seed=seed, image_miou=0.5, image_miou_exclude=0.5, image_miou_zero=0.5, per_class_iou=[0.5]*4, per_image=rows)
(result/'metrics'/f'seed_{seed:02d}.json').write_text(json.dumps(metric))
for folder in ['predictions', 'valid_mask_comparison']:
    dest = result/folder/f'seed_{seed:02d}'; dest.mkdir(parents=True)
    for i in range(20): (dest/f'image_{i}.png').write_bytes(b'fixture')
(run/'last.pt').write_bytes(b'fixture')
(run/'status.json').write_text(json.dumps(dict(state='completed', elapsed_seconds=time.time()-started)))
print('fixture done', seed, flush=True)
''')
    jobs = []
    for seed in range(10):
        run = session / f"seed_{seed:02d}"
        run.mkdir()
        job_file = run / "job.json"
        write_json(job_file, dict(seed=seed, run=str(run), result=str(result)))
        jobs.append(dict(seed=seed, state="queued", run=str(run), job_file=str(job_file)))
    code = run_queue(jobs, tmp_path, session, result, heartbeat=0.1,
                     command_builder=lambda j, gpu: [sys.executable, str(fake), j["job_file"]])
    assert code == 1
    assert jobs[3]["state"] == "failed" and jobs[3]["exit_code"] == 7
    assert sum(j["state"] == "completed" for j in jobs) == 9
    seen = [json.loads((Path(j["run"]) / "seen.json").read_text()) for j in jobs]
    assert {s["gpu"] for s in seen} == {str(i) for i in range(6)}
    for i, a in enumerate(seen):
        for b in seen[i+1:]:
            if a["gpu"] == b["gpu"]:
                assert a["ended"] <= b["started"] or b["ended"] <= a["started"]
    output = capsys.readouterr().out
    assert "[SUITE]" in output and "total_ETA=" in output and "[SUITE FAILED]" in output
    for j in jobs:
        assert f"fixture progress {j['seed']}" in output
        assert "fixture progress" in (Path(j["run"]) / "worker.log").read_text()
    assert is_complete(jobs[0], result)
    (result / "valid_mask_comparison/seed_00/image_0.png").unlink()
    assert not is_complete(jobs[0], result)  # last.pt alone must never count as success.
