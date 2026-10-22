"""A tiny CUDA tensor on each physical device. No models, backward, or optimizer."""
import concurrent.futures
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from run import inventory

CODE = """
import json,os,torch
assert torch.cuda.is_available()
assert torch.cuda.device_count() == 1
x=torch.ones(8,device='cuda')
assert x.sum().item() == 8
torch.cuda.synchronize()
properties=torch.cuda.get_device_properties(0)
print(json.dumps({'visible_devices':os.environ['CUDA_VISIBLE_DEVICES'],
                  'device_count':torch.cuda.device_count(), 'device_name':properties.name,
                  'uuid':str(getattr(properties,'uuid','unavailable')),
                  'tensor_sum':x.sum().item(), 'training_started':False}))
"""


def main():
    devices = inventory()
    assert all(g in devices for g in (6, 7))
    def check(gpu):
        env = os.environ.copy()
        env.update(CUDA_VISIBLE_DEVICES=devices[gpu]["uuid"], OMP_NUM_THREADS="1",
                   MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1")
        result = subprocess.run([sys.executable, "-B", "-c", CODE], env=env,
                                capture_output=True, text=True, timeout=90)
        if result.returncode:
            raise RuntimeError(f"GPU {gpu}: {result.stderr}")
        row = json.loads(result.stdout)
        row["physical_gpu"] = gpu
        assert row["visible_devices"] == devices[gpu]["uuid"]
        if row["uuid"] != "unavailable":
            assert row["uuid"].removeprefix("GPU-") == devices[gpu]["uuid"].removeprefix("GPU-")
        print(f"GPU {gpu}: UUID isolation and CUDA tensor OK", flush=True)
        return row
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        rows = list(pool.map(check, (6, 7)))
    report = {"training_started": False, "test": "two simultaneous isolated CUDA contexts; one 8-element tensor each",
              "records": rows}
    (ROOT / "reports/gpu_visibility_validation.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
