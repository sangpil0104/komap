"""Package, infer with the unchanged exh14 recipe, and export the new model."""
import argparse
from pathlib import Path
import os
import subprocess
import sys
import time

from experiment import ROOT, read_json, write_json
from package_model import package


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--gpus', required=True)
    args = parser.parse_args()
    run = args.run.resolve()
    status = read_json(run / 'status.json')
    if status['phase'] != 'complete' or len(status['completed']) != 14:
        raise ValueError('Final evaluation requires all fourteen training jobs.')
    started = time.monotonic()
    model = package(run)
    env = dict(os.environ)
    env.pop('PYTHONPATH', None)
    subprocess.run([sys.executable, '-B', str(model / 'verify_bundle.py')], check=True, env=env)
    command = [sys.executable, '-B', '-u', str(model / 'infer_lite.py'),
               '--input', str(ROOT / 'data/valid/images'), '--gt', str(ROOT / 'data/valid/masks'),
               '--output', str(run / 'inference'), '--gpus', args.gpus, '--keep-probabilities']
    write_json(run / 'inference_command.json', command)
    # Workers remain in this finalizer's process group; the coordinator owns cleanup.
    subprocess.run(command, cwd=model, env=env, check=True)
    from export_results import export
    result = export(run)
    write_json(run / 'finished.json', dict(result=str(result), trained=14, finalization_seconds=time.monotonic()-started))
    print(f'RESULT: {result}', flush=True)


if __name__ == '__main__':
    main()
