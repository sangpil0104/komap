"""Run prepared P100 original-route control after transfer; no restart."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
                  OPENBLAS_NUM_THREADS='2', PYTHONDONTWRITEBYTECODE='1')
from pathlib import Path
import fcntl
import hashlib
import json
import signal
import subprocess
import time
import traceback

ROOT = Path(__file__).resolve().parent
PARENT = ROOT.parent
PY = '/workspace/venv-komap/bin/python'
GPU = '3'


def write(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n')


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def gpu_state():
    mem, util = map(int, subprocess.check_output(
        ['nvidia-smi', '-i', GPU, '--query-gpu=memory.used,utilization.gpu',
         '--format=csv,noheader,nounits'], text=True).strip().split(','))
    apps = subprocess.check_output(['nvidia-smi', '-i', GPU, '--query-compute-apps=pid',
              '--format=csv,noheader,nounits'], text=True).strip().splitlines()
    return mem, util, apps


def wait_transfer():
    deadline = time.monotonic() + 14400
    transfer_pid = json.loads((ROOT/'controller.json').read_text())['pid']
    while time.monotonic() < deadline:
        if (ROOT/'failure.json').exists():
            raise RuntimeError('transfer failed; matched control will not run')
        if (ROOT/'completion.json').exists():
            d=json.loads((ROOT/'completion.json').read_text())
            assert d['finished'] and d['fresh_valid20']
            line=subprocess.run(['ps','-p',str(transfer_pid),'-o','args='],capture_output=True,text=True).stdout.strip()
            if not line.endswith('/workspace/venv-komap/bin/python -B -u run_transfer.py'):
                return
        write(ROOT/'control_queue_status.json',{'state':'waiting_transfer_completion','time':time.time()})
        time.sleep(30)
    raise TimeoutError('transfer did not finish within four hours')


def lock_gpu3():
    deadline=time.monotonic()+120
    while time.monotonic()<deadline:
        held=[]
        try:
            for path in sorted(set(PARENT.glob('*/gpu3.lock')) | {ROOT/'control_gpu3.lock'}):
                stream=path.open('a')
                held.append(stream)
                fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
            return held
        except BlockingIOError:
            for stream in held:stream.close()
            time.sleep(5)
    raise TimeoutError('sibling GPU3 lock remained held after transfer finished')


def wait_idle():
    deadline=time.monotonic()+10800
    while time.monotonic()<deadline:
        clear=True
        for _ in range(3):
            mem,util,apps=gpu_state()
            if mem>64 or util or apps:
                clear=False
                break
            time.sleep(2)
        if clear:return
        write(ROOT/'control_queue_status.json',{'state':'waiting_gpu3_idle',
              'memory_mib':mem,'utilization_percent':util,'compute_pids':apps,'time':time.time()})
        time.sleep(30)
    raise TimeoutError('GPU3 did not become idle for matched control')


def main():
    assert not (ROOT/'control_completion.json').exists()
    assert not (ROOT/'control_failure.json').exists()
    lock=json.loads((ROOT/'control_source_lock.json').read_text())
    assert all(sha(ROOT/name)==digest for name,digest in lock.items())
    bundle=ROOT/'matched_control_bundle'
    ready=json.loads((ROOT/'matched_control_prepared.json').read_text())
    manifest=json.loads((bundle/'checkpoint_manifest.json').read_text())
    assert manifest['count']==ready['checkpoint_count']==20
    assert sha(bundle/'infer_lite.py')==ready['control_inference_sha256']
    for row in manifest['checkpoints']:
        path=Path(row['source_path'])
        assert path.is_file() and path.stat().st_size==row['bytes'] and sha(path)==row['sha256']
    wait_transfer()
    held=lock_gpu3()
    wait_idle()
    output=ROOT/'runs/p100_original_matched_valid20'
    assert not output.exists()
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=GPU,NO_ALBUMENTATIONS_UPDATE='1')
    with (ROOT/'p100_original_matched.log').open('x') as log:
        child=subprocess.Popen([PY,'-B','-u',str(bundle/'infer_lite.py'),
             '--input','/workspace/komap_model1/data/valid/images',
             '--output',str(output),'--gpus',GPU,
             '--gt','/workspace/komap_model1/data/valid/masks'],
             cwd=bundle,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        write(ROOT/'control_active_child.json',{'pid':child.pid,'stage':'p100_original_matched_valid',
              'time':time.time()})
        while child.poll() is None:
            time.sleep(5)
            if len(gpu_state()[2])>1 and child.poll() is None:
                os.killpg(child.pid,signal.SIGTERM)
                child.wait(timeout=20)
                raise RuntimeError('Foreign GPU3 compute appeared; stopped only matched control child')
        assert child.returncode==0,child.returncode
    done=json.loads((output/'completion.json').read_text())
    metrics=json.loads((output/'metrics.json').read_text())
    assert done['images']==metrics['count']==20
    write(ROOT/'control_completion.json',{'finished':True,'fresh_valid20':True,
          'miou_percent_reported':100*metrics['image_miou'],
          'elapsed_seconds':done['elapsed_seconds'],'weights':20,
          'p100_checkpoint_bytes':manifest['bytes'],'test_used':False,'time':time.time()})


if __name__=='__main__':
    try:main()
    except BaseException:
        write(ROOT/'control_failure.json',{'traceback':traceback.format_exc(),'time':time.time()})
        raise
