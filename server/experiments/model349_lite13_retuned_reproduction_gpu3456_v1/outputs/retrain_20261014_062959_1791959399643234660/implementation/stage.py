"""Record exact commands and completion after a successful subprocess, not an early checkpoint."""
from pathlib import Path
import os, sys, subprocess, time
from common import now, write, sha

def main():
    name, artifact, *args=sys.argv[1:]
    out=Path(os.environ['OUT']);command=[sys.executable,'-B','-u',*args]
    row=dict(name=name,artifact=artifact,command=command,cwd=str(Path.cwd()),
             env={k:os.environ.get(k) for k in ['CUDA_VISIBLE_DEVICES','PYTHONPATH','OMP_NUM_THREADS']})
    if os.environ.get('PLAN_ONLY')=='1':
        write(out/'plan'/f'{name}.json',row);return
    receipt=out/'stages'/f'{name}.json';log=out/'logs'/f'{name}.log'
    if receipt.exists() or Path(artifact).exists(): raise FileExistsError(f'Refusing to replace prior stage: {name}')
    row.update(status='running',started_at=now(),started_unix=time.time(),log=str(log));write(receipt,row)
    print(f'START {name} GPU={row["env"]["CUDA_VISIBLE_DEVICES"]} log={log}',flush=True)
    t=time.perf_counter()
    with log.open('w') as stream:
        rc=subprocess.call(command,stdout=stream,stderr=subprocess.STDOUT)
    row.update(returncode=rc,wall_seconds=time.perf_counter()-t,finished_at=now())
    if rc==0 and Path(artifact).is_file():row.update(status='done',artifact_sha256=sha(artifact))
    else:row.update(status='failed')
    write(receipt,row)
    print(f'{row["status"].upper()} {name} elapsed={row["wall_seconds"]:.1f}s log={log}',flush=True)
    if row['status']!='done':
        print(log.read_text(errors='replace')[-6000:],flush=True)
        raise SystemExit(rc or 1)

if __name__=='__main__':main()
