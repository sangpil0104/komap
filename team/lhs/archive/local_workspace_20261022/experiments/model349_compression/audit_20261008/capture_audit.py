import hashlib,json,platform,subprocess
from datetime import datetime,timezone
from pathlib import Path
BASE=Path('/workspace/model349_compress'); root=BASE/'audit_20261008'
def cmd(args): return subprocess.check_output(args,text=True,stderr=subprocess.STDOUT)
meta={'utc':datetime.now(timezone.utc).isoformat(),'platform':platform.platform(),'gpu':cmd(['nvidia-smi','--query-gpu=index,uuid,name,memory.used,utilization.gpu','--format=csv']),'compute_apps':cmd(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name,used_memory','--format=csv']),'processes':cmd(['ps','-eo','pid,ppid,args']),'storage':cmd(['df','-h','/workspace']),'memory':cmd(['free','-h']),'preexisting_runs':{}}
for name in ('lite14_fp16_valid20','lite14_fp16_tent3_valid20','lite12_fp16_fold326_13_valid20'):
    p=BASE/'runs'/name; row={'exists':p.exists(),'completion_exists':(p/'completion.json').is_file()}
    if p.exists(): row['files']=[{'path':str(f.relative_to(p)),'bytes':f.stat().st_size} for f in sorted(p.rglob('*')) if f.is_file()]
    meta['preexisting_runs'][name]=row
for label,path in [('historical349',Path('/workspace/model349_release_verify/runs/valid20_p100_gpu7')),('historical15',BASE/'runs/lite_rep_free_bundle_v2_valid20'),('historical14',BASE/'runs/lite14_ema_pruned_valid20')]:
    link=root/'runs'/label
    if not link.exists(): link.symlink_to(path,target_is_directory=True)
(root/'server_audit.json').write_text(json.dumps(meta,indent=2)+'\n')
print('audit saved')
