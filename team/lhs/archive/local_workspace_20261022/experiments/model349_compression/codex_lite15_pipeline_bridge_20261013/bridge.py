from pathlib import Path
import subprocess,json,time,io,tarfile,traceback
W=Path(__file__).resolve().parents[3];R=W/'experiments/model349_compression';O=Path(__file__).resolve().parent
ANA=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10','anacnu'];P100=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10','-p','20670','root@168.188.128.36']
def cmd(host,s,input=None):return subprocess.run(host+[s],input=input,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=30,check=True).stdout
def read(host,path):
 b=cmd(host,"if test -f '"+path+"'; then cat '"+path+"'; else printf '{}'; fi");return json.loads(b)
def marker(host,path,d):cmd(host,"cat > '"+path+"'",json.dumps(d,indent=2).encode())
def collect(host,path,names,local):
 local.mkdir(parents=True,exist_ok=True);b=cmd(host,"tar -C '"+path+"' -cf - "+' '.join(names));tf=tarfile.open(fileobj=io.BytesIO(b))
 for member in tf:
  assert member.name in names and member.isfile();(local/member.name).write_bytes(tf.extractfile(member).read())
def dump(name,d):(O/name).write_text(json.dumps(d,ensure_ascii=False,indent=2))
def main():
 if (O/'completion.json').exists():return
 done=set();rem='/workspace/model349_compress/codex_lite15_B4_HR_reduction_20261013';hr='/home/sangpil/komap/codex_lite15_HRtrain70_evaluation_20261013';warm='/home/sangpil/komap/codex_lite15_B4_train70_EMA_SWA_20261013';share='/workspace/model349_compress/codex_lite15_model_sharing_20261013'
 while True:
  try:
   for host,path,label in [(P100,rem,'removal'),(ANA,hr,'HRtrain70'),(ANA,warm,'B4warm'),(P100,share,'sharing')]:
    failure=read(host,path+'/failure.json')
    if failure:dump('status.json',{'stage':'dependency failure; preserve jobs for inspection','phase':label,'failure':failure,'time':time.time()});raise RuntimeError(label+' failure; no downstream marker issued')
   if 'removal' not in done and (d:=read(P100,rem+'/completion.json')):
    collect(P100,rem,['completion.json','ranking.json','per_image.json','preflight.json'],R/'codex_lite15_B4_HR_reduction_20261013');marker(ANA,hr+'/removal_ready.json',{'source':'P100 removal/HR subset screen','completion':d,'time':time.time()});done.add('removal')
   if 'HRtrain70' not in done and (d:=read(ANA,hr+'/completion.json')):
    collect(ANA,hr,['completion.json','ranking.json','per_image.json'],R/'codex_lite15_HRtrain70_evaluation_20261013');done.add('HRtrain70')
   if 'B4warm' not in done and (d:=read(ANA,warm+'/completion.json')):
    collect(ANA,warm,['completion.json','ranking.json','per_image.json','evaluation_completion.json','source_receipt.json'],R/'codex_lite15_B4_train70_EMA_SWA_20261013');marker(P100,share+'/warm_phase_ready.json',{'source':'ANA warmstart/EMA/SWA/soup complete','completion':d,'time':time.time()});done.add('B4warm')
   if 'sharing' not in done and (d:=read(P100,share+'/completion.json')):
    collect(P100,share,['completion.json','ranking.json','per_image.json'],R/'codex_lite15_model_sharing_20261013');done.add('sharing')
   dump('status.json',{'stage':'sequential pipeline coordination','completed_phases':sorted(done),'time':time.time(),'automatic_adoption':False})
   if done=={'removal','HRtrain70','B4warm','sharing'}:dump('completion.json',{'time':time.time(),'phases':sorted(done),'all_phase_results_collected':True,'release_replaced':False});return
  except Exception:dump('last_error.json',{'time':time.time(),'trace':traceback.format_exc(),'next_retry_seconds':20})
  time.sleep(20)
if __name__=='__main__':main()
