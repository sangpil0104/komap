"""Parse every distinct training recipe; stop before constructing/training models."""
from pathlib import Path
import os,sys,subprocess,json
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root))
from common import read_json,write_json,environment,stamp
from worker import command_for
output=root/'checks'/('parsed_'+stamp());output.mkdir();seen=set();rows=[]
for task in read_json(root/'plan.json')['tasks']:
    if task['kind']!='train' or task['recipe'] in seen:continue
    seen.add(task['recipe'])
    argv=command_for(task,output/'not_started',parent='/tmp/selected_parent.pt' if task['group']=='347' else None,
                     features='/tmp/selected_features' if task['group']=='F' else None)
    target=output/(task['recipe']+'.json');env=environment(root);env['CUDA_VISIBLE_DEVICES']=''
    subprocess.run([sys.executable,'-B',str(root/'checks/parse_only.py'),argv[3],str(target),*argv[4:]],env=env,check=True)
    rows.append(dict(task=task['id'],recipe=task['recipe'],parsed=str(target)))
write_json(output/'complete.json',dict(ok=True,recipes=len(rows),rows=rows,training_started=False))
print('ALL RECIPES PARSED:',len(rows),'NO TRAINING')
