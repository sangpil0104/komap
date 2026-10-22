"""Stop each actual training entrypoint at argparse, before model construction."""
from pathlib import Path
import sys,subprocess
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root))
from common import command_for,environment,write_json,stamp
out=root/'checks'/('parsed_'+stamp());out.mkdir()
for c in ['322','326','329']:
    argv=command_for(c,out/'not-started');env=environment(root);env['CUDA_VISIBLE_DEVICES']=''
    subprocess.run([sys.executable,'-B',str(root/'checks/parse_only.py'),argv[3],str(out/f'{c}.json'),*argv[4:]],env=env,check=True)
write_json(out/'complete.json',dict(ok=True,recipes=3,training_started=False))
print('ALL THREE TRAINING RECIPES PARSED; NO TRAINING')
