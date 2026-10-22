"""Run a reduced expert with the unchanged R14 and fusion rules."""
from pathlib import Path
import argparse,subprocess,sys,json
ROOT=Path(__file__).resolve().parent

def main():
 p=argparse.ArgumentParser()
 p.add_argument('--parent-package',type=Path,required=True,help='Extracted Model349_fusion805 reproduction package')
 p.add_argument('--assets',type=Path,required=True)
 p.add_argument('--input',type=Path,required=True)
 p.add_argument('--output',type=Path,required=True)
 p.add_argument('--hr2-folds',type=int,nargs='+',choices=[1,2,3,4])
 p.add_argument('--layout',choices=['hr2_subset','control','seed1','seed2','b4','hr','hr1','hr2','drop_seed2_fold4','drop_seed2_folds34','drop_seed2_folds234'],required=True)
 p.add_argument('--stage',type=int,choices=[1,2],default=2)
 p.add_argument('--gpus',default='0')
 p.add_argument('--release',type=Path,help='Optional existing R14 output directory')
 a=p.parse_args()
 if a.layout=='hr2_subset' and (not a.hr2_folds or len(set(a.hr2_folds))!=len(a.hr2_folds)):p.error('hr2_subset requires unique --hr2-folds')
 if a.layout!='hr2_subset' and a.hr2_folds:p.error('--hr2-folds is only valid with hr2_subset')
 assert not a.output.exists();a.output.mkdir(parents=True)
 def run(script,*args):subprocess.run([sys.executable,str(script),*map(str,args)],check=True)
 release=a.release or a.output/'release'
 if not a.release:run(a.parent_package/'Model349_exh14/infer_lite.py','--input',a.input,'--output',release,'--gpus',a.gpus)
 import os
 env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=a.gpus
 subprocess.run([sys.executable,str(ROOT/'infer_expert.py'),'--parent-package',str(a.parent_package),'--input',str(a.input),'--assets',str(a.assets),'--output',str(a.output/'expert'),'--layout',a.layout,'--device','cuda',*(['--hr2-folds',*map(str,a.hr2_folds)] if a.hr2_folds else [])],env=env,check=True)
 if a.stage==1:run(a.parent_package/'infer_stage1.py','--release',release,'--expert',a.output/'expert','--output',a.output/'fusion')
 else:run(a.parent_package/'infer_fusion.py','--input',a.input,'--release-labels',release/'predicted_labels','--expert-probabilities',a.output/'expert','--refiner',a.assets/'refiner/epoch8.pt','--output',a.output/'fusion','--device','cpu')
 (a.output/'completion.json').write_text(json.dumps({'layout':a.layout,'stage':a.stage,'gt_read':False,'parent_release_supplied':bool(a.release)},indent=2)+'\n')
if __name__=='__main__':main()
