"""CPU-only recipe, metric, package-inventory and scheduling regression checks."""
import fcntl,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from common import ROOT,GPUS,make_plan,validate_plan,command_for,read_json,write_json,sha,blocking_experiments
from preflight import recipe_check
from report_utils import class_ious

class R70Design(unittest.TestCase):
    def test_recipe_and_plan(self):
        self.assertTrue(recipe_check()['only_R_B4_changes'])
        plan=validate_plan(make_plan());self.assertEqual(len(plan['tasks']),9)
        self.assertEqual(sum(t['kind']=='train' for t in plan['tasks']),3)
        self.assertEqual(plan['gpus'],[4,5,6,7])
        for c in ['322','326','329']:
            argv=command_for(c,Path('/not-started'))
            self.assertEqual(argv[argv.index('--data')+1],str(ROOT/'train_data'))
            self.assertEqual(argv[argv.index('--iters')+1],'21000')
            self.assertIn('--no_valid_eval',argv)

    def test_full70_paths_and_control_paths(self):
        import sys
        sys.path.insert(0,str(ROOT/'frozen'));import infer_lite
        new=infer_lite.load_config(ROOT/'compact_source/selected_R_lite20_r70.yaml')
        old=infer_lite.load_config(ROOT/'compact_source/selected_R_lite20.yaml')
        for c,n in [('322',1),('326',2),('329',3)]:
            self.assertEqual(len(infer_lite.b4_member_dirs(old,c)),n)
            self.assertEqual(infer_lite.b4_member_dirs(new,c),[infer_lite.B4_DIRS[c]+'_full70'])
        self.assertEqual(set(infer_lite.high_candidates(new)),{'restart51','pairwise'})

    def test_metric_absent_zero_and_tie_priority(self):
        from export_results import labels
        mask=np.asarray([[0,0,2,2,3,3]],np.uint8)
        self.assertEqual(class_ious(mask,mask),[1.,0.,1.,1.]);self.assertEqual(np.mean(class_ious(mask,mask)),.75)
        p=np.ones((4,1,1),np.float32)/4
        self.assertEqual(labels(p).item(),2)

    def test_existing_search_lock_is_respected(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'search.lock';path.touch()
            with patch('common.EXTERNAL_LOCKS',[path]),path.open('r') as f:
                fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
                self.assertEqual(blocking_experiments(),[str(path)])
                fcntl.flock(f,fcntl.LOCK_UN)
                self.assertEqual(blocking_experiments(),[])

    def test_changed_completed_file_rejected(self):
        from run import verify_completed
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'completed').mkdir();artifact=root/'state';artifact.write_text('original')
            task=dict(id='one',kind='train',deps=[])
            write_json(root/'plan.json',dict(tasks=[task]))
            write_json(root/'completed/one.json',dict(**task,artifacts={str(artifact):sha(artifact)}))
            verify_completed(root);artifact.write_text('changed')
            with self.assertRaises(ValueError):verify_completed(root)

    def test_eta_respects_training_and_inference_dependencies(self):
        from run import estimate_work_eta
        plan=make_plan()
        self.assertEqual(estimate_work_eta(plan,{}, {},lambda t:t['expected_seconds'],0),5340)

    def test_all_four_gpus_and_resume_with_dummy_workers(self):
        from run import run_tasks,verify_completed
        plan=make_plan()
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            for name in ['implementation','completed','logs']:(root/name).mkdir()
            write_json(root/'plan.json',plan)
            (root/'implementation/worker.py').write_text('''import argparse,json,os,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--run',type=Path);p.add_argument('--task');a=p.parse_args()
t=next(t for t in json.loads((a.run/'plan.json').read_text())['tasks'] if t['id']==a.task)
time.sleep(.1)
t.update(artifacts={},visible=os.environ['CUDA_VISIBLE_DEVICES'],physical=os.environ['KOMAP_PHYSICAL_GPU'])
(a.run/'completed'/(a.task+'.json')).write_text(json.dumps(t))
''')
            devices={g:dict(uuid=f'TEST-GPU-{g}',used_mib=1,utilization=0) for g in GPUS}
            with (root/'lock').open('a+') as lock,patch('run.inventory',return_value=devices),patch('run.blocking_experiments',return_value=[]):
                fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                done=run_tasks(root,lock.fileno());verify_completed(root)
                saved={p.name:p.stat().st_mtime_ns for p in (root/'completed').glob('*.json')}
                run_tasks(root,lock.fileno())
                self.assertEqual(saved,{p.name:p.stat().st_mtime_ns for p in (root/'completed').glob('*.json')})
            gpu=[r for r in done.values() if r['kind']!='cpu']
            self.assertEqual({r['physical'] for r in gpu},{'4','5','6','7'})
            self.assertTrue(all(r['visible']=='TEST-GPU-'+r['physical'] for r in gpu))

if __name__=='__main__':unittest.main()
