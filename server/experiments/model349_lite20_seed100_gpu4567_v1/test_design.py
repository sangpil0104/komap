"""CPU-only checks for search semantics; never construct a training model."""
import copy,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from common import (ROOT,BASE,make_plan,validate_plan,baseline_weights,weight_key,write_json,sha)
from worker import command_for
from engine import apply_candidate,baseline_context,group_name,samples
from report_utils import class_ious

class SearchDesign(unittest.TestCase):
    def test_count_and_dependencies(self):
        plan=make_plan();validate_plan(plan)
        self.assertEqual(len(plan['tasks']),122)
        tasks={t['id']:t for t in plan['tasks']}
        self.assertEqual(len([t for t in tasks.values() if t['kind']=='train' and t['group']!='F']),92)
        self.assertEqual([tasks[f'347_{i:02d}']['parent_rank'] for i in range(12)],[None]*4+[0]*4+[1]*4)
        self.assertTrue(all(tasks[f'F_{i:02d}']['deps']==['features'] for i in range(8)))

    def test_generated_training_arguments(self):
        for t in make_plan()['tasks']:
            if t['kind']!='train':continue
            argv=command_for(t,'/tmp/new_training',parent='/tmp/selected_parent.pt' if t['group']=='347' else None,
                             features='/tmp/selected_features' if t['group']=='F' else None)
            self.assertEqual(argv[argv.index('--seed')+1],str(t['seed']))
            self.assertIn('/tmp/new_training',argv)
            self.assertTrue(Path(argv[3]).is_file())
            if t['group']=='347':self.assertEqual(argv[argv.index('--init-checkpoint')+1],'/tmp/selected_parent.pt')
            if t['group']=='E_b4':
                self.assertEqual(argv[argv.index('--init_checkpoint')+1],str(BASE/f'expert/model329/fold{t["fold"]}/last.pt'))
                self.assertIn('--deterministic_workers',argv)
            if t['group']=='F':self.assertIn('/tmp/selected_features',argv)

    def test_parent_child_are_coupled(self):
        parent_key=weight_key('336');child_key=weight_key('347')
        pair=dict(id='child',context=dict(probabilities={'336':'parent_prob','347':'child_prob'},
                                         weights={parent_key:'parent',child_key:'child'},choices={}))
        single=dict(id='parent2',context=dict(probabilities={'336':'new_prob'},weights={parent_key:'new'},choices={}))
        ctx=apply_candidate(baseline_context(),pair,'347')
        self.assertEqual(ctx['weights'][parent_key],'parent')
        changed=apply_candidate(ctx,single,'336')
        self.assertNotIn(child_key,changed['weights']);self.assertNotIn('347',changed['probabilities'])
        self.assertEqual(apply_candidate(ctx,None,'HR_pair'),baseline_context())

    def test_empty_class_is_zero(self):
        target=np.asarray([[0,0,2,2,3,3]],np.uint8)
        self.assertEqual(class_ious(target,target),[1.,0.,1.,1.])
        self.assertEqual(np.mean(class_ious(target,target)),.75)

    def test_active_routes_and_weights(self):
        weights=baseline_weights();self.assertEqual(len(weights),20)
        self.assertEqual(sum(k.startswith('R/') for k in weights),11)
        rows=samples();self.assertEqual(sum(r['high_family']=='restart61' for r in rows),1)
        self.assertEqual(next(r['name'] for r in rows if r['high_family']=='restart61'),'A71_step2_sample1')

    def test_resume_rejects_changed_artifacts(self):
        from run import verify_completed
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'completed').mkdir();artifact=root/'checkpoint';artifact.write_text('saved')
            task=dict(id='one',deps=[],kind='train',seed=3407)
            write_json(root/'plan.json',dict(tasks=[task]))
            write_json(root/'completed/one.json',dict(**task,artifacts={str(artifact):sha(artifact)}))
            verify_completed(root)
            artifact.write_text('changed')
            with self.assertRaises(ValueError):verify_completed(root)

    def test_selection_never_loses_best_single(self):
        from engine import select_re
        c=dict(id='269_00',group='269',image_miou=.85,context=dict(probabilities={'269':'good'},weights={weight_key('269'):'w'},choices={'269':'269_00'}))
        def evaluate(run,ctx):return dict(image_miou=.85 if ctx['probabilities'].get('269')=='good' else .80,context=copy.deepcopy(ctx))
        with patch('engine.candidate_pool',return_value=[c]),patch('engine.evaluate',side_effect=evaluate):
            result=select_re(Path('/unused'));self.assertEqual(result['image_miou'],.85)

    def test_scheduler_gpu_mapping_and_completed_resume(self):
        import fcntl
        from run import run_tasks,verify_completed
        from common import GPUS,read_json
        tasks=[dict(id='baseline',kind='cpu',group='baseline',deps=[],priority=0,expected_seconds=1),
               *[dict(id=f'fake_{n}',kind='train',group='fake',deps=['baseline'],priority=1,seed=3407+n,expected_seconds=1) for n in range(4)]]
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            for name in ['implementation','completed','logs']:(root/name).mkdir()
            write_json(root/'plan.json',dict(tasks=tasks))
            (root/'implementation/worker.py').write_text('''import argparse,json,os,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--run',type=Path);p.add_argument('--task');a=p.parse_args()
t=next(t for t in json.loads((a.run/'plan.json').read_text())['tasks'] if t['id']==a.task)
time.sleep(.15)
t.update(artifacts={},visible=os.environ['CUDA_VISIBLE_DEVICES'],physical=os.environ['KOMAP_PHYSICAL_GPU'])
(a.run/'completed'/(a.task+'.json')).write_text(json.dumps(t))
''')
            devices={g:dict(uuid=f'TEST-GPU-{g}',used_mib=1,utilization=0) for g in GPUS}
            with (root/'lock').open('a+') as lock,patch('run.validate_plan',side_effect=lambda p:p),patch('run.inventory',return_value=devices):
                fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                completed=run_tasks(root,lock.fileno());verify_completed(root)
                saved={p.name:p.stat().st_mtime_ns for p in (root/'completed').glob('*.json')}
                run_tasks(root,lock.fileno())
                self.assertEqual(saved,{p.name:p.stat().st_mtime_ns for p in (root/'completed').glob('*.json')})
            self.assertEqual({r['physical'] for r in completed.values() if r['kind']=='train'},{'4','5','6','7'})
            self.assertTrue(all(r['visible']=='TEST-GPU-'+r['physical'] for r in completed.values() if r['kind']=='train'))
            self.assertEqual(completed['baseline']['visible'],'')

if __name__=='__main__':unittest.main()
