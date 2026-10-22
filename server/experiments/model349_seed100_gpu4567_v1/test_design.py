"""CPU-only tests for metrics, scheduling, routing, and executable commands."""
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from common import ROOT, BASE, make_plan, validate_plan, class_ious, read_json, write_json
from worker import command_for
from run import eligible, can_launch

class SearchTests(unittest.TestCase):
    def test_empty_class_zero_and_image_first(self):
        truth=[]; pred=[]
        for c in [0,2,3]:
            for d in [0,2,3]:
                count=16 if c==d else 1
                truth.extend([c]*count); pred.extend([d]*count)
        values=class_ious(np.array(pred).reshape(6,9),np.array(truth).reshape(6,9))
        np.testing.assert_allclose(values,[.8,0,.8,.8])
        self.assertAlmostEqual(float(values.mean()),.6)
        # An image with only one perfectly predicted class scores 25%,
        # irrespective of its size. Never aggregate pixels across images.
        one=class_ious(np.zeros((100,100),int),np.zeros((100,100),int)).mean()
        self.assertEqual(one,.25)
        self.assertAlmostEqual((values.mean()+one)/2,.425)

    def test_plan_commands_and_parent_dependencies(self):
        plan=validate_plan(make_plan())
        for task in plan['tasks']:
            if task['kind']!='train': continue
            dest=Path('/tmp/seed100_command_only')/task['id']
            command=command_for(task,dest,'/tmp/parent.pt' if 'parent_rank' in task else None)
            self.assertTrue(Path(command[3]).is_file())
            self.assertEqual(command[command.index('--seed')+1],str(task['seed']))
            output_flag='--out' if 'fold' in task else '--output-dir'
            self.assertEqual(command[command.index(output_flag)+1],str(dest))
            for flag in ['--checkpoint','--data-dir','--data-root','--data']:
                if flag in command: self.assertTrue(Path(command[command.index(flag)+1]).exists())
            if 'fold' in task: self.assertIn('--deterministic_workers',command)
        pending={t['id']:t for t in plan['tasks']}
        self.assertNotIn('m347_s3443',[t['id'] for t in eligible(pending,{})])
        self.assertNotIn('m329_set1',[t['id'] for t in eligible(pending,{'m329_s3451':{}})])
        long=next(t for t in plan['tasks'] if t['id']=='m336_s3431')
        self.assertFalse(can_launch(long,{},60))
        self.assertTrue(can_launch(long,{},4000))

    def test_selection_preserves_baseline_and_measured_winner(self):
        from finalize import select_combination
        def rec(identifier,model,score,selection,**extra):
            return dict(id=identifier,model=model,image_miou=score,selection=selection,**extra)
        r1=rec('good','269',.81,{'269':'good'})
        r2=rec('bad','347',.79,{'347':'bad'})
        def score(s): return .8+.01*('269' in s)-.01*('347' in s)
        active,selection,value,_=select_combination([r1,r2],score)
        self.assertEqual(selection,{'269':'good'}); self.assertAlmostEqual(value,.81)
        active,selection,value,_=select_combination([r2],score)
        self.assertEqual(selection,{}); self.assertEqual(value,.8)
        parent=rec('parent','336',.802,{'336':'parent'})
        child=rec('child','347',.82,{'336':'parent','347':'child'},parent_id='parent')
        def coupled(s): return .82 if s==child['selection'] else .8
        _,selection,value,_=select_combination([parent,child],coupled)
        self.assertEqual(selection,child['selection']); self.assertEqual(value,.82)

    def test_hrnet_worker_seed_resets_private_rng(self):
        # Load only this module; never instantiate or train a model.
        import random
        import torch
        path=ROOT/'source/experiments/model336_hrnet_best/komap_seg/dataset.py'
        sys.path.insert(0,str(path.parent.parent))
        import komap_seg.dataset as ds_module
        from types import SimpleNamespace
        class Transform:
            def set_random_seed(self,seed): self.seed=seed
        ds=SimpleNamespace(rng=random.Random(0),transform=Transform())
        with patch.object(torch.utils.data,'get_worker_info',return_value=SimpleNamespace(dataset=ds)), \
             patch.object(torch,'initial_seed',return_value=3407):
            ds_module.seed_worker(0); first=ds.rng.random()
            ds_module.seed_worker(0); second=ds.rng.random()
        self.assertEqual(first,second); self.assertEqual(ds.transform.seed,3407)

    def test_training_guard(self):
        env=os.environ.copy(); env.pop('MODEL349_USER_STARTED',None); env['CUDA_VISIBLE_DEVICES']=''
        result=subprocess.run([sys.executable,'-B',str(ROOT/'worker.py'),'--run','/tmp/unstarted','--task','none'],
                              env=env,capture_output=True,text=True)
        self.assertNotEqual(result.returncode,0)
        self.assertIn('user-started',result.stderr)

    def test_scheduler_real_subprocesses_resume_and_budget(self):
        # Synthetic JSON writers only: no torch imports, GPU operations, or learning.
        from run import run_tasks
        from unittest.mock import patch
        with tempfile.TemporaryDirectory(prefix='seed100_scheduler_check_') as temporary:
            run=Path(temporary)
            for folder in ['implementation','completed','logs']: (run/folder).mkdir()
            (run/'implementation/worker.py').write_text('''import argparse,json,os,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--run');p.add_argument('--task');a=p.parse_args()
r=Path(a.run); t=next(t for t in json.loads((r/'plan.json').read_text())['tasks'] if t['id']==a.task)
time.sleep(.08)
t['visible_gpu']=os.environ['CUDA_VISIBLE_DEVICES']
(r/'completed'/f'{a.task}.json').write_text(json.dumps(t))
''')
            tasks=[dict(id=f'fake{i}',kind='train',model='mock',seed=3407+i,deps=[],
                        priority=1,expected_seconds=.1) for i in range(5)]
            write_json(run/'plan.json',{'tasks':tasks})
            devices={g:{'uuid':f'fake-uuid-{g}'} for g in [4,5,6,7]}
            with (run/'lock').open('w') as lock, patch('run.validate_plan',side_effect=lambda x:x):
                self.assertTrue(run_tasks(run,[4,5,6,7],devices,.02,.01,lock.fileno(),'pause'))
                records=[read_json(p) for p in (run/'completed').glob('*.json')]
                self.assertEqual(len(records),5)
                self.assertTrue(all(r['visible_gpu'] in {f'fake-uuid-{g}' for g in devices} for r in records))
                events=(run/'events.jsonl').read_text().splitlines()
                self.assertTrue(run_tasks(run,[4,5,6,7],devices,.02,.01,lock.fileno(),'pause'))
                self.assertEqual(len((run/'events.jsonl').read_text().splitlines()),len(events))
                tasks.append(dict(id='too_long',kind='train',model='slow',deps=[],priority=1,expected_seconds=5000))
                write_json(run/'plan.json',{'tasks':tasks})
                self.assertFalse(run_tasks(run,[4,5,6,7],devices,.02,.01,lock.fileno(),'pause'))
                self.assertEqual(read_json(run/'status.json')['pending'],['too_long'])

if __name__=='__main__': unittest.main(verbosity=2)
