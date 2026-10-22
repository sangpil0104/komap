"""CPU-only behavioral checks. No real worker training or CUDA operation."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from PIL import Image
from common import ROOT, HIGH, FOLDS, make_plan, validate_plan, class_ious, read_json, write_json, fp16_checkpoint
from worker import command_for

class SearchTests(unittest.TestCase):
    def test_empty_class_zero_and_image_first(self):
        truth=[]; pred=[]
        for c in [0,2,3]:
            for d in [0,2,3]:
                n=16 if c==d else 1
                truth.extend([c]*n); pred.extend([d]*n)
        ious=class_ious(np.array(pred).reshape(6,9),np.array(truth).reshape(6,9))
        np.testing.assert_allclose(ious,[.8,0,.8,.8]); self.assertAlmostEqual(ious.mean(),.6)
        large=class_ious(np.zeros((100,100),int),np.zeros((100,100),int)).mean()
        self.assertEqual(large,.25); self.assertAlmostEqual((ious.mean()+large)/2,.425)

    def test_100_commands_and_fold_dependencies(self):
        from run import eligible
        plan=validate_plan(make_plan())
        candidates=[t for t in plan['tasks'] if t['kind']=='ensemble' or (t['kind']=='train' and 'fold' not in t)]
        self.assertEqual(len(candidates),84)
        pending={t['id']:t for t in plan['tasks']}
        self.assertEqual({t['kind'] for t in eligible(pending,{})},{'baseline'})
        for task in plan['tasks']:
            if task['kind']!='train': continue
            dest=Path('/tmp/not_executed_lite12')/task['id']
            command=command_for(task,dest)
            self.assertTrue(Path(command[3]).is_file())
            self.assertEqual(command[command.index('--seed')+1],str(task['seed']))
            self.assertEqual(command[command.index('--out' if 'fold' in task else '--output-dir')+1],str(dest))
            for flag in ['--data','--data-dir','--data-root','--checkpoint']:
                if flag in command: self.assertTrue(Path(command[command.index(flag)+1]).exists())
            if 'fold' in task:
                self.assertIn('--deterministic_workers',command)
                self.assertIn(task['fold'],FOLDS[task['model']])
        invalid=make_plan(); invalid['tasks'][0]['id']='tampered'
        with self.assertRaises(ValueError): validate_plan(invalid)

    def test_training_entrypoint_help(self):
        env=os.environ.copy(); env.update(CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1',
            NO_ALBUMENTATIONS_UPDATE='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',
            PYTHONPATH=f'{ROOT}/source/experiments:{ROOT}/source/experiments/historical_scripts')
        checked=set()
        for task in make_plan()['tasks']:
            if task['kind']!='train': continue
            command=command_for(task,Path('/tmp/not_executed_lite12')/task['id'])
            if command[3] in checked: continue
            result=subprocess.run(command+['--help'],env=env,capture_output=True,text=True,timeout=90)
            self.assertEqual(result.returncode,0,result.stderr)
            checked.add(command[3])
        self.assertEqual(len(checked),5)

    def test_baseline_and_best_single_candidate_preserved(self):
        from finalize import select_combination
        def rec(model,score):
            return dict(id=model,model=model,image_miou=score,selection={model:model},replacements={})
        good=rec('269',.81); bad=rec('336',.79)
        fn=lambda s:.8+.01*('269' in s)-.01*('336' in s)
        _,selected,value,_=select_combination([good,bad],fn)
        self.assertEqual(selected,{'269':'269'}); self.assertAlmostEqual(value,.81)
        self.assertEqual(select_combination([bad],fn)[1],{})
        a=rec('109',.82); b=rec('restart51',.81)
        fn=lambda s:.8+.02*('109' in s)+.01*('restart51' in s)
        self.assertAlmostEqual(select_combination([a,b],fn)[2],.83)

    def test_fp16_conversion_preserves_original_and_integer_buffers(self):
        import torch
        from common import sha
        with tempfile.TemporaryDirectory() as tmp:
            src=Path(tmp)/'original.pt'; dst=Path(tmp)/'copy.pt'
            torch.save({'model':{'float':torch.tensor([1.12345]),'integer':torch.tensor([3])},'optimizer':'omit'},src)
            digest=sha(src); fp16_checkpoint(src,dst)
            saved=torch.load(dst,map_location='cpu',weights_only=True)
            self.assertEqual(saved['model']['float'].dtype,torch.float16)
            self.assertEqual(saved['model']['integer'].dtype,torch.int64)
            self.assertEqual(sha(src),digest); self.assertNotIn('optimizer',saved)
            with self.assertRaises(FileExistsError): fp16_checkpoint(src,dst)

    def test_training_guard(self):
        env=os.environ.copy(); env.pop('LITE12_USER_STARTED',None); env['CUDA_VISIBLE_DEVICES']=''
        result=subprocess.run([sys.executable,'-B',str(ROOT/'worker.py'),'--run','/tmp/not_started','--task','none'],
                              env=env,capture_output=True,text=True)
        self.assertNotEqual(result.returncode,0); self.assertIn('user-started',result.stderr)

    def test_scheduler_gpu_mapping_resume_and_busy_wait(self):
        from run import run_tasks
        # Fake workers write JSON only. They do not import ML libraries or train.
        with tempfile.TemporaryDirectory(prefix='lite12_scheduler_') as tmp:
            run=Path(tmp)
            for folder in ['implementation','completed','logs']: (run/folder).mkdir()
            (run/'implementation/worker.py').write_text('''import argparse,json,os,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--run');p.add_argument('--task');a=p.parse_args()
r=Path(a.run);t=next(t for t in json.loads((r/'plan.json').read_text())['tasks'] if t['id']==a.task)
time.sleep(.05);t['visible_gpu']=os.environ['CUDA_VISIBLE_DEVICES']
(r/'completed'/f'{a.task}.json').write_text(json.dumps(t))
''')
            tasks=[dict(id=f'fake{i}',kind='train',model='mock',seed=i,deps=[],priority=1,expected_seconds=.1) for i in range(5)]
            devices={g:dict(uuid=f'uuid-{g}',used_mib=0) for g in [4,5,6,7]}
            # A currently occupied GPU must never receive a job.
            devices[5]['used_mib']=17000
            write_json(run/'plan.json',dict(tasks=tasks))
            with (run/'lock').open('w') as lock, patch('run.validate_plan',side_effect=lambda x:x), patch('run.inventory',return_value=devices):
                self.assertTrue(run_tasks(run,[4,5,6,7],devices,.02,.01,lock.fileno(),'finish-all'))
                rows=[read_json(p) for p in (run/'completed').glob('*.json')]
                self.assertEqual(len(rows),5)
                self.assertTrue(all(r['visible_gpu'] in ['uuid-4','uuid-6','uuid-7'] for r in rows))
                events=(run/'events.jsonl').read_text()
                self.assertTrue(run_tasks(run,[4,5,6,7],devices,.02,.01,lock.fileno(),'finish-all'))
                self.assertEqual(events,(run/'events.jsonl').read_text())
                tasks.append(dict(id='too_long',kind='train',model='slow',deps=[],priority=1,expected_seconds=5000))
                write_json(run/'plan.json',dict(tasks=tasks))
                self.assertFalse(run_tasks(run,[4,5,6,7],devices,.02,.01,lock.fileno(),'pause'))

    def test_evaluation_routing_and_comparison_pixels(self):
        import pickle
        import engine
        from common import PALETTE
        with tempfile.TemporaryDirectory(prefix='lite12_eval_') as tmp:
            root=Path(tmp); rows=[]
            for folder in ['base','replacement', '269','326','329','336']: (root/folder).mkdir()
            for i in range(20):
                name=f'image{i}'; family=HIGH[i%4]
                raw=root/f'{name}_image.png'; gt=root/f'{name}_mask.png'
                Image.new('RGB',(8,8),(12,34,56)).save(raw)
                Image.fromarray(np.ones((8,8),np.uint8)).save(gt)
                rows.append(dict(name=name,path=str(raw),gt=str(gt),size=[8,8],high_family=family))
                with (root/'base'/f'{name}.pkl').open('wb') as f: pickle.dump(dict(label=0),f)
                if family=='109':
                    with (root/'replacement'/f'{name}.pkl').open('wb') as f: pickle.dump(dict(label=1),f)
                for component in ['269','326','329','336']: np.save(root/component/f'{name}.npy',np.full((4,8,8),.25,np.float16))
            baseline={c:str(root/c) for c in ['base','269','326','329','336']}
            with patch('engine.samples',return_value=rows), patch('engine.baseline_selection',return_value=baseline), \
                 patch.object(engine.frozen,'predict_fixed',side_effect=lambda item,*_:np.full((8,8),item['label'],np.uint8)):
                value=engine.evaluate(root,{'109':str(root/'replacement')},root/'evaluation',comparisons=True,quiet=True)
            self.assertAlmostEqual(value['image_miou'],.0625)
            self.assertEqual(len(list((root/'evaluation/valid_mask_comparison').glob('*.png'))),20)
            error=Image.open(root/'evaluation/valid_mask_comparison/image1.png')
            correct=Image.open(root/'evaluation/valid_mask_comparison/image0.png')
            self.assertEqual(error.getpixel((1080,340)),PALETTE[1])
            self.assertEqual(correct.getpixel((1080,340)),(255,255,255))
            self.assertEqual(error.getpixel((360,910)),PALETTE[0])
            self.assertEqual(error.getpixel((1080,910)),PALETTE[1])

if __name__=='__main__': unittest.main(verbosity=2)
