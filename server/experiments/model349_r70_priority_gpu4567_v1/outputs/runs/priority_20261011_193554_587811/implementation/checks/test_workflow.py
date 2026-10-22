"""Meaningful CPU checks: actual dummy-process draining and stage failure/resume order."""
import json, os, signal, subprocess, sys, tempfile, threading, time, unittest
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import priority_common as c
import pause_search as draining
import workflow as w

class DrainTest(unittest.TestCase):
    def test_finishes_active_task_before_stop_and_preserves_checkpoint(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); seed=root/'seed'; run=seed/'outputs/runs/fake'
            (run/'implementation').mkdir(parents=True);(run/'completed').mkdir();(root/'work').mkdir()
            c.write(seed/'outputs/latest_run.json',{'run':str(run)})
            c.write(run/'plan.json',{});c.write(run/'implementation_manifest.json',{})
            worker=run/'implementation/worker.py'
            worker.write_text('''import sys,time,json,hashlib
from pathlib import Path
r=Path(sys.argv[sys.argv.index('--run')+1]);(r/'ready').touch()
while not (r/'release').exists():time.sleep(.01)
p=r/'checkpoint.pt';p.write_bytes(b'completed dummy checkpoint')
(r/'completed/task.json').write_text(json.dumps({'kind':'train','checkpoint':str(p),'artifacts':{str(p):hashlib.sha256(p.read_bytes()).hexdigest()}}))
''')
            driver=seed/'run.py'
            driver.write_text('''import sys,subprocess,signal,time,json,fcntl
from pathlib import Path
r=Path(__file__).parent/'outputs/runs/fake'
lock=(r.parents[1]/'search.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX)
signal.signal(signal.SIGTERM,lambda *a:sys.exit(0))
p=subprocess.Popen([sys.executable,str(r/'implementation/worker.py'),'--run',str(r),'--task','task'])
(r/'status.json').write_text(json.dumps({'active':[{'task':'task','log':str(r/'empty.log')}]}))
while True:
 if p.poll() is not None:(r/'unwanted_next_task').touch()
 time.sleep(.02)
''')
            child=subprocess.Popen([sys.executable,str(driver),'train','--execute'],start_new_session=True)
            errors=[]
            try:
                deadline=time.monotonic()+10
                while not (run/'ready').exists():
                    self.assertLess(time.monotonic(),deadline);time.sleep(.02)
                def release_when_stopped():
                    try:
                        while c.process(child.pid)['state']!='T':
                            if time.monotonic()>deadline:raise TimeoutError('not suspended')
                            time.sleep(.01)
                        while not (root/'work/paused_search/active_tasks.json').exists():
                            if time.monotonic()>deadline:raise TimeoutError('worker not recorded')
                            time.sleep(.01)
                        self.assertFalse((run/'checkpoint.pt').exists())
                        (run/'release').touch()
                    except BaseException as e:errors.append(e)
                thread=threading.Thread(target=release_when_stopped);thread.start()
                with patch.multiple(draining,SEED=seed,SEED_RUN=run): result=draining.pause(root/'work')
                thread.join(timeout=5);self.assertFalse(thread.is_alive());self.assertFalse(errors)
                child.wait(timeout=5)
                self.assertEqual(result['training_completed'],1)
                self.assertEqual(result['unfinished_drained_tasks'],[])
                self.assertEqual(len(c.read(root/'work/paused_search/active_tasks.json')),1)
                self.assertFalse((run/'unwanted_next_task').exists())
                self.assertTrue((Path(result['saved_records'])/'completed/task.json').exists())
                self.assertFalse(c.lock_busy(seed/'outputs/search.lock'))
            finally:
                if child.poll() is None:
                    os.killpg(child.pid,signal.SIGCONT);os.killpg(child.pid,signal.SIGKILL);child.wait()

    def test_pid_reuse_is_never_signalled(self):
        with patch.object(draining,'alive',return_value=False),patch.object(draining.os,'kill') as kill:
            draining.signal_identity({'pid':123,'start':'old'},signal.SIGTERM)
            kill.assert_not_called()

class StageTest(unittest.TestCase):
    def run_sequence(self, failure=False, resumed=False):
        with tempfile.TemporaryDirectory() as temp:
            work=Path(temp);r70=work/'r70';seedrun=work/'seed';events=[]
            (work/'implementation').mkdir();c.write(work/'implementation/code_manifest.json',{})
            c.write(work/'state.json',{'phase':'created','r70_run':None})
            c.write(r70/'status.json',{'phase':'complete'});c.write(seedrun/'status.json',{'phase':'complete'})
            completed={str(i):{'kind':'train'} for i in range(100)}
            completed['export']={'kind':'cpu','result':str(work/'seed_result'),'artifacts':{}}
            def pause(path):events.append('pause');c.write(work/'pause_complete.json',{})
            def child(command,work,phase,fd,**kwargs):
                events.append(phase)
                if phase=='r70_train_and_report':
                    if failure:raise RuntimeError('simulated training failure')
                    w.update(work,r70_run=str(r70));c.write(r70/'completed/export.json',{'result':str(work/'r70_result'),'artifacts':{}})
                elif phase=='eight_seed_report':c.write(work/'evaluation_export.json',{'result':str(work/'eight_result'),'files_sha256':{}})
                elif phase=='resume_original_seed100':
                    self.assertEqual(command[-2:],['--resume',str(seedrun)])
                    self.assertTrue((work/'evaluation_export.json').exists())
            if resumed:
                c.write(work/'pause_complete.json',{})
                c.write(work/'r70_complete.json',{'result':str(work/'r70_result'),'artifacts':{}})
                c.write(work/'evaluation_export.json',{'result':str(work/'eight_result'),'files_sha256':{}})
            with patch.multiple(w,SEED_RUN=seedrun,verify_source=lambda:None,verify_files=lambda *a:None,
                pause=pause,real_r70_latest=lambda:None,validate_run_path=lambda p,e:Path(p),lock_busy=lambda p:False,
                run_child=child,evaluate_pool=lambda *a:events.append('evaluate'),records=lambda p:completed):
                if failure:
                    with self.assertRaisesRegex(RuntimeError,'simulated'):w.sequence(work,0)
                else:
                    w.sequence(work,0);self.assertEqual(c.read(work/'state.json')['phase'],'complete')
            return events

    def test_exact_four_stage_order(self):
        self.assertEqual(self.run_sequence(),['pause','r70_train_and_report','evaluate','eight_seed_report','resume_original_seed100'])

    def test_failed_r70_never_starts_evaluation_or_seed_resume(self):
        self.assertEqual(self.run_sequence(failure=True),['pause','r70_train_and_report'])

    def test_completed_reports_are_skipped_on_resume(self):
        self.assertEqual(self.run_sequence(resumed=True),['resume_original_seed100'])

    def test_non_run_fixture_is_rejected(self):
        with self.assertRaisesRegex(ValueError,'Not a real experiment run'):
            w.validate_run_path(c.R70/'checks/package_fixture_20261011_190842_055142',c.R70)

    def test_gpu_inventory_selects_only_physical_4_to_7(self):
        raw='\n'.join(f'{i}, GPU-{i}, 1, 0' for i in range(8))
        with patch.object(c.subprocess,'check_output',return_value=raw):
            devices=c.inventory()
        self.assertEqual(list(devices),[4,5,6,7]);self.assertEqual(devices[4]['uuid'],'GPU-4')

    def test_evaluation_workers_are_bound_to_four_allowed_uuids(self):
        with tempfile.TemporaryDirectory() as temp:
            work=Path(temp);(work/'logs').mkdir();c.write(work/'state.json',{})
            calls=[]
            class Finished:
                pid=99999
                def poll(self):return 0
            def popen(command,**kwargs):
                identifier=command[-1];env=kwargs['env'];calls.append((identifier,env['CUDA_VISIBLE_DEVICES'],env['KOMAP_PHYSICAL_GPU']))
                c.write(work/f'evaluated/{identifier}.json',{'id':identifier})
                return Finished()
            devices={g:dict(uuid=f'GPU-{g}',memory=1,utilization=0) for g in [4,5,6,7]}
            with patch.object(w,'inventory',return_value=devices),patch.object(w,'lock_busy',return_value=False),\
                 patch.object(w.subprocess,'Popen',side_effect=popen),patch.object(w.time,'sleep'):
                w.evaluate_pool(work,0)
            self.assertEqual([r[0] for r in calls],c.IDS)
            self.assertEqual([r[1] for r in calls],[f'GPU-{g}' for g in [4,5,6,7]*2])
            self.assertEqual([r[2] for r in calls],[str(g) for g in [4,5,6,7]*2])

if __name__=='__main__':unittest.main(verbosity=2)
