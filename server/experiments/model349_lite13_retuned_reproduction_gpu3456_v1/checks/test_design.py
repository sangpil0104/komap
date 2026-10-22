from pathlib import Path
import json, os, subprocess, sys, tempfile, unittest
from unittest.mock import patch
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'source/training/train_route/experiments'))
from local_image_metrics import summary
from report_utils import class_ious,comparison
from common import read,write
from timed_predict import run_prediction
from export_results import PALETTE

class Design(unittest.TestCase):
    def test_image_metric(self):
        # Three classes each with intersection 8 / union 10, class 1 absent => .6.
        conf=np.zeros((4,4),int)
        for c in [0,2,3]:conf[c,c]=8
        conf[0,2]=conf[2,3]=conf[3,0]=1
        gt=[];pred=[]
        for c in range(4):
            for d in range(4):gt += [c]*conf[c,d];pred += [d]*conf[c,d]
        p=np.array(pred).reshape(3,9);g=np.array(gt).reshape(3,9)
        self.assertAlmostEqual(np.mean(class_ious(p,g)),.6)
        result=summary([p,np.zeros((100,100),int)],[g,np.ones((100,100),int)],list('abcd'))
        self.assertAlmostEqual(result['miou'],.3)
        self.assertNotIn('global_miou',result)
    def test_panel_positions(self):
        pred=np.zeros((100,300),np.uint8);gt=pred.copy();gt[10,20]=1
        original=Image.new('RGB',(300,100),(10,20,30))
        panel=np.asarray(comparison(original,pred,gt,PALETTE,'fixture',.5))
        self.assertEqual(tuple(panel[114,0]),(10,20,30))
        self.assertEqual(tuple(panel[124,320]),tuple(PALETTE[1]))
        self.assertEqual(tuple(panel[114,300]),(255,255,255))
        self.assertEqual(tuple(panel[248,0]),tuple(PALETTE[0]))
        self.assertEqual(tuple(panel[258,320]),tuple(PALETTE[1]))
    def test_failed_stage_never_marked_done(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'checks') as tmp:
            out=Path(tmp);(out/'logs').mkdir();script=out/'fake.py';artifact=out/'last.pt'
            script.write_text('from pathlib import Path\nPath('+repr(str(artifact))+').write_text("early")\nraise SystemExit(7)\n')
            r=subprocess.run([sys.executable,str(ROOT/'stage.py'),'fixture',str(artifact),str(script)],env=dict(os.environ,OUT=str(out)),capture_output=True,text=True)
            self.assertEqual(r.returncode,7)
            self.assertEqual(read(out/'stages/fixture.json')['status'],'failed')
    def test_real_child_timing_boundary_and_inventory(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'checks') as tmp:
            r=Path(tmp);(r/'images').mkdir();(r/'source').mkdir();(r/'weights').mkdir()
            for i in range(10):(r/f'images/{i}_image.png').touch()
            script='''#!/usr/bin/env bash
set -e
mkdir -p "$3/fusion/predicted_labels" "$3/fusion/predicted_masks"
for i in {0..9}; do touch "$3/fusion/predicted_labels/$i.png" "$3/fusion/predicted_masks/$i.png"; done
echo '{}' > "$3/completion.json"
echo '[{"script":"fixture","wall_seconds":0.1}]' > "$3/phase_timings.json"
sleep 0.15
'''
            (r/'source/run_predict.sh').write_text(script)
            with patch('timed_predict.devices',return_value={}):
                record=run_prediction(r/'source',r/'weights',r/'images',r/'out','uuid3,uuid4,uuid5,uuid6','test')
            self.assertGreaterEqual(record['wall_seconds'],.15)
            self.assertEqual(record['seconds_per_image'],record['wall_seconds']/10)
            self.assertTrue((r/'out/inference_timing.json').is_file())
            with self.assertRaises(FileExistsError):run_prediction(r/'source',r/'weights',r/'images',r/'out','x','test')
    def test_fixed_plan(self):
        plan=read(ROOT/'plan.json');rows={r['name']:r for r in plan['tasks']}
        self.assertEqual(len(rows),48)
        self.assertEqual(sum(Path(r['artifact']).suffix=='.pt' for r in rows.values()),33)
        self.assertEqual(plan['gpus'],[3,4,5,6])
        self.assertIn('--no_valid_eval',rows['b4_326_fold2']['command'])
        self.assertNotIn('image_mean_miou_exclude',str(rows))
        for f in [1,3]:self.assertIn(str(ROOT/f'checks/plan_output/E/model329/fold{f}/last.pt'),rows[f'b4ws_fold{f}']['command'])

if __name__=='__main__':unittest.main(verbosity=2)
