from pathlib import Path
import os,sys,tempfile,unittest
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from common import read
from prediction_checks import validate_prediction_inventory
from timed_predict import run_prediction

class ResumeTests(unittest.TestCase):
    def test_actual_failed_run_outputs_are_complete(self):
        parent=Path(read(ROOT/'parent.json')['parent_run'])
        expected={p.stem.removesuffix('_image') for p in (ROOT/'data/valid/images').glob('*_image.png')}
        # This exact old check caused the failure after all actual outputs were saved.
        self.assertNotEqual({p.stem for p in (parent/'pred_valid/fusion/predicted_masks').glob('*.png')},expected)
        self.assertEqual(validate_prediction_inventory(ROOT/'data/valid/images',parent/'pred_valid',20)['images'],20)

    def fixture(self,base,bad=False):
        for name in ['source','weights','images']:(base/name).mkdir()
        for i in range(10):(base/f'images/A{i}_image.png').touch()
        script='''#!/usr/bin/env bash
set -e
mkdir -p "$3/fusion/predicted_labels" "$3/fusion/predicted_masks"
for i in {0..9}; do touch "$3/fusion/predicted_labels/A$i.png" "$3/fusion/predicted_masks/A${i}_mask.png"; done
echo '{}' > "$3/completion.json"
echo '[{"script":"fixture","wall_seconds":0.1}]' > "$3/phase_timings.json"
sleep 0.1
'''
        if bad:script+='rm "$3/fusion/predicted_masks/A0_mask.png"\n'
        (base/'source/run_predict.sh').write_text(script)

    def test_timer_accepts_upstream_mask_suffix(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'checks') as tmp:
            base=Path(tmp);self.fixture(base)
            with patch('timed_predict.devices',return_value={}):
                row=run_prediction(base/'source',base/'weights',base/'images',base/'out','uuid3,uuid4,uuid5,uuid6','test')
            self.assertTrue(row['inventory_verified']);self.assertGreaterEqual(row['wall_seconds'],.1)
            self.assertEqual(row['seconds_per_image'],row['wall_seconds']/10)

    def test_timer_preserves_duration_on_inventory_failure(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'checks') as tmp:
            base=Path(tmp);self.fixture(base,bad=True)
            with patch('timed_predict.devices',return_value={}):
                with self.assertRaisesRegex(ValueError,'A0_mask.png'):
                    run_prediction(base/'source',base/'weights',base/'images',base/'out','uuid3,uuid4,uuid5,uuid6','test')
            row=read(base/'out/inference_timing.json')
            self.assertFalse(row['inventory_verified']);self.assertGreaterEqual(row['wall_seconds'],.1)
            self.assertIn('validation_error',row)

    def test_extra_and_wrongly_named_masks_rejected(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'checks') as tmp:
            base=Path(tmp);self.fixture(base)
            with patch('timed_predict.devices',return_value={}):run_prediction(base/'source',base/'weights',base/'images',base/'out','x','test')
            path=base/'out/fusion/predicted_masks/A0_mask.png';path.rename(path.with_name('A0.png'))
            with self.assertRaisesRegex(ValueError,'unexpected'):validate_prediction_inventory(base/'images',base/'out',10)

    def test_no_training_entrypoint_in_resume(self):
        code=(ROOT/'run.py').read_text()
        for name in ['run_retrain.sh','train_refiner.py','train_bnfreeze.py']:
            self.assertNotIn(name,code)
        self.assertIn("choices=['check','plan','resume']",code)
        self.assertIn('new_training_jobs=0',(ROOT/'preflight.py').read_text())

if __name__=='__main__':unittest.main(verbosity=2)
