"""Independent hand-calculated image-mIoU example across all training families."""
from pathlib import Path
import importlib.util
import sys
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'source/experiments'),str(ROOT/'source/experiments/historical_scripts')]
from local_image_metrics import summary
from restart_ensemble import metrics
from export_results import class_ious
from experiment import write_json

truth=np.repeat(np.array([0,2,3],np.uint8),9).reshape(3,9)
pred=truth.copy();pred[0,0],pred[1,0],pred[2,0]=2,3,0
assert np.isclose(np.mean(class_ious(pred,truth)),.6)
assert np.isclose(summary([pred,truth],[truth,truth],['a','b','c','d'])['image_mean_miou'],.675)
assert np.isclose(metrics([pred,truth],[truth,truth])['image_mean_miou'],.675)
for family in ['model322_team_b4','model326_hybrid_fullstats']:
    spec=importlib.util.spec_from_file_location(f'test_{family}',ROOT/f'source/experiments/{family}/KoMaP/common.py')
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    score,compat,iou=mod.image_miou(pred,truth)
    assert np.isclose(score,.6) and score==compat
    np.testing.assert_allclose(iou,[.8,0,.8,.8])
sys.path.insert(0,str(ROOT/'source/experiments/model336_hrnet_best'))
from komap_seg.metrics import confusion_matrix,iou_per_class
ious=iou_per_class(confusion_matrix(torch.from_numpy(pred.astype(np.int64)),torch.from_numpy(truth.astype(np.int64))))
assert np.isclose(float(ious.mean()),.6)
write_json(ROOT/'checks/metrics.json',dict(ok=True,training_started=False,example_score_percent=60.,
    image_average_percent=67.5,checked=['ASPP/restart','B4 gray','B4 zcdf','HRNet','final result export'],
    empty_class_zero=True,global_miou_computed=False))
print('PASS: all training metric families and final export use image mIoU, absent classes = 0.')
