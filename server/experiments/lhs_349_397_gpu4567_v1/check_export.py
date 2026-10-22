from pathlib import Path
import tempfile,json,subprocess,sys
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parent
palette=np.array([(153,127,76),(25,76,153),(76,178,76),(204,204,204)],dtype=np.uint8)
with tempfile.TemporaryDirectory(prefix='lhs397_export_check_') as tmp:
 repo=Path(tmp); run=repo/'run'; run.mkdir(); (repo/'results').mkdir()
 a=repo/'experiments/lhs_349_397_gpu4567_v1/model349'; a.mkdir(parents=True); (a/'adapters').symlink_to(ROOT/'model349/adapters',target_is_directory=True)
 for d in ['images','masks']: (repo/'dataset/valid'/d).mkdir(parents=True)
 probs={}
 for i in range(20):
  stem=f'A{i}_step1_sample1'; labels=np.tile(np.arange(4,dtype=np.uint8),(4,1)) if i<10 else np.zeros((4,4),np.uint8)
  Image.fromarray(palette[labels]).save(repo/'dataset/valid/masks'/f'{stem}_mask.png')
  Image.new('RGB',(4,4),'white').save(repo/'dataset/valid/images'/f'{stem}_image.png')
  probs[stem]=np.eye(4,dtype=np.float32)[labels].transpose(2,0,1)
 for family,seeds in [('b4',[0]),('hr',[1,2])]:
  for seed in seeds:
   for fold in range(1,5):
    out=run/'valid'/f'{family}_s{seed}_f{fold}'; out.mkdir(parents=True)
    np.savez_compressed(out/('probabilities.npz' if family=='b4' else 'valid_probabilities.npz'),**probs)
 subprocess.run([sys.executable,str(ROOT/'model397/export_valid.py'),str(run),str(repo)],check=True)
 out=Path((run/'result_path.txt').read_text().strip()); metrics=json.loads((out/'metrics.json').read_text())
 assert metrics['image_miou']==.625,metrics
 assert len(list((out/'valid_mask_comparison').glob('*.png')))==20
 panel=np.array(Image.open(out/'valid_mask_comparison/A0_step1_sample1.png'))
 assert panel.shape==(160,8,3) and (panel[76:80,4:]==255).all()
 assert (panel[156:160,0:4]==palette[np.tile(np.arange(4),(4,1))]).all()
 print('PASS: channel order, empty-class IoU=0, 20-image mean=0.625, comparison layout and white correct pixels')
(ROOT/'reports/export_validation.json').write_text(json.dumps(dict(passed=True,synthetic_valid_images=20,expected_image_miou=.625,actual_image_miou=metrics['image_miou'],training_started=False),indent=2))
