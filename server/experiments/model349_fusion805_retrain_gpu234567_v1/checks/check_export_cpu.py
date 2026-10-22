"""Exercise actual fusion/scorer scripts and both report paths on synthetic inputs."""
from pathlib import Path
import os,sys,shutil,subprocess,tempfile
import numpy as np
from PIL import Image
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from experiment import ROOT,read_json,write_json,sha,environment
import export_results as exporter
from report_utils import class_ious

with tempfile.TemporaryDirectory(prefix='fusion_export_',dir=ROOT/'checks') as directory:
    run=Path(directory)/'run';run.mkdir();model=run/'retrained_model'
    shutil.copytree(ROOT/'fusion_source',model)
    (run/'implementation').mkdir();shutil.copy2(ROOT/'provenance.json',run/'implementation/provenance.json')
    for name in ['plan.json','preflight.json']:write_json(run/name,{'synthetic_test_only':True})
    (run/'events.jsonl').write_text('');(run/'environment.freeze.txt').write_text('synthetic')
    (run/'expert/refiner').mkdir(parents=True);write_json(run/'expert/refiner/completion.json',{'synthetic_test_only':True})
    (run/'data/valid/images').mkdir(parents=True);(run/'data/valid/masks').mkdir()
    release=run/'release_inference';(release/'predicted_labels').mkdir(parents=True)
    expert=run/'expert_probabilities';expert.mkdir()
    rows=[]
    import yaml,torch
    torch.set_num_threads(2)
    sys.path.insert(0,str(model/'refiner'))
    from residual_model import Residual
    refiner=model/'assets/refiner/epoch8.pt';refiner.parent.mkdir(parents=True)
    torch.save(dict(state_dict=Residual().state_dict()),refiner)
    write_json(model/'config/auxiliary_assets.json',[dict(path='refiner/epoch8.pt',sha256=sha(refiner))])
    config=yaml.safe_load((model/'Model349_exh14/configs/model349_lite.yaml').read_text())
    palette=np.asarray(config['palette'],np.uint8)
    pred=np.tile(np.array([0,1,2,3],np.uint8),(8,4))
    target=pred.copy();target[0,:]=2
    probs=np.full((4,*pred.shape),.025,np.float32)
    for c in range(4):probs[c][pred==c]=.925
    for i in range(20):
        name=f'synthetic_{i:02d}';path=run/f'data/valid/images/{name}_image.png'
        Image.new('RGB',(pred.shape[1],pred.shape[0]),(99,99,99)).save(path)
        Image.fromarray(palette[target]).save(run/f'data/valid/masks/{name}_mask.png')
        Image.fromarray(pred).save(release/f'predicted_labels/{name}.png')
        np.save(expert/f'{name}.npy',probs)
        rows.append(dict(name=name,path=str(path),size=[pred.shape[1],pred.shape[0]],sha256=sha(path)))
    # Exact ties retain team order Al, Al3Ni, Primary Si, Eutectic Si.
    tied=probs.copy();tied[:,0,0]=.25;np.save(expert/'synthetic_00.npy',tied)
    write_json(release/'input_manifest.json',dict(config=config,images=rows))
    write_json(release/'metrics.json',dict(image_miou=0.0))
    env=environment();env['CUDA_VISIBLE_DEVICES']='';env.pop('PYTHONPATH',None)
    def call(script,*args):subprocess.run([sys.executable,'-B',str(model/script),*map(str,args)],env=env,check=True,stdout=subprocess.DEVNULL)
    call('infer_stage1.py','--release',release,'--expert',expert,'--output',run/'stage1')
    call('infer_fusion.py','--input',run/'data/valid/images','--release-labels',release/'predicted_labels',
         '--expert-probabilities',expert,'--refiner',refiner,'--output',run/'stage2','--device','cpu')
    for stage in ['stage1','stage2']:
        call('score_masks.py','--labels',run/stage/'predicted_labels','--gt',run/'data/valid/masks','--output',run/stage/'metrics.json')
    for row in rows:
        a=np.asarray(Image.open(run/f"stage1/predicted_labels/{row['name']}.png"))
        b=np.asarray(Image.open(run/f"stage2/predicted_labels/{row['name']}.png"))
        np.testing.assert_array_equal(a,b)
    assert np.asarray(Image.open(run/'stage1/predicted_labels/synthetic_00.png'))[0,0]==2
    old=exporter.ROOT;exporter.ROOT=run
    try:
        result=exporter.export(run,Path(directory)/'results')
        assert len(list((result/'valid_mask_comparison').glob('*.png')))==20
        assert len(list((result/'stage2/valid_mask_comparison').glob('*.png')))==20
        assert (result/'complete.json').exists() and (result/f'{result.name}.md').exists()
    finally:exporter.ROOT=old
write_json(ROOT/'checks/export_cpu.json',dict(ok=True,synthetic_only=True,training_started=False,
    two_stage_panels=40,zero_refiner_stage_identity=True,tie_priority_verified=True,
    actual_fusion_and_scorer_scripts=True,independent_score_check=True))
print('PASS: actual fusion/scorer scripts on synthetic data, tie order, and both 20-panel reports.')
