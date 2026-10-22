from copy import deepcopy
import json
from pathlib import Path
import numpy as np
import pytest
import torch
from alphase.models.cnn import PhaseCNN
from alphase.losses.segmentation import segmentation_loss as original_loss
from alphase.utils import json_hash
from hundredlab.catalog import catalog
from hundredlab.workspace import config,output_path
from hundredlab.models import WarmCNN,build_model,configure_trainable
from hundredlab.losses import segmentation_loss
from hundredlab.optimization import optimizer_for
from hundredlab.dataset import HundredDataset,d4_array
from hundredlab.report import render_report
from hundredlab.evaluation import begin_output,finish_output

torch.set_num_threads(2)
NAMES=list(catalog())

def test_exact_100_and_distinct():
    assert len(NAMES)==100
    assert [sum(x['group']==g for x in catalog().values()) for g in ('recommended1','local19','explore80')]==[1,19,80]
    fingerprints=[]
    for name in NAMES:
        cfg=config(name)
        effective={k:cfg[k] for k in ('model','train','augmentation','sampling','loss')}
        effective['parent']=cfg['hundred']['parent'];effective['hard']=cfg['hundred']['hard']
        fingerprints.append(json_hash(effective))
    assert len(set(fingerprints))==100

@pytest.mark.parametrize('adapter',['aspp','ppm','large_kernel','axial','bottleneck','se','eca','cbam','skip_gate','decoder_refine','highres','channel_adapter'])
def test_warm_identity_and_trainable_additions(adapter):
    torch.manual_seed(31);parent=PhaseCNN().eval();model=WarmCNN(dict(adapter=adapter,variant=2)).eval()
    missing,unexpected=model.load_state_dict(parent.state_dict(),strict=False)
    assert missing and not unexpected
    x=torch.randn(1,3,64,96)
    with torch.no_grad():torch.testing.assert_close(model(x)['logits'],parent(x)['logits'],rtol=0,atol=0)
    model(x)['logits'].square().mean().backward()
    assert any(p.grad is not None and p.grad.abs().sum()>0 for n,p in model.named_parameters() if n.startswith('additions.'))

@pytest.mark.parametrize('block',['residual','separable','convnext','attention'])
@pytest.mark.parametrize('width',[16,24,32])
def test_scratch_shape_and_gradient(block,width):
    from hundredlab.models import ScratchUNet
    model=ScratchUNet(dict(block=block,width=width))
    out=model(torch.randn(1,3,64,96));assert out['logits'].shape==(1,4,64,96)
    out['logits'].square().mean().backward()
    assert model.classifier.weight.grad.isfinite().all()

def test_loss_parity_ignore_and_actual_weighting():
    cfg=config(NAMES[0]);z=torch.randn(1,4,7,9,requires_grad=True);t=torch.randint(0,4,(1,7,9));t[:,:2]=255
    outputs=dict(logits=z)
    a,_=original_loss(outputs,t,cfg['loss']);b,_=segmentation_loss(outputs,t,cfg['loss'])
    torch.testing.assert_close(a,b)
    w={**cfg['loss'],'dice':0.,'lovasz':0.,'class_weights':[1.,2.,3.,1.]}
    got,_=segmentation_loss(outputs,t,w)
    expected=torch.nn.functional.cross_entropy(z,t,weight=torch.tensor(w['class_weights']),ignore_index=255)
    torch.testing.assert_close(got,expected)
    got.backward();assert not z.grad[:,:,:2].any()


def test_fractional_focal_saturated_logits_have_finite_gradient():
    cfg=config('m048_focal_0p5')
    z=torch.tensor([[[[100.]],[[0.]],[[0.]],[[0.]]]],requires_grad=True)
    loss,_=segmentation_loss(dict(logits=z),torch.zeros((1,1,1),dtype=torch.long),cfg['loss'])
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(z.grad).all()

@pytest.mark.parametrize('index',range(44,64))
def test_loss_candidates_finite(index):
    cfg=config(NAMES[index]);z=torch.randn(1,4,9,11,requires_grad=True);t=torch.randint(0,4,(1,9,11));t[:,:2]=255
    out=dict(logits=z,aux_logits=torch.randn(1,4,3,4,requires_grad=True),boundary_logits=torch.randn(1,1,9,11,requires_grad=True))
    loss,_=segmentation_loss(out,t,cfg['loss']);assert loss.isfinite();loss.backward();assert z.grad.isfinite().all()
    assert not z.grad[:,:,:2].any()

def test_trainable_policies_and_lr():
    cfg=config(NAMES[14]);model=build_model(cfg,False)
    for stage in cfg['train']['stages']:
        configure_trainable(model,cfg,stage);opt,_=optimizer_for(model,cfg,stage)
        params=[n for g in opt.param_groups for n in g['names']]
        if stage['policy']=='new':assert all(n.startswith('additions.') for n in params)
        else:assert any(n.startswith('detail.') for n in params)
        assert not any(n.startswith(('aux_head.','boundary_head.')) for n in params)

def test_smoke_path_config_separation():
    cfg=config(NAMES[0]);smoke=config(NAMES[0],True)
    assert cfg['hundred']['smoke'] is False and smoke['hundred']['smoke'] is True
    assert smoke['train']['stages'][0]['max_updates']==2
    assert cfg['train']['stages'][0]['max_updates']==600
    with pytest.raises(ValueError):output_path('../../escape')

def test_no_invented_report_completion():
    text,done=render_report([])
    assert not done and '0/100' in text and '| 100 |' in text
    text,done=render_report([],{'m001_recommended_fullframe':'failed'},True)
    assert not done and 'SMOKE' in text and '실패' in text

def test_interrupted_evaluation_kept(tmp_path):
    directory=tmp_path/'eval';expected={'hash':'a'}
    assert begin_output(directory,expected)
    (directory/'partial.txt').write_text('partial')
    assert begin_output(directory,expected)
    assert len(list(tmp_path.glob('eval.interrupted-*')))==1
    (directory/'result.txt').write_text('ok');finish_output(directory,expected)
    assert not begin_output(directory,expected)
    (directory/'result.txt').write_text('changed')
    with pytest.raises(ValueError):begin_output(directory,expected)

def test_d4_hard_alignment():
    a=np.arange(35).reshape(5,7);h=a==17
    for k in range(8):assert np.array_equal(d4_array(a,k)==17,d4_array(h,k))


@pytest.mark.parametrize('index',range(76,86))
def test_data_variants_change_input_and_keep_valid_labels(index,monkeypatch):
    import hundredlab.dataset as dataset
    gray=np.tile(np.arange(64,dtype=np.uint8)*4,(64,1))
    target=np.tile(np.arange(64,dtype=np.uint8)%4,(64,1))
    monkeypatch.setattr(dataset,'load_image',lambda p:gray)
    monkeypatch.setattr(dataset,'load_target',lambda p:target)
    row=dict(image_id='train',image_path='img',mask_path='mask')
    cfg=config(NAMES[index]);cfg['augmentation'].update(d4=False,extra_probability=1.)
    if index<79:cfg['augmentation']['photometric_probability']=1.
    control=deepcopy(cfg);control['augmentation']=dict(cfg['augmentation'],extra_probability=0.,photometric_probability=0.)
    sample=HundredDataset([row],cfg,cfg['train']['stages'][0])[3]
    plain=HundredDataset([row],control,control['train']['stages'][0])[3]
    assert not torch.equal(sample['detail'],plain['detail'])
    if index==85:assert (sample['target']==255).sum()>0
    else:assert torch.equal(sample['target'],plain['target'])


def test_class_sampling_uses_only_train_rows(monkeypatch):
    import hundredlab.dataset as dataset
    monkeypatch.setattr(dataset,'load_image',lambda p:np.ones((32,32),dtype=np.uint8)*128)
    monkeypatch.setattr(dataset,'load_target',lambda p:np.zeros((32,32),dtype=np.uint8))
    cfg=config(NAMES[86]);rows=[dict(image_id=str(i),image_path='i',mask_path='t',phase1_fraction=v) for i,v in enumerate([0.,1.])]
    data=HundredDataset(rows,cfg,cfg['train']['stages'][0])
    sampled=[data[i]['image_id'] for i in range(200)]
    assert set(sampled)<=set(r['image_id'] for r in rows)
    assert sampled.count('1')>150


def test_suite_attempts_all_100_even_after_one_failure(monkeypatch,tmp_path):
    from argparse import Namespace
    import hundredlab.cli as cli
    import hundredlab.evaluation as evaluation
    import hundredlab.report as reporting
    attempted=[];evaluated=[];reports=[]
    monkeypatch.setattr(cli,'controller',lambda *a:None)
    monkeypatch.setattr(cli,'preflight',lambda *a:{'dataset_sha256':'test'})
    monkeypatch.setattr(cli,'output_path',lambda *a:tmp_path.joinpath(*a))
    monkeypatch.setattr(cli,'inspect_run',lambda *a:(False,False))
    def launch(args):
        attempted.append(args.experiment)
        if args.experiment==NAMES[3]:raise RuntimeError('test failure')
    monkeypatch.setattr(cli,'launch_train',launch)
    monkeypatch.setattr(evaluation,'evaluate_run',lambda *a:evaluated.append(a))
    def report(failures,smoke,publish=False):
        reports.append((dict(failures),publish));return [],False
    monkeypatch.setattr(reporting,'report',report)
    with pytest.raises(RuntimeError,match='1개 미완료'):
        cli.suite(Namespace(experiments=NAMES,gpus=8,smoke=True))
    assert attempted==NAMES and len(evaluated)==198
    assert reports[-1][1] and list(reports[-1][0])==[NAMES[3]]
    assert json.loads((tmp_path/'smoke_suite_status.json').read_text())['state']=='completed_with_failures'


def test_report_publishes_only_complete_real_suite_and_preserves_old(monkeypatch,tmp_path):
    import hundredlab.report as reporting
    from hundredlab.workspace import ROOT
    rows=[]
    for name,item in catalog().items():
        for mode in ('plain','d4'):
            rows.append(dict(experiment=name,group=item['group'],inference=mode,complete=True,
                image_mIoU=1.,global_mIoU=2.,delta_baseline_pp=-77.,common19_delta_pp=-78.,
                improved_images=0,safe_checkpoint_uses_initial=True,minutes=.01,
                **{f'iou_{c}':1. for c in range(4)}))
    monkeypatch.setattr(reporting,'PROJECT',tmp_path)
    monkeypatch.setattr(reporting,'output_path',lambda *a:tmp_path.joinpath('outputs',*a))
    monkeypatch.setattr(reporting,'completed_rows',lambda *a:rows)
    target=tmp_path/'model_results_0920.md';target.write_text('preserve me')
    reporting.report(smoke=True,publish=True)
    assert target.read_text()=='preserve me'
    reporting.report(publish=True)
    assert '100/100 완료' in target.read_text()
    assert list((tmp_path/'outputs/report_history').glob('*.md'))[0].read_text()=='preserve me'
    text,done=reporting.render_report(rows[:-1])
    assert not done
