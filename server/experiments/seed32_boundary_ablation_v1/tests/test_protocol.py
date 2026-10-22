from copy import deepcopy
import json
from pathlib import Path
import types

import numpy as np
import pytest
import torch
from PIL import Image

from boundarylab.config import CHECKPOINT, ENCODER_WEIGHTS, make_plan, selected_arms, validate_plan
from boundarylab.data import TrainDataset, indexed_seed, native_roi, one_sample_collate
from boundarylab.inference import d4, inverse_d4, predict
from boundarylab.io import immutable_torch_save, read_json, reserve_result, sha256, write_json
from boundarylab.losses import ce_dice, native_loss
from boundarylab.metrics import PALETTE, boundary_band, mask_ids, metrics, summarize
from boundarylab.models import BoundaryModel, NativeRefiner, restore_probs
from boundarylab.report import comparison_panel, finish_report, result_manifest, verify_result

ROOT = Path(__file__).resolve().parents[1]


def test_exactly_one_factor_and_fresh_initialization():
    arms = make_plan()
    assert len(arms) == 13
    assert sum(a['kind']=='train' for a in arms) == 10
    for a in arms:
        if a['kind']=='train':
            assert a['initialization'] == str(ENCODER_WEIGHTS)
            assert a['initialization'] != str(CHECKPOINT)
    bad = deepcopy(arms)
    bad[-1]['config']['crop'] = 768
    with pytest.raises(ValueError, match='ONE factor'):
        validate_plan(bad)
    selected = selected_arms(['T08_refiner_no_gray'])
    assert [a['id'] for a in selected] == ['R00_seed32','T00_control','T01_native_geometry','T02_native_loss','T07_native_refiner','T08_refiner_no_gray']
    validate_plan(selected)


def test_metric_users_exact_absent_class_example():
    target = np.repeat(np.array([0,1,3], np.uint8), 90).reshape(9,30)
    pred = target.ravel().copy()
    for k, c in enumerate((1,3,0)):
        pred[k*90+80:(k+1)*90] = c
    row = metrics(target, pred.reshape(target.shape))
    assert row['class_iou'] == pytest.approx([.8,.8,0,.8])
    assert row['image_miou'] == pytest.approx(.6)
    assert summarize([dict(row,image_id=str(i)) for i in range(20)])['image_miou'] == pytest.approx(.6)
    perfect = metrics(target, target)
    assert perfect['image_miou'] == .75  # absent Primary Si is still zero


def test_palette_strict_and_boundary_excludes_image_border():
    ids = np.tile(np.arange(4, dtype=np.uint8), (10,4))
    np.testing.assert_array_equal(mask_ids(PALETTE[ids]), ids)
    with pytest.raises(ValueError, match='Unknown'):
        mask_ids(np.zeros((2,2,3), np.uint8))
    assert not boundary_band(np.zeros((8,8), np.uint8), 2).any()
    ids = np.zeros((8,8),np.uint8)
    ids[:,4:] = 1
    assert boundary_band(ids,0)[:,3:5].all()
    assert not boundary_band(ids,0)[:,[0,7]].any()


def test_geometry_identity_and_label_alignment():
    gray = np.arange(400,dtype=np.uint8).reshape(20,20)
    label = (gray%4).astype(np.uint8)
    g, y = native_roi(gray,label,(9.5,9.5),8)
    np.testing.assert_array_equal(g,gray[6:14,6:14])
    np.testing.assert_array_equal(y,label[6:14,6:14])
    for k in range(8):
        x = torch.arange(24).reshape(1,1,4,6)
        assert torch.equal(inverse_d4(d4(x,k),k),x)


def fixture_rows(tmp_path):
    h,w=96,128
    yy,xx=np.indices((h,w))
    gray=((xx+yy)*1.5).clip(0,255).astype(np.uint8)
    target=((xx//20+yy//24)%4).astype(np.uint8)
    image=tmp_path/'image.png'; mask=tmp_path/'mask.png'
    Image.fromarray(gray).save(image); Image.fromarray(PALETTE[target]).save(mask)
    return [dict(image_id='train_only',split='train',image=str(image),mask=str(mask))]


def test_stateless_sampling_native_pair_and_train_only(tmp_path):
    rows=fixture_rows(tmp_path)
    arms={a['id']:a for a in make_plan(crop=64,steps=2)}
    cfg=deepcopy(arms['T01_native_geometry']['config']);cfg['target_width']=256
    ds=TrainDataset(rows,cfg)
    a=ds[7];b=ds[7]
    for key in ('detail','target','native_gray','native_target'):
        assert torch.equal(a[key],b[key])
    expected=torch.nn.functional.interpolate(a['native_target'][None,None].float(),(64,64),mode='nearest-exact')[0,0].long()
    assert torch.equal(a['target'],expected)
    assert one_sample_collate([a])['detail'].shape==(1,3,64,64)
    invalid=[dict(rows[0],split='valid')]
    with pytest.raises(ValueError,match='ONLY train'):
        TrainDataset(invalid,cfg)
    all_indices=[list(range(rank,16,8)) for rank in range(8)]
    assert sorted(i for rank in all_indices for i in rank)==list(range(16))
    assert list(range(8+3,16,8))==all_indices[3][1:]
    assert indexed_seed(32,7,'sample')==indexed_seed(32,7,'sample')


def test_no_gaussian_keeps_target_geometry_and_other_photometry(tmp_path):
    rows=fixture_rows(tmp_path)
    arms={a['id']:a for a in make_plan(crop=64,steps=4)}
    c=deepcopy(arms['T00_control']['config']);c['target_width']=256
    n=deepcopy(c);n['gaussian_blur']=False
    base,off=TrainDataset(rows,c),TrainDataset(rows,n)
    changed=0
    for i in range(32):
        a,b=base[i],off[i]
        assert torch.equal(a['target'],b['target'])
        changed += not torch.equal(a['detail'],b['detail'])
    assert 0 < changed < 32


def test_area_restore_and_native_loss_have_gradients():
    logits=torch.randn(1,4,16,16,requires_grad=True)
    prob=restore_probs(logits.softmax(1),(5,7),'area')
    assert torch.allclose(prob.sum(1),torch.ones(1,5,7),atol=1e-6)
    target=torch.randint(0,4,(1,5,7))
    native_loss(prob,target).backward()
    assert logits.grad.isfinite().all() and logits.grad.abs().sum()>0
    other=restore_probs(logits.detach().softmax(1),(5,7),'bilinear')
    assert not torch.allclose(prob,other)


def test_boundary_weight_changes_only_weighted_ce():
    x=torch.randn(1,4,8,8,requires_grad=True)
    y=torch.zeros((1,8,8),dtype=torch.long);y[:,:,4:]=1
    band=torch.zeros_like(y,dtype=torch.float32);band[:,:,3:5]=1
    baseline=ce_dice(x,y,band,0)
    weighted=ce_dice(x,y,band,1)
    assert weighted != baseline
    assert torch.allclose(ce_dice(x,y,torch.zeros_like(band),1),baseline)


def test_fresh_b4_decoder_and_zero_detail_branch():
    arms={a['id']:a for a in make_plan(crop=64)}
    torch.manual_seed(32)
    base=BoundaryModel(arms['T00_control']['config'],encoder_weights=ENCODER_WEIGHTS).eval()
    torch.manual_seed(32)
    variant=BoundaryModel(arms['T05_detail_branch']['config'],encoder_weights=ENCODER_WEIGHTS).eval()
    for k,v in base.base.state_dict().items():
        assert torch.equal(v,variant.base.state_dict()[k]),k
    original=torch.load(CHECKPOINT,map_location='cpu',weights_only=True)
    assert not torch.equal(base.base.segmentation_head[0].weight,original['segmentation_head.0.weight'])
    x=torch.randn(1,3,64,64)
    with torch.no_grad():
        assert torch.equal(base.detail_logits(x),variant.detail_logits(x))


def test_native_refiner_gray_ablation_and_gradient():
    torch.manual_seed(32);a=NativeRefiner(True)
    torch.manual_seed(32);b=NativeRefiner(False)
    p=torch.rand(1,4,16,16);p/=p.sum(1,keepdim=True)
    gray=torch.randn(1,1,16,16)
    assert torch.allclose(a(gray,p),p,atol=1e-7)
    assert sum(x.numel() for x in a.parameters())==sum(x.numel() for x in b.parameters())
    with torch.no_grad():
        a.head.weight.fill_(.01)
        # Different outputs by class are needed to survive softmax invariance.
        a.head.weight[0].mul_(2)
        b.load_state_dict(a.state_dict())
    assert torch.equal(b(gray,p),b(gray+3,p))
    assert not torch.allclose(a(gray,p),a(gray+3,p))
    native_loss(a(gray,p),torch.randint(0,4,(1,16,16))).backward()
    assert a.features[0][0].weight.grad.abs().sum()>0


def test_inference_matches_archived_code_on_cpu():
    source=ROOT/'vendor/original_common.py'
    legacy=types.ModuleType('legacy_test')
    exec(compile(source.read_text(),str(source),'exec'),legacy.__dict__)
    class Toy(torch.nn.Module):
        native_refiner=None
        def forward(self,x):
            g=x[:,:1]
            return torch.cat([g,-g,g*.5,g.square()*.2],1)
        detail_logits=forward
    model=Toy().eval()
    gray=np.arange(80*64,dtype=np.uint8).reshape(64,80)
    cfg=make_plan()[0]['config'].copy();cfg['target_width']=160
    ours=predict(model,gray,cfg,'cpu')
    mem=legacy.make_member(model,'imagenet',target_width=160)
    theirs=legacy.predict_probs([mem],gray,tta=True,device='cpu')
    assert torch.equal(ours,theirs)


def test_panel_pixels_layout_and_immutable_checkpoint(tmp_path):
    target=np.zeros((20,40),np.uint8);target[:,20:]=2
    pred=target.copy();pred[4,25]=0
    original=np.full((20,40,3),33,np.uint8)
    score=metrics(target,pred)['image_miou']
    panel=np.array(comparison_panel(original,target,pred,score,'fixture'))
    # Header108 + panel caption44; top-right origin x=width+gap.
    np.testing.assert_array_equal(panel[108+44+4,40+24+25],PALETTE[2])
    np.testing.assert_array_equal(panel[108+44+3,40+24+25],[255,255,255])
    np.testing.assert_array_equal(panel[108+44,0],original[0,0])
    torch_path=tmp_path/'checkpoint.pt'
    immutable_torch_save({'value':torch.tensor([1])},torch_path)
    before=sha256(torch_path)
    with pytest.raises(FileExistsError):
        immutable_torch_save({'value':torch.tensor([2])},torch_path)
    assert sha256(torch_path)==before


def test_result_contract_and_tamper_detection(tmp_path):
    result=reserve_result(tmp_path)
    second=reserve_result(tmp_path)
    assert result!=second
    for name in ('predictions','valid_mask_comparison'):
        (result/name).mkdir()
    rows=[]
    for i in range(20):
        name=str(i)
        target=np.tile(np.arange(4,dtype=np.uint8),(8,2))
        rows.append(dict(image_id=name,**metrics(target,target)))
        for folder in ('predictions','valid_mask_comparison'):
            Image.fromarray(PALETTE[target]).save(result/folder/f'{name}.png')
    finish_report(result,make_plan()[0],rows,{'weights_sha256':'fixture'})
    write_json(result/'complete.json',{'files':result_manifest(result)})
    assert verify_result(result,[str(i) for i in range(20)])['image_miou']==1
    (result/'per_image.csv').write_text('tampered')
    with pytest.raises(ValueError,match='changed'):
        verify_result(result)
