"""Fixed supervised restart61 continuation; no teacher, validation selection or release replacement."""
import os,sys,json,time,hashlib,pickle,traceback,subprocess,random,copy
from pathlib import Path
from functools import lru_cache
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import yaml
K=Path.home()/'komap';O=K/'codex_restart61_globalstats_20261012';O.mkdir(exist_ok=True)
B=K/'Model349_lite20_b4ws_bundle';P=B/'Model349_fusion805';X=P/'Model349_exh14'
sys.path.insert(0,str(X));sys.path.insert(0,str(X/'src'))
from encoder_structure_search import build_model
from model181_raw_zcdf_aspp import representation
from model253_highres_scale_finetune import resize_pair
from model1_unet_resnet34 import decode_mask
from advanced_inference import blend_window,views,transform_view,inverse_view
from style_tta_search import transform_image
from hard_source_search import route_params
from prior_representation_search import percentile_image
from infer_lite import assemble
SRC=K/'weights/all/R/restart61_pairwise_margin/best.pt';T0=time.time()
def dump(path,obj):path.write_text(json.dumps(obj,indent=2,ensure_ascii=False)+'\n')
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for c in iter(lambda:f.read(8*1024*1024),b''):h.update(c)
    return h.hexdigest()
def load(path,device):
    m=build_model('aspp',pretrained=False,checkpoint=None)
    state=torch.load(path,map_location='cpu',weights_only=False)['model'];m.load_state_dict(state,strict=True)
    return m.to(device)
def image_pairs():
    out=[]
    for p in sorted((K/'data/train/images').glob('*_image.png')):
        mask=K/'data/train/masks'/p.name.replace('_image.png','_mask.png');assert mask.exists();out.append((p,mask))
    assert len(out)==70;return out
class Patches(torch.utils.data.Dataset):
    def __init__(self,pairs):self.pairs=pairs
    def __len__(self):return len(self.pairs)*8
    @lru_cache(maxsize=140)
    def get_full(self,i,scale):
        im,mp=self.pairs[i];raw=np.asarray(Image.open(im).convert('RGB'));target=decode_mask(Image.open(mp).convert('RGB'))
        if scale==2:
            raw=np.asarray(Image.fromarray(raw).resize((raw.shape[1]*2,raw.shape[0]*2),Image.Resampling.BICUBIC));target=np.asarray(Image.fromarray(target.astype(np.uint8)).resize((target.shape[1]*2,target.shape[0]*2),Image.Resampling.NEAREST)).astype(np.int64)
        return representation(raw),target
    def __getitem__(self,index):
        # Four patches per image at each scale; complete-image stats computed before cropping.
        i=index%70;scale=1 if (index//70)%2==0 else 2;patch=512 if scale==1 else 768
        feat,mask=self.get_full(i,scale);h,w=mask.shape
        ph,pw=max(0,patch-h),max(0,patch-w)
        if ph or pw:feat=np.pad(feat,((0,0),(0,ph),(0,pw)),mode='reflect');mask=np.pad(mask,((0,ph),(0,pw)),mode='reflect')
        h,w=mask.shape
        # Class-focused crops retain the original family's rare-phase emphasis.
        if random.random()<.9:
            cls=random.choices([0,1,3],weights=[.35,.25,.40])[0];ys,xs=np.where(mask==cls)
        else:ys,xs=np.array([],dtype=int),np.array([],dtype=int)
        if len(xs):j=random.randrange(len(xs));cy,cx=int(ys[j]),int(xs[j])
        else:cy,cx=random.randrange(h),random.randrange(w)
        y=min(max(cy-patch//2,0),h-patch);x=min(max(cx-patch//2,0),w-patch)
        feat=feat[:,y:y+patch,x:x+patch].copy();mask=mask[y:y+patch,x:x+patch].copy()
        if random.random()<.5:feat=np.flip(feat,2);mask=np.flip(mask,1)
        if random.random()<.5:feat=np.flip(feat,1);mask=np.flip(mask,0)
        turns=random.randrange(4);feat=np.rot90(feat,turns,(1,2));mask=np.rot90(mask,turns)
        return torch.from_numpy(feat.copy()),torch.from_numpy(mask.copy()).long()
def loss(logits,target):
    p=logits.softmax(1);oh=F.one_hot(target,4).permute(0,3,1,2).float()
    dice=1-((2*(p*oh).sum((0,2,3))+1)/(p.sum((0,2,3))+oh.sum((0,2,3))+1)).mean()
    ce=F.cross_entropy(logits,target,weight=logits.new_tensor([1.15,1.45,.70,2.0]))
    pair=(target==0)|(target==3);other=torch.where(target==0,3,0)
    margin=F.relu(.20-(logits.gather(1,target[:,None]).squeeze(1)-logits.gather(1,other[:,None]).squeeze(1)))
    return ce+dice+.10*(margin[pair].mean() if pair.any() else logits.sum()*0)
def train(pairs):
    dest=O/'last.pt'
    if (O/'training_complete.json').exists():return dest
    if dest.exists():raise RuntimeError('unverified final checkpoint exists')
    random.seed(819269);np.random.seed(819269);torch.manual_seed(819269);torch.cuda.manual_seed_all(819269)
    m=load(SRC,'cuda');enc=[];rest=[]
    for n,p in m.named_parameters():(enc if n.startswith(('stem.','enc')) else rest).append(p)
    opt=torch.optim.AdamW([{'params':enc,'lr':6e-6},{'params':rest,'lr':3e-5}],weight_decay=1e-4)
    scaler=torch.amp.GradScaler('cuda',init_scale=1024)
    dl=torch.utils.data.DataLoader(Patches(pairs),batch_size=1,shuffle=True,num_workers=0)
    history=[]
    for epoch in range(1,11):
        m.train()
        for mod in m.modules():
            if isinstance(mod,torch.nn.modules.batchnorm._BatchNorm):mod.eval()
        opt.zero_grad(set_to_none=True);total=0.;start=time.time()
        for j,(a,t) in enumerate(dl):
            a,t=a.cuda(),t.cuda()
            with torch.autocast('cuda',dtype=torch.float16):v=loss(m(a),t)
            if not torch.isfinite(v):raise RuntimeError('nonfinite train loss')
            scaler.scale(v/2).backward();total+=float(v.detach())
            if (j+1)%2==0:
                scaler.unscale_(opt);torch.nn.utils.clip_grad_norm_(m.parameters(),1.);scaler.step(opt);scaler.update();opt.zero_grad(set_to_none=True)
            if (j+1)%70==0:print(f'epoch {epoch}/10 patches {j+1}/560 loss {total/(j+1):.5f} elapsed {time.time()-start:.1f}s',flush=True)
        history.append({'epoch':epoch,'loss':total/len(dl),'seconds':time.time()-start});dump(O/'history.json',history)
        checkpoints=O/'checkpoints';checkpoints.mkdir(exist_ok=True)
        epoch_path=checkpoints/f'epoch{epoch:02d}.pt'
        torch.save({'model':{n:v.detach().cpu() for n,v in m.state_dict().items()},'epoch':epoch,'train_loss':total/len(dl),'recipe':'fixed 10epoch global raw/Z/CDF multiscale continuation; validation used only after training for checkpoint selection'},epoch_path)
        history[-1]['checkpoint_sha256']=sha(epoch_path);dump(O/'history.json',history)
    torch.save({'model':m.cpu().state_dict(),'epoch':10,'recipe':'global raw/Z/CDF, scale1 patch512 + scale2 patch768, train70 only'},dest)
    dump(O/'training_complete.json',{'sha256':sha(dest),'epoch':10,'time':time.time()});del m;torch.cuda.empty_cache();return dest
def full_features(raw):
    scaled,_=resize_pair(raw,np.zeros(raw.shape[:2],dtype=np.int64),1.,768)
    return representation(scaled)
def starts(size):
    a=list(range(0,max(size-768,0)+1,576));last=max(size-768,0)
    if a[-1]!=last:a.append(last)
    return a
def adapt(path,raw):
    m=load(path,'cuda');m.eval();feat=full_features(raw);ys,xs=starts(feat.shape[1]),starts(feat.shape[2])
    loc=list(dict.fromkeys([(ys[0],xs[0]),(ys[0],xs[-1]),(ys[-1],xs[0]),(ys[-1],xs[-1])]))
    inp=torch.stack([torch.from_numpy(feat[:,y:y+768,x:x+768].copy()) for y,x in loc]).cuda()
    params=[]
    for mod in m.modules():
        if isinstance(mod,torch.nn.modules.batchnorm._BatchNorm):
            mod.train()
            if mod.weight is not None:params.append(mod.weight)
            if mod.bias is not None:params.append(mod.bias)
    m.context.pool[2].eval();ids={id(p) for p in params}
    for p in m.parameters():p.requires_grad_(id(p) in ids)
    opt=torch.optim.Adam(params,lr=1e-5)
    for _ in range(5):
        opt.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.float16):
            q=m(inp).softmax(1);entropy=-(q*q.clamp_min(1e-7).log()).sum(1).mean()
        if not torch.isfinite(entropy):raise RuntimeError('nonfinite TENT entropy')
        entropy.backward();opt.step()
    m.eval();return m
@torch.no_grad()
def predict(m,raw):
    feat=full_features(raw);h,w=feat.shape[1:];win=blend_window(768,'gaussian');out=np.zeros((4,h,w),np.float32);ws=np.zeros((h,w),np.float32)
    for y in starts(h):
        for x in starts(w):
            a=torch.from_numpy(feat[:,y:y+768,x:x+768].copy())[None].cuda();p=None
            for r,f in views('d4'):
                with torch.autocast('cuda',dtype=torch.float16):v=m(transform_view(a,r,f)).softmax(1)
                v=inverse_view(v,r,f).float();p=v if p is None else p+v
            p=(p/8).cpu().numpy()[0];out[:,y:y+768,x:x+768]+=p*win;ws[y:y+768,x:x+768]+=win
    out/=np.maximum(ws[None],1e-7)
    out=np.stack([np.asarray(Image.fromarray(a).resize((raw.shape[1],raw.shape[0]),Image.Resampling.BILINEAR)) for a in out]);assert np.isfinite(out).all()
    return out/np.maximum(out.sum(0,keepdims=True),1e-7)
def score(work,name):
    dest=O/name/'fusion';dest.parent.mkdir(exist_ok=True)
    if not (dest/'completion.json').exists():
        with (O/name/'fusion.log').open('w') as f:
            subprocess.run([sys.executable,P/'infer_fusion.py','--input',K/'data/valid/images','--release-labels',work/'predicted_labels','--expert-probabilities',K/'l18h/e2e_valid/expert','--refiner',K/'weights/all/assets/refiner/epoch8.pt','--output',dest,'--device','cpu'],stdout=f,stderr=subprocess.STDOUT,check=True)
    mp=O/name/'metrics.json'
    with (O/name/'score.log').open('w') as f:
        subprocess.run(['bash',B/'score.sh',dest/'predicted_labels',K/'data/valid/masks',mp],env=dict(os.environ,PYTHON=sys.executable),stdout=f,stderr=subprocess.STDOUT,check=True)
    return json.loads(mp.read_text())
def main():
    if '--check' in sys.argv:
        pairs=image_pairs();m=load(SRC,'cpu');x,t=Patches(pairs)[0];assert x.shape[0]==3 and t.min()>=0 and t.max()<4
        assert np.isfinite(x.numpy()).all();print('CHECK_OK: strict parent load, 70 train pairs, global feature/label shapes',x.shape,t.shape);return
    if (O/'epoch_selection_completion.json').exists():return
    if (O/'completion.json').exists() and '--select-epochs' not in sys.argv:return
    while subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():print('waiting for compute GPU',flush=True);time.sleep(20)
    dump(O/'active_child.json',{'pid':os.getpid(),'stage':'training','time':time.time()})
    pairs=image_pairs();inputs={str(p):sha(p) for pair in pairs for p in pair};inputs[str(SRC)]=sha(SRC)
    dump(O/'input_sha.json',inputs);path=train(pairs)
    orig=K/'l18h/e2e_valid/release';manifest=json.loads((orig/'input_manifest.json').read_text());cfg=yaml.safe_load((B/'Model349_fusion805_lite23_b4ws/selected_R_lite18h.yaml').read_text())
    cal=json.loads((X/cfg['calibration']).read_text())
    result={'lite18h':json.loads((K/'l18h/e2e_valid_metrics.json').read_text())}
    def evaluate(name,base_path,remove):
        work=O/name/'R';work.mkdir(parents=True,exist_ok=True);(work/'probabilities').mkdir(exist_ok=True)
        c=copy.deepcopy(cfg)
        if remove:c['inference']['active_components'].remove('269');c['route']['base']['eutectic_gamma']=0.
        data=copy.deepcopy(manifest);data['config']=c;dump(work/'input_manifest.json',data)
        for component in c['inference']['active_components']:
            target=base_path if component=='base' else orig/'probabilities'/component
            link=work/'probabilities'/component
            if not link.exists():link.symlink_to(target,target_is_directory=True)
        if not (work/'metrics.json').exists():assemble(work,K/'data/valid/masks')
        return score(work,name)
    result['no269']=evaluate('no269',orig/'probabilities/base',True)
    epoch_scores=[];best_epoch=None;best_value=float('-inf')
    for epoch in range(1,11):
        path=O/'checkpoints'/f'epoch{epoch:02d}.pt';assert path.exists()
        candidate_base=O/f'epoch{epoch:02d}_base'
        if epoch==10 and not candidate_base.exists() and (O/'candidate_base').exists():candidate_base.symlink_to(O/'candidate_base',target_is_directory=True)
        candidate_base.mkdir(exist_ok=True)
        dump(O/'active_child.json',{'pid':os.getpid(),'stage':'epoch checkpoint TENT+D4 valid inference','epoch':epoch,'time':time.time()})
        for row in manifest['images']:
            name=row['name'];out=candidate_base/(name+'.pkl')
            if out.exists():continue
            raw=np.asarray(Image.open(row['path']).convert('RGB'));m=adapt(path,raw)
            style=transform_image(raw,'standardize',cal['train_brightness_mean'],cal['train_brightness_std'])
            high=[predict(m,im) for im in (raw,style)];gray=raw[...,0].astype(np.float32);params=dict(route_params(float(gray.mean()),float(gray.std())),pair_weight=0.)
            q=((1-params['style_weight'])*high[0]+params['style_weight']*high[1]).astype(np.float32);q/=np.maximum(q.sum(0,keepdims=True),1e-7)
            item={'preaux':q,'z':(gray-gray.mean())/max(float(gray.std()),1e-5),'q':percentile_image(gray),'prior_z_stats':cal['priors']['z'],'prior_q_stats':cal['priors']['q'],'params':params}
            with out.open('wb') as f:pickle.dump(item,f,pickle.HIGHEST_PROTOCOL)
            del m;torch.cuda.empty_cache();print('epoch',epoch,'candidate inferred',name,flush=True)
        name=f'epoch{epoch:02d}_no269';metrics=evaluate(name,candidate_base,True);result[name]=metrics
        value=metrics['image_miou']*100;improved=value>best_value+1e-12
        if improved:best_value=value;best_epoch=epoch
        epoch_scores.append({'epoch':epoch,'final_valid20_percent':value,'new_best':improved,'checkpoint_sha256':sha(path)})
        dump(O/'epoch_scores.json',{'rows':epoch_scores,'last_record_improvement_epoch':best_epoch,'valid_selected':True})
        print(f'EPOCH_SCORE {epoch} final={value:.8f} new_best={improved}',flush=True)
    # Secondary diagnostic for the selected epoch only; it does not select the epoch.
    result['candidate_no269']=result[f'epoch{best_epoch:02d}_no269']
    result['candidate_keep269']=evaluate('selected_keep269',O/f'epoch{best_epoch:02d}_base',False)
    import shutil
    shutil.copy2(O/'checkpoints'/f'epoch{best_epoch:02d}.pt',O/'selected_candidate.pt')
    for p,s in inputs.items():assert sha(Path(p))==s,'Input changed '+p
    summary={'scores_percent':{n:v['image_miou']*100 for n,v in result.items()},'best_epoch':best_epoch,'selection_rule':'last strict record improvement among epochs1-10 by fixed R+E+F valid20 score, model269 removed','valid_selected':True,'beats_lite18h':best_value>result['lite18h']['image_miou']*100,'input_sha_unchanged':True,'release_replaced':False,'valid_used_for_training':False,'elapsed_seconds':time.time()-T0}
    for ref in ('lite18h','no269'):
        a={x['image']:x['image_miou'] for x in result['candidate_no269']['images']};b={x['image']:x['image_miou'] for x in result[ref]['images']};d={n:100*(a[n]-b[n]) for n in a}
        summary['candidate_no269_vs_'+ref]={'delta_pp':sum(d.values())/len(d),'per_image_delta_pp':d,'worst_image':min(d,key=d.get),'improved':sum(v>0 for v in d.values()),'worse':sum(v<0 for v in d.values())}
    dump(O/'result.json',summary);dump(O/'completion.json',{'time':time.time(),'result':'result.json'});dump(O/'epoch_selection_completion.json',{'time':time.time(),'best_epoch':best_epoch,'valid_selected':True});print(summary['scores_percent'],flush=True)
try:main()
except Exception:
    dump(O/'failure.json',{'time':time.time(),'error':traceback.format_exc()});raise
