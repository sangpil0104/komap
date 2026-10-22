"""One validation preview from a preserved intermediate checkpoint; no training."""
from pathlib import Path
import json
import sys

import numpy as np
from PIL import Image
import torch
from tqdm import tqdm

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
sys.path.insert(0, str(ROOT))
from runtime import comparison, image_metrics, save_json, sha256, strict_mask
from komap_seg.dataset import KomapFullDataset
from komap_seg.inference import predict_probs, _TTA_OPS
from komap_seg.model import build_model
from komap_seg.palette import PALETTE


def main():
    torch.set_num_threads(1)
    checkpoint = OUT / 'checkpoint_snapshot.pt'
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    cfg = json.loads((OUT / 'training_config.json').read_text())
    epoch, updates = state['epoch'], state['optimizer_updates']
    model = build_model(**state['model_config'], encoder_weights=None)
    model.load_state_dict(state['state_dict'], strict=True)
    model.to('cuda:0').eval()
    data = Path(cfg['data_root'])
    ds = KomapFullDataset(data/'valid/images', data/'valid/masks')
    image_id = 'A11_step1_sample4'
    index = next(i for i, pair in enumerate(ds.pairs) if pair['stem'] == image_id)
    sample = ds[index]
    target = strict_mask(data/'valid/masks'/f'{image_id}_mask.png')
    total = None
    with torch.inference_mode():
        for forward, inverse in tqdm(_TTA_OPS, desc=f'epoch {epoch} preview TTA', unit='flip'):
            prob = predict_probs(model, forward(sample['image']), patch_size=cfg['patch_size'], overlap=cfg['overlap'], input_scale=cfg['input_scale'], batch_size=1, device=torch.device('cuda:0'))
            prob = inverse(prob)
            total = prob if total is None else total + prob
    pred = (total / len(_TTA_OPS)).argmax(0).numpy().astype(np.uint8)
    metrics = image_metrics(pred, target)
    mask_path = OUT/f'{image_id}_predicted_mask_epoch{epoch}.png'
    comparison_path = OUT/f'{image_id}_comparison_epoch{epoch}.png'
    original = np.array(Image.open(data/'valid/images'/f'{image_id}_image.png').convert('RGB'))
    Image.fromarray(PALETTE[pred]).save(mask_path)
    comparison(comparison_path, original, pred, target, metrics['image_miou'])
    assert np.array_equal(strict_mask(mask_path), pred)
    report = {'model_name':'HRNet_taeho','preview_kind':'intermediate_checkpoint_single_image', 'epoch':epoch, 'planned_epochs':cfg['epochs'], 'checkpoint_optimizer_updates':updates, 'preview_optimizer_updates':0, 'image_id':image_id,'height':pred.shape[0],'width':pred.shape[1], 'checkpoint_sha256':sha256(checkpoint),'physical_gpu':5,'inference_batch_size':1, 'tta':'four_flip', 'patch_size':cfg['patch_size'],'input_scale':cfg['input_scale'],'overlap':cfg['overlap'], 'predicted_mask':str(mask_path),'comparison':str(comparison_path),**metrics}
    save_json(OUT/'preview.json',report)
    (OUT/'README.md').write_text(f'# HRNet_taeho 중간 예측 미리보기\n\n- 학습 {epoch}/{cfg["epochs"]} epoch 체크포인트, {updates} optimizer updates.\n- 추가 학습 없이 추론만 실행. 최종 학습 결과가 아닙니다.\n- 검증 이미지: {image_id}, {pred.shape[0]}×{pred.shape[1]}.\n- 이 이미지의 image mIoU: {metrics["image_miou"]*100:.4f}%. 20장 평균이 아닙니다.\n- [예측 마스크]({mask_path.name})\n- [원본·오류·예측·정답 비교]({comparison_path.name})\n')
    print(json.dumps(report,indent=2),flush=True)


if __name__ == '__main__':
    main()
