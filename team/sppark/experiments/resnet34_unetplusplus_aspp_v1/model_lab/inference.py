from __future__ import annotations

from pathlib import Path
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from model_lab.dataset import KomapEvalDataset, save_mask
from model_lab.evaluation import predict_full_frame
from model_lab.workspace import get_dataset_samples, OUTPUTS_DIR


def run_inference_on_test(
    model: torch.nn.Module,
    output_dir: str | Path | None = None,
    d4: bool = True,
    device: torch.device = torch.device("cuda:0"),
) -> list[str]:
    """Generates 10 test predictions as RGBA PNGs matching the competition color palette."""
    if output_dir is None:
        output_dir = OUTPUTS_DIR / "predictions"
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    samples = get_dataset_samples("test")
    ds = KomapEvalDataset(samples, has_masks=False)
    loader = DataLoader(ds, batch_size=1, shuffle=False)

    saved_paths = []
    pbar = tqdm(loader, desc=f"Predicting Test Masks ({'D4' if d4 else 'Plain'})", unit="image")
    for batch in pbar:
        img = batch["image"][0]
        h = int(batch["original_h"][0])
        w = int(batch["original_w"][0])
        image_id = batch["image_id"][0]

        pred_labels = predict_full_frame(model, img, h, w, d4=d4, device=device)
        mask_file = out_path / f"{image_id}_mask.png"
        save_mask(pred_labels, mask_file, rgba=True)
        saved_paths.append(str(mask_file))

    print(f"Successfully generated {len(saved_paths)} test masks in {out_path}")
    return saved_paths

