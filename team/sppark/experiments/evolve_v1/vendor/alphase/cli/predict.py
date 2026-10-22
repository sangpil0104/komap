import argparse
from pathlib import Path
import numpy as np
from PIL import Image
from omegaconf import OmegaConf
from tqdm import tqdm

from alphase.cli.common import load_trained, prepare_output
from alphase.data.palette import save_mask
from alphase.inference.predictor import predict_probabilities
from alphase.utils import sha256, write_json

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", default="dataset/test/images")
    parser.add_argument("--output", default="submission/predictions")
    parser.add_argument("--weights", choices=["selected", "raw", "ema"], default="selected")
    parser.add_argument("--rgb", action="store_true")
    parser.add_argument("--save-probabilities", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--set", nargs="*", default=[])
    args = parser.parse_args()
    model, cfg, device, state, selected = load_trained(args.checkpoint, args.set, args.weights)
    files = sorted(Path(args.input).glob("*_image.png"))
    if not files:
        raise ValueError("No *_image.png inputs")
    out = prepare_output(args.output, args.overwrite)
    OmegaConf.save(OmegaConf.create(cfg), out/"resolved_config.yaml")
    records = []
    for path in tqdm(files, desc="Predict"):
        with Image.open(path) as image:
            gray = np.asarray(image.convert("L"), dtype=np.float32)/255
        prob = predict_probabilities(model, gray, cfg, device)
        output = out/path.name.replace("_image.png", "_mask.png")
        save_mask(prob.argmax(0), output, rgba=not args.rgb)
        if args.save_probabilities:
            folder = out/"probabilities"
            folder.mkdir(exist_ok=True)
            np.save(folder/(path.stem+".npy"), prob)
        records.append(dict(image_id=path.stem.removesuffix("_image"), width=gray.shape[1], height=gray.shape[0], output=output.name, sha256=sha256(output)))
    write_json(out/"inference_manifest.json", dict(checkpoint_sha256=sha256(args.checkpoint), weights=selected, train_ids=state["train_ids"], inference=cfg["inference"], images=records))
    print(f"Saved {len(records)} masks to {out}")
