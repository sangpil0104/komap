import argparse
from pathlib import Path
import pandas as pd
from omegaconf import OmegaConf

from alphase.cli.common import load_trained, prepare_output
from alphase.evaluation.evaluate import evaluate
from alphase.utils import sha256, write_json

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="valid", choices=["valid", "heldout"])
    parser.add_argument("--output")
    parser.add_argument("--weights", choices=["selected", "raw", "ema"], default="selected")
    parser.add_argument("--save-probabilities", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--set", nargs="*", default=[])
    args = parser.parse_args()
    model, cfg, device, state, selected = load_trained(args.checkpoint, args.set, args.weights)
    table = pd.read_csv(cfg["data"]["manifest"]).fillna("")
    rows = table[table.source_split == "valid"] if args.split == "valid" else table[table.image_id.isin(state["validation_ids"])]
    if set(rows.image_id) & set(state["train_ids"]):
        raise ValueError("Requested validation images were used for training; independent validation is not possible")
    out = prepare_output(args.output or str(Path(args.checkpoint).parent/"evaluation"), args.overwrite)
    OmegaConf.save(OmegaConf.create(cfg), out/"resolved_config.yaml")
    summary = evaluate(model, rows.to_dict("records"), cfg, device, out, args.save_probabilities)
    write_json(out/"provenance.json", dict(checkpoint_sha256=sha256(args.checkpoint), selected_weights=selected, train_ids=state["train_ids"]))
    print(summary)
