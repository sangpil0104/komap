from pathlib import Path

import torch

from alphase.models.cnn import PhaseCNN
from alphase.models.hybrid import PhaseHybrid


def build_model(cfg, load_pretrained=True):
    model = cfg["model"]
    channels = len(cfg["data"]["image_channels"])
    if model["name"] == "cnn_unet":
        result = PhaseCNN(channels)
    elif model["name"] == "alphase_hybrid_v1":
        repo = Path(model["dino_repo"]).resolve()
        weight = model.get("pretrained_checkpoint")
        if not (repo / "hubconf.py").is_file():
            raise FileNotFoundError(f"DINOv3 repository missing: {repo}")
        if load_pretrained and (not weight or not Path(weight).is_file()):
            raise FileNotFoundError("DINOv3 official checkpoint is required. Set model.pretrained_checkpoint=/path/to/original_filename.pth. Use configs/cnn_unet.yaml to train without it.")
        kwargs = dict(pretrained=load_pretrained)
        # Keep filename-based architecture selection when restoring a trained ViT-L.
        # pretrained=False reconstructs the architecture without reading this old file.
        if weight:
            kwargs["weights"] = str(Path(weight).resolve())
        backbone = torch.hub.load(str(repo), model["backbone_name"], source="local", **kwargs)
        if backbone.embed_dim != model["dino_embed_dim"]:
            raise ValueError("Unexpected DINO embedding dimension")
        result = PhaseHybrid(backbone, model["dino_layers"], channels, model["context_channels"])
    else:
        raise ValueError(f"Unknown model: {model['name']}")
    # Disabled auxiliary heads must not create unused DDP parameters.
    if not cfg["loss"]["auxiliary"]:
        result.aux_head.requires_grad_(False)
    if not cfg["loss"]["boundary"]:
        result.boundary_head.requires_grad_(False)
    return result
