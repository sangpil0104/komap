from pathlib import Path
import torch
from omegaconf import OmegaConf

from alphase.models.factory import build_model


def load_trained(checkpoint, overrides=(), weights="selected"):
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg = OmegaConf.to_container(OmegaConf.merge(OmegaConf.create(state["resolved_config"]), OmegaConf.from_dotlist(list(overrides))), resolve=True)
    torch.set_num_threads(cfg["train"].get("cpu_threads", 4))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg, load_pretrained=False)
    selected = state["selected_weights"] if weights == "selected" else weights
    model.load_state_dict(state["ema_model"] if selected == "ema" else state["model"])
    return model.to(device).eval(), cfg, device, state, selected


def prepare_output(output, overwrite=False):
    output = Path(output)
    if output.exists() and any(output.iterdir()) and not overwrite:
        raise FileExistsError(f"Output already exists: {output}; use a new path or --overwrite")
    if output.exists() and any(output.iterdir()) and overwrite:
        import time
        output.rename(output.with_name(output.name+f".backup-{time.time_ns()}"))
    output.mkdir(parents=True, exist_ok=True)
    return output
