"""Integration diagnostics; default uses synthetic data and no optimizer updates."""

from copy import deepcopy
import torch
from .models import load_model, NativeRefiner
from .geometry import prepare_inputs
from .losses import losses
from .io import ROOT, json_write
from .data import model_inputs


def smoke(
    cfg,
    steps=4,
    device="cuda:0",
    full_size=False,
    real_train_patches=False,
    execute=False,
):
    if real_train_patches and not execute:
        print(
            "DRY RUN: real train-patch overfit smoke requires --execute (user starts it)."
        )
        return
    c = deepcopy(cfg)
    if not full_size:
        c["geometry"]["detail_input_size"] = 128
        c["geometry"]["context_input_size"] = 64
    p = c["geometry"]["detail_input_size"]
    cp = c["training"]["initialize_from"] or c["paths"]["legacy_checkpoint"]
    torch.set_num_threads(1)
    torch.manual_seed(0)
    model = load_model(c, cp, device).train()
    model.set_base_frozen(False)
    gray = torch.rand(1, 1, 81, 113, device=device)
    inputs = prepare_inputs(gray, (8, 10, 49, 49), c)
    count = c["training"]["batch_size"]
    inputs = {
        key: (
            value.expand(count, *value.shape[1:]).contiguous()
            if isinstance(value, torch.Tensor)
            else value * count
        )
        for key, value in inputs.items()
    }
    target = (
        torch.arange(p * p, device=device)
        .reshape(p, p)
        .remainder(4)[None]
        .expand(count, -1, -1)
    )
    batch = {
        **inputs,
        "target_detail": target,
        "target_native": [
            torch.arange(49 * 49, device=device).reshape(49, 49).remainder(4)
            for _ in range(count)
        ],
        "geometry": [{"d4_k": 0, "d4_flip": False} for _ in range(count)],
    }
    result = {
        "data": "synthetic",
        "optimizer_updates": 0,
        "batch_size": count,
        "detail_size": p,
        "context_size": c["geometry"]["context_input_size"],
        "full_size": full_size,
        "steps": steps,
        "branches": c["model"]["new_branches"],
    }
    if real_train_patches:
        from .data import TrainSampler

        sampler = TrainSampler(c)
        batch = sampler.batch(0, 0, device)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
        values = []
        for step in range(steps):
            opt.zero_grad(set_to_none=True)
            with torch.autocast(
                "cuda", dtype=torch.bfloat16, enabled=str(device).startswith("cuda")
            ):
                output = model(**model_inputs(batch))
            loss = losses(output, batch, c)["total"]
            loss.backward()
            opt.step()
            values.append(float(loss.detach()))
        result.update(
            data="actual_train_patches_overfit_only",
            optimizer_updates=steps,
            losses=values,
        )
    else:
        records = []
        for step in range(max(2, steps)):
            model.zero_grad(set_to_none=True)
            with torch.autocast(
                "cuda", dtype=torch.bfloat16, enabled=str(device).startswith("cuda")
            ):
                output = model(**inputs)
            if step == 0:
                result["zero_residual_max_difference"] = float(
                    (output["logits_detail"] - output["base_logits_detail"]).abs().max()
                )
                assert result["zero_residual_max_difference"] == 0
            loss = losses(output, batch, c)["total"]
            loss.backward()
            norms = {
                name: float(param.grad.float().norm())
                for name, param in model.named_parameters()
                if param.grad is not None
            }
            assert all(
                torch.isfinite(p.grad).all()
                for p in model.parameters()
                if p.grad is not None
            )
            records.append(
                {
                    "loss": float(loss.detach()),
                    "base_gradient_sum": sum(
                        v for k, v in norms.items() if k.startswith("base.")
                    ),
                    "texture_gradient_sum": sum(
                        v for k, v in norms.items() if k.startswith("texture.")
                    ),
                    "context_gradient_sum": sum(
                        v for k, v in norms.items() if k.startswith("context.")
                    ),
                    "head_gradient_sum": sum(
                        v for k, v in norms.items() if k.startswith("residual.")
                    ),
                }
            )
            # Perturb only test-local head to examine earlier branch connectivity.
            # No dataset training or optimizer update; no weights are persisted.
            if step == 0 and model.residual is not None:
                with torch.no_grad():
                    model.residual.weight.normal_(0, 0.001)
        result["backward_diagnostics"] = records
        for key in ["texture", "context"]:
            if c["model"]["new_branches"][key]:
                assert records[1][key + "_gradient_sum"] > 0
        result["finite_gradients"] = True
    if str(device).startswith("cuda"):
        result["peak_memory_GiB"] = torch.cuda.max_memory_allocated() / 2**30
    json_write(
        ROOT
        / "reports"
        / f'smoke_{cfg["run"]["name"]}_{"full" if full_size else "small"}.json',
        result,
    )
    return result
