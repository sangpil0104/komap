from copy import deepcopy
import json
import numpy as np
import pytest
import torch
from PIL import Image
from alphase.io import ROOT, sha256, jsonl_write, ids_hash
from alphase.config import load_config, validate, merge
from alphase.contracts import CLASS_NAMES, ids_to_rgb
from alphase.geometry import (
    native_side,
    native_target,
    crop_resize,
    tile_positions,
    blend_weights,
    prepare_inputs,
)
from alphase.data import mixed_scores, hard_scores, TrainSampler, sample_seed
from alphase.cache import write_probability, load_probability, stream_mean, ensemble
from alphase.evaluation import unique_records
from alphase.oof import validate_lineage, LINEAGE_FIELDS
from alphase.models import EnhancedB4, NativeRefiner
from alphase.losses import losses
from alphase.training import optimizer_for, EMA, rng_state, restore_rng


def test_all_configs():
    for p in (ROOT / "configs").rglob("*.yaml"):
        load_config(p)


def test_config_errors(tmp_path, cfg):
    with pytest.raises(ValueError, match="Unknown"):
        merge(cfg, {"foo": 1})
    with pytest.raises(TypeError):
        merge(cfg, {"training": {"batch_size": "2"}})
    p = tmp_path / "a.yaml"
    p.write_text("inherit: a.yaml\n")
    with pytest.raises(ValueError, match="cycle"):
        load_config(p)
    c = deepcopy(cfg)
    c["training"]["initialize_from"] = None
    with pytest.raises(ValueError, match="required"):
        validate(c)
    for patch in [
        {"model": {"new_branches": {"residual_head": False}}},
        {"loss": {"boundary_enabled": True}},
        {"sampling": {"allow_validation_mining": True}},
        {"geometry": {"align_corners": True}},
    ]:
        with pytest.raises(ValueError):
            validate(merge(cfg, patch))


def test_native_rounding_target_and_coverage():
    assert native_side(675) == 259
    y = torch.arange(35).reshape(5, 7)
    roi, valid = native_target(y, (-2, 1, 6, 6))
    assert torch.equal(roi[:4, 2:], y[1:, 0:4]) and (roi[~valid] == 255).all()
    x = y.float()[None, None]
    actual = crop_resize(x, (1, 1, 3, 3), (3, 3))
    torch.testing.assert_close(actual[0, 0], y[1:4, 1:4].float())
    for h, w, n in [(13, 17, 7), (1, 3, 5), (37, 21, 12), (480, 675, 259)]:
        sums = torch.zeros(h, w)
        for yy in tile_positions(h, n):
            for xx in tile_positions(w, n):
                hh = min(n, h - yy)
                ww = min(n, w - xx)
                sums[yy : yy + hh, xx : xx + ww] += blend_weights(n, "cpu")[:hh, :ww]
        assert (sums > 0).all()


def test_sampler_mixed_and_hard():
    y = np.zeros((21, 31), np.uint8)
    assert mixed_scores(y).sum() == 0
    y[:, :15] = 1
    y[:, 15:] = 3
    assert mixed_scores(y).sum() > 0
    p = np.full((4, 21, 31), 0.25, np.float32)
    p[3] = 0.7
    p[:3] = 0.1
    score = hard_scores(y, p)
    assert score[y == 1].mean() > score[y == 3].mean()


def toy_rows(tmp_path):
    rows = []
    for i in range(2):
        x = np.arange(40 * 50, dtype=np.uint8).reshape(40, 50)
        y = np.full_like(x, i)
        ip = tmp_path / f"A{i}_image.png"
        mp = tmp_path / f"A{i}_mask.png"
        Image.fromarray(x).save(ip)
        Image.fromarray(ids_to_rgb(y)).save(mp)
        rows.append(
            dict(
                image_id=f"A{i}",
                split="train",
                image_path=str(ip),
                mask_path=str(mp),
                native_hw=[40, 50],
                image_sha256=sha256(ip),
                mask_sha256=sha256(mp),
                group_hint=f"A{i}",
            )
        )
    return rows


def test_sampler_isolation_fallback_resume(tmp_path, cfg):
    rows = toy_rows(tmp_path)
    c = deepcopy(cfg)
    c["geometry"].update(detail_input_size=32, context_input_size=16, target_width=64)
    c["sampling"].update(random=0.0, mixed=1.0, hard=0.0)
    sampler = TrainSampler(c, rows)
    a = sampler.sample(12, 1, 0, "cpu")
    sampler2 = TrainSampler(c, rows)
    b = sampler2.sample(12, 1, 0, "cpu")
    assert a["image_id"] == b["image_id"] and a["geometry"] == b["geometry"]
    torch.testing.assert_close(a["detail_rgb"], b["detail_rgb"], rtol=0, atol=0)
    assert sampler.counts["mixed_fallback"] == 1 and sampler.counts["random"] == 1
    rows[0]["split"] = "valid"
    with pytest.raises(ValueError, match="train-only"):
        TrainSampler(c, rows)


def meta(image_id="A1"):
    return dict(
        schema_version=1,
        image_id=image_id,
        source_split="valid",
        native_hw=[3, 5],
        class_names=list(CLASS_NAMES),
        space="native",
        representation="probability",
        dtype="float32",
        checkpoint_sha256="checkpoint",
        image_sha256="image",
        inference_config_hash="cfg",
        training_ids_hash=ids_hash(["train"]),
        checkpoint_selection_ids_hash=ids_hash([]),
    )


def test_cache_corruption_k1_duplicates(tmp_path):
    p = np.random.default_rng(0).random((4, 3, 5), dtype=np.float32)
    p /= p.sum(0)
    r = write_probability(tmp_path, meta(), p)
    np.testing.assert_array_equal(stream_mean([(tmp_path, r)]), p)
    np.testing.assert_array_equal(stream_mean([(tmp_path, r)] * 100), p)
    bad = deepcopy(r)
    bad["class_names"] = list(reversed(CLASS_NAMES))
    with pytest.raises(ValueError):
        load_probability(bad, tmp_path)
    bad = deepcopy(r)
    bad["native_hw"] = [5, 3]
    with pytest.raises(ValueError):
        load_probability(bad, tmp_path)
    (tmp_path / r["probability_path"]).write_bytes(b"corrupt")
    with pytest.raises(ValueError):
        load_probability(r, tmp_path)


def test_stream_ensemble_resume(tmp_path, cfg):
    manifests = []
    p = np.full((4, 3, 5), 0.25, np.float32)
    for i in range(3):
        out = tmp_path / f"seed{i}"
        rows = [write_probability(out, meta(k), p) for k in ["A1", "A2"]]
        jsonl_write(out / "manifest.jsonl", rows)
        manifests.append(out / "manifest.jsonl")
    out = tmp_path / "combined"
    result = ensemble(cfg, manifests, out)
    result2 = ensemble(cfg, manifests, out)
    assert result == result2
    with pytest.raises(FileExistsError):
        write_probability(tmp_path / "seed0", meta(), p)


def line():
    return {
        "predicted_image_id": "hold",
        "fold_id": 0,
        "teacher_weight_training_ids": ["train"],
        "teacher_ancestor_checkpoint_training_ids": [],
        "teacher_checkpoint_selection_ids": [],
        "teacher_bn_calibration_ids": [],
        "teacher_postprocessor_training_ids": [],
        "teacher_config_hash": "hash",
        "teacher_checkpoint_hashes": ["hash"],
        "teacher_ensemble_definition": {"seeds": [0]},
    }


@pytest.mark.parametrize("field", LINEAGE_FIELDS)
def test_oof_rejects_all_paths(field):
    l = line()
    assert validate_lineage(l)
    l[field].append("hold")
    with pytest.raises(ValueError, match="leakage"):
        validate_lineage(l)
    l[field] = "UNKNOWN"
    with pytest.raises(ValueError, match="Unknown"):
        validate_lineage(l)


def test_ddp_duplicate_ids_and_rng():
    a = {"image_id": "a", "score": 0.8}
    b = {"image_id": "b", "score": 0.9}
    assert unique_records([a, a, b]) == [a, b]
    with pytest.raises(ValueError):
        unique_records([a, dict(a, score=0.1)])
    state = rng_state()
    x = torch.rand(5)
    restore_rng(state)
    assert torch.equal(torch.rand(5), x)


@pytest.mark.parametrize(
    "texture,context,residual",
    [
        (False, False, False),
        (False, False, True),
        (True, False, True),
        (False, True, True),
        (True, True, True),
    ],
)
def test_real_b4_branch_contract(cfg, texture, context, residual):
    c = deepcopy(cfg)
    c["geometry"].update(detail_input_size=64, context_input_size=32)
    c["model"]["new_branches"].update(
        texture=texture, context=context, residual_head=residual
    )
    model = EnhancedB4(c).eval()
    image = torch.rand(1, 1, 39, 55)
    inputs = prepare_inputs(image, (3, 4, 23, 23), c, k=1, flip=True)
    with torch.no_grad():
        actual = model(**inputs)
        expected = model.base(inputs["detail_rgb"])
    torch.testing.assert_close(actual["logits_detail"], expected, rtol=0, atol=0)
    assert (model.texture is not None) == texture and (
        model.context is not None
    ) == context


def test_native_gradients_bn_freeze_and_ema(cfg):
    c = deepcopy(cfg)
    c["geometry"].update(detail_input_size=64, context_input_size=32)
    m = EnhancedB4(c)
    m.set_base_frozen(True)
    m.train()
    before = {k: v.clone() for k, v in m.base.state_dict().items()}
    inputs = prepare_inputs(torch.rand(1, 1, 41, 53), (3, 4, 21, 21), c)
    y = torch.randint(4, (1, 64, 64))
    native = torch.randint(4, (21, 21))
    batch = {
        "target_detail": y,
        "target_native": [native],
        "geometry": [{"d4_k": 0, "d4_flip": False}],
    }
    opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=0.001)
    norms = []
    for _ in range(2):
        opt.zero_grad()
        output = m(**inputs)
        loss = losses(output, batch, c)
        assert loss["native_nll"].requires_grad
        loss["total"].backward()
        norms.append(
            sum(
                float(p.grad.norm())
                for p in m.texture.parameters()
                if p.grad is not None
            )
        )
        opt.step()
    assert norms[0] == 0 and norms[1] > 0
    for k, v in m.base.state_dict().items():
        torch.testing.assert_close(v, before[k], atol=0, rtol=0)
    m.set_base_frozen(False)
    m.eval()
    m.zero_grad()
    losses(m(**inputs), batch, c)["total"].backward()
    assert (
        sum(
            float(p.grad.norm())
            for p in m.base.encoder.parameters()
            if p.grad is not None
        )
        > 0
    )
    ema = EMA(m, 0.9)
    with torch.no_grad():
        next(m.parameters()).add_(0.1)
    ema.update(m)
    assert ema.updates == 1
    for a, b in zip(ema.model.buffers(), m.buffers()):
        torch.testing.assert_close(a, b, atol=0, rtol=0)


def test_refiner_zero_and_gradients():
    m = NativeRefiner()
    gray = torch.rand(2, 1, 33, 47)
    p = torch.rand(2, 4, 33, 47).softmax(1)
    logits = m(gray, p)
    torch.testing.assert_close(logits.softmax(1), p, atol=1e-6, rtol=1e-5)
    torch.nn.functional.cross_entropy(logits, torch.randint(4, (2, 33, 47))).backward()
    assert m.head.weight.grad.norm() > 0


def test_online_hard_train_only_and_state(tmp_path, cfg):
    rows = toy_rows(tmp_path)
    c = deepcopy(cfg)
    c["geometry"].update(detail_input_size=32, context_input_size=16, target_width=64)
    c["sampling"].update(
        random=0.0, mixed=0.0, hard=1.0, hard_warmup_updates=2, hard_refresh_updates=3
    )
    sampler = TrainSampler(c, rows)
    calls = []

    def mine(gray):
        calls.append(gray.shape)
        return np.full((4, *gray.shape), 0.25, np.float32)

    sampler.mining_callback = mine
    sampler.sample(0, 0, 0, "cpu")
    assert not calls
    sampler.sample(2, 0, 0, "cpu")
    assert len(calls) == 1
    state = sampler.state_dict()
    restored = TrainSampler(c, rows)
    restored.load_state_dict(state)
    restored.mining_callback = mine
    a = sampler.sample(2, 0, 0, "cpu")
    b = restored.sample(2, 0, 0, "cpu")
    assert a["geometry"] == b["geometry"] and len(calls) == 1
    assert a["image_id"] in ["A0", "A1"]


def test_inference_odd_size_no_gt(cfg):
    from alphase.inference import predict_probs

    c = deepcopy(cfg)
    c["geometry"].update(detail_input_size=32, context_input_size=16, target_width=48)
    c["inference"]["d4"] = True

    class Constant(torch.nn.Module):
        def forward(self, detail_rgb, **kwargs):
            logits = torch.zeros(len(detail_rgb), 4, *detail_rgb.shape[-2:])
            logits[:, 1] = 2
            return {"logits_detail": logits}

    gray = np.arange(19 * 27, dtype=np.uint8).reshape(19, 27)
    p = predict_probs(Constant(), gray, c, "cpu")
    assert p.shape == (4, 19, 27) and (p.argmax(0) == 1).all()
    torch.testing.assert_close(p.sum(0), torch.ones(19, 27))
