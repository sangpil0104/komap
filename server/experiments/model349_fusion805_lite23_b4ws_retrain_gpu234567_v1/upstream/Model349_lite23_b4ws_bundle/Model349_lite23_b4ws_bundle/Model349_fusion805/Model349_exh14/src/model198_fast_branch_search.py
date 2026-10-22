# Model349 inference dependency; extracted from the recorded source snapshot.

# Unused training/search entrypoints are intentionally omitted.

from __future__ import annotations


import numpy as np

from scipy.ndimage import gaussian_filter

from aspp768_addition import cleanup_primary

from model183_metric_calibration import prior_terms

def replace_classes(base: np.ndarray, source: np.ndarray | None,
                    classes: tuple[int, ...], gamma: float) -> np.ndarray:
    if source is None or abs(gamma) < 1e-12 or not classes:
        return base
    output = base.copy()
    for class_id in classes:
        output[class_id] = (1.0 - gamma) * base[class_id] + gamma * source[class_id]
    return output / np.maximum(output.sum(axis=0, keepdims=True), 1e-7)

def bilateral_weights(guide: np.ndarray, radius: int = 2,
                      sigma_spatial: float = 1.0,
                      sigma_range: float = 0.60) -> list[np.ndarray]:
    pad = radius
    guide_pad = np.pad(guide, pad, mode="reflect")
    center = guide_pad[pad:pad + guide.shape[0], pad:pad + guide.shape[1]]
    spatial_denominator = 2.0 * sigma_spatial * sigma_spatial
    range_denominator = 2.0 * sigma_range * sigma_range
    weights = []
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            spatial_distance = float(dx * dx + dy * dy)
            spatial_weight = np.exp(-spatial_distance / spatial_denominator)
            neighbour_guide = guide_pad[
                pad + dy:pad + dy + guide.shape[0],
                pad + dx:pad + dx + guide.shape[1],
            ]
            weights.append((spatial_weight * np.exp(
                -((neighbour_guide - center) ** 2) / range_denominator
            )).astype(np.float32))
    return weights

def apply_bilateral(field: np.ndarray, guide_weights: list[np.ndarray],
                    radius: int = 2) -> np.ndarray:
    pad = radius
    field_pad = np.pad(field, pad, mode="reflect")
    numerator = np.zeros_like(field, dtype=np.float32)
    denominator = np.zeros_like(field, dtype=np.float32)
    index = 0
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            neighbour = field_pad[
                pad + dy:pad + dy + field.shape[0],
                pad + dx:pad + dx + field.shape[1],
            ]
            weight = guide_weights[index]
            numerator += weight * neighbour
            denominator += weight
            index += 1
    return numerator / np.maximum(denominator, 1e-7)

def predict(item: dict, mixed: np.ndarray, setting: dict,
            weights: list[np.ndarray], exact: bool) -> np.ndarray:
    base_prior, delta = prior_terms(item, mixed)
    prior = setting["prior_scale"] * base_prior
    prior[0] += (setting["prior_scale"] * setting["beta_scale"]
                 * item["params"]["beta"] * delta)
    logits = np.log(np.maximum(mixed, 1e-7)) + 0.09 * prior
    logits[2] += item["params"]["al_bias"] + setting["al_delta"]
    logits[3] += item["params"]["eut_bias"] + setting["eut_delta"]
    if exact:
        refined = logits.copy()
        refined[1] = (setting["gaussian_blend"] * gaussian_filter(
            logits[1], sigma=0.50, mode="reflect")
            + (1.0 - setting["gaussian_blend"]) * apply_bilateral(logits[1], weights))
        refined[3] = ((1.0 - setting["eut_edge_blend"]) * logits[3]
                      + setting["eut_edge_blend"] * apply_bilateral(logits[3], weights))
        prediction = cleanup_primary(
            refined.argmax(axis=0).astype(np.uint8), refined,
            setting["cleanup_threshold"])
        if float((prediction == 1).mean()) <= 0.01:
            mask = prediction == 1
            alternative = refined.copy()
            alternative[1] = -np.inf
            prediction[mask] = alternative[:, mask].argmax(axis=0).astype(np.uint8)
    else:
        # The first pass only ranks branch/calibration candidates.  It avoids
        # the expensive spatial filters; all finalists are re-evaluated below
        # with the exact model-197 route.
        prediction = logits.argmax(axis=0).astype(np.uint8)
    return prediction
