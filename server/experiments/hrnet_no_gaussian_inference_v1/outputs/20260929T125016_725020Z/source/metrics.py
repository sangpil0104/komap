"""Image-wise four-class IoU, ALWAYS empty union=0. No global mIoU."""
import cv2
import numpy as np

CLASSES = ('Al', 'Al3Ni', 'Primary Si', 'Eutectic Si')
PALETTE = np.array([[153,127,76], [25,76,153], [76,178,76], [204,204,204]], np.uint8)


def mask_ids(rgb):
    rgb = np.asarray(rgb)
    if rgb.ndim != 3 or rgb.shape[2] not in (3, 4):
        raise ValueError('Expected RGB/RGBA mask')
    if rgb.shape[2] == 4 and not (rgb[..., 3] == 255).all():
        raise ValueError('Unexpected alpha')
    out = np.full(rgb.shape[:2], 255, np.uint8)
    for k, color in enumerate(PALETTE):
        out[(rgb[..., :3] == color).all(-1)] = k
    if (out == 255).any():
        raise ValueError('Unknown mask color')
    return out


def boundary_band(target, radius=2):
    valid = target < 4
    edge = np.zeros_like(valid)
    d = (target[1:] != target[:-1]) & valid[1:] & valid[:-1]
    edge[1:] |= d
    edge[:-1] |= d
    d = (target[:, 1:] != target[:, :-1]) & valid[:, 1:] & valid[:, :-1]
    edge[:, 1:] |= d
    edge[:, :-1] |= d
    if radius:
        edge = cv2.dilate(edge.astype(np.uint8), np.ones((2*radius+1, 2*radius+1), np.uint8)).astype(bool)
    return edge & valid


def metrics(target, pred):
    if target.shape != pred.shape or (pred > 3).any() or (pred < 0).any():
        raise ValueError('Prediction shape/classes mismatch')
    valid = target < 4
    cm = np.bincount((4*target[valid].astype(np.int64) + pred[valid]), minlength=16).reshape(4,4)
    tp = cm.diagonal().astype(np.float64)
    union = cm.sum(0) + cm.sum(1) - tp
    iou = np.divide(tp, union, out=np.zeros(4, np.float64), where=union > 0)
    bands = {}
    for radius in (1, 2, 3):
        band = boundary_band(target, radius)
        bands[str(radius)] = float((pred[band] != target[band]).mean()) if band.any() else None
    return dict(image_miou=float(iou.mean()), class_iou=iou.tolist(),
                confusion_matrix=cm.tolist(), boundary_error=bands)


def summarize(rows):
    if not rows:
        raise ValueError('Empty evaluation')
    if len({r['image_id'] for r in rows}) != len(rows):
        raise ValueError('Duplicate validation images')
    return dict(image_miou=float(np.mean([r['image_miou'] for r in rows])),
                class_iou=np.mean([r['class_iou'] for r in rows], axis=0).tolist(),
                num_images=len(rows), empty_union=0, class_names=CLASSES,
                primary_metric='image_miou', rows=rows)
