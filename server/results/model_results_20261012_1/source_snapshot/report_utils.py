"""Image-first four-class score and exact 2x2 validation comparison export."""
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path
import csv
import shutil

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

from experiment import ROOT, REPO, VALID, inference, read_json, write_json


def class_ious(prediction, target):
    if prediction.shape != target.shape or prediction.ndim != 2:
        raise ValueError('Prediction and ground truth must have equal H,W shapes.')
    if not (np.isin(prediction, range(4)).all() and np.isin(target, range(4)).all()):
        raise ValueError('Expected class labels 0..3.')
    values = []
    for class_id in range(4):
        intersection = np.count_nonzero((prediction == class_id) & (target == class_id))
        union = np.count_nonzero((prediction == class_id) | (target == class_id))
        values.append(float(intersection / union) if union else 0.0)
    return values


def comparison(original, prediction, target, palette, name, score):
    original = original.convert('RGB')
    width, height = original.size
    palette = np.asarray(palette, dtype=np.uint8)
    if target.shape != (height, width):
        raise ValueError('Original image and mask shapes differ.')
    errors = np.full((height, width, 3), 255, dtype=np.uint8)
    mismatch = prediction != target
    errors[mismatch] = palette[target[mismatch]]
    font_path = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
    font = ImageFont.truetype(font_path, 22) if Path(font_path).is_file() else ImageFont.load_default()
    title_height, label_height = 80, 34
    panel = Image.new('RGB', (2 * width, title_height + 2 * (height + label_height)), 'white')
    draw = ImageDraw.Draw(panel)
    draw.text((12, 8), name, fill='black', font=font)
    draw.text((12, 40), f'image mIoU: {score * 100:.6f}%    100% - image mIoU: {(1 - score) * 100:.6f}%', fill='black', font=font)
    tiles = [(original, 'Original image', 0, 0),
             (Image.fromarray(errors), 'Errors (ground-truth color; correct = white)', 1, 0),
             (Image.fromarray(palette[prediction]), 'Predicted mask', 0, 1),
             (Image.fromarray(palette[target]), 'Ground-truth mask', 1, 1)]
    for tile, label, column, row in tiles:
        x, y = column * width, title_height + row * (height + label_height)
        draw.text((x + 10, y + 4), label, fill='black', font=font)
        panel.paste(tile, (x, y + label_height))
    return panel


def allocate_result(results_root):
    results_root.mkdir(parents=True, exist_ok=True)
    date = datetime.now(ZoneInfo('Asia/Seoul')).strftime('%Y%m%d')
    for number in range(1, 1000000):
        destination = results_root / f'model_results_{date}_{number}'
        try:
            destination.mkdir()
            return destination
        except FileExistsError:
            continue
    raise RuntimeError('Could not allocate a new result directory.')

