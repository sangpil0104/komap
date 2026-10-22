import csv
import io
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from metrics import CLASSES, PALETTE, summarize


def comparison_panel(original, target, prediction, score, image_id):
    original = np.asarray(original.convert('RGB')) if isinstance(original, Image.Image) else original
    h, w = target.shape
    if original.shape != (h,w,3):
        raise ValueError('Panel must preserve original image resolution')
    error = np.full((h,w,3), 255, np.uint8)
    wrong = prediction != target
    error[wrong] = PALETTE[target[wrong]]
    gap, header, caption = 24, 108, 44
    canvas = Image.new('RGB', (2*w+gap, 2*(h+caption)+gap+header), 'white')
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 28)
    small = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 24)
    draw.text((10,4), image_id, font=small, fill='black')
    draw.text((10,36), f'image mIoU: {100*score:.4f}%', font=font, fill='black')
    draw.text((10,71), f'100% - image mIoU: {100*(1-score):.4f}%', font=font, fill='black')
    panels = [(original, 'Original image', 0, header),
              (error, 'Errors: ground-truth color', w+gap, header),
              (PALETTE[prediction], 'Predicted mask', 0, header+h+caption+gap),
              (PALETTE[target], 'Ground-truth mask', w+gap, header+h+caption+gap)]
    for array, title, x, y in panels:
        draw.text((x+8,y+5), title, font=small, fill='black')
        canvas.paste(Image.fromarray(array), (x,y+caption))
    return canvas

