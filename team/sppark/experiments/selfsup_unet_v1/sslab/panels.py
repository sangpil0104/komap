import numpy as np
from PIL import Image, ImageDraw, ImageFont
from .common import CLASS_NAMES, PALETTE

def font(size):
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()


def error_rgb(target, prediction):
    error = np.full((*target.shape, 3), 255, dtype=np.uint8)
    wrong = target != prediction
    error[wrong] = PALETTE[target[wrong]]
    return error


def comparison(path, original, target, prediction, image_id, score):
    height, width = target.shape
    margin, gap, header, top, footer = 14, 16, 38, 52, 76
    canvas_width = max(2 * width + 2 * margin + gap, 1050)
    canvas = Image.new("RGB", (canvas_width, top + 2 * (header + height) + gap + footer), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((margin, 12), f"{image_id} | image mIoU: {score:.4f}% (empty=1)", fill="black", font=font(22))
    positions = [(margin, top + header), (margin + width + gap, top + header),
                 (margin, top + 2 * header + height + gap),
                 (margin + width + gap, top + 2 * header + height + gap)]
    arrays = [original, error_rgb(target, prediction), PALETTE[prediction], PALETTE[target]]
    titles = ["Original image", "Errors: GT color; correct = white", "Predicted mask", "Ground-truth mask"]
    for position, array, title in zip(positions, arrays, titles):
        canvas.paste(Image.fromarray(array), position)
        draw.text((position[0], position[1] - 30), title, fill="black", font=font(19))
    y = canvas.height - footer + 12
    for class_id, name in enumerate(CLASS_NAMES):
        x = margin + 240 * class_id
        draw.rectangle((x, y, x + 18, y + 18), fill=tuple(PALETTE[class_id]))
        draw.text((x + 26, y - 2), name, fill="black", font=font(18))
    draw.text((margin, y + 30), "Gray errors = Eutectic Si (204); white = correct (255).", fill="black", font=font(17))
    canvas.save(path)
    # Verify the actual saved panels, including the frequently confused top-right panel.
    with Image.open(path) as reopened:
        for (x, y), expected in zip(positions, arrays):
            actual = np.asarray(reopened.crop((x, y, x + width, y + height)))
            if not np.array_equal(actual, expected):
                raise RuntimeError(f"Saved panel differs from source pixels: {path}")
    return {"image_id": image_id, "native_panel_size": [width, height],
            "panels_verified": 4, "error_rule": "wrong=GT palette, correct=white"}
